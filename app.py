import logging
import re
import uuid
from datetime import datetime, timezone

import streamlit as st

from classbuild.course_config import (
    DEFAULT_DIFFICULTY,
    DEFAULT_LESSONS,
    DIFFICULTY_LEVELS,
    MAX_LESSONS,
    MIN_LESSONS,
    CourseConfigError,
    validate_course_config,
)
from classbuild.export import course_filename, course_to_markdown
from classbuild.llm import GroqClientError
from classbuild.persistence import (
    CourseStore,
    CourseStorageError,
    SavedCourse,
    build_saved_course,
    new_course_id,
)
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.lesson_generator import LessonGenerator
from classbuild.services.structured_output import CourseGenerationError
from classbuild.services.quiz_generator import QuizGenerator
from classbuild.rag.document_state import (
    COURSE_MATERIAL_KEY,
    DOC_KEY,
    attach_material_to_course,
    material_for_course,
    sync_learning_material,
)
from classbuild.rag.embeddings import EmbeddingError, EmbeddingService
from classbuild.rag.search_state import current_results, get_or_build_index, run_search
from classbuild.rag.retriever import RetrievalError
from classbuild.services.grounded_lesson_generator import (
    GroundedLessonGenerator,
    InsufficientEvidenceError,
)
from classbuild.services.lesson_state import remove_stale_grounded_lessons
from classbuild.services.quiz_grader import QuizSubmissionError, grade_quiz
from classbuild.services.chat_state import (
    append_exchange,
    clear_thread,
    get_history,
    prune_chats,
    thread_key,
    valid_thread_keys,
)
from classbuild.services.lesson_chat import (
    ChatAnswerError,
    ChatInputError,
    LessonChatContext,
    LessonChatService,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
CONFIG_KEY = "course_config"
OUTLINE_KEY = "course_outline"
SELECTED_LESSON_KEY = "selected_lesson"
LESSONS_KEY = "lesson_contents"
QUIZZES_KEY = "quizzes"
COURSE_ID_KEY = "course_id"
RESULTS_KEY = "quiz_results"
UPLOADER_KEY = "pdf_uploader"
EVIDENCE_NOTICES_KEY = "evidence_notices"
LOW_SIMILARITY = 0.30
ATTEMPTS_KEY = "quiz_attempts"
CHATS_KEY = "lesson_chats"

# Stage 3: persistence / export keys
SAVED_ID_KEY = "saved_course_id"       # 32-hex storage id (different from COURSE_ID_KEY)
SAVED_CREATED_KEY = "saved_created_at"  # datetime of first save
SAVED_AT_KEY = "saved_at_hash"         # content_hash at last save (for "unsaved changes")
SAVED_HASH_KEY = "saved_content_hash"  # alias used by save logic
SOURCE_DOC_KEY = "source_doc"          # SourceDocument for a loaded PDF-based course
SAVE_FLASH_KEY = "save_flash"          # one-shot save success message

# Keys cleared when a new course replaces the old one
COURSE_STATE_KEYS = [
    CONFIG_KEY, OUTLINE_KEY, SELECTED_LESSON_KEY, LESSONS_KEY, QUIZZES_KEY,
    COURSE_ID_KEY, RESULTS_KEY, ATTEMPTS_KEY, COURSE_MATERIAL_KEY, EVIDENCE_NOTICES_KEY,
    CHATS_KEY, SAVED_ID_KEY, SAVED_CREATED_KEY, SAVED_AT_KEY, SAVED_HASH_KEY, SOURCE_DOC_KEY,
]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def new_defaults():
    return {
        CONFIG_KEY: None,
        OUTLINE_KEY: None,
        SELECTED_LESSON_KEY: None,
        LESSONS_KEY: {},
        QUIZZES_KEY: {},
        COURSE_ID_KEY: None,
        RESULTS_KEY: {},
        ATTEMPTS_KEY: {},
        COURSE_MATERIAL_KEY: None,
        EVIDENCE_NOTICES_KEY: {},
        CHATS_KEY: {},
        SAVED_ID_KEY: None,
        SAVED_CREATED_KEY: None,
        SAVED_AT_KEY: None,
        SAVED_HASH_KEY: None,
        SOURCE_DOC_KEY: None,
    }


def init_state():
    for key, value in new_defaults().items():
        st.session_state.setdefault(key, value)


def reset_course_state():
    st.session_state.update(new_defaults())


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

def show_generation_error(exc):
    if isinstance(exc, (GroqClientError, CourseGenerationError, EmbeddingError, RetrievalError)):
        st.error(str(exc))
        problems = getattr(exc, "problems", None)
        if problems:
            with st.expander("Details"):
                for problem in problems:
                    st.write(f"- {problem}")
    else:
        logger.error("Unexpected error during generation", exc_info=exc)
        st.error(f"Something went wrong ({type(exc).__name__}). Please try again.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header():
    st.title("Mini ClassBuild")
    st.write(
        "Enter a topic, choose a difficulty and a lesson count, "
        "and Mini ClassBuild will turn it into a structured course."
    )


# ---------------------------------------------------------------------------
# Keep-previous-course notice
# ---------------------------------------------------------------------------

def keep_previous_course_notice():
    """Tell the user the course still shown is the previous one."""
    if st.session_state.get(OUTLINE_KEY) is not None:
        st.info(
            "The request failed. The course shown below is the previous one — "
            "it has not changed."
        )


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

def render_form():
    with st.form("course_form"):
        topic = st.text_input("Course Topic", placeholder="e.g. Introduction to Python")
        difficulty = st.radio(
            "Difficulty",
            DIFFICULTY_LEVELS,
            index=DIFFICULTY_LEVELS.index(DEFAULT_DIFFICULTY),
            horizontal=True,
        )
        lesson_count = st.slider("Number of lessons", MIN_LESSONS, MAX_LESSONS, DEFAULT_LESSONS)
        submitted = st.form_submit_button("Generate Course Outline")

    if not submitted:
        return

    try:
        config = validate_course_config(topic, difficulty, lesson_count)
    except CourseConfigError as exc:
        for message in exc.errors:
            st.warning(message)
        return

    # Step 10: generate (and validate) the outline BEFORE clearing old state
    try:
        with st.spinner("Generating your course outline..."):
            outline = CourseGenerator().generate_outline(config)
    except Exception as exc:
        show_generation_error(exc)
        keep_previous_course_notice()
        return

    # Only clear the old course AFTER the new one has passed validation
    reset_course_state()
    st.session_state[CONFIG_KEY] = config.to_dict()
    st.session_state[OUTLINE_KEY] = outline
    st.session_state[COURSE_ID_KEY] = uuid.uuid4().hex[:8]
    attach_material_to_course(st.session_state)


# ---------------------------------------------------------------------------
# Outline + Markdown download
# ---------------------------------------------------------------------------

def render_outline():
    outline = st.session_state[OUTLINE_KEY]
    if outline is None:
        return

    st.header(outline.title)
    st.caption(f"Difficulty: {outline.difficulty} · {len(outline.lessons)} lessons")
    st.write(outline.description)

    st.subheader("Course objectives")
    for objective in outline.learning_objectives:
        st.markdown(f"- {objective}")

    st.subheader("Lessons")
    for lesson in outline.lessons:
        with st.expander(f"Lesson {lesson.lesson_number}: {lesson.title}"):
            st.write(lesson.description)
            st.markdown("**Learning objectives**")
            for objective in lesson.learning_objectives:
                st.markdown(f"- {objective}")

    # Markdown download button
    config = st.session_state.get(CONFIG_KEY) or {}
    topic = config.get("topic", outline.title)
    md_text = course_to_markdown(
        outline,
        topic,
        st.session_state[LESSONS_KEY],
        st.session_state[QUIZZES_KEY],
    )
    st.download_button(
        "⬇️ Download Course as Markdown",
        data=md_text,
        file_name=course_filename(outline.title),
        mime="text/markdown",
    )

    render_lesson_section(outline)


# ---------------------------------------------------------------------------
# Lesson generation
# ---------------------------------------------------------------------------

def generate_lesson_content(lesson_number, document=None):
    lessons = st.session_state[LESSONS_KEY]
    if lesson_number in lessons:
        return True  # already generated: no API call

    outline = st.session_state[OUTLINE_KEY]
    notices = st.session_state[EVIDENCE_NOTICES_KEY]
    try:
        with st.spinner("Generating lesson content..."):
            if document is not None:
                content = generate_grounded_lesson(outline, lesson_number, document)
            else:
                content = LessonGenerator().generate_lesson(outline, lesson_number)
    except InsufficientEvidenceError as exc:
        # Not an error: the PDF simply does not cover this lesson.
        notices[lesson_number] = {
            "message": str(exc),
            "fingerprint": document.fingerprint,
            "passages": exc.passages,
        }
        st.rerun()  # redraw so the warning and the fallback button appear
        return False
    except Exception as exc:
        show_generation_error(exc)
        return False

    notices.pop(lesson_number, None)
    lessons[lesson_number] = content
    # Remember which lesson was last generated (Step 11)
    st.session_state[SELECTED_LESSON_KEY] = lesson_number
    return True


# ---------------------------------------------------------------------------
# Lesson section (picker + content)
# ---------------------------------------------------------------------------

def render_lesson_section(outline):
    st.divider()
    st.subheader("Lesson content")

    drop_stale_grounded_lessons()
    prune_stale_chats()
    document = render_pdf_reference_option()

    course_id = st.session_state[COURSE_ID_KEY]
    titles = {lesson.lesson_number: lesson.title for lesson in outline.lessons}

    # Restore selected lesson from state (after a restore or a generate)
    saved_selection = st.session_state.get(SELECTED_LESSON_KEY)
    default_index = 0
    lesson_numbers = list(titles)
    if saved_selection is not None and saved_selection in lesson_numbers:
        default_index = lesson_numbers.index(saved_selection)

    selected = st.selectbox(
        "Choose a lesson",
        lesson_numbers,
        index=default_index,
        format_func=lambda number: f"Lesson {number}: {titles[number]}",
        key=f"lesson_picker_{course_id}",
    )
    st.session_state[SELECTED_LESSON_KEY] = selected

    lessons = st.session_state[LESSONS_KEY]
    if selected not in lessons:
        render_generate_lesson_buttons(selected, document)

    if selected in lessons:
        render_lesson_content(lessons[selected])
        render_chat_section(outline, lessons[selected], document)
        render_quiz_section(selected)


# ---------------------------------------------------------------------------
# Lesson content display
# ---------------------------------------------------------------------------

def render_lesson_content(content):
    st.markdown(f"### Lesson {content.lesson_number}: {content.title}")
    if content.is_grounded:
        pages = ", ".join(str(page) for page in content.cited_pages)
        # Even if the PDF isn't loaded, show saved citations as history
        if pdf_features_allowed():
            st.text(f"Based on {plain(content.grounding.source_filename)} · cited page(s): {pages}")
        else:
            st.text(
                f"Originally based on {plain(content.grounding.source_filename)} · "
                f"cited page(s): {pages} (PDF not currently loaded)"
            )
    st.write(content.introduction)

    for section in content.sections:
        st.markdown(f"#### {section.heading}")
        st.write(section.explanation)
        st.markdown("**Example**")
        st.code(section.example, language=None, wrap_lines=True)
        if section.source_ids:
            labels = ", ".join(source_label(content, item) for item in section.source_ids)
            st.text(f"Sources: {labels}")

    st.markdown("#### Summary")
    st.write(content.summary)

    st.markdown("#### Exercises")
    for index, exercise in enumerate(content.exercises, start=1):
        st.markdown(f"**Exercise {index}.** {exercise.question}")
        with st.expander(f"Hint for exercise {index}"):
            st.write(exercise.hint)

    if content.is_grounded and pdf_features_allowed():
        render_lesson_sources(content)


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

def quiz_widget_prefix(lesson_number):
    course_id = st.session_state[COURSE_ID_KEY]
    attempt = st.session_state[ATTEMPTS_KEY].get(lesson_number, 0)
    return f"quiz_{course_id}_l{lesson_number}_a{attempt}"


def retake_quiz(lesson_number):
    """Callback for Try Again: runs before the page redraws."""
    st.session_state[RESULTS_KEY].pop(lesson_number, None)
    attempts = st.session_state[ATTEMPTS_KEY]
    attempts[lesson_number] = attempts.get(lesson_number, 0) + 1


def generate_quiz(lesson_number):
    quizzes = st.session_state[QUIZZES_KEY]
    if lesson_number in quizzes:
        return True

    outline = st.session_state[OUTLINE_KEY]
    content = st.session_state[LESSONS_KEY].get(lesson_number)
    try:
        with st.spinner("Generating quiz..."):
            quiz = QuizGenerator().generate_quiz(outline, lesson_number, content)
    except Exception as exc:
        show_generation_error(exc)
        return False

    quizzes[lesson_number] = quiz
    # Remember which lesson was last used (Step 11)
    st.session_state[SELECTED_LESSON_KEY] = lesson_number
    return True


def show_question_text(number, text, bold=True):
    stem, _, code = text.partition("\n")
    st.markdown(f"**{number}. {stem}**" if bold else f"{number}. {stem}")
    if code.strip():
        st.code(code.strip("\n"), language="python")


def render_quiz_section(lesson_number):
    st.divider()
    st.subheader("Quiz")

    quizzes = st.session_state[QUIZZES_KEY]
    if lesson_number not in quizzes:
        course_id = st.session_state[COURSE_ID_KEY]
        if st.button("Generate Quiz", key=f"generate_quiz_{course_id}_{lesson_number}"):
            generate_quiz(lesson_number)

    if lesson_number not in quizzes:
        return

    result = st.session_state[RESULTS_KEY].get(lesson_number)
    if result is None:
        render_quiz_form(lesson_number, quizzes[lesson_number])
    else:
        render_quiz_result(lesson_number, result)


def render_quiz_form(lesson_number, quiz):
    prefix = quiz_widget_prefix(lesson_number)
    with st.form(f"{prefix}_form"):
        for number, question in enumerate(quiz.questions, start=1):
            show_question_text(number, question.question)
            st.radio(
                f"Answer for question {number}",
                options=list(range(len(question.options))),
                format_func=lambda index, options=question.options: options[index],
                index=None,
                key=f"{prefix}_q{number}",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Submit Answers")

    if not submitted:
        return

    answers = [
        st.session_state.get(f"{prefix}_q{number}")
        for number in range(1, len(quiz.questions) + 1)
    ]
    try:
        result = grade_quiz(quiz, answers)
    except QuizSubmissionError as exc:
        st.warning(str(exc))
        return

    st.session_state[RESULTS_KEY][lesson_number] = result
    st.rerun()


def render_quiz_result(lesson_number, result):
    message = f"Score: {result.correct_count} / {result.total_questions} ({result.percentage:.0f}%)"
    if result.percentage >= 80:
        st.success(message)
    elif result.percentage >= 50:
        st.info(message)
    else:
        st.warning(message)

    for item in result.question_results:
        icon = "✅" if item.is_correct else "❌"
        show_question_text(f"{icon} {item.number}", item.question)
        st.write(f"Your answer: {item.selected_text}")
        if not item.is_correct:
            st.write(f"Correct answer: {item.correct_text}")
        st.caption(item.explanation)

    st.button(
        "Try Again",
        key=f"retake_{quiz_widget_prefix(lesson_number)}",
        on_click=retake_quiz,
        args=(lesson_number,),
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def chat_message_caption(message):
    if message.role != "assistant":
        return None
    if message.mode == "pdf":
        if message.status == "answered":
            pages = ", ".join(str(page) for page in message.cited_pages)
            return f"From the uploaded PDF · cited page(s): {pages}"
        return "The uploaded PDF did not have enough evidence to answer this."
    if message.truncated:
        return "The answer was cut off by the length limit."
    return None


def render_chat_message(message):
    with st.chat_message(message.role):
        st.markdown(message.content)
        caption = chat_message_caption(message)
        if caption:
            st.caption(caption)


def chat_index_for(document):
    if document is None:
        return None, None
    embeddings = EmbeddingService()
    index = get_or_build_index(st.session_state, document, embeddings)
    return embeddings, index


def chat_thread_id(key):
    return "_".join(str(part) for part in key)


def render_chat_section(outline, content, document):
    st.divider()
    st.subheader("Ask about this lesson")

    # Hide PDF answers when PDF features are not allowed
    if document is not None and not pdf_features_allowed():
        req = required_source_document()
        st.caption(
            f"Answers use this lesson's content and your recent messages. "
            f"To use PDF answers, upload {plain(req.filename) if req else 'the original PDF'}."
        )
        document = None
    else:
        st.caption(
            "Answers are grounded in the uploaded PDF, with checked citations."
            if document is not None
            else "Answers use this lesson's content and your recent messages."
        )

    try:
        ctx = LessonChatContext.from_course(outline, content.lesson_number, content)
    except ChatAnswerError as exc:
        st.info(str(exc))
        return

    course_id = st.session_state[COURSE_ID_KEY]
    chats = st.session_state[CHATS_KEY]
    key = thread_key(course_id, content, document.fingerprint if document is not None else None)
    history = get_history(chats, key)
    thread_id = chat_thread_id(key)

    for message in history:
        render_chat_message(message)

    question = st.chat_input("Ask a question about this lesson", key=f"chat_input_{thread_id}")
    if question:
        try:
            embeddings, index = chat_index_for(document)
            service = LessonChatService(embeddings=embeddings)
            reply = service.answer(ctx, question, history=history, document=document, index=index)
        except ChatInputError as exc:
            st.warning(str(exc))
        except Exception as exc:
            show_generation_error(exc)
        else:
            append_exchange(chats, key, reply.question, reply.answer)
            st.rerun()

    if history:
        st.button(
            "Clear Chat",
            key=f"clear_chat_{thread_id}",
            on_click=clear_thread,
            args=(chats, key),
        )


# ---------------------------------------------------------------------------
# Learning materials (PDF upload)
# ---------------------------------------------------------------------------

def render_learning_materials():
    st.subheader("Learning materials (optional)")
    st.caption("Upload a PDF to use as reference material. It is read on your computer.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key=UPLOADER_KEY)
    upload = (uploaded.name, uploaded.getvalue()) if uploaded is not None else None
    status = sync_learning_material(st.session_state, upload)

    if status.error:
        st.error(status.error)
        return
    if status.document is None:
        return

    document = status.document
    st.success(
        f"Extracted text from {document.pages_with_text} of {document.page_count} pages "
        f"and created {document.chunk_count} chunks."
    )
    render_document_summary(document)
    render_pdf_search(document)


def render_document_summary(document):
    columns = st.columns(4)
    columns[0].metric("Pages", document.page_count)
    columns[1].metric("Words", document.word_count)
    columns[2].metric("Chunks", document.chunk_count)
    columns[3].metric("Pages without text", len(document.empty_pages))

    if document.empty_pages:
        pages = ", ".join(str(number) for number in document.empty_pages)
        st.warning(f"No text could be extracted from page(s): {pages}. They may be images.")

    st.caption(
        f"Chunks are up to {document.chunk_size} characters with "
        f"{document.chunk_overlap} characters of overlap. "
        f"Fingerprint: {document.fingerprint[:12]}"
    )

    with st.expander("Preview chunks"):
        shown = document.chunks[:20]
        for chunk in shown:
            st.markdown(
                f"**{chunk.chunk_id}** · page {chunk.page_number} · {chunk.char_count} characters"
            )
            st.text(chunk.text)
        if len(document.chunks) > len(shown):
            st.caption(f"Showing the first {len(shown)} of {document.chunk_count} chunks.")


def render_pdf_search(document):
    st.markdown("#### Search the PDF")
    st.caption(
        "Finds the passages closest in meaning to your question. "
        "The first search loads a small model (about 90 MB download the first time)."
    )

    key = f"pdf_search_{document.fingerprint[:12]}"
    with st.form(f"{key}_form"):
        query = st.text_input(
            "Question",
            key=f"{key}_query",
            placeholder="e.g. What does dynamically typed mean in Python?",
        )
        submitted = st.form_submit_button("Search PDF")

    if submitted:
        try:
            with st.spinner("Searching..."):
                run_search(st.session_state, query)
        except (EmbeddingError, RetrievalError) as exc:
            st.error(str(exc))
            return
        except Exception:
            logger.exception("Unexpected error during PDF search")
            st.error("Something went wrong while searching the PDF. Please try again.")
            return

    saved = current_results(st.session_state)
    if saved:
        render_search_results(saved["results"])


def render_search_results(results):
    st.caption(f"Top {len(results)} passages for your question")
    if results and results[0].score < LOW_SIMILARITY:
        st.warning(
            "These passages have low similarity, so they probably do not answer "
            "your question. A top result is not proof of an answer."
        )
    for result in results:
        st.markdown(
            f"**#{result.rank}** · page {result.page_number} · {result.chunk_id} "
            f"· similarity {result.score:.3f}"
        )
        st.text(result.text)
    st.caption("Similarity measures closeness of meaning. It does not prove correctness.")


# ---------------------------------------------------------------------------
# PDF reference option + stale-lesson helpers
# ---------------------------------------------------------------------------

def current_pdf_fingerprint():
    document = st.session_state.get(DOC_KEY)
    return document.fingerprint if document is not None else None


def required_source_document():
    """The SourceDocument a loaded course needs (or None for a fresh course)."""
    return st.session_state.get(SOURCE_DOC_KEY)


def pdf_features_allowed():
    """
    PDF-based generation and PDF answers are allowed when:
    - there is no required source document (fresh/non-grounded course), OR
    - the uploaded PDF has the same fingerprint as the required one.
    """
    req = required_source_document()
    if req is None:
        return True
    return current_pdf_fingerprint() == req.fingerprint


def drop_stale_grounded_lessons():
    """Remove PDF-based lessons whose PDF is no longer loaded."""
    # For a loaded course, grounded lessons are saved content — do not remove them.
    if st.session_state.get(SAVED_ID_KEY) is not None:
        return
    removed = remove_stale_grounded_lessons(
        st.session_state[LESSONS_KEY],
        current_pdf_fingerprint(),
        st.session_state[QUIZZES_KEY],
        st.session_state[RESULTS_KEY],
        st.session_state[ATTEMPTS_KEY],
    )
    if removed:
        numbers = ", ".join(str(number) for number in sorted(removed))
        st.warning(f"The PDF changed, so the PDF-based lesson(s) {numbers} were removed.")


def prune_stale_chats():
    valid = valid_thread_keys(
        st.session_state[COURSE_ID_KEY],
        st.session_state[LESSONS_KEY],
        current_pdf_fingerprint(),
    )
    prune_chats(st.session_state[CHATS_KEY], valid)


def render_pdf_reference_option():
    """Checkbox, off by default. Returns the document to ground new lessons in, or None."""
    # Hide the option entirely when PDF features are blocked
    if not pdf_features_allowed():
        req = required_source_document()
        if req is not None:
            st.info(
                f"This course was based on **{plain(req.filename)}**. "
                "Upload that PDF to enable PDF-based generation and answers. "
                "A different file with the same name will not unlock them."
            )
        return None

    document = material_for_course(st.session_state)
    if document is None:
        render_course_material_note()
        return None

    course_id = st.session_state[COURSE_ID_KEY]
    use_pdf = st.checkbox(
        "Use Uploaded PDF as Reference",
        key=f"use_pdf_{course_id}_{document.fingerprint[:12]}",
        help="New lessons are based on passages from your PDF and show page citations.",
    )
    return document if use_pdf else None


def render_course_material_note():
    if st.session_state.get(DOC_KEY) is not None:
        st.info(
            "The loaded PDF was not part of this course. "
            "Generate a new course to use it as reference."
        )


def generate_grounded_lesson(outline, lesson_number, document):
    service = EmbeddingService()
    index = get_or_build_index(st.session_state, document, service)
    generator = GroundedLessonGenerator(embeddings=service)
    return generator.generate_lesson(outline, lesson_number, document, index)


def current_evidence_notice(lesson_number):
    notice = st.session_state[EVIDENCE_NOTICES_KEY].get(lesson_number)
    if notice and notice["fingerprint"] == current_pdf_fingerprint():
        return notice
    return None


def render_generate_lesson_buttons(selected, document):
    course_id = st.session_state[COURSE_ID_KEY]
    notice = current_evidence_notice(selected) if document is not None else None

    if notice is not None:
        st.warning(notice["message"])
        with st.expander("Passages that were checked"):
            for passage in notice["passages"]:
                st.text(
                    f"{passage.chunk_id} · page {passage.page_number} "
                    f"· similarity {passage.score:.3f}"
                )
                st.text(passage.text)
        if st.button("Generate Without PDF", key=f"generate_plain_{course_id}_{selected}"):
            if generate_lesson_content(selected, None):
                st.rerun()
        return

    label = "Generate Lesson Content (from PDF)" if document is not None else "Generate Lesson Content"
    if st.button(label, key=f"generate_lesson_{course_id}_{selected}"):
        generate_lesson_content(selected, document)


def source_label(content, source_id):
    for source in content.grounding.sources:
        if source.chunk_id == source_id:
            return f"{source.chunk_id} (page {source.page_number})"
    return source_id


def render_lesson_sources(content):
    st.markdown("#### Sources")
    for source in content.grounding.sources:
        st.text(
            f"{source.chunk_id} · {source.source_filename} · page {source.page_number} "
            f"· similarity {source.similarity:.3f}"
        )
        with st.expander(f"Show passage {source.chunk_id}"):
            st.text(source.text)


# ---------------------------------------------------------------------------
# Stage 3 — Save / Load / Export
# ---------------------------------------------------------------------------

def plain(text: str) -> str:
    """Escape Markdown special characters so a filename is displayed as plain text."""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)


def _get_store() -> CourseStore:
    return CourseStore()


def current_saved_course(course_id: str) -> SavedCourse | None:
    """Build a validated SavedCourse from session state. Returns None on failure."""
    outline = st.session_state.get(OUTLINE_KEY)
    if outline is None:
        return None
    config = st.session_state.get(CONFIG_KEY) or {}
    topic = config.get("topic", outline.title)
    try:
        return build_saved_course(
            course_id=course_id,
            topic=topic,
            outline=outline,
            lessons=st.session_state[LESSONS_KEY],
            quizzes=st.session_state[QUIZZES_KEY],
            created_at=st.session_state.get(SAVED_CREATED_KEY),
            selected_lesson=st.session_state.get(SELECTED_LESSON_KEY),
        )
    except Exception as exc:
        logger.warning("Could not build SavedCourse: %s", exc)
        return None


def save_current_course(store: CourseStore):
    """
    Save or update the current course.
    Each failure kind gets its own message and leaves state unchanged.
    """
    # Ensure we have a stable 32-hex storage id
    saved_id = st.session_state.get(SAVED_ID_KEY)
    if saved_id is None:
        saved_id = new_course_id()

    course = current_saved_course(saved_id)
    if course is None:
        st.error("There is no course to save yet. Generate a course outline first.")
        return

    try:
        store.save(course)
    except CourseStorageError as exc:
        st.error(f"Could not save the course: {exc}")
        return
    except Exception as exc:
        logger.error("Unexpected error while saving course", exc_info=exc)
        st.error(f"Unexpected error while saving ({type(exc).__name__}). Please try again.")
        return

    # Persist the storage id and hash so "unsaved changes" can be detected
    st.session_state[SAVED_ID_KEY] = saved_id
    st.session_state[SAVED_CREATED_KEY] = course.created_at
    st.session_state[SAVED_AT_KEY] = course.content_hash()
    st.session_state[SAVE_FLASH_KEY] = True


def restore_saved_course(course: SavedCourse):
    """
    Replace the course on screen with a saved one.
    Does NOT call Groq. Resets quiz answers, chats and widget state
    by assigning a new session course id.
    """
    reset_course_state()

    st.session_state[OUTLINE_KEY] = course.outline
    st.session_state[CONFIG_KEY] = {
        "topic": course.topic,
        "difficulty": course.difficulty,
        "lesson_count": len(course.outline.lessons),
    }
    st.session_state[LESSONS_KEY] = course.lessons_by_number()
    st.session_state[QUIZZES_KEY] = course.quizzes_by_number()
    st.session_state[SELECTED_LESSON_KEY] = course.selected_lesson
    st.session_state[COURSE_ID_KEY] = uuid.uuid4().hex[:8]  # new session id → fresh widget state

    # Persistence metadata
    st.session_state[SAVED_ID_KEY] = course.course_id
    st.session_state[SAVED_CREATED_KEY] = course.created_at
    st.session_state[SAVED_AT_KEY] = course.content_hash()

    # Remember the required PDF so PDF features are gated correctly
    if course.source_document is not None:
        st.session_state[SOURCE_DOC_KEY] = course.source_document


def render_saved_courses(container):
    """Render the Saved Courses panel inside `container`."""
    with container:
        st.subheader("Saved Courses")

        store = _get_store()
        outline = st.session_state.get(OUTLINE_KEY)

        # ---- Save / Update button ----
        if outline is not None:
            saved_id = st.session_state.get(SAVED_ID_KEY)
            if saved_id is None:
                label = "💾 Save Course"
                unsaved_note = ""
            else:
                # Check whether content has changed since last save
                course = current_saved_course(saved_id)
                current_hash = course.content_hash() if course else None
                last_hash = st.session_state.get(SAVED_AT_KEY)
                has_changes = current_hash != last_hash
                label = "💾 Update Saved Course" if has_changes else "✅ Saved"
                unsaved_note = "*(unsaved changes)*" if has_changes else ""

            col1, col2 = st.columns([2, 3])
            with col1:
                if st.button(label, disabled=(label == "✅ Saved")):
                    save_current_course(store)
                    st.rerun()
            with col2:
                if unsaved_note:
                    st.caption(unsaved_note)

        # Flash a success message once after saving
        if st.session_state.pop(SAVE_FLASH_KEY, False):
            st.success("Course saved.")

        # ---- Listing ----
        listing = store.list_courses()

        for err in listing.errors:
            st.warning(f"⚠️ Damaged file: {err}")

        if not listing.courses:
            st.caption("No saved courses yet.")
            return

        options = {
            s.course_id: (
                f"{s.title} ({s.difficulty}) · "
                f"{s.generated_lesson_count}/{s.lesson_count} lessons"
                + (" · PDF-based" if s.has_grounded_lessons else "")
                + f" · saved {s.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )
            for s in listing.courses
        }

        selected_id = st.selectbox(
            "Select a saved course",
            list(options.keys()),
            format_func=lambda cid: options[cid],
        )

        if st.button("📂 Load Selected Course"):
            try:
                course = store.load(selected_id)
            except CourseStorageError as exc:
                st.error(f"Could not load the course: {exc}")
                return
            restore_saved_course(course)
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Mini ClassBuild", layout="centered")
    init_state()
    render_header()
    render_learning_materials()
    render_form()

    # Reserve a container for the Saved Courses panel.
    # It is filled AFTER render_outline() so lessons and quizzes generated
    # in this same rerun are already in session state when the panel runs.
    saved_panel = st.container()

    render_outline()

    render_saved_courses(saved_panel)


main()
