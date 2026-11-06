import logging
import uuid

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
from classbuild.llm import GroqClientError
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    }


def init_state():
    for key, value in new_defaults().items():
        st.session_state.setdefault(key, value)


def reset_course_state():
    st.session_state.update(new_defaults())


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


def render_header():
    st.title("Mini ClassBuild")
    st.write(
        "Enter a topic, choose a difficulty and a lesson count, "
        "and Mini ClassBuild will turn it into a structured course."
    )


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

    # A new request never shows the previous course or its lessons.
    reset_course_state()

    try:
        config = validate_course_config(topic, difficulty, lesson_count)
    except CourseConfigError as exc:
        for message in exc.errors:
            st.warning(message)
        return

    try:
        with st.spinner("Generating your course outline..."):
            outline = CourseGenerator().generate_outline(config)
    except Exception as exc:
        show_generation_error(exc)
        return

    st.session_state[CONFIG_KEY] = config.to_dict()
    st.session_state[OUTLINE_KEY] = outline
    st.session_state[COURSE_ID_KEY] = uuid.uuid4().hex[:8]
    attach_material_to_course(st.session_state)

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

    render_lesson_section(outline)


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
    return True


def render_lesson_section(outline):
    st.divider()
    st.subheader("Lesson content")

    drop_stale_grounded_lessons()
    document = render_pdf_reference_option()

    course_id = st.session_state[COURSE_ID_KEY]
    titles = {lesson.lesson_number: lesson.title for lesson in outline.lessons}
    selected = st.selectbox(
        "Choose a lesson",
        list(titles),
        format_func=lambda number: f"Lesson {number}: {titles[number]}",
        key=f"lesson_picker_{course_id}",
    )
    st.session_state[SELECTED_LESSON_KEY] = selected

    lessons = st.session_state[LESSONS_KEY]
    if selected not in lessons:
        render_generate_lesson_buttons(selected, document)

    if selected in lessons:
        render_lesson_content(lessons[selected])
        render_quiz_section(selected)

def render_lesson_content(content):
    st.markdown(f"### Lesson {content.lesson_number}: {content.title}")
    if content.is_grounded:
        pages = ", ".join(str(page) for page in content.cited_pages)
        st.text(f"Based on {content.grounding.source_filename} · cited page(s): {pages}")
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

    if content.is_grounded:
        render_lesson_sources(content)

def quiz_widget_prefix(lesson_number):
    # Course id + lesson + attempt, so quizzes from different courses,
    # lessons or attempts never share radio-button state.
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
        return True  # already generated: no API call

    outline = st.session_state[OUTLINE_KEY]
    content = st.session_state[LESSONS_KEY].get(lesson_number)
    try:
        with st.spinner("Generating quiz..."):
            quiz = QuizGenerator().generate_quiz(outline, lesson_number, content)
    except Exception as exc:
        show_generation_error(exc)
        return False

    quizzes[lesson_number] = quiz
    return True


def show_question_text(number, text, bold=True):
    # Questions can contain code after the first line, so show it as code.
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
                index=None,  # nothing pre-selected
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
        result = grade_quiz(quiz, answers)  # local grading, no Groq call
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
            st.text(chunk.text)  # PDF text is untrusted: never render it as Markdown
        if len(document.chunks) > len(shown):
            st.caption(f"Showing the first {len(shown)} of {document.chunk_count} chunks.")
def render_pdf_search(document):
    st.markdown("#### Search the PDF")
    st.caption(
        "Finds the passages closest in meaning to your question. "
        "The first search loads a small model (about 90 MB download the first time)."
    )

    # The key includes the PDF fingerprint, so a new PDF always gets fresh widgets.
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
        st.text(result.text)  # PDF text is untrusted: never render it as Markdown
    st.caption("Similarity measures closeness of meaning. It does not prove correctness.")

def current_pdf_fingerprint():
    document = st.session_state.get(DOC_KEY)
    return document.fingerprint if document is not None else None


def drop_stale_grounded_lessons():
    """Remove PDF-based lessons whose PDF is no longer loaded."""
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


def render_pdf_reference_option():
    """Checkbox, off by default. Returns the document to ground new lessons in, or None."""
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
    index = get_or_build_index(st.session_state, document, service)  # cached per PDF
    generator = GroundedLessonGenerator(embeddings=service)
    return generator.generate_lesson(outline, lesson_number, document, index)


def current_evidence_notice(lesson_number):
    """A lesson's 'not enough evidence' notice, only if it refers to the loaded PDF."""
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
            st.text(source.text)  # PDF text is untrusted: never render it as Markdown

def main():
    st.set_page_config(page_title="Mini ClassBuild", layout="centered")
    init_state()
    render_header()
    render_learning_materials()
    render_form()
    render_outline()


main()