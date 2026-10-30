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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_KEY = "course_config"
OUTLINE_KEY = "course_outline"
SELECTED_LESSON_KEY = "selected_lesson"
LESSONS_KEY = "lesson_contents"
QUIZZES_KEY = "quizzes"
COURSE_ID_KEY = "course_id"


def new_defaults():
    return {
        CONFIG_KEY: None,
        OUTLINE_KEY: None,
        SELECTED_LESSON_KEY: None,
        LESSONS_KEY: {},
        QUIZZES_KEY: {},
        COURSE_ID_KEY: None,
    }


def init_state():
    for key, value in new_defaults().items():
        st.session_state.setdefault(key, value)


def reset_course_state():
    st.session_state.update(new_defaults())


def show_generation_error(exc):
    if isinstance(exc, (GroqClientError, CourseGenerationError)):
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


def generate_lesson_content(lesson_number):
    lessons = st.session_state[LESSONS_KEY]
    if lesson_number in lessons:
        return True  # already generated: no API call

    try:
        with st.spinner("Generating lesson content..."):
            content = LessonGenerator().generate_lesson(
                st.session_state[OUTLINE_KEY], lesson_number
            )
    except Exception as exc:
        show_generation_error(exc)
        return False

    lessons[lesson_number] = content
    return True


def render_lesson_section(outline):
    st.divider()
    st.subheader("Lesson content")

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
        if st.button("Generate Lesson Content", key=f"generate_lesson_{course_id}_{selected}"):
            generate_lesson_content(selected)

    if selected in lessons:
        render_lesson_content(lessons[selected])


def render_lesson_content(content):
    st.markdown(f"### Lesson {content.lesson_number}: {content.title}")
    st.write(content.introduction)

    for section in content.sections:
        st.markdown(f"#### {section.heading}")
        st.write(section.explanation)
        st.markdown("**Example**")
        st.code(section.example, language=None, wrap_lines=True)

    st.markdown("#### Summary")
    st.write(content.summary)

    st.markdown("#### Exercises")
    for index, exercise in enumerate(content.exercises, start=1):
        st.markdown(f"**Exercise {index}.** {exercise.question}")
        with st.expander(f"Hint for exercise {index}"):
            st.write(exercise.hint)


def main():
    st.set_page_config(page_title="Mini ClassBuild", layout="centered")
    init_state()
    render_header()
    render_form()
    render_outline()


main()