import logging

from classbuild.course_config import validate_course_config
from classbuild.llm import GroqClientError
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.lesson_generator import LessonGenerator
from classbuild.services.structured_output import CourseGenerationError

logging.basicConfig(level=logging.INFO)

config = validate_course_config("Introduction to Python", "Beginner", 3)
try:
    outline = CourseGenerator().generate_outline(config)
    lesson = LessonGenerator().generate_lesson(outline, 2)
except (CourseGenerationError, GroqClientError) as exc:
    print(f"[FAILED] {exc}")
else:
    print(f"[OK] Lesson {lesson.lesson_number}: {lesson.title}")
    print("Sections:", len(lesson.sections), "| Exercises:", len(lesson.exercises))
    for section in lesson.sections:
        print("  -", section.heading)