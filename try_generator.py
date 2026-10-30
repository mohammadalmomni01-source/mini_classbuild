import logging

from classbuild.course_config import validate_course_config
from classbuild.llm import GroqClientError
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.structured_output import CourseGenerationError

logging.basicConfig(level=logging.INFO)

for count in (2, 3, 5):
    config = validate_course_config("Introduction to Python", "Beginner", count)
    try:
        outline = CourseGenerator().generate_outline(config)
    except (CourseGenerationError, GroqClientError) as exc:
        print(f"[FAILED] {count} lessons: {exc}")
        continue
    print(f"[OK] asked for {count}, got {len(outline.lessons)}: {outline.title}")
    for lesson in outline.lessons:
        print(f"   {lesson.lesson_number}. {lesson.title}")