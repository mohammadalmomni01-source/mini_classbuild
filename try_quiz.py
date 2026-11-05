import logging

from classbuild.course_config import validate_course_config
from classbuild.llm import GroqClientError
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.lesson_generator import LessonGenerator
from classbuild.services.quiz_generator import QuizGenerator
from classbuild.services.quiz_grader import grade_quiz
from classbuild.services.structured_output import CourseGenerationError

logging.basicConfig(level=logging.INFO)

config = validate_course_config("Introduction to Python", "Beginner", 3)
try:
    outline = CourseGenerator().generate_outline(config)
    content = LessonGenerator().generate_lesson(outline, 2)
    quiz = QuizGenerator().generate_quiz(outline, 2, content)
except (CourseGenerationError, GroqClientError) as exc:
    print(f"[FAILED] {exc}")
else:
    print(f"[OK] Quiz for lesson {quiz.lesson_number}: {quiz.lesson_title}")
    for number, item in enumerate(quiz.questions, start=1):
        print(f"{number}. {item.question} ({len(item.options)} options, answer {item.correct_answer})")

    all_right = [item.correct_answer for item in quiz.questions]
    result = grade_quiz(quiz, all_right)
    print(f"Local grading, all correct: {result.correct_count}/{result.total_questions}")

    all_wrong = [(item.correct_answer + 1) % 4 for item in quiz.questions]
    result = grade_quiz(quiz, all_wrong)
    print(f"Local grading, all wrong: {result.correct_count}/{result.total_questions}")