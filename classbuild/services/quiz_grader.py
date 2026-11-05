from dataclasses import dataclass

from classbuild.models.quiz import OPTION_COUNT


class QuizSubmissionError(ValueError):
    def __init__(self, message, unanswered=None):
        super().__init__(message)
        self.unanswered = unanswered or []


@dataclass(frozen=True)
class QuestionResult:
    number: int
    question: str
    options: tuple
    selected_answer: int
    correct_answer: int
    is_correct: bool
    explanation: str

    @property
    def selected_text(self):
        return self.options[self.selected_answer]

    @property
    def correct_text(self):
        return self.options[self.correct_answer]


@dataclass(frozen=True)
class QuizResult:
    lesson_number: int
    correct_count: int
    total_questions: int
    percentage: float
    question_results: tuple


def _is_valid_index(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < OPTION_COUNT


def grade_quiz(quiz, answers):
    """answers: one chosen option index (0-3) per question, or None if unanswered."""
    if len(answers) != len(quiz.questions):
        raise QuizSubmissionError("Please answer every question.")

    unanswered = [number for number, answer in enumerate(answers, start=1) if answer is None]
    if unanswered:
        missing = ", ".join(str(number) for number in unanswered)
        raise QuizSubmissionError(f"Please answer every question. Missing: {missing}.", unanswered)

    if not all(_is_valid_index(answer) for answer in answers):
        raise QuizSubmissionError("One or more answers are not valid choices.")

    results = []
    for number, (question, answer) in enumerate(zip(quiz.questions, answers), start=1):
        results.append(
            QuestionResult(
                number=number,
                question=question.question,
                options=tuple(question.options),
                selected_answer=answer,
                correct_answer=question.correct_answer,
                is_correct=answer == question.correct_answer,
                explanation=question.explanation,
            )
        )

    correct = sum(1 for item in results if item.is_correct)
    total = len(results)
    return QuizResult(
        lesson_number=quiz.lesson_number,
        correct_count=correct,
        total_questions=total,
        percentage=correct / total * 100,
        question_results=tuple(results),
    )