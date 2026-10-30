from dataclasses import asdict, dataclass

DIFFICULTY_LEVELS = ("Beginner", "Intermediate", "Advanced")
DEFAULT_DIFFICULTY = "Beginner"
MIN_LESSONS = 2
MAX_LESSONS = 5
DEFAULT_LESSONS = 3


class CourseConfigError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class CourseConfig:
    topic: str
    difficulty: str
    lesson_count: int

    def to_dict(self):
        return asdict(self)


def validate_course_config(topic, difficulty, lesson_count):
    errors = []
    clean_topic = (topic or "").strip()
    if not clean_topic:
        errors.append("Please enter a course topic.")
    if difficulty not in DIFFICULTY_LEVELS:
        errors.append("Difficulty must be Beginner, Intermediate or Advanced.")
    if (
        isinstance(lesson_count, bool)
        or not isinstance(lesson_count, int)
        or not MIN_LESSONS <= lesson_count <= MAX_LESSONS
    ):
        errors.append(
            f"Lesson count must be a whole number from {MIN_LESSONS} to {MAX_LESSONS}."
        )
    if errors:
        raise CourseConfigError(errors)
    return CourseConfig(clean_topic, difficulty, lesson_count)