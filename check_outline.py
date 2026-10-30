import copy

from pydantic import ValidationError

from classbuild.models.course import validate_outline

good = {
    "title": "Intro to Python",
    "description": "A first course in Python.",
    "difficulty": "Beginner",
    "learning_objectives": ["Write simple programs", "Use variables", "Use loops"],
    "lessons": [
        {
            "lesson_number": 1,
            "title": "Variables",
            "description": "Store data.",
            "learning_objectives": ["Create variables", "Use types"],
        },
        {
            "lesson_number": 2,
            "title": "Loops",
            "description": "Repeat actions.",
            "learning_objectives": ["Write for loops", "Use while loops"],
        },
    ],
}

print("Valid outline:", validate_outline(good, 2, "Beginner").title)

missing_title = copy.deepcopy(good)
del missing_title["title"]

cases = [
    ("missing title", missing_title, 2),
    ("wrong lesson count", good, 3),
]
for label, data, count in cases:
    try:
        validate_outline(data, count, "Beginner")
        print(f"[BUG] {label}: accepted")
    except ValidationError as exc:
        print(f"[OK] {label}: rejected -> {exc.errors()[0]['msg']}")