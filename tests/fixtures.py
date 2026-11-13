"""
tests/fixtures.py
Test fixtures for models, documents, and outline generation.
"""
from classbuild.rag.models import DocumentChunk, ProcessedDocument


def valid_outline(lesson_count: int = 3, difficulty: str = "Beginner") -> dict:
    """Return a valid dictionary representing a CourseOutline."""
    return {
        "title": "Introduction to Python",
        "description": "Learn the basics of Python programming from scratch.",
        "difficulty": difficulty,
        "learning_objectives": [
            "Understand variables and data types in Python.",
            "Write basic control flow statements like if/else and loops.",
            "Define and invoke functions with arguments and return values.",
        ],
        "lessons": [
            {
                "lesson_number": i,
                "title": f"Lesson Title {i}",
                "description": f"Detailed description for lesson number {i}.",
                "learning_objectives": [
                    f"Objective 1 for lesson {i}",
                    f"Objective 2 for lesson {i}",
                ],
            }
            for i in range(1, lesson_count + 1)
        ],
    }


def valid_lesson(lesson_number: int = 1, title: str = "Lesson Title 1", is_grounded: bool = False) -> dict:
    """Return a valid dictionary representing LessonContent."""
    data = {
        "lesson_number": lesson_number,
        "title": title,
        "introduction": "This is an introductory paragraph explaining what we will learn.",
        "sections": [
            {
                "heading": "Core Concepts",
                "explanation": "Here is an in-depth explanation of the topic.",
                "example": "print('Hello world!')",
                "source_ids": ["p1-c1"] if is_grounded else [],
            },
            {
                "heading": "Practical Application",
                "explanation": "Here is how to apply the knowledge practically.",
                "example": "x = 42\nprint(x * 2)",
                "source_ids": ["p1-c1"] if is_grounded else [],
            },
        ],
        "summary": "In this lesson we covered fundamental syntax and operations.",
        "exercises": [
            {
                "question": "What will print(1 + 1) output?",
                "hint": "It performs basic addition.",
            },
            {
                "question": "How do you define a function in Python?",
                "hint": "Use the def keyword.",
            },
        ],
    }
    if is_grounded:
        data["grounding"] = {
            "document_fingerprint": "a" * 32,
            "source_filename": "sample.pdf",
            "retrieval_query": "Introduction to Python",
            "sources": [
                {
                    "chunk_id": "p1-c1",
                    "source_filename": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 1,
                    "similarity": 0.85,
                    "text": "Python is a popular programming language.",
                }
            ],
        }
    return data


def valid_quiz(lesson_number: int = 1, lesson_title: str = "Lesson Title 1") -> dict:
    """Return a valid dictionary representing a Quiz."""
    return {
        "lesson_number": lesson_number,
        "lesson_title": lesson_title,
        "questions": [
            {
                "question": f"Question {i}: What is the primary purpose of topic {i}?",
                "options": [
                    f"Option A for Q{i}",
                    f"Option B for Q{i}",
                    f"Option C for Q{i}",
                    f"Option D for Q{i}",
                ],
                "correct_answer": 0,
                "explanation": f"Explanation for question {i}.",
            }
            for i in range(1, 6)
        ],
    }


def sample_document() -> ProcessedDocument:
    """Return a sample ProcessedDocument."""
    chunks = [
        DocumentChunk(
            chunk_id="p1-c1",
            source_filename="sample.pdf",
            page_number=1,
            chunk_index=1,
            text="Python is a high-level, general-purpose programming language.",
        ),
        DocumentChunk(
            chunk_id="p1-c2",
            source_filename="sample.pdf",
            page_number=1,
            chunk_index=2,
            text="Its design philosophy emphasizes code readability with indentation.",
        ),
    ]
    return ProcessedDocument(
        filename="sample.pdf",
        fingerprint="a" * 32,
        page_count=1,
        pages_with_text=1,
        word_count=20,
        char_count=120,
        empty_pages=[],
        chunks=chunks,
        chunk_size=1000,
        chunk_overlap=200,
    )
