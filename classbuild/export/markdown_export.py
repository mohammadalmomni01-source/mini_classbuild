"""
classbuild/export/markdown_export.py
Converts a course into one Markdown document.
Pure Python: no Groq, no Streamlit.
"""
import re

OPTION_LETTERS = "ABCD"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _one_line(text: str) -> str:
    """Collapse text to a single line for use in list items."""
    return re.sub(r"\s+", " ", text).strip()


def _safe_filename_part(title: str) -> str:
    """Turn a title into a safe ASCII slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug or "course"


def _sources_line(lesson, source_ids: list[str]) -> str:
    """A section's citation line: 'Sources: p1-c2 (page 3), …'"""
    if not source_ids or not lesson.is_grounded:
        return ""
    parts = []
    for chunk_id in source_ids:
        for source in lesson.grounding.sources:
            if source.chunk_id == chunk_id:
                parts.append(f"{chunk_id} (page {source.page_number})")
                break
        else:
            parts.append(chunk_id)
    return f"*Sources: {', '.join(parts)}*"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def course_filename(title: str) -> str:
    """
    A safe download name from the title.
    Example: 'Introduction to Python' → 'introduction-to-python.md'
    """
    return _safe_filename_part(title) + ".md"


def course_to_markdown(outline, topic: str, lessons: dict, quizzes: dict) -> str:
    """
    Returns the Markdown text for the complete course.
    - Lessons not generated yet are noted rather than skipped.
    - The Answer Key appears only if there is at least one quiz.
    - outline: CourseOutline
    - topic: str (original search topic)
    - lessons: dict {lesson_number: LessonContent}
    - quizzes: dict {lesson_number: Quiz}
    """
    lines: list[str] = []

    # ---- Header ----
    lines.append(f"# {outline.title}")
    lines.append("")
    lines.append(f"**Topic:** {_one_line(topic)}  ")
    lines.append(f"**Difficulty:** {outline.difficulty}  ")
    lines.append(f"**Lessons:** {len(outline.lessons)}")
    lines.append("")
    lines.append(outline.description)
    lines.append("")

    # ---- Course Objectives ----
    lines.append("## Course Objectives")
    lines.append("")
    for obj in outline.learning_objectives:
        lines.append(f"- {_one_line(obj)}")
    lines.append("")

    # ---- Lesson Outline ----
    lines.append("## Lesson Outline")
    lines.append("")
    for lo in outline.lessons:
        lines.append(f"### Lesson {lo.lesson_number}: {lo.title}")
        lines.append("")
        lines.append(_one_line(lo.description))
        lines.append("")
        for obj in lo.learning_objectives:
            lines.append(f"- {_one_line(obj)}")
        lines.append("")

    # ---- Generated Lessons ----
    lines.append("---")
    lines.append("")
    lines.append("## Lessons")
    lines.append("")

    for lo in outline.lessons:
        num = lo.lesson_number
        lines.append(f"## Lesson {num}: {lo.title}")
        lines.append("")

        if num not in lessons:
            lines.append("*This lesson has not been generated yet.*")
            lines.append("")
            continue

        content = lessons[num]

        # Grounding note
        if content.is_grounded:
            pages = ", ".join(str(p) for p in content.cited_pages)
            lines.append(
                f"*Based on {content.grounding.source_filename} · "
                f"cited page(s): {pages}*"
            )
            lines.append("")

        lines.append(content.introduction)
        lines.append("")

        for section in content.sections:
            lines.append(f"### {section.heading}")
            lines.append("")
            lines.append(section.explanation)
            lines.append("")
            lines.append("**Example**")
            lines.append("")
            lines.append(f"```\n{section.example}\n```")
            lines.append("")
            if section.source_ids:
                src = _sources_line(content, section.source_ids)
                if src:
                    lines.append(src)
                    lines.append("")

        lines.append("### Summary")
        lines.append("")
        lines.append(content.summary)
        lines.append("")

        lines.append("### Exercises")
        lines.append("")
        for idx, ex in enumerate(content.exercises, start=1):
            lines.append(f"**Exercise {idx}.** {_one_line(ex.question)}")
            lines.append("")
            lines.append(f"> *Hint:* {_one_line(ex.hint)}")
            lines.append("")

        # PDF sources block
        if content.is_grounded:
            lines.append("#### Sources")
            lines.append("")
            for source in content.grounding.sources:
                lines.append(
                    f"- **{source.chunk_id}** · {source.source_filename} · "
                    f"page {source.page_number} · similarity {source.similarity:.3f}"
                )
            lines.append("")

    # ---- Quizzes (questions only, no answers) ----
    if quizzes:
        lines.append("---")
        lines.append("")
        lines.append("## Quizzes")
        lines.append("")

        answer_key: list[tuple] = []

        for lo in outline.lessons:
            num = lo.lesson_number
            if num not in quizzes:
                continue
            quiz = quizzes[num]
            lines.append(f"### Quiz: Lesson {num} — {lo.title}")
            lines.append("")
            for q_idx, question in enumerate(quiz.questions, start=1):
                stem, _, code = question.question.partition("\n")
                lines.append(f"**{q_idx}. {stem.strip()}**")
                if code.strip():
                    lines.append("")
                    lines.append(f"```python\n{code.strip()}\n```")
                lines.append("")
                for opt_idx, option in enumerate(question.options):
                    letter = OPTION_LETTERS[opt_idx]
                    lines.append(f"{letter}. {option}")
                lines.append("")
                answer_key.append((num, q_idx, question))
            lines.append("")

        # ---- Answer Key ----
        lines.append("---")
        lines.append("")
        lines.append("## Answer Key")
        lines.append("")
        for lesson_num, q_idx, question in answer_key:
            correct_letter = OPTION_LETTERS[question.correct_answer]
            correct_text = question.options[question.correct_answer]
            lines.append(
                f"**Lesson {lesson_num}, Q{q_idx}:** "
                f"{correct_letter}. {_one_line(correct_text)}"
            )
            lines.append(f"> {_one_line(question.explanation)}")
            lines.append("")

    return "\n".join(lines)
