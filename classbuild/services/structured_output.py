import logging

from pydantic import ValidationError

from classbuild.llm import InvalidJSONError

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # first try + one retry


class CourseGenerationError(Exception):
    """Generation problems that are not Groq API errors."""


class StructuredGenerationError(CourseGenerationError):
    """No usable JSON even after the retry."""


class ContentValidationError(CourseGenerationError):
    """The JSON was valid but broke a schema rule."""

    def __init__(self, message, problems):
        super().__init__(message)
        self.problems = problems


def describe_validation_error(exc):
    return [
        f"{'.'.join(str(p) for p in e['loc']) or 'content'}: {e['msg']}"
        for e in exc.errors()
    ]


def generate_validated(client, *, prompt, system_prompt, max_completion_tokens, validate, what):
    """Ask Groq for JSON, validate it, and retry once if something is wrong."""
    current_prompt = prompt
    problems = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            data = client.generate_json(
                prompt=current_prompt,
                system_prompt=system_prompt,
                max_completion_tokens=max_completion_tokens,
            )
        except InvalidJSONError as exc:
            logger.warning("Attempt %d/%d gave bad JSON: %s", attempt, MAX_ATTEMPTS, exc)
            problems = []
            continue

        try:
            return validate(data)
        except ValidationError as exc:
            problems = describe_validation_error(exc)
            logger.warning(
                "Attempt %d/%d failed validation: %s", attempt, MAX_ATTEMPTS, problems
            )
            current_prompt = (
                prompt
                + "\n\nYour previous answer was rejected for these reasons:\n- "
                + "\n- ".join(problems)
                + "\nReturn a corrected JSON object."
            )

    if problems:
        raise ContentValidationError(f"The AI's {what} did not match your request.", problems)
    raise StructuredGenerationError(
        f"The AI couldn't produce a valid {what} after {MAX_ATTEMPTS} attempts. Please try again."
    )