import json
import os
import re
from dataclasses import dataclass

import groq
from dotenv import load_dotenv

from classbuild.llm.exceptions import (
    AuthenticationError,
    EmptyResponseError,
    GroqAPIError,
    GroqConnectionError,
    GroqTimeoutError,
    InvalidJSONError,
    MissingAPIKeyError,
    RateLimitError,
)

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
REASONING_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_GROQ_KEY_RE = re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a Groq API key with [redacted]."""
    return _GROQ_KEY_RE.sub("[redacted]", text)


@dataclass(frozen=True)
class ChatText:
    text: str
    finish_reason: str | None  # "length" when the token limit cut the answer


class GroqClient:
    def __init__(self, api_key=None, model=None, timeout_seconds=60, max_retries=None):
        from classbuild.config import get_settings  # local import to avoid circular deps

        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.model = (model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)).strip()
        if not self.api_key:
            raise MissingAPIKeyError("GROQ_API_KEY is missing. Add it to your .env file.")

        settings = get_settings()
        effective_retries = max_retries if max_retries is not None else settings.max_retries

        self._client = groq.Groq(
            api_key=self.api_key,
            timeout=timeout_seconds,
            max_retries=effective_retries,
        )

    def _error_detail(self, exc) -> str:
        """Human-readable detail from an APIStatusError, with secrets redacted."""
        try:
            body = exc.body or {}
            message = (body.get("error") or {}).get("message") or str(exc)
        except Exception:
            message = str(exc)
        return redact_secrets(message)

    def _create(self, messages, max_completion_tokens, reasoning_effort="low",
                response_format=None):
        """One chat request. Every SDK error becomes one of our clear errors."""
        extra = {}
        if self.model in REASONING_MODELS and reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            extra["response_format"] = response_format
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                **extra,
            )
        except groq.APITimeoutError as exc:
            raise GroqTimeoutError("The request to Groq timed out. Try again.") from exc
        except groq.APIConnectionError as exc:
            raise GroqConnectionError("Could not reach Groq. Check your internet.") from exc
        except groq.AuthenticationError as exc:
            raise AuthenticationError("The API key was rejected. Check GROQ_API_KEY.") from exc
        except groq.RateLimitError as exc:
            raise RateLimitError("Rate limit reached. Wait a moment and retry.") from exc
        except groq.BadRequestError as exc:
            if "json_validate_failed" in str(exc):
                raise InvalidJSONError("The model did not return valid JSON.") from exc
            raise GroqAPIError(
                f"Groq rejected the request: {redact_secrets(str(exc))}"
            ) from exc
        except groq.APIStatusError as exc:
            raise GroqAPIError(
                f"Groq returned an error (HTTP {exc.status_code}): {self._error_detail(exc)}"
            ) from exc
        return response.choices[0]

    def generate_json(self, prompt, system_prompt, max_completion_tokens=3000, history=None):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])  # earlier chat turns, if any
        messages.append({"role": "user", "content": prompt})

        choice = self._create(
            messages, max_completion_tokens, response_format={"type": "json_object"}
        )
        if choice.finish_reason == "length":
            raise InvalidJSONError("The model's answer was cut off before it finished.")
        text = (choice.message.content or "").strip()
        if not text:
            raise EmptyResponseError("Groq answered with an empty reply.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidJSONError("The model did not return valid JSON.") from exc

    def chat(self, messages, max_completion_tokens=2000):
        """Multi-turn plain text. The last message must be a non-empty user message."""
        last = messages[-1] if messages else {}
        if last.get("role") != "user" or not str(last.get("content", "")).strip():
            raise ValueError("The last message must be a non-empty user message.")

        choice = self._create(messages, max_completion_tokens)
        text = (choice.message.content or "").strip()
        if not text:
            hint = " It was cut off; try a shorter question." if choice.finish_reason == "length" else ""
            raise EmptyResponseError("Groq answered with an empty reply." + hint)
        return ChatText(text=text, finish_reason=choice.finish_reason)
