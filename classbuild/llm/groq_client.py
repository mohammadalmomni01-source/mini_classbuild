import json
import os

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


class GroqClient:
    def __init__(self, api_key=None, model=None, timeout_seconds=60):
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.model = (model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)).strip()
        if not self.api_key:
            raise MissingAPIKeyError("GROQ_API_KEY is missing. Add it to your .env file.")
        self._client = groq.Groq(api_key=self.api_key, timeout=timeout_seconds)

    def generate_json(self, prompt, system_prompt, max_completion_tokens=3000):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        extra = {"reasoning_effort": "low"} if self.model in REASONING_MODELS else {}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
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
            raise GroqAPIError(f"Groq rejected the request: {exc}") from exc
        except groq.APIStatusError as exc:
            raise GroqAPIError(f"Groq returned an error (HTTP {exc.status_code}).") from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise InvalidJSONError("The model's answer was cut off before it finished.")
        text = (choice.message.content or "").strip()
        if not text:
            raise EmptyResponseError("Groq answered with an empty reply.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidJSONError("The model did not return valid JSON.") from exc