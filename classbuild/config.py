"""
classbuild/config.py
Application-level settings read from environment variables.
"""
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_RETRIES = 1
MAX_ALLOWED_RETRIES = 3


@dataclass(frozen=True)
class Settings:
    max_retries: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read GROQ_MAX_RETRIES; invalid text falls back to 1; value is clamped to 0–3."""
    raw = os.getenv("GROQ_MAX_RETRIES", "").strip()
    try:
        value = int(raw)
    except (ValueError, TypeError):
        value = DEFAULT_MAX_RETRIES
    value = max(0, min(MAX_ALLOWED_RETRIES, value))
    return Settings(max_retries=value)
