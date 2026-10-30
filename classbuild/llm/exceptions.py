class GroqClientError(Exception):
    """Base class for every Groq-related error."""


class MissingAPIKeyError(GroqClientError):
    """GROQ_API_KEY is not set."""


class AuthenticationError(GroqClientError):
    """The API key was rejected."""


class RateLimitError(GroqClientError):
    """Too many requests in a short time."""


class GroqTimeoutError(GroqClientError):
    """The request took too long."""


class GroqConnectionError(GroqClientError):
    """Groq could not be reached."""


class EmptyResponseError(GroqClientError):
    """Groq answered with no text."""


class InvalidJSONError(GroqClientError):
    """The model's answer was not valid JSON."""


class GroqAPIError(GroqClientError):
    """Any other API error."""