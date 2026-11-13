import unittest
from pydantic import BaseModel

from classbuild.llm import InvalidJSONError
from classbuild.services.structured_output import (
    StructuredGenerationError,
    generate_validated,
)


class DummyModel(BaseModel):
    message: str


class FakeGroq:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class StructuredOutputTests(unittest.TestCase):
    def test_invalid_json_triggers_retry(self):
        # 2 failures -> triggers retry and raises StructuredGenerationError
        client = FakeGroq([
            InvalidJSONError("Bad JSON on attempt 1"),
            InvalidJSONError("Bad JSON on attempt 2"),
        ])
        with self.assertRaises(StructuredGenerationError):
            generate_validated(
                client,
                prompt="Hello",
                system_prompt="System",
                max_completion_tokens=100,
                validate=DummyModel.model_validate,
                what="test",
            )
        self.assertEqual(len(client.calls), 2)

    def test_valid_json_on_retry_accepted(self):
        client = FakeGroq([
            InvalidJSONError("Bad JSON on attempt 1"),
            {"message": "Success on attempt 2"},
        ])
        result = generate_validated(
            client,
            prompt="Hello",
            system_prompt="System",
            max_completion_tokens=100,
            validate=DummyModel.model_validate,
            what="test",
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result.message, "Success on attempt 2")


if __name__ == "__main__":
    unittest.main()
