from __future__ import annotations

import unittest

from models.response_parsing import chat_completion_payload, chat_messages, extract_chat_response


class ResponseParsingTest(unittest.TestCase):
    def test_extracts_message_content(self) -> None:
        self.assertEqual(
            extract_chat_response({"choices": [{"message": {"content": "hello"}}]}),
            "hello",
        )

    def test_extracts_text_fallback(self) -> None:
        self.assertEqual(extract_chat_response({"choices": [{"text": "hello"}]}), "hello")

    def test_empty_choices_return_empty_string(self) -> None:
        self.assertEqual(extract_chat_response({"choices": []}), "")

    def test_builds_chat_messages(self) -> None:
        self.assertEqual(
            chat_messages("system", "user"),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        )

    def test_builds_chat_completion_payload(self) -> None:
        payload = chat_completion_payload("model", "system", "user", 0.1, 32)

        self.assertEqual(payload["model"], "model")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(payload["messages"][1]["content"], "user")


if __name__ == "__main__":
    unittest.main()
