from __future__ import annotations

import unittest

from models.response_parsing import extract_chat_response


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


if __name__ == "__main__":
    unittest.main()
