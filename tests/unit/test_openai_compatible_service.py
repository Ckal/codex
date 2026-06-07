from __future__ import annotations

import json
import unittest
from typing import Any

import requests

from models.model_catalog import load_model_catalog
from models.openai_compatible_service import (
    OpenAICompatibleConfig,
    OpenAICompatibleService,
)


def _json_response(payload: dict[str, Any], status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    return response


class CapturingPost:
    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}
        self.url = ""

    def __call__(self, url: str, **kwargs: Any) -> requests.Response:
        self.url = url
        self.payload = dict(kwargs["json"])
        return _json_response({"choices": [{"message": {"content": "local answer"}}]})


class OpenAICompatibleServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_model_catalog("config/models.yaml")["minicpm5_1b"]

    def test_status_reports_reachable_server(self) -> None:
        def get_models(url: str, **kwargs: Any) -> requests.Response:
            self.assertEqual(url, "http://local.test/v1/models")
            self.assertEqual(kwargs["timeout"], 2)
            return _json_response({"data": [{"id": "loaded-model"}]})

        status = OpenAICompatibleService.status("http://local.test", get_models)

        self.assertTrue(status.available)
        self.assertEqual(status.name, "openai-compatible")

    def test_status_reports_unreachable_server(self) -> None:
        def get_models(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise requests.ConnectionError("not listening")

        status = OpenAICompatibleService.status("http://local.test", get_models)

        self.assertFalse(status.available)
        self.assertIn("not reachable", status.detail)

    def test_chat_posts_openai_compatible_payload_with_model_override(self) -> None:
        def get_models(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            return _json_response({"data": [{"id": "lm-studio-model"}]})

        post_chat = CapturingPost()
        service = OpenAICompatibleService(
            self.model,
            OpenAICompatibleConfig(
                base_url="http://local.test/",
                model_name="lm-studio-model",
                api_key="local-key",
            ),
            get_func=get_models,
            post_func=post_chat,
        )

        answer = service.chat("be concise", "hello")

        self.assertEqual(answer, "local answer")
        self.assertEqual(post_chat.url, "http://local.test/v1/chat/completions")
        self.assertEqual(post_chat.payload["model"], "lm-studio-model")
        self.assertEqual(post_chat.payload["messages"][0]["role"], "system")
        self.assertEqual(post_chat.payload["messages"][1]["content"], "hello")

    def test_chat_returns_unavailable_message_without_posting(self) -> None:
        def get_models(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise requests.ConnectionError("server down")

        def post_chat(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise AssertionError("chat request should not be sent when status fails")

        service = OpenAICompatibleService(
            self.model,
            OpenAICompatibleConfig(base_url="http://local.test"),
            get_func=get_models,
            post_func=post_chat,
        )

        answer = service.chat("system", "prompt")

        self.assertIn("[OpenAI-compatible backend unavailable]", answer)
        self.assertIn("server down", answer)

    def test_chat_returns_request_failure_message(self) -> None:
        def get_models(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            return _json_response({"data": [{"id": "loaded-model"}]})

        def post_chat(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise requests.Timeout("slow local model")

        service = OpenAICompatibleService(
            self.model,
            OpenAICompatibleConfig(base_url="http://local.test"),
            get_func=get_models,
            post_func=post_chat,
        )

        answer = service.chat("system", "prompt")

        self.assertIn("[OpenAI-compatible request failed]", answer)
        self.assertIn("slow local model", answer)


if __name__ == "__main__":
    unittest.main()
