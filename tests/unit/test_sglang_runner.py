from __future__ import annotations

import unittest
from typing import Any, cast

import requests

from models.model_catalog import load_model_catalog
from models.sglang_runner import SGLangConfig, SGLangService, build_sglang_run_plan


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _json_response(payload: dict[str, Any], status_code: int = 200) -> requests.Response:
    return cast(requests.Response, FakeResponse(payload, status_code))


class CapturingPost:
    def __init__(self) -> None:
        self.url = ""
        self.payload: dict[str, Any] = {}

    def __call__(self, url: str, **kwargs: Any) -> requests.Response:
        self.url = url
        self.payload = dict(kwargs.get("json", {}))
        return _json_response({"choices": [{"message": {"content": "sglang answer"}}]})


class SGLangRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_model_catalog("config/models.yaml")["minicpm5_1b"]

    def test_builds_start_stop_health_and_chat_plan(self) -> None:
        plan = build_sglang_run_plan(
            self.model,
            SGLangConfig(host="127.0.0.1", port=31000, tp_size=2),
        )

        self.assertIn("sglang.launch_server", plan.start_command)
        self.assertIn("--tool-call-parser", plan.start_command)
        self.assertEqual(plan.health_url, "http://127.0.0.1:30000/health")
        self.assertFalse(plan.startup_downloads)
        self.assertFalse(plan.auto_model_load)

    def test_status_reports_reachable_server(self) -> None:
        def get_health(url: str, **kwargs: Any) -> requests.Response:
            self.assertEqual(url, "http://local-sglang/health")
            self.assertEqual(kwargs["timeout"], 2)
            return _json_response({"status": "ok"})

        status = SGLangService.status(
            "http://local-sglang",
            which_func=lambda name: "python" if name == "python" else None,
            find_spec=lambda name: object() if name == "sglang" else None,
            get_func=get_health,
        )

        self.assertTrue(status.available)

    def test_status_reports_unreachable_server_and_missing_package(self) -> None:
        def get_health(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise requests.ConnectionError("offline")

        status = SGLangService.status(
            "http://local-sglang",
            which_func=lambda name: None,
            find_spec=lambda name: None,
            get_func=get_health,
        )

        self.assertFalse(status.available)
        self.assertIn("offline", status.detail)
        self.assertIn("not installed", status.detail)

    def test_chat_posts_openai_compatible_payload(self) -> None:
        def get_health(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            return _json_response({"status": "ok"})

        post_chat = CapturingPost()
        service = SGLangService(
            self.model,
            SGLangConfig(base_url="http://local-sglang"),
            get_func=get_health,
            post_func=post_chat,
        )

        response = service.chat("system", "prompt")

        self.assertEqual(response, "sglang answer")
        self.assertEqual(post_chat.url, "http://local-sglang/v1/chat/completions")
        self.assertEqual(post_chat.payload["model"], "openbmb/MiniCPM5-1B")
        self.assertEqual(post_chat.payload["messages"][1]["content"], "prompt")

    def test_unavailable_chat_does_not_post(self) -> None:
        def get_health(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise requests.ConnectionError("offline")

        def post_chat(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise AssertionError("chat should not post when health fails")

        service = SGLangService(
            self.model,
            SGLangConfig(base_url="http://local-sglang"),
            get_func=get_health,
            post_func=post_chat,
        )

        response = service.chat("system", "prompt")

        self.assertIn("[SGLang unavailable]", response)

    def test_stop_server_posts_shutdown(self) -> None:
        post_chat = CapturingPost()
        service = SGLangService(
            self.model,
            SGLangConfig(base_url="http://local-sglang"),
            post_func=post_chat,
        )

        response = service.stop_server()

        self.assertEqual(response, "SGLang shutdown request sent.")
        self.assertEqual(post_chat.url, "http://local-sglang/shutdown")


if __name__ == "__main__":
    unittest.main()
