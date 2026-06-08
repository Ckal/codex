from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import requests

from models.model_catalog import load_model_catalog
from models.vllm_runner import (
    VLLMConfig,
    VLLMService,
    build_vllm_run_plan,
    fetch_vllm_metrics,
    log_vllm_benchmark,
    parse_vllm_metrics,
)
from tracking.trackio_client import TrackingClient, TrackingConfig, read_trace_rows


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        text: str = "",
        status_code: int = 200,
    ) -> None:
        self.payload = payload or {}
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class CapturingPost:
    def __init__(self) -> None:
        self.url = ""
        self.payload: dict[str, Any] = {}

    def __call__(self, url: str, **kwargs: Any) -> requests.Response:
        self.url = url
        self.payload = dict(kwargs["json"])
        return cast(
            requests.Response,
            FakeResponse({"choices": [{"message": {"content": "vllm answer"}}]}),
        )


class VLLMRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_model_catalog("config/models.yaml")["minicpm5_1b"]

    def test_builds_vllm_run_plan(self) -> None:
        plan = build_vllm_run_plan(
            self.model,
            VLLMConfig(port=8100, tensor_parallel_size=2, max_model_len=2048),
        )

        self.assertEqual(plan.health_url, "http://127.0.0.1:8000/health")
        self.assertIn("serve", plan.start_command)
        self.assertIn("--tensor-parallel-size", plan.start_command)
        self.assertFalse(plan.startup_downloads)

    def test_status_reports_missing_package_and_unreachable_server(self) -> None:
        def get_health(url: str, **kwargs: Any) -> requests.Response:
            del url, kwargs
            raise requests.ConnectionError("offline")

        status = VLLMService.status(
            "http://local-vllm",
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
            return cast(requests.Response, FakeResponse({"status": "ok"}))

        post_chat = CapturingPost()
        service = VLLMService(
            self.model,
            VLLMConfig(base_url="http://local-vllm"),
            get_func=get_health,
            post_func=post_chat,
        )

        answer = service.chat("system", "prompt")

        self.assertEqual(answer, "vllm answer")
        self.assertEqual(post_chat.url, "http://local-vllm/v1/chat/completions")
        self.assertEqual(post_chat.payload["model"], "openbmb/MiniCPM5-1B")

    def test_parses_prometheus_metrics(self) -> None:
        metrics = parse_vllm_metrics(
            "# HELP demo demo\n"
            "vllm:num_requests_running 2\n"
            'vllm:gpu_cache_usage_perc{gpu="0"} 0.42\n'
        )

        self.assertEqual(metrics["vllm:num_requests_running"], 2.0)
        self.assertEqual(metrics["vllm:gpu_cache_usage_perc"], 0.42)

    def test_fetches_metrics(self) -> None:
        def get_metrics(url: str, **kwargs: Any) -> requests.Response:
            self.assertEqual(url, "http://local-vllm/metrics")
            self.assertEqual(kwargs["timeout"], 5)
            return cast(requests.Response, FakeResponse(text="vllm:num_requests_running 1\n"))

        metrics = fetch_vllm_metrics("http://local-vllm", get_metrics)

        self.assertEqual(metrics["vllm:num_requests_running"], 1.0)

    def test_logs_vllm_benchmark_to_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            client = TrackingClient(TrackingConfig(local_path=str(path)))

            saved = log_vllm_benchmark({"latency": 1.2}, "model", client)

            self.assertEqual(saved, str(path))
            self.assertEqual(read_trace_rows(path)[0]["event"], "vllm_benchmark")


if __name__ == "__main__":
    unittest.main()
