from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import requests

from models.base import BackendStatus
from models.model_catalog import ModelInfo
from models.server_helpers import host_port_args, request_openai_chat_text
from tracking.trackio_client import TrackingClient


@dataclass(frozen=True)
class VLLMConfig:
    base_url: str = "http://127.0.0.1:8000"
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: int = 4096
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: float = 120


@dataclass(frozen=True)
class VLLMRunPlan:
    start_command: list[str]
    stop_note: str
    health_url: str
    metrics_url: str
    chat_url: str
    startup_downloads: bool
    auto_model_load: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMService:
    """vLLM OpenAI-compatible client and local server command planner."""

    def __init__(
        self,
        model: ModelInfo,
        config: VLLMConfig | None = None,
        get_func: Callable[..., requests.Response] = requests.get,
        post_func: Callable[..., requests.Response] = requests.post,
    ) -> None:
        self.model = model
        self.config = config or VLLMConfig()
        self.get_func = get_func
        self.post_func = post_func

    @staticmethod
    def status(
        base_url: str = "http://127.0.0.1:8000",
        which_func: Callable[[str], str | None] = shutil.which,
        find_spec: Callable[[str], object | None] = importlib.util.find_spec,
        get_func: Callable[..., requests.Response] = requests.get,
    ) -> BackendStatus:
        has_vllm = which_func("vllm") is not None or find_spec("vllm") is not None
        try:
            response = get_func(f"{base_url.rstrip('/')}/health", timeout=2)
        except requests.RequestException as exc:
            detail = f"vLLM server is not reachable: {exc}"
            if not has_vllm:
                detail += " Python package vllm or vllm CLI is also not installed."
            return BackendStatus("vllm", False, detail)
        if response.ok:
            return BackendStatus("vllm", True, "vLLM server is reachable.")
        return BackendStatus("vllm", False, f"vLLM responded with HTTP {response.status_code}.")

    def run_plan(self) -> VLLMRunPlan:
        return build_vllm_run_plan(self.model, self.config)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        status = self.status(self.config.base_url, get_func=self.get_func)
        if not status.available:
            return (
                "[vLLM unavailable]\n\n"
                f"{status.detail}\n\n"
                "Install vLLM, start the planned local server command, then retry."
            )
        cfg = self.config
        sampling = (cfg.temperature, cfg.max_tokens, cfg.timeout_seconds)
        return request_openai_chat_text(
            self.post_func,
            cfg.base_url,
            self.model.hf_id,
            system_prompt,
            user_prompt,
            *sampling,
            "vLLM",
        )


def build_vllm_run_plan(model: ModelInfo, config: VLLMConfig | None = None) -> VLLMRunPlan:
    cfg = config or VLLMConfig()
    base_url = cfg.base_url.rstrip("/")
    return VLLMRunPlan(
        start_command=[
            "vllm",
            "serve",
            model.hf_id,
            *host_port_args(cfg.host, cfg.port),
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
            "--dtype",
            cfg.dtype,
            "--max-model-len",
            str(cfg.max_model_len),
        ],
        stop_note="Stop the foreground vLLM process with Ctrl+C or your process manager.",
        health_url=f"{base_url}/health",
        metrics_url=f"{base_url}/metrics",
        chat_url=f"{base_url}/v1/chat/completions",
        startup_downloads=False,
        auto_model_load=False,
    )


def parse_vllm_metrics(metrics_text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        metric_name = name.split("{", 1)[0]
        try:
            parsed[metric_name] = float(value.strip())
        except ValueError:
            continue
    return parsed


def fetch_vllm_metrics(
    base_url: str = "http://127.0.0.1:8000",
    get_func: Callable[..., requests.Response] = requests.get,
) -> dict[str, float]:
    response = get_func(f"{base_url.rstrip('/')}/metrics", timeout=5)
    response.raise_for_status()
    return parse_vllm_metrics(response.text)


def log_vllm_benchmark(
    metrics: dict[str, float],
    model_id: str,
    client: TrackingClient | None = None,
) -> str:
    tracking_client = client or TrackingClient()
    path = tracking_client.log(
        "vllm_benchmark",
        {
            "model_id": model_id,
            "metrics": metrics,
        },
    )
    return str(path)
