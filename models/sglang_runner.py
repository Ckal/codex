from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import requests

from models.base import BackendStatus
from models.http_chat import post_chat_completion
from models.model_catalog import ModelInfo
from models.response_parsing import extract_chat_response


@dataclass(frozen=True)
class SGLangConfig:
    base_url: str = "http://127.0.0.1:30000"
    host: str = "127.0.0.1"
    port: int = 30000
    tp_size: int = 1
    tool_parser: str = "minicpm"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: float = 120


@dataclass(frozen=True)
class SGLangRunPlan:
    start_command: list[str]
    stop_url: str
    health_url: str
    chat_url: str
    startup_downloads: bool
    auto_model_load: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SGLangService:
    """SGLang OpenAI-compatible client and local server command planner."""

    def __init__(
        self,
        model: ModelInfo,
        config: SGLangConfig | None = None,
        get_func: Callable[..., requests.Response] = requests.get,
        post_func: Callable[..., requests.Response] = requests.post,
    ) -> None:
        self.model = model
        self.config = config or SGLangConfig()
        self.get_func = get_func
        self.post_func = post_func

    @staticmethod
    def status(
        base_url: str = "http://127.0.0.1:30000",
        which_func: Callable[[str], str | None] = shutil.which,
        find_spec: Callable[[str], object | None] = importlib.util.find_spec,
        get_func: Callable[..., requests.Response] = requests.get,
    ) -> BackendStatus:
        has_cli = which_func("python") is not None and find_spec("sglang") is not None
        try:
            response = get_func(f"{base_url.rstrip('/')}/health", timeout=2)
        except requests.RequestException as exc:
            detail = f"SGLang server is not reachable: {exc}"
            if not has_cli:
                detail += " Python package sglang is also not installed."
            return BackendStatus("sglang", False, detail)
        if response.ok:
            return BackendStatus("sglang", True, "SGLang server is reachable.")
        return BackendStatus("sglang", False, f"SGLang responded with HTTP {response.status_code}.")

    def run_plan(self) -> SGLangRunPlan:
        return build_sglang_run_plan(self.model, self.config)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        status = self.status(self.config.base_url, get_func=self.get_func)
        if not status.available:
            return (
                "[SGLang unavailable]\n\n"
                f"{status.detail}\n\n"
                "Install SGLang, start the planned local server command, then retry."
            )

        try:
            data = post_chat_completion(
                post_func=self.post_func,
                url=f"{self.config.base_url.rstrip('/')}/v1/chat/completions",
                model=self.model.hf_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            return f"[SGLang request failed]\n\n{exc}"
        return extract_chat_response(data)

    def stop_server(self) -> str:
        try:
            response = self.post_func(
                f"{self.config.base_url.rstrip('/')}/shutdown",
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return f"[SGLang stop failed]\n\n{exc}"
        return "SGLang shutdown request sent."


def build_sglang_run_plan(
    model: ModelInfo,
    config: SGLangConfig | None = None,
) -> SGLangRunPlan:
    cfg = config or SGLangConfig()
    base_url = cfg.base_url.rstrip("/")
    return SGLangRunPlan(
        start_command=[
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model.hf_id,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "--tp-size",
            str(cfg.tp_size),
            "--tool-call-parser",
            cfg.tool_parser,
        ],
        stop_url=f"{base_url}/shutdown",
        health_url=f"{base_url}/health",
        chat_url=f"{base_url}/v1/chat/completions",
        startup_downloads=False,
        auto_model_load=False,
        notes=[
            "The workbench does not start SGLang on app startup.",
            "Run the command explicitly after installing SGLang and choosing hardware.",
            "MiniCPM tool parsing is configured through --tool-call-parser.",
        ],
    )
