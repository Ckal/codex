from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from models.base import BackendStatus
from models.http_chat import post_chat_completion
from models.model_catalog import ModelInfo
from models.response_parsing import extract_chat_response


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str = "http://127.0.0.1:1234"
    api_key: str = "lm-studio"
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: float = 120

    @property
    def api_base(self) -> str:
        return self.base_url.rstrip("/")

    def request_model(self, fallback: str) -> str:
        return self.model_name.strip() or fallback


class OpenAICompatibleService:
    """Local OpenAI-compatible chat client for LM Studio, vLLM, or similar servers."""

    def __init__(
        self,
        model: ModelInfo,
        config: OpenAICompatibleConfig | None = None,
        get_func: Callable[..., requests.Response] = requests.get,
        post_func: Callable[..., requests.Response] = requests.post,
    ) -> None:
        self.model = model
        self.config = config or OpenAICompatibleConfig()
        self.get_func = get_func
        self.post_func = post_func

    @staticmethod
    def status(
        base_url: str = "http://127.0.0.1:1234",
        get_func: Callable[..., requests.Response] = requests.get,
    ) -> BackendStatus:
        url = base_url.rstrip("/")
        try:
            response = get_func(f"{url}/v1/models", timeout=2)
            response.raise_for_status()
        except requests.RequestException as exc:
            return BackendStatus(
                "openai-compatible",
                False,
                f"OpenAI-compatible local server is not reachable at {url}: {exc}",
            )
        return BackendStatus(
            "openai-compatible",
            True,
            f"OpenAI-compatible local server is reachable at {url}.",
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        status = self.status(self.config.api_base, self.get_func)
        if not status.available:
            return (
                "[OpenAI-compatible backend unavailable]\n\n"
                f"{status.detail}\n\n"
                "Start LM Studio or another local OpenAI-compatible server before retrying."
            )

        try:
            data = post_chat_completion(
                self.post_func,
                f"{self.config.api_base}/v1/chat/completions",
                self.config.request_model(self.model.hf_id),
                system_prompt,
                user_prompt,
                self.config.temperature,
                self.config.max_tokens,
                self.config.timeout_seconds,
                {"Authorization": f"Bearer {self.config.api_key}"},
            )
        except requests.RequestException as exc:
            return f"[OpenAI-compatible request failed]\n\n{exc}"
        return self._extract_response(data)

    @staticmethod
    def _extract_response(data: dict[str, Any]) -> str:
        return extract_chat_response(data)
