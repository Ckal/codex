from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from models.base import BackendStatus
from models.model_catalog import ModelInfo
from models.response_parsing import extract_chat_response
from models.server_helpers import request_openai_chat_text


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

        return request_openai_chat_text(
            post_func=self.post_func,
            base_url=self.config.api_base,
            model=self.config.request_model(self.model.hf_id),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
            failure_label="OpenAI-compatible",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

    @staticmethod
    def _extract_response(data: dict[str, Any]) -> str:
        return extract_chat_response(data)
