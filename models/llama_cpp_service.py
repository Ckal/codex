from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from models.base import BackendStatus
from models.model_catalog import ModelInfo
from models.response_parsing import extract_chat_response


@dataclass(frozen=True)
class LlamaCppConfig:
    """Runtime configuration for a local llama.cpp server."""

    server_url: str = "http://127.0.0.1:8080"
    model_path: str = ""
    mmproj_path: str = ""


class LlamaCppService:
    """llama.cpp HTTP client for local GGUF inference."""

    def __init__(
        self,
        model: ModelInfo,
        config: LlamaCppConfig | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self.model = model
        self.config = config or LlamaCppConfig()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def status(
        which_func: Callable[[str], str | None] = shutil.which,
        get_func: Callable[..., requests.Response] = requests.get,
        server_url: str = "http://127.0.0.1:8080",
    ) -> BackendStatus:
        if which_func("llama-server") is None:
            return BackendStatus("llama.cpp", False, "llama-server was not found on PATH.")

        try:
            response = get_func(f"{server_url}/health", timeout=2)
        except requests.RequestException as exc:
            return BackendStatus(
                "llama.cpp",
                False,
                f"llama-server is installed but not reachable: {exc}",
            )

        if response.ok:
            return BackendStatus("llama.cpp", True, "llama-server is installed and reachable.")
        return BackendStatus(
            "llama.cpp",
            False,
            f"llama-server responded with HTTP {response.status_code}.",
        )

    def launch_command(self) -> list[str]:
        if not self.config.model_path:
            return []

        command = ["llama-server", "-m", self.config.model_path]
        if self.config.mmproj_path:
            command.extend(["--mmproj", self.config.mmproj_path])
        return command

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._post_chat(messages)

    def vision_chat(self, has_image: bool, prompt: str, image=None) -> str:
        del image
        if has_image:
            return (
                "[llama.cpp vision note]\n\n"
                "Image upload requires a running llama-server with an mmproj file. "
                "The current scaffold validates the server path but does not yet serialize "
                "Gradio images into llama.cpp multimodal payloads."
            )
        return self._post_chat([{"role": "user", "content": prompt}])

    def _post_chat(self, messages: list[dict[str, str]]) -> str:
        status = self.status(server_url=self.config.server_url)
        if not status.available:
            return (
                "[llama.cpp unavailable]\n\n"
                f"{status.detail}\n\n"
                "Install llama.cpp, start llama-server with an explicit GGUF model, "
                "then retry."
            )

        try:
            response = requests.post(
                f"{self.config.server_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return f"[llama.cpp request failed]\n\n{exc}"

        return self._extract_response(dict(response.json()))

    @staticmethod
    def _extract_response(data: dict[str, Any]) -> str:
        return extract_chat_response(data)


def local_file_status(path: str) -> str:
    if not path:
        return "not configured"
    return "found" if Path(path).exists() else "missing"
