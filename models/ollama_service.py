from __future__ import annotations

import base64
import shutil
from collections.abc import Callable
from io import BytesIO
from typing import Any

import requests

from models.base import BackendStatus
from models.model_catalog import ModelInfo


class OllamaService:
    """Ollama-backed local inference service."""

    base_url = "http://127.0.0.1:11434"

    def __init__(self, model: ModelInfo, timeout_seconds: float = 60) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def status(
        which_func: Callable[[str], str | None] = shutil.which,
        get_func: Callable[..., requests.Response] = requests.get,
    ) -> BackendStatus:
        if which_func("ollama") is None:
            return BackendStatus("ollama", False, "Ollama executable was not found on PATH.")

        try:
            response = get_func(f"{OllamaService.base_url}/api/tags", timeout=2)
        except requests.RequestException as exc:
            return BackendStatus("ollama", False, f"Ollama is installed but not reachable: {exc}")

        if response.ok:
            return BackendStatus("ollama", True, "Ollama is installed and reachable.")
        return BackendStatus(
            "ollama",
            False,
            f"Ollama responded with HTTP {response.status_code}.",
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model.hf_id,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._post_chat(payload)
        return str(data.get("message", {}).get("content", ""))

    def vision_chat(self, has_image: bool, prompt: str, image=None) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if has_image and image is not None:
            message["images"] = [self._encode_image(image)]

        payload = {
            "model": self.model.hf_id,
            "stream": False,
            "messages": [message],
        }
        data = self._post_chat(payload)
        return str(data.get("message", {}).get("content", ""))

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = self.status()
        if not status.available:
            return {
                "message": {
                    "content": (
                        "[Ollama unavailable]\n\n"
                        f"{status.detail}\n\n"
                        "Install/start Ollama and pull the selected model explicitly "
                        "before retrying."
                    )
                }
            }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return {"message": {"content": f"[Ollama request failed]\n\n{exc}"}}

        return dict(response.json())

    @staticmethod
    def _encode_image(image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
