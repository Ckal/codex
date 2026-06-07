from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from models.base import BackendStatus
from models.model_catalog import ModelInfo
from models.response_parsing import extract_chat_response


@dataclass(frozen=True)
class LlamaCppPythonConfig:
    """Runtime configuration for llama-cpp-python."""

    model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    temperature: float = 0.7
    max_tokens: int = 512


class LlamaCppPythonService:
    """Direct llama-cpp-python GGUF inference service."""

    def __init__(
        self,
        model: ModelInfo,
        config: LlamaCppPythonConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or LlamaCppPythonConfig()

    @staticmethod
    def status(model_path: str = "") -> BackendStatus:
        if importlib.util.find_spec("llama_cpp") is None:
            return BackendStatus(
                "llama-cpp-python",
                False,
                "Python package llama-cpp-python is not installed in the current environment.",
            )
        if not model_path:
            return BackendStatus(
                "llama-cpp-python",
                False,
                "llama-cpp-python is installed, but no GGUF model path is configured.",
            )
        if not Path(model_path).exists():
            return BackendStatus(
                "llama-cpp-python",
                False,
                f"Configured GGUF model was not found: {model_path}",
            )
        return BackendStatus("llama-cpp-python", True, "llama-cpp-python is ready.")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        status = self.status(self.config.model_path)
        if not status.available:
            return (
                "[llama-cpp-python unavailable]\n\n"
                f"{status.detail}\n\n"
                "Install llama-cpp-python and configure a local GGUF path before retrying."
            )

        llama = self._load_llama()
        response = llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return self._extract_response(response)

    def vision_chat(self, has_image: bool, prompt: str, image=None) -> str:
        del image
        if has_image:
            return (
                "[llama-cpp-python vision note]\n\n"
                "Direct multimodal llama-cpp-python support requires model-specific mmproj "
                "wiring and image serialization. Use llama-server for the current vision path."
            )
        return self.chat("", prompt)

    def _load_llama(self):
        llama_module = import_module("llama_cpp")
        llama_class = llama_module.Llama

        return llama_class(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )

    @staticmethod
    def _extract_response(data: dict[str, Any]) -> str:
        return extract_chat_response(data)
