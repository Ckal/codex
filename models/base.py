from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from models.model_catalog import ModelInfo


@dataclass(frozen=True)
class BackendStatus:
    """Current availability of an inference backend."""

    name: str
    available: bool
    detail: str


class TextModelService(Protocol):
    """Text chat service contract."""

    model: ModelInfo

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Return a text response."""


class VisionModelService(Protocol):
    """Vision chat service contract."""

    model: ModelInfo

    def vision_chat(self, has_image: bool, prompt: str, image=None) -> str:
        """Return a vision-language response."""
