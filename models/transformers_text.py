from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, cast

from models.base import BackendStatus
from models.hf_components import load_tokenizer_and_causal_lm
from models.model_catalog import ModelInfo


@dataclass(frozen=True)
class TransformersTextConfig:
    trust_remote_code: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True


class TransformersTextService:
    """Optional Transformers text backend with lazy model loading."""

    def __init__(
        self,
        model: ModelInfo,
        config: TransformersTextConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TransformersTextConfig(
            trust_remote_code=model.trust_remote_code
        )
        self._model = None
        self._tokenizer = None

    @staticmethod
    def status() -> BackendStatus:
        if importlib.util.find_spec("transformers") is None:
            return BackendStatus(
                "transformers",
                False,
                "Python package transformers is not installed in the current environment.",
            )
        return BackendStatus("transformers", True, "Transformers package is installed.")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        status = self.status()
        if not status.available:
            return (
                "[Transformers unavailable]\n\n"
                f"{status.detail}\n\n"
                "Install transformers/torch and select this backend only when local hardware "
                "can load the chosen model."
            )

        model, tokenizer = self._load_components()
        prompt = self._format_chat_prompt(tokenizer, system_prompt, user_prompt)
        encoded = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**encoded, **self.generation_kwargs())
        decoded = cast(str, tokenizer.decode(outputs[0], skip_special_tokens=True))
        return decoded[len(prompt) :].strip() or decoded.strip()

    def stream_chat(self, system_prompt: str, user_prompt: str) -> list[str]:
        response = self.chat(system_prompt, user_prompt)
        return [token for token in response.split(" ") if token]

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
        }

    def _load_components(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        self._model, self._tokenizer = load_tokenizer_and_causal_lm(
            self.model,
            self.config.trust_remote_code,
            self.config.device_map,
            self.config.torch_dtype,
        )
        return self._model, self._tokenizer

    @staticmethod
    def _format_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(rendered)

        parts = [f"{message['role']}: {message['content']}" for message in messages]
        parts.append("assistant:")
        return "\n".join(parts)
