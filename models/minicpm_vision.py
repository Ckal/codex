from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, cast

from models.base import BackendStatus
from models.hf_components import load_processor_and_image_text_model
from models.model_catalog import ModelInfo


@dataclass(frozen=True)
class MiniCPMVisionConfig:
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.2
    do_sample: bool = False


class MiniCPMVisionService:
    """Optional MiniCPM vision backend using Transformers image-text-to-text APIs."""

    def __init__(
        self,
        model: ModelInfo,
        config: MiniCPMVisionConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or MiniCPMVisionConfig(
            trust_remote_code=model.trust_remote_code or True
        )
        self._model = None
        self._processor = None

    @staticmethod
    def status() -> BackendStatus:
        if importlib.util.find_spec("transformers") is None:
            return BackendStatus(
                "transformers-vision",
                False,
                "Python package transformers is not installed in the current environment.",
            )
        return BackendStatus(
            "transformers-vision",
            True,
            "Transformers package is installed; vision model loads only when selected.",
        )

    def vision_chat(self, has_image: bool, prompt: str, image=None) -> str:
        if not has_image:
            return "[MiniCPM vision unavailable]\n\nAdd an image before running this backend."

        status = self.status()
        if not status.available:
            return (
                "[MiniCPM vision unavailable]\n\n"
                f"{status.detail}\n\n"
                "Install transformers/torch and select this backend only when local hardware "
                "can load the chosen vision model."
            )

        model, processor = self._load_components()
        messages = self.format_messages(prompt, image, thinking=self.model.thinking_mode)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        outputs = model.generate(**inputs, **self.generation_kwargs())
        decoded = cast(str, processor.decode(outputs[0], skip_special_tokens=True))
        return decoded.strip()

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
        }

    @staticmethod
    def format_messages(prompt: str, image, thinking: bool = False) -> list[dict[str, Any]]:
        text = prompt.strip() or "Describe the image."
        if thinking:
            text = f"{text}\nThink carefully before answering."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    @staticmethod
    def video_support_plan() -> dict[str, object]:
        return {
            "implemented": False,
            "reason": "Current Gradio tab accepts one image. Video requires frame sampling first.",
            "next_steps": [
                "Add video upload input.",
                "Sample frames locally without uploading media.",
                "Pass frame list through the model-specific processor template.",
                "Add tests with synthetic frame placeholders before enabling execution.",
            ],
        }

    def _load_components(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        self._model, self._processor = load_processor_and_image_text_model(
            self.model,
            self.config.trust_remote_code,
            self.config.device_map,
            self.config.torch_dtype,
        )
        return self._model, self._processor
