from __future__ import annotations

from models.model_catalog import ModelInfo


class PlaceholderModelService:
    """Deterministic scaffold service used until a real backend is configured."""

    def __init__(self, model: ModelInfo) -> None:
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return (
            f"[Placeholder response from {self.model.display_name}]\n\n"
            "Real inference is not wired yet. Configure Ollama, llama.cpp, Transformers, "
            "or SGLang before using this as a hackathon demo.\n\n"
            f"System prompt: {system_prompt or '(none)'}\n"
            f"User prompt: {user_prompt}"
        )

    def vision_chat(self, has_image: bool, prompt: str, image=None) -> str:
        del image
        image_state = "an uploaded image" if has_image else "no image"
        return (
            f"[Placeholder vision response from {self.model.display_name}]\n\n"
            f"I received {image_state}. Real MiniCPM-V inference is not wired yet.\n\n"
            f"Prompt: {prompt}"
        )
