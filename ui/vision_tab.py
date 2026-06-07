from __future__ import annotations

import gradio as gr

from core.app_state import APP_STATE, emit_inference_response
from core.events import Event, EventType
from core.tab_feedback import emit_tab_error, status_ok
from models.model_catalog import ModelInfo, model_choices, model_summary
from models.service_factory import BACKENDS, create_vision_service


def build_vision_tab(catalog: dict[str, ModelInfo]) -> None:
    vision_models = model_choices(catalog, "vision")
    if not vision_models:
        vision_models = [mid for mid, model in catalog.items() if model.type == "omnimodal"]
    default_model = vision_models[0]

    with gr.Row():
        model_id = gr.Dropdown(vision_models, value=default_model, label="Vision model")
        backend = gr.Dropdown(BACKENDS, value="placeholder", label="Backend")
        thinking = gr.Checkbox(label="Thinking mode", value=False)

    image = gr.Image(type="pil", label="Image")
    prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe or ask about the image...")
    model_meta = gr.JSON(model_summary(catalog[default_model]), label="Model card")
    run = gr.Button("Run vision", variant="primary")
    output = gr.Textbox(label="Response", lines=10)
    status = gr.Markdown(status_ok("Ready."))

    def select_model(selected: str) -> dict:
        return model_summary(catalog[selected])

    def respond(
        selected: str,
        selected_backend: str,
        _thinking: bool,
        img,
        text: str,
    ) -> tuple[str, str]:
        if img is None and not text.strip():
            return (
                "",
                emit_tab_error(
                    "Vision",
                    "Add an image or prompt before running vision.",
                    {"model_id": selected, "backend": selected_backend},
                ),
            )
        APP_STATE.emit(
            Event(
                EventType.INFERENCE_REQUEST,
                {
                    "mode": "vision",
                    "model_id": selected,
                    "backend": selected_backend,
                    "has_image": img is not None,
                    "thinking": _thinking,
                    "prompt_chars": len(text),
                },
            )
        )
        try:
            response = create_vision_service(catalog[selected], selected_backend).vision_chat(
                img is not None,
                text,
                img,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return (
                "",
                emit_tab_error(
                    "Vision",
                    str(exc),
                    {"model_id": selected, "backend": selected_backend},
                ),
            )
        if _thinking:
            response += (
                "\n\nThinking mode requested. Real backend will map this to the model template."
            )
        emit_inference_response("vision", selected, selected_backend, response)
        return response, status_ok("Vision response generated.")

    model_id.change(select_model, model_id, model_meta)
    run.click(respond, [model_id, backend, thinking, image, prompt], [output, status])
