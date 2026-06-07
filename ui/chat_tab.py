from __future__ import annotations

import gradio as gr

from core.app_state import APP_STATE, emit_inference_response
from core.events import Event, EventType
from core.tab_feedback import emit_tab_error, status_ok
from models.model_catalog import ModelInfo, model_choices, model_summary
from models.service_factory import BACKENDS, create_text_service
from ui.progress import CLICK_PROGRESS


def build_chat_tab(catalog: dict[str, ModelInfo]) -> None:
    text_models = model_choices(catalog, "text") or list(catalog)
    default_model = text_models[0]

    with gr.Row():
        model_id = gr.Dropdown(text_models, value=default_model, label="Model")
        backend = gr.Dropdown(BACKENDS, value="placeholder", label="Backend")
        model_meta = gr.JSON(model_summary(catalog[default_model]), label="Model card")

    system_prompt = gr.Textbox(label="System prompt", lines=2)
    user_prompt = gr.Textbox(
        label="Prompt",
        lines=6,
        placeholder="Ask the local workbench something...",
    )
    run = gr.Button("Run", variant="primary")
    output = gr.Textbox(label="Response", lines=12)
    status = gr.Markdown(status_ok("Ready."))

    def select_model(selected: str) -> dict:
        return model_summary(catalog[selected])

    def respond(
        selected: str,
        selected_backend: str,
        system: str,
        prompt: str,
    ) -> tuple[str, str]:
        if not prompt.strip():
            return (
                "",
                emit_tab_error(
                    "Chat",
                    "Enter a prompt before running chat.",
                    {"model_id": selected, "backend": selected_backend},
                ),
            )
        APP_STATE.emit(
            Event(
                EventType.INFERENCE_REQUEST,
                {
                    "mode": "text",
                    "model_id": selected,
                    "backend": selected_backend,
                    "prompt_chars": len(prompt),
                },
            )
        )
        try:
            response = create_text_service(catalog[selected], selected_backend).chat(system, prompt)
        except (RuntimeError, ValueError, OSError) as exc:
            return (
                "",
                emit_tab_error(
                    "Chat",
                    str(exc),
                    {"model_id": selected, "backend": selected_backend},
                ),
            )
        emit_inference_response("text", selected, selected_backend, response)
        return response, status_ok("Chat response generated.")

    model_id.change(select_model, model_id, model_meta)
    run.click(
        respond,
        [model_id, backend, system_prompt, user_prompt],
        [output, status],
        show_progress=CLICK_PROGRESS,
    )
