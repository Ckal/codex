from __future__ import annotations

from typing import Any

import gradio as gr

from models.model_catalog import ModelInfo


def create_serving_controls(
    catalog: dict[str, ModelInfo],
    backend_name: str,
    default_base_url: str,
    default_port: int,
) -> dict[str, Any]:
    controls: dict[str, Any] = {
        "selected": gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Model config"),
        "base_url": gr.Textbox(label=f"{backend_name} base URL", value=default_base_url),
    }
    with gr.Row():
        controls["host"] = gr.Textbox(label="Host", value="127.0.0.1")
        controls["port"] = gr.Number(label="Port", value=default_port, precision=0)
        controls["parallel"] = gr.Number(label="Parallel size", value=1, precision=0)
    return controls
