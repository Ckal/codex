from __future__ import annotations

from typing import cast

import gradio as gr

from core.app_logging import configure_app_logging
from models.model_catalog import load_model_catalog
from ui.agent_tab import build_agent_tab
from ui.chat_tab import build_chat_tab
from ui.dataset_tab import build_dataset_tab
from ui.export_tab import build_export_tab
from ui.notes_tab import build_notes_tab
from ui.status_tab import build_status_tab
from ui.traces_tab import build_traces_tab
from ui.train_tab import build_train_tab
from ui.vision_tab import build_vision_tab

APP_THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
APP_CSS = """
.app-title { margin-bottom: 0.25rem; }
.app-subtitle { color: var(--body-text-color-subdued); margin-top: 0; }
"""


def build_app() -> gr.Blocks:
    configure_app_logging()
    catalog = load_model_catalog("config/models.yaml")

    with gr.Blocks(title="OpenBMB Local AI Workbench") as demo:
        gr.Markdown(
            """
            # OpenBMB Local AI Workbench
            Small-model experimentation for Gradio + Hugging Face Spaces.
            """,
            elem_classes=["app-title"],
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                build_chat_tab(catalog)
            with gr.Tab("Vision"):
                build_vision_tab(catalog)
            with gr.Tab("Dataset"):
                build_dataset_tab()
            with gr.Tab("Train"):
                build_train_tab()
            with gr.Tab("Export"):
                build_export_tab(catalog)
            with gr.Tab("Field Notes"):
                build_notes_tab(catalog)
            with gr.Tab("Traces"):
                build_traces_tab()
            with gr.Tab("Agent"):
                build_agent_tab()
            with gr.Tab("Status"):
                build_status_tab(catalog)

    return cast(gr.Blocks, demo)


if __name__ == "__main__":
    build_app().launch(server_port=7860, share=False, theme=APP_THEME, css=APP_CSS)
