from __future__ import annotations

import os

import gradio as gr

from plant.app import APP_CSS, build_app

os.environ.setdefault("WORKBENCH_DEPLOYMENT", "space")

demo = build_app(model_mode="openbmb")

if __name__ == "__main__":
    demo.launch(
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
        css=APP_CSS,
        mcp_server=True,
    )
