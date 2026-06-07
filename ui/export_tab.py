from __future__ import annotations

import gradio as gr

from models.model_catalog import ModelInfo
from training.export import (
    QUANTIZATION_CHOICES,
    build_export_plan,
    list_exported_files,
)


def build_export_tab(catalog: dict[str, ModelInfo]) -> None:
    model_id = gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Model")
    quant = gr.Dropdown(QUANTIZATION_CHOICES, value="Q4_K_M", label="Quantization")
    output_dir = gr.Textbox(label="Output directory", value="exports")
    run = gr.Button("Prepare export plan", variant="primary")
    refresh = gr.Button("Refresh exported files")
    output = gr.JSON(label="Export plan")
    files = gr.Dataframe(headers=["path", "bytes"], label="Exported files", interactive=False)

    def plan_export(
        selected: str,
        selected_quant: str,
        directory: str,
    ) -> tuple[dict, list[list[str]]]:
        model = catalog[selected]
        plan = build_export_plan(model, selected_quant, directory)
        return plan.as_dict(), list_exported_files(directory)

    run.click(plan_export, [model_id, quant, output_dir], [output, files])
    refresh.click(list_exported_files, output_dir, files)
