from __future__ import annotations

from pathlib import Path

import gradio as gr

from models.model_catalog import ModelInfo
from training.export import (
    QUANTIZATION_CHOICES,
    build_export_plan,
    list_exported_files,
)
from ui.progress import CLICK_PROGRESS


def export_download_paths(plan: dict | None, exported_rows: list[list[str]]) -> list[str]:
    """Return existing exported files for the Gradio download output."""
    existing_paths = [
        Path(row[0])
        for row in exported_rows
        if row and Path(row[0]).is_file()
    ]
    if not plan:
        return [str(path) for path in existing_paths]

    planned_paths = _planned_export_paths(plan)
    planned_keys = {_path_key(path) for path in planned_paths}
    planned_existing = [
        path for path in existing_paths if _path_key(path) in planned_keys
    ]
    other_existing = [
        path for path in existing_paths if _path_key(path) not in planned_keys
    ]
    return [str(path) for path in [*planned_existing, *other_existing]]


def _planned_export_paths(plan: dict) -> list[Path]:
    output_dir = Path(str(plan.get("output_dir") or ""))
    paths: list[Path] = []

    official_file = str(plan.get("official_gguf_file") or "")
    if official_file:
        paths.append(output_dir / official_file)

    convert_command = plan.get("convert_command") or []
    if "--outfile" in convert_command:
        outfile_index = convert_command.index("--outfile") + 1
        if outfile_index < len(convert_command):
            paths.append(Path(str(convert_command[outfile_index])))

    quantize_command = plan.get("quantize_command") or []
    if len(quantize_command) >= 3:
        paths.append(Path(str(quantize_command[2])))

    return paths


def _path_key(path: Path) -> str:
    return str(path.resolve())


def build_export_tab(catalog: dict[str, ModelInfo]) -> None:
    model_id = gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Model")
    quant = gr.Dropdown(QUANTIZATION_CHOICES, value="Q4_K_M", label="Quantization")
    output_dir = gr.Textbox(label="Output directory", value="exports")
    run = gr.Button("Prepare export plan", variant="primary")
    refresh = gr.Button("Refresh exported files")
    output = gr.JSON(label="Export plan")
    files = gr.Dataframe(headers=["path", "bytes"], label="Exported files", interactive=False)
    downloads = gr.File(
        label="Download exported files",
        file_count="multiple",
        interactive=False,
    )

    def plan_export(
        selected: str,
        selected_quant: str,
        directory: str,
    ) -> tuple[dict, list[list[str]], list[str]]:
        model = catalog[selected]
        plan = build_export_plan(model, selected_quant, directory)
        exported_rows = list_exported_files(directory)
        plan_data = plan.as_dict()
        return plan_data, exported_rows, export_download_paths(plan_data, exported_rows)

    def refresh_exports(directory: str) -> tuple[list[list[str]], list[str]]:
        exported_rows = list_exported_files(directory)
        return exported_rows, export_download_paths(None, exported_rows)

    run.click(
        plan_export,
        [model_id, quant, output_dir],
        [output, files, downloads],
        show_progress=CLICK_PROGRESS,
    )
    refresh.click(
        refresh_exports,
        output_dir,
        [files, downloads],
        show_progress=CLICK_PROGRESS,
    )
