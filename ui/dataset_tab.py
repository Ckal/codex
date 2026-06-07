from __future__ import annotations

import gradio as gr

from core.app_state import APP_STATE
from core.events import Event, EventType
from core.tab_feedback import emit_tab_error, status_ok
from datasets.loader import (
    dataset_statistics,
    preview_huggingface_dataset,
    preview_local_dataset,
)


def build_dataset_tab() -> None:
    dataset_id = gr.Textbox(
        label="Dataset ID or local path",
        placeholder="Example: tatsu-lab/alpaca",
    )
    split = gr.Textbox(label="Split", value="train")
    preview = gr.Dataframe(headers=["field", "value"], label="Preview")
    stats = gr.JSON(label="Dataset statistics")
    status = gr.Markdown(status_ok("Ready."))
    load = gr.Button("Preview dataset", variant="primary")
    load_hf = gr.Button("Preview Hugging Face dataset")
    inspect = gr.Button("Calculate local stats")

    def preview_dataset(ds_id: str, split_name: str) -> tuple[list[list[str]], str]:
        if not ds_id:
            message = "Enter a local CSV or JSONL path."
            return [["status", message]], emit_tab_error("Dataset", message)
        try:
            result = preview_local_dataset(ds_id)
            APP_STATE.emit(
                Event(
                    EventType.DATASET_LOADED,
                    {
                        "source": result.source,
                        "rows": result.rows,
                        "columns": result.columns,
                        "split": split_name,
                    },
                )
            )
            return result.as_table(), status_ok("Local dataset preview loaded.")
        except (FileNotFoundError, ValueError, OSError) as exc:
            return [["error", str(exc)]], emit_tab_error(
                "Dataset",
                str(exc),
                {"source": ds_id, "split": split_name},
            )

    load.click(preview_dataset, [dataset_id, split], [preview, status])

    def preview_hf_dataset(ds_id: str, split_name: str) -> tuple[list[list[str]], str]:
        if not ds_id:
            message = "Enter a Hugging Face dataset ID."
            return [["status", message]], emit_tab_error("Dataset", message)
        try:
            result = preview_huggingface_dataset(ds_id, split_name)
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            return [["error", str(exc)]], emit_tab_error(
                "Dataset",
                str(exc),
                {"source": ds_id, "split": split_name},
            )
        return result.as_table(), status_ok("Hugging Face dataset preview loaded.")

    def calculate_stats(ds_id: str) -> tuple[dict, str]:
        if not ds_id:
            message = "Enter a local CSV or JSONL path."
            return {"status": message}, emit_tab_error("Dataset", message)
        try:
            return dataset_statistics(ds_id).as_dict(), status_ok("Local dataset stats calculated.")
        except (FileNotFoundError, ValueError, OSError) as exc:
            return {"error": str(exc)}, emit_tab_error(
                "Dataset",
                str(exc),
                {"source": ds_id},
            )

    load_hf.click(preview_hf_dataset, [dataset_id, split], [preview, status])
    inspect.click(calculate_stats, dataset_id, [stats, status])
