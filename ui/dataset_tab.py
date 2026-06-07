from __future__ import annotations

import gradio as gr

from core.app_state import APP_STATE
from core.events import Event, EventType
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
    load = gr.Button("Preview dataset", variant="primary")
    load_hf = gr.Button("Preview Hugging Face dataset")
    inspect = gr.Button("Calculate local stats")

    def preview_dataset(ds_id: str, split_name: str) -> list[list[str]]:
        if not ds_id:
            return [["status", "Enter a local CSV or JSONL path."]]
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
            return result.as_table()
        except (FileNotFoundError, ValueError, OSError) as exc:
            return [["error", str(exc)]]

    load.click(preview_dataset, [dataset_id, split], preview)

    def preview_hf_dataset(ds_id: str, split_name: str) -> list[list[str]]:
        if not ds_id:
            return [["status", "Enter a Hugging Face dataset ID."]]
        return preview_huggingface_dataset(ds_id, split_name).as_table()

    def calculate_stats(ds_id: str) -> dict:
        if not ds_id:
            return {"status": "Enter a local CSV or JSONL path."}
        try:
            return dataset_statistics(ds_id).as_dict()
        except (FileNotFoundError, ValueError, OSError) as exc:
            return {"error": str(exc)}

    load_hf.click(preview_hf_dataset, [dataset_id, split], preview)
    inspect.click(calculate_stats, dataset_id, stats)
