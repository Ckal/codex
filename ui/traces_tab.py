from __future__ import annotations

import gradio as gr

from core.app_state import APP_STATE
from tracking.trackio_client import export_traces, read_trace_rows, tracking_status_dict


def recent_events() -> list[dict]:
    return APP_STATE.recent_events()


def build_traces_tab() -> None:
    gr.Markdown(
        "Recent local app events with JSONL fallback tracking and optional Trackio integration."
    )
    run_name = gr.Textbox(label="Run name", placeholder="demo-run")
    event = gr.Textbox(label="Event", placeholder="inference, dataset_loaded, training_step")
    preview = gr.Button("Preview trace event", variant="primary")
    refresh = gr.Button("Refresh recent events")
    export = gr.Button("Export local traces")
    output = gr.JSON(label="Trace preview")
    trace_rows = gr.JSON(label="Local trace rows")
    status = gr.JSON(tracking_status_dict(), label="Tracking status")

    def preview_trace(run: str, event_name: str) -> dict:
        return {
            "status": "preview",
            "run_name": run or "demo-run",
            "event": event_name or "manual_preview",
            "tracking": tracking_status_dict(),
        }

    preview.click(preview_trace, [run_name, event], output)
    refresh.click(recent_events, outputs=output)
    refresh.click(read_trace_rows, outputs=trace_rows)
    refresh.click(tracking_status_dict, outputs=status)

    def export_local_traces() -> dict:
        path = export_traces()
        return {"exported_to": str(path)}

    export.click(export_local_traces, outputs=output)
