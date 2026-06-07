from __future__ import annotations

import gradio as gr

from agent.runner import (
    export_agent_traces,
    export_agent_traces_hf_dataset,
    run_agent_loop,
    save_agent_trace,
)


def build_agent_tab() -> None:
    gr.Markdown("Agent mode drafts a local research -> plan -> implement -> verify trace.")
    task = gr.Textbox(label="Agent task", lines=5, placeholder="Example: improve field-note export")
    plan = gr.Button("Draft agent trace", variant="primary")
    export = gr.Button("Export traces")
    export_dataset = gr.Button("Export traces dataset")
    output = gr.Textbox(label="Plan", lines=10)
    trace = gr.JSON(label="Trace")

    def draft_plan(task_text: str) -> tuple[str, dict]:
        session = run_agent_loop(task_text)
        save_agent_trace(session)
        return session.as_markdown(), session.as_dict()

    def export_trace_file() -> dict:
        path = export_agent_traces()
        return {"exported_to": str(path)}

    def export_trace_dataset() -> dict:
        path = export_agent_traces_hf_dataset()
        return {"exported_to": str(path)}

    plan.click(draft_plan, task, [output, trace])
    export.click(export_trace_file, outputs=trace)
    export_dataset.click(export_trace_dataset, outputs=trace)
