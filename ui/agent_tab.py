from __future__ import annotations

import gradio as gr


def build_agent_tab() -> None:
    gr.Markdown("Agent mode is planned as a research -> plan -> implement helper.")
    task = gr.Textbox(label="Agent task", lines=5, placeholder="Example: improve field-note export")
    plan = gr.Button("Draft agent plan", variant="primary")
    output = gr.Textbox(label="Plan", lines=10)

    def draft_plan(task_text: str) -> str:
        return (
            "Agent mode is not autonomous yet.\n\n"
            f"Task: {task_text or '(none)'}\n\n"
            "Planned loop:\n"
            "1. Research relevant docs/files.\n"
            "2. Propose a small implementation plan.\n"
            "3. Execute with explicit verification.\n"
            "4. Save trace for sharing."
        )

    plan.click(draft_plan, task, output)
