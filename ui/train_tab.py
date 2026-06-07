from __future__ import annotations

import gradio as gr

from training.evaluation import (
    attach_perplexity,
    compare_base_vs_tuned,
    default_prompt_cases,
    evaluate_responses,
    log_eval_report,
)
from training.planner import build_training_plan
from ui.progress import CLICK_PROGRESS


def build_train_tab() -> None:
    gr.Markdown("LoRA training is planned for the next implementation slice.")
    with gr.Row():
        rank = gr.Slider(4, 64, value=16, step=4, label="LoRA rank")
        epochs = gr.Slider(1, 5, value=1, step=1, label="Epochs")
    dataset = gr.Textbox(label="Training dataset", placeholder="data/field_notes.jsonl")
    start = gr.Button("Prepare training plan", variant="primary")
    output = gr.Textbox(label="Plan", lines=8)

    def plan_training(rank_value: int, epoch_value: int, dataset_path: str) -> str:
        plan = build_training_plan(
            dataset_path=dataset_path,
            rank=rank_value,
            epochs=epoch_value,
        )
        return plan.as_text()

    start.click(
        plan_training,
        [rank, epochs, dataset],
        output,
        show_progress=CLICK_PROGRESS,
    )

    gr.Markdown("### Local evaluation")
    base_responses = gr.Textbox(
        label="Base responses",
        lines=4,
        placeholder="One response per default prompt case",
    )
    tuned_responses = gr.Textbox(
        label="Tuned responses",
        lines=4,
        placeholder="One response per default prompt case",
    )
    tuned_losses = gr.Textbox(
        label="Optional tuned losses",
        lines=2,
        placeholder="Optional negative log likelihood values, comma or newline separated",
    )
    run_eval = gr.Button("Run local evaluation")
    eval_summary = gr.JSON(label="Evaluation summary")
    eval_table = gr.Dataframe(
        headers=["prompt", "expected", "actual", "exact_match", "notes"],
        label="Tuned qualitative table",
        interactive=False,
    )

    def evaluate_local(
        base_text: str,
        tuned_text: str,
        loss_text: str,
    ) -> tuple[dict, list[list[str]]]:
        cases = default_prompt_cases()
        base_report = evaluate_responses(cases, base_text.splitlines())
        tuned_report = evaluate_responses(cases, tuned_text.splitlines())
        tuned_report = attach_perplexity(tuned_report, parse_losses(loss_text))
        comparison = compare_base_vs_tuned(base_report, tuned_report)
        log_eval_report(tuned_report)
        summary = comparison.as_dict()
        summary["tuned_perplexity"] = tuned_report.perplexity
        return summary, tuned_report.as_table()

    def parse_losses(loss_text: str) -> list[float]:
        cleaned = loss_text.replace(",", "\n")
        return [float(value.strip()) for value in cleaned.splitlines() if value.strip()]

    run_eval.click(
        evaluate_local,
        [base_responses, tuned_responses, tuned_losses],
        [eval_summary, eval_table],
        show_progress=CLICK_PROGRESS,
    )
