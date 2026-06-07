from __future__ import annotations

import gradio as gr

from training.evaluation import (
    compare_base_vs_tuned,
    default_prompt_cases,
    evaluate_responses,
    log_eval_report,
)


def build_train_tab() -> None:
    gr.Markdown("LoRA training is planned for the next implementation slice.")
    with gr.Row():
        rank = gr.Slider(4, 64, value=16, step=4, label="LoRA rank")
        epochs = gr.Slider(1, 5, value=1, step=1, label="Epochs")
    dataset = gr.Textbox(label="Training dataset", placeholder="data/field_notes.jsonl")
    start = gr.Button("Prepare training plan", variant="primary")
    output = gr.Textbox(label="Plan", lines=8)

    def plan_training(rank_value: int, epoch_value: int, dataset_path: str) -> str:
        return (
            "Training is not started automatically in the MVP.\n\n"
            f"Dataset: {dataset_path or '(none selected)'}\n"
            f"LoRA rank: {rank_value}\n"
            f"Epochs: {epoch_value}\n\n"
            "Next step: wire this to PEFT/TRL or SWIFT after the local model path is chosen."
        )

    start.click(plan_training, [rank, epochs, dataset], output)

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
    run_eval = gr.Button("Run local evaluation")
    eval_summary = gr.JSON(label="Evaluation summary")
    eval_table = gr.Dataframe(
        headers=["prompt", "expected", "actual", "exact_match", "notes"],
        label="Tuned qualitative table",
        interactive=False,
    )

    def evaluate_local(base_text: str, tuned_text: str) -> tuple[dict, list[list[str]]]:
        cases = default_prompt_cases()
        base_report = evaluate_responses(cases, base_text.splitlines())
        tuned_report = evaluate_responses(cases, tuned_text.splitlines())
        comparison = compare_base_vs_tuned(base_report, tuned_report)
        log_eval_report(tuned_report)
        return comparison.as_dict(), tuned_report.as_table()

    run_eval.click(evaluate_local, [base_responses, tuned_responses], [eval_summary, eval_table])
