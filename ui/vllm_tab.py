from __future__ import annotations

from typing import Any

import gradio as gr
import requests

from models.model_catalog import ModelInfo
from models.vllm_runner import (
    VLLMConfig,
    VLLMService,
    build_vllm_run_plan,
    fetch_vllm_metrics,
    log_vllm_benchmark,
)
from ui.progress import CLICK_PROGRESS
from ui.server_controls import create_serving_controls


def build_vllm_tab(catalog: dict[str, ModelInfo]) -> None:
    gr.Markdown("vLLM serving plans and local metrics checks.")
    controls = create_vllm_controls(catalog)
    command = gr.Textbox(label="vLLM command / status", interactive=False)
    output = gr.JSON(label="vLLM plan, status, or metrics")
    prepare_inputs = [
        controls[key]
        for key in (
            "selected",
            "base_url",
            "host",
            "port",
            "parallel",
            "dtype",
            "max_model_len",
        )
    ]

    controls["prepare"].click(
        lambda *args: prepare_vllm(catalog, *args),
        prepare_inputs,
        [command, output],
        show_progress=CLICK_PROGRESS,
    )
    controls["check"].click(check_vllm, controls["base_url"], output, show_progress=CLICK_PROGRESS)
    controls["metrics"].click(
        get_metrics,
        controls["base_url"],
        output,
        show_progress=CLICK_PROGRESS,
    )
    controls["log_metrics"].click(
        log_current_metrics,
        [controls["selected"], controls["base_url"]],
        command,
        show_progress=CLICK_PROGRESS,
    )


def create_vllm_controls(catalog: dict[str, ModelInfo]) -> dict[str, Any]:
    controls = create_serving_controls(catalog, "vLLM", "http://127.0.0.1:8000", 8000)
    controls["dtype"] = gr.Textbox(label="dtype", value="auto")
    controls["max_model_len"] = gr.Number(label="Max model length", value=4096, precision=0)
    with gr.Row():
        controls["prepare"] = gr.Button("Prepare vLLM command", variant="primary")
        controls["check"] = gr.Button("Check vLLM")
        controls["metrics"] = gr.Button("Fetch metrics")
        controls["log_metrics"] = gr.Button("Log benchmark")
    return controls


def config_from_inputs(
    url: str,
    server_host: str,
    server_port: int | float,
    parallel_size: int | float,
    dtype_value: str,
    model_len: int | float,
) -> VLLMConfig:
    return VLLMConfig(
        base_url=url.strip() or "http://127.0.0.1:8000",
        host=server_host.strip() or "127.0.0.1",
        port=int(server_port),
        tensor_parallel_size=int(parallel_size),
        dtype=dtype_value.strip() or "auto",
        max_model_len=int(model_len),
    )


def prepare_vllm(
    catalog: dict[str, ModelInfo],
    model_id: str,
    url: str,
    server_host: str,
    server_port: int | float,
    parallel_size: int | float,
    dtype_value: str,
    model_len: int | float,
) -> tuple[str, dict]:
    config = config_from_inputs(
        url,
        server_host,
        server_port,
        parallel_size,
        dtype_value,
        model_len,
    )
    plan = build_vllm_run_plan(catalog[model_id], config)
    return " ".join(plan.start_command), plan.to_dict()


def check_vllm(url: str) -> dict:
    status = VLLMService.status(url.strip() or "http://127.0.0.1:8000")
    return {"backend": status.name, "available": status.available, "detail": status.detail}


def get_metrics(url: str) -> dict:
    try:
        return fetch_vllm_metrics(url.strip() or "http://127.0.0.1:8000")
    except (OSError, requests.RequestException) as exc:
        return {"error": str(exc)}


def log_current_metrics(model_id: str, url: str) -> str:
    parsed = get_metrics(url)
    if "error" in parsed:
        return str(parsed["error"])
    return f"Logged vLLM benchmark to {log_vllm_benchmark(parsed, model_id)}"
