from __future__ import annotations

import gradio as gr

from models.local_backend_config import (
    LocalBackendConfig,
    build_llama_server_command,
    load_local_backend_config,
    local_backend_summary,
    save_local_backend_config,
)
from models.model_catalog import ModelInfo, model_summary, validate_catalog
from models.ollama_service import OllamaService
from models.service_factory import backend_statuses
from ui.progress import CLICK_PROGRESS


def build_status_tab(catalog: dict[str, ModelInfo]) -> None:
    gr.Markdown("Model and backend status. Real backend checks will be added after local setup.")
    gr.Dataframe(
        headers=[
            "config_id",
            "hf_id",
            "type",
            "backend",
            "parameters_b",
            "context_length",
            "thinking",
        ],
        value=[
            [
                model.config_id,
                model.hf_id,
                model.type,
                model.backend,
                model.parameters_b,
                model.context_length,
                model.thinking_mode,
            ]
            for model in catalog.values()
        ],
        label="Configured models",
        interactive=False,
    )
    gr.JSON(validate_catalog(catalog), label="Validation warnings")
    gr.Dataframe(
        headers=["backend", "available", "detail"],
        value=[
            [status.name, status.available, status.detail]
            for status in backend_statuses()
        ],
        label="Backend status",
        interactive=False,
    )
    selected = gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Inspect model")
    details = gr.JSON(model_summary(catalog[next(iter(catalog))]), label="Details")

    def inspect(model_id: str) -> dict:
        return model_summary(catalog[model_id])

    selected.change(inspect, selected, details)

    build_llama_cpp_setup_panel()
    build_ollama_setup_panel(catalog)


def build_ollama_setup_panel(catalog: dict[str, ModelInfo]) -> None:
    gr.Markdown("### Ollama local setup")
    selected = gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Model config")
    ollama_name = gr.Textbox(
        label="Ollama model name",
        placeholder="Example: minicpm-v or a local Ollama model tag",
    )
    refresh = gr.Button("List local Ollama models")
    prepare = gr.Button("Prepare pull command", variant="primary")
    local_models = gr.JSON(label="Local Ollama models")
    pull_command = gr.Textbox(label="Ollama pull command", interactive=False)

    def list_models() -> dict:
        models = OllamaService.list_local_models()
        return {
            "models": models,
            "note": "Empty means Ollama is not running, not installed, or has no local models.",
        }

    def prepare_pull(model_id: str, model_name: str) -> str:
        name = model_name.strip() or catalog[model_id].hf_id
        return " ".join(OllamaService.pull_command(name))

    refresh.click(list_models, outputs=local_models, show_progress=CLICK_PROGRESS)
    prepare.click(
        prepare_pull,
        [selected, ollama_name],
        pull_command,
        show_progress=CLICK_PROGRESS,
    )


def build_llama_cpp_setup_panel() -> None:
    gr.Markdown("### llama.cpp local setup")
    local_config = load_local_backend_config()
    server_url = gr.Textbox(
        label="llama-server URL",
        value=local_config.llama_cpp_server_url,
    )
    gguf_path = gr.Textbox(
        label="GGUF model path",
        value=local_config.gguf_path,
        placeholder="C:\\models\\MiniCPM5-1B-Q4_K_M.gguf",
    )
    gguf_file = gr.File(label="Pick GGUF model", file_types=[".gguf"], type="filepath")
    mmproj_path = gr.Textbox(
        label="mmproj path",
        value=local_config.mmproj_path,
        placeholder="Optional vision projector GGUF path",
    )
    mmproj_file = gr.File(label="Pick mmproj", file_types=[".gguf"], type="filepath")
    with gr.Row():
        n_ctx = gr.Number(label="Context length", value=local_config.n_ctx, precision=0)
        n_gpu_layers = gr.Number(
            label="GPU layers",
            value=local_config.n_gpu_layers,
            precision=0,
        )
    prepare = gr.Button("Prepare local model config", variant="primary")
    command = gr.Textbox(label="llama-server command", interactive=False)
    local_summary = gr.JSON(local_backend_summary(local_config), label="Local backend config")

    def prepare_local_config(
        url: str,
        model_path: str,
        model_file: str | None,
        projector_path: str,
        projector_file: str | None,
        context_length: int | float,
        gpu_layers: int | float,
    ) -> tuple[str, dict]:
        config = LocalBackendConfig(
            llama_cpp_server_url=url or "http://127.0.0.1:8080",
            gguf_path=model_file or model_path,
            mmproj_path=projector_file or projector_path,
            n_ctx=int(context_length),
            n_gpu_layers=int(gpu_layers),
        )
        save_local_backend_config(config)
        built_command = build_llama_server_command(config)
        return " ".join(built_command), local_backend_summary(config)

    prepare.click(
        prepare_local_config,
        [server_url, gguf_path, gguf_file, mmproj_path, mmproj_file, n_ctx, n_gpu_layers],
        [command, local_summary],
        show_progress=CLICK_PROGRESS,
    )
