from __future__ import annotations

from typing import Any

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
from models.openai_compatible_service import OpenAICompatibleService
from models.service_factory import backend_statuses
from models.sglang_runner import SGLangConfig, SGLangService, build_sglang_run_plan
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
            "capabilities",
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
                ", ".join(
                    sorted(
                        {
                            capability
                            for capabilities in model.backend_capabilities.values()
                            for capability in capabilities
                        }
                    )
                ),
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
    build_openai_compatible_setup_panel()
    build_sglang_setup_panel(catalog)
    build_ollama_setup_panel(catalog)


def build_sglang_setup_panel(catalog: dict[str, ModelInfo]) -> None:
    gr.Markdown("### SGLang local setup")
    controls = create_sglang_controls(catalog)
    command = gr.Textbox(label="SGLang start command", interactive=False)
    summary = gr.JSON(label="SGLang plan/status")

    def config_from_inputs(
        url: str,
        server_host: str,
        server_port: int | float,
        tp: int | float,
        parser: str,
    ) -> SGLangConfig:
        return SGLangConfig(
            base_url=url.strip() or "http://127.0.0.1:30000",
            host=server_host.strip() or "127.0.0.1",
            port=int(server_port),
            tp_size=int(tp),
            tool_parser=parser.strip() or "minicpm",
        )

    def prepare_sglang(
        model_id: str,
        url: str,
        server_host: str,
        server_port: int | float,
        tp: int | float,
        parser: str,
    ) -> tuple[str, dict]:
        config = config_from_inputs(url, server_host, server_port, tp, parser)
        plan = build_sglang_run_plan(catalog[model_id], config)
        return " ".join(plan.start_command), plan.to_dict()

    def check_sglang(url: str) -> dict:
        status = SGLangService.status(url.strip() or "http://127.0.0.1:30000")
        return {"backend": status.name, "available": status.available, "detail": status.detail}

    def stop_sglang(model_id: str, url: str) -> str:
        service = SGLangService(
            catalog[model_id],
            SGLangConfig(base_url=url.strip() or "http://127.0.0.1:30000"),
        )
        return service.stop_server()

    controls["prepare"].click(
        prepare_sglang,
        [
            controls["selected"],
            controls["base_url"],
            controls["host"],
            controls["port"],
            controls["tp_size"],
            controls["tool_parser"],
        ],
        [command, summary],
        show_progress=CLICK_PROGRESS,
    )
    controls["check"].click(
        check_sglang,
        controls["base_url"],
        summary,
        show_progress=CLICK_PROGRESS,
    )
    controls["stop"].click(
        stop_sglang,
        [controls["selected"], controls["base_url"]],
        command,
        show_progress=CLICK_PROGRESS,
    )


def create_sglang_controls(catalog: dict[str, ModelInfo]) -> dict[str, Any]:
    controls: dict[str, Any] = {
        "selected": gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Model config"),
        "base_url": gr.Textbox(label="SGLang base URL", value="http://127.0.0.1:30000"),
    }
    with gr.Row():
        controls["host"] = gr.Textbox(label="Host", value="127.0.0.1")
        controls["port"] = gr.Number(label="Port", value=30000, precision=0)
        controls["tp_size"] = gr.Number(label="Tensor parallel size", value=1, precision=0)
    controls["tool_parser"] = gr.Textbox(label="Tool parser", value="minicpm")
    with gr.Row():
        controls["prepare"] = gr.Button("Prepare SGLang command", variant="primary")
        controls["check"] = gr.Button("Check SGLang")
        controls["stop"] = gr.Button("Request SGLang stop")
    return controls


def build_openai_compatible_setup_panel() -> None:
    gr.Markdown("### LM Studio / OpenAI-compatible local setup")
    local_config = load_local_backend_config()
    base_url = gr.Textbox(
        label="Server URL",
        value=local_config.openai_compatible_base_url,
        placeholder="http://127.0.0.1:1234",
    )
    model_name = gr.Textbox(
        label="Served model name",
        value=local_config.openai_compatible_model_name,
        placeholder="Optional; leave blank to use the selected HF model ID",
    )
    prepare = gr.Button("Save OpenAI-compatible config", variant="primary")
    check = gr.Button("Check server")
    summary = gr.JSON(local_backend_summary(local_config), label="OpenAI-compatible config")
    status = gr.JSON(label="OpenAI-compatible status")

    def save_openai_config(url: str, served_model_name: str) -> dict:
        current = load_local_backend_config()
        config = LocalBackendConfig(
            llama_cpp_server_url=current.llama_cpp_server_url,
            openai_compatible_base_url=url.strip() or LocalBackendConfig.openai_compatible_base_url,
            openai_compatible_model_name=served_model_name.strip(),
            gguf_path=current.gguf_path,
            mmproj_path=current.mmproj_path,
            n_ctx=current.n_ctx,
            n_gpu_layers=current.n_gpu_layers,
        )
        save_local_backend_config(config)
        return local_backend_summary(config)

    def check_openai_server(url: str) -> dict:
        server_status = OpenAICompatibleService.status(
            url.strip() or LocalBackendConfig.openai_compatible_base_url
        )
        return {
            "backend": server_status.name,
            "available": server_status.available,
            "detail": server_status.detail,
        }

    prepare.click(
        save_openai_config,
        [base_url, model_name],
        summary,
        show_progress=CLICK_PROGRESS,
    )
    check.click(check_openai_server, base_url, status, show_progress=CLICK_PROGRESS)


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
        current = load_local_backend_config()
        config = LocalBackendConfig(
            llama_cpp_server_url=url or "http://127.0.0.1:8080",
            openai_compatible_base_url=current.openai_compatible_base_url,
            openai_compatible_model_name=current.openai_compatible_model_name,
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
