from __future__ import annotations

from collections.abc import Callable

from core.registry import Registry
from models.base import BackendStatus, TextModelService, VisionModelService
from models.llama_cpp_python_service import LlamaCppPythonConfig, LlamaCppPythonService
from models.llama_cpp_service import LlamaCppConfig, LlamaCppService
from models.local_backend_config import load_local_backend_config
from models.minicpm_vision import MiniCPMVisionService
from models.model_catalog import ModelInfo
from models.ollama_service import OllamaService
from models.openai_compatible_service import OpenAICompatibleConfig, OpenAICompatibleService
from models.placeholder_service import PlaceholderModelService
from models.sglang_runner import SGLangConfig, SGLangService
from models.transformers_text import TransformersTextService

TextFactory = Callable[[ModelInfo], TextModelService]
VisionFactory = Callable[[ModelInfo], VisionModelService]

TEXT_SERVICE_REGISTRY: Registry[TextFactory] = Registry()
TEXT_SERVICE_REGISTRY.register("placeholder", PlaceholderModelService)
TEXT_SERVICE_REGISTRY.register("llama.cpp", LlamaCppService)
TEXT_SERVICE_REGISTRY.register("llama-cpp-python", LlamaCppPythonService)
TEXT_SERVICE_REGISTRY.register("ollama", OllamaService)
TEXT_SERVICE_REGISTRY.register("transformers", TransformersTextService)
TEXT_SERVICE_REGISTRY.register("openai-compatible", OpenAICompatibleService)
TEXT_SERVICE_REGISTRY.register("sglang", SGLangService)

VISION_SERVICE_REGISTRY: Registry[VisionFactory] = Registry()
VISION_SERVICE_REGISTRY.register("placeholder", PlaceholderModelService)
VISION_SERVICE_REGISTRY.register("llama.cpp", LlamaCppService)
VISION_SERVICE_REGISTRY.register("llama-cpp-python", LlamaCppPythonService)
VISION_SERVICE_REGISTRY.register("ollama", OllamaService)
VISION_SERVICE_REGISTRY.register("transformers", MiniCPMVisionService)

BACKENDS = TEXT_SERVICE_REGISTRY.list()


def create_text_service(model: ModelInfo, backend: str) -> TextModelService:
    if backend == "llama.cpp":
        config = load_local_backend_config()
        return LlamaCppService(
            model,
            LlamaCppConfig(
                server_url=config.llama_cpp_server_url,
                model_path=config.gguf_path,
                mmproj_path=config.mmproj_path,
            ),
        )
    if backend == "llama-cpp-python":
        config = load_local_backend_config()
        return LlamaCppPythonService(
            model,
            LlamaCppPythonConfig(
                model_path=config.gguf_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
            ),
        )
    if backend == "openai-compatible":
        config = load_local_backend_config()
        return OpenAICompatibleService(
            model,
            OpenAICompatibleConfig(
                base_url=config.openai_compatible_base_url,
                model_name=config.openai_compatible_model_name,
            ),
        )
    if backend == "sglang":
        return SGLangService(model, SGLangConfig())
    return TEXT_SERVICE_REGISTRY.get(backend)(model)


def create_vision_service(model: ModelInfo, backend: str) -> VisionModelService:
    if backend == "llama.cpp":
        config = load_local_backend_config()
        return LlamaCppService(
            model,
            LlamaCppConfig(
                server_url=config.llama_cpp_server_url,
                model_path=config.gguf_path,
                mmproj_path=config.mmproj_path,
            ),
        )
    if backend == "llama-cpp-python":
        config = load_local_backend_config()
        return LlamaCppPythonService(
            model,
            LlamaCppPythonConfig(
                model_path=config.gguf_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
            ),
        )
    return VISION_SERVICE_REGISTRY.get(backend)(model)


def backend_statuses() -> list[BackendStatus]:
    return [
        BackendStatus("placeholder", True, "Deterministic placeholder backend is available."),
        LlamaCppService.status(),
        LlamaCppPythonService.status(),
        OllamaService.status(),
        TransformersTextService.status(),
        OpenAICompatibleService.status(load_local_backend_config().openai_compatible_base_url),
        SGLangService.status(),
        MiniCPMVisionService.status(),
    ]
