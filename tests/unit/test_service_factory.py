from __future__ import annotations

import unittest

from models.llama_cpp_python_service import LlamaCppPythonService
from models.llama_cpp_service import LlamaCppService
from models.minicpm_vision import MiniCPMVisionService
from models.model_catalog import load_model_catalog
from models.ollama_service import OllamaService
from models.openai_compatible_service import OpenAICompatibleService
from models.placeholder_service import PlaceholderModelService
from models.service_factory import (
    TEXT_SERVICE_REGISTRY,
    VISION_SERVICE_REGISTRY,
    backend_statuses,
    create_text_service,
    create_vision_service,
)
from models.sglang_runner import SGLangService
from models.transformers_text import TransformersTextService


class ServiceFactoryTest(unittest.TestCase):
    def test_creates_placeholder_service_by_default(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "placeholder")

        self.assertIsInstance(service, PlaceholderModelService)

    def test_creates_ollama_service_when_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "ollama")

        self.assertIsInstance(service, OllamaService)

    def test_creates_llama_cpp_service_when_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "llama.cpp")

        self.assertIsInstance(service, LlamaCppService)

    def test_creates_llama_cpp_python_service_when_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "llama-cpp-python")

        self.assertIsInstance(service, LlamaCppPythonService)

    def test_backend_statuses_include_placeholder_and_ollama(self) -> None:
        statuses = backend_statuses()

        self.assertIn("placeholder", [status.name for status in statuses])
        self.assertIn("llama.cpp", [status.name for status in statuses])
        self.assertIn("llama-cpp-python", [status.name for status in statuses])
        self.assertIn("ollama", [status.name for status in statuses])
        self.assertIn("transformers", [status.name for status in statuses])
        self.assertIn("openai-compatible", [status.name for status in statuses])
        self.assertIn("sglang", [status.name for status in statuses])
        self.assertIn("transformers-vision", [status.name for status in statuses])

    def test_model_service_registries_include_all_backends(self) -> None:
        expected = [
            "placeholder",
            "llama.cpp",
            "llama-cpp-python",
            "ollama",
            "transformers",
            "openai-compatible",
            "sglang",
        ]

        self.assertEqual(TEXT_SERVICE_REGISTRY.list(), expected)
        self.assertEqual(
            VISION_SERVICE_REGISTRY.list(),
            ["placeholder", "llama.cpp", "llama-cpp-python", "ollama", "transformers"],
        )

    def test_creates_transformers_service_when_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "transformers")

        self.assertIsInstance(service, TransformersTextService)

    def test_creates_openai_compatible_service_when_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "openai-compatible")

        self.assertIsInstance(service, OpenAICompatibleService)

    def test_creates_sglang_service_when_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_text_service(catalog["minicpm5_1b"], "sglang")

        self.assertIsInstance(service, SGLangService)

    def test_creates_minicpm_vision_service_when_transformers_selected(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = create_vision_service(catalog["minicpm_v46"], "transformers")

        self.assertIsInstance(service, MiniCPMVisionService)


if __name__ == "__main__":
    unittest.main()
