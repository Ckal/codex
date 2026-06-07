from __future__ import annotations

import unittest

from models.llama_cpp_python_service import LlamaCppPythonService
from models.llama_cpp_service import LlamaCppService
from models.model_catalog import load_model_catalog
from models.ollama_service import OllamaService
from models.placeholder_service import PlaceholderModelService
from models.service_factory import (
    TEXT_SERVICE_REGISTRY,
    VISION_SERVICE_REGISTRY,
    backend_statuses,
    create_text_service,
)


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

    def test_model_service_registries_include_all_backends(self) -> None:
        expected = ["placeholder", "llama.cpp", "llama-cpp-python", "ollama"]

        self.assertEqual(TEXT_SERVICE_REGISTRY.list(), expected)
        self.assertEqual(VISION_SERVICE_REGISTRY.list(), expected)


if __name__ == "__main__":
    unittest.main()
