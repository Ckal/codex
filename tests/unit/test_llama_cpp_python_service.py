from __future__ import annotations

import unittest

from models.llama_cpp_python_service import LlamaCppPythonService
from models.model_catalog import load_model_catalog
from models.service_factory import create_text_service


class LlamaCppPythonServiceTest(unittest.TestCase):
    def test_factory_creates_llama_cpp_python_service(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        service = create_text_service(catalog["minicpm5_1b"], "llama-cpp-python")

        self.assertIsInstance(service, LlamaCppPythonService)

    def test_status_reports_missing_package_or_missing_model(self) -> None:
        status = LlamaCppPythonService.status()

        self.assertFalse(status.available)
        self.assertTrue(
            "not installed" in status.detail or "no GGUF model path" in status.detail
        )

    def test_chat_returns_clear_unavailable_message(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        response = LlamaCppPythonService(catalog["minicpm5_1b"]).chat("", "Hello")

        self.assertIn("[llama-cpp-python unavailable]", response)

    def test_extracts_chat_completion_response(self) -> None:
        response = LlamaCppPythonService._extract_response(
            {"choices": [{"message": {"content": "hello"}}]}
        )

        self.assertEqual(response, "hello")


if __name__ == "__main__":
    unittest.main()
