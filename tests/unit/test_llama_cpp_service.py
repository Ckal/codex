from __future__ import annotations

import unittest

import requests

from models.llama_cpp_service import LlamaCppConfig, LlamaCppService, local_file_status
from models.model_catalog import load_model_catalog
from models.service_factory import create_text_service


class LlamaCppServiceTest(unittest.TestCase):
    def test_factory_creates_llama_cpp_service(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        service = create_text_service(catalog["minicpm5_1b"], "llama.cpp")

        self.assertIsInstance(service, LlamaCppService)

    def test_status_reports_missing_executable(self) -> None:
        status = LlamaCppService.status(which_func=lambda _name: None)

        self.assertFalse(status.available)
        self.assertIn("not found", status.detail)

    def test_launch_command_includes_model_and_mmproj(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = LlamaCppService(
            catalog["minicpm_v46"],
            LlamaCppConfig(model_path="model.gguf", mmproj_path="mmproj.gguf"),
        )

        self.assertEqual(
            service.launch_command(),
            ["llama-server", "-m", "model.gguf", "--mmproj", "mmproj.gguf"],
        )

    def test_launch_command_is_empty_without_model_path(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        service = LlamaCppService(catalog["minicpm5_1b"])

        self.assertEqual(service.launch_command(), [])

    def test_chat_returns_clear_unavailable_message(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        response = LlamaCppService(catalog["minicpm5_1b"]).chat("", "Hello")

        self.assertIn("[llama.cpp unavailable]", response)

    def test_extracts_openai_compatible_response(self) -> None:
        response = LlamaCppService._extract_response(
            {"choices": [{"message": {"content": "hello"}}]}
        )

        self.assertEqual(response, "hello")

    def test_status_reports_unreachable_server(self) -> None:
        def raise_request_error(*_args, **_kwargs):
            raise requests.ConnectionError("offline")

        status = LlamaCppService.status(
            which_func=lambda _name: "llama-server",
            get_func=raise_request_error,
        )

        self.assertFalse(status.available)
        self.assertIn("not reachable", status.detail)

    def test_local_file_status(self) -> None:
        self.assertEqual(local_file_status(""), "not configured")
        self.assertEqual(local_file_status("missing-file.gguf"), "missing")


if __name__ == "__main__":
    unittest.main()
