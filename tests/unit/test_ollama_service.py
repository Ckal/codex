from __future__ import annotations

import unittest

import models.ollama_service as ollama_module
from models.model_catalog import load_model_catalog
from models.ollama_service import OllamaService


class OllamaServiceTest(unittest.TestCase):
    def test_status_reports_missing_executable(self) -> None:
        status = OllamaService.status(which_func=lambda _name: None)

        self.assertFalse(status.available)
        self.assertIn("not found", status.detail)

    def test_chat_returns_clear_unavailable_message(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        class UnavailableOllamaService(OllamaService):
            @staticmethod
            def status(*_args, **_kwargs) -> ollama_module.BackendStatus:
                return ollama_module.BackendStatus("ollama", False, "Ollama test unavailable.")

        response = UnavailableOllamaService(catalog["minicpm5_1b"]).chat("", "Hello")

        self.assertIn("[Ollama unavailable]", response)
        self.assertIn("Ollama test unavailable.", response)


if __name__ == "__main__":
    unittest.main()
