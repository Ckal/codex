from __future__ import annotations

import unittest

import requests

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

    def test_lists_local_models(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response._content = (  # pylint: disable=protected-access
            b'{"models": [{"name": "minicpm:latest"}, {"name": "llama:latest"}]}'
        )

        models = OllamaService.list_local_models(get_func=lambda *_args, **_kwargs: response)

        self.assertEqual(models, ["minicpm:latest", "llama:latest"])

    def test_list_local_models_returns_empty_when_unreachable(self) -> None:
        def raise_error(*_args, **_kwargs):
            raise requests.ConnectionError("offline")

        self.assertEqual(OllamaService.list_local_models(get_func=raise_error), [])

    def test_builds_explicit_pull_command(self) -> None:
        self.assertEqual(
            OllamaService.pull_command(" minicpm:latest "),
            ["ollama", "pull", "minicpm:latest"],
        )
        self.assertEqual(OllamaService.pull_command(""), [])


if __name__ == "__main__":
    unittest.main()
