from __future__ import annotations

import unittest

from models.model_catalog import load_model_catalog
from models.transformers_text import TransformersTextConfig, TransformersTextService


class TransformersTextServiceTest(unittest.TestCase):
    def test_status_reports_missing_package_or_installed_package(self) -> None:
        status = TransformersTextService.status()

        self.assertEqual(status.name, "transformers")
        self.assertIsInstance(status.available, bool)

    def test_unavailable_chat_does_not_download_weights(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = TransformersTextService(catalog["minicpm5_1b"])

        response = service.chat("Be concise.", "Hello")

        if not TransformersTextService.status().available:
            self.assertIn("[Transformers unavailable]", response)

    def test_generation_kwargs_are_explicit(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = TransformersTextService(
            catalog["minicpm5_1b"],
            TransformersTextConfig(max_new_tokens=32, temperature=0.2, do_sample=False),
        )

        self.assertEqual(
            service.generation_kwargs(),
            {"max_new_tokens": 32, "temperature": 0.2, "do_sample": False},
        )

    def test_fallback_chat_prompt_format(self) -> None:
        class PlainTokenizer:
            pass

        prompt = TransformersTextService._format_chat_prompt(
            PlainTokenizer(),
            "System",
            "User",
        )

        self.assertIn("system: System", prompt)
        self.assertIn("user: User", prompt)
        self.assertTrue(prompt.endswith("assistant:"))


if __name__ == "__main__":
    unittest.main()
