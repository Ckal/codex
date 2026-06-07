from __future__ import annotations

import unittest
from typing import cast

from models.minicpm_vision import MiniCPMVisionConfig, MiniCPMVisionService
from models.model_catalog import load_model_catalog


class MiniCPMVisionServiceTest(unittest.TestCase):
    def test_status_reports_transformers_availability(self) -> None:
        status = MiniCPMVisionService.status()

        self.assertEqual(status.name, "transformers-vision")
        self.assertIsInstance(status.available, bool)

    def test_unavailable_or_missing_image_path_does_not_load_weights(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = MiniCPMVisionService(catalog["minicpm_v46"])

        response = service.vision_chat(False, "Describe it.")

        self.assertIn("[MiniCPM vision unavailable]", response)

    def test_generation_kwargs_are_explicit(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        service = MiniCPMVisionService(
            catalog["minicpm_v46"],
            MiniCPMVisionConfig(max_new_tokens=64, temperature=0.1, do_sample=False),
        )

        self.assertEqual(
            service.generation_kwargs(),
            {"max_new_tokens": 64, "temperature": 0.1, "do_sample": False},
        )

    def test_formats_image_text_prompt_with_thinking_instruction(self) -> None:
        messages = MiniCPMVisionService.format_messages(
            "Read the label.",
            image="image-object",
            thinking=True,
        )

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["type"], "text")
        self.assertIn("Think carefully", messages[0]["content"][1]["text"])

    def test_video_support_plan_is_documented_but_not_enabled(self) -> None:
        plan = MiniCPMVisionService.video_support_plan()
        next_steps = cast(list[str], plan["next_steps"])

        self.assertFalse(plan["implemented"])
        self.assertTrue(any("Sample frames locally" in step for step in next_steps))


if __name__ == "__main__":
    unittest.main()
