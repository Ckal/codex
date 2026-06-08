from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.lora_trainer import (
    build_lora_training_request,
    lora_dependency_report,
    vision_finetuning_plan,
)


class LoraTrainerTest(unittest.TestCase):
    def test_dependency_report_is_serializable(self) -> None:
        report = lora_dependency_report().as_dict()

        self.assertIn("ready", report)
        self.assertIn("peft_available", report)
        self.assertIn("trl_available", report)

    def test_builds_non_executing_lora_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "train.jsonl"
            dataset.write_text('{"prompt":"hello","correction":"world"}\n', encoding="utf-8")

            request = build_lora_training_request(
                "minicpm5_1b",
                str(dataset),
                rank=8,
                epochs=2,
                output_root=Path(tmp) / "out",
            )

            data = request.as_dict()
            self.assertFalse(data["execute_training"])
            self.assertEqual(data["plan"]["lora"]["rank"], 8)
            self.assertIn("--model-id", data["command_preview"])

    def test_vision_finetuning_plan_mentions_swift_and_llama_factory(self) -> None:
        plan = vision_finetuning_plan()

        self.assertFalse(plan["implemented"])
        self.assertIn("SWIFT", plan["recommended_tools"])
        self.assertIn("LLaMA-Factory", plan["recommended_tools"])


if __name__ == "__main__":
    unittest.main()
