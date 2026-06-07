from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.planner import (
    LoraConfig,
    TrainingConfig,
    build_training_plan,
    load_training_config,
    training_hardware_notes,
    validate_training_plan,
)


class TrainingPlannerTest(unittest.TestCase):
    def test_loads_training_config(self) -> None:
        lora, training = load_training_config("config/training.yaml")

        self.assertEqual(lora.rank, 16)
        self.assertEqual(training.epochs, 1)

    def test_validates_missing_dataset(self) -> None:
        errors = validate_training_plan(
            "missing.jsonl",
            LoraConfig(rank=16, alpha=32, dropout=0.05),
            TrainingConfig(epochs=1, batch_size=2, grad_accum=4, lr=0.0002, report_to="none"),
        )

        self.assertIn("does not exist", errors[0])

    def test_builds_valid_dry_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "field_notes.jsonl"
            dataset.write_text('{"prompt":"hello","correction":"world"}\n', encoding="utf-8")

            plan = build_training_plan(
                str(dataset),
                rank=8,
                epochs=2,
                output_root=Path(tmp) / "out",
            )

            self.assertEqual(plan.lora.rank, 8)
            self.assertEqual(plan.training.epochs, 2)
            self.assertEqual(plan.validation_errors, [])
            self.assertFalse(plan.executes_training)
            self.assertIn("field_notes", plan.output_dir)

    def test_hardware_notes_warn_about_git_weights(self) -> None:
        self.assertTrue(any("git" in note for note in training_hardware_notes()))


if __name__ == "__main__":
    unittest.main()
