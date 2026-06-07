from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.synthetic import (
    SyntheticExample,
    augment_examples,
    export_synthetic_jsonl,
    generate_synthetic_examples,
    quality_filter_examples,
    validate_synthetic_example,
)


class SyntheticDatasetTest(unittest.TestCase):
    def test_generates_deterministic_examples(self) -> None:
        examples = generate_synthetic_examples("field notes", 2)

        self.assertEqual(len(examples), 2)
        self.assertIn("field notes #1", examples[0].prompt)
        self.assertIn("synthetic", examples[0].tags)

    def test_rejects_negative_count(self) -> None:
        with self.assertRaises(ValueError):
            generate_synthetic_examples("demo", -1)

    def test_validates_required_fields(self) -> None:
        errors = validate_synthetic_example(
            SyntheticExample(prompt="", response="", correction="", tags=[])
        )

        self.assertIn("prompt is required", errors)
        self.assertIn("at least one tag is required", errors)

    def test_filters_low_quality_examples(self) -> None:
        good = generate_synthetic_examples("demo", 1)[0]
        bad = SyntheticExample(prompt="same", response="same", correction="better", tags=["x"])

        filtered = quality_filter_examples([good, bad])

        self.assertEqual(filtered, [good])

    def test_augments_examples(self) -> None:
        examples = generate_synthetic_examples("demo", 1)

        augmented = augment_examples(examples)

        self.assertEqual(len(augmented), 2)
        self.assertIn("augmented", augmented[1].tags)
        self.assertIn("verification detail", augmented[1].prompt.casefold())

    def test_exports_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            examples = generate_synthetic_examples("demo", 1)

            exported = export_synthetic_jsonl(examples, path)

            self.assertEqual(exported, path)
            self.assertIn('"prompt"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
