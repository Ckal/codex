from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.loader import (
    dataset_statistics,
    preview_huggingface_dataset,
    preview_local_dataset,
)


class DatasetLoaderTest(unittest.TestCase):
    def test_previews_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.csv"
            path.write_text("prompt,response\nhello,world\nfoo,bar\n", encoding="utf-8")

            preview = preview_local_dataset(path)

            self.assertEqual(preview.rows, 2)
            self.assertEqual(preview.columns, ["prompt", "response"])
            self.assertEqual(preview.samples[0]["prompt"], "hello")

    def test_previews_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text('{"prompt":"hello","response":"world"}\n', encoding="utf-8")

            preview = preview_local_dataset(path)

            self.assertEqual(preview.rows, 1)
            self.assertEqual(preview.columns, ["prompt", "response"])
            self.assertIn("sample_1", preview.as_table()[3][0])

    def test_rejects_jsonl_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text('["not", "an", "object"]\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                preview_local_dataset(path)

    def test_rejects_unsupported_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.txt"
            path.write_text("hello", encoding="utf-8")

            with self.assertRaises(ValueError):
                preview_local_dataset(path)

    def test_calculates_dataset_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.csv"
            path.write_text("prompt,response\nhello,world\nempty,\n", encoding="utf-8")

            stats = dataset_statistics(path)

            self.assertEqual(stats.rows, 2)
            self.assertEqual(stats.columns, 2)
            self.assertEqual(stats.non_empty_by_column["response"], 1)

    def test_calculates_jsonl_dataset_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text(
                '{"prompt":"hello","response":"world"}\n'
                '{"prompt":"empty","response":""}\n',
                encoding="utf-8",
            )

            stats = dataset_statistics(path)

            self.assertEqual(stats.as_dict()["rows"], 2)
            self.assertEqual(stats.non_empty_by_column["response"], 1)

    def test_huggingface_preview_reports_missing_optional_dependency(self) -> None:
        preview = preview_huggingface_dataset("demo/dataset")

        self.assertEqual(preview.dataset_id, "demo/dataset")
        self.assertIn("datasets", preview.status)


if __name__ == "__main__":
    unittest.main()
