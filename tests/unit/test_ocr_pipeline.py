from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datasets.field_notes import FieldNote, FieldNoteStore
from datasets.ocr import (
    export_corrected_ocr_notes,
    import_uncertain_predictions,
    load_ocr_predictions,
    ocr_import_summary,
    uncertain_predictions,
)
from ui.notes_tab import import_ocr_predictions, preview_ocr_predictions


class OCRPipelineTest(unittest.TestCase):
    def test_loads_ocr_predictions_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_path": "receipt.png",
                        "text": "Total 12.30",
                        "confidence": 0.91,
                        "page": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            predictions = load_ocr_predictions(path)

            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0].source_path, "receipt.png")
            self.assertEqual(predictions[0].confidence, 0.91)
            self.assertEqual(predictions[0].page, "1")

    def test_loads_ocr_predictions_from_csv_alias_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr.csv"
            path.write_text(
                "image_path,prediction,score\nlabel.png,Best before 2026,0.72\n",
                encoding="utf-8",
            )

            predictions = load_ocr_predictions(path)

            self.assertEqual(predictions[0].source_path, "label.png")
            self.assertEqual(predictions[0].text, "Best before 2026")
            self.assertEqual(predictions[0].confidence, 0.72)

    def test_filters_uncertain_predictions_by_threshold_and_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr.jsonl"
            rows = [
                {"source_path": "a.png", "text": "clear", "confidence": 0.95},
                {"source_path": "b.png", "text": "maybe", "confidence": 0.7},
                {"source_path": "c.png", "text": "", "confidence": 0.99},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            uncertain = uncertain_predictions(load_ocr_predictions(path), 0.8)

            self.assertEqual([item.source_path for item in uncertain], ["b.png", "c.png"])

    def test_imports_uncertain_predictions_to_field_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictions_path = Path(tmp) / "ocr.jsonl"
            predictions_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_path": "uncertain.png",
                                "text": "hel1o",
                                "confidence": 0.5,
                            }
                        ),
                        json.dumps(
                            {
                                "source_path": "confident.png",
                                "text": "hello",
                                "confidence": 0.99,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            store = FieldNoteStore(Path(tmp) / "field_notes.csv")

            imported = import_uncertain_predictions(
                store,
                load_ocr_predictions(predictions_path),
                "minicpm5_1b",
                confidence_threshold=0.8,
            )

            notes = store.list_notes(tag="ocr")
            self.assertEqual(imported, 1)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].response, "hel1o")
            self.assertEqual(notes[0].image_path, "uncertain.png")
            self.assertFalse(notes[0].use_for_training)

    def test_exports_corrected_ocr_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "field_notes.csv")
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Review OCR text for receipt.png.",
                    response="Tota1",
                    correction="Total",
                    tags="ocr,uncertain",
                    image_path="receipt.png",
                )
            )
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Other correction",
                    response="abc",
                    correction="abc",
                    tags="demo",
                )
            )

            output = export_corrected_ocr_notes(store, Path(tmp) / "ocr_corrections.jsonl")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_path"], "receipt.png")
            self.assertEqual(rows[0]["predicted_text"], "Tota1")
            self.assertEqual(rows[0]["corrected_text"], "Total")

    def test_ocr_import_summary_reports_uncertain_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr.jsonl"
            path.write_text(
                json.dumps({"source_path": "a.png", "text": "?", "confidence": 0.1}),
                encoding="utf-8",
            )

            summary = ocr_import_summary(path, 0.8)

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["uncertain_rows"], 1)
            self.assertEqual(summary["confidence_threshold"], 0.8)

    def test_notes_tab_ocr_preview_callback_reports_missing_path(self) -> None:
        self.assertEqual(
            preview_ocr_predictions("", 0.8),
            {"error": "Enter a local OCR prediction file path."},
        )

    def test_notes_tab_ocr_import_callback_binds_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr.jsonl"
            path.write_text(
                json.dumps({"source_path": "a.png", "text": "?", "confidence": 0.1}),
                encoding="utf-8",
            )
            store = FieldNoteStore(Path(tmp) / "field_notes.csv")

            status = import_ocr_predictions(store, "minicpm5_1b", str(path), 0.8)

            self.assertIn("Imported 1 uncertain OCR predictions", status)
            self.assertEqual(len(store.list_notes(tag="ocr")), 1)


if __name__ == "__main__":
    unittest.main()
