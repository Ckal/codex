from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from datasets.field_notes import FieldNote, FieldNoteStore, SQLiteFieldNoteStore


class FieldNoteStoreTest(unittest.TestCase):
    def test_save_writes_correction_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field_notes.csv"
            store = FieldNoteStore(path)

            saved_path = store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="What plant is this?",
                    response="Unknown plant",
                    correction="It is basil.",
                    tags="plant-id,demo",
                    image_path="images/basil.jpg",
                    video_path="",
                    use_for_training=True,
                )
            )

            self.assertEqual(saved_path, path)
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_id"], "minicpm5_1b")
            self.assertEqual(rows[0]["correction"], "It is basil.")
            self.assertEqual(rows[0]["image_path"], "images/basil.jpg")
            self.assertEqual(rows[0]["use_for_training"], "True")

    def test_created_at_uses_utc_offset_without_python_311_utc_import(self) -> None:
        note = FieldNote.create(
            model_id="minicpm5_1b",
            prompt="Prompt",
            response="Response",
            correction="Correction",
            tags="demo",
        )

        self.assertTrue(note.created_at.endswith("+00:00"))

    def test_exports_corrected_notes_to_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field_notes.csv"
            output = Path(tmp) / "field_notes.jsonl"
            store = FieldNoteStore(path)
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt",
                    response="Response",
                    correction="Correction",
                    tags="demo",
                )
            )
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt 2",
                    response="Response 2",
                    correction="",
                    tags="demo",
                )
            )
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt 3",
                    response="Response 3",
                    correction="Correction 3",
                    tags="demo",
                    use_for_training=False,
                )
            )

            exported = store.export_jsonl(output)

            self.assertEqual(exported, output)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("Correction", lines[0])

    def test_filters_notes_by_tag_and_training_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "field_notes.csv")
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt",
                    response="Response",
                    correction="Correction",
                    tags="ocr,plant-id",
                    use_for_training=True,
                )
            )
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt 2",
                    response="Response 2",
                    correction="Correction 2",
                    tags="audio",
                    use_for_training=False,
                )
            )

            self.assertEqual(len(store.list_notes(tag="plant-id")), 1)
            self.assertEqual(len(store.list_notes(training_only=True)), 1)

    def test_exports_local_hf_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "field_notes.csv")
            target = Path(tmp) / "hf_dataset"
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt",
                    response="Response",
                    correction="Correction",
                    tags="demo",
                )
            )

            exported = store.export_hf_dataset(target)

            self.assertEqual(exported, target)
            self.assertTrue((target / "data.jsonl").exists())
            self.assertTrue((target / "README.md").exists())

    def test_sqlite_store_saves_and_lists_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteFieldNoteStore(Path(tmp) / "field_notes.sqlite")
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Prompt",
                    response="Response",
                    correction="Correction",
                    tags="demo",
                    image_path="image.png",
                    use_for_training=True,
                )
            )

            notes = store.list_notes(corrected_only=True, tag="demo", training_only=True)

            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].image_path, "image.png")


if __name__ == "__main__":
    unittest.main()
