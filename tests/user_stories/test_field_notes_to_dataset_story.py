from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.field_notes import FieldNote, FieldNoteStore
from datasets.loader import preview_local_dataset


class FieldNotesToDatasetStoryTest(unittest.TestCase):
    def test_user_exports_corrections_then_previews_training_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            note_path = Path(tmp) / "field_notes.csv"
            jsonl_path = Path(tmp) / "training.jsonl"
            store = FieldNoteStore(note_path)
            store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="Identify this plant.",
                    response="Unknown.",
                    correction="Basil.",
                    tags="plant-id",
                    image_path="images/basil.jpg",
                    use_for_training=True,
                )
            )

            store.export_jsonl(jsonl_path)
            preview = preview_local_dataset(jsonl_path)

            self.assertEqual(preview.rows, 1)
            self.assertIn("correction", preview.columns)
            self.assertIn("image_path", preview.columns)
            self.assertEqual(preview.samples[0]["correction"], "Basil.")


if __name__ == "__main__":
    unittest.main()
