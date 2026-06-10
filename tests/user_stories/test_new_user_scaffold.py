from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.deployment import DeploymentPolicy
from datasets.field_notes import FieldNote, FieldNoteStore
from models.model_catalog import load_model_catalog, validate_catalog
from models.service_factory import create_text_service


class NewUserScaffoldStoryTest(unittest.TestCase):
    def test_new_user_can_inspect_model_try_chat_and_save_correction(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        warnings = validate_catalog(catalog)
        service = create_text_service(
            catalog["minicpm5_1b"],
            "placeholder",
            DeploymentPolicy("local"),
        )

        response = service.chat("Be concise.", "What can this workbench do?")

        self.assertIn("Real inference is not wired yet", response)
        self.assertFalse(any("placeholder backend" in warning for warning in warnings))

        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "field_notes.csv")
            saved_path = store.save(
                FieldNote.create(
                    model_id="minicpm5_1b",
                    prompt="What can this workbench do?",
                    response=response,
                    correction="It should explain the local-first workflow.",
                    tags="new-user,e2e",
                )
            )

            self.assertTrue(saved_path.exists())


if __name__ == "__main__":
    unittest.main()
