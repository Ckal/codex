from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.field_notes import FieldNote, FieldNoteStore
from plant.app import build_app, load_config
from plant.plant_loader import (
    FieldNotesPlantExporter,
    LocalFolderLoader,
    SpeciesIndexBuilder,
    demo_species,
)
from plant.plant_service import DemoPlantVisionService, extract_json_object, parse_plant_response
from plant.plant_tab import (
    identify_plant_callback,
    plant_training_plan,
    render_result_card,
    save_field_note_callback,
    species_table,
)
from plant.plant_tools import dataset_stats, search_species, set_services, training_plan
from plant.training import build_plant_training_plan, write_llamafactory_dataset_info


class PlantReferenceAppTest(unittest.TestCase):
    def test_plant_config_loads_domain_and_model(self) -> None:
        config = load_config("plant/models.yaml")

        self.assertEqual(config["domain"]["name"], "plant_discovery")
        self.assertEqual(config["models"]["plant_vlm"]["hf_id"], "openbmb/MiniCPM-V-4.6")

    def test_no_model_app_builds_gradio_blocks(self) -> None:
        demo = build_app(no_model=True)

        self.assertEqual(type(demo).__name__, "Blocks")

    def test_default_app_builds_openbmb_service_without_loading_weights(self) -> None:
        demo = build_app()

        self.assertEqual(type(demo).__name__, "Blocks")

    def test_demo_service_returns_structured_plant_result(self) -> None:
        result = DemoPlantVisionService().identify(object(), force_thinking=True)

        self.assertEqual(result.latin_name, "Bellis perennis")
        self.assertGreater(result.confidence, 0.8)
        self.assertEqual(result.to_dict()["family"], "Asteraceae")

    def test_demo_service_reports_no_llm_usage(self) -> None:
        status = DemoPlantVisionService().service_status()

        self.assertFalse(status["uses_llm"])
        self.assertEqual(status["mode"], "demo")

    def test_extract_json_repairs_common_model_wrapping(self) -> None:
        parsed = extract_json_object('```json\n{"latin_name":"Rosa canina",}\n```')

        self.assertEqual(parsed["latin_name"], "Rosa canina")

    def test_parse_plant_response_builds_schema(self) -> None:
        result = parse_plant_response(
            '{"common_name":"Oak","latin_name":"Quercus robur","confidence":0.91}',
            model_used="demo",
        )

        self.assertEqual(result.genus, "Quercus")
        self.assertEqual(result.model_used, "demo")

    def test_local_folder_loader_uses_species_folder_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            species_dir = Path(tmp) / "Acer_palmatum"
            species_dir.mkdir()
            (species_dir / "leaf.jpg").write_bytes(b"not a real image but good metadata")

            loader = LocalFolderLoader(tmp)

            self.assertEqual(loader.species_list(), ["Acer palmatum"])
            self.assertEqual(loader.iter_records()[0].latin_name, "Acer palmatum")

    def test_species_index_falls_back_to_demo_species(self) -> None:
        index = SpeciesIndexBuilder(root="missing-plant-root").build({"datasets": {}})

        self.assertIn("Bellis perennis", index)

    def test_species_table_filters_by_query_and_family(self) -> None:
        rows = species_table(demo_species(), "rose", "Rosaceae")

        self.assertEqual(rows[0][0], "Rosa canina")

    def test_render_result_card_escapes_model_text(self) -> None:
        html = render_result_card(
            {
                "common_name": "<script>alert(1)</script>",
                "latin_name": "Bellis perennis",
                "confidence": 0.5,
            }
        )

        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)

    def test_identify_callback_handles_missing_image(self) -> None:
        _card, data, state = identify_plant_callback(
            DemoPlantVisionService(),
            None,
            None,
            "standard",
        )

        self.assertEqual(data["common_name"], "No image")
        self.assertEqual(state["latin_name"], "Upload a plant image")

    def test_save_field_note_and_export_training_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "plant_notes.csv")
            message = save_field_note_callback(
                store,
                {"latin_name": "Wrong plant", "model_used": "demo"},
                "Bellis perennis",
                "Corrected from field observation.",
            )
            exporter = FieldNotesPlantExporter(store.path)
            output = exporter.export_jsonl(Path(tmp) / "plant_training.jsonl")

            self.assertIn("Saved field note", message)
            self.assertEqual(len(store.list_notes(corrected_only=True)), 1)
            self.assertIn("Bellis perennis", output.read_text(encoding="utf-8"))

    def test_training_plan_is_non_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "plant_notes.csv")
            store.save(
                FieldNote.create(
                    model_id="demo",
                    prompt="Identify",
                    response="Wrong",
                    correction="Rosa canina",
                    tags="plant-discovery",
                )
            )

            plan = plant_training_plan(store)

            self.assertFalse(plan["execute_training"])
            self.assertEqual(plan["corrected_examples"], 1)
            self.assertIn("swift", plan["swift_command"])

    def test_build_plant_training_plan_documents_openbmb_and_adapter_mode(self) -> None:
        plan = build_plant_training_plan(corrected_examples=42)

        self.assertFalse(plan.execute_training)
        self.assertEqual(plan.base_model, "openbmb/MiniCPM-V-4.6")
        self.assertEqual(plan.corrected_examples, 42)
        self.assertIn("--model-mode", plan.use_trained_model_command)

    def test_write_llamafactory_dataset_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = write_llamafactory_dataset_info(
                dataset_path="data/plant_training.jsonl",
                output_path=Path(tmp) / "dataset_info.json",
            )

            self.assertIn("plant_discovery", output.read_text(encoding="utf-8"))

    def test_tools_use_registered_services_without_hard_mcp_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FieldNoteStore(Path(tmp) / "plant_notes.csv")
            set_services(DemoPlantVisionService(), store, demo_species())

            self.assertEqual(search_species("daisy")[0]["latin_name"], "Bellis perennis")
            self.assertEqual(dataset_stats()["species_index_size"], len(demo_species()))
            self.assertFalse(training_plan()["execute_training"])
            self.assertIn("swift_command", training_plan())


if __name__ == "__main__":
    unittest.main()
