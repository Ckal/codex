from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.model_catalog import load_model_catalog
from training.export import (
    ToolStatus,
    build_export_plan,
    detect_llama_cpp_tools,
    list_exported_files,
)
from ui.export_tab import export_download_paths


class ExportPlanTest(unittest.TestCase):
    def test_detects_llama_cpp_tools_without_requiring_install(self) -> None:
        statuses = detect_llama_cpp_tools(which_func=lambda name: f"C:/tools/{name}.exe")

        self.assertEqual(
            [status.name for status in statuses],
            ["llama-server", "llama-cli", "llama-quantize"],
        )
        self.assertTrue(all(status.available for status in statuses))

    def test_builds_non_executing_official_gguf_plan(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        tools = [ToolStatus("llama-quantize", False, "")]

        plan = build_export_plan(catalog["minicpm_v46"], "Q4_K_M", "exports", tools)
        data = plan.as_dict()

        self.assertEqual(data["official_gguf_repo"], "openbmb/MiniCPM-V-4.6-gguf")
        self.assertIn("huggingface-cli", data["download_command"])
        self.assertIn("llama-quantize", data["quantize_command"])
        self.assertFalse(data["executes_commands"])
        self.assertFalse(data["startup_downloads"])

    def test_rejects_unknown_quantization(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        with self.assertRaises(ValueError):
            build_export_plan(catalog["minicpm5_1b"], "Q2_UNKNOWN")

    def test_lists_exported_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            file_path = output / "model.gguf"
            file_path.write_bytes(b"gguf")

            rows = list_exported_files(output)

            self.assertEqual(rows, [[str(file_path), "4"]])

    def test_export_download_paths_prioritizes_existing_planned_files(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        tools = [ToolStatus("llama-quantize", False, "")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "minicpm_v46"
            model_dir.mkdir()
            extra_file = root / "notes.txt"
            extra_file.write_text("manual note", encoding="utf-8")

            plan = build_export_plan(catalog["minicpm_v46"], "Q4_K_M", root, tools)
            planned_file = model_dir / plan.official_gguf_file
            planned_file.write_bytes(b"planned gguf")

            rows = list_exported_files(root)
            paths = export_download_paths(plan.as_dict(), rows)

            self.assertEqual(paths[0], str(planned_file))
            self.assertIn(str(extra_file), paths)

    def test_export_download_paths_omits_missing_planned_files(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        tools = [ToolStatus("llama-quantize", False, "")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra_file = root / "already-exported.gguf"
            extra_file.write_bytes(b"existing")

            plan = build_export_plan(catalog["minicpm_v46"], "Q4_K_M", root, tools)
            paths = export_download_paths(plan.as_dict(), list_exported_files(root))

            self.assertEqual(paths, [str(extra_file)])


if __name__ == "__main__":
    unittest.main()
