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


if __name__ == "__main__":
    unittest.main()
