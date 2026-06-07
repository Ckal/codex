from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcp_tools.tools import (
    dataset_stats_tool,
    hf_dataset_preview_tool,
    model_inference_tool,
    safe_calculator_tool,
    tool_registry,
    vindex_plan_tool,
)


class McpToolsTest(unittest.TestCase):
    def test_dataset_stats_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("prompt,response\nhello,world\nempty,\n", encoding="utf-8")

            result = dataset_stats_tool(str(path))

            self.assertEqual(result.name, "dataset_stats")
            self.assertEqual(result.payload["rows"], 2)
            self.assertEqual(result.payload["non_empty_by_column"]["response"], 1)

    def test_safe_calculator_tool(self) -> None:
        result = safe_calculator_tool("2 + 3 * 4")

        self.assertEqual(result.payload["value"], 14)

    def test_safe_calculator_rejects_names(self) -> None:
        with self.assertRaises(ValueError):
            safe_calculator_tool("__import__('os').system('echo nope')")

    def test_model_inference_tool_uses_placeholder_backend(self) -> None:
        result = model_inference_tool("Hello")

        self.assertIn("Placeholder response", result.payload["response"])

    def test_hf_dataset_preview_tool_reports_missing_optional_package(self) -> None:
        result = hf_dataset_preview_tool("demo/dataset")

        self.assertEqual(result.name, "hf_dataset_preview")
        self.assertIn("status", result.payload)

    def test_tool_registry_contains_expected_tools(self) -> None:
        registry = tool_registry()

        self.assertIn("dataset_stats", registry)
        self.assertIn("safe_calculator", registry)
        self.assertIn("model_inference", registry)
        self.assertIn("vindex_plan", registry)

    def test_vindex_plan_tool_returns_non_executing_plan(self) -> None:
        result = vindex_plan_tool("logit_lens", {"model_id": "demo", "text": "hello"})

        self.assertEqual(result.name, "vindex_plan")
        self.assertFalse(result.payload["ready_for_execution"])
        self.assertEqual(result.payload["call_plan"]["method"], "logit_lens")


if __name__ == "__main__":
    unittest.main()
