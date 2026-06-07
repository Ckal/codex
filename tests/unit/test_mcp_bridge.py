from __future__ import annotations

import unittest

from mcp_tools.bridge import MCP_PATH, invoke_mcp_tool, mcp_manifest, mcp_tool_definitions


class McpBridgeTest(unittest.TestCase):
    def test_manifest_documents_gradio_native_path(self) -> None:
        manifest = mcp_manifest()

        self.assertEqual(manifest["path"], MCP_PATH)
        self.assertEqual(manifest["mode"], "gradio_native_mcp_server")
        self.assertIn("launch(mcp_server=True)", manifest["served_by"])
        self.assertTrue(manifest["tools"])

    def test_tool_definitions_include_registry_tools(self) -> None:
        names = [definition.name for definition in mcp_tool_definitions()]

        self.assertIn("safe_calculator", names)
        self.assertIn("model_inference", names)

    def test_invokes_registered_tool(self) -> None:
        result = invoke_mcp_tool("safe_calculator", {"expression": "2 + 2"})

        self.assertEqual(result.name, "safe_calculator")
        self.assertEqual(result.payload["value"], 4)

    def test_rejects_unknown_tool(self) -> None:
        with self.assertRaises(KeyError):
            invoke_mcp_tool("missing", {})


if __name__ == "__main__":
    unittest.main()
