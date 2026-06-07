from dataclasses import asdict, dataclass
from typing import Any, cast

from mcp_tools.tools import ToolResult, tool_registry

MCP_PATH = "/gradio_api/mcp/sse"
MCP_MODE = "gradio_native_mcp_server"


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    description: str
    endpoint: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


TOOL_DESCRIPTIONS = {
    "dataset_stats": "Return row, column, and non-empty statistics for a local CSV/JSONL file.",
    "hf_dataset_preview": (
        "Preview a Hugging Face dataset when optional dependencies are installed."
    ),
    "safe_calculator": "Evaluate numeric arithmetic expressions only.",
    "model_inference": "Run a text prompt through the selected local model backend.",
}


def mcp_tool_definitions() -> list[McpToolDefinition]:
    return [
        McpToolDefinition(
            name=name,
            description=TOOL_DESCRIPTIONS.get(name, "Local workbench tool."),
            endpoint=f"{MCP_PATH}#{name}",
        )
        for name in sorted(tool_registry())
    ]


def mcp_manifest() -> dict[str, Any]:
    return {
        "mode": MCP_MODE,
        "path": MCP_PATH,
        "tools": [definition.as_dict() for definition in mcp_tool_definitions()],
        "served_by": "Gradio launch(mcp_server=True)",
    }


def invoke_mcp_tool(name: str, payload: dict[str, Any]) -> ToolResult:
    registry = tool_registry()
    if name not in registry:
        raise KeyError(f"Unknown MCP tool: {name}")
    tool = registry[name]
    return cast(ToolResult, tool(**payload))
