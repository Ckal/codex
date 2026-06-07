from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets.loader import dataset_statistics, preview_huggingface_dataset
from models.model_catalog import load_model_catalog
from models.service_factory import create_text_service


@dataclass(frozen=True)
class ToolResult:
    """Simple serializable result for local tool calls."""

    name: str
    payload: dict[str, Any]


def dataset_stats_tool(path: str) -> ToolResult:
    return ToolResult("dataset_stats", dataset_statistics(path).as_dict())


def hf_dataset_preview_tool(dataset_id: str, split: str = "train") -> ToolResult:
    preview = preview_huggingface_dataset(dataset_id, split)
    return ToolResult(
        "hf_dataset_preview",
        {
            "dataset_id": preview.dataset_id,
            "split": preview.split,
            "rows": preview.rows,
            "columns": preview.columns,
            "samples": preview.samples,
            "status": preview.status,
        },
    )


def safe_calculator_tool(expression: str) -> ToolResult:
    value = _safe_eval(expression)
    return ToolResult("safe_calculator", {"expression": expression, "value": value})


def model_inference_tool(
    prompt: str,
    model_id: str = "minicpm5_1b",
    backend: str = "placeholder",
    system_prompt: str = "",
) -> ToolResult:
    catalog = load_model_catalog("config/models.yaml")
    service = create_text_service(catalog[model_id], backend)
    response = service.chat(system_prompt, prompt)
    return ToolResult(
        "model_inference",
        {
            "model_id": model_id,
            "backend": backend,
            "response": response,
        },
    )


def tool_registry() -> dict[str, Any]:
    return {
        "dataset_stats": dataset_stats_tool,
        "hf_dataset_preview": hf_dataset_preview_tool,
        "safe_calculator": safe_calculator_tool,
        "model_inference": model_inference_tool,
    }


def _safe_eval(expression: str) -> float:
    operators: dict[type[ast.operator] | type[ast.unaryop], Callable[..., float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return float(operators[type(node.op)](eval_node(node.left), eval_node(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return float(operators[type(node.op)](eval_node(node.operand)))
        raise ValueError("Only numeric arithmetic expressions are allowed.")

    return eval_node(ast.parse(expression, mode="eval"))
