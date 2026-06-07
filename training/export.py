from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.model_catalog import ModelInfo

QUANTIZATION_CHOICES = ["F16", "Q4_K_M", "Q5_K_M", "Q8_0"]


@dataclass(frozen=True)
class ToolStatus:
    """Availability of a local export-related tool."""

    name: str
    available: bool
    path: str


@dataclass(frozen=True)
class ExportPlan:
    """Non-executing GGUF export plan."""

    model_id: str
    hf_id: str
    quantization: str
    output_dir: str
    official_gguf_repo: str
    official_gguf_file: str
    download_command: list[str]
    convert_command: list[str]
    quantize_command: list[str]
    tool_statuses: list[ToolStatus]
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hf_id": self.hf_id,
            "quantization": self.quantization,
            "output_dir": self.output_dir,
            "official_gguf_repo": self.official_gguf_repo,
            "official_gguf_file": self.official_gguf_file,
            "download_command": self.download_command,
            "convert_command": self.convert_command,
            "quantize_command": self.quantize_command,
            "tools": [tool.__dict__ for tool in self.tool_statuses],
            "notes": self.notes,
            "executes_commands": False,
            "startup_downloads": False,
        }


def detect_llama_cpp_tools(
    which_func: Callable[[str], str | None] = shutil.which,
) -> list[ToolStatus]:
    tool_names = ["llama-server", "llama-cli", "llama-quantize"]
    return [
        ToolStatus(name=name, available=bool(path := which_func(name)), path=path or "")
        for name in tool_names
    ]


def list_exported_files(output_dir: str | Path = "exports") -> list[list[str]]:
    root = Path(output_dir)
    if not root.exists():
        return []

    rows: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append([str(path), str(path.stat().st_size)])
    return rows


def build_export_plan(
    model: ModelInfo,
    quantization: str,
    output_dir: str | Path = "exports",
    tools: list[ToolStatus] | None = None,
) -> ExportPlan:
    if quantization not in QUANTIZATION_CHOICES:
        raise ValueError(f"Unsupported quantization: {quantization}")

    root = Path(output_dir)
    model_output_dir = root / model.config_id
    gguf = model.gguf or {}
    repo = str(gguf.get("repo", ""))
    official_file = str(
        gguf.get("main_file")
        or f"{model.display_name.replace(' ', '-')}-{quantization}.gguf"
    )
    base_gguf = model_output_dir / "converted-f16.gguf"
    quantized_gguf = model_output_dir / official_file

    notes = [
        "This plan does not execute downloads, conversion, or quantization.",
        "Run commands manually after installing dependencies and verifying paths.",
    ]
    if not repo:
        notes.append("No official GGUF repo is configured for this model.")

    download_command = []
    if repo:
        download_command = [
            "huggingface-cli",
            "download",
            repo,
            official_file,
            "--local-dir",
            str(model_output_dir),
        ]

    convert_command = [
        "python",
        "path\\to\\llama.cpp\\convert_hf_to_gguf.py",
        model.hf_id,
        "--outfile",
        str(base_gguf),
    ]

    quantize_command = [
        "llama-quantize",
        str(base_gguf),
        str(quantized_gguf),
        quantization,
    ]

    return ExportPlan(
        model_id=model.config_id,
        hf_id=model.hf_id,
        quantization=quantization,
        output_dir=str(model_output_dir),
        official_gguf_repo=repo,
        official_gguf_file=official_file,
        download_command=download_command,
        convert_command=convert_command,
        quantize_command=quantize_command,
        tool_statuses=tools or detect_llama_cpp_tools(),
        notes=notes,
    )
