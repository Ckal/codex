from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelInfo:
    """Configured model metadata."""

    config_id: str
    hf_id: str
    display_name: str
    type: str
    parameters_b: float
    backend: str
    context_length: int
    notes: str
    thinking_mode: bool
    gguf: dict[str, Any]


def load_model_catalog(path: str) -> dict[str, ModelInfo]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    catalog = {}
    for config_id, cfg in raw.get("models", {}).items():
        catalog[config_id] = ModelInfo(
            config_id=config_id,
            hf_id=cfg["hf_id"],
            display_name=cfg.get("display_name", config_id),
            type=cfg.get("type", "text"),
            parameters_b=float(cfg.get("parameters_b", 0)),
            backend=cfg.get("backend", "placeholder"),
            context_length=int(cfg.get("context_length", 0)),
            notes=cfg.get("notes", ""),
            thinking_mode=bool(cfg.get("thinking_mode", False)),
            gguf=cfg.get("gguf", {}),
        )
    return catalog


def model_choices(catalog: dict[str, ModelInfo], model_type: str | None = None) -> list[str]:
    choices = []
    for model in catalog.values():
        if model_type and model.type != model_type:
            continue
        choices.append(model.config_id)
    return choices


def model_summary(model: ModelInfo) -> dict[str, Any]:
    return {
        "config_id": model.config_id,
        "hf_id": model.hf_id,
        "display_name": model.display_name,
        "type": model.type,
        "parameters_b": model.parameters_b,
        "backend": model.backend,
        "context_length": model.context_length,
        "thinking_mode": model.thinking_mode,
        "gguf": model.gguf,
        "notes": model.notes,
    }


def validate_catalog(catalog: dict[str, ModelInfo], max_parameters_b: float = 32) -> list[str]:
    warnings = []
    for model in catalog.values():
        if model.parameters_b > max_parameters_b:
            warnings.append(
                f"{model.config_id} has {model.parameters_b}B parameters, "
                f"above the {max_parameters_b}B limit."
            )
        if model.backend == "placeholder":
            warnings.append(
                f"{model.config_id} uses placeholder backend; real inference is not wired yet."
            )
        if not model.hf_id:
            warnings.append(f"{model.config_id} is missing hf_id.")
    return warnings
