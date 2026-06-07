from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoraConfig:
    """LoRA settings for dry-run training plans."""

    rank: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings for dry-run validation."""

    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    report_to: str


@dataclass(frozen=True)
class TrainingPlan:
    """Non-executing local training plan."""

    dataset_path: str
    output_dir: str
    lora: LoraConfig
    training: TrainingConfig
    validation_errors: list[str]
    hardware_notes: list[str]
    executes_training: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "validation_errors": self.validation_errors,
            "hardware_notes": self.hardware_notes,
            "executes_training": self.executes_training,
        }

    def as_text(self) -> str:
        status = "valid" if not self.validation_errors else "blocked"
        lines = [
            "Training dry-run plan",
            f"Status: {status}",
            f"Dataset: {self.dataset_path or '(none selected)'}",
            f"Output directory: {self.output_dir}",
            f"LoRA rank: {self.lora.rank}",
            f"LoRA alpha: {self.lora.alpha}",
            f"LoRA dropout: {self.lora.dropout}",
            f"Epochs: {self.training.epochs}",
            f"Batch size: {self.training.batch_size}",
            f"Gradient accumulation: {self.training.grad_accum}",
            f"Learning rate: {self.training.lr}",
            f"Report to: {self.training.report_to}",
            f"Executes training: {self.executes_training}",
            "",
            "Validation:",
            *[f"- {item}" for item in (self.validation_errors or ["passed"])],
            "",
            "Hardware notes:",
            *[f"- {item}" for item in self.hardware_notes],
        ]
        return "\n".join(lines)


def load_training_config(
    path: str | Path = "config/training.yaml",
) -> tuple[LoraConfig, TrainingConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    lora = raw.get("lora", {})
    training = raw.get("training", {})
    return (
        LoraConfig(
            rank=int(lora.get("rank", 16)),
            alpha=int(lora.get("alpha", 32)),
            dropout=float(lora.get("dropout", 0.05)),
        ),
        TrainingConfig(
            epochs=int(training.get("epochs", 1)),
            batch_size=int(training.get("batch_size", 2)),
            grad_accum=int(training.get("grad_accum", 4)),
            lr=float(training.get("lr", 0.0002)),
            report_to=str(training.get("report_to", "none")),
        ),
    )


def build_training_plan(
    dataset_path: str,
    rank: int | None = None,
    epochs: int | None = None,
    output_root: str | Path = "outputs/checkpoints",
    config_path: str | Path = "config/training.yaml",
) -> TrainingPlan:
    lora, training = load_training_config(config_path)
    if rank is not None:
        lora = LoraConfig(rank=rank, alpha=max(rank * 2, lora.alpha), dropout=lora.dropout)
    if epochs is not None:
        training = TrainingConfig(
            epochs=epochs,
            batch_size=training.batch_size,
            grad_accum=training.grad_accum,
            lr=training.lr,
            report_to=training.report_to,
        )

    output_dir = str(Path(output_root) / _safe_output_name(dataset_path))
    return TrainingPlan(
        dataset_path=dataset_path,
        output_dir=output_dir,
        lora=lora,
        training=training,
        validation_errors=validate_training_plan(dataset_path, lora, training),
        hardware_notes=training_hardware_notes(),
    )


def validate_training_plan(
    dataset_path: str,
    lora: LoraConfig,
    training: TrainingConfig,
) -> list[str]:
    errors = []
    if not dataset_path:
        errors.append("Training dataset path is required.")
    elif not Path(dataset_path).exists():
        errors.append(f"Training dataset does not exist: {dataset_path}")
    if lora.rank <= 0:
        errors.append("LoRA rank must be positive.")
    if not 0 <= lora.dropout < 1:
        errors.append("LoRA dropout must be between 0 and 1.")
    if training.epochs <= 0:
        errors.append("Epochs must be positive.")
    if training.batch_size <= 0:
        errors.append("Batch size must be positive.")
    if training.grad_accum <= 0:
        errors.append("Gradient accumulation must be positive.")
    if training.lr <= 0:
        errors.append("Learning rate must be positive.")
    return errors


def training_hardware_notes() -> list[str]:
    return [
        "MiniCPM 1B LoRA may run on modest CUDA hardware; CPU-only training will be slow.",
        "MiniCPM 8B and vision fine-tuning need substantially more VRAM "
        "or quantized/adapted tooling.",
        "Install PEFT/TRL, SWIFT, or LLaMA-Factory only after choosing the final training path.",
        "Keep checkpoints and model weights out of git.",
    ]


def _safe_output_name(dataset_path: str) -> str:
    if not dataset_path:
        return "unselected"
    stem = Path(dataset_path).stem or "dataset"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
