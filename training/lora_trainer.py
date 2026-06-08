from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.planner import TrainingPlan, build_training_plan


@dataclass(frozen=True)
class LoraTrainerDependencyReport:
    peft_available: bool
    trl_available: bool
    transformers_available: bool
    torch_available: bool

    @property
    def ready(self) -> bool:
        return all(asdict(self).values())

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ready"] = self.ready
        return data


@dataclass(frozen=True)
class LoraTrainingRequest:
    model_id: str
    dataset_path: str
    output_dir: str
    plan: TrainingPlan
    dependency_report: LoraTrainerDependencyReport
    execute_training: bool
    command_preview: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "plan": self.plan.as_dict(),
            "dependency_report": self.dependency_report.as_dict(),
            "execute_training": self.execute_training,
            "command_preview": self.command_preview,
        }


def lora_dependency_report() -> LoraTrainerDependencyReport:
    return LoraTrainerDependencyReport(
        peft_available=importlib.util.find_spec("peft") is not None,
        trl_available=importlib.util.find_spec("trl") is not None,
        transformers_available=importlib.util.find_spec("transformers") is not None,
        torch_available=importlib.util.find_spec("torch") is not None,
    )


def build_lora_training_request(
    model_id: str,
    dataset_path: str,
    rank: int = 16,
    epochs: int = 1,
    output_root: str | Path = "outputs/checkpoints",
) -> LoraTrainingRequest:
    plan = build_training_plan(
        dataset_path=dataset_path,
        rank=rank,
        epochs=epochs,
        output_root=output_root,
    )
    report = lora_dependency_report()
    return LoraTrainingRequest(
        model_id=model_id,
        dataset_path=dataset_path,
        output_dir=plan.output_dir,
        plan=plan,
        dependency_report=report,
        execute_training=False,
        command_preview=[
            "python",
            "-m",
            "training.lora_trainer",
            "--model-id",
            model_id,
            "--dataset",
            dataset_path,
            "--output-dir",
            plan.output_dir,
        ],
    )


def vision_finetuning_plan() -> dict[str, Any]:
    return {
        "implemented": False,
        "recommended_tools": ["SWIFT", "LLaMA-Factory"],
        "local_first_steps": [
            "Export corrected vision/OCR field notes to JSONL.",
            "Choose MiniCPM-V model and verify local inference first.",
            "Select SWIFT or LLaMA-Factory after hardware is known.",
            "Keep checkpoints, datasets with private media, and model weights out of git.",
        ],
        "blocked_until": [
            "Final vision dataset schema is selected.",
            "GPU/VRAM target is known.",
            "Training framework dependency is approved for installation.",
        ],
    }
