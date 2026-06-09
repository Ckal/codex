from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PlantTrainingPlan:
    execute_training: bool
    backend: str
    base_model: str
    dataset_path: str
    output_dir: str
    adapter_repo: str
    corrected_examples: int
    minimum_recommended_examples: int
    dependency_report: dict[str, bool]
    swift_command: list[str]
    llamafactory_command: list[str]
    publish_commands: list[list[str]]
    use_trained_model_command: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plant_training_plan(
    config_path: str | Path = "plant/models.yaml",
    dataset_path: str | Path = "data/plant_training.jsonl",
    output_dir: str | Path = "checkpoints/plant_lora",
    adapter_repo: str = "your-username/minicpm-v46-plant-lora",
    corrected_examples: int = 0,
) -> PlantTrainingPlan:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    models = cfg.get("models", {})
    training = cfg.get("training", {})
    lora = training.get("lora", {})
    sft = training.get("sft", {})
    base_model = str(
        models.get("plant_vlm", {}).get("hf_id")
        or models.get("plant_vlm_finetuned", {}).get("base_model")
        or "openbmb/MiniCPM-V-4.6"
    )
    rank = int(lora.get("rank", 16))
    epochs = int(sft.get("epochs", 3))
    batch_size = int(sft.get("batch_size", 4))
    grad_accum = int(sft.get("grad_accum", 4))
    lr = str(sft.get("lr", "2.0e-4"))
    output = str(output_dir)
    dataset = str(dataset_path)

    return PlantTrainingPlan(
        execute_training=False,
        backend="SWIFT vision LoRA preferred; LLaMA-Factory plan documented as alternative",
        base_model=base_model,
        dataset_path=dataset,
        output_dir=output,
        adapter_repo=adapter_repo,
        corrected_examples=corrected_examples,
        minimum_recommended_examples=30,
        dependency_report=plant_training_dependency_report(),
        swift_command=[
            "swift",
            "sft",
            "--model",
            base_model,
            "--dataset",
            dataset,
            "--lora_rank",
            str(rank),
            "--num_train_epochs",
            str(epochs),
            "--per_device_train_batch_size",
            str(batch_size),
            "--gradient_accumulation_steps",
            str(grad_accum),
            "--learning_rate",
            lr,
            "--freeze_vit",
            "true",
            "--output_dir",
            output,
        ],
        llamafactory_command=[
            "llamafactory-cli",
            "train",
            "plant_lora.yaml",
        ],
        publish_commands=[
            [
                "huggingface-cli",
                "repo",
                "create",
                adapter_repo,
                "--type",
                "model",
            ],
            [
                "huggingface-cli",
                "upload",
                adapter_repo,
                output,
                ".",
            ],
        ],
        use_trained_model_command=[
            "python",
            "-m",
            "plant.app",
            "--model-mode",
            "finetuned",
            "--port",
            "7861",
        ],
        notes=[
            "This plan does not start training automatically.",
            "Use OpenBMB MiniCPM-V zero-shot first, then collect corrected plant examples.",
            "Train only after reviewing dataset quality and GPU memory.",
            "Set models.plant_vlm_finetuned.adapter_id to your real adapter repo after upload.",
        ],
    )


def plant_training_dependency_report() -> dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "datasets": importlib.util.find_spec("datasets") is not None,
        "peft": importlib.util.find_spec("peft") is not None,
        "trl": importlib.util.find_spec("trl") is not None,
        "swift": importlib.util.find_spec("swift") is not None,
    }


def write_llamafactory_dataset_info(
    dataset_path: str | Path = "data/plant_training.jsonl",
    output_path: str | Path = "data/llamafactory_dataset_info.json",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        (
            "{\n"
            '  "plant_discovery": {\n'
            f'    "file_name": "{Path(dataset_path).as_posix()}",\n'
            '    "formatting": "alpaca",\n'
            '    "columns": {\n'
            '      "prompt": "instruction",\n'
            '      "response": "response"\n'
            "    }\n"
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    return output
