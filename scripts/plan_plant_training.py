from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from plant.training import build_plant_training_plan, write_llamafactory_dataset_info

    parser = argparse.ArgumentParser(description="Create a Plant Discovery training plan.")
    parser.add_argument("--config", default="plant/models.yaml")
    parser.add_argument("--dataset", default="data/plant_training.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/plant_lora")
    parser.add_argument("--adapter-repo", default="your-username/minicpm-v46-plant-lora")
    parser.add_argument("--corrected-examples", type=int, default=0)
    parser.add_argument(
        "--write-llamafactory-info",
        action="store_true",
        help="Write a LLaMA-Factory dataset_info JSON preview.",
    )
    args = parser.parse_args()

    plan = build_plant_training_plan(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        adapter_repo=args.adapter_repo,
        corrected_examples=args.corrected_examples,
    )
    payload = plan.to_dict()
    if args.write_llamafactory_info:
        payload["llamafactory_dataset_info"] = str(
            write_llamafactory_dataset_info(args.dataset)
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
