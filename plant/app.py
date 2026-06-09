from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import gradio as gr
import yaml

from datasets.field_notes import FieldNoteStore
from plant.plant_loader import SpeciesIndexBuilder
from plant.plant_service import DemoPlantVisionService, PlantVisionService
from plant.plant_tab import build_plant_tab
from plant.plant_tools import set_services

ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "models.yaml"

APP_CSS = """
.plant-shell {
  max-width: 1120px !important;
}
.plant-title {
  margin-bottom: 0.35rem;
}
footer {
  display: none !important;
}
"""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Plant config must be a mapping: {config_path}")
    return data


def build_app(
    config_path: str | Path = DEFAULT_CONFIG,
    no_model: bool = False,
    data_dir: str | Path = "data",
) -> gr.Blocks:
    cfg = load_config(config_path)
    root = Path(config_path).parent
    species_index = SpeciesIndexBuilder(root=root).build(cfg)
    note_store = FieldNoteStore(Path(data_dir) / "plant_field_notes.csv")

    plant_service: Any
    if no_model:
        plant_service = DemoPlantVisionService()
    else:
        plant_service = PlantVisionService.from_config(config_path)

    set_services(plant_service, note_store, species_index)

    domain = cfg.get("domain", {})
    title = str(domain.get("title") or "Plant Discovery")
    description = str(
        domain.get("description")
        or "Identify plants, correct mistakes, and export local training data."
    )

    with gr.Blocks(
        title=title,
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            f"# {title}\n\n{description}",
            elem_classes=["plant-title"],
        )
        gr.Markdown(
            "Local-first reference app generated around the OpenBMB Workbench template. "
            "Model loading happens only after an explicit identify action."
        )
        build_plant_tab(plant_service, note_store, species_index)

    return cast(gr.Blocks, demo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant Discovery reference app")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Use deterministic demo identification instead of loading a vision model.",
    )
    args = parser.parse_args()

    demo = build_app(args.config, no_model=args.no_model)
    print(f"Starting Plant Discovery on http://127.0.0.1:{args.port}")
    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
        css=APP_CSS,
        mcp_server=True,
    )


if __name__ == "__main__":
    main()
