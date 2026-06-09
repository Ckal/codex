from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.field_notes import FieldNote, FieldNoteStore
from plant.plant_loader import FieldNotesPlantExporter
from plant.training import build_plant_training_plan

_plant_service: Any | None = None
_note_store: FieldNoteStore | None = None
_species_index: dict[str, dict[str, Any]] = {}


def set_services(
    plant_service: Any | None,
    note_store: FieldNoteStore | None,
    species_index: dict[str, dict[str, Any]] | None,
) -> None:
    global _plant_service, _note_store, _species_index
    _plant_service = plant_service
    _note_store = note_store
    _species_index = species_index or {}


def identify_plant(image_path: str, detail_level: str = "standard") -> dict[str, Any]:
    if _plant_service is None:
        return {"status": "error", "message": "Plant service is not initialised."}
    path = Path(image_path)
    if not path.exists():
        return {"status": "error", "message": f"Image not found: {image_path}"}
    result = _plant_service.identify(path, force_thinking=(detail_level == "thinking"))
    return dict(result.to_dict())


def search_species(
    query: str,
    family_filter: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    query_value = query.lower().strip()
    family_value = (family_filter or "").lower().strip()
    results: list[dict[str, Any]] = []
    for latin_name, meta in sorted(_species_index.items()):
        common = str(meta.get("common_name") or "")
        family = str(meta.get("family") or "")
        haystack = " ".join([latin_name, common, family]).lower()
        if query_value and query_value not in haystack:
            continue
        if family_value and family_value != family.lower():
            continue
        results.append(
            {
                "latin_name": latin_name,
                "common_name": common,
                "family": family,
                "genus": meta.get("genus", ""),
                "source": meta.get("source", ""),
            }
        )
        if len(results) >= limit:
            break
    return results


def save_field_correction(
    predicted_latin: str,
    correct_latin: str,
    image_path: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if _note_store is None:
        return {"status": "error", "message": "Field note store is not initialised."}
    if not correct_latin.strip():
        return {"status": "error", "message": "A correction is required."}

    note = FieldNote.create(
        model_id="plant_vlm",
        prompt="Identify this plant species from the image.",
        response=predicted_latin,
        correction=correct_latin.strip(),
        tags="plant-discovery,mcp-correction",
        image_path=image_path,
        use_for_training=True,
    )
    _note_store.save(note)
    return {
        "status": "saved",
        "message": f"Correction saved: {predicted_latin} -> {correct_latin}",
        "notes": notes,
    }


def export_corrections(output_path: str = "data/plant_training.jsonl") -> dict[str, Any]:
    if _note_store is None:
        return {"status": "error", "message": "Field note store is not initialised."}
    exporter = FieldNotesPlantExporter(_note_store.path)
    path = exporter.export_jsonl(output_path)
    rows = exporter.load_corrections()
    return {
        "status": "exported",
        "output_path": str(path),
        "rows": len(rows),
        "execute_training": False,
    }


def dataset_stats() -> dict[str, Any]:
    notes = _note_store.list_notes() if _note_store else []
    corrections = [note for note in notes if note.correction.strip()]
    return {
        "species_index_size": len(_species_index),
        "field_notes": len(notes),
        "corrections": len(corrections),
    }


def training_plan(
    dataset_path: str = "data/plant_training.jsonl",
    lora_rank: int = 16,
    epochs: int = 3,
) -> dict[str, Any]:
    plan = build_plant_training_plan(
        dataset_path=dataset_path,
        corrected_examples=dataset_stats().get("corrections", 0),
    ).to_dict()
    plan["requested_lora_rank"] = lora_rank
    plan["requested_epochs"] = epochs
    return plan


def build_mcp_server():
    """Build a FastMCP server only when the optional mcp package is installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install mcp[cli] to expose Plant Discovery tools over MCP.") from exc

    mcp = FastMCP(
        name="PlantDiscovery",
        description="Plant identification and correction-loop tools.",
    )
    mcp.tool()(identify_plant)
    mcp.tool()(search_species)
    mcp.tool()(save_field_correction)
    mcp.tool()(export_corrections)
    mcp.tool()(dataset_stats)
    mcp.tool()(training_plan)
    return mcp


if __name__ == "__main__":
    server = build_mcp_server()
    server.run(transport="sse", port=8081)
