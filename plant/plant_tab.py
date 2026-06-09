from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr

from datasets.field_notes import FieldNote, FieldNoteStore
from plant.plant_loader import FieldNotesPlantExporter
from plant.plant_service import PlantID


def build_plant_tab(
    plant_service: Any,
    note_store: FieldNoteStore,
    species_index: dict[str, dict[str, Any]],
) -> None:
    with gr.Tabs():
        with gr.Tab("Identify"):
            build_identify_panel(plant_service, note_store)
        with gr.Tab("Field Guide"):
            build_field_guide_panel(species_index)
        with gr.Tab("Corrections"):
            build_corrections_panel(note_store)
        with gr.Tab("Stats"):
            build_stats_panel(note_store, species_index)


def build_identify_panel(plant_service: Any, note_store: FieldNoteStore) -> None:
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Primary plant image", height=280)
            extra_images = gr.Gallery(
                label="Extra angles",
                type="pil",
                columns=2,
                height=180,
            )
            mode = gr.Dropdown(
                ["standard", "thinking"],
                value="standard",
                label="Analysis mode",
            )
            identify = gr.Button("Identify plant", variant="primary")
        with gr.Column(scale=1):
            result_card = gr.HTML(label="Result")
            result_json = gr.JSON(label="Structured result")
            correction = gr.Textbox(label="Human correction", placeholder="Genus species")
            notes = gr.Textbox(label="Correction notes", lines=2)
            save = gr.Button("Save correction / field note")
            save_status = gr.Markdown()

    last_result = gr.State({})

    identify.click(
        lambda img, gallery, selected_mode: identify_plant_callback(
            plant_service,
            img,
            gallery,
            selected_mode,
        ),
        [image, extra_images, mode],
        [result_card, result_json, last_result],
    )
    save.click(
        lambda result, corrected, note_text: save_field_note_callback(
            note_store,
            result,
            corrected,
            note_text,
        ),
        [last_result, correction, notes],
        save_status,
    )


def build_field_guide_panel(species_index: dict[str, dict[str, Any]]) -> None:
    query = gr.Textbox(label="Search", placeholder="Common name, Latin name, family")
    family = gr.Dropdown(
        choices=available_families(species_index),
        label="Family",
        allow_custom_value=True,
    )
    search = gr.Button("Search species", variant="primary")
    table = gr.Dataframe(
        headers=["Latin name", "Common name", "Family", "Genus", "Source"],
        interactive=False,
        wrap=True,
    )
    detail = gr.HTML(label="Species detail")

    search.click(lambda text, fam: species_table(species_index, text, fam), [query, family], table)

    def on_species_select(rows: list[list[str]], event: gr.SelectData) -> str:
        return species_detail(species_index, event, rows)

    table.select(on_species_select, table, detail)


def build_corrections_panel(note_store: FieldNoteStore) -> None:
    gr.Markdown("Corrections are saved locally and can be exported for later LoRA training.")
    refresh = gr.Button("Load corrections")
    export = gr.Button("Export training JSONL", variant="primary")
    training = gr.Button("Prepare training plan")
    rows = gr.Dataframe(
        headers=["created_at", "prediction", "correction", "tags", "image_path"],
        interactive=False,
        wrap=True,
    )
    status = gr.JSON(label="Export or training plan")

    refresh.click(lambda: correction_rows(note_store), outputs=rows)
    export.click(lambda: export_corrections(note_store), outputs=status)
    training.click(lambda: plant_training_plan(note_store), outputs=status)


def build_stats_panel(
    note_store: FieldNoteStore,
    species_index: dict[str, dict[str, Any]],
) -> None:
    refresh = gr.Button("Refresh stats")
    stats = gr.JSON(label="Plant app stats")
    refresh.click(lambda: plant_stats(note_store, species_index), outputs=stats)


def identify_plant_callback(
    plant_service: Any,
    image: Any,
    gallery: list[Any] | None,
    mode: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if image is None:
        empty = {
            "common_name": "No image",
            "latin_name": "Upload a plant image",
            "confidence": 0.0,
            "notes": "Add a photo before identifying.",
        }
        return render_result_card(empty), empty, empty

    extras = normalize_gallery_images(gallery)
    result = plant_service.identify(
        image,
        extra_images=extras,
        force_thinking=(mode == "thinking"),
    )
    result_data = result.to_dict() if isinstance(result, PlantID) else dict(result)
    return render_result_card(result_data), result_data, result_data


def save_field_note_callback(
    note_store: FieldNoteStore,
    result: dict[str, Any],
    correction: str,
    notes: str,
) -> str:
    if not result:
        return "Run an identification before saving a field note."
    predicted = str(result.get("latin_name") or "Unknown")
    note = FieldNote.create(
        model_id=str(result.get("model_used") or "plant_vlm"),
        prompt="Identify this plant species from the image.",
        response=predicted,
        correction=correction.strip(),
        tags="plant-discovery,correction" if correction.strip() else "plant-discovery",
        image_path=str(result.get("image_path") or ""),
        use_for_training=bool(correction.strip()),
    )
    note_store.save(note)
    suffix = f" Correction: {correction.strip()}." if correction.strip() else ""
    note_hint = f" Notes: {notes.strip()}." if notes.strip() else ""
    return f"Saved field note for {predicted}.{suffix}{note_hint}"


def render_result_card(result: dict[str, Any]) -> str:
    confidence = _confidence(result.get("confidence"))
    confidence_percent = int(confidence * 100)
    color = "#0f766e" if confidence >= 0.80 else "#a16207" if confidence >= 0.55 else "#b91c1c"
    features = "".join(
        f"<li>{html.escape(str(feature))}</li>"
        for feature in result.get("key_features", [])[:5]
    )
    care = "".join(
        f"<li>{html.escape(str(tip))}</li>"
        for tip in result.get("care_tips", [])[:4]
    )
    common_name = html.escape(str(result.get("common_name", "Unknown")))
    latin_name = html.escape(str(result.get("latin_name", "")))
    family = html.escape(str(result.get("family", "")))
    genus = html.escape(str(result.get("genus", "")))
    notes = html.escape(str(result.get("notes", "")))
    return f"""
<section style="border:1px solid #d8ded7;border-radius:8px;padding:16px;background:#fff;">
  <div style="display:flex;justify-content:space-between;gap:16px;">
    <div>
      <h2 style="margin:0;font-size:22px;">{common_name}</h2>
      <p style="margin:4px 0;color:#475569;font-style:italic;">
        {latin_name}
      </p>
      <p style="margin:0;color:#64748b;">
        {family} / {genus}
      </p>
    </div>
    <div style="text-align:right;color:{color};font-weight:700;font-size:26px;">
      {confidence_percent}%
    </div>
  </div>
  <div style="margin:12px 0;background:#e5e7eb;border-radius:4px;height:6px;">
    <div style="width:{confidence_percent}%;background:{color};border-radius:4px;height:6px;"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div><strong>Key features</strong><ul>{features}</ul></div>
    <div><strong>Care tips</strong><ul>{care}</ul></div>
  </div>
  <p style="color:#475569;">{notes}</p>
</section>
"""


def normalize_gallery_images(gallery: list[Any] | None) -> list[Any]:
    images: list[Any] = []
    for item in gallery or []:
        if isinstance(item, (list, tuple)) and item:
            images.append(item[0])
        elif item is not None:
            images.append(item)
    return images[:3]


def available_families(species_index: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        {str(meta.get("family")) for meta in species_index.values() if meta.get("family")}
    )


def species_table(
    species_index: dict[str, dict[str, Any]],
    query: str,
    family: str | None,
    limit: int = 100,
) -> list[list[str]]:
    query_value = query.lower().strip()
    family_value = (family or "").lower().strip()
    rows: list[list[str]] = []
    for latin_name, meta in sorted(species_index.items()):
        common = str(meta.get("common_name") or "")
        family_name = str(meta.get("family") or "")
        haystack = " ".join([latin_name, common, family_name]).lower()
        if query_value and query_value not in haystack:
            continue
        if family_value and family_value != family_name.lower():
            continue
        rows.append(
            [
                latin_name,
                common,
                family_name,
                str(meta.get("genus") or ""),
                str(meta.get("source") or ""),
            ]
        )
        if len(rows) >= limit:
            break
    return rows


def species_detail(
    species_index: dict[str, dict[str, Any]],
    event: gr.SelectData,
    rows: list[list[str]] | None,
) -> str:
    if rows is None or event.index is None:
        return ""
    row_index = event.index[0] if isinstance(event.index, tuple) else int(event.index)
    if row_index >= len(rows):
        return ""
    latin_name = rows[row_index][0]
    meta = species_index.get(latin_name, {})
    countries = ", ".join(meta.get("habitat_countries", [])[:8]) or "unknown"
    return (
        "<section style='border:1px solid #d8ded7;border-radius:8px;padding:14px;'>"
        f"<strong>{html.escape(str(meta.get('common_name') or latin_name))}</strong>"
        f"<p><em>{html.escape(latin_name)}</em></p>"
        f"<p>Family: {html.escape(str(meta.get('family') or 'unknown'))}</p>"
        f"<p>Known regions: {html.escape(countries)}</p>"
        "</section>"
    )


def correction_rows(note_store: FieldNoteStore) -> list[list[str]]:
    return [
        [
            note.created_at,
            note.response,
            note.correction or "-",
            note.tags,
            note.image_path,
        ]
        for note in note_store.list_notes()
    ]


def export_corrections(note_store: FieldNoteStore) -> dict[str, Any]:
    exporter = FieldNotesPlantExporter(note_store.path)
    output = exporter.export_jsonl(Path("data") / "plant_training.jsonl")
    rows = exporter.load_corrections()
    return {
        "output_path": str(output),
        "rows": len(rows),
        "execute_training": False,
        "message": "Exported corrected plant notes. Training remains a separate explicit step.",
    }


def plant_training_plan(note_store: FieldNoteStore) -> dict[str, Any]:
    corrected = len(note_store.list_notes(corrected_only=True, training_only=True))
    return {
        "execute_training": False,
        "corrected_examples": corrected,
        "minimum_recommended_examples": 30,
        "backend": "SWIFT or LLaMA-Factory vision LoRA",
        "command_preview": [
            "swift",
            "sft",
            "--model",
            "openbmb/MiniCPM-V-4.6",
            "--dataset",
            "data/plant_training.jsonl",
            "--freeze_vit",
            "true",
            "--output_dir",
            "checkpoints/plant_lora",
        ],
        "notes": [
            "Do not start training from the public UI.",
            "Collect enough corrections first.",
            "Run training locally after checking GPU memory and dataset quality.",
        ],
    }


def plant_stats(
    note_store: FieldNoteStore,
    species_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    notes = note_store.list_notes()
    corrections = [note for note in notes if note.correction.strip()]
    return {
        "species_index_size": len(species_index),
        "field_notes": len(notes),
        "corrections": len(corrections),
        "training_ready": len([note for note in corrections if note.use_for_training]),
    }


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(parsed, 0.0), 1.0)
