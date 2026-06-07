from __future__ import annotations

from functools import partial

import gradio as gr

from core.app_state import APP_STATE
from core.events import Event, EventType
from datasets.field_notes import FieldNote, FieldNoteStore
from datasets.ocr import (
    export_corrected_ocr_notes,
    import_uncertain_predictions,
    load_ocr_predictions,
    ocr_import_summary,
)
from models.model_catalog import ModelInfo
from ui.progress import CLICK_PROGRESS


def _export_notes(store: FieldNoteStore) -> str:
    path = store.export_jsonl()
    return f"Exported corrected notes to {path}"


def _export_dataset(store: FieldNoteStore) -> str:
    path = store.export_hf_dataset()
    return f"Exported local HF Dataset files to {path}"


def _export_ocr_notes(store: FieldNoteStore) -> str:
    path = export_corrected_ocr_notes(store)
    return f"Exported corrected OCR notes to {path}"


def build_notes_tab(catalog: dict[str, ModelInfo]) -> None:
    store = FieldNoteStore()
    model_id = gr.Dropdown(list(catalog), value=next(iter(catalog)), label="Model")
    prompt = gr.Textbox(label="Prompt", lines=3)
    response = gr.Textbox(label="Model response", lines=4)
    correction = gr.Textbox(label="Human correction", lines=4)
    tags = gr.Textbox(label="Tags", placeholder="ocr, plant-id, demo")
    image_path = gr.Textbox(label="Image path", placeholder="Optional local image path")
    video_path = gr.Textbox(label="Video path", placeholder="Optional local video path")
    use_for_training = gr.Checkbox(label="Use for training", value=True)
    save = gr.Button("Save field note", variant="primary")
    export = gr.Button("Export corrected JSONL")
    export_hf = gr.Button("Export local HF Dataset")
    status = gr.Textbox(label="Status", interactive=False)

    def save_note(
        selected: str,
        prompt_text: str,
        response_text: str,
        correction_text: str,
        tag_text: str,
        image: str,
        video: str,
        training: bool,
    ) -> str:
        note = FieldNote.create(
            model_id=selected,
            prompt=prompt_text,
            response=response_text,
            correction=correction_text,
            tags=tag_text,
            image_path=image,
            video_path=video,
            use_for_training=training,
        )
        path = store.save(note)
        APP_STATE.emit(
            Event(
                EventType.FIELD_NOTE_SAVED,
                {
                    "model_id": selected,
                    "path": str(path),
                    "has_correction": bool(correction_text.strip()),
                    "tags": tag_text,
                    "use_for_training": training,
                },
            )
        )
        return f"Saved to {path}"

    save.click(
        save_note,
        [model_id, prompt, response, correction, tags, image_path, video_path, use_for_training],
        status,
        show_progress=CLICK_PROGRESS,
    )

    export.click(
        partial(_export_notes, store),
        outputs=status,
        show_progress=CLICK_PROGRESS,
    )
    export_hf.click(
        partial(_export_dataset, store),
        outputs=status,
        show_progress=CLICK_PROGRESS,
    )

    build_ocr_import_panel(store, model_id, status)


def build_ocr_import_panel(
    store: FieldNoteStore,
    model_id: gr.Dropdown,
    status: gr.Textbox,
) -> None:
    gr.Markdown("### OCR correction import")
    ocr_path = gr.Textbox(
        label="OCR predictions file",
        placeholder="Local .csv, .jsonl, or .ndjson with source_path,text,confidence",
    )
    ocr_threshold = gr.Slider(
        label="Uncertain confidence threshold",
        minimum=0,
        maximum=1,
        value=0.8,
        step=0.01,
    )
    with gr.Row():
        preview_ocr = gr.Button("Preview uncertain OCR")
        import_ocr = gr.Button("Import uncertain OCR", variant="primary")
        export_ocr = gr.Button("Export corrected OCR JSONL")
    ocr_preview = gr.JSON(label="OCR import preview")

    preview_ocr.click(
        preview_ocr_predictions,
        [ocr_path, ocr_threshold],
        ocr_preview,
        show_progress=CLICK_PROGRESS,
    )
    import_ocr.click(
        partial(import_ocr_predictions, store),
        [model_id, ocr_path, ocr_threshold],
        status,
        show_progress=CLICK_PROGRESS,
    )
    export_ocr.click(
        partial(_export_ocr_notes, store),
        outputs=status,
        show_progress=CLICK_PROGRESS,
    )


def preview_ocr_predictions(path: str, threshold: float) -> dict:
    if not path.strip():
        return {"error": "Enter a local OCR prediction file path."}
    try:
        return ocr_import_summary(path, threshold)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {"error": str(exc)}


def import_ocr_predictions(
    store: FieldNoteStore,
    selected: str,
    path: str,
    threshold: float,
) -> str:
    if not path.strip():
        return "Enter a local OCR prediction file path."
    try:
        predictions = load_ocr_predictions(path)
        imported = import_uncertain_predictions(
            store,
            predictions,
            selected,
            confidence_threshold=threshold,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        return str(exc)
    APP_STATE.emit(
        Event(
            EventType.FIELD_NOTE_SAVED,
            {
                "model_id": selected,
                "path": str(store.path),
                "source": path,
                "imported": imported,
                "tags": "ocr,uncertain",
                "use_for_training": False,
            },
        )
    )
    return f"Imported {imported} uncertain OCR predictions to {store.path}"
