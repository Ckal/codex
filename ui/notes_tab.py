from __future__ import annotations

from functools import partial

import gradio as gr

from core.app_state import APP_STATE
from core.events import Event, EventType
from datasets.field_notes import FieldNote, FieldNoteStore
from models.model_catalog import ModelInfo


def _export_notes(store: FieldNoteStore) -> str:
    path = store.export_jsonl()
    return f"Exported corrected notes to {path}"


def _export_dataset(store: FieldNoteStore) -> str:
    path = store.export_hf_dataset()
    return f"Exported local HF Dataset files to {path}"


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
    )

    export.click(partial(_export_notes, store), outputs=status)
    export_hf.click(partial(_export_dataset, store), outputs=status)
