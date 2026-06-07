from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets.field_notes import FieldNote, FieldNoteStore


@dataclass(frozen=True)
class OCRPrediction:
    """One local OCR prediction that may need human correction."""

    source_path: str
    text: str
    confidence: float
    page: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> OCRPrediction:
        source_path = _first_present(row, ["source_path", "image_path", "file_path", "path"])
        text = _first_present(row, ["text", "prediction", "ocr_text", "response"])
        confidence_raw = _first_present(row, ["confidence", "score", "probability"], "0")
        return cls(
            source_path=source_path,
            text=text,
            confidence=_parse_confidence(confidence_raw),
            page=str(row.get("page", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_ocr_predictions(path: str | Path) -> list[OCRPrediction]:
    """Load OCR predictions from local CSV, JSONL, or NDJSON files."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"OCR prediction file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as f:
            return [OCRPrediction.from_row(row) for row in csv.DictReader(f)]
    if suffix in {".jsonl", ".ndjson"}:
        return [
            OCRPrediction.from_row(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError("OCR predictions must be a .csv, .jsonl, or .ndjson file.")


def uncertain_predictions(
    predictions: list[OCRPrediction],
    confidence_threshold: float = 0.8,
) -> list[OCRPrediction]:
    return [
        prediction
        for prediction in predictions
        if prediction.confidence <= confidence_threshold or not prediction.text.strip()
    ]


def import_uncertain_predictions(
    store: FieldNoteStore,
    predictions: list[OCRPrediction],
    model_id: str,
    confidence_threshold: float = 0.8,
    tags: str = "ocr,uncertain",
) -> int:
    imported = 0
    for prediction in uncertain_predictions(predictions, confidence_threshold):
        page_note = f" page {prediction.page}" if prediction.page else ""
        note = FieldNote.create(
            model_id=model_id,
            prompt=f"Review OCR text for {prediction.source_path}{page_note}.",
            response=prediction.text,
            correction="",
            tags=tags,
            image_path=prediction.source_path,
            use_for_training=False,
        )
        store.save(note)
        imported += 1
    return imported


def export_corrected_ocr_notes(
    store: FieldNoteStore,
    output_path: str | Path = "data/ocr_corrections.jsonl",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    notes = store.list_notes(corrected_only=True, tag="ocr")
    with output.open("w", encoding="utf-8") as f:
        for note in notes:
            row = {
                "source_path": note.image_path,
                "predicted_text": note.response,
                "corrected_text": note.correction,
                "model_id": note.model_id,
                "created_at": note.created_at,
                "tags": note.tags,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def ocr_import_summary(path: str | Path, confidence_threshold: float = 0.8) -> dict[str, object]:
    predictions = load_ocr_predictions(path)
    uncertain = uncertain_predictions(predictions, confidence_threshold)
    return {
        "source": str(path),
        "rows": len(predictions),
        "uncertain_rows": len(uncertain),
        "confidence_threshold": confidence_threshold,
        "sample": [prediction.to_dict() for prediction in uncertain[:5]],
    }


def _first_present(row: dict[str, Any], names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return str(value)
    return default


def _parse_confidence(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0
