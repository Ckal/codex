from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetPreview:
    """Small local dataset preview result."""

    source: str
    rows: int
    columns: list[str]
    samples: list[dict[str, Any]]

    def as_table(self) -> list[list[str]]:
        table = [
            ["source", self.source],
            ["rows", str(self.rows)],
            ["columns", ", ".join(self.columns)],
        ]
        for index, sample in enumerate(self.samples, start=1):
            table.append([f"sample_{index}", json.dumps(sample, ensure_ascii=False)])
        return table


@dataclass(frozen=True)
class DatasetStats:
    """Basic dataset statistics for local preview and tools."""

    source: str
    rows: int
    columns: int
    column_names: list[str]
    non_empty_by_column: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows": self.rows,
            "columns": self.columns,
            "column_names": self.column_names,
            "non_empty_by_column": self.non_empty_by_column,
        }


@dataclass(frozen=True)
class HuggingFaceDatasetPreview:
    """Small preview result for optional Hugging Face datasets integration."""

    dataset_id: str
    split: str
    rows: int
    columns: list[str]
    samples: list[dict[str, Any]]
    status: str = "loaded"

    def as_table(self) -> list[list[str]]:
        table = [
            ["dataset_id", self.dataset_id],
            ["split", self.split],
            ["rows", str(self.rows)],
            ["columns", ", ".join(self.columns)],
            ["status", self.status],
        ]
        for index, sample in enumerate(self.samples, start=1):
            table.append([f"sample_{index}", json.dumps(sample, ensure_ascii=False)])
        return table


def preview_local_dataset(path: str | Path, limit: int = 5) -> DatasetPreview:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".csv":
        return _preview_csv(dataset_path, limit)
    if dataset_path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _preview_jsonl(dataset_path, limit)
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def dataset_statistics(path: str | Path) -> DatasetStats:
    preview = preview_local_dataset(path, limit=0)
    non_empty_by_column = {column: 0 for column in preview.columns}

    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".csv":
        with dataset_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for column in preview.columns:
                    if str(row.get(column, "")).strip():
                        non_empty_by_column[column] += 1
    else:
        with dataset_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for column in preview.columns:
                    if str(row.get(column, "")).strip():
                        non_empty_by_column[column] += 1

    return DatasetStats(
        source=preview.source,
        rows=preview.rows,
        columns=len(preview.columns),
        column_names=preview.columns,
        non_empty_by_column=non_empty_by_column,
    )


def preview_huggingface_dataset(
    dataset_id: str,
    split: str = "train",
    limit: int = 5,
) -> HuggingFaceDatasetPreview:
    datasets_module = import_module("datasets")
    load_dataset = getattr(datasets_module, "load_dataset", None)
    if load_dataset is None:
        return HuggingFaceDatasetPreview(
            dataset_id=dataset_id,
            split=split,
            rows=0,
            columns=[],
            samples=[],
            status="Hugging Face package 'datasets' is not installed.",
        )

    dataset = load_dataset(dataset_id, split=split)
    rows = len(dataset)
    columns = list(dataset.column_names)
    samples = [dict(dataset[index]) for index in range(min(limit, rows))]
    return HuggingFaceDatasetPreview(dataset_id, split, rows, columns, samples)


def _preview_csv(path: Path, limit: int) -> DatasetPreview:
    samples: list[dict[str, Any]] = []
    rows = 0
    columns: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            if len(samples) < limit:
                samples.append(dict(row))

    return DatasetPreview(str(path), rows, columns, samples)


def _preview_jsonl(path: Path, limit: int) -> DatasetPreview:
    samples: list[dict[str, Any]] = []
    rows = 0
    columns: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("JSONL rows must be objects")
            columns.update(item)
            if len(samples) < limit:
                samples.append(item)

    return DatasetPreview(str(path), rows, sorted(columns), samples)
