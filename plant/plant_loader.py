from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlantRecord:
    image_path: str = ""
    latin_name: str = ""
    common_name: str = ""
    family: str = ""
    genus: str = ""
    source: str = ""
    split: str = "train"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_training_row(self) -> dict[str, Any]:
        response = {
            "common_name": self.common_name or self.latin_name,
            "latin_name": self.latin_name,
            "family": self.family,
            "genus": self.genus or genus_from_latin(self.latin_name),
            "confidence": 1.0,
            "key_features": self.extra.get("key_features", []),
            "care_tips": self.extra.get("care_tips", []),
            "toxicity": self.extra.get("toxicity", {"humans": "unknown", "pets": "unknown"}),
            "habitat": self.extra.get("habitat", ""),
            "bloom_season": self.extra.get("bloom_season", ""),
            "similar_species": self.extra.get("similar_species", []),
            "notes": self.extra.get("notes", ""),
        }
        return {
            "image_path": self.image_path,
            "instruction": "Identify this plant species from the image.",
            "response": json.dumps(response, ensure_ascii=False),
            "latin_name": self.latin_name,
            "family": self.family,
            "source": self.source,
            "split": self.split,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalFolderLoader:
    """Load image metadata from folders named after species."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def iter_records(
        self,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> list[PlantRecord]:
        if not self.root.exists():
            return []

        records: list[PlantRecord] = []
        for species_dir in sorted(self.root.iterdir()):
            if not species_dir.is_dir():
                continue
            latin_name = species_dir.name.replace("_", " ")
            for image_path in sorted(species_dir.rglob("*")):
                if image_path.suffix.lower() not in extensions:
                    continue
                records.append(
                    PlantRecord(
                        image_path=str(image_path),
                        latin_name=latin_name,
                        common_name=latin_name,
                        genus=genus_from_latin(latin_name),
                        source="local_folder",
                    )
                )
        return records

    def species_list(self) -> list[str]:
        if not self.root.exists():
            return []
        return [
            path.name.replace("_", " ")
            for path in sorted(self.root.iterdir())
            if path.is_dir()
        ]


class GBIFLoader:
    """Load a small Darwin Core TSV export for metadata enrichment."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def load_metadata(self) -> dict[str, dict[str, Any]]:
        if not self.csv_path.exists():
            return {}

        metadata: dict[str, dict[str, Any]] = {}
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                species = row.get("species", "").strip()
                if not species:
                    continue
                entry = metadata.setdefault(
                    species,
                    {
                        "family": row.get("family", ""),
                        "genus": row.get("genus", ""),
                        "countries": set(),
                    },
                )
                country = row.get("country", "").strip()
                if country:
                    entry["countries"].add(country)

        for value in metadata.values():
            value["countries"] = sorted(value["countries"])
        return metadata


class SpeciesIndexBuilder:
    """Build an in-memory plant species index without network by default."""

    def __init__(self, root: str | Path = "plant") -> None:
        self.root = Path(root)

    def build(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        index = self._from_local_folder(config)
        self._enrich_from_cached_labels(index)
        self._enrich_from_gbif(index)
        return index or demo_species()

    def _from_local_folder(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        dataset_cfg = config.get("datasets", {}).get("local_field_guide", {})
        configured_path = Path(str(dataset_cfg.get("path", "data/field_guide")))
        local_path = (
            configured_path
            if configured_path.is_absolute()
            else self.root / configured_path
        )
        loader = LocalFolderLoader(local_path)
        index: dict[str, dict[str, Any]] = {}
        for species in loader.species_list():
            index[species] = {
                "common_name": species,
                "family": "",
                "genus": genus_from_latin(species),
                "n_images": 0,
                "source": "local_folder",
            }
        return index

    def _enrich_from_cached_labels(self, index: dict[str, dict[str, Any]]) -> None:
        labels_path = self.root / "data" / "plantnet_labels.json"
        if not labels_path.exists():
            return
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            return
        for latin_name, meta in labels.items():
            if not isinstance(meta, dict):
                continue
            entry = index.setdefault(
                latin_name,
                {
                    "common_name": latin_name,
                    "family": "",
                    "genus": genus_from_latin(latin_name),
                    "n_images": 0,
                    "source": "plantnet_cache",
                },
            )
            for key, value in meta.items():
                if value and not entry.get(key):
                    entry[key] = value

    def _enrich_from_gbif(self, index: dict[str, dict[str, Any]]) -> None:
        metadata = GBIFLoader(self.root / "data" / "gbif_occurrences.tsv").load_metadata()
        for latin_name, values in metadata.items():
            if latin_name not in index:
                continue
            index[latin_name].setdefault("habitat_countries", values.get("countries", []))
            if values.get("family") and not index[latin_name].get("family"):
                index[latin_name]["family"] = values["family"]


class FieldNotesPlantExporter:
    """Read corrected field notes and export plant training rows."""

    def __init__(self, csv_path: str | Path = "data/plant_field_notes.csv") -> None:
        self.csv_path = Path(csv_path)

    def load_corrections(self) -> list[PlantRecord]:
        if not self.csv_path.exists():
            return []

        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        records: list[PlantRecord] = []
        for row in rows:
            correction = row.get("correction", "").strip()
            if not correction:
                continue
            records.append(
                PlantRecord(
                    image_path=row.get("image_path", ""),
                    latin_name=correction,
                    common_name=correction,
                    genus=genus_from_latin(correction),
                    source="field_notes",
                    extra={"original_prediction": row.get("response", "")},
                )
            )
        return records

    def export_jsonl(self, output_path: str | Path = "data/plant_training.jsonl") -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in self.load_corrections():
                handle.write(json.dumps(record.to_training_row(), ensure_ascii=False) + "\n")
        return output


class SQLitePlantNoteReader:
    """Compatibility reader for old plant note SQLite files."""

    def __init__(self, db_path: str | Path = "data/field_notes.db") -> None:
        self.db_path = Path(db_path)

    def count_rows(self) -> int:
        if not self.db_path.exists():
            return 0
        connection = sqlite3.connect(self.db_path)
        try:
            row = connection.execute("SELECT COUNT(*) FROM notes").fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()


def demo_species() -> dict[str, dict[str, Any]]:
    return {
        "Acer palmatum": {
            "common_name": "Japanese Maple",
            "family": "Sapindaceae",
            "genus": "Acer",
            "source": "demo",
        },
        "Bellis perennis": {
            "common_name": "Common Daisy",
            "family": "Asteraceae",
            "genus": "Bellis",
            "source": "demo",
        },
        "Rosa canina": {
            "common_name": "Dog Rose",
            "family": "Rosaceae",
            "genus": "Rosa",
            "source": "demo",
        },
        "Quercus robur": {
            "common_name": "English Oak",
            "family": "Fagaceae",
            "genus": "Quercus",
            "source": "demo",
        },
        "Urtica dioica": {
            "common_name": "Stinging Nettle",
            "family": "Urticaceae",
            "genus": "Urtica",
            "source": "demo",
        },
    }


def genus_from_latin(latin_name: str) -> str:
    return latin_name.split()[0] if " " in latin_name else latin_name
