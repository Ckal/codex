from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from core.file_exports import copy_text_file_or_empty


@dataclass(frozen=True)
class TrackingConfig:
    """Tracking settings loaded from config/training.yaml."""

    enabled: bool = False
    project: str = "openbmb-local-ai-workbench"
    local_path: str = "data/traces.jsonl"


@dataclass(frozen=True)
class TrackingStatus:
    """Availability and mode of the tracking client."""

    enabled: bool
    trackio_available: bool
    project: str
    local_path: str
    mode: str


def load_tracking_config(path: str | Path = "config/training.yaml") -> TrackingConfig:
    config_path = Path(path)
    if not config_path.exists():
        return TrackingConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    trackio = raw.get("trackio", {})
    return TrackingConfig(
        enabled=bool(trackio.get("enabled", False)),
        project=str(trackio.get("project", TrackingConfig.project)),
        local_path=str(trackio.get("local_path", TrackingConfig.local_path)),
    )


class TrackingClient:
    """Optional Trackio wrapper with JSONL fallback logging."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or load_tracking_config()
        self._trackio_run: Any | None = None
        self._trackio_module: Any | None = None

    def status(self) -> TrackingStatus:
        trackio_available = importlib.util.find_spec("trackio") is not None
        mode = "trackio" if self.config.enabled and trackio_available else "local_jsonl"
        return TrackingStatus(
            enabled=self.config.enabled,
            trackio_available=trackio_available,
            project=self.config.project,
            local_path=self.config.local_path,
            mode=mode,
        )

    def init(self, run_name: str = "local-run") -> TrackingStatus:
        status = self.status()
        if status.mode == "trackio":
            self._trackio_module = import_module("trackio")
            self._trackio_run = self._trackio_module.init(
                project=self.config.project,
                name=run_name,
            )
        return status

    def log(self, event_name: str, payload: dict[str, Any]) -> Path:
        record = {"event": event_name, "payload": payload}
        output = Path(self.config.local_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self._trackio_module is not None:
            self._trackio_module.log({event_name: payload})

        return output

    def finish(self) -> None:
        if self._trackio_module is not None:
            finish_func = getattr(self._trackio_module, "finish", None)
            if finish_func is not None:
                finish_func()
        self._trackio_run = None
        self._trackio_module = None


def export_traces(
    source_path: str | Path = "data/traces.jsonl",
    output_path: str | Path = "exports/traces.jsonl",
) -> Path:
    return copy_text_file_or_empty(source_path, output_path)


def read_trace_rows(path: str | Path = "data/traces.jsonl", limit: int = 20) -> list[dict]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines()[-limit:]:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tracking_status_dict(client: TrackingClient | None = None) -> dict[str, Any]:
    status = (client or TrackingClient()).status()
    return asdict(status)
