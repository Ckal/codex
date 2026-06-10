from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    """Python 3.10-compatible subset of enum.StrEnum."""

    def __str__(self) -> str:
        return str(self.value)


class EventType(StrEnum):
    """Event names shared across future services."""

    DATASET_LOADED = "dataset_loaded"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    UI_ERROR = "ui_error"
    FIELD_NOTE_SAVED = "field_note_saved"
    TRAINING_STARTED = "training_started"
    EXPORT_STARTED = "export_started"


@dataclass
class Event:
    """A lightweight event payload."""

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """In-process event dispatcher for low-volume app events."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {}

    def on(self, event_type: EventType) -> Callable:
        def decorator(fn: Callable[[Event], None]) -> Callable[[Event], None]:
            self._handlers.setdefault(event_type, []).append(fn)
            return fn

        return decorator

    def emit(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            handler(event)


bus = EventBus()
