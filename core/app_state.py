from __future__ import annotations

import logging
from dataclasses import asdict

from core.events import Event, EventBus, EventType
from tracking.trackio_client import TrackingClient


class AppState:
    """Shared in-process state for UI events and lightweight traces."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        tracking_client: TrackingClient | None = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.events: list[Event] = []
        self.logger = logging.getLogger("openbmb_workbench")
        self.tracking_client = tracking_client or TrackingClient()
        self.tracking_client.init()

    def emit(self, event: Event) -> None:
        self.events.append(event)
        self.logger.info("event=%s payload=%s", event.type.value, event.payload)
        self.tracking_client.log(event.type.value, event.payload)
        self.event_bus.emit(event)

    def recent_events(self, limit: int = 20) -> list[dict]:
        recent = self.events[-limit:]
        return [asdict(event) for event in recent]


def emit_inference_response(
    mode: str,
    model_id: str,
    backend: str,
    response: str,
    state: AppState | None = None,
) -> None:
    target_state = state or APP_STATE
    target_state.emit(
        Event(
            EventType.INFERENCE_RESPONSE,
            {
                "mode": mode,
                "model_id": model_id,
                "backend": backend,
                "response_chars": len(response),
            },
        )
    )


APP_STATE = AppState()
