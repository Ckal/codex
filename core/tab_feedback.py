from __future__ import annotations

from typing import Any

from core.app_state import APP_STATE, AppState
from core.events import Event, EventType


def status_ok(message: str) -> str:
    return f"Status: {message}"


def emit_tab_error(
    tab: str,
    message: str,
    context: dict[str, Any] | None = None,
    state: AppState | None = None,
) -> str:
    target_state = state or APP_STATE
    payload = {"tab": tab, "message": message}
    if context:
        payload.update(context)
    target_state.emit(Event(EventType.UI_ERROR, payload))
    return f"Error: {message}"
