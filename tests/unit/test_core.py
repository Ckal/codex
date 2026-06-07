from __future__ import annotations

import unittest

from core.app_state import AppState
from core.events import Event, EventBus, EventType
from core.registry import Registry


class CoreTest(unittest.TestCase):
    def test_event_bus_calls_registered_handler(self) -> None:
        bus = EventBus()
        seen = []

        @bus.on(EventType.DATASET_LOADED)
        def handler(event: Event) -> None:
            seen.append(event.payload["rows"])

        bus.emit(Event(EventType.DATASET_LOADED, {"rows": 3}))

        self.assertEqual(seen, [3])

    def test_registry_registers_and_gets_items(self) -> None:
        registry: Registry[int] = Registry()
        registry.register("answer", 42)

        self.assertEqual(registry.get("answer"), 42)
        self.assertEqual(registry.list(), ["answer"])

    def test_registry_raises_for_missing_item(self) -> None:
        registry: Registry[int] = Registry()

        with self.assertRaises(KeyError):
            registry.get("missing")

    def test_app_state_records_and_dispatches_events(self) -> None:
        bus = EventBus()
        state = AppState(bus)
        seen = []

        @bus.on(EventType.FIELD_NOTE_SAVED)
        def handler(event: Event) -> None:
            seen.append(event.payload["model_id"])

        state.emit(Event(EventType.FIELD_NOTE_SAVED, {"model_id": "minicpm"}))

        self.assertEqual(seen, ["minicpm"])
        self.assertEqual(state.recent_events()[0]["payload"]["model_id"], "minicpm")


if __name__ == "__main__":
    unittest.main()
