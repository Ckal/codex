from __future__ import annotations

import unittest

from core.app_state import AppState
from core.events import EventType
from core.tab_feedback import emit_tab_error, status_ok


class TabFeedbackTest(unittest.TestCase):
    def test_status_ok_formats_tab_status(self) -> None:
        self.assertEqual(status_ok("Ready."), "Status: Ready.")

    def test_emit_tab_error_records_ui_error_event(self) -> None:
        state = AppState()

        message = emit_tab_error(
            "Chat",
            "Enter a prompt before running chat.",
            {"model_id": "minicpm5_1b"},
            state=state,
        )

        self.assertEqual(message, "Error: Enter a prompt before running chat.")
        event = state.recent_events()[0]
        self.assertEqual(event["type"], EventType.UI_ERROR)
        self.assertEqual(event["payload"]["tab"], "Chat")
        self.assertEqual(event["payload"]["model_id"], "minicpm5_1b")


if __name__ == "__main__":
    unittest.main()
