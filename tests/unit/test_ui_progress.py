from __future__ import annotations

import unittest

from ui.progress import CLICK_PROGRESS


class UiProgressTest(unittest.TestCase):
    def test_click_progress_uses_full_gradio_indicator(self) -> None:
        self.assertEqual(CLICK_PROGRESS, "full")

    def test_ui_tabs_remain_importable_with_progress_helper(self) -> None:
        from ui.agent_tab import build_agent_tab
        from ui.chat_tab import build_chat_tab
        from ui.dataset_tab import build_dataset_tab
        from ui.export_tab import build_export_tab
        from ui.notes_tab import build_notes_tab
        from ui.status_tab import build_status_tab
        from ui.traces_tab import build_traces_tab
        from ui.train_tab import build_train_tab
        from ui.vision_tab import build_vision_tab

        self.assertTrue(callable(build_agent_tab))
        self.assertTrue(callable(build_chat_tab))
        self.assertTrue(callable(build_dataset_tab))
        self.assertTrue(callable(build_export_tab))
        self.assertTrue(callable(build_notes_tab))
        self.assertTrue(callable(build_status_tab))
        self.assertTrue(callable(build_traces_tab))
        self.assertTrue(callable(build_train_tab))
        self.assertTrue(callable(build_vision_tab))


if __name__ == "__main__":
    unittest.main()
