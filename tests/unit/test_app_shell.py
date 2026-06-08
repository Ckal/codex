from __future__ import annotations

import unittest
from pathlib import Path

from app import APP_CSS


class AppShellTest(unittest.TestCase):
    def test_app_css_includes_compact_responsive_rules(self) -> None:
        self.assertIn("@media (max-width: 720px)", APP_CSS)
        self.assertIn("max-width: 1180px", APP_CSS)
        self.assertIn("overflow-x: auto", APP_CSS)
        self.assertIn("min-height: 2.5rem", APP_CSS)

    def test_app_launch_enables_gradio_mcp_server(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("mcp_server=True", source)

    def test_app_disables_gradio_analytics_for_local_first_launch(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("analytics_enabled=False", source)

    def test_app_includes_vllm_tab(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('with gr.Tab("vLLM")', source)
        self.assertIn("build_vllm_tab(catalog)", source)


if __name__ == "__main__":
    unittest.main()
