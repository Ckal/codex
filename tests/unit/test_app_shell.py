from __future__ import annotations

import unittest

from app import APP_CSS


class AppShellTest(unittest.TestCase):
    def test_app_css_includes_compact_responsive_rules(self) -> None:
        self.assertIn("@media (max-width: 720px)", APP_CSS)
        self.assertIn("max-width: 1180px", APP_CSS)
        self.assertIn("overflow-x: auto", APP_CSS)
        self.assertIn("min-height: 2.5rem", APP_CSS)


if __name__ == "__main__":
    unittest.main()
