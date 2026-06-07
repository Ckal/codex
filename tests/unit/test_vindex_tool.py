from __future__ import annotations

import unittest

import requests

from mcp_tools.vindex_tool import (
    VINDEX_METHODS,
    build_vindex_call_plan,
    vindex_dependency_report,
    vindex_verification_report,
)


class VINDEXToolTest(unittest.TestCase):
    def test_builds_non_executing_call_plan_for_supported_method(self) -> None:
        plan = build_vindex_call_plan(
            "logit_lens",
            {"model_id": "minicpm5_1b", "text": "Acer"},
            "http://local-vindex",
        )

        self.assertEqual(plan.method, "logit_lens")
        self.assertEqual(plan.endpoint, "http://local-vindex/logit_lens")
        self.assertFalse(plan.execute)
        self.assertEqual(plan.payload["text"], "Acer")

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            build_vindex_call_plan("unknown", {})

    def test_caps_star_spread_neighbors(self) -> None:
        plan = build_vindex_call_plan("star_spread", {"n_neighbors": 12})

        self.assertEqual(plan.payload["n_neighbors"], 5)
        self.assertTrue(any("capped at 5" in note for note in plan.safety_notes))

    def test_caps_calibrated_edit_window(self) -> None:
        plan = build_vindex_call_plan("calibrated_edit", {"causal_window": 9})

        self.assertEqual(plan.payload["causal_window"], 3)
        self.assertTrue(any("capped at 3" in note for note in plan.safety_notes))

    def test_dependency_report_includes_methods_and_unreachable_server(self) -> None:
        def get_health(url: str, **kwargs):
            del url, kwargs
            raise requests.ConnectionError("server down")

        report = vindex_dependency_report("http://local-vindex", get_health)

        self.assertFalse(report["server_reachable"])
        self.assertEqual(report["supported_methods"], sorted(VINDEX_METHODS))
        self.assertIn("server down", report["server_detail"])

    def test_verification_report_keeps_execution_disabled(self) -> None:
        def get_health(url: str, **kwargs):
            del url, kwargs
            response = requests.Response()
            response.status_code = 200
            return response

        report = vindex_verification_report(
            "protect_relations",
            {"protected_triplets": []},
            "http://local-vindex",
            get_health,
        )

        self.assertFalse(report["ready_for_execution"])
        self.assertTrue(report["dependency_report"]["server_reachable"])
        self.assertEqual(report["call_plan"]["method"], "protect_relations")


if __name__ == "__main__":
    unittest.main()
