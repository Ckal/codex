from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.evaluation import (
    compare_base_vs_tuned,
    default_prompt_cases,
    evaluate_responses,
    load_prompt_cases,
    log_eval_report,
)


class EvaluationTest(unittest.TestCase):
    def test_evaluates_exact_match_rate(self) -> None:
        cases = default_prompt_cases()

        report = evaluate_responses(cases, ["field note", "wrong", "no"])

        self.assertEqual(report.exact_match_rate, 2 / 3)
        self.assertEqual(report.rows[1].notes, "review")

    def test_compares_base_vs_tuned_reports(self) -> None:
        cases = default_prompt_cases()
        base = evaluate_responses(cases, ["wrong", "wrong", "wrong"])
        tuned = evaluate_responses(cases, ["field note", "jsonl", "no"])

        comparison = compare_base_vs_tuned(base, tuned)

        self.assertEqual(comparison.delta, 1.0)

    def test_loads_prompt_cases_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"prompt": "Prompt", "expected": "Answer"}) + "\n",
                encoding="utf-8",
            )

            cases = load_prompt_cases(path)

            self.assertEqual(cases[0].prompt, "Prompt")
            self.assertEqual(cases[0].expected, "Answer")

    def test_logs_eval_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            report = evaluate_responses(default_prompt_cases(), ["field note"])

            saved = log_eval_report(report, path)

            self.assertEqual(saved, path)
            self.assertIn("exact_match_rate", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
