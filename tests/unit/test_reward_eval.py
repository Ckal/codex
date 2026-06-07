from __future__ import annotations

import unittest

from training.reward_eval import RewardCriteria, RewardEvaluator


class RewardEvalTest(unittest.TestCase):
    def test_scores_are_deterministic_and_local(self) -> None:
        evaluator = RewardEvaluator()

        first = evaluator.evaluate(
            "Should model weights download on startup?",
            "No. Keep inference local and do not download on startup.",
        )
        second = evaluator.evaluate(
            "Should model weights download on startup?",
            "No. Keep inference local and do not download on startup.",
        )

        self.assertEqual(first.score, second.score)
        self.assertIn("positive_terms", first.notes)

    def test_best_of_n_selects_highest_reward(self) -> None:
        evaluator = RewardEvaluator()

        best = evaluator.best_of_n(
            "What format should corrected training data use?",
            ["wrong", "Use JSONL for corrected field note training data."],
        )

        self.assertEqual(best.response, "Use JSONL for corrected field note training data.")
        self.assertEqual(best.rank, 1)
        self.assertEqual(best.index, 1)

    def test_candidate_ranking_uses_input_order_for_ties(self) -> None:
        evaluator = RewardEvaluator(RewardCriteria(positive_terms=(), negative_terms=()))

        ranked = evaluator.rank_candidates("Prompt", ["same score", "same score"])

        self.assertEqual([row.index for row in ranked], [0, 1])

    def test_best_of_n_requires_candidates(self) -> None:
        evaluator = RewardEvaluator()

        with self.assertRaises(ValueError):
            evaluator.best_of_n("Prompt", [])

    def test_creates_dpo_pairs_from_ranked_responses(self) -> None:
        evaluator = RewardEvaluator()

        pairs = evaluator.create_dpo_pairs(
            {
                "How should field notes be exported?": [
                    "unknown",
                    "Export corrected field note data as JSONL.",
                ]
            }
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen, "Export corrected field note data as JSONL.")
        self.assertEqual(pairs[0].rejected, "unknown")
        self.assertGreater(pairs[0].reward_gap, 0)

    def test_skips_dpo_pairs_without_enough_gap(self) -> None:
        evaluator = RewardEvaluator(RewardCriteria(positive_terms=(), negative_terms=()))

        pairs = evaluator.create_dpo_pairs(
            {"Prompt": ["same score", "same score"]},
            min_reward_gap=0.0,
        )

        self.assertEqual(pairs, [])

    def test_reports_lora_vs_base_rewards_from_sequences(self) -> None:
        evaluator = RewardEvaluator()

        report = evaluator.eval_lora_vs_base(
            ["Should inference stay local?", "What format should corrected data use?"],
            ["unknown", "wrong"],
            ["Yes, keep inference local and offline.", "Use JSONL for field note corrections."],
        )

        self.assertGreater(report.lora_mean, report.base_mean)
        self.assertEqual(report.lora_win_rate, 1.0)
        self.assertEqual(report.rows[0].winner, "lora")
        self.assertIn("lora_mean", report.as_dict())
        self.assertEqual(report.as_table()[0][-1], "lora")

    def test_reports_lora_vs_base_rewards_from_mappings(self) -> None:
        evaluator = RewardEvaluator()

        report = evaluator.eval_lora_vs_base(
            ["Prompt"],
            {"Prompt": "Use JSONL."},
            {"Prompt": "wrong"},
        )

        self.assertEqual(report.rows[0].winner, "base")

    def test_missing_sequence_response_is_scored_as_empty(self) -> None:
        evaluator = RewardEvaluator()

        report = evaluator.eval_lora_vs_base(["Prompt"], [], [])

        self.assertEqual(report.base_mean, 0.0)
        self.assertEqual(report.lora_mean, 0.0)
        self.assertEqual(report.rows[0].winner, "tie")


if __name__ == "__main__":
    unittest.main()
