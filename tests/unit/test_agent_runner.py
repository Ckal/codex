from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.runner import (
    AGENT_SYSTEM_PROMPT,
    default_safety_gates,
    export_agent_traces,
    export_agent_traces_hf_dataset,
    run_agent_loop,
    run_paper_to_code_loop,
    save_agent_trace,
)


class AgentRunnerTest(unittest.TestCase):
    def test_agent_loop_creates_research_plan_implementation_trace(self) -> None:
        session = run_agent_loop("Improve dataset stats")

        phases = [step.phase for step in session.steps]
        self.assertIn("research", phases)
        self.assertIn("plan", phases)
        self.assertIn("implement", phases)
        self.assertIn("verify", phases)
        self.assertIn("safe_calculator", session.tools)
        self.assertTrue(session.safety_gates)
        self.assertEqual(session.as_dict()["system_prompt"], AGENT_SYSTEM_PROMPT)

    def test_agent_loop_can_use_safe_calculator_tool(self) -> None:
        session = run_agent_loop("calculate: 2 + 2")

        self.assertIn("tool:safe_calculator", [step.phase for step in session.steps])
        self.assertIn("4.0", session.as_markdown())

    def test_saves_and_exports_agent_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "agent_traces.jsonl"
            output = Path(tmp) / "exports" / "agent_traces.jsonl"
            session = run_agent_loop("Improve docs")

            saved = save_agent_trace(session, source)
            exported = export_agent_traces(source, output)

            self.assertEqual(saved, source)
            self.assertEqual(exported, output)
            self.assertIn("Improve docs", output.read_text(encoding="utf-8"))

    def test_exports_agent_trace_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "agent_traces.jsonl"
            target = Path(tmp) / "dataset"
            save_agent_trace(run_agent_loop("Trace me"), source)

            exported = export_agent_traces_hf_dataset(source, target)

            self.assertEqual(exported, target)
            self.assertTrue((target / "data.jsonl").exists())
            self.assertTrue((target / "README.md").exists())

    def test_paper_to_code_trace_has_required_phases_and_safety_gates(self) -> None:
        session = run_paper_to_code_loop(
            "Demo paper",
            "Claims a local reward model can rank completions.",
            "Implement deterministic reward eval.",
        )

        phases = [step.phase for step in session.steps]
        self.assertEqual(phases, ["research", "plan", "implement", "verify"])
        self.assertIn("Paper-to-code", session.task)
        self.assertIn("No shell commands", session.as_markdown())

    def test_default_safety_gates_block_external_side_effects(self) -> None:
        gates = default_safety_gates()

        self.assertTrue(any("downloaded automatically" in gate for gate in gates))
        self.assertTrue(any("matching test" in gate for gate in gates))


if __name__ == "__main__":
    unittest.main()
