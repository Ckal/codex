from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.runner import (
    AGENT_SYSTEM_PROMPT,
    export_agent_traces,
    export_agent_traces_hf_dataset,
    run_agent_loop,
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


if __name__ == "__main__":
    unittest.main()
