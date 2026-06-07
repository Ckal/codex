from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.file_exports import copy_text_file_or_empty
from mcp_tools.tools import safe_calculator_tool, tool_registry

AGENT_SYSTEM_PROMPT = (
    "You are a local workbench agent. Research the request, draft a small plan, "
    "name tools you would use, and require verification before marking work done."
)


@dataclass(frozen=True)
class AgentStep:
    """One deterministic agent trace step."""

    phase: str
    content: str


@dataclass(frozen=True)
class AgentSession:
    """Agent session trace."""

    task: str
    steps: list[AgentStep]
    tools: list[str]
    limitations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "steps": [asdict(step) for step in self.steps],
            "tools": self.tools,
            "limitations": self.limitations,
            "system_prompt": AGENT_SYSTEM_PROMPT,
        }

    def as_markdown(self) -> str:
        lines = [f"Task: {self.task or '(none)'}", ""]
        for step in self.steps:
            lines.append(f"{step.phase}: {step.content}")
        lines.append("")
        lines.append(f"Tools: {', '.join(self.tools)}")
        lines.append(f"Limitations: {'; '.join(self.limitations)}")
        return "\n".join(lines)


def run_agent_loop(task: str) -> AgentSession:
    tools = sorted(tool_registry())
    steps = [
        AgentStep("research", _research_summary(task)),
        AgentStep("plan", _plan_summary(task)),
        AgentStep("implement", _implementation_summary(task)),
        AgentStep("verify", "Run unit tests, smoke checks, quality gates, and update docs/tasks."),
    ]

    calculator_result = _maybe_calculate(task)
    if calculator_result is not None:
        steps.insert(
            1,
            AgentStep(
                "tool:safe_calculator",
                json.dumps(calculator_result.payload, ensure_ascii=False),
            ),
        )

    return AgentSession(
        task=task,
        steps=steps,
        tools=tools,
        limitations=[
            "Does not execute shell commands.",
            "Does not commit, push, deploy, download models, or call external services.",
            "Requires Codex or a human to apply and verify implementation changes.",
        ],
    )


def save_agent_trace(
    session: AgentSession,
    path: str | Path = "data/agent_traces.jsonl",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(session.as_dict(), ensure_ascii=False) + "\n")
    return output


def export_agent_traces(
    source_path: str | Path = "data/agent_traces.jsonl",
    output_path: str | Path = "exports/agent_traces.jsonl",
) -> Path:
    return copy_text_file_or_empty(source_path, output_path)


def export_agent_traces_hf_dataset(
    source_path: str | Path = "data/agent_traces.jsonl",
    output_dir: str | Path = "exports/agent_traces_dataset",
) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    data_file = target / "data.jsonl"
    if Path(source_path).exists():
        data_file.write_text(Path(source_path).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        data_file.write_text("", encoding="utf-8")
    (target / "README.md").write_text(
        "# Agent Traces Dataset\n\n"
        "Local Hugging Face Dataset-style export for OpenBMB Local AI Workbench traces.\n",
        encoding="utf-8",
    )
    return target


def _research_summary(task: str) -> str:
    if not task.strip():
        return "No task provided. Ask for a concrete task before implementation."
    return "Inspect PRD/tasks/docs, identify affected modules, and check existing tests."


def _plan_summary(task: str) -> str:
    if any(word in task.casefold() for word in ["deploy", "push", "github", "huggingface"]):
        return "Prepare repo/deploy steps, verify auth/remotes, then push only after tests pass."
    return "Make a focused implementation slice, add or update tests, then update docs."


def _implementation_summary(task: str) -> str:
    if "model" in task.casefold():
        return "Use configured backend services and avoid startup downloads."
    return "Apply changes in the smallest relevant modules and keep unrelated files untouched."


def _maybe_calculate(task: str):
    prefix = "calculate:"
    if task.casefold().strip().startswith(prefix):
        expression = task.split(":", 1)[1].strip()
        return safe_calculator_tool(expression)
    return None
