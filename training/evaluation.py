from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tracking.trackio_client import TrackingClient


@dataclass(frozen=True)
class PromptCase:
    """One prompt and expected answer for lightweight evaluation."""

    prompt: str
    expected: str


@dataclass(frozen=True)
class EvalRow:
    """One evaluated response row."""

    prompt: str
    expected: str
    actual: str
    exact_match: bool
    notes: str


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation report."""

    rows: list[EvalRow]
    exact_match_rate: float

    def as_table(self) -> list[list[str]]:
        return [
            [
                row.prompt,
                row.expected,
                row.actual,
                str(row.exact_match),
                row.notes,
            ]
            for row in self.rows
        ]

    def as_dict(self) -> dict:
        return {
            "exact_match_rate": self.exact_match_rate,
            "rows": [asdict(row) for row in self.rows],
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Base-vs-tuned comparison using exact-match rates."""

    base: EvalReport
    tuned: EvalReport
    delta: float

    def as_dict(self) -> dict:
        return {
            "base_exact_match_rate": self.base.exact_match_rate,
            "tuned_exact_match_rate": self.tuned.exact_match_rate,
            "delta": self.delta,
        }


def default_prompt_cases() -> list[PromptCase]:
    return [
        PromptCase("Identify the correction target.", "field note"),
        PromptCase("What format should corrected training data use?", "jsonl"),
        PromptCase("Should model weights download on startup?", "no"),
    ]


def load_prompt_cases(path: str | Path) -> list[PromptCase]:
    rows = _read_jsonl(path)
    return [PromptCase(prompt=str(row["prompt"]), expected=str(row["expected"])) for row in rows]


def evaluate_responses(cases: list[PromptCase], responses: list[str]) -> EvalReport:
    rows = []
    for case, actual in zip(cases, responses, strict=False):
        exact = _normalize(case.expected) == _normalize(actual)
        rows.append(
            EvalRow(
                prompt=case.prompt,
                expected=case.expected,
                actual=actual,
                exact_match=exact,
                notes="exact" if exact else "review",
            )
        )
    exact_match_rate = 0.0
    if rows:
        exact_match_rate = sum(1 for row in rows if row.exact_match) / len(rows)
    return EvalReport(rows=rows, exact_match_rate=exact_match_rate)


def compare_base_vs_tuned(base: EvalReport, tuned: EvalReport) -> ComparisonReport:
    return ComparisonReport(
        base=base,
        tuned=tuned,
        delta=tuned.exact_match_rate - base.exact_match_rate,
    )


def log_eval_report(report: EvalReport, path: str | Path = "data/eval_results.jsonl") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report.as_dict(), ensure_ascii=False) + "\n")
    return output


def log_eval_metrics(
    report: EvalReport,
    client: TrackingClient | None = None,
) -> Path:
    tracker = client or TrackingClient()
    tracker.init(run_name="evaluation")
    return tracker.log(
        "training_metrics",
        {
            "exact_match_rate": report.exact_match_rate,
            "rows": len(report.rows),
        },
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())
