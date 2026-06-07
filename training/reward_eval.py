from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(frozen=True)
class RewardCriteria:
    """Deterministic local scoring criteria for reward-style evaluation."""

    positive_terms: tuple[str, ...] = (
        "correct",
        "jsonl",
        "local",
        "offline",
        "no download",
        "field note",
        "concise",
    )
    negative_terms: tuple[str, ...] = (
        "download on startup",
        "cloud api",
        "unknown",
        "incorrect",
        "wrong",
    )
    max_response_chars: int = 1200


@dataclass(frozen=True)
class RewardScore:
    """One prompt/response reward score with local heuristic details."""

    prompt: str
    response: str
    score: float
    notes: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate response ranked by reward."""

    prompt: str
    response: str
    reward: float
    rank: int
    index: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DPOPair:
    """A deterministic chosen/rejected pair ready for DPO-style datasets."""

    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    reward_gap: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardComparisonRow:
    """Per-prompt base-vs-LoRA reward comparison."""

    prompt: str
    base_response: str
    lora_response: str
    base_reward: float
    lora_reward: float
    delta: float
    winner: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardComparisonReport:
    """Aggregate LoRA-vs-base reward report."""

    rows: list[RewardComparisonRow]
    base_mean: float
    lora_mean: float
    delta: float
    lora_win_rate: float

    def as_table(self) -> list[list[str]]:
        return [
            [
                row.prompt,
                row.base_response,
                row.lora_response,
                f"{row.base_reward:.3f}",
                f"{row.lora_reward:.3f}",
                f"{row.delta:.3f}",
                row.winner,
            ]
            for row in self.rows
        ]

    def as_dict(self) -> dict:
        return {
            "base_mean": self.base_mean,
            "lora_mean": self.lora_mean,
            "delta": self.delta,
            "lora_win_rate": self.lora_win_rate,
            "rows": [row.as_dict() for row in self.rows],
        }


class RewardEvaluator:
    """Local deterministic reward evaluator.

    This helper never loads models, downloads weights, or calls external services. It scores
    already-supplied responses with transparent lexical heuristics so reward workflows can be
    prototyped before a real reward model is wired in.
    """

    def __init__(self, criteria: RewardCriteria | None = None) -> None:
        self.criteria = criteria or RewardCriteria()

    def score(self, prompt: str, response: str) -> float:
        return self.evaluate(prompt, response).score

    def evaluate(self, prompt: str, response: str) -> RewardScore:
        prompt_tokens = _content_tokens(prompt)
        response_tokens = _content_tokens(response)
        normalized_response = _normalize(response)
        notes = []

        score = 0.0
        if response.strip():
            score += 0.2
            notes.append("non_empty")
        else:
            notes.append("empty")

        if response_tokens:
            score += min(len(response_tokens) / 40, 1.0) * 0.25
            notes.append("substantive")

        if prompt_tokens and response_tokens:
            overlap = len(set(prompt_tokens) & set(response_tokens)) / len(set(prompt_tokens))
            score += min(overlap, 1.0) * 0.25
            if overlap:
                notes.append("prompt_overlap")

        positive_hits = _term_hits(normalized_response, self.criteria.positive_terms)
        negative_hits = _term_hits(normalized_response, self.criteria.negative_terms)
        score += min(positive_hits, 4) * 0.15
        score -= min(negative_hits, 4) * 0.2
        if positive_hits:
            notes.append(f"positive_terms:{positive_hits}")
        if negative_hits:
            notes.append(f"negative_terms:{negative_hits}")

        if len(response) > self.criteria.max_response_chars:
            score -= 0.15
            notes.append("too_long")

        return RewardScore(
            prompt=prompt,
            response=response,
            score=round(score, 6),
            notes=", ".join(notes),
        )

    def rank_candidates(self, prompt: str, candidates: Sequence[str]) -> list[ScoredCandidate]:
        scored = [
            (index, candidate, self.score(prompt, candidate))
            for index, candidate in enumerate(candidates)
        ]
        ranked = sorted(scored, key=lambda item: (-item[2], item[0]))
        return [
            ScoredCandidate(
                prompt=prompt,
                response=response,
                reward=reward,
                rank=rank,
                index=index,
            )
            for rank, (index, response, reward) in enumerate(ranked, start=1)
        ]

    def best_of_n(self, prompt: str, candidates: Sequence[str]) -> ScoredCandidate:
        ranked = self.rank_candidates(prompt, candidates)
        if not ranked:
            raise ValueError("At least one candidate is required.")
        return ranked[0]

    def create_dpo_pairs(
        self,
        prompt_responses: Mapping[str, Sequence[str]],
        min_reward_gap: float = 0.0,
    ) -> list[DPOPair]:
        pairs = []
        for prompt, responses in prompt_responses.items():
            ranked = self.rank_candidates(prompt, responses)
            if len(ranked) < 2:
                continue
            best = ranked[0]
            worst = ranked[-1]
            reward_gap = round(best.reward - worst.reward, 6)
            if reward_gap <= min_reward_gap:
                continue
            pairs.append(
                DPOPair(
                    prompt=prompt,
                    chosen=best.response,
                    rejected=worst.response,
                    chosen_reward=best.reward,
                    rejected_reward=worst.reward,
                    reward_gap=reward_gap,
                )
            )
        return pairs

    def eval_lora_vs_base(
        self,
        prompts: Sequence[str],
        base_responses: Mapping[str, str] | Sequence[str],
        lora_responses: Mapping[str, str] | Sequence[str],
    ) -> RewardComparisonReport:
        rows = []
        for index, prompt in enumerate(prompts):
            base_response = _response_for(prompt, index, base_responses)
            lora_response = _response_for(prompt, index, lora_responses)
            base_reward = self.score(prompt, base_response)
            lora_reward = self.score(prompt, lora_response)
            delta = round(lora_reward - base_reward, 6)
            rows.append(
                RewardComparisonRow(
                    prompt=prompt,
                    base_response=base_response,
                    lora_response=lora_response,
                    base_reward=base_reward,
                    lora_reward=lora_reward,
                    delta=delta,
                    winner=_winner(delta),
                )
            )

        base_mean = _mean([row.base_reward for row in rows])
        lora_mean = _mean([row.lora_reward for row in rows])
        lora_wins = sum(1 for row in rows if row.winner == "lora")
        return RewardComparisonReport(
            rows=rows,
            base_mean=base_mean,
            lora_mean=lora_mean,
            delta=round(lora_mean - base_mean, 6),
            lora_win_rate=round(lora_wins / len(rows), 6) if rows else 0.0,
        )


def _response_for(
    prompt: str,
    index: int,
    responses: Mapping[str, str] | Sequence[str],
) -> str:
    if isinstance(responses, Mapping):
        return responses.get(prompt, "")
    if index >= len(responses):
        return ""
    return responses[index]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _winner(delta: float) -> str:
    if delta > 0:
        return "lora"
    if delta < 0:
        return "base"
    return "tie"


def _term_hits(text: str, terms: Sequence[str]) -> int:
    return sum(1 for term in terms if _normalize(term) in text)


def _content_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in _STOPWORDS
    ]


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())
