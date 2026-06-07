from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyntheticExample:
    prompt: str
    response: str
    correction: str
    tags: list[str]
    source: str = "synthetic"

    def as_dict(self) -> dict:
        return asdict(self)


def generate_synthetic_examples(
    topic: str,
    count: int,
    tag: str = "synthetic",
) -> list[SyntheticExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    cleaned_topic = topic.strip() or "local AI workflow"
    return [
        SyntheticExample(
            prompt=f"Write a concise field note about {cleaned_topic} #{index}.",
            response=f"Draft note {index}: {cleaned_topic}.",
            correction=f"Corrected note {index}: {cleaned_topic} with concrete evidence.",
            tags=[tag, cleaned_topic.replace(" ", "-").lower()],
        )
        for index in range(1, count + 1)
    ]


def validate_synthetic_example(example: SyntheticExample) -> list[str]:
    errors = []
    if not example.prompt.strip():
        errors.append("prompt is required")
    if not example.response.strip():
        errors.append("response is required")
    if not example.correction.strip():
        errors.append("correction is required")
    if not example.tags:
        errors.append("at least one tag is required")
    return errors


def quality_filter_examples(examples: list[SyntheticExample]) -> list[SyntheticExample]:
    return [
        example
        for example in examples
        if not validate_synthetic_example(example)
        and example.prompt != example.response
        and example.response != example.correction
    ]


def augment_examples(
    examples: list[SyntheticExample],
    extra_tag: str = "augmented",
) -> list[SyntheticExample]:
    augmented = list(examples)
    for example in examples:
        tags = list(dict.fromkeys([*example.tags, extra_tag]))
        augmented.append(
            SyntheticExample(
                prompt=f"{example.prompt} Include one verification detail.",
                response=example.response,
                correction=f"{example.correction} Verification detail recorded.",
                tags=tags,
                source=example.source,
            )
        )
    return augmented


def export_synthetic_jsonl(
    examples: list[SyntheticExample],
    output_path: str | Path = "data/synthetic_examples.jsonl",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
    return output
