from __future__ import annotations

from typing import Any


def chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat_completion_payload(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": chat_messages(system_prompt, user_prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def extract_chat_response(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message", {})
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
    return str(first.get("text", ""))
