from __future__ import annotations

from typing import Any


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
