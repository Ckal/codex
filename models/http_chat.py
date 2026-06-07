from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import requests

from models.response_parsing import chat_completion_payload


def post_json(
    post_func: Callable[..., requests.Response],
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = post_func(
        url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


def post_chat_completion(
    post_func: Callable[..., requests.Response],
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return post_json(
        post_func,
        url,
        chat_completion_payload(model, system_prompt, user_prompt, temperature, max_tokens),
        timeout,
        headers,
    )
