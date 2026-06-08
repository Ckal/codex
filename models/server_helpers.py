from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

from models.http_chat import post_chat_completion
from models.response_parsing import extract_chat_response


def host_port_args(host: str, port: int) -> list[str]:
    return ["--host", host, "--port", str(port)]


def request_openai_chat_completion(
    post_func: Callable[..., requests.Response],
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return post_chat_completion(
        post_func=post_func,
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        headers=headers,
    )


def request_openai_chat_text(
    post_func: Callable[..., requests.Response],
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    failure_label: str,
    headers: dict[str, str] | None = None,
) -> str:
    try:
        data = request_openai_chat_completion(
            post_func=post_func,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            headers=headers,
        )
    except requests.RequestException as exc:
        return f"[{failure_label} request failed]\n\n{exc}"
    return extract_chat_response(data)
