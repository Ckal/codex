from __future__ import annotations

import os
from dataclasses import dataclass

LOCAL_DEPLOYMENT = "local"
SPACE_DEPLOYMENT = "space"
VALID_DEPLOYMENTS = {LOCAL_DEPLOYMENT, SPACE_DEPLOYMENT}
PLACEHOLDER_BACKEND = "placeholder"


@dataclass(frozen=True)
class DeploymentPolicy:
    mode: str = LOCAL_DEPLOYMENT

    @property
    def is_space(self) -> bool:
        return self.mode == SPACE_DEPLOYMENT

    @property
    def allow_placeholder_backend(self) -> bool:
        return not self.is_space

    @property
    def allow_demo_mode(self) -> bool:
        return not self.is_space


def deployment_mode(env_value: str | None = None) -> str:
    raw_value = env_value if env_value is not None else os.getenv("WORKBENCH_DEPLOYMENT", "")
    value = raw_value.strip().lower() or LOCAL_DEPLOYMENT
    if value not in VALID_DEPLOYMENTS:
        raise ValueError(
            "WORKBENCH_DEPLOYMENT must be 'local' or 'space'; "
            f"got {raw_value!r}."
        )
    return value


def current_policy() -> DeploymentPolicy:
    return DeploymentPolicy(deployment_mode())


def filter_backends_for_policy(backends: list[str], policy: DeploymentPolicy) -> list[str]:
    if policy.allow_placeholder_backend:
        return backends
    return [backend for backend in backends if backend != PLACEHOLDER_BACKEND]


def default_backend_for_policy(
    backends: list[str],
    preferred: str,
    policy: DeploymentPolicy,
) -> str:
    allowed = filter_backends_for_policy(backends, policy)
    if preferred in allowed:
        return preferred
    if allowed:
        return allowed[0]
    raise ValueError("No real backend is available for this deployment policy.")


def ensure_backend_allowed(backend: str, policy: DeploymentPolicy) -> None:
    if backend == PLACEHOLDER_BACKEND and not policy.allow_placeholder_backend:
        raise ValueError(
            "Placeholder inference is disabled in WORKBENCH_DEPLOYMENT=space. "
            "Select a real backend such as transformers, Ollama, llama.cpp, "
            "SGLang, or an OpenAI-compatible local server."
        )


def ensure_demo_mode_allowed(policy: DeploymentPolicy) -> None:
    if not policy.allow_demo_mode:
        raise ValueError(
            "Demo/no-model mode is disabled in WORKBENCH_DEPLOYMENT=space. "
            "Use a real OpenBMB model configuration for the deployed app."
        )
