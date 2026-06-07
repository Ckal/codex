from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import requests

VINDEX_BASE_URL = "http://127.0.0.1:8765"
VINDEX_METHODS = {
    "logit_lens": "/logit_lens",
    "slot_neighbors": "/slot_neighbors",
    "layer_contribution": "/layer_contribution",
    "transition_spectrum": "/transition_spectrum",
    "calibrated_edit": "/calibrated_edit",
    "derive_scale": "/derive_scale",
    "star_spread": "/star_spread",
    "protect_relations": "/protect_relations",
}


@dataclass(frozen=True)
class VINDEXCallPlan:
    method: str
    endpoint: str
    payload: dict[str, Any]
    execute: bool
    safety_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vindex_dependency_report(
    base_url: str = VINDEX_BASE_URL,
    get_func: Callable[..., requests.Response] = requests.get,
) -> dict[str, Any]:
    package_available = importlib.util.find_spec("vindex") is not None
    server = _server_status(base_url, get_func)
    return {
        "package_available": package_available,
        "server_base_url": base_url.rstrip("/"),
        "server_reachable": server["reachable"],
        "server_detail": server["detail"],
        "supported_methods": sorted(VINDEX_METHODS),
        "dependency": "Install or run VINDEX locally; no cloud endpoint is configured.",
    }


def build_vindex_call_plan(
    method: str,
    payload: dict[str, Any],
    base_url: str = VINDEX_BASE_URL,
) -> VINDEXCallPlan:
    if method not in VINDEX_METHODS:
        raise ValueError(f"Unsupported VINDEX method: {method}")

    safe_payload = _safe_payload(method, payload)
    return VINDEXCallPlan(
        method=method,
        endpoint=f"{base_url.rstrip('/')}{VINDEX_METHODS[method]}",
        payload=safe_payload,
        execute=False,
        safety_notes=_safety_notes(method, payload, safe_payload),
    )


def vindex_verification_report(
    method: str = "logit_lens",
    payload: dict[str, Any] | None = None,
    base_url: str = VINDEX_BASE_URL,
    get_func: Callable[..., requests.Response] = requests.get,
) -> dict[str, Any]:
    call_plan = build_vindex_call_plan(method, payload or {"model_id": "", "text": ""}, base_url)
    return {
        "dependency_report": vindex_dependency_report(base_url, get_func),
        "call_plan": call_plan.to_dict(),
        "ready_for_execution": False,
        "reason": (
            "The workbench records a safe VINDEX boundary first. Execute only after the local "
            "VINDEX package/server, model loading, and edit safety checks are verified."
        ),
    }


def _server_status(
    base_url: str,
    get_func: Callable[..., requests.Response],
) -> dict[str, object]:
    url = base_url.rstrip("/")
    try:
        response = get_func(f"{url}/health", timeout=2)
    except requests.RequestException as exc:
        return {"reachable": False, "detail": f"VINDEX server not reachable: {exc}"}
    if response.ok:
        return {"reachable": True, "detail": "VINDEX health endpoint is reachable."}
    return {
        "reachable": False,
        "detail": f"VINDEX health endpoint responded with HTTP {response.status_code}.",
    }


def _safe_payload(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    safe_payload = dict(payload)
    if method == "star_spread":
        requested = int(safe_payload.get("n_neighbors", 5))
        safe_payload["n_neighbors"] = min(requested, 5)
    if method == "calibrated_edit":
        requested_window = int(safe_payload.get("causal_window", 3))
        safe_payload["causal_window"] = min(requested_window, 3)
    return safe_payload


def _safety_notes(
    method: str,
    original_payload: dict[str, Any],
    safe_payload: dict[str, Any],
) -> list[str]:
    notes = [
        "Non-executing plan only; no model weights are edited by this workbench helper.",
        "Run VINDEX locally and verify protected relations before any calibrated edit.",
    ]
    if method == "star_spread" and original_payload != safe_payload:
        notes.append("n_neighbors was capped at 5 to avoid over-editing related slots.")
    if method == "calibrated_edit" and original_payload != safe_payload:
        notes.append("causal_window was capped at 3 layers around the phase layer.")
    return notes
