from __future__ import annotations

import time

import pytest

from models.model_catalog import load_model_catalog, validate_catalog
from models.placeholder_service import PlaceholderModelService


@pytest.mark.performance
def test_catalog_load_and_validate_is_fast_enough_for_app_start() -> None:
    start = time.perf_counter()

    catalog = load_model_catalog("config/models.yaml")
    validate_catalog(catalog)

    elapsed = time.perf_counter() - start
    assert elapsed < 0.25


@pytest.mark.performance
def test_placeholder_response_is_fast_enough_for_ui_smoke_flow() -> None:
    catalog = load_model_catalog("config/models.yaml")
    service = PlaceholderModelService(catalog["minicpm5_1b"])

    start = time.perf_counter()
    response = service.chat("Be concise.", "Smoke test")
    elapsed = time.perf_counter() - start

    assert "Real inference is not wired yet" in response
    assert elapsed < 0.05
