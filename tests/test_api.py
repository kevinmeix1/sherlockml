"""FastAPI boundary tests for the local SherlockML command room."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents.engineer import PIPELINE_CONTRACT
from api.main import app, reset_state, run_investigation


@pytest.fixture(autouse=True)
def _restore_contract() -> None:
    reset_state()
    yield
    reset_state()


def test_health_and_incident_catalogue_are_available() -> None:
    client = TestClient(app)

    health = client.get("/health")
    catalogue = client.get("/api/incidents")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert catalogue.status_code == 200
    assert [item["id"] for item in catalogue.json()["incidents"]] == [
        "data_drift",
        "pipeline_bug",
        "model_regression",
    ]


def test_investigation_endpoint_returns_live_dashboard_contract() -> None:
    response = TestClient(app).post("/api/incidents/data_drift/investigate")

    assert response.status_code == 200
    case = response.json()
    assert case["source"] == "live-engine"
    assert case["incident_type"] == "data_drift"
    assert case["recovery"]["approved"] is True
    assert case["timeline"][-1]["phase"] == "REPORT"
    assert case["experiments"][1]["f1"] > case["experiments"][0]["f1"]


def test_invalid_incident_is_rejected_and_feature_alias_is_accepted() -> None:
    client = TestClient(app)

    invalid = client.post("/api/incidents/not-a-real-incident/investigate")
    alias = client.post("/api/incidents/feature_pipeline_bug/investigate")

    assert invalid.status_code == 422
    assert "Unsupported incident type" in invalid.json()["detail"]
    assert alias.status_code == 200
    assert alias.json()["incident_type"] == "pipeline_bug"


def test_reset_restores_the_default_pipeline_contract() -> None:
    before = json.loads(Path(PIPELINE_CONTRACT).read_text())
    run_investigation("data_drift")
    changed = json.loads(Path(PIPELINE_CONTRACT).read_text())

    response = TestClient(app).post("/api/reset")
    restored = json.loads(Path(PIPELINE_CONTRACT).read_text())

    assert response.status_code == 200
    assert changed != before
    assert restored == before
