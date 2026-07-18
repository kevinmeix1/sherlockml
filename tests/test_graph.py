"""Contract tests for the deterministic LangGraph investigation workflow."""

from __future__ import annotations

import pytest

from agents.graph import (
    SUPPORTED_INCIDENTS,
    normalize_incident_type,
    reset_local_state,
    run_case,
    to_dashboard_contract,
)


@pytest.fixture(autouse=True)
def _restore_contract() -> None:
    """Keep a test repair from changing the next test's starting state."""

    reset_local_state()
    yield
    reset_local_state()


def test_normalize_feature_pipeline_bug_alias() -> None:
    assert normalize_incident_type("feature_pipeline_bug") == "pipeline_bug"
    assert normalize_incident_type("Pipeline Bug") == "pipeline_bug"
    assert set(SUPPORTED_INCIDENTS) == {"data_drift", "pipeline_bug", "model_regression"}


def test_graph_produces_a_complete_dashboard_case() -> None:
    state = run_case("data_drift")
    case = to_dashboard_contract(state)

    assert case["source"] == "live-engine"
    assert case["case_id"] == "CASE-DRIFT-001"
    assert case["incident_type"] == "data_drift"
    assert len(case["evidence"]) >= 3
    assert len(case["suspects"]) == 3
    assert len(case["war_room"]) >= 5
    assert len(case["timeline"]) == 9
    assert case["timeline"][0] == {
        "time": "10:01",
        "phase": "ALERT",
        "text": "Synthetic data drift monitor crossed its reliability threshold.",
    }
    assert {"before", "after_incident", "recovered"}.issubset(case["health"])
    assert len(case["experiments"]) == 2
    assert case["artifacts"]["report"]
    assert case["artifacts"]["patch"]


def test_timeline_is_repeatable_between_clean_runs() -> None:
    first = to_dashboard_contract(run_case("model_regression"))
    reset_local_state()
    second = to_dashboard_contract(run_case("model_regression"))

    assert first["timeline"] == second["timeline"]
    assert first["case_id"] == second["case_id"] == "CASE-REGRESSION-001"
    assert first["experiments"] == second["experiments"]
