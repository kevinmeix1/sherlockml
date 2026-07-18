"""Contract tests for SherlockML's deterministic ML incident laboratory."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.data import generate_fraud_data
from ml.train import run_repair_experiment, train_model
from simulator.incidents import normalize_incident_kind, simulate_incident


@pytest.mark.parametrize("kind", ["data_drift", "pipeline_bug", "model_regression"])
def test_each_incident_recovers_after_the_candidate_treatment(kind: str) -> None:
    """The demo must show a true decline and a passing, measurable recovery."""

    simulation = simulate_incident(kind, seed=2026)
    result = run_repair_experiment(simulation)
    healthy = simulation.baseline["healthy_metrics"]
    incident = simulation.baseline["incident_metrics"]

    assert incident["f1"] < healthy["f1"]
    assert result["after"]["f1"] > result["before"]["f1"] + 0.08
    assert result["after"]["f1"] >= 0.58
    assert result["approved"] is True


def test_incident_windows_are_temporal_and_repeatable() -> None:
    """Repair data precedes the evaluation incident and a seed reproduces it."""

    first = simulate_incident("data_drift", seed=2042)
    second = simulate_incident("data_drift", seed=2042)

    assert (
        first.training_data["event_time"].max()
        < first.healthy_validation_data["event_time"].min()
    )
    assert (
        first.healthy_validation_data["event_time"].max()
        < first.incident_data["event_time"].min()
    )
    assert (
        first.repair_validation_data["event_time"].max()
        < first.incident_data["event_time"].min()
    )
    assert first.context() == second.context()


def test_bad_input_fails_with_an_explainable_contract_error() -> None:
    """Schema and scenario failures should be explicit, not silent degradation."""

    with pytest.raises(ValueError, match="unsupported incident kind"):
        normalize_incident_kind("unknown_incident")

    malformed = pd.DataFrame({"transaction_amount": [10.0, 20.0]})
    validation = generate_fraud_data(60, seed=11)
    with pytest.raises(ValueError, match="missing required columns"):
        train_model(malformed, validation)
