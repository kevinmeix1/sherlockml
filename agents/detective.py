"""The lead investigator turns raw telemetry into a coherent case file."""

from __future__ import annotations

from typing import Any


def investigate(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Create evidence and competing hypotheses from a normalized snapshot."""
    incident = snapshot["incident"]
    baseline = snapshot["baseline"]
    health = snapshot["health"]
    diagnostics = snapshot["diagnostics"]
    pipeline = snapshot["pipeline"]

    evidence: list[dict[str, Any]] = [
        {
            "id": "E-01",
            "source": "Production telemetry",
            "title": "Model-health breach",
            "finding": (
                f"F1 fell from {health['healthy_f1']:.2f} to {baseline['f1']:.2f}; "
                f"the reliability policy floor is {health['f1_floor']:.2f}."
            ),
            "strength": "high",
            "value": f"{baseline['f1']:.2f} F1",
        },
        {
            "id": "E-02",
            "source": "Distribution monitor",
            "title": "Feature movement",
            "finding": diagnostics["headline"],
            "strength": diagnostics["severity"],
            "value": diagnostics["headline_value"],
        },
        {
            "id": "E-03",
            "source": "Pipeline registry",
            "title": "Serving/training contract",
            "finding": pipeline["finding"],
            "strength": pipeline["severity"],
            "value": pipeline["version"],
        },
    ]

    suspects: list[dict[str, Any]] = [
        {
            "name": "Feature distribution shift",
            "confidence": _confidence(incident["kind"], "data_drift"),
            "rationale": diagnostics["headline"],
            "status": "primary" if incident["kind"] == "data_drift" else "open",
        },
        {
            "name": "Preprocessing contract defect",
            "confidence": _confidence(incident["kind"], "pipeline_bug"),
            "rationale": pipeline["finding"],
            "status": "primary" if incident["kind"] == "pipeline_bug" else "open",
        },
        {
            "name": "Model training regression",
            "confidence": _confidence(incident["kind"], "model_regression"),
            "rationale": "The deployed estimator differs from the last known healthy recipe.",
            "status": "primary" if incident["kind"] == "model_regression" else "open",
        },
    ]
    return {"evidence": evidence, "suspects": suspects}


def _confidence(actual_incident: str, suspect: str) -> int:
    if actual_incident == suspect:
        return {"data_drift": 88, "pipeline_bug": 91, "model_regression": 84}[suspect]
    return {"data_drift": 32, "pipeline_bug": 41, "model_regression": 27}[suspect]
