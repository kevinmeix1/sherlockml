"""Translate a technical diagnosis into a model-health treatment plan."""

from __future__ import annotations

from typing import Any


def prescribe(
    consensus: dict[str, Any],
    experiment: dict[str, Any],
    incident: dict[str, Any],
) -> dict[str, Any]:
    approved = experiment["validation"]["approved"]
    recovery_probability = 94 if approved else 58
    treatment_by_incident = {
        "data_drift": [
            "Refresh the training window with recent labelled traffic.",
            "Promote behavior features that now carry fraud signal.",
            "Add PSI and KS guardrails before the next deployment.",
        ],
        "pipeline_bug": [
            "Restore the validated feature transformation contract.",
            "Add missing-value and scaling assertions at serving time.",
            "Gate releases on parity between training and serving transforms.",
        ],
        "model_regression": [
            "Restore the last known-good estimator recipe.",
            "Compare candidate hyperparameters against the champion model.",
            "Require regression tests before promotion.",
        ],
    }
    return {
        "patient": "Fraud Detection Model",
        "condition": consensus["name"],
        "severity": "HIGH",
        "treatment": treatment_by_incident[incident["kind"]],
        "recovery_probability": recovery_probability,
        "decision": "CLEARED FOR HUMAN REVIEW" if approved else "HOLD FOR HUMAN REVIEW",
        "bedside_note": (
            "The candidate has crossed all pre-defined reliability gates. Human approval is still "
            "required before any production promotion."
            if approved
            else "The candidate did not clear the safety gates; no production promotion is advised."
        ),
    }
