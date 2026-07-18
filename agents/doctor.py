"""Translate a technical diagnosis into a model-health treatment plan."""

from __future__ import annotations

from typing import Any


def prescribe(
    consensus: dict[str, Any],
    experiment: dict[str, Any],
    incident: dict[str, Any],
) -> dict[str, Any]:
    validation = experiment.get("validation", {})
    approved = bool(validation.get("approved", experiment.get("approved", False)))
    improvement = experiment.get("improvement", experiment.get("deltas", {}))
    candidate = experiment.get("candidate", experiment.get("after", {}))
    gates = list(validation.get("gates", []))
    passed = sum(1 for gate in gates if gate.get("passed"))
    recovery_probability = _recovery_probability(approved, improvement, candidate, gates)

    treatment_source = experiment.get("proposed_fix", {}).get("changes")
    if not treatment_source:
        treatment_source = _default_treatment(incident["kind"])
    treatment = [str(step) for step in treatment_source]

    return {
        "patient": "Fraud Detection Model",
        "condition": consensus["name"],
        "severity": "HIGH" if not approved else "MODERATE",
        "treatment": treatment,
        "recovery_probability": recovery_probability,
        "decision": "CLEARED FOR HUMAN REVIEW" if approved else "HOLD FOR HUMAN REVIEW",
        "bedside_note": _bedside_note(approved, passed, len(gates), improvement),
    }


def _recovery_probability(
    approved: bool,
    improvement: dict[str, Any],
    candidate: dict[str, Any],
    gates: list[dict[str, Any]],
) -> int:
    f1_delta = float(improvement.get("f1", 0))
    candidate_f1 = float(candidate.get("f1", 0))
    candidate_recall = float(candidate.get("recall", 0))
    gate_ratio = (sum(1 for gate in gates if gate.get("passed")) / len(gates)) if gates else 0.0

    score = 42.0
    score += min(28.0, max(f1_delta, 0) * 120.0)
    score += min(18.0, candidate_f1 * 22.0)
    score += min(10.0, candidate_recall * 12.0)
    score += gate_ratio * 12.0
    if approved:
        score += 8.0
    return int(min(97, max(35, round(score))))


def _bedside_note(approved: bool, passed: int, total: int, improvement: dict[str, Any]) -> str:
    f1_delta = float(improvement.get("f1", 0))
    if approved:
        return (
            f"The candidate cleared {passed}/{total} recovery gates with an F1 lift of "
            f"{f1_delta:+.3f}. Human approval is still required before any production promotion."
        )
    return (
        f"The candidate cleared only {passed}/{total} recovery gates (F1 delta {f1_delta:+.3f}); "
        "no production promotion is advised until the treatment plan is revised."
    )


def _default_treatment(incident_kind: str) -> list[str]:
    return {
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
    }[incident_kind]
