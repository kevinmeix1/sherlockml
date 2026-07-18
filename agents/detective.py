"""The lead investigator turns raw telemetry into a coherent case file."""

from __future__ import annotations

from typing import Any

from ml.data import FEATURE_COLUMNS

_SUSPECT_KEYS = (
    ("Feature distribution shift", "data_drift"),
    ("Preprocessing contract defect", "pipeline_bug"),
    ("Model training regression", "model_regression"),
)


def investigate(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Create evidence and competing hypotheses from a normalized snapshot."""
    incident = snapshot["incident"]
    baseline = snapshot.get("baseline", {})
    metadata = snapshot.get("metadata", {})
    health = snapshot["health"]
    diagnostics = snapshot["diagnostics"]
    pipeline = snapshot["pipeline"]

    evidence: list[dict[str, Any]] = [
        {
            "id": "E-01",
            "source": "Production telemetry",
            "title": "Model-health breach",
            "finding": (
                f"F1 fell from {health['healthy_f1']:.2f} to {baseline.get('f1', 0):.2f}; "
                f"the reliability policy floor is {health['f1_floor']:.2f}."
            ),
            "strength": "high",
            "value": f"{baseline.get('f1', 0):.2f} F1",
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

    scores = _score_suspects(incident["kind"], diagnostics, pipeline, baseline, metadata)
    suspects: list[dict[str, Any]] = []
    for name, key in _SUSPECT_KEYS:
        confidence = int(round(scores[key]))
        suspects.append(
            {
                "name": name,
                "confidence": confidence,
                "rationale": _rationale(key, diagnostics, pipeline, baseline, metadata),
                "status": "primary" if incident["kind"] == key else "open",
            }
        )
    suspects.sort(key=lambda item: item["confidence"], reverse=True)
    for suspect in suspects:
        if suspect["name"] == next(n for n, k in _SUSPECT_KEYS if k == incident["kind"]):
            suspect["status"] = "primary"
        elif suspect["status"] == "primary":
            suspect["status"] = "open"
    return {"evidence": evidence, "suspects": suspects}


def _score_suspects(
    incident_kind: str,
    diagnostics: dict[str, Any],
    pipeline: dict[str, Any],
    baseline: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, float]:
    features = diagnostics.get("features", [])
    max_psi = max((_number(feature.get("psi")) for feature in features), default=0.0)
    critical_count = sum(1 for feature in features if feature.get("severity") == "critical")
    warning_count = sum(1 for feature in features if feature.get("severity") == "warning")
    max_missing = max(
        (_number(feature.get("missing_rate_current")) for feature in features), default=0.0
    )

    contract = pipeline.get("contract", {})
    preprocessing = contract.get("preprocessing", {})
    parity_enabled = bool(preprocessing.get("parity_check"))
    feature_gap = len(set(FEATURE_COLUMNS) - set(contract.get("selected_features", [])))

    active = baseline.get("active_model", metadata.get("active_model", {}))
    champion = baseline.get("champion_metadata", {})
    active_features = set(active.get("feature_columns", []))
    champion_features = set(champion.get("feature_columns", []))

    drift_score = min(
        95.0,
        28.0 + critical_count * 16.0 + warning_count * 8.0 + min(max_psi, 8.0) * 7.0,
    )
    pipeline_score = min(
        95.0,
        24.0 + max_missing * 140.0 + (0 if parity_enabled else 22.0) + feature_gap * 6.0,
    )
    regression_score = 26.0
    if metadata.get("active_model_regressed") or active.get("name") != champion.get("name"):
        regression_score += 48.0
    if active_features and champion_features and active_features != champion_features:
        regression_score += 18.0

    scores = {
        "data_drift": drift_score,
        "pipeline_bug": pipeline_score,
        "model_regression": min(95.0, regression_score),
    }
    scores[incident_kind] = min(95.0, scores[incident_kind] + 6.0)
    return scores


def _rationale(
    suspect: str,
    diagnostics: dict[str, Any],
    pipeline: dict[str, Any],
    baseline: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    if suspect == "data_drift":
        return diagnostics["headline"]
    if suspect == "pipeline_bug":
        return pipeline["finding"]
    active = baseline.get("active_model", metadata.get("active_model", {}))
    champion = baseline.get("champion_metadata", {})
    if active.get("feature_columns") and champion.get("feature_columns"):
        return (
            f"Deployed model uses {len(active['feature_columns'])} features; "
            f"champion validated {len(champion['feature_columns'])}."
        )
    return "The deployed estimator differs from the last known healthy recipe."


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default