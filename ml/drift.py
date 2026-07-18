"""Explainable drift and data-quality diagnostics for an ML incident."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

_EPSILON = 1e-6


def _as_distribution(values: pd.Series, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values.dropna().astype(float), bins=bins)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / max(len(counts), 1))
    return np.clip(counts / total, _EPSILON, None)


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """Compute PSI using reference quantile bins.

    PSI below .10 is generally stable; .10-.25 is a warning; .25+ is a
    material shift.  The function is deliberately defensive for constant and
    sparse features so a bad payload cannot crash the investigation.
    """

    reference_values = reference.dropna().astype(float)
    current_values = current.dropna().astype(float)
    if reference_values.empty or current_values.empty:
        return 0.0
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected = _as_distribution(reference_values, edges)
    actual = _as_distribution(current_values, edges)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """PSI analogue for category frequency changes."""

    reference_counts = reference.fillna("<missing>").astype(str).value_counts(normalize=True)
    current_counts = current.fillna("<missing>").astype(str).value_counts(normalize=True)
    categories = reference_counts.index.union(current_counts.index)
    expected = np.clip(
        reference_counts.reindex(categories, fill_value=0).to_numpy(), _EPSILON, None
    )
    actual = np.clip(current_counts.reindex(categories, fill_value=0).to_numpy(), _EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _severity(psi: float, ks_pvalue: float) -> str:
    if psi >= 0.25 or ks_pvalue < 0.001:
        return "critical"
    if psi >= 0.10 or ks_pvalue < 0.05:
        return "warning"
    return "stable"


def _ensure_features(frame: pd.DataFrame, features: Iterable[str]) -> None:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"diagnostic frame is missing required features: {', '.join(missing)}")


def build_diagnostics(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    target_column: str = "fraud_label",
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
) -> dict[str, Any]:
    """Create a serialisable evidence payload for SherlockML's agents.

    It intentionally contains both signal diagnostics (PSI/KS/correlation) and
    pipeline-health checks (missingness/class balance), because a visual drift
    chart alone cannot distinguish data drift from a broken feature pipeline.
    """

    numeric = tuple(numeric_features)
    categorical = tuple(categorical_features)
    _ensure_features(reference, numeric + categorical)
    _ensure_features(current, numeric + categorical)

    feature_diagnostics: list[dict[str, Any]] = []
    for feature in numeric:
        ref_values = reference[feature].dropna().astype(float)
        cur_values = current[feature].dropna().astype(float)
        if ref_values.empty or cur_values.empty:
            ks_statistic, ks_pvalue = 0.0, 1.0
        else:
            ks = ks_2samp(ref_values, cur_values)
            ks_statistic, ks_pvalue = float(ks.statistic), float(ks.pvalue)
        psi = population_stability_index(reference[feature], current[feature])
        reference_mean = float(ref_values.mean()) if not ref_values.empty else 0.0
        current_mean = float(cur_values.mean()) if not cur_values.empty else 0.0
        relative_mean_shift = (current_mean - reference_mean) / max(abs(reference_mean), _EPSILON)
        feature_diagnostics.append(
            {
                "feature": feature,
                "type": "numeric",
                "psi": round(psi, 4),
                "ks_statistic": round(ks_statistic, 4),
                "ks_pvalue": round(ks_pvalue, 6),
                "reference_mean": round(reference_mean, 4),
                "current_mean": round(current_mean, 4),
                "relative_mean_shift": round(float(relative_mean_shift), 4),
                "missing_rate_reference": round(float(reference[feature].isna().mean()), 4),
                "missing_rate_current": round(float(current[feature].isna().mean()), 4),
                "severity": _severity(psi, ks_pvalue),
            }
        )

    for feature in categorical:
        psi = categorical_population_stability_index(reference[feature], current[feature])
        feature_diagnostics.append(
            {
                "feature": feature,
                "type": "categorical",
                "psi": round(psi, 4),
                "ks_statistic": None,
                "ks_pvalue": None,
                "reference_mean": None,
                "current_mean": None,
                "relative_mean_shift": None,
                "missing_rate_reference": round(float(reference[feature].isna().mean()), 4),
                "missing_rate_current": round(float(current[feature].isna().mean()), 4),
                "severity": "critical" if psi >= 0.25 else "warning" if psi >= 0.10 else "stable",
            }
        )

    correlation_changes: list[dict[str, Any]] = []
    reference_corr = reference.loc[:, numeric].corr(numeric_only=True)
    current_corr = current.loc[:, numeric].corr(numeric_only=True)
    for first, second in combinations(numeric, 2):
        before = float(reference_corr.loc[first, second])
        after = float(current_corr.loc[first, second])
        correlation_changes.append(
            {
                "pair": f"{first} ↔ {second}",
                "reference": round(before, 4),
                "current": round(after, 4),
                "absolute_change": round(abs(after - before), 4),
            }
        )
    correlation_changes.sort(key=lambda item: item["absolute_change"], reverse=True)

    def _class_balance(frame: pd.DataFrame) -> dict[str, Any]:
        if target_column not in frame:
            return {"available": False}
        prevalence = float(frame[target_column].astype(int).mean())
        return {
            "available": True,
            "positive_rate": round(prevalence, 4),
            "negative_rate": round(1 - prevalence, 4),
            "imbalance_ratio": round((1 - prevalence) / max(prevalence, _EPSILON), 3),
        }

    feature_diagnostics.sort(
        key=lambda item: (float(item["psi"]), float(item["missing_rate_current"])), reverse=True
    )
    critical = [item["feature"] for item in feature_diagnostics if item["severity"] == "critical"]
    warnings = [item["feature"] for item in feature_diagnostics if item["severity"] == "warning"]
    strongest = feature_diagnostics[0] if feature_diagnostics else None

    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "feature_diagnostics": feature_diagnostics,
        "correlation_changes": correlation_changes[:5],
        "class_imbalance": {
            "reference": _class_balance(reference),
            "current": _class_balance(current),
        },
        "summary": {
            "critical_features": critical,
            "warning_features": warnings,
            "strongest_evidence": strongest["feature"] if strongest else None,
            "drift_detected": bool(critical or warnings),
            "feature_count_checked": len(FEATURE_COLUMNS),
        },
    }
