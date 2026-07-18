"""Metric contracts for deterministic model-health comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ModelMetrics:
    """Small serialisable metric record shown in the case file and dashboard."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    positive_rate: float
    predicted_positive_rate: float
    latency_ms: float
    sample_size: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in asdict(self).items()
        }


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    latency_ms: float,
) -> ModelMetrics:
    """Calculate stable classification metrics without warning on edge cases."""

    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(probabilities, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.5
    return ModelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        positive_rate=float(y_true.mean()),
        predicted_positive_rate=float(y_pred.mean()),
        latency_ms=float(latency_ms),
        sample_size=int(len(y_true)),
    )


def metrics_delta(
    before: ModelMetrics | dict[str, Any], after: ModelMetrics | dict[str, Any]
) -> dict[str, float]:
    """Return the directional recovery delta used by the Doctor and report."""

    before_values = before.to_dict() if isinstance(before, ModelMetrics) else before
    after_values = after.to_dict() if isinstance(after, ModelMetrics) else after
    fields = ("accuracy", "precision", "recall", "f1", "roc_auc", "latency_ms")
    return {
        field: round(float(after_values[field]) - float(before_values[field]), 4)
        for field in fields
    }
