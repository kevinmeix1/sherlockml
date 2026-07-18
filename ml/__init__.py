"""Deterministic machine-learning primitives used by SherlockML."""

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, generate_fraud_data
from .drift import build_diagnostics
from .evaluation import ModelMetrics, metrics_delta
from .train import (
    ModelArtifact,
    evaluate_model,
    run_baseline,
    run_repair_experiment,
    train_model,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "ModelArtifact",
    "ModelMetrics",
    "build_diagnostics",
    "evaluate_model",
    "generate_fraud_data",
    "metrics_delta",
    "run_baseline",
    "run_repair_experiment",
    "train_model",
]
