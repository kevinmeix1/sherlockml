"""Training and recovery experiments for the SherlockML fraud model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .evaluation import ModelMetrics, calculate_metrics, metrics_delta

if TYPE_CHECKING:
    from simulator.incidents import IncidentSimulation

ModelKind = Literal["logistic_regression", "xgboost"]


@dataclass
class ModelArtifact:
    """A trained model plus the contract needed to safely score it."""

    estimator: Pipeline
    name: str
    feature_columns: tuple[str, ...]
    decision_threshold: float
    engine: str
    estimated_latency_ms: float
    training_rows: int
    preprocessing_version: str = "fraud-features-v1"

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "feature_columns": list(self.feature_columns),
            "decision_threshold": round(self.decision_threshold, 3),
            "estimated_latency_ms": self.estimated_latency_ms,
            "training_rows": self.training_rows,
            "preprocessing_version": self.preprocessing_version,
        }


def _one_hot_encoder() -> OneHotEncoder:
    """Support both current and older scikit-learn releases."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - only reached with older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_groups(feature_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    requested = tuple(feature_columns)
    numeric = [feature for feature in requested if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in requested if feature in CATEGORICAL_FEATURES]
    unknown = set(requested).difference(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if unknown:
        raise ValueError(f"unsupported feature columns: {', '.join(sorted(unknown))}")
    if not numeric and not categorical:
        raise ValueError("at least one supported feature column is required")
    return numeric, categorical


def _preprocessor(feature_columns: Iterable[str]) -> ColumnTransformer:
    numeric, categorical = _feature_groups(feature_columns)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _xgboost_or_fallback(seed: int) -> tuple[Any, str, float]:
    """Build XGBoost when available, with an explicit deterministic fallback."""

    try:
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                n_estimators=110,
                max_depth=4,
                learning_rate=0.07,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
            ),
            "xgboost",
            18.6,
        )
    except ImportError:
        return (
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=150,
                max_leaf_nodes=24,
                l2_regularization=0.1,
                random_state=seed,
            ),
            "hist_gradient_boosting_fallback",
            15.2,
        )


def _make_estimator(model_kind: ModelKind, seed: int) -> tuple[Any, str, float]:
    if model_kind == "logistic_regression":
        return (
            LogisticRegression(
                class_weight="balanced", max_iter=800, random_state=seed, solver="lbfgs"
            ),
            "scikit_learn_logistic_regression",
            8.4,
        )
    if model_kind == "xgboost":
        return _xgboost_or_fallback(seed)
    raise ValueError(f"unknown model kind: {model_kind}")


def _validate_training_frame(frame: pd.DataFrame, feature_columns: Iterable[str]) -> None:
    needed = set(feature_columns) | {"fraud_label"}
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise ValueError(f"training data is missing required columns: {', '.join(missing)}")
    if frame["fraud_label"].nunique() < 2:
        raise ValueError("training data must contain both fraud classes")


def _choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Pick a validation F1 threshold deterministically, favouring recall on ties."""

    candidates = np.linspace(0.10, 0.90, 65)
    best: tuple[float, float, float] = (-1.0, -1.0, 0.5)  # f1, recall, threshold
    y_true = np.asarray(labels, dtype=int)
    for threshold in candidates:
        metrics = calculate_metrics(y_true, probabilities, threshold=float(threshold), latency_ms=0)
        candidate = (metrics.f1, metrics.recall, -float(threshold))
        # The negative threshold produces a stable preference for a lower
        # operational threshold where F1 and recall are exactly tied.
        if candidate > (best[0], best[1], -best[2]):
            best = (metrics.f1, metrics.recall, float(threshold))
    return best[2]


def train_model(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    model_kind: ModelKind = "xgboost",
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    seed: int = 2026,
    decision_threshold: float | None = None,
    name: str | None = None,
) -> ModelArtifact:
    """Train a production-style feature pipeline and tune its validation threshold."""

    features = tuple(feature_columns)
    _validate_training_frame(training_data, features)
    _validate_training_frame(validation_data, features)
    estimator, engine, latency = _make_estimator(model_kind, seed)
    pipeline = Pipeline(
        [("preprocess", _preprocessor(features)), ("classifier", estimator)]
    )
    pipeline.fit(training_data.loc[:, features], training_data["fraud_label"].astype(int))
    validation_probabilities = pipeline.predict_proba(validation_data.loc[:, features])[:, 1]
    threshold = (
        float(decision_threshold)
        if decision_threshold is not None
        else _choose_threshold(validation_probabilities, validation_data["fraud_label"].to_numpy())
    )
    return ModelArtifact(
        estimator=pipeline,
        name=name or model_kind,
        feature_columns=features,
        decision_threshold=threshold,
        engine=engine,
        estimated_latency_ms=latency,
        training_rows=len(training_data),
    )


def evaluate_model(artifact: ModelArtifact, data: pd.DataFrame) -> ModelMetrics:
    """Score an artifact while enforcing its input-schema contract."""

    missing = sorted(set(artifact.feature_columns).difference(data.columns))
    if missing:
        raise ValueError(f"prediction data is missing required columns: {', '.join(missing)}")
    if "fraud_label" not in data:
        raise ValueError("prediction data is missing required column: fraud_label")
    probabilities = artifact.estimator.predict_proba(data.loc[:, artifact.feature_columns])[:, 1]
    return calculate_metrics(
        data["fraud_label"].to_numpy(),
        probabilities,
        threshold=artifact.decision_threshold,
        latency_ms=artifact.estimated_latency_ms,
    )


def _comparison_row(artifact: ModelArtifact, metrics: ModelMetrics) -> dict[str, Any]:
    return {**artifact.metadata(), **metrics.to_dict()}


def run_baseline(
    training_data: pd.DataFrame, validation_data: pd.DataFrame, *, seed: int = 2026
) -> dict[str, Any]:
    """Train and compare the required Logistic Regression and XGBoost tracks.

    The returned result is deliberately mixed: serialisable fields are safe for
    API/dashboard use and ``champion_artifact`` is an in-memory handoff for the
    agent graph and experiment runner.
    """

    logistic = train_model(
        training_data,
        validation_data,
        model_kind="logistic_regression",
        seed=seed,
        name="Logistic Regression",
    )
    boosted = train_model(
        training_data,
        validation_data,
        model_kind="xgboost",
        seed=seed,
        name="XGBoost",
    )
    logistic_metrics = evaluate_model(logistic, validation_data)
    boosted_metrics = evaluate_model(boosted, validation_data)
    # F1 is the primary fraud-operation score; ROC-AUC breaks a rare tie.
    champion, champion_metrics = max(
        ((logistic, logistic_metrics), (boosted, boosted_metrics)),
        key=lambda pair: (pair[1].f1, pair[1].roc_auc),
    )
    comparison = [
        _comparison_row(logistic, logistic_metrics),
        _comparison_row(boosted, boosted_metrics),
    ]
    return {
        "model_comparison": comparison,
        "champion": champion.name,
        "champion_metrics": champion_metrics.to_dict(),
        "champion_metadata": champion.metadata(),
        "champion_artifact": champion,
        "artifacts": {"logistic_regression": logistic, "xgboost": boosted},
    }


def run_repair_experiment(simulation: IncidentSimulation) -> dict[str, Any]:
    """Apply the incident-specific recovery treatment and prove the outcome.

    No labels from the current incident window are used for candidate training.
    ``repair_training_data`` and ``repair_validation_data`` model a prior,
    newly labelled time window.  This keeps the demo's recovery claim honest.
    """

    before = evaluate_model(simulation.active_artifact, simulation.incident_data)
    kind = simulation.kind
    if kind == "pipeline_bug":
        candidate = simulation.baseline["champion_artifact"]
        after = evaluate_model(candidate, simulation.repair_evaluation_data)
        proposed_fix = {
            "type": "pipeline_contract_repair",
            "changes": [
                "Restore the versioned preprocessor used during training.",
                "Reject unknown feature-schema versions before scoring.",
                "Impute missing values inside the persisted sklearn pipeline.",
            ],
            "code_change": "pipeline_version: v2-bugged → fraud-features-v1",
        }
    else:
        candidate_baseline = run_baseline(
            simulation.repair_training_data,
            simulation.repair_validation_data,
            seed=simulation.seed + 17,
        )
        candidate = candidate_baseline["champion_artifact"]
        after = evaluate_model(candidate, simulation.repair_evaluation_data)
        if kind == "data_drift":
            proposed_fix = {
                "type": "drift_aware_retraining",
                "changes": [
                    "Add the latest labelled post-shift transaction window.",
                    "Retune the fraud decision threshold on the recent validation window.",
                    "Enable PSI and KS alerts on all seven serving features.",
                ],
                "code_change": (
                    "training_window: historical-only → historical + labelled post-shift window"
                ),
            }
        else:
            proposed_fix = {
                "type": "model_regression_rollback",
                "changes": [
                    "Replace the two-feature regressed model with the validated feature pipeline.",
                    "Restore champion-selection gates for F1 and ROC-AUC.",
                    "Block deployment when candidate F1 is below the healthy baseline.",
                ],
                "code_change": (
                    "model_spec: amount/customer_age logistic → validated champion pipeline"
                ),
            }

    before_dict = before.to_dict()
    after_dict = after.to_dict()
    improvement = metrics_delta(before, after)
    approved = bool(
        improvement["f1"] >= 0.08
        and after.f1 >= 0.58
        and after.recall >= 0.60
    )
    return {
        "incident_kind": kind,
        "proposed_fix": proposed_fix,
        "before": before_dict,
        "after": after_dict,
        "improvement": improvement,
        "approved": approved,
        "approval_reason": (
            "Candidate clears recovery gates: +0.08 F1, F1 ≥ 0.58, and recall ≥ 0.60."
            if approved
            else "Candidate did not clear the deterministic recovery gate; human review required."
        ),
        "candidate_metadata": candidate.metadata(),
        "candidate_artifact": candidate,
        "experiment_rows": [
            {"version": "v1", "change": "incident state", **before_dict},
            {"version": "v2", "change": proposed_fix["type"], **after_dict},
        ],
    }
