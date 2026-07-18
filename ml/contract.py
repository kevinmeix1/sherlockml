"""Load and apply the versioned pipeline contract used by training and repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = ROOT / "models" / "pipeline_contract.json"


DEFAULT_CONTRACT: dict[str, Any] = {
    "version": "fraud-v1.7",
    "selected_features": [
        "transaction_amount",
        "transaction_frequency",
        "customer_age",
        "account_age_days",
    ],
    "preprocessing": {"scaler": "standard", "impute_missing": "median", "parity_check": False},
    "training_window_days": 14,
    "estimator": {"champion": "xgboost", "max_depth": 2, "learning_rate": 0.05},
    "release_gate": {"minimum_f1": 0.78, "minimum_recall": 0.72},
}


def load_pipeline_contract(path: Path | None = None) -> dict[str, Any]:
    """Return the current pipeline contract, creating the demo default if needed."""

    contract_path = path or DEFAULT_CONTRACT_PATH
    if not contract_path.exists():
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(DEFAULT_CONTRACT, indent=2) + "\n")
    return json.loads(contract_path.read_text())


def contract_feature_columns(contract: dict[str, Any]) -> tuple[str, ...]:
    """Feature columns declared by the contract, validated against the schema."""

    raw = contract.get("selected_features", list(FEATURE_COLUMNS))
    columns = tuple(str(feature) for feature in raw)
    unknown = set(columns).difference(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"contract references unknown features: {', '.join(sorted(unknown))}")
    if not columns:
        raise ValueError("contract must declare at least one selected feature")
    return columns


def filter_training_window(
    frame: pd.DataFrame,
    window_days: int,
    *,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Keep only rows inside the contract's labelled training window."""

    if window_days <= 0 or time_column not in frame.columns:
        return frame
    latest = pd.Timestamp(frame[time_column].max())
    cutoff = latest - pd.Timedelta(days=int(window_days))
    return frame.loc[frame[time_column] >= cutoff].copy()


def estimator_overrides(contract: dict[str, Any]) -> dict[str, Any]:
    """Pass hyperparameters from the contract into the training module."""

    estimator = contract.get("estimator", {})
    overrides: dict[str, Any] = {}
    for key in ("max_depth", "learning_rate", "n_estimators"):
        if key in estimator:
            overrides[key] = estimator[key]
    return overrides


def contract_model_kind(contract: dict[str, Any]) -> str:
    return str(contract.get("estimator", {}).get("champion", "xgboost"))
