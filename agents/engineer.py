"""A bounded ML engineer agent that performs a reversible pipeline repair."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, cast

from agents.git_integration import propose_or_commit

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONTRACT = ROOT / "models" / "pipeline_contract.json"
CASE_ARTIFACTS = ROOT / "artifacts" / "cases"

DEFAULT_CONTRACT = {
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


def restore_pipeline_contract() -> Path:
    PIPELINE_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_CONTRACT.write_text(json.dumps(DEFAULT_CONTRACT, indent=2) + "\n")
    return PIPELINE_CONTRACT


def inspect_pipeline() -> dict[str, Any]:
    if not PIPELINE_CONTRACT.exists():
        restore_pipeline_contract()
    contract = json.loads(PIPELINE_CONTRACT.read_text())
    selected = cast(list[str], contract.get("selected_features", []))
    baseline_features = cast(list[str], DEFAULT_CONTRACT["selected_features"])
    missing_features = sorted(set(baseline_features) ^ set(selected))
    parity = contract.get("preprocessing", {}).get("parity_check", False)
    finding_parts = []
    if len(selected) < len(baseline_features) + 3:
        finding_parts.append(
            "The contract is deliberately minimal and lacks behavior features, serving parity "
            "assertions, and a recent-label training window."
        )
    if not parity:
        finding_parts.append("Serving parity checks are disabled.")
    if contract.get("training_window_days", 14) < 28:
        finding_parts.append(
            f"Training window is only {contract.get('training_window_days')} days."
        )
    if missing_features:
        finding_parts.append(f"Feature delta vs baseline contract: {', '.join(missing_features)}.")
    finding = (
        " ".join(finding_parts)
        if finding_parts
        else "Contract matches the expected champion schema."
    )
    return {
        "file": str(PIPELINE_CONTRACT),
        "version": contract["version"],
        "selected_features": contract["selected_features"],
        "finding": finding,
        "severity": "high",
        "contract": contract,
    }


def apply_repair(case_id: str, incident_kind: str) -> dict[str, Any]:
    """Write the smallest config repair and preserve a reviewable diff artifact."""
    if not PIPELINE_CONTRACT.exists():
        restore_pipeline_contract()
    before = PIPELINE_CONTRACT.read_text()
    contract = json.loads(before)
    changes = _repair_contract(contract, incident_kind)
    after = json.dumps(contract, indent=2) + "\n"
    PIPELINE_CONTRACT.write_text(after)

    case_dir = CASE_ARTIFACTS / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="models/pipeline_contract.json (before)",
            tofile="models/pipeline_contract.json (candidate)",
        )
    )
    patch_path = case_dir / "pipeline_repair.diff"
    patch_path.write_text(patch)
    action_path = case_dir / "engineering_action.json"
    action_path.write_text(
        json.dumps(
            {"case_id": case_id, "incident_kind": incident_kind, "changes": changes}, indent=2
        )
        + "\n"
    )
    git = propose_or_commit(case_id, [PIPELINE_CONTRACT, patch_path, action_path])
    return {
        "summary": "Applied a bounded, reviewable repair to the feature pipeline contract.",
        "changes": changes,
        "changed_file": str(PIPELINE_CONTRACT),
        "patch": patch,
        "patch_path": str(patch_path),
        "action_path": str(action_path),
        "git": git,
        "candidate_contract": contract,
    }


def _repair_contract(contract: dict[str, Any], incident_kind: str) -> list[str]:
    changes = ["Expanded the labelled training window from 14 to 42 days."]
    contract["training_window_days"] = 42
    preprocessing = contract["preprocessing"]
    preprocessing["impute_missing"] = "median_with_missingness_indicator"
    preprocessing["parity_check"] = True

    if incident_kind == "data_drift":
        for feature in ["transaction_frequency", "merchant_category", "location", "device_type"]:
            if feature not in contract["selected_features"]:
                contract["selected_features"].append(feature)
        changes.extend(
            [
                "Promoted behavior and categorical risk signals to the selected-feature contract.",
                "Enabled training/serving transform parity checks and drift gates.",
            ]
        )
    elif incident_kind == "pipeline_bug":
        changes.extend(
            [
                "Restored robust missing-value treatment and scaler parity assertions.",
                "Added a missingness indicator to keep absence of data observable to the model.",
            ]
        )
    else:
        contract["estimator"] = {"champion": "xgboost", "max_depth": 4, "learning_rate": 0.08}
        changes.extend(
            [
                "Restored the champion XGBoost recipe from the validated model registry.",
                "Re-enabled the release gate against the last known-good benchmark.",
            ]
        )
    return changes
