"""Production incident scenarios for the SherlockML detective workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml.data import FEATURE_COLUMNS, generate_fraud_data
from ml.drift import build_diagnostics
from ml.train import ModelArtifact, evaluate_model, run_baseline, train_model

IncidentKind = Literal["data_drift", "pipeline_bug", "model_regression"]
_ALIASES: dict[str, IncidentKind] = {
    "data_drift": "data_drift",
    "pipeline_bug": "pipeline_bug",
    "feature_pipeline_bug": "pipeline_bug",
    "model_regression": "model_regression",
}


def normalize_incident_kind(kind: str) -> IncidentKind:
    """Normalize dashboard/API vocabulary to the three core scenarios."""

    normalized = kind.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _ALIASES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_ALIASES))
        raise ValueError(f"unsupported incident kind '{kind}'. Supported: {supported}") from error


@dataclass
class IncidentSimulation:
    """All deterministic evidence and inputs required for one investigation."""

    kind: IncidentKind
    seed: int
    training_data: pd.DataFrame
    healthy_validation_data: pd.DataFrame
    incident_data: pd.DataFrame
    repair_training_data: pd.DataFrame
    repair_validation_data: pd.DataFrame
    repair_evaluation_data: pd.DataFrame
    baseline: dict[str, Any]
    active_artifact: ModelArtifact
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def context(self) -> dict[str, Any]:
        """Return a JSON-friendly context for the LangGraph agents and API."""

        baseline = self.baseline
        return {
            "incident_kind": self.kind,
            "seed": self.seed,
            "baseline": {
                "champion": baseline["champion"],
                "champion_metrics": baseline["champion_metrics"],
                "champion_metadata": baseline["champion_metadata"],
                "model_comparison": baseline["model_comparison"],
                "healthy_metrics": baseline["healthy_metrics"],
                "incident_metrics": baseline["incident_metrics"],
                "active_model": baseline["active_model"],
            },
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }


def _inject_pipeline_bug(clean_data: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Simulate an inference-only feature-contract and scaling regression."""

    rng = np.random.default_rng(seed)
    broken = clean_data.copy(deep=True)
    # Version v2 sent normalized numeric values into a model that expects raw
    # values, then mapped production categories to values unseen at training.
    broken["transaction_amount"] = (broken["transaction_amount"] / 1_000).round(4)
    broken["transaction_frequency"] = np.log1p(broken["transaction_frequency"]).round(4)
    broken["customer_age"] = ((broken["customer_age"] - 40) / 20).round(4)
    broken["account_age_days"] = (broken["account_age_days"] / 3_650).round(5)
    broken["merchant_category"] = "unmapped_v2_category"
    broken["location"] = "unknown_region"
    broken.loc[rng.random(len(broken)) < 0.42, "device_type"] = np.nan
    return broken


def _scenario_metadata(kind: IncidentKind) -> dict[str, Any]:
    common = {
        "model_name": "SherlockML Fraud Detection Model",
        "feature_contract": list(FEATURE_COLUMNS),
        "simulation": True,
        "data_policy": "synthetic and deterministic; no customer data",
    }
    stories: dict[IncidentKind, dict[str, Any]] = {
        "data_drift": {
            "title": "The high-velocity population shift",
            "symptom": "Fraud F1 deteriorated after customer behaviour changed.",
            "expected_root_cause": "data_drift",
            "evidence_hint": "transaction_amount and transaction_frequency shift materially.",
        },
        "pipeline_bug": {
            "title": "The broken feature contract",
            "symptom": "A serving preprocessing release silently altered feature values.",
            "expected_root_cause": "pipeline_bug",
            "evidence_hint": "missingness and category/scale changes dominate diagnostics.",
        },
        "model_regression": {
            "title": "The weakened model deployment",
            "symptom": "The active model uses a reduced feature set after a bad release.",
            "expected_root_cause": "model_regression",
            "evidence_hint": "Data is stable; active model metadata is inconsistent with champion.",
        },
    }
    return {**common, **stories[kind]}


def simulate_incident(kind: str = "data_drift", *, seed: int = 2026) -> IncidentSimulation:
    """Build one reproducible end-to-end incident with an honest repair window.

    All timestamps are intentionally ordered:

    * Jan: original training population
    * Feb: healthy validation population
    * Mar/Apr: post-shift labels available for a repair (data-drift only)
    * May: incident evaluation window
    """

    incident_kind = normalize_incident_kind(kind)
    training_data = generate_fraud_data(
        2_200, seed=seed, start_time="2026-01-01", profile="healthy"
    )
    healthy_validation = generate_fraud_data(
        850, seed=seed + 1, start_time="2026-02-10", profile="healthy"
    )
    baseline = run_baseline(training_data, healthy_validation, seed=seed)
    champion = baseline["champion_artifact"]

    if incident_kind == "data_drift":
        incident_data = generate_fraud_data(
            1_100, seed=seed + 2, start_time="2026-05-10", profile="drifted"
        )
        repair_training = pd.concat(
            [
                training_data,
                generate_fraud_data(
                    1_900, seed=seed + 3, start_time="2026-03-01", profile="drifted"
                ),
            ],
            ignore_index=True,
        )
        repair_validation = generate_fraud_data(
            800, seed=seed + 4, start_time="2026-04-10", profile="drifted"
        )
        repair_evaluation = incident_data
        active_artifact = champion
        reference_data = healthy_validation
    elif incident_kind == "pipeline_bug":
        clean_incident = generate_fraud_data(
            1_100, seed=seed + 2, start_time="2026-05-10", profile="healthy"
        )
        incident_data = _inject_pipeline_bug(clean_incident, seed=seed + 3)
        repair_training = training_data
        repair_validation = healthy_validation
        repair_evaluation = clean_incident
        active_artifact = champion
        reference_data = healthy_validation
    else:
        incident_data = generate_fraud_data(
            1_100, seed=seed + 2, start_time="2026-05-10", profile="healthy"
        )
        # A plausible bad release: a lightweight logistic model was deployed
        # with only two columns and a stale threshold, bypassing the champion.
        active_artifact = train_model(
            training_data,
            healthy_validation,
            model_kind="logistic_regression",
            feature_columns=("transaction_amount", "customer_age"),
            decision_threshold=0.82,
            seed=seed + 9,
            name="Regressed two-feature logistic model",
        )
        repair_training = training_data
        repair_validation = healthy_validation
        repair_evaluation = incident_data
        reference_data = healthy_validation

    healthy_metrics = evaluate_model(champion, healthy_validation).to_dict()
    incident_metrics = evaluate_model(active_artifact, incident_data).to_dict()
    baseline["healthy_metrics"] = healthy_metrics
    baseline["incident_metrics"] = incident_metrics
    baseline["active_model"] = active_artifact.metadata()
    diagnostics = build_diagnostics(reference_data, incident_data)
    metadata = _scenario_metadata(incident_kind)
    metadata.update(
        {
            "time_windows": {
                "training_end": str(training_data["event_time"].max()),
                "validation_start": str(healthy_validation["event_time"].min()),
                "incident_start": str(incident_data["event_time"].min()),
            },
            "active_model_regressed": active_artifact.name != champion.name,
        }
    )
    return IncidentSimulation(
        kind=incident_kind,
        seed=seed,
        training_data=training_data,
        healthy_validation_data=healthy_validation,
        incident_data=incident_data,
        repair_training_data=repair_training,
        repair_validation_data=repair_validation,
        repair_evaluation_data=repair_evaluation,
        baseline=baseline,
        active_artifact=active_artifact,
        diagnostics=diagnostics,
        metadata=metadata,
    )
