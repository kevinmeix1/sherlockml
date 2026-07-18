"""Offline-first experiment tracking with a local MLflow SQLite backend."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "artifacts" / "experiments"
MLFLOW_DATABASE = ROOT / "artifacts" / "mlflow_tracking.db"


def track_experiment(
    case_id: str,
    incident_kind: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    changes: list[str],
) -> dict[str, Any]:
    """Persist a portable experiment record and use MLflow if it is available."""
    payload = {
        "case_id": case_id,
        "incident_kind": incident_kind,
        "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        "baseline": baseline,
        "candidate": candidate,
        "changes": changes,
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = EXPERIMENT_DIR / f"{case_id.lower()}-experiment.json"
    local_path.write_text(json.dumps(payload, indent=2) + "\n")

    try:
        import mlflow

        MLFLOW_DATABASE.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{MLFLOW_DATABASE}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("sherlockml-recovery")
        with mlflow.start_run(run_name=f"{case_id}-candidate") as run:
            mlflow.log_params({"incident_kind": incident_kind, "change_count": len(changes)})
            for prefix, metrics in (("baseline", baseline), ("candidate", candidate)):
                for name in ("accuracy", "f1", "precision", "recall", "latency_ms"):
                    if name in metrics:
                        mlflow.log_metric(f"{prefix}_{name}", float(metrics[name]))
            run_id = run.info.run_id
        return {
            "backend": "mlflow-sqlite",
            "run_id": run_id,
            "local_record": str(local_path),
            "tracking_uri": tracking_uri,
        }
    except Exception as error:  # The portable JSON record remains the reliable demo artifact.
        return {
            "backend": "local-json-fallback",
            "run_id": None,
            "local_record": str(local_path),
            "tracking_uri": None,
            "note": f"MLflow unavailable: {error.__class__.__name__}",
        }
