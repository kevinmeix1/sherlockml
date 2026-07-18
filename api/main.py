"""Local FastAPI surface for SherlockML's deterministic incident command room."""

from __future__ import annotations

from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException

from agents.graph import (
    SUPPORTED_INCIDENTS,
    normalize_incident_type,
    reset_local_state,
    run_case,
    to_dashboard_contract,
)

app = FastAPI(
    title="SherlockML Incident API",
    version="0.1.0",
    description=(
        "A local, deterministic ML-reliability investigation interface. "
        "A recovery approval always requires human review."
    ),
)

# The demo deliberately mutates one local pipeline-contract file while the
# engineer prepares a reviewable repair. Serialising runs makes that mutation
# deterministic and avoids one browser click overwriting another case's diff.
_investigation_lock = RLock()

_INCIDENT_CATALOG = {
    "data_drift": {
        "label": "Data drift",
        "description": "A population shift degrades the deployed fraud model.",
    },
    "pipeline_bug": {
        "label": "Feature pipeline bug",
        "description": "A serving transformation no longer matches the trained contract.",
    },
    "model_regression": {
        "label": "Model regression",
        "description": "A weak release recipe replaces the validated model path.",
    },
}


def run_investigation(incident_type: str) -> dict[str, Any]:
    """Run one full case and return the dashboard's documented response shape.

    This function intentionally remains synchronous so Streamlit can import it
    directly without requiring an HTTP process.  The same function backs the
    FastAPI endpoint below.
    """

    try:
        canonical = normalize_incident_type(incident_type)
    except (AttributeError, ValueError) as error:
        raise ValueError(str(error)) from error

    with _investigation_lock:
        # Every run begins from the same feature-contract baseline, then leaves
        # its repair and diff in local artifacts for the operator to inspect.
        reset_local_state()
        state = run_case(canonical)
        return to_dashboard_contract(state)


def reset_state() -> dict[str, str]:
    """Restore the deterministic input contract without deleting evidence."""

    with _investigation_lock:
        return reset_local_state()


def reset_demo() -> dict[str, str]:
    """Dashboard-friendly alias for :func:`reset_state`."""

    return reset_state()


@app.get("/health")
def health() -> dict[str, Any]:
    """Cheap local liveness endpoint; it does not run an ML experiment."""

    return {
        "status": "ok",
        "service": "sherlockml",
        "mode": "deterministic-local",
        "requires_human_approval": True,
        "supported_incidents": list(SUPPORTED_INCIDENTS),
    }


@app.get("/api/incidents")
def incidents() -> dict[str, Any]:
    """Expose named, safe-to-demo scenarios to the dashboard or a judge."""

    return {
        "incidents": [
            {"id": identifier, **_INCIDENT_CATALOG[identifier]}
            for identifier in SUPPORTED_INCIDENTS
        ]
    }


@app.post("/api/incidents/{incident_type}/investigate")
def investigate_incident(incident_type: str) -> dict[str, Any]:
    """Execute a bounded investigation for a declared synthetic incident."""

    try:
        return run_investigation(incident_type)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:  # Preserve the exception type without leaking tracebacks.
        raise HTTPException(
            status_code=500,
            detail=f"SherlockML could not complete the investigation: {error.__class__.__name__}",
        ) from error


@app.post("/api/reset")
def reset() -> dict[str, str]:
    """Restore the feature contract to the deterministic baseline."""

    return reset_state()
