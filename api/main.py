"""Local FastAPI surface for SherlockML's deterministic incident command room."""

from __future__ import annotations

import json
import os
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from agents.graph import (
    SUPPORTED_INCIDENTS,
    normalize_incident_type,
    reset_local_state,
    run_case,
    stream_case,
    to_dashboard_contract,
)

app = FastAPI(
    title="SherlockML Incident API",
    version="0.2.0",
    description=(
        "A local, deterministic ML-reliability investigation interface. "
        "Agents compute evidence-led scores, repairs honour the pipeline contract, "
        "and recovery approval always requires human review."
    ),
)

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

_NODE_LABELS = {
    "observe": "Collecting evidence",
    "statistician": "Running PSI/KS diagnostics",
    "infra": "Checking serving telemetry",
    "war_room": "Debating root cause",
    "engineer": "Writing contract repair",
    "experiment": "Training and validating candidate",
    "escalate": "Recording validation hold",
    "doctor": "Prescribing treatment",
    "tracker": "Logging experiment",
    "report": "Generating recovery report",
}


def _strict_gates_enabled() -> bool:
    return os.environ.get("SHERLOCKML_STRICT_GATES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def run_investigation(incident_type: str, *, strict_gates: bool | None = None) -> dict[str, Any]:
    """Run one full case and return the dashboard's documented response shape."""

    try:
        canonical = normalize_incident_type(incident_type)
    except (AttributeError, ValueError) as error:
        raise ValueError(str(error)) from error

    use_strict = _strict_gates_enabled() if strict_gates is None else strict_gates
    with _investigation_lock:
        reset_local_state()
        state = run_case(canonical, strict_gates=use_strict)
        return to_dashboard_contract(state)


def stream_investigation(
    incident_type: str, *, strict_gates: bool | None = None
) -> StreamingResponse:
    """Stream node-by-node progress as newline-delimited JSON events."""

    try:
        canonical = normalize_incident_type(incident_type)
    except (AttributeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    use_strict = _strict_gates_enabled() if strict_gates is None else strict_gates

    def event_generator():
        with _investigation_lock:
            reset_local_state()
            state: dict[str, Any] = {
                "incident_type": canonical,
                "timeline": [],
                "strict_gates": use_strict,
            }
            for node_name, update in stream_case(canonical, strict_gates=use_strict):
                state.update(update)
                payload = {
                    "node": node_name,
                    "label": _NODE_LABELS.get(node_name, node_name),
                    "timeline": state.get("timeline", []),
                }
                yield json.dumps(payload) + "\n"
            yield json.dumps(
                {"node": "complete", "case": to_dashboard_contract(state)}
            ) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


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
        "strict_gates_env": "SHERLOCKML_STRICT_GATES",
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
def investigate_incident(incident_type: str, strict_gates: bool = False) -> dict[str, Any]:
    """Execute a bounded investigation for a declared synthetic incident."""

    try:
        return run_investigation(incident_type, strict_gates=strict_gates)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"SherlockML could not complete the investigation: {error.__class__.__name__}",
        ) from error


@app.post("/api/incidents/{incident_type}/investigate/stream")
def investigate_incident_stream(
    incident_type: str, strict_gates: bool = False
) -> StreamingResponse:
    """Stream investigation progress for live dashboard updates."""

    return stream_investigation(incident_type, strict_gates=strict_gates)


@app.post("/api/reset")
def reset() -> dict[str, str]:
    """Restore the feature contract to the deterministic baseline."""

    return reset_state()
