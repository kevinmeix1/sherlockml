"""SherlockML's interactive incident-command dashboard.

Run from the repository root:

    streamlit run dashboard/app.py

The dashboard deliberately owns presentation state only.  It calls
``api.main.run_investigation(incident_type)`` when that engine is available,
then normalises its result into a stable view model.  This lets the UI remain
usable while the orchestration layer is evolving, without pretending that a
preview case is a live investigation.
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

try:  # Charts degrade to an explanatory empty state when plotly is absent.
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover - exercised only in minimal environments
    go = None
    _HAS_PLOTLY = False

APP_TITLE = "SherlockML | ML Incident Command"
PAGES = (
    "Model Health",
    "Case File",
    "War Room",
    "Model Doctor",
    "Recovery Results",
)
PAGE_ICONS = {
    "Model Health": "🫀",
    "Case File": "🗂️",
    "War Room": "⚔️",
    "Model Doctor": "🩺",
    "Recovery Results": "📈",
}
INCIDENTS = {
    "data_drift": "Data drift",
    "feature_pipeline_bug": "Feature pipeline bug",
    "model_regression": "Model regression",
}
INCIDENT_HINTS = {
    "data_drift": "Customer behaviour shifts under a frozen model.",
    "feature_pipeline_bug": "A serving transform silently breaks the feature contract.",
    "model_regression": "A weak release recipe replaces the validated champion.",
}

_PALETTE = {
    "mint": "#5de0b5",
    "coral": "#ff7462",
    "gold": "#f2c86f",
    "blue": "#89c7ff",
    "muted": "#9caab0",
    "grid": "rgba(190, 215, 211, .10)",
}
_PLOTLY_CONFIG = {"displayModeBar": False}

_STEPS = (
    ("ALERT", "🚨"),
    ("EVIDENCE", "🔍"),
    ("DEBATE", "⚖️"),
    ("REPAIR", "🔧"),
    ("VERIFY", "🧪"),
    ("REPORT", "📋"),
)
_PHASE_TO_STEP = {
    "ALERT": 0,
    "COLLECT": 1,
    "ANALYZE": 1,
    "RULE OUT": 1,
    "DEBATE": 2,
    "REPAIR": 3,
    "TREAT": 3,
    "VERIFY": 4,
    "DECIDE": 4,
    "RECORD": 5,
    "REPORT": 5,
}

_AGENT_ICONS = (
    ("sherlock", "🕵️"),
    ("detective", "🕵️"),
    ("ada", "📊"),
    ("statist", "📊"),
    ("linus", "👨‍💻"),
    ("engineer", "👨‍💻"),
    ("sre", "☁️"),
    ("infra", "☁️"),
    ("moriarty", "⚖️"),
    ("moderator", "⚖️"),
    ("doctor", "🩺"),
)


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")


def _agent_icon(name: str) -> str:
    lowered = name.lower()
    for token, icon in _AGENT_ICONS:
        if token in lowered:
            return icon
    return "🤖"


def _preview_case(incident_type: str) -> dict[str, Any]:
    """A clearly labelled UI-only preview used before the API exists."""

    labels = {
        "data_drift": {
            "title": "Feature distribution shift",
            "cause": "Transaction amount and frequency drifted beyond the model's training envelope.",
            "treatment": "Recalibrate scaling, retrain on recent labelled traffic, and install PSI guardrails.",
        },
        "feature_pipeline_bug": {
            "title": "Feature transformation defect",
            "cause": "The production preprocessing contract no longer matches the training transform.",
            "treatment": "Restore the feature contract, add a schema gate, and retrain a verified artifact.",
        },
        "model_regression": {
            "title": "Model release regression",
            "cause": "A recent model configuration change reduced recall on high-risk transactions.",
            "treatment": "Roll back the unsafe configuration, retrain the candidate, and gate promotion on recall.",
        },
    }
    selected = labels.get(incident_type, labels["data_drift"])
    return {
        "source": "dashboard-preview",
        "case_id": "PREVIEW-001",
        "incident_type": incident_type,
        "incident_title": selected["title"],
        "status": "Preview — engine not connected",
        "health": {
            "before": 77.0,
            "after_incident": 49.0,
            "recovered": 78.0,
            "status": "critical",
            "model_name": "Fraud Sentinel / xgboost-v17",
            "latency_ms": 112.0,
        },
        "evidence": [
            {
                "id": "E-01",
                "title": "Performance signal",
                "detail": "F1 crossed the intervention threshold after a production distribution shift.",
                "strength": "high",
                "source": "production metrics",
            },
            {
                "id": "E-02",
                "title": "Feature contract",
                "detail": "A feature-level integrity check shows a mismatch between train and serve paths.",
                "strength": "high",
                "source": "pipeline diff",
            },
            {
                "id": "E-03",
                "title": "Release activity",
                "detail": "The latest release is in the incident window and needs causal verification.",
                "strength": "medium",
                "source": "git history",
            },
        ],
        "suspects": [
            {
                "name": "Feature distribution shift",
                "confidence": 86,
                "verdict": "primary",
                "rationale": "Drift statistic exceeds the configured alert threshold.",
            },
            {
                "name": "Preprocessing mismatch",
                "confidence": 68,
                "verdict": "contributing",
                "rationale": "The serving feature contract differs from the trained artifact.",
            },
            {
                "name": "Release configuration change",
                "confidence": 31,
                "verdict": "ruled out",
                "rationale": "Timing is suspicious, but the experiment does not reproduce it alone.",
            },
        ],
        "timeline": [
            {
                "time": "10:01",
                "phase": "ALERT",
                "text": "Production monitor detected a reliability breach.",
            },
            {
                "time": "10:02",
                "phase": "COLLECT",
                "text": "Detective gathered metrics, logs, data profiles, and release metadata.",
            },
            {
                "time": "10:03",
                "phase": "DEBATE",
                "text": "War Room compared competing root-cause hypotheses.",
            },
            {
                "time": "10:05",
                "phase": "TREAT",
                "text": "Model Doctor drafted a reversible recovery prescription.",
            },
            {
                "time": "10:07",
                "phase": "VERIFY",
                "text": "Experiment Agent validated the candidate against the policy gates.",
            },
        ],
        "war_room": [
            {
                "agent": "Detective",
                "role": "Lead investigator",
                "stance": "evidence",
                "icon": "🕵️",
                "message": "The first reliable clue is a sharp shift in the high-value transaction cohort.",
            },
            {
                "agent": "Statistician",
                "role": "Distribution analyst",
                "stance": "analysis",
                "icon": "📊",
                "message": "PSI and KS both reject the healthy-distribution hypothesis. Drift is material.",
            },
            {
                "agent": "ML Engineer",
                "role": "Pipeline owner",
                "stance": "challenge",
                "icon": "👨‍💻",
                "message": "The pipeline contract changed too. I would not call it pure data drift yet.",
            },
            {
                "agent": "Infra",
                "role": "Reliability engineer",
                "stance": "infra",
                "icon": "☁️",
                "message": "Serving latency is elevated but stable; the incident is not a platform outage.",
            },
            {
                "agent": "Moderator",
                "role": "Decision maker",
                "stance": "decision",
                "icon": "⚖️",
                "message": "Consensus: drift is primary, contract mismatch is a contributor. Authorize a guarded repair experiment.",
            },
        ],
        "drift": {
            "features": [
                {"feature": "transaction_amount", "type": "numeric", "psi": 0.42, "severity": "critical"},
                {"feature": "transaction_frequency", "type": "numeric", "psi": 0.31, "severity": "critical"},
                {"feature": "location", "type": "categorical", "psi": 0.18, "severity": "warning"},
                {"feature": "merchant_category", "type": "categorical", "psi": 0.12, "severity": "warning"},
                {"feature": "customer_age", "type": "numeric", "psi": 0.06, "severity": "stable"},
                {"feature": "device_type", "type": "categorical", "psi": 0.04, "severity": "stable"},
                {"feature": "account_age_days", "type": "numeric", "psi": 0.03, "severity": "stable"},
            ]
        },
        "metrics_compare": {
            "baseline": {"accuracy": 0.76, "precision": 0.69, "recall": 0.66, "f1": 0.71},
            "candidate": {"accuracy": 0.91, "precision": 0.87, "recall": 0.84, "f1": 0.86},
        },
        "diagnosis": {
            "patient": "Fraud Sentinel / xgboost-v17",
            "condition": selected["title"],
            "severity": "High",
            "summary": selected["cause"],
            "recovery_probability": 94,
        },
        "treatment": [
            {
                "step": "Stabilize",
                "action": "Freeze risky promotion and preserve the current production evidence.",
            },
            {"step": "Repair", "action": selected["treatment"]},
            {
                "step": "Protect",
                "action": "Add automated data-contract and drift checks before deployment.",
            },
        ],
        "experiments": [
            {
                "version": "v17",
                "change": "Production baseline",
                "accuracy": 76.0,
                "f1": 0.71,
                "recall": 0.66,
                "status": "failed",
            },
            {
                "version": "v18-candidate",
                "change": "Contract repair + retraining",
                "accuracy": 91.0,
                "f1": 0.86,
                "recall": 0.84,
                "status": "passed",
            },
        ],
        "recovery": {
            "approved": True,
            "summary": "Candidate meets the defined recovery gates and is ready for human approval.",
            "git_commit": "fix: restore fraud feature contract and drift guardrails",
        },
        "artifacts": {},
        "raw": {},
    }


def _as_dict(value: Any) -> dict[str, Any]:
    """Turn pydantic/dataclass-ish response objects into a safe dictionary."""

    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return list(value.values())
    return [value]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _metric(value: Any) -> float:
    """Normalise 0..1 and 0..100 metric representations to percentages."""

    numeric = _number(value)
    return numeric * 100 if 0 < numeric <= 1 else numeric


def _normalise_items(items: Any, label_key: str = "title") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(items), start=1):
        data = _as_dict(item)
        if not data:
            data = {label_key: str(item)}
        data.setdefault("id", f"E-{index:02d}")
        data.setdefault(
            label_key, _first(data, "name", "label", "event", default=f"Evidence {index}")
        )
        data.setdefault(
            "detail",
            _first(
                data, "description", "summary", "message", "text", default="No narrative supplied."
            ),
        )
        output.append(data)
    return output


def normalise_investigation(payload: Any) -> dict[str, Any]:
    """Accept permissive API output and produce the dashboard's view contract.

    The API can return either the contract documented in ``dashboard/README.md``
    or a LangGraph state with semantically equivalent aliases.  Missing values
    are represented honestly as "awaiting evidence", rather than making the
    page fail during a live demo.
    """

    raw = _as_dict(payload)
    incident = _as_dict(_first(raw, "incident", "case", "incident_report", default={}))
    health_raw = _as_dict(_first(raw, "health", "model_health", "metrics", default={}))
    diagnosis_raw = _as_dict(_first(raw, "diagnosis", "medical_report", "root_cause", default={}))
    recovery_raw = _as_dict(_first(raw, "recovery", "validation", "approval", default={}))
    raw_experiments = _first(raw, "experiments", "experiment_results", "results", default=[])

    case_id = str(
        _first(
            raw,
            "case_id",
            "incident_id",
            default=_first(incident, "id", "case_id", default="CASE-PENDING"),
        )
    )
    incident_type = str(
        _first(
            raw,
            "incident_type",
            "type",
            default=_first(incident, "type", "incident_type", default="unknown"),
        )
    )
    title = str(
        _first(
            raw,
            "incident_title",
            "title",
            default=_first(
                incident, "title", "name", "description", default="ML reliability incident"
            ),
        )
    )
    status = str(
        _first(
            raw,
            "status",
            "phase",
            default=_first(incident, "status", default="Investigation complete"),
        )
    )

    before_health = _metric(
        _first(
            health_raw,
            "before",
            "baseline",
            "healthy_score",
            default=_first(raw, "health_before", default=92),
        )
    )
    incident_health = _metric(
        _first(
            health_raw,
            "after_incident",
            "current",
            "score",
            "health_score",
            default=_first(raw, "health_after_incident", default=61),
        )
    )
    recovered_health = _metric(
        _first(
            health_raw,
            "recovered",
            "after",
            "candidate",
            default=_first(raw, "health_recovered", default=incident_health),
        )
    )
    model_name = str(
        _first(
            health_raw,
            "model_name",
            "model",
            "version",
            default=_first(raw, "model_name", default="Fraud Sentinel"),
        )
    )

    evidence = _normalise_items(_first(raw, "evidence", "evidence_board", "clues", default=[]))
    suspects = _normalise_items(
        _first(raw, "suspects", "hypotheses", "root_cause_candidates", default=[]), "name"
    )
    for suspect in suspects:
        suspect["confidence"] = _metric(
            _first(suspect, "confidence", "score", "probability", default=0)
        )
        suspect.setdefault("verdict", _first(suspect, "status", "outcome", default="under review"))
        suspect.setdefault(
            "rationale",
            _first(suspect, "detail", "description", "summary", default="Awaiting evidence."),
        )

    timeline = _normalise_items(_first(raw, "timeline", "events", "activity", default=[]), "text")
    for entry in timeline:
        entry.setdefault("time", _first(entry, "timestamp", "at", default="--:--"))
        entry.setdefault("phase", _first(entry, "stage", "type", default="UPDATE"))
        entry["text"] = str(
            _first(
                entry, "text", "message", "detail", "description", default="Investigation update"
            )
        )

    messages = _normalise_items(
        _first(raw, "war_room", "conversation", "debate", "messages", default=[]), "message"
    )
    for message in messages:
        message.setdefault("agent", _first(message, "speaker", "author", "role", default="Agent"))
        message.setdefault("role", _first(message, "title", "team", default="Investigation team"))
        message.setdefault("stance", _first(message, "type", "status", default="analysis"))
        message["icon"] = str(
            _first(message, "icon", default=_agent_icon(str(message["agent"])))
        )
        message["message"] = str(
            _first(message, "message", "text", "detail", default="No message supplied.")
        )

    drift_raw = _first(raw, "drift", "drift_features", "feature_diagnostics", default={})
    drift_items = (
        _as_list(_as_dict(drift_raw).get("features"))
        if isinstance(drift_raw, Mapping)
        else _as_list(drift_raw)
    )
    drift_features: list[dict[str, Any]] = []
    for item in drift_items:
        data = _as_dict(item)
        if not data:
            continue
        drift_features.append(
            {
                "feature": str(
                    _first(data, "feature", "name", default=f"feature-{len(drift_features) + 1}")
                ),
                "type": str(_first(data, "type", default="numeric")),
                "psi": _number(_first(data, "psi", default=0)),
                "severity": str(_first(data, "severity", default="stable")),
            }
        )

    metrics_raw = _as_dict(_first(raw, "metrics_compare", default={}))
    metric_names = ("accuracy", "precision", "recall", "f1")

    def _metric_side(side: str) -> dict[str, float]:
        side_raw = _as_dict(metrics_raw.get(side))
        values: dict[str, float] = {}
        for name in metric_names:
            numeric = _number(side_raw.get(name, 0.0))
            values[name] = numeric / 100 if numeric > 1 else numeric
        return values

    metrics_compare = {
        "baseline": _metric_side("baseline"),
        "candidate": _metric_side("candidate"),
    }

    experiments: list[dict[str, Any]] = []
    for index, experiment in enumerate(_as_list(raw_experiments), start=1):
        item = _as_dict(experiment)
        if not item:
            item = {"version": f"run-{index}", "change": str(experiment)}
        experiments.append(
            {
                "version": str(
                    _first(item, "version", "name", "run_name", "id", default=f"run-{index}")
                ),
                "change": str(
                    _first(item, "change", "description", "treatment", default="Experiment")
                ),
                "accuracy": _metric(_first(item, "accuracy", "acc", default=0)),
                "f1": _number(_first(item, "f1", "f1_score", default=0)),
                "recall": _number(_first(item, "recall", default=0)),
                "status": str(_first(item, "status", "outcome", default="completed")),
            }
        )

    treatment = _normalise_items(
        _first(raw, "treatment", "prescription", "actions", "recommended_actions", default=[]),
        "action",
    )
    for index, step in enumerate(treatment, start=1):
        step.setdefault("step", _first(step, "title", "name", default=f"Step {index}"))
        step["action"] = str(
            _first(step, "action", "detail", "description", "text", default="No action specified.")
        )

    condition = str(
        _first(
            diagnosis_raw,
            "condition",
            "title",
            "root_cause",
            default=_first(raw, "root_cause", default="Diagnosis pending"),
        )
    )
    diagnosis_summary = str(
        _first(
            diagnosis_raw,
            "summary",
            "description",
            "detail",
            default=_first(
                raw,
                "diagnosis_summary",
                default="The diagnostic engine has not supplied a narrative.",
            ),
        )
    )
    approved = bool(
        _first(
            recovery_raw,
            "approved",
            "passed",
            "safe_to_promote",
            default=_first(raw, "approved", default=False),
        )
    )
    artifacts = {
        str(key): str(value)
        for key, value in _as_dict(_first(raw, "artifacts", default={})).items()
        if value
    }

    return {
        "source": str(_first(raw, "source", "run_source", default="live-engine")),
        "case_id": case_id,
        "incident_type": incident_type,
        "incident_title": title,
        "status": status,
        "health": {
            "before": before_health,
            "after_incident": incident_health,
            "recovered": recovered_health,
            "status": str(_first(health_raw, "status", "state", default="critical")),
            "model_name": model_name,
            "latency_ms": _number(_first(health_raw, "latency_ms", "latency", default=0)),
        },
        "evidence": evidence,
        "suspects": suspects,
        "timeline": timeline,
        "war_room": messages,
        "drift": {"features": drift_features},
        "metrics_compare": metrics_compare,
        "diagnosis": {
            "patient": str(_first(diagnosis_raw, "patient", "model_name", default=model_name)),
            "condition": condition,
            "severity": str(_first(diagnosis_raw, "severity", "risk", default="Unknown")),
            "summary": diagnosis_summary,
            "recovery_probability": _metric(
                _first(diagnosis_raw, "recovery_probability", "confidence", default=0)
            ),
        },
        "treatment": treatment,
        "experiments": experiments,
        "recovery": {
            "approved": approved,
            "summary": str(
                _first(
                    recovery_raw,
                    "summary",
                    "reason",
                    "message",
                    default="Awaiting validation evidence.",
                )
            ),
            "gates": [
                {
                    "name": str(_first(gate, "name", default="Gate")),
                    "passed": bool(_first(gate, "passed", default=False)),
                    "detail": str(_first(gate, "detail", default="")),
                }
                for gate in _as_list(recovery_raw.get("gates"))
            ],
            "git_commit": str(
                _first(
                    recovery_raw,
                    "git_commit",
                    "commit",
                    "commit_message",
                    default="No generated commit recorded.",
                )
            ),
        },
        "artifacts": artifacts,
        "raw": raw,
    }


def _resolve_engine() -> tuple[Callable[[str], Any] | None, Callable[[], Any] | None, str | None]:
    """Locate the core API without imposing a startup order on the project."""

    try:
        module = importlib.import_module("api.main")
    except Exception as error:  # Import errors should not kill a demo UI.
        return None, None, f"Core API unavailable: {error.__class__.__name__}"

    investigate = getattr(module, "run_investigation", None)
    reset = getattr(module, "reset_demo", None) or getattr(module, "reset_state", None)
    if not callable(investigate):
        return (
            None,
            reset if callable(reset) else None,
            "Core API has not exposed run_investigation yet.",
        )
    return investigate, reset if callable(reset) else None, None


async def _resolve_awaitable(outcome: Awaitable[Any]) -> Any:
    return await outcome


def _call_maybe_async(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    outcome = function(*args, **kwargs)
    if inspect.isawaitable(outcome):
        import asyncio

        return asyncio.run(_resolve_awaitable(outcome))
    return outcome


_NODE_PROGRESS = {
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


def run_investigation(
    incident_type: str,
    *,
    strict_gates: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Run the core engine with optional per-node progress callbacks."""

    investigate, _, issue = _resolve_engine()
    if investigate is None:
        return _preview_case(incident_type), issue
    try:
        payload = _call_maybe_async(investigate, incident_type, strict_gates=strict_gates)
        return normalise_investigation(payload), None
    except TypeError:
        try:
            from agents.graph import normalize_incident_type, stream_case, to_dashboard_contract
            from api.main import reset_local_state

            reset_local_state()
            state: dict[str, Any] = {
                "incident_type": normalize_incident_type(incident_type),
                "timeline": [],
                "strict_gates": strict_gates,
            }
            for node_name, update in stream_case(incident_type, strict_gates=strict_gates):
                state.update(update)
                if progress is not None:
                    progress(_NODE_PROGRESS.get(node_name, node_name))
            return normalise_investigation(to_dashboard_contract(state)), None
        except Exception as error:
            prior = st.session_state.get("case")
            return (
                prior if isinstance(prior, dict) else _preview_case(incident_type),
                f"Investigation engine raised {error.__class__.__name__}: {error}",
            )
    except Exception as error:
        prior = st.session_state.get("case")
        return (
            prior if isinstance(prior, dict) else _preview_case(incident_type),
            f"Investigation engine raised {error.__class__.__name__}: {error}",
        )


def reset_investigation() -> str | None:
    """Reset local presentation state and ask the core to reset when supported."""

    _, reset, issue = _resolve_engine()
    if reset is not None:
        try:
            _call_maybe_async(reset)
        except Exception as error:
            issue = f"Core reset raised {error.__class__.__name__}: {error}"
    st.session_state.case = None
    st.session_state.engine_notice = issue
    return issue


def _escape(value: Any) -> str:
    """Minimal escaping for strings inserted into controlled HTML templates."""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _status_colour(status: str) -> str:
    value = status.lower()
    if any(token in value for token in ("pass", "approved", "healthy", "recovered", "primary")):
        return "mint"
    if any(token in value for token in ("fail", "critical", "high", "sick", "reject")):
        return "coral"
    if any(token in value for token in ("warning", "medium", "contributing", "review")):
        return "gold"
    return "blue"


def _stance_colour(stance: str) -> str:
    value = stance.lower()
    if any(token in value for token in ("dissent", "challenge", "objection")):
        return "coral"
    if any(token in value for token in ("consensus", "decision", "verdict")):
        return "mint"
    if any(token in value for token in ("operational", "infra", "opens")):
        return "gold"
    return "blue"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');
          :root {
            --ink: #07131b;
            --ink-soft: #0c1d27;
            --paper: #f5f1e8;
            --muted: #9caab0;
            --line: rgba(190, 215, 211, .16);
            --mint: #5de0b5;
            --coral: #ff7462;
            --gold: #f2c86f;
            --blue: #89c7ff;
          }
          .stApp { background: radial-gradient(circle at 12% -8%, #17394a 0, transparent 34%), radial-gradient(circle at 96% 6%, #213c36 0, transparent 29%), var(--ink); color: var(--paper); }
          .stApp, .stApp [data-testid="stMarkdownContainer"] { font-family: 'DM Sans', sans-serif; }
          #MainMenu, footer, header { visibility: hidden; }
          [data-testid="stSidebar"] { background: #091820; border-right: 1px solid var(--line); }
          [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--paper); }
          [data-testid="stSidebar"] .stRadio label { color: #d9e4e4 !important; font-family: 'DM Mono', monospace; font-size: .8rem; }
          [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stCaption { color: var(--muted) !important; font-family: 'DM Mono', monospace; }
          [data-testid="stSidebar"] div[data-baseweb="select"] > div { background: rgba(255,255,255,.07); border-color: var(--line); color: var(--paper); }
          .block-container { max-width: 1280px; padding-top: 2.2rem; padding-bottom: 3rem; }

          @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
          @keyframes grow { from { width: 0; } }
          @keyframes pulse-coral { 0% { box-shadow: 0 0 0 0 rgba(255,116,98,.38); } 70% { box-shadow: 0 0 0 20px rgba(255,116,98,0); } 100% { box-shadow: 0 0 0 0 rgba(255,116,98,0); } }
          @keyframes pulse-mint { 0% { box-shadow: 0 0 0 0 rgba(93,224,181,.34); } 70% { box-shadow: 0 0 0 20px rgba(93,224,181,0); } 100% { box-shadow: 0 0 0 0 rgba(93,224,181,0); } }
          @keyframes ecg-move { from { stroke-dashoffset: 1280; } to { stroke-dashoffset: 0; } }
          @keyframes hero-sheen { 0% { transform: translateX(-70%) skewX(-14deg); } 100% { transform: translateX(240%) skewX(-14deg); } }

          .sherlock-hero { position: relative; overflow: hidden; padding: 1.65rem 1.85rem 1.55rem; background: linear-gradient(112deg, rgba(23, 52, 60, .93), rgba(10, 27, 34, .88)); border: 1px solid rgba(137, 199, 255, .24); border-radius: 18px; margin-bottom: 1.15rem; box-shadow: 0 18px 48px rgba(0,0,0,.22); }
          .sherlock-hero:after { content:''; position:absolute; width: 320px; height: 320px; right:-105px; top:-185px; border-radius:50%; border: 1px solid rgba(93,224,181,.22); box-shadow: 0 0 0 34px rgba(93,224,181,.03), 0 0 0 70px rgba(93,224,181,.025); }
          .hero-sheen { position:absolute; top:0; bottom:0; width:26%; background:linear-gradient(90deg, transparent, rgba(137,199,255,.05), transparent); animation: hero-sheen 6.5s ease-in-out infinite; pointer-events:none; }
          .eyebrow { font-family:'DM Mono', monospace; color: var(--mint); letter-spacing:.13em; font-size:.72rem; text-transform:uppercase; }
          .hero-title { margin:.25rem 0 .1rem; font-family:'Playfair Display', serif; font-size:2.25rem; color:var(--paper); letter-spacing:-.035em; }
          .hero-subtitle { color:#b7c8c7; font-size:.96rem; max-width: 47rem; }
          .case-chip { display:inline-block; margin:.75rem .45rem 0 0; padding:.28rem .6rem; border:1px solid rgba(93,224,181,.34); border-radius:999px; color:var(--mint); background:rgba(93,224,181,.08); font: .68rem 'DM Mono', monospace; letter-spacing:.06em; }
          .case-chip.coral { border-color:rgba(255,116,98,.5); color:var(--coral); background:rgba(255,116,98,.08); }
          .case-chip.gold { border-color:rgba(242,200,111,.5); color:var(--gold); background:rgba(242,200,111,.08); }
          .case-chip.blue { border-color:rgba(137,199,255,.5); color:var(--blue); background:rgba(137,199,255,.08); }
          .section-label { font: .72rem 'DM Mono', monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--mint); margin:1.55rem 0 .6rem; }

          .stepper { display:flex; gap:.4rem; margin:.2rem 0 1.15rem; }
          .step { flex:1; display:flex; flex-direction:column; align-items:center; gap:.28rem; padding:.62rem .3rem .55rem; border-radius:11px; border:1px solid var(--line); background:rgba(13,34,42,.55); color:var(--muted); font:.62rem 'DM Mono', monospace; letter-spacing:.09em; animation: rise .5s cubic-bezier(.2,.7,.3,1) both; }
          .step-icon { font-size:.95rem; filter:grayscale(1) opacity(.55); }
          .step.done { border-color:rgba(93,224,181,.38); background:rgba(93,224,181,.07); color:var(--mint); }
          .step.done .step-icon { filter:none; }
          .step.current { border-color:rgba(242,200,111,.55); background:rgba(242,200,111,.08); color:var(--gold); }
          .step.current .step-icon { filter:none; }

          .avatar-card { display:flex; align-items:center; gap:1.15rem; padding:1.05rem 1.2rem; background:linear-gradient(140deg, rgba(17,41,48,.92), rgba(8,23,29,.88)); border:1px solid var(--line); border-radius:14px; min-height:138px; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .avatar-ring { flex:0 0 78px; height:78px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2.15rem; }
          .avatar-ring.coral { background:rgba(255,116,98,.1); border:1px solid rgba(255,116,98,.5); animation: pulse-coral 2.1s infinite; }
          .avatar-ring.mint { background:rgba(93,224,181,.1); border:1px solid rgba(93,224,181,.5); animation: pulse-mint 2.4s infinite; }
          .avatar-ring.blue { background:rgba(137,199,255,.1); border:1px solid rgba(137,199,255,.5); }
          .avatar-body { flex:1; min-width:0; }
          .avatar-name { color:var(--paper); font-weight:600; font-size:1.02rem; margin-bottom:.3rem; }
          .ecg { width:100%; height:44px; margin-top:.5rem; }
          .ecg path { fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray: 640; animation: ecg-move 3.4s linear infinite; }
          .ecg.mint path { stroke: var(--mint); filter: drop-shadow(0 0 4px rgba(93,224,181,.55)); }
          .ecg.coral path { stroke: var(--coral); filter: drop-shadow(0 0 4px rgba(255,116,98,.55)); }

          .metric-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.72rem; }
          .metric-card { min-height:108px; padding: .9rem .95rem; background: rgba(13, 34, 42, .83); border:1px solid var(--line); border-radius: 13px; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .metric-label { color:#9daeb1; font: .66rem 'DM Mono', monospace; letter-spacing:.08em; text-transform:uppercase; }
          .metric-value { font: 600 1.7rem 'DM Sans', sans-serif; color:var(--paper); margin-top:.25rem; letter-spacing:-.04em; }
          .metric-note { color:#aebbbb; font-size:.76rem; margin-top:.25rem; }
          .signal { display:inline-flex; align-items:center; gap:.35rem; font: .68rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.06em; }
          .dot { width:.45rem; height:.45rem; border-radius:50%; display:inline-block; }
          .mint { color:var(--mint); } .mint .dot, .dot.mint { background:var(--mint); }
          .coral { color:var(--coral); } .coral .dot, .dot.coral { background:var(--coral); }
          .gold { color:var(--gold); } .gold .dot, .dot.gold { background:var(--gold); }
          .blue { color:var(--blue); } .blue .dot, .dot.blue { background:var(--blue); }

          .timeline { position:relative; margin:.4rem 0 1.1rem; }
          .timeline:before { content:''; position:absolute; left: 5.1rem; top:.55rem; bottom:.55rem; width:1px; background:linear-gradient(var(--mint), rgba(93,224,181,.08)); }
          .timeline-row { display:grid; grid-template-columns:4.45rem 1.3rem 1fr; gap:.5rem; align-items:start; padding:.47rem 0; animation: rise .5s cubic-bezier(.2,.7,.3,1) both; }
          .timeline-time { color:var(--gold); font: .68rem 'DM Mono', monospace; padding-top:.16rem; }
          .timeline-node { width:.63rem; height:.63rem; margin-top:.18rem; border-radius:50%; background:var(--ink); border:2px solid var(--mint); box-shadow:0 0 8px rgba(93,224,181,.45); z-index:1; }
          .timeline-phase { color:var(--mint); font: .64rem 'DM Mono', monospace; letter-spacing:.08em; }
          .timeline-text { color:#d5e0df; font-size:.88rem; line-height:1.42; margin-top:.1rem; }

          .evidence-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.78rem; }
          .evidence-card { position:relative; padding:1rem 1rem 2.3rem; min-height:150px; background:linear-gradient(150deg, rgba(17,41,48,.91), rgba(8,23,29,.86)); border:1px solid var(--line); border-radius:13px; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; transition: transform .25s ease, border-color .25s ease; }
          .evidence-card:hover { transform: translateY(-3px); border-color: rgba(93,224,181,.4); }
          .evidence-id { color:var(--gold); font: .65rem 'DM Mono', monospace; letter-spacing:.1em; }
          .evidence-title { color:var(--paper); font-weight:600; margin:.56rem 0 .4rem; }
          .evidence-detail { color:#b3c1c1; font-size:.81rem; line-height:1.4; }
          .evidence-source { position:absolute; bottom:.78rem; color:#6f8d91; font: .62rem 'DM Mono', monospace; text-transform:uppercase; }

          .suspect { margin:.48rem 0; padding:.75rem .85rem .8rem; background:rgba(14,36,43,.74); border-left:3px solid var(--blue); border-radius:0 10px 10px 0; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .suspect.primary { border-left-color:var(--coral); } .suspect.contributing { border-left-color:var(--gold); }
          .suspect-top { display:flex; justify-content:space-between; align-items:baseline; gap:.8rem; color:var(--paper); font-weight:600; }
          .confidence { color:var(--gold); font: .73rem 'DM Mono', monospace; white-space:nowrap; }
          .suspect-meta { color:#9eafb1; font-size:.77rem; margin-top:.22rem; }
          .conf-track { margin-top:.55rem; height:.42rem; border-radius:999px; background:rgba(255,255,255,.07); overflow:hidden; }
          .conf-fill { height:100%; border-radius:999px; animation: grow 1.1s cubic-bezier(.2,.7,.3,1) both; }
          .conf-fill.coral { background:linear-gradient(90deg, rgba(255,116,98,.55), var(--coral)); }
          .conf-fill.gold { background:linear-gradient(90deg, rgba(242,200,111,.55), var(--gold)); }
          .conf-fill.blue { background:linear-gradient(90deg, rgba(137,199,255,.45), var(--blue)); }
          .conf-fill.mint { background:linear-gradient(90deg, rgba(93,224,181,.5), var(--mint)); }

          .chat-row { display:flex; gap:.8rem; margin:.85rem 0; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .chat-avatar { flex:0 0 2.8rem; height:2.8rem; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:1.28rem; background:rgba(137,199,255,.09); border:1px solid rgba(137,199,255,.3); box-shadow:0 4px 14px rgba(0,0,0,.25); }
          .chat-avatar.coral { background:rgba(255,116,98,.1); border-color:rgba(255,116,98,.42); }
          .chat-avatar.mint { background:rgba(93,224,181,.1); border-color:rgba(93,224,181,.42); }
          .chat-avatar.gold { background:rgba(242,200,111,.1); border-color:rgba(242,200,111,.42); }
          .chat-bubble { flex:1; padding:.78rem .95rem .8rem; background:linear-gradient(150deg, rgba(17,41,48,.9), rgba(9,25,31,.86)); border:1px solid var(--line); border-radius:4px 14px 14px 14px; }
          .chat-head { display:flex; flex-wrap:wrap; align-items:center; gap:.55rem; }
          .chat-name { color:var(--paper); font-weight:600; }
          .chat-role { color:#7f9da1; font:.68rem 'DM Mono', monospace; }
          .stance-chip { margin-left:auto; padding:.14rem .5rem; border-radius:999px; font:.6rem 'DM Mono', monospace; letter-spacing:.07em; text-transform:uppercase; border:1px solid var(--line); }
          .stance-chip.coral { color:var(--coral); border-color:rgba(255,116,98,.45); background:rgba(255,116,98,.08); }
          .stance-chip.mint { color:var(--mint); border-color:rgba(93,224,181,.45); background:rgba(93,224,181,.08); }
          .stance-chip.gold { color:var(--gold); border-color:rgba(242,200,111,.45); background:rgba(242,200,111,.08); }
          .stance-chip.blue { color:var(--blue); border-color:rgba(137,199,255,.45); background:rgba(137,199,255,.08); }
          .chat-copy { color:#c5d2d1; font-size:.88rem; line-height:1.48; margin-top:.35rem; }

          .verdict-card { display:flex; align-items:center; gap:1.2rem; padding:1.1rem 1.25rem; background:linear-gradient(135deg, rgba(20,55,52,.85), rgba(8,24,30,.9)); border:1px solid rgba(93,224,181,.28); border-radius:14px; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .prob-ring { flex:0 0 104px; height:104px; border-radius:50%; display:flex; align-items:center; justify-content:center; }
          .prob-core { width:80px; height:80px; border-radius:50%; background:#0a1b22; display:flex; flex-direction:column; align-items:center; justify-content:center; }
          .prob-value { font:600 1.32rem 'DM Sans', sans-serif; color:var(--mint); letter-spacing:-.03em; }
          .prob-caption { font:.52rem 'DM Mono', monospace; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }

          .doctor-card { padding:1.2rem; background:linear-gradient(135deg, rgba(20,55,52,.82), rgba(8,24,30,.88)); border:1px solid rgba(93,224,181,.22); border-radius:14px; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .diagnosis-name { margin:.35rem 0; color:var(--paper); font:600 1.45rem 'Playfair Display',serif; }
          .diagnosis-copy { color:#c0d0ce; line-height:1.53; font-size:.9rem; }
          .rx-card { display:grid; grid-template-columns: 2.1rem 1fr; gap:.7rem; padding:.8rem 0; border-bottom:1px solid var(--line); animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .rx-card:last-child { border-bottom:0; }
          .rx-num { color:var(--mint); font:.75rem 'DM Mono',monospace; padding-top:.14rem; }
          .rx-step { color:var(--paper); font-weight:600; font-size:.9rem; }
          .rx-action { color:#acbdbc; font-size:.81rem; line-height:1.42; margin-top:.13rem; }

          .comparison { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
          .experiment-card { padding:1rem; border-radius:13px; border:1px solid var(--line); background:rgba(13,34,42,.78); animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .experiment-card.pass { border-color:rgba(93,224,181,.35); background:rgba(93,224,181,.07); }
          .experiment-version { color:var(--blue); font:.67rem 'DM Mono',monospace; letter-spacing:.08em; }
          .experiment-change { color:var(--paper); font-weight:600; margin:.35rem 0 .75rem; }
          .exp-metrics { display:flex; gap:1rem; flex-wrap:wrap; }
          .exp-metric { color:#9baeb0; font:.66rem 'DM Mono',monospace; }
          .exp-metric strong { display:block; color:var(--paper); font:600 1.15rem 'DM Sans',sans-serif; margin-top:.1rem; }

          .delta-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.72rem; margin:.2rem 0 .4rem; }
          .delta-chip { padding:.75rem .8rem; border-radius:12px; border:1px solid var(--line); background:rgba(13,34,42,.7); text-align:center; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .delta-value { font:600 1.35rem 'DM Sans',sans-serif; letter-spacing:-.03em; }
          .delta-label { margin-top:.15rem; font:.62rem 'DM Mono',monospace; color:var(--muted); letter-spacing:.09em; text-transform:uppercase; }

          .approval { padding:1.1rem 1.2rem; border:1px solid rgba(93,224,181,.34); border-radius:13px; background:rgba(93,224,181,.08); animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .approval-title { color:var(--mint); font:.74rem 'DM Mono',monospace; letter-spacing:.09em; }
          .approval-copy { color:#dceae6; margin-top:.35rem; font-size:.9rem; }

          .terminal { border-radius:13px; overflow:hidden; border:1px solid rgba(93,224,181,.25); background:#040d12; animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
          .terminal-bar { display:flex; align-items:center; gap:.4rem; padding:.5rem .8rem; background:rgba(255,255,255,.04); border-bottom:1px solid rgba(255,255,255,.06); }
          .t-dot { width:.6rem; height:.6rem; border-radius:50%; }
          .t-dot.red { background:#ff5f56; } .t-dot.yellow { background:#ffbd2e; } .t-dot.green { background:#27c93f; }
          .terminal-title { margin-left:.4rem; color:#7f9da1; font:.64rem 'DM Mono',monospace; letter-spacing:.06em; }
          .terminal-body { padding:.85rem 1rem; font:.78rem 'DM Mono',monospace; color:#c5e8dc; line-height:1.7; overflow-x:auto; }
          .terminal-body .prompt { color:var(--mint); margin-right:.45rem; }
          .terminal-body .t-muted { color:#5f7d81; }

          .gate-board { margin:.35rem 0 1rem; }
          .gate-row { display:grid; grid-template-columns: 5.5rem 1fr 1.4fr; gap:.65rem; align-items:center; padding:.55rem .2rem; border-bottom:1px solid var(--line); font-size:.84rem; }
          .gate-row:last-child { border-bottom:0; }
          .gate-name { color:var(--paper); font-weight:600; }
          .gate-detail { color:#9eafb1; font:.74rem 'DM Mono', monospace; }
          .artifact-chip { padding:.4rem .7rem; border-radius:9px; border:1px solid rgba(137,199,255,.28); background:rgba(137,199,255,.06); color:var(--blue); font:.68rem 'DM Mono',monospace; }
          .artifact-chip span { color:#7f9da1; margin-right:.45rem; text-transform:uppercase; letter-spacing:.07em; }

          .empty-state { padding:1.4rem; border:1px dashed rgba(137,199,255,.3); border-radius:12px; color:#aebfc0; font-size:.88rem; }
          .stButton > button { width:100%; border-radius:10px; border:1px solid rgba(93,224,181,.45); background:rgba(93,224,181,.1); color:#e8fffa; font-family:'DM Mono',monospace; font-size:.76rem; letter-spacing:.04em; }
          .stButton > button:hover { border-color:var(--mint); color:var(--mint); background:rgba(93,224,181,.16); }
          .stAlert { border-radius:10px; }
          [data-testid="stPlotlyChart"] { background: rgba(13,34,42,.6); border:1px solid var(--line); border-radius:13px; padding:.45rem .45rem .1rem; }
          @media (max-width: 800px) { .metric-grid, .delta-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .evidence-grid { grid-template-columns:1fr; } .comparison { grid-template-columns:1fr; } .hero-title { font-size:1.8rem; } .stepper { flex-wrap:wrap; } .step { min-width:28%; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plotly figure builders (all styled to the dashboard's dark palette)
# ---------------------------------------------------------------------------


def _apply_layout(figure: Any, height: int, margin_top: int = 18) -> Any:
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 14, "t": margin_top, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#c5d2d1", "size": 13},
        showlegend=False,
    )
    return figure


def _plot(figure: Any) -> None:
    st.plotly_chart(figure, use_container_width=True, config=_PLOTLY_CONFIG)


def _health_gauge_figure(health: Mapping[str, Any]) -> Any:
    recovered = _number(health.get("recovered"))
    before = _number(health.get("before"))
    bar_colour = _PALETTE["mint"] if recovered >= before - 5 else _PALETTE["coral"]
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=recovered,
            number={"suffix": "%", "font": {"size": 42, "family": "DM Sans, sans-serif"}},
            delta={
                "reference": before,
                "suffix": " pts vs baseline",
                "font": {"size": 13},
                "increasing": {"color": _PALETTE["mint"]},
                "decreasing": {"color": _PALETTE["coral"]},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(190,215,211,.25)",
                    "tickfont": {"size": 10, "color": _PALETTE["muted"]},
                },
                "bar": {"color": bar_colour, "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 55], "color": "rgba(255,116,98,.15)"},
                    {"range": [55, 75], "color": "rgba(242,200,111,.13)"},
                    {"range": [75, 100], "color": "rgba(93,224,181,.12)"},
                ],
                "threshold": {
                    "line": {"color": _PALETTE["gold"], "width": 3},
                    "thickness": 0.85,
                    "value": min(before, 100),
                },
            },
        )
    )
    return _apply_layout(figure, height=252, margin_top=30)


def _trajectory_figure(health: Mapping[str, Any]) -> Any:
    stages = ["Baseline", "Incident", "Recovered"]
    values = [
        _number(health.get("before")),
        _number(health.get("after_incident")),
        _number(health.get("recovered")),
    ]
    figure = go.Figure(
        go.Scatter(
            x=stages,
            y=values,
            mode="lines+markers+text",
            text=[f"{value:.0f}%" for value in values],
            textposition="top center",
            textfont={"family": "DM Mono, monospace", "size": 13, "color": "#e8f4ef"},
            line={"color": _PALETTE["blue"], "width": 3, "shape": "spline"},
            marker={
                "size": 13,
                "color": [_PALETTE["blue"], _PALETTE["coral"], _PALETTE["mint"]],
                "line": {"color": "rgba(7,19,27,.9)", "width": 2},
            },
            fill="tozeroy",
            fillcolor="rgba(137,199,255,.06)",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_yaxes(
        range=[0, max(values + [100]) * 1.18],
        gridcolor=_PALETTE["grid"],
        zeroline=False,
        ticksuffix="%",
    )
    figure.update_xaxes(showgrid=False)
    applied = _apply_layout(figure, height=280, margin_top=34)
    applied.update_layout(margin={"l": 36, "r": 44, "t": 34, "b": 12})
    return applied


def _drift_figure(features: list[dict[str, Any]]) -> Any:
    ranked = sorted(features, key=lambda item: _number(item.get("psi")), reverse=True)[:7]
    ranked.reverse()  # horizontal bars read bottom-up
    severity_colour = {
        "critical": _PALETTE["coral"],
        "warning": _PALETTE["gold"],
        "stable": "rgba(137,199,255,.55)",
    }
    values = [max(_number(item.get("psi")), 0.005) for item in ranked]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[str(item.get("feature", "feature")) for item in ranked],
            orientation="h",
            marker={
                "color": [
                    severity_colour.get(str(item.get("severity", "stable")), _PALETTE["blue"])
                    for item in ranked
                ],
                "line": {"width": 0},
            },
            text=[f"{_number(item.get('psi')):.2f}" for item in ranked],
            textposition="outside",
            textfont={"family": "DM Mono, monospace", "size": 11, "color": "#d5e0df"},
            cliponaxis=False,
            hovertemplate="%{y}: PSI %{x:.3f}<extra></extra>",
        )
    )
    figure.add_vline(
        x=0.25,
        line_dash="dot",
        line_color=_PALETTE["gold"],
        annotation_text="0.25 drift threshold",
        annotation_position="top",
        annotation_font={"family": "DM Mono, monospace", "size": 10, "color": _PALETTE["gold"]},
    )
    figure.update_xaxes(
        type="log",
        title={"text": "population stability index (log scale)", "font": {"size": 11}},
        gridcolor=_PALETTE["grid"],
        zeroline=False,
        tickvals=[0.01, 0.05, 0.25, 1, 5, 25],
        ticktext=["0.01", "0.05", "0.25", "1", "5", "25"],
        tickfont={"family": "DM Mono, monospace", "size": 10},
    )
    figure.update_yaxes(
        showgrid=False, tickfont={"family": "DM Mono, monospace", "size": 11}
    )
    applied = _apply_layout(figure, height=max(250, 44 * len(ranked) + 60), margin_top=42)
    applied.update_layout(margin={"l": 10, "r": 48, "t": 42, "b": 10})
    return applied


def _comparison_figure(metrics_compare: Mapping[str, Any], experiments: list[dict[str, Any]]) -> Any:
    metric_names = ("accuracy", "precision", "recall", "f1")
    baseline = _as_dict(metrics_compare.get("baseline"))
    candidate = _as_dict(metrics_compare.get("candidate"))
    baseline_label = experiments[0]["version"] if experiments else "baseline"
    candidate_label = experiments[-1]["version"] if experiments else "candidate"
    labels = [name.capitalize() for name in metric_names]
    baseline_values = [round(_number(baseline.get(name)) * 100, 1) for name in metric_names]
    candidate_values = [round(_number(candidate.get(name)) * 100, 1) for name in metric_names]
    figure = go.Figure(
        [
            go.Bar(
                name=baseline_label,
                x=labels,
                y=baseline_values,
                marker={"color": "rgba(255,116,98,.72)", "line": {"width": 0}},
                text=[f"{value:.0f}" for value in baseline_values],
                textposition="outside",
                textfont={"family": "DM Mono, monospace", "size": 11},
                hovertemplate=baseline_label + " · %{x}: %{y:.1f}%<extra></extra>",
            ),
            go.Bar(
                name=candidate_label,
                x=labels,
                y=candidate_values,
                marker={"color": "rgba(93,224,181,.85)", "line": {"width": 0}},
                text=[f"{value:.0f}" for value in candidate_values],
                textposition="outside",
                textfont={"family": "DM Mono, monospace", "size": 11},
                hovertemplate=candidate_label + " · %{x}: %{y:.1f}%<extra></extra>",
            ),
        ]
    )
    figure.update_layout(barmode="group", bargap=0.34, bargroupgap=0.08)
    figure.update_yaxes(
        range=[0, 118], gridcolor=_PALETTE["grid"], zeroline=False, ticksuffix="%"
    )
    figure.update_xaxes(showgrid=False, tickfont={"family": "DM Mono, monospace", "size": 11})
    applied = _apply_layout(figure, height=300, margin_top=26)
    applied.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "y": 1.16,
            "x": 0,
            "font": {"family": "DM Mono, monospace", "size": 11, "color": "#d5e0df"},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return applied


# ---------------------------------------------------------------------------
# HTML component renderers
# ---------------------------------------------------------------------------


def _render_hero(case: dict[str, Any]) -> None:
    live = case["source"] != "dashboard-preview"
    source = "LIVE INVESTIGATION" if live else "INTERACTIVE PREVIEW"
    source_colour = "mint" if live else "gold"
    health_colour = "coral" if "critical" in str(case["health"]["status"]).lower() else "mint"
    st.markdown(
        f"""
        <div class="sherlock-hero">
          <div class="hero-sheen"></div>
          <div class="eyebrow">SherlockML / autonomous ML incident command</div>
          <div class="hero-title">{_escape(case["incident_title"])}</div>
          <div class="hero-subtitle">A detective-and-doctor workflow for evidence, diagnosis, reversible treatment, and validated recovery.</div>
          <div>
            <span class="case-chip {source_colour}">{_escape(case["case_id"])} · {source}</span>
            <span class="case-chip {health_colour}">PATIENT {("RECOVERED" if health_colour == "mint" else "CRITICAL")}</span>
            <span class="case-chip blue">{_escape(case["status"]).upper()}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stepper(timeline: list[dict[str, Any]]) -> None:
    reached: set[int] = set()
    for entry in timeline:
        index = _PHASE_TO_STEP.get(str(entry.get("phase", "")).upper())
        if index is not None:
            reached.add(index)
    current = max(reached) if reached else -1
    cells = "".join(
        f'<div class="step {"current" if index == current else "done" if index in reached else ""}" style="animation-delay:{index * 60}ms">'
        f'<span class="step-icon">{icon}</span><span>{label}</span></div>'
        for index, (label, icon) in enumerate(_STEPS)
    )
    st.markdown(f'<div class="stepper">{cells}</div>', unsafe_allow_html=True)


_ECG_PATH = (
    "M0,35 H70 L84,12 L98,56 L112,35 H190 L204,18 L218,50 L232,35 H310 L324,8 "
    "L338,60 L352,35 H430 L444,16 L458,52 L472,35 H600"
)


def _render_avatar(case: dict[str, Any]) -> None:
    health = case["health"]
    status = str(health.get("status", "critical")).lower()
    if "recover" in status or "healthy" in status:
        face, tone, label = "😌", "mint", "stabilised — cleared for human review"
    elif "critical" in status:
        face, tone, label = "🤒", "coral", "critical — under active treatment"
    else:
        face, tone, label = "🙂", "blue", "under observation"
    st.markdown(
        f"""
        <div class="avatar-card">
          <div class="avatar-ring {tone}"><span>{face}</span></div>
          <div class="avatar-body">
            <div class="avatar-name">{_escape(health.get("model_name", "Production model"))}</div>
            <span class="signal {tone}"><span class="dot"></span>{_escape(label)}</span>
            <svg class="ecg {tone}" viewBox="0 0 600 70" preserveAspectRatio="none"><path d="{_ECG_PATH}"/></svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, note: str, colour: str = "blue", delay: int = 0) -> str:
    return (
        f'<div class="metric-card" style="animation-delay:{delay}ms">'
        f'<div class="metric-label">{_escape(label)}</div>'
        f'<div class="metric-value">{_escape(value)}</div><div class="metric-note">'
        f'<span class="signal {colour}"><span class="dot"></span>{_escape(note)}</span>'
        "</div></div>"
    )


def _empty(message: str) -> None:
    st.markdown(f'<div class="empty-state">{_escape(message)}</div>', unsafe_allow_html=True)


def _render_metrics(case: dict[str, Any]) -> None:
    health = case["health"]
    deterioration = health["after_incident"] - health["before"]
    recovery = health["recovered"] - health["after_incident"]
    cards = "".join(
        (
            _metric_card(
                "Baseline health",
                f"{health['before']:.0f}%",
                "pre-incident reference",
                "blue",
                0,
            ),
            _metric_card(
                "Health on alert",
                f"{health['after_incident']:.0f}%",
                f"{deterioration:+.0f} pts from baseline",
                "coral",
                70,
            ),
            _metric_card(
                "Recovery signal",
                f"{health['recovered']:.0f}%",
                f"{recovery:+.0f} pts candidate lift",
                "mint",
                140,
            ),
            _metric_card(
                "Serving latency",
                f"{health['latency_ms']:.0f} ms" if health["latency_ms"] else "—",
                case["status"],
                _status_colour(case["status"]),
                210,
            ),
        )
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def _render_timeline(timeline: list[dict[str, Any]], compact: bool = False) -> None:
    if not timeline:
        _empty("The engine has not emitted an investigation timeline yet.")
        return
    entries = timeline[:3] if compact else timeline
    rows = "".join(
        f"""
        <div class="timeline-row" style="animation-delay:{index * 70}ms">
          <div class="timeline-time">{_escape(entry["time"])}</div>
          <div class="timeline-node"></div>
          <div><div class="timeline-phase">{_escape(entry["phase"])}</div><div class="timeline-text">{_escape(entry["text"])}</div></div>
        </div>
        """.strip()
        for index, entry in enumerate(entries)
    )
    st.markdown(f'<div class="timeline">{rows}</div>', unsafe_allow_html=True)


def _render_evidence(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        _empty(
            "No evidence has been received. Trigger an investigation to populate the case board."
        )
        return
    cards = "".join(
        f"""
        <div class="evidence-card" style="animation-delay:{index * 90}ms">
          <div class="evidence-id">{_escape(item["id"])} · {_escape(item.get("strength", "signal"))}</div>
          <div class="evidence-title">{_escape(item["title"])}</div>
          <div class="evidence-detail">{_escape(item["detail"])}</div>
          <div class="evidence-source">{_escape(item.get("source", "investigation artifact"))}</div>
        </div>
        """.strip()
        for index, item in enumerate(evidence)
    )
    st.markdown(f'<div class="evidence-grid">{cards}</div>', unsafe_allow_html=True)


def _render_suspects(suspects: list[dict[str, Any]]) -> None:
    if not suspects:
        _empty("No hypotheses generated yet.")
        return
    cards = "".join(
        f"""
        <div class="suspect {_escape(str(item["verdict"]).lower())}" style="animation-delay:{index * 90}ms">
          <div class="suspect-top"><span>{_escape(item["name"])}</span><span class="confidence">{item["confidence"]:.0f}% confidence</span></div>
          <div class="suspect-meta">{_escape(item["verdict"])} · {_escape(item["rationale"])}</div>
          <div class="conf-track"><div class="conf-fill {_status_colour(str(item["verdict"]))}" style="width:{min(max(item["confidence"], 2), 100):.0f}%"></div></div>
        </div>
        """.strip()
        for index, item in enumerate(suspects)
    )
    st.markdown(cards, unsafe_allow_html=True)


def _render_war_room(messages: list[dict[str, Any]]) -> None:
    if not messages:
        _empty("The War Room is waiting for the investigation agents to report in.")
        return
    transcript = "".join(
        f"""
        <div class="chat-row" style="animation-delay:{index * 140}ms">
          <div class="chat-avatar {_stance_colour(str(message.get("stance", "")))}">{message.get("icon", "🤖")}</div>
          <div class="chat-bubble">
            <div class="chat-head">
              <span class="chat-name">{_escape(message["agent"])}</span>
              <span class="chat-role">{_escape(message["role"])}</span>
              <span class="stance-chip {_stance_colour(str(message.get("stance", "")))}">{_escape(message.get("stance", "analysis"))}</span>
            </div>
            <div class="chat-copy">{_escape(message["message"])}</div>
          </div>
        </div>
        """.strip()
        for index, message in enumerate(messages)
    )
    st.markdown(transcript, unsafe_allow_html=True)


def _render_consensus(case: dict[str, Any]) -> None:
    suspects = case["suspects"]
    if not suspects:
        _empty("No consensus recorded yet.")
        return
    primary = suspects[0]
    confidence = min(max(_number(primary.get("confidence")), 0), 100)
    st.markdown(
        f"""
        <div class="verdict-card">
          <div class="prob-ring" style="background: conic-gradient(var(--mint) {confidence * 3.6:.0f}deg, rgba(255,255,255,.06) 0deg);">
            <div class="prob-core"><span class="prob-value">{confidence:.0f}%</span><span class="prob-caption">confidence</span></div>
          </div>
          <div>
            <div class="eyebrow">War room verdict</div>
            <div class="diagnosis-name">{_escape(primary["name"])}</div>
            <div class="diagnosis-copy">{_escape(primary["rationale"])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_doctor(case: dict[str, Any]) -> None:
    diagnosis = case["diagnosis"]
    treatment = case["treatment"]
    probability = min(max(_number(diagnosis.get("recovery_probability")), 0), 100)
    col_a, col_b = st.columns((1.1, 1), gap="large")
    with col_a:
        st.markdown(
            f"""
            <div class="doctor-card">
              <div class="eyebrow">Patient record · {_escape(diagnosis["patient"])}</div>
              <div class="diagnosis-name">{_escape(diagnosis["condition"])}</div>
              <div class="signal {_status_colour(diagnosis["severity"])}"><span class="dot"></span>{_escape(diagnosis["severity"])} severity</div>
              <div class="diagnosis-copy" style="margin-top:.5rem">{_escape(diagnosis["summary"])}</div>
              <div style="display:flex; align-items:center; gap:1rem; margin-top:1rem">
                <div class="prob-ring" style="flex:0 0 88px; height:88px; background: conic-gradient(var(--mint) {probability * 3.6:.0f}deg, rgba(255,255,255,.06) 0deg);">
                  <div class="prob-core" style="width:66px; height:66px;"><span class="prob-value" style="font-size:1.05rem">{probability:.0f}%</span></div>
                </div>
                <div class="metric-note">Estimated recovery probability, given the validated treatment plan and deterministic gates.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="section-label" style="margin-top:0">Treatment plan</div>',
            unsafe_allow_html=True,
        )
        if treatment:
            rows = "".join(
                f"""
                <div class="rx-card" style="animation-delay:{index * 110}ms"><div class="rx-num">{index:02d}</div><div><div class="rx-step">{_escape(item["step"])}</div><div class="rx-action">{_escape(item["action"])}</div></div></div>
                """.strip()
                for index, item in enumerate(treatment, start=1)
            )
            st.markdown(rows, unsafe_allow_html=True)
        else:
            _empty("No treatment plan has been authorised yet.")


def _render_experiments(case: dict[str, Any]) -> None:
    experiments = case["experiments"]
    if not experiments:
        _empty("No experiment runs have been attached to this case.")
        return
    cards = "".join(
        f"""
        <div class="experiment-card {"pass" if _status_colour(item["status"]) == "mint" else ""}" style="animation-delay:{index * 110}ms">
          <div class="experiment-version">{_escape(item["version"])} · {_escape(item["status"]).upper()}</div>
          <div class="experiment-change">{_escape(item["change"])}</div>
          <div class="exp-metrics"><div class="exp-metric">ACCURACY<strong>{item["accuracy"]:.0f}%</strong></div><div class="exp-metric">F1<strong>{item["f1"]:.2f}</strong></div><div class="exp-metric">RECALL<strong>{item["recall"]:.2f}</strong></div></div>
        </div>
        """.strip()
        for index, item in enumerate(experiments)
    )
    st.markdown(f'<div class="comparison">{cards}</div>', unsafe_allow_html=True)


def _render_deltas(metrics_compare: Mapping[str, Any]) -> None:
    baseline = _as_dict(metrics_compare.get("baseline"))
    candidate = _as_dict(metrics_compare.get("candidate"))
    if not any(_number(value) for value in candidate.values()):
        return
    chips: list[str] = []
    for index, name in enumerate(("f1", "recall", "precision", "accuracy")):
        delta = _number(candidate.get(name)) - _number(baseline.get(name))
        colour = "mint" if delta >= 0 else "coral"
        chips.append(
            f'<div class="delta-chip" style="animation-delay:{index * 80}ms">'
            f'<div class="delta-value {colour}">{delta * 100:+.1f}</div>'
            f'<div class="delta-label">{name} · pts</div></div>'
        )
    st.markdown(f'<div class="delta-grid">{"".join(chips)}</div>', unsafe_allow_html=True)


def _render_commit(case: dict[str, Any]) -> None:
    recovery = case["recovery"]
    raw_git = _as_dict(_as_dict(case.get("raw")).get("raw")).get("git")
    git = _as_dict(raw_git)
    sha = str(git.get("sha") or "").strip()
    mode = str(git.get("mode", "proposed"))
    line_two = (
        f'<div><span class="prompt">✓</span>[{_escape(sha)}] committed by the autonomous engineer</div>'
        if sha
        else f'<div class="t-muted"># mode: {_escape(mode)} — set SHERLOCKML_AUTOCOMMIT=1 for a real local commit</div>'
    )
    st.markdown(
        f"""
        <div class="terminal">
          <div class="terminal-bar"><span class="t-dot red"></span><span class="t-dot yellow"></span><span class="t-dot green"></span><span class="terminal-title">sherlockml · autonomous engineer</span></div>
          <div class="terminal-body">
            <div><span class="prompt">$</span>git commit -m "{_escape(recovery["git_commit"])}"</div>
            {line_two}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_artifacts(case: dict[str, Any]) -> None:
    artifacts = _as_dict(case.get("artifacts"))
    if not artifacts:
        return
    chips = "".join(
        f'<div class="artifact-chip" title="{_escape(path)}"><span>{_escape(name.replace("_", " "))}</span>{_escape(Path(str(path)).name)}</div>'
        for name, path in artifacts.items()
    )
    st.markdown('<div class="section-label">Case artifacts</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="artifact-grid">{chips}</div>', unsafe_allow_html=True)


def _render_validation_gates(gates: list[dict[str, Any]]) -> None:
    if not gates:
        return
    rows = "".join(
        f"""
        <div class="gate-row">
          <span class="signal {_status_colour('pass' if gate['passed'] else 'fail')}">
            <span class="dot"></span>{'PASS' if gate['passed'] else 'FAIL'}
          </span>
          <span class="gate-name">{_escape(gate['name'])}</span>
          <span class="gate-detail">{_escape(gate.get('detail', ''))}</span>
        </div>
        """.strip()
        for gate in gates
    )
    st.markdown(
        f'<div class="section-label">Validation gates</div><div class="gate-board">{rows}</div>',
        unsafe_allow_html=True,
    )


def _render_recovery(case: dict[str, Any]) -> None:
    health = case["health"]
    recovery = case["recovery"]
    col_a, col_b = st.columns((1.15, 1), gap="large")
    with col_a:
        st.markdown(
            '<div class="section-label" style="margin-top:0">Baseline vs candidate</div>',
            unsafe_allow_html=True,
        )
        candidate_metrics = _as_dict(_as_dict(case.get("metrics_compare")).get("candidate"))
        has_comparison = any(_number(value) for value in candidate_metrics.values())
        if _HAS_PLOTLY and has_comparison:
            _plot(_comparison_figure(case["metrics_compare"], case["experiments"]))
        else:
            _render_experiments(case)
        _render_deltas(case["metrics_compare"])
        _render_validation_gates(case["recovery"].get("gates", []))
    with col_b:
        st.markdown(
            '<div class="section-label" style="margin-top:0">Health trajectory</div>',
            unsafe_allow_html=True,
        )
        if _HAS_PLOTLY:
            _plot(_trajectory_figure(health))
        if recovery["approved"]:
            heading, colour = "RECOVERY APPROVED FOR HUMAN REVIEW", "mint"
        else:
            heading, colour = "RECOVERY NOT YET APPROVED", "gold"
        st.markdown(
            f"""
            <div class="approval">
              <div class="approval-title {colour}">{heading}</div>
              <div class="approval-copy">{_escape(recovery["summary"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-label">Autonomous engineering record</div>', unsafe_allow_html=True)
    left, right = st.columns((1.15, 1), gap="large")
    with left:
        _render_commit(case)
    with right:
        _render_experiments(case)
    _render_artifacts(case)


def _render_page(page: str, case: dict[str, Any]) -> None:
    _render_hero(case)
    if page == "Model Health":
        _render_stepper(case["timeline"])
        col_a, col_b = st.columns((1.1, 1), gap="large")
        with col_a:
            _render_avatar(case)
            st.markdown('<div class="section-label">Vital signs</div>', unsafe_allow_html=True)
            _render_metrics(case)
        with col_b:
            if _HAS_PLOTLY:
                _plot(_health_gauge_figure(case["health"]))
            else:
                _empty("Install plotly to display the model-health gauge.")
        st.markdown('<div class="section-label">Investigation pulse</div>', unsafe_allow_html=True)
        _render_timeline(case["timeline"])
        st.markdown('<div class="section-label">Leading hypothesis</div>', unsafe_allow_html=True)
        _render_suspects(case["suspects"][:2])
    elif page == "Case File":
        st.markdown(
            '<div class="section-label" style="margin-top:.2rem">Evidence board</div>',
            unsafe_allow_html=True,
        )
        _render_evidence(case["evidence"])
        left, right = st.columns((1, 1.1), gap="large")
        with left:
            st.markdown('<div class="section-label">Suspect ledger</div>', unsafe_allow_html=True)
            _render_suspects(case["suspects"])
        with right:
            st.markdown(
                '<div class="section-label">Feature drift fingerprint</div>',
                unsafe_allow_html=True,
            )
            drift_features = _as_list(_as_dict(case.get("drift")).get("features"))
            if drift_features and _HAS_PLOTLY:
                _plot(_drift_figure([_as_dict(item) for item in drift_features]))
            elif drift_features:
                _empty("Install plotly to display the PSI drift chart.")
            else:
                _empty("The statistician has not supplied feature diagnostics yet.")
        st.markdown('<div class="section-label">Chain of events</div>', unsafe_allow_html=True)
        _render_timeline(case["timeline"], compact=True)
    elif page == "War Room":
        st.markdown(
            '<div class="section-label" style="margin-top:.2rem">Live multi-agent transcript</div>',
            unsafe_allow_html=True,
        )
        _render_war_room(case["war_room"])
        st.markdown('<div class="section-label">Moderator conclusion</div>', unsafe_allow_html=True)
        _render_consensus(case)
    elif page == "Model Doctor":
        _render_doctor(case)
        st.markdown(
            '<div class="section-label">Evidence supporting treatment</div>', unsafe_allow_html=True
        )
        _render_evidence(case["evidence"][:3])
    elif page == "Recovery Results":
        _render_recovery(case)
        with st.expander("Raw case payload (debug)", expanded=False):
            st.code(
                json.dumps(case["raw"], indent=2, default=str)
                if case["raw"]
                else "No live payload received yet.",
                language="json",
            )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE, page_icon="🔎", layout="wide", initial_sidebar_state="expanded"
    )
    _inject_styles()

    if "case" not in st.session_state:
        st.session_state.case = None
    if "engine_notice" not in st.session_state:
        st.session_state.engine_notice = None

    with st.sidebar:
        st.markdown(
            "<div class='eyebrow'>SherlockML</div><div class='hero-title' style='font-size:1.42rem'>Incident Command</div>",
            unsafe_allow_html=True,
        )
        st.caption("Autonomous ML detective & doctor")
        st.divider()
        page = st.radio(
            "Command view",
            PAGES,
            label_visibility="collapsed",
            format_func=lambda name: f"{PAGE_ICONS.get(name, '•')}  {name}",
        )
        st.markdown("<div class='section-label'>Scenario control</div>", unsafe_allow_html=True)
        incident_type = st.selectbox(
            "Incident type",
            options=list(INCIDENTS),
            format_func=lambda key: INCIDENTS[key],
            help="Choose the failure injected into the synthetic production system.",
        )
        st.caption(INCIDENT_HINTS.get(incident_type, ""))
        strict_gates = st.checkbox(
            "Demo failed-recovery path (strict gates)",
            help="Raises the F1 improvement threshold so the candidate is rejected for demo purposes.",
        )
        if st.button("🔍 Investigate incident", type="primary", use_container_width=True):
            with st.status("Investigation in progress…", expanded=True) as status:
                def _progress(label: str) -> None:
                    status.write(label)

                case, notice = run_investigation(
                    incident_type, strict_gates=strict_gates, progress=_progress
                )
                status.update(label="Investigation complete", state="complete")
            st.session_state.case = case
            st.session_state.engine_notice = notice
            st.session_state.last_run_at = _now()
        if st.button("↺ Reset command room", use_container_width=True):
            reset_investigation()
            st.session_state.last_run_at = None
        st.divider()
        active_case = st.session_state.get("case")
        if active_case:
            approved = bool(_as_dict(active_case.get("recovery")).get("approved"))
            chip_colour = "mint" if approved else "gold"
            chip_text = "RECOVERY APPROVED" if approved else "UNDER INVESTIGATION"
            st.markdown(
                f"<span class='case-chip blue'>{_escape(active_case.get('case_id', 'CASE'))}</span>"
                f"<span class='case-chip {chip_colour}'>{chip_text}</span>",
                unsafe_allow_html=True,
            )
        if st.session_state.get("last_run_at"):
            st.caption(f"Last run · {st.session_state.last_run_at}")
        st.caption("The dashboard distinguishes live engine output from its UI-only preview.")

    if st.session_state.engine_notice:
        st.warning(st.session_state.engine_notice)

    if st.session_state.case is None:
        st.session_state.case = _preview_case("data_drift")
        st.info(
            "Select an incident in the command panel, then choose **Investigate incident** to run the connected core engine."
        )

    _render_page(page, st.session_state.case)


if __name__ == "__main__":
    main()
