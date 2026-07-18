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
from typing import Any

import streamlit as st

APP_TITLE = "SherlockML | ML Incident Command"
PAGES = (
    "Model Health",
    "Case File",
    "War Room",
    "Model Doctor",
    "Recovery Results",
)
INCIDENTS = {
    "data_drift": "Data drift",
    "feature_pipeline_bug": "Feature pipeline bug",
    "model_regression": "Model regression",
}


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")


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
            "before": 92.0,
            "after_incident": 61.0,
            "recovered": 90.0,
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
                "message": "The first reliable clue is a sharp shift in the high-value transaction cohort.",
            },
            {
                "agent": "Statistician",
                "role": "Distribution analyst",
                "stance": "analysis",
                "message": "PSI and KS both reject the healthy-distribution hypothesis. Drift is material.",
            },
            {
                "agent": "ML Engineer",
                "role": "Pipeline owner",
                "stance": "challenge",
                "message": "The pipeline contract changed too. I would not call it pure data drift yet.",
            },
            {
                "agent": "Infra",
                "role": "Reliability engineer",
                "stance": "infra",
                "message": "Serving latency is elevated but stable; the incident is not a platform outage.",
            },
            {
                "agent": "Moderator",
                "role": "Decision maker",
                "stance": "decision",
                "message": "Consensus: drift is primary, contract mismatch is a contributor. Authorize a guarded repair experiment.",
            },
        ],
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
        message["message"] = str(
            _first(message, "message", "text", "detail", default="No message supplied.")
        )

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


def _call_maybe_async(function: Callable[..., Any], *args: Any) -> Any:
    outcome = function(*args)
    if inspect.isawaitable(outcome):
        import asyncio

        return asyncio.run(_resolve_awaitable(outcome))
    return outcome


def run_investigation(incident_type: str) -> tuple[dict[str, Any], str | None]:
    """Run the core engine, or return a transparent preview if it is absent."""

    investigate, _, issue = _resolve_engine()
    if investigate is None:
        return _preview_case(incident_type), issue
    try:
        return normalise_investigation(_call_maybe_async(investigate, incident_type)), None
    except Exception as error:  # Keep the UX usable and expose an actionable error.
        return _preview_case(
            incident_type
        ), f"Investigation engine raised {error.__class__.__name__}: {error}"


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
          .sherlock-hero { position: relative; overflow: hidden; padding: 1.65rem 1.85rem 1.55rem; background: linear-gradient(112deg, rgba(23, 52, 60, .93), rgba(10, 27, 34, .88)); border: 1px solid rgba(137, 199, 255, .24); border-radius: 18px; margin-bottom: 1.15rem; box-shadow: 0 18px 48px rgba(0,0,0,.22); }
          .sherlock-hero:after { content:''; position:absolute; width: 320px; height: 320px; right:-105px; top:-185px; border-radius:50%; border: 1px solid rgba(93,224,181,.22); box-shadow: 0 0 0 34px rgba(93,224,181,.03), 0 0 0 70px rgba(93,224,181,.025); }
          .eyebrow { font-family:'DM Mono', monospace; color: var(--mint); letter-spacing:.13em; font-size:.72rem; text-transform:uppercase; }
          .hero-title { margin:.25rem 0 .1rem; font-family:'Playfair Display', serif; font-size:2.25rem; color:var(--paper); letter-spacing:-.035em; }
          .hero-subtitle { color:#b7c8c7; font-size:.96rem; max-width: 47rem; }
          .case-chip { display:inline-block; margin-top:.75rem; padding:.28rem .6rem; border:1px solid rgba(93,224,181,.34); border-radius:999px; color:var(--mint); background:rgba(93,224,181,.08); font: .68rem 'DM Mono', monospace; letter-spacing:.06em; }
          .case-chip.coral { border-color:rgba(255,116,98,.5); color:var(--coral); background:rgba(255,116,98,.08); }
          .section-label { font: .72rem 'DM Mono', monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--mint); margin:1.55rem 0 .6rem; }
          .metric-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.72rem; }
          .metric-card { min-height:108px; padding: .9rem .95rem; background: rgba(13, 34, 42, .83); border:1px solid var(--line); border-radius: 13px; }
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
          .timeline-row { display:grid; grid-template-columns:4.45rem 1.3rem 1fr; gap:.5rem; align-items:start; padding:.47rem 0; }
          .timeline-time { color:var(--gold); font: .68rem 'DM Mono', monospace; padding-top:.16rem; }
          .timeline-node { width:.63rem; height:.63rem; margin-top:.18rem; border-radius:50%; background:var(--ink); border:2px solid var(--mint); z-index:1; }
          .timeline-phase { color:var(--mint); font: .64rem 'DM Mono', monospace; letter-spacing:.08em; }
          .timeline-text { color:#d5e0df; font-size:.88rem; line-height:1.42; margin-top:.1rem; }
          .evidence-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.78rem; }
          .evidence-card { position:relative; padding:1rem; min-height:150px; background:linear-gradient(150deg, rgba(17,41,48,.91), rgba(8,23,29,.86)); border:1px solid var(--line); border-radius:13px; }
          .evidence-id { color:var(--gold); font: .65rem 'DM Mono', monospace; letter-spacing:.1em; }
          .evidence-title { color:var(--paper); font-weight:600; margin:.56rem 0 .4rem; }
          .evidence-detail { color:#b3c1c1; font-size:.81rem; line-height:1.4; }
          .evidence-source { position:absolute; bottom:.78rem; color:#6f8d91; font: .62rem 'DM Mono', monospace; text-transform:uppercase; }
          .suspect { margin:.48rem 0; padding:.75rem .85rem; background:rgba(14,36,43,.74); border-left:3px solid var(--blue); border-radius:0 10px 10px 0; }
          .suspect.primary { border-left-color:var(--coral); } .suspect.contributing { border-left-color:var(--gold); }
          .suspect-top { display:flex; justify-content:space-between; align-items:baseline; gap:.8rem; color:var(--paper); font-weight:600; }
          .confidence { color:var(--gold); font: .73rem 'DM Mono', monospace; white-space:nowrap; }
          .suspect-meta { color:#9eafb1; font-size:.77rem; margin-top:.22rem; }
          .thread { display:flex; gap:.75rem; padding:.78rem 0; border-bottom:1px solid var(--line); }
          .thread:last-child { border-bottom:0; }
          .agent-seal { flex:0 0 2.35rem; height:2.35rem; display:flex; align-items:center; justify-content:center; border-radius:50%; background:rgba(137,199,255,.1); border:1px solid rgba(137,199,255,.28); color:var(--blue); font: .72rem 'DM Mono', monospace; }
          .thread-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:.5rem; }
          .thread-agent { color:var(--paper); font-weight:600; }
          .thread-role { color:#7f9da1; font:.68rem 'DM Mono', monospace; }
          .thread-copy { color:#c5d2d1; font-size:.88rem; line-height:1.48; margin-top:.22rem; }
          .doctor-card { padding:1.2rem; background:linear-gradient(135deg, rgba(20,55,52,.82), rgba(8,24,30,.88)); border:1px solid rgba(93,224,181,.22); border-radius:14px; }
          .diagnosis-name { margin:.35rem 0; color:var(--paper); font:600 1.45rem 'Playfair Display',serif; }
          .diagnosis-copy { color:#c0d0ce; line-height:1.53; font-size:.9rem; }
          .rx-card { display:grid; grid-template-columns: 2.1rem 1fr; gap:.7rem; padding:.8rem 0; border-bottom:1px solid var(--line); }
          .rx-card:last-child { border-bottom:0; }
          .rx-num { color:var(--mint); font:.75rem 'DM Mono',monospace; padding-top:.14rem; }
          .rx-step { color:var(--paper); font-weight:600; font-size:.9rem; }
          .rx-action { color:#acbdbc; font-size:.81rem; line-height:1.42; margin-top:.13rem; }
          .comparison { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
          .experiment-card { padding:1rem; border-radius:13px; border:1px solid var(--line); background:rgba(13,34,42,.78); }
          .experiment-card.pass { border-color:rgba(93,224,181,.35); background:rgba(93,224,181,.07); }
          .experiment-version { color:var(--blue); font:.67rem 'DM Mono',monospace; letter-spacing:.08em; }
          .experiment-change { color:var(--paper); font-weight:600; margin:.35rem 0 .75rem; }
          .exp-metrics { display:flex; gap:1rem; flex-wrap:wrap; }
          .exp-metric { color:#9baeb0; font:.66rem 'DM Mono',monospace; }
          .exp-metric strong { display:block; color:var(--paper); font:600 1.15rem 'DM Sans',sans-serif; margin-top:.1rem; }
          .approval { padding:1.1rem 1.2rem; border:1px solid rgba(93,224,181,.34); border-radius:13px; background:rgba(93,224,181,.08); }
          .approval-title { color:var(--mint); font:.74rem 'DM Mono',monospace; letter-spacing:.09em; }
          .approval-copy { color:#dceae6; margin-top:.35rem; font-size:.9rem; }
          .empty-state { padding:1.4rem; border:1px dashed rgba(137,199,255,.3); border-radius:12px; color:#aebfc0; font-size:.88rem; }
          .stButton > button { width:100%; border-radius:10px; border:1px solid rgba(93,224,181,.45); background:rgba(93,224,181,.1); color:#e8fffa; font-family:'DM Mono',monospace; font-size:.76rem; letter-spacing:.04em; }
          .stButton > button:hover { border-color:var(--mint); color:var(--mint); background:rgba(93,224,181,.16); }
          .stAlert { border-radius:10px; }
          @media (max-width: 800px) { .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .evidence-grid { grid-template-columns:1fr; } .comparison { grid-template-columns:1fr; } .hero-title { font-size:1.8rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(case: dict[str, Any]) -> None:
    source = (
        "LIVE INVESTIGATION" if case["source"] != "dashboard-preview" else "INTERACTIVE PREVIEW"
    )
    colour = "coral" if "critical" in str(case["health"]["status"]).lower() else "mint"
    st.markdown(
        f"""
        <div class="sherlock-hero">
          <div class="eyebrow">SherlockML / autonomous ML incident command</div>
          <div class="hero-title">{_escape(case["incident_title"])}</div>
          <div class="hero-subtitle">A detective-and-doctor workflow for evidence, diagnosis, reversible treatment, and validated recovery.</div>
          <span class="case-chip {colour}">{_escape(case["case_id"])} · {source}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, note: str, colour: str = "blue") -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{_escape(label)}</div>'
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
            _metric_card("Patient", health["model_name"], "production model", "blue"),
            _metric_card(
                "Health on alert",
                f"{health['after_incident']:.0f}%",
                f"{deterioration:+.0f} pts from baseline",
                "coral",
            ),
            _metric_card(
                "Recovery signal",
                f"{health['recovered']:.0f}%",
                f"{recovery:+.0f} pts candidate lift",
                "mint",
            ),
            _metric_card(
                "Serving latency",
                f"{health['latency_ms']:.0f} ms" if health["latency_ms"] else "—",
                case["status"],
                _status_colour(case["status"]),
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
        <div class="timeline-row">
          <div class="timeline-time">{_escape(entry["time"])}</div>
          <div class="timeline-node"></div>
          <div><div class="timeline-phase">{_escape(entry["phase"])}</div><div class="timeline-text">{_escape(entry["text"])}</div></div>
        </div>
        """.strip()
        for entry in entries
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
        <div class="evidence-card">
          <div class="evidence-id">{_escape(item["id"])} · {_escape(item.get("strength", "signal"))}</div>
          <div class="evidence-title">{_escape(item["title"])}</div>
          <div class="evidence-detail">{_escape(item["detail"])}</div>
          <div class="evidence-source">{_escape(item.get("source", "investigation artifact"))}</div>
        </div>
        """.strip()
        for item in evidence
    )
    st.markdown(f'<div class="evidence-grid">{cards}</div>', unsafe_allow_html=True)


def _render_suspects(suspects: list[dict[str, Any]]) -> None:
    if not suspects:
        _empty("No hypotheses generated yet.")
        return
    cards = "".join(
        f"""
        <div class="suspect {_escape(str(item["verdict"]).lower())}">
          <div class="suspect-top"><span>{_escape(item["name"])}</span><span class="confidence">{item["confidence"]:.0f}% confidence</span></div>
          <div class="suspect-meta">{_escape(item["verdict"])} · {_escape(item["rationale"])}</div>
        </div>
        """.strip()
        for item in suspects
    )
    st.markdown(cards, unsafe_allow_html=True)


def _render_war_room(messages: list[dict[str, Any]]) -> None:
    if not messages:
        _empty("The War Room is waiting for the investigation agents to report in.")
        return
    transcript = "".join(
        f"""
        <div class="thread">
          <div class="agent-seal">{_escape(str(message["agent"])[:2].upper())}</div>
          <div><div class="thread-head"><span class="thread-agent">{_escape(message["agent"])}</span><span class="thread-role">{_escape(message["role"])}</span></div><div class="thread-copy">{_escape(message["message"])}</div></div>
        </div>
        """.strip()
        for message in messages
    )
    st.markdown(transcript, unsafe_allow_html=True)


def _render_doctor(case: dict[str, Any]) -> None:
    diagnosis = case["diagnosis"]
    treatment = case["treatment"]
    col_a, col_b = st.columns((1.1, 1), gap="large")
    with col_a:
        st.markdown(
            f"""
            <div class="doctor-card">
              <div class="eyebrow">Patient record · {_escape(diagnosis["patient"])}</div>
              <div class="diagnosis-name">{_escape(diagnosis["condition"])}</div>
              <div class="signal {_status_colour(diagnosis["severity"])}"><span class="dot"></span>{_escape(diagnosis["severity"])} severity</div>
              <div class="diagnosis-copy">{_escape(diagnosis["summary"])}</div>
              <div class="metric-note" style="margin-top:.85rem">Estimated recovery probability · <span class="mint">{diagnosis["recovery_probability"]:.0f}%</span></div>
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
                <div class="rx-card"><div class="rx-num">{index:02d}</div><div><div class="rx-step">{_escape(item["step"])}</div><div class="rx-action">{_escape(item["action"])}</div></div></div>
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
        <div class="experiment-card {"pass" if _status_colour(item["status"]) == "mint" else ""}">
          <div class="experiment-version">{_escape(item["version"])} · {_escape(item["status"]).upper()}</div>
          <div class="experiment-change">{_escape(item["change"])}</div>
          <div class="exp-metrics"><div class="exp-metric">ACCURACY<strong>{item["accuracy"]:.0f}%</strong></div><div class="exp-metric">F1<strong>{item["f1"]:.2f}</strong></div><div class="exp-metric">RECALL<strong>{item["recall"]:.2f}</strong></div></div>
        </div>
        """.strip()
        for item in experiments
    )
    st.markdown(f'<div class="comparison">{cards}</div>', unsafe_allow_html=True)


def _render_recovery(case: dict[str, Any]) -> None:
    recovery = case["recovery"]
    experiments = case["experiments"]
    health = case["health"]
    col_a, col_b = st.columns((1.15, 1), gap="large")
    with col_a:
        _render_experiments(case)
    with col_b:
        if recovery["approved"]:
            heading = "RECOVERY APPROVED FOR HUMAN REVIEW"
            colour = "mint"
        else:
            heading = "RECOVERY NOT YET APPROVED"
            colour = "gold"
        st.markdown(
            f"""
            <div class="approval">
              <div class="approval-title {colour}">{heading}</div>
              <div class="approval-copy">{_escape(recovery["summary"])}</div>
              <div class="metric-note" style="margin-top:.7rem">Health trajectory · {health["before"]:.0f}% → {health["after_incident"]:.0f}% → {health["recovered"]:.0f}%</div>
              <div class="metric-note" style="margin-top:.35rem">Git artifact · {_escape(recovery["git_commit"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if experiments:
            baseline, candidate = experiments[0], experiments[-1]
            delta = candidate["f1"] - baseline["f1"]
            st.markdown(
                f"<div class='section-label'>Measured gain</div><div class='metric-card'><div class='metric-label'>F1 movement</div><div class='metric-value'>{delta:+.2f}</div><div class='metric-note'>from {baseline['version']} to {candidate['version']}</div></div>",
                unsafe_allow_html=True,
            )


def _render_page(page: str, case: dict[str, Any]) -> None:
    _render_hero(case)
    if page == "Model Health":
        _render_metrics(case)
        st.markdown('<div class="section-label">Investigation pulse</div>', unsafe_allow_html=True)
        _render_timeline(case["timeline"])
        st.markdown('<div class="section-label">Leading hypothesis</div>', unsafe_allow_html=True)
        _render_suspects(case["suspects"][:2])
    elif page == "Case File":
        st.markdown('<div class="section-label">Evidence board</div>', unsafe_allow_html=True)
        _render_evidence(case["evidence"])
        left, right = st.columns((1.1, 1), gap="large")
        with left:
            st.markdown('<div class="section-label">Suspect ledger</div>', unsafe_allow_html=True)
            _render_suspects(case["suspects"])
        with right:
            st.markdown('<div class="section-label">Chain of events</div>', unsafe_allow_html=True)
            _render_timeline(case["timeline"], compact=True)
    elif page == "War Room":
        st.markdown(
            '<div class="section-label">Live multi-agent transcript</div>', unsafe_allow_html=True
        )
        _render_war_room(case["war_room"])
        st.markdown('<div class="section-label">Moderator conclusion</div>', unsafe_allow_html=True)
        _render_suspects(case["suspects"][:1])
    elif page == "Model Doctor":
        _render_doctor(case)
        st.markdown(
            '<div class="section-label">Evidence supporting treatment</div>', unsafe_allow_html=True
        )
        _render_evidence(case["evidence"][:3])
    elif page == "Recovery Results":
        st.markdown(
            '<div class="section-label">Recovery experiment comparison</div>',
            unsafe_allow_html=True,
        )
        _render_recovery(case)

    with st.expander("View incident payload", expanded=False):
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
        page = st.radio("Command view", PAGES, label_visibility="collapsed")
        st.markdown("<div class='section-label'>Scenario control</div>", unsafe_allow_html=True)
        incident_type = st.selectbox(
            "Incident type",
            options=list(INCIDENTS),
            format_func=lambda key: INCIDENTS[key],
            help="Choose the failure injected into the synthetic production system.",
        )
        if st.button("Investigate incident", type="primary", use_container_width=True):
            with st.spinner(
                "Agents are collecting evidence, debating suspects, and validating a treatment…"
            ):
                case, notice = run_investigation(incident_type)
            st.session_state.case = case
            st.session_state.engine_notice = notice
            st.session_state.last_run_at = _now()
        if st.button("Reset command room", use_container_width=True):
            reset_investigation()
            st.session_state.last_run_at = None
        st.divider()
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
