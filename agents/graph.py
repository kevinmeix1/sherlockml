"""LangGraph orchestration for a bounded SherlockML investigation.

The graph deliberately carries an inspectable case record rather than a chat
transcript.  Each node calls one narrow specialist function and adds its
observable output to the case state.  The final state is converted into the
stable dashboard contract by :func:`to_dashboard_contract`.

No node deploys a model.  The only mutation is the local, reviewable pipeline
contract repair performed by ``agents.engineer``; callers should use
``reset_local_state`` to restore the deterministic demo baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

SUPPORTED_INCIDENTS = ("data_drift", "pipeline_bug", "model_regression")

_INCIDENT_ALIASES = {
    "data_drift": "data_drift",
    "data-drift": "data_drift",
    "pipeline_bug": "pipeline_bug",
    "pipeline-bug": "pipeline_bug",
    "feature_pipeline_bug": "pipeline_bug",
    "feature-pipeline-bug": "pipeline_bug",
    "model_regression": "model_regression",
    "model-regression": "model_regression",
}

_CASE_IDS = {
    "data_drift": "CASE-DRIFT-001",
    "pipeline_bug": "CASE-PIPELINE-001",
    "model_regression": "CASE-REGRESSION-001",
}

_INCIDENT_TITLES = {
    "data_drift": "Feature distribution shift",
    "pipeline_bug": "Feature transformation defect",
    "model_regression": "Model release regression",
}


class InvestigationState(TypedDict, total=False):
    """Data that crosses the supervisor graph's node boundaries."""

    incident_type: str
    case_id: str
    source: str
    simulation: Any
    incident: dict[str, Any]
    baseline: dict[str, Any]
    healthy: dict[str, Any]
    health: dict[str, Any]
    diagnostics: dict[str, Any]
    operations: dict[str, Any]
    pipeline: dict[str, Any]
    evidence: list[dict[str, Any]]
    suspects: list[dict[str, Any]]
    statistician: dict[str, Any]
    infra: dict[str, Any]
    war_room: dict[str, Any]
    engineering_action: dict[str, Any]
    experiment: dict[str, Any]
    doctor: dict[str, Any]
    tracking: dict[str, Any]
    report: dict[str, Any]
    timeline: list[dict[str, str]]
    status: str


def normalize_incident_type(incident_type: str) -> str:
    """Return a canonical incident name or raise a useful validation error."""

    normalised = incident_type.strip().lower().replace(" ", "_")
    canonical = _INCIDENT_ALIASES.get(normalised)
    if canonical is None:
        supported = ", ".join(SUPPORTED_INCIDENTS)
        raise ValueError(
            f"Unsupported incident type '{incident_type}'. Choose one of: {supported}."
        )
    return canonical


def reset_local_state() -> dict[str, str]:
    """Restore the mutable demo contract without deleting prior case artifacts."""

    from agents.engineer import restore_pipeline_contract

    restored = restore_pipeline_contract()
    return {
        "status": "reset",
        "message": "Restored the deterministic baseline pipeline contract.",
        "contract": str(restored),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        # IncidentSimulation is intentionally an in-memory dataclass because
        # its dataframes are not API payloads.  ``vars`` keeps those references
        # intact for the experiment node without attempting an expensive deep
        # serialisation.
        return dict(vars(value))
    return {}


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        data = _as_dict(value)
        if data:
            return data
    return {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_fraction(value: Any, default: float = 0.0) -> float:
    """Accept either a percentage or a 0..1 metric and return a fraction."""

    metric = _number(value, default)
    return metric / 100 if metric > 1 else metric


def _timeline(state: InvestigationState, phase: str, text: str) -> list[dict[str, str]]:
    """Append a fixed-clock event so a fresh default run is reproducible."""

    events = list(state.get("timeline", []))
    events.append({"time": f"10:{len(events) + 1:02d}", "phase": phase, "text": text})
    return events


def _canonical_incident(simulation: dict[str, Any], incident_type: str) -> dict[str, Any]:
    metadata = _as_dict(simulation.get("metadata"))
    incident = _first_mapping(simulation.get("incident"), metadata.get("incident"))
    incident.setdefault("kind", incident_type)
    incident.setdefault("title", metadata.get("title", _INCIDENT_TITLES[incident_type]))
    incident.setdefault("symptom", metadata.get("symptom", "A model-health policy was breached."))
    incident.setdefault("severity", "high")
    return incident


def _canonical_diagnostics(simulation: dict[str, Any], incident_type: str) -> dict[str, Any]:
    diagnostics = _as_dict(simulation.get("diagnostics"))
    raw_features = diagnostics.get("features", diagnostics.get("feature_diagnostics"))
    features: list[dict[str, Any]] = []
    if isinstance(raw_features, list):
        for raw_feature in raw_features:
            feature = _as_dict(raw_feature)
            if not feature:
                continue
            # ``ml.drift.build_diagnostics`` uses ``feature_diagnostics`` and
            # categorical rows legitimately have no KS value.  The specialist
            # statistician expects a numeric field because it prints it in the
            # War Room narrative, so a neutral p-value is the honest bridge.
            feature["ks_pvalue"] = _number(feature.get("ks_pvalue"), 1.0)
            feature["psi"] = _number(feature.get("psi"))
            feature.setdefault("severity", "stable")
            features.append(feature)
    if not features:
        features = [
            {
                "feature": "transaction_amount",
                "psi": 0.31 if incident_type == "data_drift" else 0.08,
                "ks_pvalue": 0.002 if incident_type == "data_drift" else 0.31,
            }
        ]
    features.sort(key=lambda item: _number(item.get("psi")), reverse=True)
    diagnostics["features"] = features

    raw_imbalance = _as_dict(diagnostics.get("class_imbalance"))
    reference_imbalance = _as_dict(raw_imbalance.get("reference"))
    current_imbalance = _as_dict(raw_imbalance.get("current"))
    if reference_imbalance or current_imbalance:
        reference_rate = _metric_fraction(reference_imbalance.get("positive_rate", 0.0))
        current_rate = _metric_fraction(current_imbalance.get("positive_rate", 0.0))
        diagnostics["class_imbalance"] = {
            "baseline_rate": reference_rate,
            "current_rate": current_rate,
            "delta_percentage_points": round((current_rate - reference_rate) * 100, 2),
        }
    else:
        diagnostics.setdefault(
            "class_imbalance",
            {"delta_percentage_points": 0.0, "baseline_rate": 0.0, "current_rate": 0.0},
        )

    correlations = diagnostics.get("correlation_change", diagnostics.get("correlation_changes"))
    diagnostics["correlation_change"] = (
        correlations or "No material correlation signal was supplied."
    )
    strongest = features[0]
    summary = _as_dict(diagnostics.get("summary"))
    diagnostics.setdefault(
        "headline",
        {
            "data_drift": ("Production transaction behavior is outside the training distribution."),
            "pipeline_bug": (
                "Serving feature transformations no longer match the training contract."
            ),
            "model_regression": (
                "The deployed model recipe no longer matches the validated champion."
            ),
        }.get(
            incident_type,
            f"{strongest['feature']} is the strongest distributional evidence in this case.",
        ),
    )
    diagnostics.setdefault(
        "headline_value",
        (
            f"PSI={_number(strongest.get('psi')):.2f} · "
            f"{summary.get('feature_count_checked', len(features))} features checked"
        ),
    )
    diagnostics.setdefault("severity", str(strongest.get("severity", "high")))
    return diagnostics


def _canonical_baseline(simulation: dict[str, Any]) -> dict[str, Any]:
    baseline_bundle = _as_dict(simulation.get("baseline"))
    baseline = _first_mapping(
        baseline_bundle.get("incident_metrics"),
        simulation.get("baseline"),
        simulation.get("before"),
        _as_dict(simulation.get("metadata")).get("incident_metrics"),
    )
    for name in ("accuracy", "f1", "precision", "recall", "latency_ms"):
        baseline.setdefault(name, 0.0)
    return baseline


def _canonical_healthy(simulation: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(simulation.get("metadata"))
    baseline_bundle = _as_dict(simulation.get("baseline"))
    healthy = _first_mapping(
        baseline_bundle.get("healthy_metrics"),
        simulation.get("healthy"),
        simulation.get("healthy_baseline"),
        metadata.get("healthy"),
        metadata.get("healthy_metrics"),
    )
    if not healthy:
        healthy = dict(baseline)
        healthy["f1"] = max(_metric_fraction(baseline.get("f1")), 0.82)
        healthy["accuracy"] = max(_metric_fraction(baseline.get("accuracy")), 0.88)
        healthy["recall"] = max(_metric_fraction(baseline.get("recall")), 0.74)
    return healthy


def _canonical_operations(simulation: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(simulation.get("metadata"))
    operations = _first_mapping(simulation.get("operations"), metadata.get("operations"))
    operations.setdefault("latency_ms", baseline.get("latency_ms", 112.0) or 112.0)
    operations.setdefault("error_rate", 0.002)
    operations.setdefault("memory_mb", 512.0)
    return operations


def _observation_node(state: InvestigationState) -> dict[str, Any]:
    """Create a reproducible incident snapshot and initial evidence board."""

    from agents.detective import investigate
    from agents.engineer import inspect_pipeline
    from simulator.incidents import simulate_incident

    incident_type = normalize_incident_type(state["incident_type"])
    simulation_payload = simulate_incident(incident_type)
    context = getattr(simulation_payload, "context", None)
    simulation = _as_dict(context()) if callable(context) else _as_dict(simulation_payload)
    baseline = _canonical_baseline(simulation)
    healthy = _canonical_healthy(simulation, baseline)
    diagnostics = _canonical_diagnostics(simulation, incident_type)
    operations = _canonical_operations(simulation, baseline)
    pipeline = inspect_pipeline()
    incident = _canonical_incident(simulation, incident_type)
    metadata = _as_dict(simulation.get("metadata"))
    health = {
        "healthy_f1": _metric_fraction(
            healthy.get("f1", metadata.get("healthy_f1", baseline.get("f1", 0.0)))
        ),
        "f1_floor": _metric_fraction(metadata.get("f1_floor", 0.72)),
    }
    case_file = investigate(
        {
            "incident": incident,
            "baseline": baseline,
            "health": health,
            "diagnostics": diagnostics,
            "pipeline": pipeline,
        }
    )
    return {
        "case_id": _CASE_IDS[incident_type],
        "source": "live-engine",
        "simulation": simulation_payload,
        "incident": incident,
        "baseline": baseline,
        "healthy": healthy,
        "health": health,
        "diagnostics": diagnostics,
        "operations": operations,
        "pipeline": pipeline,
        "evidence": list(case_file["evidence"]),
        "suspects": list(case_file["suspects"]),
        "timeline": _timeline(
            state,
            "ALERT",
            (
                f"Synthetic {incident_type.replace('_', ' ')} monitor crossed "
                "its reliability threshold."
            ),
        ),
        "status": "Evidence collection in progress",
    }


def _statistics_node(state: InvestigationState) -> dict[str, Any]:
    from agents.statistician import analyze

    statistician = analyze(state["diagnostics"])
    return {
        "statistician": statistician,
        "timeline": _timeline(
            state,
            "ANALYZE",
            (
                f"Statistician ranked {statistician['strongest_feature']} "
                "as the strongest data signal."
            ),
        ),
    }


def _infra_node(state: InvestigationState) -> dict[str, Any]:
    from agents.infra import inspect

    infrastructure = inspect(state["operations"])
    return {
        "infra": infrastructure,
        "timeline": _timeline(
            state,
            "RULE OUT",
            "Infrastructure checks completed; serving telemetry was added to the case file.",
        ),
    }


def _war_room_node(state: InvestigationState) -> dict[str, Any]:
    from agents.moderator import convene

    war_room = convene(state["incident"], state["statistician"], state["infra"], state["suspects"])
    return {
        "war_room": war_room,
        "timeline": _timeline(
            state,
            "DEBATE",
            "The War Room compared the competing root-cause hypotheses and recorded a consensus.",
        ),
    }


def _engineering_node(state: InvestigationState) -> dict[str, Any]:
    from agents.engineer import apply_repair

    action = apply_repair(state["case_id"], state["incident"]["kind"])
    return {
        "engineering_action": action,
        "timeline": _timeline(
            state,
            "REPAIR",
            (
                "The ML Engineer wrote a bounded pipeline-contract patch and "
                "preserved its diff artifact."
            ),
        ),
    }


def _canonical_experiment(
    raw_experiment: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Adapt the training module's portable result to reporting's stable schema."""

    # The training module keeps the fitted candidate artifact in memory for
    # programmatic callers.  A recovery report must remain JSON serialisable,
    # so retain its public metadata and measurements—not the estimator object.
    serializable_raw = {
        key: value
        for key, value in raw_experiment.items()
        if key not in {"candidate_artifact", "champion_artifact", "artifacts"}
    }
    before = _first_mapping(raw_experiment.get("before"), raw_experiment.get("baseline"), baseline)
    after = _first_mapping(raw_experiment.get("after"), raw_experiment.get("candidate"))
    improvement = _first_mapping(raw_experiment.get("improvement"), raw_experiment.get("deltas"))
    candidate = dict(after)
    for metric in ("accuracy", "f1", "precision", "recall", "latency_ms"):
        before.setdefault(metric, baseline.get(metric, 0.0))
        candidate.setdefault(metric, before.get(metric, 0.0))
        improvement.setdefault(metric, _number(candidate[metric]) - _number(before[metric]))

    approved = bool(
        raw_experiment.get(
            "approved",
            _as_dict(raw_experiment.get("validation")).get("approved", False),
        )
    )
    validation = _as_dict(raw_experiment.get("validation"))
    validation.setdefault("approved", approved)
    validation.setdefault("gates", raw_experiment.get("gates", []))
    validation.setdefault(
        "summary",
        "Candidate cleared deterministic recovery gates."
        if approved
        else "Candidate did not clear all deterministic recovery gates.",
    )
    return {
        **serializable_raw,
        "baseline": before,
        "before": before,
        "candidate": candidate,
        "after": candidate,
        "deltas": improvement,
        "improvement": improvement,
        "validation": validation,
        "approved": approved,
        "experiment_rows": raw_experiment.get("experiment_rows", []),
    }


def _experiment_node(state: InvestigationState) -> dict[str, Any]:
    from ml.train import run_repair_experiment

    raw_experiment = _as_dict(run_repair_experiment(state["simulation"]))
    experiment = _canonical_experiment(raw_experiment, state["baseline"])
    return {
        "experiment": experiment,
        "timeline": _timeline(
            state,
            "VERIFY",
            (
                "The candidate was trained and evaluated against the same "
                "deterministic incident fixture."
            ),
        ),
    }


def _doctor_node(state: InvestigationState) -> dict[str, Any]:
    from agents.doctor import prescribe

    doctor = prescribe(state["war_room"]["consensus"], state["experiment"], state["incident"])
    return {
        "doctor": doctor,
        "status": "Recovery approved for human review"
        if state["experiment"]["validation"]["approved"]
        else "Recovery requires further human investigation",
        "timeline": _timeline(
            state,
            "DECIDE",
            "Model Doctor recorded the treatment plan and a human-review recommendation.",
        ),
    }


def _tracking_node(state: InvestigationState) -> dict[str, Any]:
    from agents.tracker import track_experiment

    tracking = track_experiment(
        state["case_id"],
        state["incident"]["kind"],
        state["experiment"]["baseline"],
        state["experiment"]["candidate"],
        state["engineering_action"]["changes"],
    )
    return {
        "tracking": tracking,
        "timeline": _timeline(
            state,
            "RECORD",
            "The experiment comparison and local evidence record were persisted for review.",
        ),
    }


def _report_node(state: InvestigationState) -> dict[str, Any]:
    from agents.reporting import build_report

    report = build_report(cast(dict[str, Any], state))
    return {
        "report": report,
        "timeline": _timeline(
            state,
            "REPORT",
            "SherlockML generated an evidence-linked recovery report and artifact paths.",
        ),
    }


def build_investigation_graph() -> Any:
    """Build the deterministic supervisor graph without a hidden LLM dependency."""

    workflow = StateGraph(InvestigationState)
    workflow.add_node("observe", _observation_node)
    workflow.add_node("statistician", _statistics_node)
    workflow.add_node("infra", _infra_node)
    workflow.add_node("war_room", _war_room_node)
    workflow.add_node("engineer", _engineering_node)
    workflow.add_node("experiment", _experiment_node)
    workflow.add_node("doctor", _doctor_node)
    workflow.add_node("tracker", _tracking_node)
    workflow.add_node("report", _report_node)
    workflow.add_edge(START, "observe")
    workflow.add_edge("observe", "statistician")
    workflow.add_edge("statistician", "infra")
    workflow.add_edge("infra", "war_room")
    workflow.add_edge("war_room", "engineer")
    workflow.add_edge("engineer", "experiment")
    workflow.add_edge("experiment", "doctor")
    workflow.add_edge("doctor", "tracker")
    workflow.add_edge("tracker", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def run_case(incident_type: str) -> dict[str, Any]:
    """Execute one named incident through the complete LangGraph workflow."""

    canonical = normalize_incident_type(incident_type)
    graph = build_investigation_graph()
    result = graph.invoke({"incident_type": canonical, "timeline": []})
    return dict(cast(Mapping[str, Any], result))


def _health_score(metrics: Mapping[str, Any]) -> float:
    """A small transparent health score: F1 expressed as a percentage."""

    return round(_metric_fraction(metrics.get("f1", 0.0)) * 100, 1)


def _experiment_cards(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    experiment = _as_dict(state.get("experiment"))
    baseline = _as_dict(experiment.get("baseline"))
    candidate = _as_dict(experiment.get("candidate"))
    approved = bool(_as_dict(experiment.get("validation")).get("approved", False))
    return [
        {
            "version": "v17-production",
            "change": "Production baseline under the synthetic incident.",
            "accuracy": _health_score({"f1": baseline.get("accuracy", 0.0)}),
            "f1": _metric_fraction(baseline.get("f1", 0.0)),
            "recall": _metric_fraction(baseline.get("recall", 0.0)),
            "status": "failed",
        },
        {
            "version": "v18-candidate",
            "change": "Bounded contract repair and deterministic retraining.",
            "accuracy": _health_score({"f1": candidate.get("accuracy", 0.0)}),
            "f1": _metric_fraction(candidate.get("f1", 0.0)),
            "recall": _metric_fraction(candidate.get("recall", 0.0)),
            "status": "passed" if approved else "needs review",
        },
    ]


def to_dashboard_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project graph state into the public contract documented by the dashboard."""

    incident = _as_dict(state.get("incident"))
    baseline = _as_dict(state.get("baseline"))
    healthy = _as_dict(state.get("healthy"))
    experiment = _as_dict(state.get("experiment"))
    candidate = _as_dict(experiment.get("candidate"))
    validation = _as_dict(experiment.get("validation"))
    doctor = _as_dict(state.get("doctor"))
    engineering = _as_dict(state.get("engineering_action"))
    war_room = _as_dict(state.get("war_room"))
    report = _as_dict(state.get("report"))
    pipeline = _as_dict(state.get("pipeline"))
    kind = str(incident.get("kind", state.get("incident_type", "data_drift")))
    approved = bool(validation.get("approved", False))

    evidence: list[dict[str, Any]] = []
    for item in state.get("evidence", []):
        clue = _as_dict(item)
        evidence.append(
            {
                "id": str(clue.get("id", f"E-{len(evidence) + 1:02d}")),
                "title": str(clue.get("title", "Investigation evidence")),
                "detail": str(clue.get("finding", clue.get("detail", "No narrative supplied."))),
                "strength": str(clue.get("strength", "medium")),
                "source": str(clue.get("source", "SherlockML")),
            }
        )

    suspects: list[dict[str, Any]] = []
    for item in state.get("suspects", []):
        suspect = _as_dict(item)
        suspects.append(
            {
                "name": str(suspect.get("name", "Unclassified hypothesis")),
                "confidence": _number(suspect.get("confidence", 0)),
                "verdict": str(suspect.get("status", suspect.get("verdict", "open"))),
                "rationale": str(suspect.get("rationale", "Awaiting evidence.")),
            }
        )

    messages: list[dict[str, Any]] = []
    for item in war_room.get("messages", []):
        message = _as_dict(item)
        messages.append(
            {
                "agent": str(message.get("speaker", message.get("agent", "Agent"))),
                "role": str(message.get("role", "Investigation specialist")),
                "stance": str(message.get("stance", "analysis")),
                "icon": str(message.get("icon", "🤖")),
                "message": str(message.get("message", "No message supplied.")),
            }
        )

    diagnostics = _as_dict(state.get("diagnostics"))
    drift_features: list[dict[str, Any]] = []
    for item in diagnostics.get("features", []):
        feature = _as_dict(item)
        if not feature:
            continue
        drift_features.append(
            {
                "feature": str(feature.get("feature", "unknown")),
                "type": str(feature.get("type", "numeric")),
                "psi": _number(feature.get("psi")),
                "ks_pvalue": _number(feature.get("ks_pvalue"), 1.0),
                "severity": str(feature.get("severity", "stable")),
                "missing_rate_current": _number(feature.get("missing_rate_current")),
            }
        )

    metric_names = ("accuracy", "precision", "recall", "f1")
    experiment_before = _as_dict(experiment.get("before")) or baseline
    metrics_compare = {
        "baseline": {
            name: _metric_fraction(experiment_before.get(name, 0.0)) for name in metric_names
        },
        "candidate": {name: _metric_fraction(candidate.get(name, 0.0)) for name in metric_names},
    }

    treatments: list[dict[str, Any]] = []
    for index, action in enumerate(doctor.get("treatment", []), start=1):
        treatments.append({"step": f"Step {index}", "action": str(action)})

    health_after = _health_score(baseline)
    recovered = _health_score(candidate)
    health_before = _health_score(healthy)
    diagnosis_summary = str(
        doctor.get(
            "bedside_note",
            (
                "The investigation linked the primary signal to "
                f"{incident.get('title', _INCIDENT_TITLES.get(kind, kind))}."
            ),
        )
    )
    git = _as_dict(engineering.get("git"))
    return {
        "source": str(state.get("source", "live-engine")),
        "case_id": str(state.get("case_id", _CASE_IDS.get(kind, "CASE-001"))),
        "incident_type": kind,
        "incident_title": str(incident.get("title", _INCIDENT_TITLES.get(kind, kind))),
        "status": str(state.get("status", "Investigation complete")),
        "health": {
            "before": health_before,
            "after_incident": health_after,
            "recovered": recovered,
            "status": "recovered" if approved else "critical",
            "model_name": "Fraud Sentinel / deterministic-v17",
            "latency_ms": _number(baseline.get("latency_ms", 0.0)),
        },
        "evidence": evidence,
        "suspects": suspects,
        "timeline": list(state.get("timeline", [])),
        "war_room": messages,
        "drift": {"features": drift_features},
        "metrics_compare": metrics_compare,
        "diagnosis": {
            "patient": str(doctor.get("patient", "Fraud Detection Model")),
            "condition": str(
                doctor.get("condition", incident.get("title", "ML reliability incident"))
            ),
            "severity": str(doctor.get("severity", incident.get("severity", "HIGH"))),
            "summary": diagnosis_summary,
            "recovery_probability": _number(doctor.get("recovery_probability", 0)),
        },
        "treatment": treatments,
        "experiments": _experiment_cards(state),
        "recovery": {
            "approved": approved,
            "summary": str(
                validation.get(
                    "summary",
                    doctor.get("bedside_note", "Candidate outcome requires human review."),
                )
            ),
            "git_commit": str(
                git.get("message", engineering.get("summary", "No repair artifact recorded."))
            ),
        },
        "artifacts": {
            "report": report.get("path"),
            "report_json": report.get("machine_path"),
            "patch": engineering.get("patch_path"),
            "experiment": _as_dict(state.get("tracking")).get("local_record"),
            "pipeline_contract": pipeline.get("file"),
        },
        "raw": {
            "incident": incident,
            "validation": validation,
            "git": git,
        },
    }
