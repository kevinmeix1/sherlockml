# SherlockML dashboard

Run the command room from the repository root:

```bash
streamlit run dashboard/app.py
```

The page dynamically imports `api.main.run_investigation(incident_type)`. It
does not need the API to exist while the rest of SherlockML is being built: a
visually distinct `dashboard-preview` case is shown instead, and the UI tells
the user that it is not a live investigation.

## Recommended API contract

`run_investigation` may be a normal or `async` Python function and should
return a dictionary. The dashboard accepts aliases for many fields, but this
shape produces the richest presentation:

```python
{
    "source": "live-engine",
    "case_id": "CASE-001",
    "incident_type": "data_drift",
    "incident_title": "Feature distribution shift",
    "status": "Investigation complete",
    "health": {
        "before": 92.0,
        "after_incident": 61.0,
        "recovered": 90.0,
        "status": "critical",
        "model_name": "Fraud Sentinel / xgboost-v17",
        "latency_ms": 112.0,
    },
    "evidence": [{
        "id": "E-01", "title": "Performance signal", "detail": "…",
        "strength": "high", "source": "production metrics",
    }],
    "suspects": [{
        "name": "Feature distribution shift", "confidence": 86,
        "verdict": "primary", "rationale": "…",
    }],
    "timeline": [{"time": "10:01", "phase": "ALERT", "text": "…"}],
    "war_room": [{
        "agent": "Detective", "role": "Lead investigator",
        "stance": "evidence", "message": "…",
    }],
    "diagnosis": {
        "patient": "Fraud Sentinel", "condition": "Feature distribution shift",
        "severity": "High", "summary": "…", "recovery_probability": 94,
    },
    "treatment": [{"step": "Stabilize", "action": "…"}],
    "experiments": [{
        "version": "v18-candidate", "change": "…", "accuracy": 91.0,
        "f1": 0.86, "recall": 0.84, "status": "passed",
    }],
    "recovery": {
        "approved": True, "summary": "…", "git_commit": "fix: …",
    },
}
```

Metric values can be fractions (`0.86`) or percentages (`86`); accuracy and
health are normalised to percentages. `api.main.reset_demo()` or
`api.main.reset_state()` is optional. If exposed, the Reset command invokes it
in addition to clearing the dashboard session.
