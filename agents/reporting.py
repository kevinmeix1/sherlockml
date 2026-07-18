"""Generate the final, evidence-linked incident report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts" / "reports"


def build_report(state: dict[str, Any]) -> dict[str, str]:
    incident = state["incident"]
    baseline = state["baseline"]
    experiment = state["experiment"]
    doctor = state["doctor"]
    consensus = state["war_room"]["consensus"]
    engineer = state["engineering_action"]
    validation = experiment["validation"]
    report = f"""# SherlockML Recovery Report — {state["case_id"]}

## Incident

- **Model:** Fraud Detection Model
- **Classification:** `{incident["kind"]}`
- **Severity:** HIGH
- **Status:** {doctor["decision"]}

## Diagnosis

**Primary cause:** {consensus["name"]} ({consensus["confidence"]}% confidence)

{doctor["bedside_note"]}

## Evidence

"""
    for evidence in state["evidence"]:
        report += f"- **{evidence['id']} — {evidence['title']}:** {evidence['finding']}\n"
    report += f"""

## Controlled engineering action

{engineer["summary"]}

Changed file: `{engineer["changed_file"]}`

"""
    for change in engineer["changes"]:
        report += f"- {change}\n"
    baseline_accuracy = baseline["accuracy"]
    candidate_accuracy = experiment["candidate"]["accuracy"]
    accuracy_delta = experiment["deltas"]["accuracy"]
    baseline_f1 = baseline["f1"]
    candidate_f1 = experiment["candidate"]["f1"]
    f1_delta = experiment["deltas"]["f1"]
    baseline_recall = baseline["recall"]
    candidate_recall = experiment["candidate"]["recall"]
    recall_delta = experiment["deltas"]["recall"]
    baseline_latency = baseline["latency_ms"]
    candidate_latency = experiment["candidate"]["latency_ms"]
    latency_delta = experiment["deltas"]["latency_ms"]
    report += f"""

## Experiment results

| Metric | Before | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | {baseline_accuracy:.3f} | {candidate_accuracy:.3f} | {accuracy_delta:+.3f} |
| F1 | {baseline_f1:.3f} | {candidate_f1:.3f} | {f1_delta:+.3f} |
| Recall | {baseline_recall:.3f} | {candidate_recall:.3f} | {recall_delta:+.3f} |
| Latency | {baseline_latency:.0f} ms | {candidate_latency:.0f} ms | {latency_delta:+.0f} ms |

## Validation gates

"""
    for gate in validation["gates"]:
        report += f"- {'PASS' if gate['passed'] else 'FAIL'} — {gate['name']}: {gate['detail']}\n"
    report += f"""

## Rollout recommendation

{doctor["decision"]}. Deploy to a shadow environment, monitor drift and parity
checks, then seek an accountable human's production approval.

## Git integration

Mode: `{engineer["git"]["mode"]}`  
Proposed message: `{engineer["git"]["message"]}`  
{engineer["git"]["note"]}
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{state['case_id'].lower()}-recovery-report.md"
    report_path.write_text(report)
    machine_path = REPORTS_DIR / f"{state['case_id'].lower()}-recovery-report.json"
    machine_path.write_text(json.dumps(_machine_summary(state), indent=2) + "\n")
    return {"markdown": report, "path": str(report_path), "machine_path": str(machine_path)}


def _machine_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": state["case_id"],
        "incident": state["incident"],
        "diagnosis": state["war_room"]["consensus"],
        "experiment": state["experiment"],
        "doctor": state["doctor"],
        "engineering_action": {
            "changes": state["engineering_action"]["changes"],
            "changed_file": state["engineering_action"]["changed_file"],
            "git": state["engineering_action"]["git"],
        },
    }
