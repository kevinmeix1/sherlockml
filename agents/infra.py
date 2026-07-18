"""SRE-style operational checks used to rule in or rule out serving failures."""

from __future__ import annotations

from typing import Any


def inspect(operations: dict[str, Any]) -> dict[str, Any]:
    latency = float(operations.get("latency_ms", 0))
    error_rate = float(operations.get("error_rate", 0))
    memory = float(operations.get("memory_mb", 0))
    cpu = float(operations.get("cpu_pct", 0))
    note = str(operations.get("note", ""))

    latency_warning = latency >= 120
    error_warning = error_rate >= 0.005
    memory_warning = memory >= 800
    healthy = latency < 150 and error_rate < 0.01 and memory < 900

    if healthy and not latency_warning:
        conclusion = (
            "Serving infrastructure is within its operating envelope "
            f"(latency {latency:.0f} ms, error rate {error_rate:.3%}, memory {memory:.0f} MB); "
            "no platform event explains the model-quality loss."
        )
    elif healthy and latency_warning:
        conclusion = (
            f"Latency is mildly elevated at {latency:.0f} ms but error rate ({error_rate:.3%}) "
            "and memory remain healthy — infrastructure is unlikely to be the primary cause."
        )
    else:
        conclusion = (
            "Infrastructure signals require follow-up: "
            f"latency {latency:.0f} ms, error rate {error_rate:.3%}, memory {memory:.0f} MB."
        )

    if note:
        conclusion = f"{note} {conclusion}"

    return {
        "latency_ms": latency,
        "error_rate": error_rate,
        "memory_mb": memory,
        "cpu_pct": cpu,
        "conclusion": conclusion,
        "clears_infrastructure": healthy,
        "latency_warning": latency_warning,
        "error_warning": error_warning,
        "memory_warning": memory_warning,
    }
