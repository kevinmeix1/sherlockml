"""SRE-style operational checks used to rule in or rule out serving failures."""

from __future__ import annotations

from typing import Any


def inspect(operations: dict[str, Any]) -> dict[str, Any]:
    latency = operations["latency_ms"]
    error_rate = operations["error_rate"]
    memory = operations["memory_mb"]
    healthy = latency < 150 and error_rate < 0.01 and memory < 900
    return {
        "latency_ms": latency,
        "error_rate": error_rate,
        "memory_mb": memory,
        "conclusion": (
            "Serving infrastructure is within its operating envelope; no API or resource event "
            "explains the model-quality loss."
            if healthy
            else "Infrastructure signals require follow-up alongside the model investigation."
        ),
        "clears_infrastructure": healthy,
    }
