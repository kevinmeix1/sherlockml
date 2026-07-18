"""Small, JSON-friendly contracts shared by the specialist agents.

The application deliberately keeps the agent outputs inspectable.  There is no
hidden chain-of-thought: every conclusion cites the observable signals that
led to it.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TimelineEvent(TypedDict):
    at: str
    actor: str
    icon: str
    title: str
    detail: str


class Evidence(TypedDict):
    id: str
    source: str
    title: str
    finding: str
    strength: str
    value: str


class Suspect(TypedDict):
    name: str
    confidence: int
    rationale: str
    status: str


class WarRoomMessage(TypedDict):
    speaker: str
    role: str
    icon: str
    stance: str
    message: str


JsonDict = dict[str, Any]
