"""A transparent, deliberately adversarial war-room moderator."""

from __future__ import annotations

from typing import Any

from agents.contracts import WarRoomMessage


def convene(
    incident: dict[str, Any],
    statistician: dict[str, Any],
    infra: dict[str, Any],
    suspects: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = next(suspect for suspect in suspects if suspect["status"] == "primary")
    dissent = next(suspect for suspect in suspects if suspect["status"] != "primary")
    messages: list[WarRoomMessage] = [
        {
            "speaker": "Sherlock",
            "role": "Detective Agent",
            "icon": "🕵️",
            "stance": "opens case",
            "message": (
                f"The highest-confidence suspect is {primary['name'].lower()} "
                f"({primary['confidence']}%)."
            ),
        },
        {
            "speaker": "Ada",
            "role": "Statistician Agent",
            "icon": "📊",
            "stance": "evidence",
            "message": statistician["conclusion"],
        },
        {
            "speaker": "Linus",
            "role": "ML Engineer Agent",
            "icon": "👨‍💻",
            "stance": "dissent",
            "message": (
                f"I want to rule out {dissent['name'].lower()} before touching production. "
                "I will inspect the versioned feature contract and test the smallest safe repair."
            ),
        },
        {
            "speaker": "SRE-7",
            "role": "Infra Agent",
            "icon": "☁️",
            "stance": "operational check",
            "message": infra["conclusion"],
        },
        {
            "speaker": "Moriarty",
            "role": "War Room Moderator",
            "icon": "⚔️",
            "stance": "consensus",
            "message": (
                f"Consensus reached: {primary['name']} is the primary cause for incident "
                f"type `{incident['kind']}`. Proceed with a reversible repair and validation gate."
            ),
        },
    ]
    return {"messages": messages, "consensus": primary, "dissent": dissent}
