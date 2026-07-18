"""A transparent, deliberately adversarial war-room moderator."""

from __future__ import annotations

from typing import Any

from agents.contracts import WarRoomMessage


def convene(
    incident: dict[str, Any],
    statistician: dict[str, Any],
    infra: dict[str, Any],
    suspects: list[dict[str, Any]],
    pipeline: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = sorted(suspects, key=lambda item: item["confidence"], reverse=True)
    primary = next(suspect for suspect in ranked if suspect["status"] == "primary")
    dissent = next(
        (suspect for suspect in ranked if suspect["status"] != "primary"),
        ranked[1] if len(ranked) > 1 else ranked[0],
    )
    margin = primary["confidence"] - dissent["confidence"]
    pipeline = pipeline or {}
    baseline = baseline or {}
    contract = pipeline.get("contract", {})
    active = baseline.get("active_model", {})
    champion = baseline.get("champion_metadata", {})
    contract_version = contract.get("version", "unknown")
    active_version = active.get("preprocessing_version", "unknown")
    versions_match = contract_version == active_version or not active_version

    messages: list[WarRoomMessage] = [
        {
            "speaker": "Sherlock",
            "role": "Detective Agent",
            "icon": "🕵️",
            "stance": "opens case",
            "message": (
                f"The highest-confidence suspect is {primary['name'].lower()} "
                f"({primary['confidence']}% computed from evidence)."
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
            "speaker": "Linus",
            "role": "ML Engineer Agent",
            "icon": "👨‍💻",
            "stance": "contract check",
            "message": _contract_check_message(
                contract_version, active_version, versions_match, contract, active, champion
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
                f"Consensus reached: {primary['name']} leads by {margin:.0f} points over "
                f"{dissent['name'].lower()} for `{incident['kind']}`. "
                "Proceed with a reversible repair and explicit validation gates."
            ),
        },
    ]
    return {"messages": messages, "consensus": primary, "dissent": dissent, "margin": margin}


def _contract_check_message(
    contract_version: str,
    active_version: str,
    versions_match: bool,
    contract: dict[str, Any],
    active: dict[str, Any],
    champion: dict[str, Any],
) -> str:
    contract_features = set(contract.get("selected_features", []))
    active_features = set(active.get("feature_columns", []))
    champion_features = set(champion.get("feature_columns", []))
    if not versions_match:
        return (
            f"Contract `{contract_version}` does not match the active preprocessor "
            f"`{active_version}` — this supports a pipeline-contract investigation."
        )
    if active_features and champion_features and active_features != champion_features:
        return (
            f"Active model uses {len(active_features)} features while the champion validated "
            f"{len(champion_features)} — regression evidence without distributional drift."
        )
    if len(contract_features) < 7:
        return (
            f"Contract `{contract_version}` still omits "
            f"{7 - len(contract_features)} behaviour features from the serving schema."
        )
    return (
        f"Contract `{contract_version}` matches the active preprocessor version; "
        "no serving-schema mismatch detected on this check."
    )
