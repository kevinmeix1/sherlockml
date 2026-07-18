"""Evidence-led statistical analysis for the SherlockML war room."""

from __future__ import annotations

from typing import Any


def analyze(diagnostics: dict[str, Any]) -> dict[str, Any]:
    ranked = diagnostics["features"]
    strongest = ranked[0]
    imbalance = diagnostics["class_imbalance"]
    correlation = diagnostics["correlation_change"]
    conclusion = (
        f"{strongest['feature']} is the strongest statistical signal: PSI={strongest['psi']:.2f} "
        f"and KS p={strongest['ks_pvalue']:.3g}."
    )
    return {
        "conclusion": conclusion,
        "strongest_feature": strongest["feature"],
        "psi": strongest["psi"],
        "ks_pvalue": strongest["ks_pvalue"],
        "class_imbalance": imbalance,
        "correlation_change": correlation,
        "finding": (
            f"Class prevalence moved by {imbalance['delta_percentage_points']:+.1f} percentage "
            "points; the evidence is consistent with a changed population, not random noise."
        ),
    }
