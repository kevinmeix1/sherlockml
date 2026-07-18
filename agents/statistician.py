"""Evidence-led statistical analysis for the SherlockML war room."""

from __future__ import annotations

from typing import Any

_MATERIAL_PSI = 0.25
_WARNING_PSI = 0.10


def analyze(diagnostics: dict[str, Any]) -> dict[str, Any]:
    ranked = list(diagnostics.get("features", []))
    if not ranked:
        return {
            "conclusion": "No feature diagnostics were supplied for this case.",
            "strongest_feature": "unknown",
            "psi": 0.0,
            "ks_pvalue": 1.0,
            "class_imbalance": diagnostics.get("class_imbalance", {}),
            "correlation_change": diagnostics.get("correlation_change"),
            "drift_detected": False,
            "finding": (
                "Awaiting labelled production samples before drawing a population conclusion."
            ),
        }

    ranked.sort(key=lambda item: float(item.get("psi", 0)), reverse=True)
    strongest = ranked[0]
    imbalance = diagnostics["class_imbalance"]
    correlation = diagnostics.get("correlation_change")
    psi = float(strongest.get("psi", 0))
    feature_type = str(strongest.get("type", "numeric"))
    ks_pvalue = strongest.get("ks_pvalue")
    drift_detected = psi >= _WARNING_PSI or any(
        item.get("severity") in {"warning", "critical"} for item in ranked
    )

    if psi < _WARNING_PSI:
        conclusion = (
            "No feature exceeds the 0.25 PSI material-drift threshold; "
            f"the strongest signal is {strongest['feature']} at PSI={psi:.2f}."
        )
        finding = (
            f"Class prevalence moved by {imbalance['delta_percentage_points']:+.1f} percentage "
            "points, but feature-level drift is stable — look to model metadata and contracts."
        )
    elif feature_type == "categorical" or ks_pvalue is None:
        conclusion = (
            f"{strongest['feature']} shows categorical PSI={psi:.2f} "
            f"({'above' if psi >= _MATERIAL_PSI else 'below'} the 0.25 material threshold); "
            "KS is not applicable for categorical features."
        )
        finding = (
            f"Category frequencies shifted on {strongest['feature']}; "
            f"class prevalence moved {imbalance['delta_percentage_points']:+.1f} pts."
        )
    else:
        conclusion = (
            f"{strongest['feature']} is the strongest statistical signal: PSI={psi:.2f} "
            f"({'above' if psi >= _MATERIAL_PSI else 'below'} 0.25 material threshold) "
            f"and KS p={float(ks_pvalue):.3g}."
        )
        finding = (
            f"Class prevalence moved by {imbalance['delta_percentage_points']:+.1f} percentage "
            "points; distributional evidence is consistent with a changed population."
            if drift_detected
            else "Population prevalence moved slightly without strong per-feature drift."
        )

    return {
        "conclusion": conclusion,
        "strongest_feature": strongest["feature"],
        "psi": psi,
        "ks_pvalue": float(ks_pvalue) if ks_pvalue is not None else None,
        "class_imbalance": imbalance,
        "correlation_change": correlation,
        "drift_detected": drift_detected,
        "finding": finding,
    }
