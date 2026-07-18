"""Synthetic, deterministic fraud data for the SherlockML incident lab.

The data deliberately resembles a production table without using any real
customer information.  ``fraud_probability`` is the latent model target and
``fraud_label`` is the sampled observed outcome used to train/evaluate models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

NUMERIC_FEATURES: tuple[str, ...] = (
    "transaction_amount",
    "transaction_frequency",
    "customer_age",
    "account_age_days",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "merchant_category",
    "location",
    "device_type",
)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMNS: tuple[str, ...] = ("fraud_probability", "fraud_label")

_MERCHANTS = np.array(["grocery", "electronics", "travel", "luxury", "gaming"])
_LOCATIONS = np.array(["GB", "US", "DE", "BR", "NG"])
_DEVICES = np.array(["mobile", "desktop", "tablet"])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _risk_lookup(values: np.ndarray, mapping: dict[str, float]) -> np.ndarray:
    return np.array([mapping[str(value)] for value in values], dtype=float)


def generate_fraud_data(
    rows: int = 3_000,
    *,
    seed: int = 2026,
    start_time: str | datetime = "2026-01-01T00:00:00",
    profile: Literal["healthy", "drifted"] = "healthy",
) -> pd.DataFrame:
    """Generate a temporally ordered, deterministic synthetic fraud dataset.

    ``healthy`` represents the original deployed population.  ``drifted``
    represents a later behavioural regime: larger baskets, more rapid
    transactions, and a changed fraud mechanism.  This makes a genuine
    retraining experiment meaningful rather than merely changing a chart.
    """

    if rows < 50:
        raise ValueError("rows must be at least 50 to support model evaluation")
    if profile not in {"healthy", "drifted"}:
        raise ValueError("profile must be either 'healthy' or 'drifted'")

    rng = np.random.default_rng(seed)
    event_time = pd.Timestamp(start_time) + pd.to_timedelta(
        np.arange(rows) * 5, unit="min"
    )

    if profile == "healthy":
        transaction_amount = np.clip(rng.lognormal(3.95, 0.62, rows), 2, 1_500)
        transaction_frequency = rng.poisson(2.4, rows) + 1
        customer_age = np.clip(rng.normal(42, 13, rows), 18, 85)
        account_age_days = np.clip(rng.gamma(2.3, 260, rows), 3, 4_500)
        merchant_category = rng.choice(
            _MERCHANTS, rows, p=[0.36, 0.18, 0.12, 0.08, 0.26]
        )
        location = rng.choice(_LOCATIONS, rows, p=[0.44, 0.25, 0.12, 0.11, 0.08])
        device_type = rng.choice(_DEVICES, rows, p=[0.61, 0.29, 0.10])

        log_odds = (
            # Tuned to a realistic-but-learnable synthetic fraud prevalence
            # (roughly 8–15%), so F1 is meaningful in the recovery demo.
            -1.55
            + 0.015 * transaction_amount
            + 0.33 * transaction_frequency
            - 0.005 * account_age_days
            + _risk_lookup(
                merchant_category,
                {
                    "grocery": -0.35,
                    "electronics": 0.45,
                    "travel": 0.65,
                    "luxury": 0.95,
                    "gaming": 0.3,
                },
            )
            + _risk_lookup(
                location,
                {"GB": -0.25, "US": 0.0, "DE": -0.1, "BR": 0.45, "NG": 0.95},
            )
            + _risk_lookup(device_type, {"mobile": 0.2, "desktop": -0.15, "tablet": 0.05})
            + np.where(customer_age < 24, 0.3, 0.0)
        )
    else:
        # New customer behaviour after the incident.  The amount distribution
        # shifts materially and fraud is now driven much more by transaction
        # velocity than by large basket size.  That is a realistic enough
        # concept/covariate shift to expose a stale production model.
        transaction_amount = np.clip(rng.lognormal(5.35, 0.55, rows), 8, 2_500)
        transaction_frequency = rng.poisson(7.4, rows) + 1
        customer_age = np.clip(rng.normal(37, 12, rows), 18, 85)
        account_age_days = np.clip(rng.gamma(1.8, 170, rows), 2, 4_500)
        merchant_category = rng.choice(
            _MERCHANTS, rows, p=[0.18, 0.25, 0.13, 0.18, 0.26]
        )
        location = rng.choice(_LOCATIONS, rows, p=[0.30, 0.20, 0.08, 0.18, 0.24])
        device_type = rng.choice(_DEVICES, rows, p=[0.78, 0.14, 0.08])

        log_odds = (
            # Drifted fraud prevalence stays operationally realistic while the
            # original high-amount model now over-flags most transactions.
            -8.50
            - 0.0020 * transaction_amount
            + 0.83 * transaction_frequency
            - 0.003 * account_age_days
            + _risk_lookup(
                merchant_category,
                {
                    "grocery": -0.55,
                    "electronics": 0.55,
                    "travel": 0.35,
                    "luxury": 0.7,
                    "gaming": 0.95,
                },
            )
            + _risk_lookup(
                location,
                {"GB": -0.35, "US": -0.1, "DE": -0.25, "BR": 0.55, "NG": 1.05},
            )
            + _risk_lookup(device_type, {"mobile": 0.52, "desktop": -0.25, "tablet": 0.0})
            + np.where(customer_age < 25, 0.25, 0.0)
        )

    # A sharper latent risk surface gives the synthetic model enough signal to
    # establish a clearly healthy baseline.  Labels remain sampled outcomes,
    # rather than a leaked deterministic rule, so the incident still requires
    # real modelling and evaluation work.
    fraud_probability = _sigmoid(2.0 * log_odds + 1.3)
    fraud_label = rng.binomial(1, fraud_probability).astype(int)

    return pd.DataFrame(
        {
            "event_time": event_time,
            "transaction_amount": transaction_amount.round(2),
            "transaction_frequency": transaction_frequency.astype(int),
            "customer_age": customer_age.round(1),
            "merchant_category": merchant_category,
            "location": location,
            "device_type": device_type,
            "account_age_days": account_age_days.round(1),
            "fraud_probability": fraud_probability.round(6),
            "fraud_label": fraud_label,
        }
    )
