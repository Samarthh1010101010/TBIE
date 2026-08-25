"""
src/utils/velocity.py
─────────────────────
Within-snapshot velocity features — the single implementation.

These six features compare a member's last 30 days against their own 90-day
baseline, so they capture how fast someone is moving rather than where they
currently are.

Why this module exists
──────────────────────
`add_velocity_features` was implemented twice — once in
src/08_transition_prediction.py and once in pipeline.py — and the two had
drifted on NaN handling for `recency_days`:

    training  : recency_days NaN -> NaN -> later .fillna(0)  => recency_risk 0
    inference : recency_days .fillna(999) first              => recency_risk 999

`recency_days` is null for members who have never purchased (1,814 of 500,000
at 2025-12-31). The training path scored them 0, which reads as "purchased
today, no lapse risk". The inference path scored them 999, which reads as
"maximum lapse risk". Same member, opposite signal, depending on which side of
the pipeline they passed through.

999 is the correct sentinel: never-purchased is the high-risk end, and it is
already what src/utils/state_rules.py uses for the same column. That behaviour
is adopted here for both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sentinel for "has never purchased". Matches state_rules._col(..., fill=999.0).
NEVER_PURCHASED_RECENCY = 999.0

VELOCITY_COLS = [
    "spend_velocity",      # 30d spend vs 90d/3 — below 1.0 means slowing down
    "freq_velocity",       # 30d purchase frequency vs 90d/3
    "app_velocity",        # 30d app opens vs 90d/3
    "recency_risk",        # recency x (1 - min(freq/10, 1)) — high = lapse incoming
    "engagement_score",    # total digital touches in 30d (app + email + push)
    "spend_decline_flag",  # 1 if spending is below half the 90d pace
]

VELOCITY_CLIP_MAX = 10.0   # outlier guard on the three ratio features


def _s(df: pd.DataFrame, name: str, fill: float = 0.0) -> pd.Series:
    """Column as float with NaN filled; a zero series if the column is absent."""
    if name in df.columns:
        return df[name].fillna(fill).astype("float64")
    return pd.Series(fill, index=df.index, dtype="float64")


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the six velocity features. Fully vectorised.

    Returns a copy; the input frame is not modified.
    """
    eps = 1e-6
    out = df.copy()

    spend_30 = _s(out, "spend_total_30d")
    spend_90 = _s(out, "spend_total_90d")
    freq_30  = _s(out, "purchase_count_30d")
    freq_90  = _s(out, "purchase_count_90d")
    app_30   = _s(out, "app_open_30d")
    app_90   = _s(out, "app_open_90d")
    email_30 = _s(out, "email_open_30d")
    push_30  = _s(out, "push_open_30d")
    recency  = _s(out, "recency_days", fill=NEVER_PURCHASED_RECENCY)

    out["spend_velocity"] = spend_30 / (spend_90 / 3 + eps)
    out["freq_velocity"]  = freq_30  / (freq_90  / 3 + eps)
    out["app_velocity"]   = app_30   / (app_90   / 3 + eps)

    # Long since the last purchase AND low recent frequency = lapse signal.
    out["recency_risk"] = recency * np.clip(1.0 - freq_30 / 10.0, 0.0, 1.0)

    out["engagement_score"] = app_30 + email_30 + push_30

    # Computed after the clip below would change its meaning, so derive it from
    # the unclipped ratio: below half the 90-day pace.
    out["spend_decline_flag"] = (out["spend_velocity"] < 0.5).astype("float32")

    for col in ("spend_velocity", "freq_velocity", "app_velocity"):
        out[col] = out[col].clip(0, VELOCITY_CLIP_MAX)

    return out
