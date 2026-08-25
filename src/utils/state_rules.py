"""
src/utils/state_rules.py
────────────────────────
THE canonical 10-state lifecycle cascade.

This module is the single source of truth for state classification. Both the
offline phase (src/07_lifecycle_states.py, which produces the training labels)
and the inference path (pipeline.py) import from here.

Why this module exists
──────────────────────
The cascade used to be implemented twice — once in Phase 7 and once inline in
pipeline.py — and the two copies drifted:

    rule                Phase 7 (training)        pipeline.py (inference)
    Value Maximizer     redeem > 0.10             redeem >= 0.21
    Momentum Builder    slope > 5.0               slope >= 5.0
    Plateau Cruiser     no recency clause         recency <= 30 required

Because `y_curr` (the current state) is a feature of the transition model, a
drifted inference-side cascade feeds the model a different label distribution
than it was trained on. The numbers in METHODOLOGY.md describe the Phase 7
version, so that is the behaviour preserved here.

Ordering is priority order: the first matching rule wins. STATE_IDS encodes
that same priority (New & Uncertain = 1 … Program Skeptic = 10) and is the
encoding written into the model bundle, so training and inference agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Canonical state list, in priority order ──────────────────────────────────
STATE_ORDER = [
    "New & Uncertain",
    "Win-Back Target",
    "Lapse Risk",
    "Momentum Builder",
    "Brand Advocate",
    "Redemption Hunter",
    "Value Maximizer",
    "Silent Accumulator",
    "Plateau Cruiser",
    "Program Skeptic",     # catch-all default
]

# Priority encoding. Must match the state_id column in states/*.parquet.
STATE_IDS = {name: i for i, name in enumerate(STATE_ORDER, start=1)}

VALID_STATES = set(STATE_ORDER)

# ── Thresholds ───────────────────────────────────────────────────────────────
# Named constants rather than magic numbers so METHODOLOGY.md §4 and this file
# can be diffed line by line.
TENURE_NEW_MAX          = 90     # New & Uncertain: account younger than 90d
RECENCY_WINBACK_MIN     = 60     # Win-Back Target: lapsed 60+ days
RECENCY_LAPSE_MIN       = 30     # Lapse Risk / Momentum Builder boundary
MOMENTUM_SLOPE_MIN      = 5.0    # Momentum Builder: spend slope above this
MOMENTUM_FREQ_MIN       = 2      # Momentum Builder: at least 2 purchases in 30d
MOMENTUM_DIVERSITY_MIN  = 0.3
ADVOCATE_EMAIL_MIN      = 2
ADVOCATE_APP_MIN        = 3
ADVOCATE_TIER_MIN       = 2      # gold or above
ADVOCATE_DIVERSITY_MIN  = 0.4
REDEMPTION_RATE_MIN     = 0.30   # Redemption Hunter
MAXIMIZER_DIVERSITY_MIN = 0.50
MAXIMIZER_REDEEM_MIN    = 0.10   # NOT 0.21 — see module docstring
PLATEAU_SLOPE_MIN       = -2.0
PLATEAU_SLOPE_MAX       = 5.0


def _col(df: pd.DataFrame, name: str, fill: float = 0.0) -> np.ndarray:
    """Fetch a column as float, filling NaN and missing columns alike."""
    if name in df.columns:
        return df[name].fillna(fill).to_numpy(dtype=float)
    return np.full(len(df), fill, dtype=float)


def build_conditions(df: pd.DataFrame) -> list[np.ndarray]:
    """
    Boolean condition per state, in priority order, excluding the catch-all.

    Returned list is len(STATE_ORDER) - 1; anything matching none of them is
    Program Skeptic. Exposed separately so callers that need exclusive masks
    (e.g. Phase 7's rule_fired strings) can derive them without duplicating
    the rule logic.
    """
    p30     = _col(df, "purchase_count_30d")
    tenure  = _col(df, "tenure_days",  fill=999.0)
    recency = _col(df, "recency_days", fill=999.0)   # 999 = never purchased
    slope   = _col(df, "spend_slope_30d")
    redeem  = _col(df, "redemption_rate")
    app     = _col(df, "app_open_30d")
    email   = _col(df, "email_open_30d")
    push    = _col(df, "push_open_30d")
    social  = _col(df, "social_share_30d")
    referral= _col(df, "referral_sent_30d")
    survey  = _col(df, "survey_completed_30d")
    tier    = _col(df, "tier_ordinal")
    div     = _col(df, "category_diversity_90d")

    return [
        # New & Uncertain — purely tenure-based
        (tenure < TENURE_NEW_MAX),
        # Win-Back Target — lapsed, but showing digital re-engagement
        (recency > RECENCY_WINBACK_MIN) & ((email > 0) | (app > 0) | (push > 0)),
        # Lapse Risk — gone quiet and spend is flat or falling
        (recency > RECENCY_LAPSE_MIN) & (p30 == 0) & (slope <= 0.0),
        # Momentum Builder — accelerating spend, active, diversifying
        (slope > MOMENTUM_SLOPE_MIN) & (p30 >= MOMENTUM_FREQ_MIN)
        & (recency < RECENCY_LAPSE_MIN) & (div > MOMENTUM_DIVERSITY_MIN),
        # Brand Advocate — engaged on any channel, high tier, broad basket
        ((email >= ADVOCATE_EMAIL_MIN) | (app >= ADVOCATE_APP_MIN)
         | (social >= 1) | (referral >= 1) | (survey >= 1))
        & (tier >= ADVOCATE_TIER_MIN) & (div > ADVOCATE_DIVERSITY_MIN),
        # Redemption Hunter — redeems heavily, buys rarely
        (redeem > REDEMPTION_RATE_MIN) & (p30 <= 1),
        # Value Maximizer — broad basket and actively redeeming
        (div > MAXIMIZER_DIVERSITY_MIN) & (redeem > MAXIMIZER_REDEEM_MIN),
        # Silent Accumulator — buying, but zero digital engagement
        (p30 >= 1) & (app == 0) & (email == 0) & (push == 0),
        # Plateau Cruiser — steady, neither growing nor declining
        (slope >= PLATEAU_SLOPE_MIN) & (slope <= PLATEAU_SLOPE_MAX) & (p30 >= 1),
    ]


def classify_states(df: pd.DataFrame) -> np.ndarray:
    """
    Assign exactly one lifecycle state per row. First matching rule wins.

    Every member receives a state; unmatched rows fall through to the
    Program Skeptic catch-all. NaNs are filled before comparison, so no member
    is dropped for missing features.
    """
    conditions = build_conditions(df)
    choices    = STATE_ORDER[:-1]          # all but the catch-all
    assert len(conditions) == len(choices), (
        f"cascade arity mismatch: {len(conditions)} conditions vs "
        f"{len(choices)} labels"
    )
    return np.select(conditions, choices, default=STATE_ORDER[-1])


def state_ids_from_names(state_names) -> np.ndarray:
    """Map state name labels to their canonical priority ids."""
    s = pd.Series(state_names)
    ids = s.map(STATE_IDS)
    if ids.isna().any():
        unknown = sorted(set(s[ids.isna()]))
        raise ValueError(f"Unknown state name(s): {unknown}")
    return ids.to_numpy(dtype=int)
