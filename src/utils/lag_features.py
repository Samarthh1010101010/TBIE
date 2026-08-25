"""
src/utils/lag_features.py
─────────────────────────
Month-over-month change features, shared by training and inference.

Why these exist
───────────────
The prediction target is a TRANSITION, but nearly every base feature describes
a level: how much a member spent, how often they visited, how recently. The six
"velocity" features are ratios computed inside a single snapshot (30d vs 90d/3),
so they never compare this month's snapshot against last month's. The model was
therefore being asked to predict change from an almost change-free view of the
member.

These features difference consecutive monthly snapshots, and add two
segment-history features. `seg_prev` in particular lifts the model from a
first-order to a second-order Markov view: a member who has just arrived in a
segment behaves differently from one who has sat in it for months.

Why it lives here
─────────────────
Both src/08_transition_prediction.py (training) and pipeline.py (inference)
need to build these identically. The state cascade was previously duplicated
across those two files and silently drifted apart; this module exists so the
same mistake cannot be repeated for the lag features.
"""

from __future__ import annotations

import pandas as pd

# Base columns differenced month over month. Chosen for behavioural meaning:
# spend and frequency at two horizons, recency, the spend trend, each digital
# channel, redemption, basket breadth, order value, and tier.
LAG_BASE_COLS_ALL = [
    "spend_total_30d",
    "spend_total_90d",
    "purchase_count_30d",
    "purchase_count_90d",
    "recency_days",
    "spend_slope_30d",
    "app_open_30d",
    "email_open_30d",
    "push_open_30d",
    "redemption_rate",
    "category_diversity_90d",
    "avg_order_value_30d",
    "tier_ordinal",
]

SEG_HIST_COLS = ["seg_prev", "seg_changed"]


def resolve_lag_base_cols(available_cols) -> list[str]:
    """Restrict the delta set to columns actually present in the feature files."""
    available = set(available_cols)
    return [c for c in LAG_BASE_COLS_ALL if c in available]


def delta_col_names(lag_base_cols) -> list[str]:
    return [f"d_{c}" for c in lag_base_cols]


def lag_col_names(lag_base_cols) -> list[str]:
    return delta_col_names(lag_base_cols) + SEG_HIST_COLS


def add_delta_features(
    feat_t: pd.DataFrame,
    feat_prev: pd.DataFrame,
    lag_base_cols: list[str],
) -> pd.DataFrame:
    """
    Month-over-month deltas, aligned on member_id.

    Members absent from the prior snapshot get a delta of 0 — "no observed
    change" — rather than being dropped, which would break the fixed 500K
    member panel and the 500,000-row output contract.
    """
    missing = [c for c in lag_base_cols if c not in feat_prev.columns]
    if missing:
        raise KeyError(f"prior-month frame is missing lag base column(s): {missing}")

    prev = feat_prev[["member_id"] + lag_base_cols].rename(
        columns={c: f"prev_{c}" for c in lag_base_cols}
    )
    out = feat_t.merge(prev, on="member_id", how="left")
    for c in lag_base_cols:
        out[f"d_{c}"] = (
            out[c].fillna(0) - out[f"prev_{c}"].fillna(0)
        ).astype("float32")
    return out.drop(columns=[f"prev_{c}" for c in lag_base_cols])


def add_segment_history(
    df: pd.DataFrame,
    seg_prev_values,
    seg_curr_col: str = "seg_curr",
) -> pd.DataFrame:
    """
    Attach seg_prev and seg_changed.

    A member unseen in the prior month is treated as having already been in
    their current segment, so seg_changed reads 0 rather than inventing a move
    that was never observed.
    """
    out = df.copy()
    out["seg_prev"] = pd.Series(seg_prev_values, index=out.index).astype("float32")
    out["seg_prev"] = out["seg_prev"].fillna(out[seg_curr_col]).astype("float32")
    out["seg_changed"] = (out["seg_prev"] != out[seg_curr_col]).astype("float32")
    return out
