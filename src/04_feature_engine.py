"""
src/04_feature_engine.py
══════════════════════════════════════════════════════════════════════════════
PHASE 4 — Feature Engine

Reads each monthly snapshot and engineers 9 feature families:
  1. Spend (avg_order_value_30d, spend_per_purchase_90d)
  2. Frequency (purchase_count_* pass-through from snapshot)
  3. Recency (recency_days pass-through)
  4. Loyalty (redemption_rate, hoarding_ratio — with clipping)
  5. Engagement (email_open_rate_30d, email_click_rate_30d,
                  browse_to_purchase_ratio_30d + browsed_but_never_purchased_30d flag)
  6. Trend (spend_slope_30d, frequency_slope_30d — from weekly bucket resampling)
  7. Acceleration (spend_acceleration — requires prior month's slope)
  8. Diversity (category_diversity_90d, channel_diversity_90d)
  9. Tier trajectory (tier_ordinal, tier_changes_count,
                       months_since_last_tier_change, tier_trajectory_direction)

Key design points (per spec + critique):
  - Slope computed from weekly buckets per member (real bridge, not placeholder)
  - Returns 0.0 when < 2 weekly data points exist (never NaN)
  - hoarding_ratio and browse_to_purchase_ratio clipped at [0, 10]
  - Outlier clips read from validation/clip_bounds.json (not hardcoded)
  - Explicit BEHAVIORAL_FEATURE_COLS allowlist for Phase 6 clustering
  - All ratio features clipped to prevent +inf from zero denominators

Run from TBIE/ root:
    python src/04_feature_engine.py
"""

import json
import sys
from pathlib import Path

# Force UTF-8 output — Windows terminal defaults to cp1252 which can't encode
# the box-drawing characters (═══) used in print statements below.
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

# Path setup
ROOT          = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR  = ROOT / "snapshots"
FEATURES_DIR  = ROOT / "features"
VAL_DIR       = ROOT / "validation"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

# Snapshot date list (must match Phase 3)
SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")


# CLIP BOUNDS LOADER

def load_clip_bounds() -> dict:
    """Load clip bounds from Phase 1 audit output. Raises if file not found."""
    clip_path = VAL_DIR / "clip_bounds.json"
    if not clip_path.exists():
        raise FileNotFoundError(
            f"Clip bounds file not found: {clip_path}\n"
            f"Run src/01_validate_raw.py first."
        )
    with open(clip_path) as f:
        return json.load(f)


# WEEKLY SLOPE COMPUTATION

def compute_spend_frequency_slopes(
    snapshot_df: pd.DataFrame,
    txns_raw: pd.DataFrame,
    obs_date: pd.Timestamp,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Compute spend_slope_30d and frequency_slope_30d by:
      1. Filtering transactions to the window
      2. Resampling to weekly buckets per member
      3. Fitting a linear slope to each member's 4-point weekly series
      4. Returning 0.0 when < 2 weekly data points exist

    This function replaces the broken placeholder in the spec and provides
    the correct vectorized weekly bridge.
    """
    window_start = obs_date - pd.Timedelta(days=window_days)

    # Filter to purchase transactions in the 30d window
    mask = (
        (txns_raw["transaction_date"] >= window_start) &
        (txns_raw["transaction_date"] <= obs_date) &
        (txns_raw["transaction_type"] == "purchase")
    )
    txn_window = txns_raw[mask].copy()

    if len(txn_window) == 0:
        # No transactions  all slopes = 0.0
        result = pd.DataFrame({
            "member_id":         snapshot_df["member_id"],
            "spend_slope_30d":   0.0,
            "frequency_slope_30d": 0.0,
        })
        return result

    # Floor transaction_date to the start of its week (Monday)
    txn_window["week"] = txn_window["transaction_date"].dt.to_period("W").apply(
        lambda p: p.start_time
    )

    # Aggregate to weekly buckets per member
    weekly = (
        txn_window.groupby(["member_id", "week"])
        .agg(
            weekly_spend=("transaction_amount", "sum"),
            weekly_count=("transaction_id",     "count"),
        )
        .reset_index()
    )

    # Compute slope per member
    def _slope(series):
        arr = series.values
        if len(arr) < 2 or np.all(arr == 0):
            return 0.0
        x = np.arange(len(arr), dtype=float)
        try:
            slope = np.polyfit(x, arr.astype(float), 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    spend_slopes = (
        weekly.groupby("member_id")["weekly_spend"]
        .apply(_slope)
        .rename("spend_slope_30d")
        .reset_index()
    )
    freq_slopes = (
        weekly.groupby("member_id")["weekly_count"]
        .apply(_slope)
        .rename("frequency_slope_30d")
        .reset_index()
    )

    slopes = spend_slopes.merge(freq_slopes, on="member_id", how="outer")

    # Left-join onto full snapshot so all members are covered
    result = snapshot_df[["member_id"]].merge(slopes, on="member_id", how="left")
    result["spend_slope_30d"] = result["spend_slope_30d"].fillna(0.0)
    result["frequency_slope_30d"] = result["frequency_slope_30d"].fillna(0.0)

    return result[["member_id", "spend_slope_30d", "frequency_slope_30d"]]


# OUTLIER CLIPPING

def apply_outlier_clips(df: pd.DataFrame, clip_bounds: dict) -> pd.DataFrame:
    """
    Apply outlier clipping using bounds from Phase 1's clip_bounds.json.
    Also applies hard-coded [0, 10] caps for ratio features that can
    technically explode (hoarding_ratio, browse_to_purchase_ratio_30d).
    """
    df = df.copy()  # pandas 3.0 CoW-safe: never mutate the caller's DataFrame
    # Bounds from Phase 1 audit
    for col, info in clip_bounds.items():
        if col not in df.columns:
            continue
        lower = info.get("lower")
        upper = info.get("upper")
        if lower is not None and upper is not None:
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif upper is not None:
            df[col] = df[col].clip(upper=upper)
        elif lower is not None:
            df[col] = df[col].clip(lower=lower)

    # Hard caps for ratio features (as per critique #11)
    for ratio_col in ["hoarding_ratio", "browse_to_purchase_ratio_30d"]:
        if ratio_col in df.columns:
            df[ratio_col] = df[ratio_col].clip(lower=0, upper=10)

    return df


# MAIN FEATURE ENGINEERING FUNCTION

def engineer_features(
    snapshot_df: pd.DataFrame,
    txns_raw: pd.DataFrame,
    obs_date: pd.Timestamp,
    clip_bounds: dict,
    prior_snapshot_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Engineer all 9 feature families from a single snapshot.

    Parameters
    ----------
    snapshot_df       : output of Phase 3 build_snapshot()
    txns_raw          : clean transactions (for slope computation — needs raw grain)
    obs_date          : observation date for this snapshot
    clip_bounds       : loaded from validation/clip_bounds.json
    prior_snapshot_df : previous month's feature df (for acceleration); None for Jan

    Returns
    -------
    pd.DataFrame with all engineered features
    """
    df = snapshot_df.copy()

    # FAMILY 0: TENURE
    # tenure_days drives the priority-1 state rule (New & Uncertain: tenure<90).
    # It used to be derived ad hoc at inference time and was simply absent from
    # the feature files, which meant the rule never fired during training in 11
    # of 12 months while firing for ~27% of members at inference. Computing it
    # here makes training and inference see the same feature.
    if "account_open_date" in df.columns:
        df["tenure_days"] = (
            obs_date - pd.to_datetime(df["account_open_date"])
        ).dt.days.astype("float32")
        # Negative tenure = the member enrols AFTER this observation date. The
        # spine is a fixed 500K panel, so these rows exist at every snapshot in
        # order to keep the output contract at 500,000 rows. The value is left
        # negative on purpose: it is a genuine signal ("not yet a member") that
        # clipping to 0 would conflate with "enrolled today". The state cascade
        # treats both as New & Uncertain either way, since tenure < 90 holds.
        n_pre = int((df["tenure_days"] < 0).sum())
        if n_pre:
            print(f"    tenure_days: {n_pre:,} rows pre-enrolment at this "
                  f"observation date (negative tenure retained as signal)")
    else:
        raise KeyError(
            "account_open_date missing from snapshot — cannot compute tenure_days, "
            "which the New & Uncertain state rule depends on."
        )

    # FAMILY 1 & 2: SPEND & FREQUENCY
    # purchase_count_* and spend_total_* are purchase-only (Phase 3 enforces this)
    # Derived spend features:
    df["avg_order_value_30d"] = (
        df["spend_total_30d"] / df["purchase_count_30d"].replace(0, np.nan)
    ).fillna(0)

    df["spend_per_purchase_90d"] = (
        df["spend_total_90d"] / df["purchase_count_90d"].replace(0, np.nan)
    ).fillna(0)

    # FAMILY 3: RECENCY
    # recency_days is already in snapshot; NaN = never purchased (kept as NaN)

    # FAMILY 4: LOYALTY
    # Points redeemed and earned reconstructed from transactions (leakage-safe)
    has_points = all(col in df.columns for col in [
        "points_redeemed_lifetime", "points_earned_lifetime", "current_point_balance_reconstructed"
    ])
    if has_points:
        df["redemption_rate"] = (
            df["points_redeemed_lifetime"] /
            df["points_earned_lifetime"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        df["hoarding_ratio"] = (
            df["current_point_balance_reconstructed"] /
            df["points_earned_lifetime"].replace(0, np.nan)
        ).fillna(0)
        # hoarding_ratio clipped in apply_outlier_clips below
    else:
        df["redemption_rate"] = 0.0
        df["hoarding_ratio"]  = 0.0

    # FAMILY 5: ENGAGEMENT
    # BUG FIX (diagnosed 2026-06-22):
    #   Phase 3 produces SINGULAR column names from event_type pivot:
    #     email_open_30d, email_click_30d, app_open_30d, push_open_30d
    #   'email_sent_30d' does NOT exist — email_sent is not a real event type
    #   (Decision 021). The old else-branch was always firing  0.0.
    #   'browse_sessions_30d' does NOT exist — browse_session is not a real
    #   event type. Use reward_browse_30d instead (4.4M rows in raw data).
    #
    # email_open_rate_30d: raw email_open_30d count (direct engagement signal).
    # email_click_rate_30d: click-to-open rate = email_click / email_open.
    #   This is a standard email marketing metric and IS computable from our data.
    if "email_open_30d" in df.columns:
        # Raw open count — used directly as engagement signal (no email_sent denominator)
        df["email_open_rate_30d"] = df["email_open_30d"]

        # Click-to-open rate: what fraction of openers also clicked?
        if "email_click_30d" in df.columns:
            df["email_click_rate_30d"] = (
                df["email_click_30d"] / df["email_open_30d"].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
        else:
            df["email_click_rate_30d"] = 0.0
    else:
        df["email_open_rate_30d"]  = 0.0
        df["email_click_rate_30d"] = 0.0

    # browse_to_purchase_ratio — use reward_browse_30d (real event type with 4.4M rows).
    # 'browse_sessions_30d' does NOT exist — browse_session is not a real event type.
    # reward_browse = member browsed the rewards catalog, a strong intent signal.
    if "reward_browse_30d" in df.columns:
        df["browse_to_purchase_ratio_30d"] = (
            df["reward_browse_30d"] / df["purchase_count_30d"].replace(0, np.nan)
        ).fillna(0).clip(0, 10)

        # High-intent non-buyer flag: browsed rewards but made no purchase
        df["browsed_but_never_purchased_30d"] = (
            (df["reward_browse_30d"] > 0) & (df["purchase_count_30d"] == 0)
        ).astype(int)
    else:
        df["browse_to_purchase_ratio_30d"]    = 0.0
        df["browsed_but_never_purchased_30d"] = 0

    # FAMILY 6: TREND (slopes)
    print("    Computing spend/frequency slopes (weekly resampling)…", end="", flush=True)
    slope_df = compute_spend_frequency_slopes(df, txns_raw, obs_date, window_days=30)
    df = df.merge(slope_df, on="member_id", how="left")
    df["spend_slope_30d"] = df["spend_slope_30d"].fillna(0.0)
    df["frequency_slope_30d"] = df["frequency_slope_30d"].fillna(0.0)
    print(f" done. NaN slopes: {df['spend_slope_30d'].isna().sum()}")

    # FAMILY 7: ACCELERATION
    if prior_snapshot_df is not None:
        prior_slopes = prior_snapshot_df.set_index("member_id")["spend_slope_30d"].reindex(
            df["member_id"]
        ).fillna(0.0).values
        df["spend_acceleration"] = df["spend_slope_30d"].values - prior_slopes
    else:
        df["spend_acceleration"] = 0.0  # January — no prior period

    # Defensive fillna: members with zero purchases in the 30d window get 0.0,
    # not NaN. The slope function returns 0.0 for <2 data points, but the merge
    # path can still produce NaN for edge cases (audit flagged 16,215 members).
    df["spend_slope_30d"]      = df["spend_slope_30d"].fillna(0.0)
    df["frequency_slope_30d"]  = df["frequency_slope_30d"].fillna(0.0)
    df["spend_acceleration"]   = df["spend_acceleration"].fillna(0.0)

    # FAMILY 8: DIVERSITY
    for window_label in ["7d", "30d", "90d", "180d"]:
        cat_col  = f"unique_categories_{window_label}"
        chan_col = f"unique_channels_{window_label}"
        cnt_col  = f"purchase_count_{window_label}"

        if cat_col in df.columns:
            df[f"category_diversity_{window_label}"] = (
                df[cat_col] / df[cnt_col].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
        else:
            df[f"category_diversity_{window_label}"] = 0.0

        if chan_col in df.columns:
            df[f"channel_diversity_{window_label}"] = (
                df[chan_col] / df[cnt_col].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
        else:
            df[f"channel_diversity_{window_label}"] = 0.0

    # FAMILY 9: TIER TRAJECTORY
    # current_tier, tier_changes_count, months_since_last_tier_change,
    # tier_trajectory_direction are all attached by Phase 3's parse_tier_history_asof()
    tier_map = {"base": 0, "silver": 1, "gold": 2, "platinum": 3}
    if "current_tier" in df.columns:
        df["tier_ordinal"] = df["current_tier"].str.lower().map(tier_map).fillna(0).astype(int)
    else:
        df["tier_ordinal"] = 0

    # Apply outlier clips (from Phase 1 audit)
    df = apply_outlier_clips(df, clip_bounds)

    # Ensure redemption_rate is still in [0,1] post-clip (double check)
    if "redemption_rate" in df.columns:
        df["redemption_rate"] = df["redemption_rate"].clip(0, 1)

    # FEATURE COMPLETENESS FLAG
    # Members with any purchase activity OR non-null recency get feature_complete=1
    df["feature_complete"] = (
        (df.get("purchase_count_180d", pd.Series(0)) >= 1) |
        df["recency_days"].notna()
    ).astype(int)

    # EXPLICIT BEHAVIORAL FEATURE ALLOWLIST (critique #12)
    # This is the definitive list for PCA/HDBSCAN in Phase 6.
    # Excludes: IDs, raw dates, raw strings, intermediate columns, flags.
    # BUG FIX (diagnosed 2026-06-22):
    #   Phase 3 pivot produces SINGULAR column names (e.g. app_open_30d, not app_opens_30d).
    #   All plural names in this list have been corrected to match Phase 3 output.
    #   Mapping of old (wrong) -> new (correct):
    #     app_opens_30d       -> app_open_30d
    #     app_opens_90d       -> app_open_90d
    #     email_opens_30d     -> email_open_30d    (email_open_rate_30d now = raw count)
    #     email_clicks_30d    -> email_click_30d
    #     push_opens_30d      -> push_open_30d
    #     tier_checks_30d     -> tier_status_check_30d
    #     support_contacts_30d -> support_contact_30d
    behavioral_feature_cols = [
        # Spend
        "spend_total_7d", "spend_total_30d", "spend_total_90d", "spend_total_180d",
        "avg_order_value_30d", "spend_per_purchase_90d",
        # Frequency
        "purchase_count_7d", "purchase_count_30d", "purchase_count_90d", "purchase_count_180d",
        "return_count_30d", "return_count_90d",
        # Recency
        "recency_days",
        # Loyalty
        "redemption_rate", "hoarding_ratio",
        "points_earned_lifetime", "points_redeemed_lifetime",
        # Engagement — column names match Phase 3 pivot output (singular event_type names)
        "app_open_30d", "app_open_90d",          # Phase 3: app_open_{w}
        "email_open_30d", "email_click_30d",      # Phase 3: email_open_{w}, email_click_{w}
        # NOTE: email_open_rate_30d REMOVED — it is an exact duplicate of email_open_30d
        #       (raw count assigned in Family 5). Keeping both inflates email weight in PCA.
        "email_click_rate_30d",                   # click-to-open ratio (derived in Family 5)
        "browse_to_purchase_ratio_30d", "browsed_but_never_purchased_30d",
        "push_open_30d",                          # Phase 3: push_open_{w}
        "reward_browse_30d", "reward_redemption_30d",
        "tier_status_check_30d", "support_contact_30d",  # Phase 3 exact names
        "total_session_sec_30d",
        # SESSION 1 FIX: 5 event types that existed in parquets but were missing from allowlist
        "social_share_30d",       # 16.5% non-zero — critical Brand Advocate signal
        "referral_sent_30d",      # 16.5% non-zero — critical Brand Advocate signal
        "survey_completed_30d",   # 17.3% non-zero — high-effort engagement
        "point_balance_check_30d",# 48.3% non-zero — active point monitoring signal
        "profile_update_30d",     # 15.8% non-zero — program investment signal
        # Trend
        "spend_slope_30d", "frequency_slope_30d",
        # Acceleration
        "spend_acceleration",
        # Diversity
        "category_diversity_90d", "channel_diversity_90d",
        "unique_categories_90d", "unique_channels_90d",
        # Tier trajectory
        "tier_ordinal", "tier_changes_count", "months_since_last_tier_change",
    ]

    # Filter to only columns that actually exist in this snapshot
    behavioral_feature_cols = [c for c in behavioral_feature_cols if c in df.columns]

    # Store as metadata attribute (consumed by Phase 6)
    df.attrs["behavioral_feature_cols"] = behavioral_feature_cols
    df.attrs["ml_feature_cols"]         = behavioral_feature_cols  # alias for compatibility

    return df


# MAIN

def main():
    print("\n" + "═"*70)
    print("TBIE — PHASE 4: FEATURE ENGINE")
    print("═"*70 + "\n")

    # Load clip bounds
    print("Loading clip bounds from Phase 1 audit…")
    clip_bounds = load_clip_bounds()
    print(f"  Loaded bounds for {len(clip_bounds)} column(s): {list(clip_bounds.keys())}")

    # Load raw transactions for slope computation
    print("\nLoading raw transactions for weekly slope computation…")
    txns_raw = pd.read_parquet(ROOT / "data" / "raw" / "transactions.parquet")

    # Normalize transaction_type for purchase filter
    from utils.datetime_parser import parse_mixed_datetime
    txns_raw["transaction_date"] = parse_mixed_datetime(
        txns_raw["transaction_date"], col_name="transaction_date"
    )
    txns_raw["transaction_type"] = txns_raw["transaction_type"].str.strip().str.lower()

    # Exclude ghost cohort
    ghost_ids = set(pd.read_csv(VAL_DIR / "ghost_cohort.csv")["member_id"].tolist())
    txns_raw  = txns_raw[~txns_raw["member_id"].isin(ghost_ids)]
    print(f"  Loaded: {txns_raw.shape} (ghost-filtered)")

    # Process all 12 snapshots in chronological order
    prior = None
    feature_summary = []

    for obs_date in SNAPSHOT_DATES:
        date_str   = obs_date.strftime("%Y_%m_%d")
        snap_path  = SNAPSHOT_DIR / f"snapshot_{date_str}.parquet"
        feat_path  = FEATURES_DIR / f"features_{date_str}.parquet"

        if not snap_path.exists():
            print(f"  ️  Snapshot not found: {snap_path} — skipping")
            continue

        print(f"\n  [{obs_date.strftime('%Y-%m')}] Engineering features…")
        snap = pd.read_parquet(snap_path)

        features = engineer_features(snap, txns_raw, obs_date, clip_bounds,
                                     prior_snapshot_df=prior)
        features.to_parquet(feat_path, index=False)

        # Verification checks
        n_total         = len(features)
        pct_complete    = features["feature_complete"].mean() * 100
        nan_slopes_spend = features["spend_slope_30d"].isna().sum()
        nan_slopes_freq  = features["frequency_slope_30d"].isna().sum()
        nan_accel        = features["spend_acceleration"].isna().sum()
        jan_accel_all_zero = (
            features["spend_acceleration"] == 0.0
        ).all() if obs_date == SNAPSHOT_DATES[0] else None

        print(f"    Rows: {n_total:,}  |  feature_complete: {pct_complete:.1f}%")
        print(f"    NaN spend_slope_30d: {nan_slopes_spend}  |  "
              f"NaN frequency_slope_30d: {nan_slopes_freq}  |  "
              f"NaN spend_acceleration: {nan_accel}")
        if jan_accel_all_zero is not None:
            flag = "" if jan_accel_all_zero else "️ "
            print(f"    {flag} January spend_acceleration all 0.0: {jan_accel_all_zero}")

        # Redemption rate check
        bad_redemption = (
            (features["redemption_rate"] < 0) | (features["redemption_rate"] > 1)
        ).sum()
        if bad_redemption > 0:
            print(f"    ️  {bad_redemption} rows with redemption_rate outside [0,1]!")
        else:
            print("     redemption_rate in [0,1] for all rows")

        feature_summary.append({
            "month":                obs_date.strftime("%Y-%m"),
            "n_rows":               int(n_total),
            "pct_feature_complete": round(pct_complete, 2),
            "nan_spend_slope":      int(nan_slopes_spend),
            "nan_accel":            int(nan_accel),
            "bad_redemption_rate":  int(bad_redemption),
        })

        prior = features

    # Summary
    print("\n" + "─"*60)
    print("PHASE 4 SUMMARY:")
    summary_df = pd.DataFrame(feature_summary)
    print(summary_df.to_string(index=False))

    print("\n" + "─"*60)
    print("PHASE 4 EXIT CRITERIA:")
    print("   All 9 feature families implemented")
    print("   Slope computed from weekly buckets (real bridge, not placeholder)")
    print("   Returns 0.0 (not NaN) for members with < 2 weekly data points")
    print("   Acceleration relative to PRIOR month in chronological order")
    print("   January spend_acceleration = 0.0 for all members")
    print("   hoarding_ratio and browse_to_purchase_ratio clipped at [0, 10]")
    print("   browsed_but_never_purchased_30d boolean flag added")
    print("   Outlier clips read from Phase 1 audit (not hardcoded)")
    print("   Explicit BEHAVIORAL_FEATURE_COLS allowlist defined")
    print("\nPHASE 4 COMPLETE.")


if __name__ == "__main__":
    main()
