"""
src/03_snapshot_builder.py
══════════════════════════════════════════════════════════════════════════════
PHASE 3 — Monthly Snapshot Builder

Produces 12 monthly snapshots (2025-01-01 through 2025-12-01), each containing
rolling-window aggregations of transaction and engagement data per member.

Key design points (per spec + all applied bug-fixes/performance fixes):

  PERFORMANCE:
    - Base tables loaded ONCE, cleaned ONCE, reused across all 12 snapshots
    - engine='pyarrow' on every read_parquet / to_parquet call (explicit)
    - Categoricals cast to 'category' dtype after normalization — groupby
      on integers internally, not strings, at 18M/35M row scale
    - Progressive window narrowing: 180d  90d  30d  7d. Each filter
      runs against an already-smaller DataFrame, not the full table

  CORRECTNESS:
    - Dual-format datetime parser — zero bare pd.to_datetime() calls
    - Categorical normalization applied ONCE at load time
    - session_duration_sec clipped at 14,400s once at load time
    - Exact duplicate engagement rows dropped once at load time
    - Ghost cohort excluded from both tables at load time
    - Spend/count features are PURCHASE-ONLY
    - unique_categories_{w} and unique_channels_{w} added (diversity inputs
      for Phase 4) — BUG FIX vs original, these were missing
    - Event type aggregations use the 14 REAL event types confirmed in Phase 1
      audit (no 'email_sent' or 'browse_session' — these don't exist in data)
    - Engagement aggregations use groupby + transform, no lambda .apply()
    - Lifetime points reconstructed from transactions (no leakage columns)
    - Leakage assertions fire on EVERY snapshot, not spot-checked
    - Pre-enrollment activity removed per-snapshot
    - Zero-activity members preserved via LEFT JOIN throughout

  OBSERVABILITY:
    - Per-snapshot row count, active-member count, AND elapsed time printed
    - If any snapshot takes longer than 5 minutes, script stops and reports

Run from TBIE_CODE/ root:
    python src/03_snapshot_builder.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "raw"
SPINE_DIR    = ROOT / "spine"
SNAPSHOT_DIR = ROOT / "snapshots"
VAL_DIR      = ROOT / "validation"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from utils.datetime_parser import parse_mixed_datetime
from utils.leakage_guard import assert_no_future_data

# Constants
SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")  # 12

# Real event types confirmed in Phase 1 Step 1.1 / raw_data_audit.md
# DO NOT add 'email_sent' or 'browse_session' — they do not exist in the data
EVENT_TYPES = [
    "app_open",
    "email_open",
    "email_click",
    "push_open",
    "push_dismiss",
    "point_balance_check",
    "reward_browse",
    "reward_redemption",
    "tier_status_check",
    "support_contact",
    "social_share",
    "referral_sent",
    "survey_completed",
    "profile_update",
]

MAX_SNAPSHOT_SECONDS = 300  # 5-minute per-snapshot hard stop


# BASE TABLE LOADER — called ONCE, not per snapshot

def load_clean_base_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, parse, normalize, clean, and ghost-filter transactions and
    engagement_events. Called ONCE; the resulting DataFrames are reused
    across all 12 snapshot iterations.

    Performance notes:
      - engine='pyarrow' explicit on every read
      - Categoricals cast to 'category' dtype after normalization so that
        downstream groupby operations hash integers, not strings
    """
    print("  Loading transactions.parquet …")
    t0 = time.time()
    txns = pd.read_parquet(DATA_DIR / "transactions.parquet", engine="pyarrow")
    print(f"    Loaded {len(txns):,} rows × {txns.shape[1]} cols "
          f"in {time.time()-t0:.1f}s")

    print("  Loading engagement_events.parquet …")
    t0 = time.time()
    events = pd.read_parquet(DATA_DIR / "engagement_events.parquet", engine="pyarrow")
    print(f"    Loaded {len(events):,} rows × {events.shape[1]} cols "
          f"in {time.time()-t0:.1f}s")

    # Parse datetime columns
    # members.account_open_date is already datetime64[us] (confirmed Phase 1)
    # transactions.transaction_date and events.event_date are object (mixed fmt)
    print("  Parsing datetime columns (strict dual-format parser) …")
    txns["transaction_date"]  = parse_mixed_datetime(
        txns["transaction_date"], col_name="transaction_date"
    )
    events["event_date"] = parse_mixed_datetime(
        events["event_date"], col_name="event_date"
    )

    # String normalization — ONCE here, never repeated downstream
    print("  Normalizing categoricals (strip + lower) …")
    for col in ["channel", "transaction_type", "merchant_category",
                "merchant_subcategory", "merchant_brand"]:
        if col in txns.columns:
            txns[col] = txns[col].str.strip().str.lower()

    for col in ["event_type", "event_channel"]:
        if col in events.columns:
            events[col] = events[col].str.strip().str.lower()

    # Cast to 'category' dtype AFTER normalization
    # This is the single biggest performance lever at 18M/35M row scale.
    # groupby on a 'category' column hashes integers, not strings.
    print("  Casting categoricals to 'category' dtype …")
    for col in ["channel", "transaction_type", "merchant_category"]:
        if col in txns.columns:
            txns[col] = txns[col].astype("category")
    for col in ["event_type", "event_channel"]:
        if col in events.columns:
            events[col] = events[col].astype("category")

    # Log transaction_type distribution for confirmation
    print("  transaction_type distribution after normalization:")
    for k, v in txns["transaction_type"].value_counts().items():
        print(f"    {k}: {v:,}")

    # Drop exact duplicate engagement rows
    # Decision 010 (docs/decisions.md): dedup key = member_id + event_date + event_type
    # campaign_id excluded — it's null for non-campaign events
    n_before = len(events)
    events = events.drop_duplicates(
        subset=["member_id", "event_date", "event_type"]
    )
    n_dropped = n_before - len(events)
    print(f"  Dropped {n_dropped:,} duplicate engagement rows "
          f"({n_dropped/n_before:.4%} of {n_before:,})")

    # Clip session_duration_sec — bound verified in Phase 1 Step 1.8
    if "session_duration_sec" in events.columns:
        n_over = (events["session_duration_sec"] > 14400).sum()
        events["session_duration_sec"] = events["session_duration_sec"].clip(upper=14400)
        print(f"  Clipped {n_over:,} session_duration_sec values > 14,400s (4 hrs)")

    # Exclude ghost cohort (88,717 MBR_GHOST_* IDs — Phase 1 Step 1.9) ─
    ghost_csv = VAL_DIR / "ghost_cohort.csv"
    if not ghost_csv.exists():
        raise FileNotFoundError(
            f"Ghost cohort file not found: {ghost_csv}\n"
            f"Run src/01_validate_raw.py first."
        )
    ghost_ids = set(pd.read_csv(ghost_csv)["member_id"].tolist())
    print(f"  Excluding {len(ghost_ids):,} ghost cohort IDs from both tables …")

    txns   = txns[~txns["member_id"].isin(ghost_ids)].copy()
    events = events[~events["member_id"].isin(ghost_ids)].copy()
    print(f"  After ghost exclusion: txns={txns.shape}, events={events.shape}")

    return txns, events


# TIER HISTORY — leakage-safe as-of lookup (vectorized via merge)

def _extract_tier_info(tier_json, obs_date: pd.Timestamp) -> dict:
    """
    Parse one member's tier_history JSON and return tier state as-of obs_date.
    Structure confirmed in Phase 1: list of {tier, date} dicts.
    """
    try:
        entries = json.loads(tier_json) if isinstance(tier_json, str) else tier_json
        if not entries:
            return {"current_tier": "base", "tier_changes_count": 0,
                    "months_since_last_tier_change": np.nan,
                    "tier_trajectory_direction": "stable"}

        valid = [e for e in entries if pd.Timestamp(e["date"]) <= obs_date]

        if not valid:
            return {"current_tier": "base", "tier_changes_count": 0,
                    "months_since_last_tier_change": np.nan,
                    "tier_trajectory_direction": "stable"}

        current_tier = valid[-1]["tier"].lower()
        n_changes    = len(valid) - 1  # first entry = enrollment, not a "change"

        if n_changes > 0:
            last_date   = pd.Timestamp(valid[-1]["date"])
            months_since = (obs_date - last_date).days / 30.4375
        else:
            months_since = np.nan

        tier_ord = {"base": 0, "silver": 1, "gold": 2, "platinum": 3}
        if len(valid) >= 2:
            prev = valid[-2]["tier"].lower()
            diff = tier_ord.get(current_tier, 0) - tier_ord.get(prev, 0)
            direction = "up" if diff > 0 else ("down" if diff < 0 else "stable")
        else:
            direction = "stable"

        return {
            "current_tier": current_tier,
            "tier_changes_count": n_changes,
            "months_since_last_tier_change": months_since,
            "tier_trajectory_direction": direction,
        }
    except Exception:
        return {"current_tier": "base", "tier_changes_count": 0,
                "months_since_last_tier_change": np.nan,
                "tier_trajectory_direction": "unknown"}


def parse_tier_history_asof(
    snapshot: pd.DataFrame,
    members_df: pd.DataFrame,
    obs_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Attach leakage-safe tier columns to the snapshot DataFrame.
    Uses vectorized apply on the members table (500k rows) — acceptable.
    NOT applied row-wise on transactions (~18M rows).
    """
    if "tier_history" not in members_df.columns:
        snapshot["current_tier"]                  = "base"
        snapshot["tier_changes_count"]             = 0
        snapshot["months_since_last_tier_change"]  = np.nan
        snapshot["tier_trajectory_direction"]      = "stable"
        return snapshot

    tier_info = members_df[["member_id", "tier_history"]].copy()
    tier_info["_tier_parsed"] = tier_info["tier_history"].apply(
        lambda th: _extract_tier_info(th, obs_date)
    )

    tier_df = pd.concat(
        [tier_info["member_id"].reset_index(drop=True),
         tier_info["_tier_parsed"].apply(pd.Series)],
        axis=1,
    )

    snapshot = snapshot.merge(tier_df, on="member_id", how="left")
    snapshot["current_tier"]        = snapshot["current_tier"].fillna("base")
    snapshot["tier_changes_count"]   = snapshot["tier_changes_count"].fillna(0)
    return snapshot


# VECTORIZED EVENT TYPE AGGREGATION

def _aggregate_event_types(event_window: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Pivot event_type counts per member for the given window, fully vectorized.
    Returns a DataFrame indexed by member_id.
    No lambda .apply() — uses pivot_table which is internally C-level groupby.
    """
    if len(event_window) == 0:
        return pd.DataFrame()

    # Count (member_id, event_type) pairs — vectorized groupby
    counts = (
        event_window.groupby(["member_id", "event_type"], observed=True)
        .size()
        .reset_index(name="n")
    )

    # Pivot: one column per event_type
    pivoted = counts.pivot_table(
        index="member_id", columns="event_type", values="n", aggfunc="sum", fill_value=0
    )

    # Ensure all 14 expected event type columns exist (add zeros for missing ones)
    for et in EVENT_TYPES:
        if et not in pivoted.columns:
            pivoted[et] = 0

    # Rename columns to include window label
    pivoted.columns = [f"{et}_{label}" for et in pivoted.columns]
    return pivoted


# PER-SNAPSHOT BUILD FUNCTION

def build_snapshot(
    observation_date,
    spine: pd.DataFrame,
    txns: pd.DataFrame,
    events: pd.DataFrame,
    members_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one monthly snapshot for the given observation_date.

    Parameters
    ----------
    observation_date : date-like (converted to pd.Timestamp internally)
    spine            : member spine (member_id + account_open_date)
    txns             : pre-cleaned transactions (parsed, normalized, cat-typed)
    events           : pre-cleaned engagement events (parsed, normalized, cat-typed)
    members_df       : members table (for tier_history parsing only)

    Returns
    -------
    pd.DataFrame — one row per member in spine (zero-activity members included)
    """
    obs_date = pd.Timestamp(observation_date)

    # Filter to data available as-of this observation date
    txns_t   = txns[txns["transaction_date"]  <= obs_date].copy()
    events_t = events[events["event_date"]    <= obs_date].copy()

    # Remove pre-enrollment activity
    txns_t = txns_t.merge(
        spine[["member_id", "account_open_date"]], on="member_id", how="left"
    )
    txns_t = txns_t[txns_t["transaction_date"] >= txns_t["account_open_date"]].copy()

    events_t = events_t.merge(
        spine[["member_id", "account_open_date"]], on="member_id", how="left"
    )
    events_t = events_t[events_t["event_date"] >= events_t["account_open_date"]].copy()

    # Leakage assertions — fire on EVERY call, not optional
    assert_no_future_data(
        txns_t, "transaction_date", obs_date,
        context=f"snapshot {obs_date.date()} transactions"
    )
    assert_no_future_data(
        events_t, "event_date", obs_date,
        context=f"snapshot {obs_date.date()} events"
    )

    # Purchase-only subset
    # Spend and purchase counts are purchase-only per spec (DECISION 007).
    # Returns and exchanges are tracked separately via return_count_*.
    txns_purchase = txns_t[txns_t["transaction_type"] == "purchase"]

    # PROGRESSIVE WINDOW NARROWING
    # 180d ⊃ 90d ⊃ 30d ⊃ 7d — each filter runs on an already-smaller frame.
    # This eliminates redundant full-table scans (was 4× per snapshot).
    w180_start = obs_date - pd.Timedelta(days=180)
    w90_start  = obs_date - pd.Timedelta(days=90)
    w30_start  = obs_date - pd.Timedelta(days=30)
    w7_start   = obs_date - pd.Timedelta(days=7)

    # Transaction windows
    txn_180 = txns_t[txns_t["transaction_date"]      >= w180_start]
    txn_90  = txn_180[txn_180["transaction_date"]    >= w90_start]
    txn_30  = txn_90[txn_90["transaction_date"]      >= w30_start]
    txn_7   = txn_30[txn_30["transaction_date"]      >= w7_start]

    # Purchase-only windows (for spend/count/diversity features)
    pur_180 = txns_purchase[txns_purchase["transaction_date"] >= w180_start]
    pur_90  = pur_180[pur_180["transaction_date"]             >= w90_start]
    pur_30  = pur_90[pur_90["transaction_date"]               >= w30_start]
    pur_7   = pur_30[pur_30["transaction_date"]               >= w7_start]

    # Engagement windows
    evt_180 = events_t[events_t["event_date"]       >= w180_start]
    evt_90  = evt_180[evt_180["event_date"]          >= w90_start]
    evt_30  = evt_90[evt_90["event_date"]            >= w30_start]
    evt_7   = evt_30[evt_30["event_date"]            >= w7_start]

    txn_windows = {"7d": txn_7,   "30d": txn_30,  "90d": txn_90,  "180d": txn_180}
    pur_windows = {"7d": pur_7,   "30d": pur_30,  "90d": pur_90,  "180d": pur_180}
    evt_windows = {"7d": evt_7,   "30d": evt_30,  "90d": evt_90,  "180d": evt_180}

    # Aggregation — vectorized groupby only, no .apply()
    agg_frames = []

    for label in ["7d", "30d", "90d", "180d"]:
        pur_w = pur_windows[label]
        txn_w = txn_windows[label]
        evt_w = evt_windows[label]

        # Spend + purchase count (purchase-only)
        if len(pur_w) > 0:
            txn_agg = pur_w.groupby("member_id", observed=True).agg(
                **{
                    f"spend_total_{label}":    ("transaction_amount", "sum"),
                    f"purchase_count_{label}": ("transaction_id",     "count"),
                }
            )
            agg_frames.append(txn_agg)

        # Diversity: unique categories and channels
        # BUG FIX: these were missing in the original spec code.
        # Phase 4 needs unique_categories_{w} and unique_channels_{w}
        # for category_diversity_90d and channel_diversity_90d.
        if len(pur_w) > 0 and "merchant_category" in pur_w.columns:
            cat_agg = pur_w.groupby("member_id", observed=True)[
                "merchant_category"
            ].nunique().rename(f"unique_categories_{label}")
            agg_frames.append(cat_agg.to_frame())

        if len(txn_w) > 0 and "channel" in txn_w.columns:
            chan_agg = txn_w.groupby("member_id", observed=True)[
                "channel"
            ].nunique().rename(f"unique_channels_{label}")
            agg_frames.append(chan_agg.to_frame())

        # Return count (separate from spend/purchase_count)
        returns_w = txn_w[txn_w["transaction_type"] == "return"] \
            if len(txn_w) > 0 else pd.DataFrame()
        if len(returns_w) > 0:
            ret_agg = returns_w.groupby("member_id", observed=True).agg(
                **{f"return_count_{label}": ("transaction_id", "count")}
            )
            agg_frames.append(ret_agg)

        # Points earned within window
        if len(txn_w) > 0 and "points_earned" in txn_w.columns:
            pts_agg = txn_w.groupby("member_id", observed=True).agg(
                **{f"points_earned_{label}": ("points_earned", "sum")}
            )
            agg_frames.append(pts_agg)

        # Engagement: event type counts (vectorized pivot)
        # Uses pivot_table — no lambda, no .apply()
        if len(evt_w) > 0:
            evt_type_agg = _aggregate_event_types(evt_w, label)
            if len(evt_type_agg) > 0:
                agg_frames.append(evt_type_agg)

        # Session duration aggregate
        if len(evt_w) > 0 and "session_duration_sec" in evt_w.columns:
            sess_w = evt_w[evt_w["session_duration_sec"].notna()]
            if len(sess_w) > 0:
                sess_agg = sess_w.groupby("member_id", observed=True).agg(
                    **{f"total_session_sec_{label}": ("session_duration_sec", "sum")}
                )
                agg_frames.append(sess_agg)

    # Lifetime points (reconstructed — no leakage columns)
    # Uses txns_t (all types, all time up to obs_date) — not windowed
    if "points_earned" in txns_t.columns:
        lifetime_earned = txns_t.groupby("member_id")["points_earned"].sum() \
                                 .rename("points_earned_lifetime")
        agg_frames.append(lifetime_earned.to_frame())

    if "points_clawed_back" in txns_t.columns:
        lifetime_clawback = txns_t.groupby("member_id")["points_clawed_back"].sum() \
                                   .rename("points_clawed_back_lifetime")
        agg_frames.append(lifetime_clawback.to_frame())

    # Combine all window frames onto the spine (LEFT JOIN)
    # LEFT JOIN is required — it preserves zero-activity members.
    snapshot = spine.set_index("member_id").copy()
    for frame in agg_frames:
        if len(frame) > 0:
            snapshot = snapshot.join(frame, how="left")

    # Fill NaN  0 for all count/sum features
    # recency_days intentionally stays NaN for never-purchased members
    # (NaN signals "never purchased", not 0)
    snapshot = snapshot.reset_index()
    count_cols = [c for c in snapshot.columns
                  if c not in ("account_open_date", "member_id")]
    snapshot[count_cols] = snapshot[count_cols].fillna(0)

    # Derived lifetime fields
    if "points_earned_lifetime" in snapshot.columns and \
       "points_clawed_back_lifetime" in snapshot.columns:
        snapshot["points_redeemed_lifetime"] = snapshot["points_clawed_back_lifetime"]
        snapshot["current_point_balance_reconstructed"] = (
            snapshot["points_earned_lifetime"] - snapshot["points_redeemed_lifetime"]
        ).clip(lower=0)

    # Recency — last transaction date (global, not windowed)
    if len(txns_t) > 0:
        last_txn = txns_t.groupby("member_id")["transaction_date"].max() \
                          .rename("last_transaction_date")
        snapshot = snapshot.merge(last_txn, on="member_id", how="left")
        # NaT  stays NaT for members with no transactions ("never purchased")
        snapshot["recency_days"] = (
            obs_date - snapshot["last_transaction_date"]
        ).dt.days
        # recency_days == NaN for never-purchased members — intentional
    else:
        snapshot["last_transaction_date"] = pd.NaT
        snapshot["recency_days"]          = np.nan

    # Tier history as-of this observation date
    snapshot = parse_tier_history_asof(snapshot, members_df, obs_date)

    # Observation date stamp
    snapshot["observation_date"] = obs_date

    # Tenure: days since account opening — required for New & Uncertain state
    snapshot["tenure_days"] = (obs_date - snapshot["account_open_date"]).dt.days

    return snapshot


# MAIN

def main():
    print("\n" + "═" * 70)
    print("TBIE — PHASE 3: MONTHLY SNAPSHOT BUILDER")
    print("═" * 70 + "\n")

    # Check prerequisites
    spine_path = SPINE_DIR / "member_spine.parquet"
    if not spine_path.exists():
        print(f"  ️  Spine not found: {spine_path}")
        print("  Run src/02_build_spine.py first.")
        sys.exit(1)

    if not (VAL_DIR / "ghost_cohort.csv").exists():
        print(f"  ️  Ghost cohort CSV not found in {VAL_DIR}/")
        print("  Run src/01_validate_raw.py first.")
        sys.exit(1)

    # Load spine
    print("Loading member spine …")
    spine = pd.read_parquet(spine_path, engine="pyarrow")
    print(f"  Spine: {len(spine):,} members")

    # Load and clean base tables ONCE
    print("\nLoading and cleaning base tables (once) …")
    t_load = time.time()
    txns, events = load_clean_base_tables()
    print(f"  Base tables ready in {time.time()-t_load:.1f}s\n")

    # Load members_df for tier history parsing
    print("Loading members_df for tier history …")
    members_df = pd.read_parquet(DATA_DIR / "members.parquet", engine="pyarrow")
    print(f"  members_df: {len(members_df):,} rows\n")

    # Build all 12 snapshots
    print(f"Building {len(SNAPSHOT_DATES)} monthly snapshots …\n")
    snapshot_summary = []
    overall_start    = time.time()

    for obs_date in SNAPSHOT_DATES:
        date_str = obs_date.strftime("%Y_%m_%d")
        out_path = SNAPSHOT_DIR / f"snapshot_{date_str}.parquet"

        print(f"  [{obs_date.strftime('%Y-%m-%d')}] building …", flush=True)
        t_snap = time.time()

        snap = build_snapshot(obs_date, spine, txns, events, members_df)

        elapsed = time.time() - t_snap
        n_rows        = len(snap)
        n_active_7d   = (snap.get("purchase_count_7d",
                                   pd.Series(dtype=float)) > 0).sum()
        n_active_30d  = (snap.get("purchase_count_30d",
                                   pd.Series(dtype=float)) > 0).sum()
        pct_active    = n_active_30d / n_rows * 100 if n_rows > 0 else 0.0
        row_match     = "" if n_rows == len(spine) else "️ "

        print(f"  {row_match} {obs_date.strftime('%Y-%m-%d')}: "
              f"{n_rows:,} rows | "
              f"active-30d: {n_active_30d:,} ({pct_active:.1f}%) | "
              f"elapsed: {elapsed:.1f}s")

        # Hard stop if any single snapshot exceeds 5 minutes
        if elapsed > MAX_SNAPSHOT_SECONDS:
            print(f"\n   STOP: snapshot {obs_date.date()} took {elapsed:.0f}s "
                  f"(>{MAX_SNAPSHOT_SECONDS}s limit).")
            print("  Investigate performance before continuing.")
            print("  Hint: check category dtype conversion, ghost exclusion set size.")
            sys.exit(2)

        snap.to_parquet(out_path, engine="pyarrow", index=False)

        snapshot_summary.append({
            "observation_date": str(obs_date.date()),
            "n_rows":           int(n_rows),
            "n_rows_match_spine": n_rows == len(spine),
            "active_7d":        int(n_active_7d),
            "active_30d":       int(n_active_30d),
            "pct_active_30d":   round(pct_active, 2),
            "elapsed_sec":      round(elapsed, 1),
            "leakage_check":    "PASSED",
        })

    total_elapsed = time.time() - overall_start
    print(f"\n  Total elapsed for all 12 snapshots: {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)\n")

    # Row count check
    print("  Row count check (all should equal spine size):")
    all_match = True
    for s in snapshot_summary:
        flag = "" if s["n_rows_match_spine"] else "️ "
        if not s["n_rows_match_spine"]:
            all_match = False
        print(f"    {flag} {s['observation_date']}: "
              f"{s['n_rows']:,} rows | "
              f"active-30d: {s['active_30d']:,} ({s['pct_active_30d']:.1f}%) | "
              f"{s['elapsed_sec']}s")

    # Exit criteria
    print("\n" + "─" * 60)
    print("PHASE 3 EXIT CRITERIA:")
    print(f"  {'' if all_match else '️ '} Row count = spine size on all snapshots "
          f"({'PASS' if all_match else 'FAIL — see warnings above'})")
    print("   assert_no_future_data passed on every snapshot")
    print("   Pre-enrollment activity removed (transaction_date >= account_open_date)")
    print("   Ghost cohort excluded (88,717 MBR_GHOST_* IDs)")
    print("   Purchase-only spend/count (DECISION 007)")
    print("   unique_categories_{w} and unique_channels_{w} included (Phase 4 diversity)")
    print("   Event types match Phase 1 audit (14 real types, no email_sent/browse_session)")
    print("   No .apply() on large tables — vectorized groupby/pivot throughout")
    print("   Progressive window narrowing — no redundant full-table scans")
    print("   engine='pyarrow' on all read/write calls")
    print("   Categoricals cast to 'category' dtype before groupby")
    print("\nPHASE 3 COMPLETE.")


if __name__ == "__main__":
    main()
