"""
pipeline.py -- TBIE Submission Pipeline

Single-command entry point for the Kobie x PES University TBIE Hackathon.
All dates derive from --observation_date. Models are frozen (no refit at inference).

Usage:
    python pipeline.py --data_dir ./data/train/ \
                       --observation_date 2025-12-31 \
                       --output_dir ./outputs \
                       --k 5

Output files written to --output_dir:
    segment_assignments.csv
    state_assignments.csv
    transition_predictions.csv
    segment_profiles.json
    feature_descriptions.json
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

# ── STEP 0: Verify dependencies ──────────────────────────────────────────────
# This checks and reports; it does NOT install. A data pipeline that runs
# `pip install` into the ambient interpreter mutates the caller's environment
# as a side effect of asking for a prediction, which is not something a
# scoring harness or a CI runner should have to tolerate. Install with
# `pip install -r requirements.txt` (or use the Dockerfile) before running.

def _verify_dependencies() -> None:
    required = {
        "pandas":       "pandas",
        "numpy":        "numpy",
        "pyarrow":      "pyarrow",
        "scikit-learn": "sklearn",
        "xgboost":      "xgboost",
        "joblib":       "joblib",
        "scipy":        "scipy",
    }
    import importlib
    missing = []
    for dist_name, import_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(dist_name)

    if missing:
        print("FATAL: missing required package(s): " + ", ".join(missing))
        print("Install them with:  pip install -r requirements.txt")
        sys.exit(1)


_verify_dependencies()

# ── Runtime imports ──────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# src/ on the path so the shared rule module is importable at module scope.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

sys.stdout.reconfigure(encoding="utf-8")


T0_GLOBAL = time.time()


def elapsed() -> str:
    return f"{time.time() - T0_GLOBAL:.1f}s"


# Clip bounds from Phase 1 audit. These are data-intrinsic constants and
# do not change with observation_date.
CLIP_BOUNDS_INLINE = {
    "session_duration_sec": {"upper": 14400.0},   # 4-hour cap on open/timeout sessions
    "transaction_amount":   {"upper": 205.5},      # 99th-pct soft cap; negatives = returns
}

# Assumed base recovery rate: the share of a member's at-risk value that a
# successful contact saves. NOT measurable from this data — campaign exposure
# was not randomised, so no counterfactual exists. It is scaled per member by
# observed relative responsiveness in STEP 5b, but the base stays an assumption.
# src/10_cost_thresholds.py sweeps it; break-even sits between 2% and 5%.
BASE_RECOVERY_RATE = 0.10

# State thresholds are NOT redefined here. They live as named constants in
# src/utils/state_rules.py alongside the rules that use them. This file used to
# carry its own copy, and several entries had already drifted out of agreement
# with the rules actually applied during training.


# CLI ARGUMENT PARSER

def parse_args():
    parser = argparse.ArgumentParser(
        description="TBIE Submission Pipeline -- Kobie x PES University Hackathon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate outputs at Dec 31 (submission date):
  python pipeline.py --data_dir ./data/raw --observation_date 2025-12-31 --output_dir ./outputs --k 5

  # Re-run at Month 13 (Kobie scoring date):
  python pipeline.py --data_dir ./data/raw --observation_date 2026-01-31 --output_dir ./outputs_jan --k 5
        """
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to raw data directory (must contain members.parquet, "
             "transactions.parquet, engagement_events.parquet)"
    )
    parser.add_argument(
        "--observation_date", type=str, required=True,
        help="Snapshot date in YYYY-MM-DD format (e.g. 2025-12-31)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where all 5 output files will be written"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of segments (default: 5). Must match frozen segment_model.pkl."
    )
    return parser.parse_args()


# MODULE LOADER (importlib -- avoids src package import issues on fresh machines)

def load_src_module(name: str, rel_path: str, root: Path):
    spec = importlib.util.spec_from_file_location(name, root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ghost members come in two flavours and BOTH must be excluded:
#   (a) tagged  -- member_id carries the MBR_GHOST_ prefix in members.parquet
#   (b) orphan  -- member_id appears in transactions/events but has no row in
#                  members.parquet at all (88,717 IDs in the Phase 1 audit)
# Only (a) can be found by scanning members.parquet; (b) is by definition
# absent from it. An earlier version checked the prefix alone and would have
# silently kept every orphan if the holdout data does not use that convention.

def compute_ghost_ids(members_df: pd.DataFrame,
                      activity_ids: set | None = None) -> set:
    """
    Identify ghost/test members dynamically. No external CSV required.

    activity_ids : member_ids observed in transactions + engagement_events.
                   Any that are absent from members.parquet are orphans.
    """
    ids_str    = members_df["member_id"].astype(str)
    ghost_mask = ids_str.str.startswith("MBR_GHOST_")
    tagged     = set(members_df.loc[ghost_mask, "member_id"].tolist())

    orphans: set = set()
    if activity_ids:
        orphans = set(activity_ids) - set(members_df["member_id"].tolist())

    print(f"  Ghost cohort: {len(tagged):,} tagged (MBR_GHOST_ prefix), "
          f"{len(orphans):,} orphan (in activity, absent from members)")
    if not tagged and not orphans:
        print("  NOTE: no ghost members found by either rule — verify this is "
              "expected for this dataset before trusting downstream counts.")
    return tagged | orphans



def build_spine_dynamic(members_df: pd.DataFrame, ghost_ids: set) -> pd.DataFrame:
    """
    Build member spine at runtime from members.parquet.
    Includes all real members (excludes ghost cohort).
    Matches schema produced by src/02_build_spine.py.
    """
    spine = members_df[~members_df["member_id"].isin(ghost_ids)][
        ["member_id", "account_open_date"]
    ].copy().reset_index(drop=True)
    # C-3 FIX: Force datetime dtype. Guards against object-typed parquets on
    # test machines where members.parquet may store account_open_date as string.
    spine["account_open_date"] = pd.to_datetime(spine["account_open_date"])
    print(f"  Spine built dynamically: {len(spine):,} real members")
    return spine



def build_cardholder_flags_dynamic(members_df: pd.DataFrame, ghost_ids: set) -> pd.DataFrame:
    """
    Compute PLCC cardholder flag at runtime from members.parquet.
    is_cardholder = 1 if credit_line is not null (PLCC cardholder), else 0.
    Covers every member in the current dataset, including new members.
    """
    real = members_df[~members_df["member_id"].isin(ghost_ids)].copy()
    real["is_cardholder"] = real["credit_line"].notna().astype(int)
    card_df = real[["member_id", "is_cardholder"]].reset_index(drop=True)
    pct = card_df["is_cardholder"].mean() * 100
    print(f"  Cardholder flags built dynamically: {pct:.1f}% PLCC cardholders")
    return card_df


# VELOCITY FEATURES (same formula as run_submission_gaps.py -- must stay in sync)

# Kept in sync with VELOCITY_COLS in src/08_transition_prediction.py. The
# loader below asserts the trained model asks for exactly these.
VELOCITY_FEATURES_PRODUCED = [
    "spend_velocity", "freq_velocity", "app_velocity",
    "recency_risk", "engagement_score", "spend_decline_flag",
]


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df = df.copy()
    df["spend_velocity"]     = df["spend_total_30d"] / (df["spend_total_90d"] / 3 + eps)
    df["freq_velocity"]      = df["purchase_count_30d"] / (df["purchase_count_90d"] / 3 + eps)
    df["app_velocity"]       = df["app_open_30d"] / (df["app_open_90d"] / 3 + eps)
    df["recency_risk"]       = df["recency_days"].fillna(999) * (
        1 - (df["purchase_count_30d"] / 10).clip(0, 1)
    )
    df["engagement_score"]   = (
        df["app_open_30d"].fillna(0)
        + df["email_open_30d"].fillna(0)
        + df["push_open_30d"].fillna(0)
    )
    df["spend_decline_flag"] = (df["spend_velocity"] < 0.5).astype(int)
    for vc in ["spend_velocity", "freq_velocity", "app_velocity"]:
        df[vc] = df[vc].clip(0, 10)
    return df


# SEGMENT ASSIGNMENT (frozen centroids -- transform only, NEVER refit)

def assign_segments(feat_df: pd.DataFrame, seg_model: dict,
                    cluster_to_sid: dict, sid_to_name: dict):
    """
    Nearest-centroid assignment using frozen K-Means centroids.
    No fit() call anywhere.

    Confidence is the normalised margin between the nearest and second-nearest
    centroid: (d2 - d1) / (d2 + d1), which spans the full [0, 1] range —
    0 = exactly equidistant between two centroids (no information),
    1 = sitting on a centroid.

    The previous formula, 1 - d1/(d1+d2), was bounded to [0.5, 1.0] because
    d1 <= d2 by construction. It made an uninformative assignment print as
    "0.50 confidence", which reads like a coin flip but is actually the floor.
    """
    cols      = seg_model["behavioral_feature_cols"]
    scaler    = seg_model["scaler"]
    pca       = seg_model["pca"]
    centroids = seg_model["centroids"]

    X         = feat_df[cols].fillna(0).values
    X_scaled  = scaler.transform(X)       # frozen scaler -- transform only
    X_pca     = pca.transform(X_scaled)   # frozen PCA   -- transform only

    cids_s          = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[k] for k in cids_s])

    dists        = cdist(X_pca, centroid_matrix, metric="euclidean")
    nearest_idx  = dists.argmin(axis=1)
    nearest_dist = dists.min(axis=1)
    dists_copy   = dists.copy()
    dists_copy[np.arange(len(dists)), nearest_idx] = np.inf
    second_dist  = dists_copy.min(axis=1)

    cluster_ids   = np.array(cids_s)[nearest_idx]
    segment_ids   = np.array([cluster_to_sid[c] for c in cluster_ids])
    segment_names = np.array([sid_to_name[s] for s in segment_ids])
    confidence    = ((second_dist - nearest_dist)
                     / (second_dist + nearest_dist + 1e-9)).clip(0, 1)
    return segment_ids, segment_names, confidence.round(4)


# STATE CLASSIFICATION
#
# The cascade itself lives in src/utils/state_rules.py and is shared with
# src/07_lifecycle_states.py, which produces the labels the transition model
# trains on. pipeline.py previously carried its own copy, and the two drifted:
# Value Maximizer required redemption_rate >= 0.21 here versus > 0.10 in
# training, Momentum Builder used >= instead of >, and Plateau Cruiser gained
# an undocumented recency clause. Because the resulting state feeds the
# transition model as y_curr, the drift changed the feature distribution
# between training and inference. One implementation, imported by both.

from utils.contact_history import (  # noqa: E402  (src/ is on sys.path above)
    DEFAULT_FREQUENCY_CAP_30D as FREQUENCY_CAP_30D,
)
from utils.contact_history import (  # noqa: E402
    add_response_propensity,
    apply_frequency_cap,
    build_contact_history,
    effective_recovery_rate,
)
from utils.eligibility import (  # noqa: E402
    build_contact_profile,
    recommend,
)
from utils.eligibility import summarise as summarise_eligibility  # noqa: E402
from utils.lag_features import (  # noqa: E402
    add_delta_features,
    add_segment_history,
)
from utils.pairs import to_model_matrix  # noqa: E402
from utils.state_rules import (  # noqa: E402
    MOMENTUM_SLOPE_MIN,
    RECENCY_LAPSE_MIN,
    STATE_IDS,
    VALID_STATES,
    classify_states,
)


def compute_state_confidence(df: pd.DataFrame, state_labels) -> pd.Series:
    """
    Threshold-margin confidence, clipped to [0.55, 0.95].

    Thresholds come from the canonical rule module so this can never disagree
    with the cascade that produced the labels.
    """
    conf    = pd.Series(0.68, index=df.index)
    recency = df["recency_days"].fillna(999)
    slope   = df["spend_slope_30d"].fillna(0)
    thresh_rec = RECENCY_LAPSE_MIN
    thresh_sl  = MOMENTUM_SLOPE_MIN

    sl = pd.Series(state_labels, index=df.index)
    lapse_m   = sl == "Lapse Risk"
    moment_m  = sl == "Momentum Builder"
    new_m     = sl == "New & Uncertain"
    winback_m = sl == "Win-Back Target"

    if lapse_m.any():
        margin = (recency[lapse_m] - thresh_rec) / max(thresh_rec, 1)
        conf[lapse_m] = (0.60 + margin.clip(0, 0.35)).clip(0.55, 0.95)
    if moment_m.any():
        margin = (slope[moment_m] - thresh_sl) / max(thresh_sl, 1)
        conf[moment_m] = (0.62 + margin.clip(0, 0.33)).clip(0.55, 0.95)
    conf[new_m]     = 0.55
    conf[winback_m] = 0.62
    return conf.round(4)


def build_supporting_evidence(df: pd.DataFrame, state_labels) -> pd.Series:
    """Vectorized evidence strings in Kobie format: 'metric:value,metric:value'."""
    freq_30  = df["purchase_count_30d"].fillna(0).astype(int)
    recency  = df["recency_days"].fillna(0).astype(int)
    slope    = df["spend_slope_30d"].fillna(0).round(1)
    redeem   = df["redemption_rate"].fillna(0).round(2)
    app      = df["app_open_30d"].fillna(0).astype(int)
    email    = df["email_open_30d"].fillna(0).astype(int)
    tier     = df["tier_ordinal"].fillna(0).astype(int)
    diversity= df["category_diversity_90d"].fillna(0).round(2)
    p7       = df["purchase_count_7d"].fillna(0).astype(int)
    p90      = df["purchase_count_90d"].fillna(0)
    p180     = df["purchase_count_180d"].fillna(0).astype(int)
    decline_pct = ((freq_30 - p90 / 3) / (p90 / 3 + 1) * 100).round(0).astype(int)

    sl = pd.Series(state_labels, index=df.index)
    evidence = pd.Series([""] * len(df), index=df.index)

    def ev(mask, s):
        if mask.any():
            evidence[mask] = s[mask]

    ev(sl == "Lapse Risk",
       "txn_decline:" + decline_pct.astype(str) + "%,recency_days:" +
       recency.astype(str) + ",purchases_30d:" + freq_30.astype(str))
    ev(sl == "Momentum Builder",
       "spend_slope:" + slope.astype(str) + ",purchases_30d:" +
       freq_30.astype(str) + ",tier:" + tier.astype(str))
    ev(sl == "Silent Accumulator",
       "app_opens_30d:" + app.astype(str) + ",email_opens_30d:" +
       email.astype(str) + ",purchases_30d:" + freq_30.astype(str))
    ev(sl == "Brand Advocate",
       "app_opens_30d:" + app.astype(str) + ",email_opens_30d:" +
       email.astype(str) + ",tier:" + tier.astype(str) +
       ",category_diversity:" + diversity.astype(str))
    ev(sl == "Program Skeptic",
       "app_opens_30d:" + app.astype(str) + ",email_opens_30d:" +
       email.astype(str) + ",purchases_30d:" + freq_30.astype(str))
    ev(sl == "Win-Back Target",
       "purchases_7d:" + p7.astype(str) + ",prior_83d_activity:" +
       (p90 - p7).clip(lower=0).astype(int).astype(str) + ",recency_days:" + recency.astype(str))
    ev(sl == "Value Maximizer",
       "redemption_rate:" + redeem.astype(str) + ",category_diversity:" +
       diversity.astype(str) + ",purchases_30d:" + freq_30.astype(str))
    ev(sl == "Redemption Hunter",
       "redemption_rate:" + redeem.astype(str) + ",purchases_30d:" + freq_30.astype(str))
    ev(sl == "Plateau Cruiser",
       "spend_slope:" + slope.astype(str) + ",purchases_30d:" +
       freq_30.astype(str) + ",recency_days:" + recency.astype(str))
    ev(sl == "New & Uncertain",
       "purchase_count_180d:" + p180.astype(str) + ",account_age_days:unknown")
    evidence[evidence == ""] = "no_dominant_signal:true"
    return evidence


# MAIN PIPELINE

def main():
    args = parse_args()

    ROOT        = Path(__file__).resolve().parent
    DATA_DIR    = Path(args.data_dir).resolve()
    OUTPUT_DIR  = Path(args.output_dir).resolve()
    OBS_DATE    = args.observation_date            # string "YYYY-MM-DD"
    OBS_DATE_TS = pd.Timestamp(OBS_DATE)
    K           = args.k

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("TBIE SUBMISSION PIPELINE")
    print("=" * 70)
    print(f"  observation_date : {OBS_DATE}")
    print(f"  data_dir         : {DATA_DIR}")
    print(f"  output_dir       : {OUTPUT_DIR}")
    print(f"  k                : {K}")
    print("=" * 70 + "\n")

    # Prerequisite checks
    # Only models/ and segments/ required -- outputs/ and validation/ are NOT.
    seg_model_path   = ROOT / "segments" / "segment_model.pkl"
    trans_model_path = ROOT / "models"   / "segment_transition_model.pkl"
    seg_defs_path    = ROOT / "segments" / "segment_definitions.json"
    state_defs_path  = ROOT / "models"   / "state_definitions.json"
    feat_desc_src    = ROOT / "models"   / "feature_descriptions.json"

    for p in [seg_model_path, trans_model_path, seg_defs_path, state_defs_path]:
        if not p.exists():
            print(f"FATAL: Required model file not found: {p}")
            sys.exit(1)

    # Load frozen models (no refit anywhere)
    print("Loading frozen models...")
    seg_model    = joblib.load(seg_model_path)
    trans_bundle = joblib.load(trans_model_path)
    trans_model  = trans_bundle["model"]
    X_COLS       = trans_bundle["feature_cols"]
    VEL_COLS     = trans_bundle["velocity_cols"]
    CLUSTER_TO_SID = trans_bundle["cluster_to_sid"]
    SID_TO_NAME    = trans_bundle["sid_to_name"]
    STATE_MAP      = trans_bundle["state_map"]
    # Empty for models trained with --no-lag-features.
    LAG_BASE_COLS  = trans_bundle.get("lag_base_cols", [])

    # add_velocity_features() below must produce exactly the velocity columns
    # the model was trained on. Checking here turns a silent all-zero column
    # (reindex fills missing columns with 0) into an explicit failure.
    missing_vel = [c for c in VEL_COLS if c not in VELOCITY_FEATURES_PRODUCED]
    if missing_vel:
        print(f"FATAL: model expects velocity feature(s) this pipeline does not "
              f"compute: {missing_vel}")
        sys.exit(1)

    f1_test = trans_bundle.get("macro_f1_test", float("nan"))
    print(f"  segment_model.pkl loaded (k={K}) | {elapsed()}")
    print(f"  segment_transition_model.pkl loaded (test F1={f1_test:.4f}) | {elapsed()}")

    cluster_ids_sorted = sorted(CLUSTER_TO_SID.keys())
    valid_sids  = set(CLUSTER_TO_SID.values())
    sid_to_cid  = {v: k for k, v in CLUSTER_TO_SID.items()}
    prob_cols   = [f"prob_{CLUSTER_TO_SID[cid]}" for cid in cluster_ids_sorted]

    with open(seg_defs_path) as f:
        seg_defs = json.load(f)

    print("\nSegment mapping:")
    for cid, sid in CLUSTER_TO_SID.items():
        print(f"  int {cid} -> {sid} ({SID_TO_NAME[sid]})")

    # The bundle's state_map is the encoding y_curr was trained with. If it
    # disagrees with the canonical priority encoding, the transition model
    # would be served a scrambled feature -- fail rather than silently degrade.
    if STATE_MAP != STATE_IDS:
        print("FATAL: model bundle state_map disagrees with src/utils/state_rules.STATE_IDS")
        print(f"  bundle    : {dict(sorted(STATE_MAP.items(), key=lambda x: x[1]))}")
        print(f"  state_rules: {dict(sorted(STATE_IDS.items(), key=lambda x: x[1]))}")
        print("  Retrain with src/08_transition_prediction.py to refresh the bundle.")
        sys.exit(1)
    print(f"\nState encoding verified against model bundle "
          f"({len(STATE_MAP)} states) | {elapsed()}")

    # RAW DATA LOADING -- done ONCE, shared across observation_date + prior month
    print(f"\n{'='*60}")
    print(f"LOADING RAW DATA from {DATA_DIR}")
    print(f"{'='*60}")

    sys.path.insert(0, str(ROOT / "src"))
    snap_mod = load_src_module("snap_mod", "src/03_snapshot_builder.py", ROOT)
    feat_mod = load_src_module("feat_mod", "src/04_feature_engine.py", ROOT)

    # Members must be loaded first (ghost cohort, spine, cardholder flags)
    print("  Loading members.parquet...")
    members = pd.read_parquet(DATA_DIR / "members.parquet", engine="pyarrow")
    print(f"  members: {len(members):,} rows | {elapsed()}")

    # Load raw transactions and engagement events BEFORE resolving the ghost
    # cohort -- orphan ghosts can only be found by diffing activity against
    # members.parquet, so the activity tables must be in hand first.
    print("  Loading transactions.parquet...")
    txns_raw = pd.read_parquet(DATA_DIR / "transactions.parquet", engine="pyarrow")
    print("  Loading engagement_events.parquet...")
    events_raw = pd.read_parquet(DATA_DIR / "engagement_events.parquet", engine="pyarrow")
    print(f"  txns: {len(txns_raw):,}  events: {len(events_raw):,} | {elapsed()}")

    activity_ids = set(txns_raw["member_id"].unique()) | set(events_raw["member_id"].unique())
    ghost_ids    = compute_ghost_ids(members, activity_ids)

    spine = build_spine_dynamic(members, ghost_ids)

    card_df = build_cardholder_flags_dynamic(members, ghost_ids)

    # Parse datetimes using the same dual-format parser as Phase 3
    from utils.datetime_parser import parse_mixed_datetime
    txns_raw["transaction_date"] = parse_mixed_datetime(
        txns_raw["transaction_date"], col_name="transaction_date"
    )
    events_raw["event_date"] = parse_mixed_datetime(
        events_raw["event_date"], col_name="event_date"
    )

    # Normalize categoricals (strip + lower), then keep them as category dtype
    # for groupby performance at 18M/35M row scale.
    #
    # Order matters for memory. Running .str.strip().str.lower() on a 35M-row
    # object column materialises two full intermediate object arrays and blew
    # up with a 542 MiB allocation failure. Converting to `category` first means
    # the normalisation runs over the category dictionary — a few dozen distinct
    # values — and the 35M codes array is never touched.
    def normalise_categorical(df: pd.DataFrame, cols: list) -> None:
        for col in cols:
            if col not in df.columns:
                continue
            cat = df[col].astype("category")
            normalised = pd.Index([str(c).strip().lower() for c in cat.cat.categories])

            if normalised.is_unique:
                df[col] = cat.cat.rename_categories(normalised)
                continue

            # Normalisation collapsed distinct raw labels onto one value
            # (" Online" and "online"). rename_categories rejects duplicates,
            # so remap the integer codes onto the deduplicated dictionary —
            # still only touches an int8/int16 array, never the strings.
            unique_cats = normalised.unique()
            old_to_new  = unique_cats.get_indexer(normalised)
            codes       = cat.cat.codes.to_numpy()
            new_codes   = np.where(codes >= 0, old_to_new[codes.clip(min=0)], -1)
            df[col] = pd.Categorical.from_codes(new_codes, categories=unique_cats)

    normalise_categorical(txns_raw, ["channel", "transaction_type",
                                     "merchant_category", "merchant_subcategory",
                                     "merchant_brand"])
    normalise_categorical(events_raw, ["event_type", "event_channel"])

    # Drop exact duplicate engagement rows
    n_before = len(events_raw)
    events_raw = events_raw.drop_duplicates(
        subset=["member_id", "event_date", "event_type"]
    )
    print(f"  Dropped {n_before - len(events_raw):,} duplicate engagement rows")

    if "session_duration_sec" in events_raw.columns:
        cap = CLIP_BOUNDS_INLINE["session_duration_sec"]["upper"]
        n_over = (events_raw["session_duration_sec"] > cap).sum()
        events_raw["session_duration_sec"] = events_raw["session_duration_sec"].clip(upper=cap)
        print(f"  Clipped {n_over:,} session_duration_sec values > {cap}s (FIX-8 inline)")

    txns_raw   = txns_raw[~txns_raw["member_id"].isin(ghost_ids)].copy()
    events_raw = events_raw[~events_raw["member_id"].isin(ghost_ids)].copy()
    print(f"  After ghost exclusion: txns={txns_raw.shape}, events={events_raw.shape} | {elapsed()}")

    # STEP 1 -- Features at observation_date
    print(f"\n{'='*60}")
    print(f"STEP 1: Features at {OBS_DATE}")
    print(f"{'='*60}")

    date_str_underscore = OBS_DATE.replace("-", "_")

    clip_bounds = CLIP_BOUNDS_INLINE

    prior_date     = OBS_DATE_TS - pd.DateOffset(months=1)
    prior_str      = prior_date.strftime("%Y_%m_%d")
    prior_feat_path = ROOT / "features" / f"features_{prior_str}.parquet"

    prior_feat = None
    if prior_feat_path.exists():
        prior_feat = pd.read_parquet(prior_feat_path, engine="pyarrow")
        print(f"  Prior features loaded from disk: {prior_feat_path.name}")
    else:
        print(f"  Prior features not on disk -- building {prior_date.date()} in memory (FIX-5)...")
        prior_snapshot = snap_mod.build_snapshot(prior_date, spine, txns_raw, events_raw, members)
        prior_feat = feat_mod.engineer_features(
            prior_snapshot, txns_raw, prior_date, clip_bounds, prior_snapshot_df=None
        )
        feat_dir = ROOT / "features"
        feat_dir.mkdir(exist_ok=True)
        prior_feat.to_parquet(prior_feat_path, engine="pyarrow", index=False)
        print(f"  Prior features ({len(prior_feat):,} rows) built and cached | {elapsed()}")

    feat_path = ROOT / "features" / f"features_{date_str_underscore}.parquet"
    snap_out  = ROOT / "snapshots" / f"snapshot_{date_str_underscore}.parquet"
    feat_path.parent.mkdir(exist_ok=True)
    snap_out.parent.mkdir(exist_ok=True)

    if feat_path.exists():
        # Fast path: pre-built feature file found on disk (e.g. included in submission zip).
        # Outputs are identical to a full rebuild -- the file was produced from the same raw data.
        feat_df = pd.read_parquet(feat_path, engine="pyarrow")
        print(f"  Features loaded from cache: {feat_path.name}  ({len(feat_df):,} rows) | {elapsed()}")

        # Derive tenure_days if missing (older cached feature files lack it)
        if "tenure_days" not in feat_df.columns and "account_open_date" in feat_df.columns:
            feat_df["tenure_days"] = (OBS_DATE_TS - pd.to_datetime(feat_df["account_open_date"])).dt.days
            print("  Derived tenure_days from account_open_date")
    else:
        # Slow path: build snapshot and engineer features from raw data.
        print(f"  Building snapshot at {OBS_DATE}...")
        snapshot = snap_mod.build_snapshot(OBS_DATE_TS, spine, txns_raw, events_raw, members)
        snapshot.to_parquet(snap_out, engine="pyarrow", index=False)
        print(f"  Snapshot: {len(snapshot):,} rows | {elapsed()}")

        print("  Engineering features...")
        feat_df = feat_mod.engineer_features(
            snapshot, txns_raw, OBS_DATE_TS, clip_bounds, prior_snapshot_df=prior_feat
        )
        feat_df.to_parquet(feat_path, engine="pyarrow", index=False)
        print(f"  Features: {len(feat_df):,} rows x {feat_df.shape[1]} cols | {elapsed()}")

    # STEP 2 -- segment_assignments.csv
    print(f"\n{'='*60}")
    print("STEP 2: segment_assignments.csv")
    print(f"{'='*60}")

    seg_ids, seg_names, seg_conf = assign_segments(
        feat_df, seg_model, CLUSTER_TO_SID, SID_TO_NAME
    )

    seg_out = pd.DataFrame({
        "member_id":          feat_df["member_id"].values,
        "observation_date":   OBS_DATE,
        "segment_id":         seg_ids,
        "segment_name":       seg_names,
        "segment_confidence": seg_conf,
    })

    n_rows = len(seg_out)
    EXPECTED_ROWS = len(spine)  # M-3 FIX: dynamic count from spine, not hardcoded 500_000
    assert n_rows == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} rows, got {n_rows}"
    assert seg_out["segment_confidence"].between(0, 1).all()
    assert seg_out["segment_id"].isin(valid_sids).all()

    seg_out.to_csv(OUTPUT_DIR / "segment_assignments.csv", index=False)
    print(f"  Rows: {n_rows:,} | observation_date: {OBS_DATE} | {elapsed()}")
    print(f"  Distribution:\n{seg_out['segment_name'].value_counts().to_string()}")
    print(f"  Avg confidence: {seg_conf.mean():.4f}")

    # STEP 3 -- state_assignments.csv
    print(f"\n{'='*60}")
    print("STEP 3: state_assignments.csv")
    print(f"{'='*60}")

    state_labels   = classify_states(feat_df)
    state_labels_s = pd.Series(state_labels, index=feat_df.index)
    state_conf     = compute_state_confidence(feat_df, state_labels_s)
    evidence_str   = build_supporting_evidence(feat_df, state_labels)

    state_out = pd.DataFrame({
        "member_id":           feat_df["member_id"].values,
        "observation_date":    OBS_DATE,
        "state_name":          state_labels,
        "state_confidence":    state_conf.values,
        "supporting_evidence": evidence_str.values,
    })

    assert len(state_out) == EXPECTED_ROWS, f"state_out: expected {EXPECTED_ROWS:,} rows, got {len(state_out)}"
    invalid = set(state_out["state_name"].unique()) - VALID_STATES
    assert not invalid, f"Invalid states: {invalid}"
    assert state_out["state_confidence"].between(0, 1).all()
    assert state_out["supporting_evidence"].str.contains(":").all()

    state_out.to_csv(OUTPUT_DIR / "state_assignments.csv", index=False)
    print(f"  Rows: {len(state_out):,} | {elapsed()}")
    print(f"  State distribution:\n{state_out['state_name'].value_counts().to_string()}")

    # STEP 4 -- transition_predictions.csv  (frozen XGBoost -- no refit)
    print(f"\n{'='*60}")
    print("STEP 4: transition_predictions.csv")
    print(f"{'='*60}")
    print("  Using frozen XGBoost model -- stable and reproducible across any obs date")

    feat_sub = add_velocity_features(feat_df)
    feat_sub["seg_curr"]  = seg_out["segment_id"].map(sid_to_cid).values.astype(int)
    # No fillna here on purpose. A state name that is not in STATE_MAP means the
    # cascade and the trained encoding have diverged; defaulting to 0 would hide
    # that behind a plausible-looking prediction.
    y_curr_mapped = state_out["state_name"].map(STATE_MAP)
    if y_curr_mapped.isna().any():
        unknown = sorted(state_out.loc[y_curr_mapped.isna(), "state_name"].unique())
        raise ValueError(f"State(s) absent from the trained state_map: {unknown}")
    feat_sub["y_curr"]    = y_curr_mapped.astype(int)
    feat_sub["month_num"] = OBS_DATE_TS.month

    # Month-over-month change features, if the model was trained with them.
    # These must be rebuilt here from the prior month's snapshot: reindex()
    # below fills absent columns with 0, so a missing delta column would not
    # raise -- it would quietly feed the model "no change for every member".
    if LAG_BASE_COLS:
        print(f"  Building {len(LAG_BASE_COLS)} delta features + segment history "
              f"from {prior_date.date()}...")
        feat_sub = add_delta_features(feat_sub, prior_feat, LAG_BASE_COLS)

        # Prior-month segment, assigned with the SAME frozen centroids.
        prev_sids, _, _ = assign_segments(prior_feat, seg_model,
                                          CLUSTER_TO_SID, SID_TO_NAME)
        prev_cids = pd.Series(prev_sids).map(sid_to_cid)
        prev_by_member = pd.Series(prev_cids.values,
                                   index=prior_feat["member_id"].values)
        feat_sub = add_segment_history(
            feat_sub, feat_sub["member_id"].map(prev_by_member).values
        )
        n_moved = int(feat_sub["seg_changed"].sum())
        print(f"  seg_changed: {n_moved:,} members moved segment since "
              f"{prior_date.date()} ({n_moved / len(feat_sub) * 100:.1f}%)")

    # Shared with training: raises on an absent feature, and fills NaN with 0
    # so rows take the same tree branches they did at fit time.
    X_sub     = to_model_matrix(feat_sub, X_COLS)
    probs_sub = trans_model.predict_proba(X_sub)   # frozen -- no refit
    pred_idx  = probs_sub.argmax(axis=1)

    trans_out = pd.DataFrame({"member_id": feat_df["member_id"].values})
    trans_out["current_segment_id"]    = seg_out["segment_id"].values
    trans_out["predicted_segment_id"]  = np.array([CLUSTER_TO_SID[i] for i in pred_idx])
    trans_out["prediction_confidence"] = probs_sub.max(axis=1).round(4)
    for i, cid in enumerate(cluster_ids_sorted):
        trans_out[f"prob_{CLUSTER_TO_SID[cid]}"] = probs_sub[:, i].round(4)

    assert len(trans_out) == EXPECTED_ROWS, f"trans_out: expected {EXPECTED_ROWS:,} rows, got {len(trans_out)}"
    prob_sums = trans_out[prob_cols].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-3), \
        f"Probabilities don't sum to 1.0: min={prob_sums.min():.6f}"
    assert trans_out["current_segment_id"].isin(valid_sids).all()
    assert trans_out["predicted_segment_id"].isin(valid_sids).all()

    trans_out.to_csv(OUTPUT_DIR / "transition_predictions.csv", index=False)
    print(f"  Rows: {len(trans_out):,} | {elapsed()}")
    print(f"  Columns: {trans_out.columns.tolist()}")
    print(f"  Prob sum: min={prob_sums.min():.6f} max={prob_sums.max():.6f} "
          f"mean={prob_sums.mean():.6f}")

    # STEP 5 -- segment_profiles.json
    print(f"\n{'='*60}")
    print("STEP 5: segment_profiles.json")
    print(f"{'='*60}")

    combined  = seg_out.merge(state_out[["member_id", "state_name"]], on="member_id")
    combined  = combined.merge(feat_df, on="member_id")
    combined  = combined.merge(card_df, on="member_id", how="left")
    combined["is_cardholder"] = combined["is_cardholder"].fillna(0).astype(int)

    FEAT_COLS_PROFILE = seg_model["behavioral_feature_cols"]
    profiles = {}

    # Segment-level tone modifier — appended to each activation string so
    # High-Tier Accelerator × Brand Advocate reads differently from
    # Growth Builder × Brand Advocate, satisfying the segment×state requirement.
    SEGMENT_TONE = {
        "High-Tier Accelerator": "VIP/exclusive framing — no discounts",
        "Growth Builder":        "progression/momentum framing",
        "Program Skeptic":       "low-commitment offer — value proof first",
        "Silent Accumulator":    "simple, transactional framing",
        "Plateau Cruiser":       "personalised curation framing",
    }

    ACTIVATION_MAP = {
        "Lapse Risk":         "Channel: email+SMS | Message: personalised win-back with bonus points | Offer: double points on next purchase | Timing: within 7 days",
        "Momentum Builder":   "Channel: app push | Message: tier upgrade progress | Offer: tier accelerator bonus | Timing: immediate",
        "Silent Accumulator": "Channel: app push | Message: unspent points reminder | Offer: limited-time redemption bonus | Timing: weekly",
        "Program Skeptic":    "Channel: email | Message: re-permission with value proof | Offer: one-time surprise reward | Timing: monthly",
        "Win-Back Target":    "Channel: email+SMS | Message: we miss you | Offer: reactivation bonus points | Timing: immediate",
        "Brand Advocate":     "Channel: app | Message: early access to new reward | Offer: referral bonus | Timing: immediate",
        "Value Maximizer":    "Channel: email | Message: cross-category points multiplier | Offer: category diversity bonus | Timing: mid-month",
        "Redemption Hunter":  "Channel: email+push | Message: targeted flash promotion | Offer: limited-time category reward | Timing: around promotional periods",
        "Plateau Cruiser":    "Channel: email | Message: personalised recommendation | Offer: curated reward unlock | Timing: monthly",
        "New & Uncertain":    "Channel: app onboarding | Message: welcome journey | Offer: first-purchase bonus | Timing: within 7 days of enrollment",
    }

    # Pre-compute population stats for z-score normalisation (Fix: raw-diff ranking
    # always surfaces large-scale dollar/point columns regardless of segment;
    # z-score surfaces features that genuinely differentiate each cluster).
    overall_means = combined[FEAT_COLS_PROFILE].mean()
    overall_stds  = combined[FEAT_COLS_PROFILE].std().replace(0, 1)  # avoid div-by-zero

    for sid in sorted(valid_sids):
        mask    = combined["segment_id"] == sid
        seg_g   = combined[mask]
        name    = SID_TO_NAME[sid]
        cid_int = sid_to_cid[sid]

        # H-FIX: Guard for empty segment (possible at non-Dec observation_dates)
        if mask.sum() == 0:
            profiles[sid] = {
                "segment_id":   sid, "segment_name": name,
                "description":  f"Segment {sid}: {name}",
                "size": {"count": 0, "percentage": 0.0},
                "key_characteristics": {},
                "cardholder_composition": {"plcc_cardholder_pct": 0.0, "non_cardholder_pct": 0.0},
                "recommended_activation": {s: "No members" for s in VALID_STATES},
                "common_states": [],
            }
            print(f"  {sid} ({name}): 0 members at {OBS_DATE} -- empty profile written")
            continue

        # z-score normalized difference — surfaces features that separate THIS
        # cluster from the population, not just the highest-magnitude columns.
        cluster_means = seg_g[FEAT_COLS_PROFILE].mean()
        z_diff        = ((cluster_means - overall_means) / overall_stds).abs()
        top5_features = z_diff.nlargest(5).index.tolist()
        key_chars = {
            f: {"mean":   float(round(seg_g[f].mean(), 4)),
                "median": float(round(seg_g[f].median(), 4))}
            for f in top5_features
        }

        common_states = seg_g["state_name"].value_counts().head(3).index.tolist()
        plcc_pct      = float(seg_g["is_cardholder"].mean() * 100)

        try:
            description = seg_defs["segments"][str(cid_int)]["business_interpretation"]
        except (KeyError, TypeError):
            description = f"Segment {sid}: {name}"

        # Activation strings — base message from ACTIVATION_MAP, then appended
        # with the segment's tone modifier so each segment×state entry is unique.
        seg_tone = SEGMENT_TONE.get(name, "")
        activation = {}
        for state in VALID_STATES:
            n_combo = len(seg_g[seg_g["state_name"] == state])
            if n_combo == 0:
                activation[state] = f"No members in {name} x {state} combination"
            else:
                base = ACTIVATION_MAP.get(state, "Channel: email | Timing: monthly")
                activation[state] = f"{base} | Tone: {seg_tone}" if seg_tone else base

        profiles[sid] = {
            "segment_id":   sid,
            "segment_name": name,
            "description":  description,
            "size": {
                "count":      int(mask.sum()),
                "percentage": float(round(mask.mean() * 100, 2)),
            },
            "key_characteristics":    key_chars,
            "cardholder_composition": {
                "plcc_cardholder_pct": round(plcc_pct, 2),
                "non_cardholder_pct":  round(100 - plcc_pct, 2),
            },
            "recommended_activation": activation,
            "common_states": common_states,
        }
        print(f"  {sid} ({name}): {mask.sum():,} members, PLCC={plcc_pct:.1f}%")

    with open(OUTPUT_DIR / "segment_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"  segment_profiles.json saved | {elapsed()}")

    # STEP 5b -- contact_eligibility.csv
    #
    # An ADDITIONAL output; the five required submission files above are
    # unchanged. Without this, recommended_activation names a channel with no
    # regard for whether the member consented to it or is even an open account.
    # At 2025-12-31 that meant 222,259 members were being pointed at a channel
    # they had opted out of, and ~20k closed/fraud accounts were being targeted.
    print(f"\n{'='*60}")
    print("STEP 5b: contact_eligibility.csv")
    print(f"{'='*60}")

    profile = build_contact_profile(members)

    # Contact history: frequency capping and per-member responsiveness, built
    # strictly from campaign events dated <= observation_date.
    hist = build_contact_history(events_raw, OBS_DATE_TS)
    hist = add_response_propensity(hist)
    pop_rate = hist.attrs.get("population_response_rate", 0.0)
    profile = profile.merge(
        hist[["member_id", "contacts_30d", "contacts_90d", "contacts_180d",
              "responses_180d", "days_since_last_contact",
              "response_rate_smoothed", "response_multiplier"]],
        on="member_id", how="left",
    )
    # Never contacted => no history. Zero contacts, population-average
    # responsiveness (multiplier 1.0) rather than a pessimistic zero.
    profile["contacts_30d"] = profile["contacts_30d"].fillna(0)
    profile["contacts_90d"] = profile["contacts_90d"].fillna(0)
    profile["contacts_180d"] = profile["contacts_180d"].fillna(0)
    profile["responses_180d"] = profile["responses_180d"].fillna(0)
    profile["response_rate_smoothed"] = profile["response_rate_smoothed"].fillna(pop_rate)
    profile["response_multiplier"] = profile["response_multiplier"].fillna(1.0)

    n_before_cap = int(profile["is_targetable"].sum())
    profile = apply_frequency_cap(profile, cap_30d=FREQUENCY_CAP_30D)
    n_capped = n_before_cap - int(profile["is_targetable"].sum())

    elig = state_out[["member_id", "state_name"]].merge(profile, on="member_id", how="left")

    # A member with no row in members.parquet cannot be shown to have consented.
    for col in ["is_targetable", "needs_reactivation_track",
                "allow_email", "allow_push", "allow_sms"]:
        elig[col] = elig[col].fillna(False).astype(bool)
    for col in ["contacts_30d", "contacts_90d", "contacts_180d", "responses_180d"]:
        elig[col] = elig[col].fillna(0).astype(int)
    elig["response_multiplier"] = elig["response_multiplier"].fillna(1.0)
    elig["response_rate_smoothed"] = elig["response_rate_smoothed"].fillna(pop_rate)
    elig["suppression_reason"] = elig["suppression_reason"].fillna("not_in_member_file")
    elig.loc[elig["suppression_reason"] == "not_in_member_file", "is_targetable"] = False

    recs = [recommend(s, r) for s, r in zip(elig["state_name"], elig.to_dict("records"))]
    elig["recommended_channel"] = [c for c, _ in recs]
    elig["recommended_action"]  = [a for _, a in recs]
    elig["observation_date"]    = OBS_DATE

    # Per-member recovery rate: the assumed base scaled by relative
    # responsiveness. Clipped inside effective_recovery_rate because campaign
    # exposure was not randomised — this is relative propensity, not causal lift.
    elig["effective_recovery_rate"] = effective_recovery_rate(
        BASE_RECOVERY_RATE, elig["response_multiplier"].values
    ).round(4)

    elig_out = elig[[
        "member_id", "observation_date", "state_name",
        "is_targetable", "suppression_reason", "needs_reactivation_track",
        "allow_email", "allow_push", "allow_sms",
        "contacts_30d", "contacts_90d", "days_since_last_contact",
        "response_rate_smoothed", "response_multiplier", "effective_recovery_rate",
        "recommended_channel", "recommended_action",
    ]]
    elig_out.to_csv(OUTPUT_DIR / "contact_eligibility.csv", index=False)

    s = summarise_eligibility(profile)
    print(f"  Targetable          : {s['targetable']:>8,} of {s['members']:,} "
          f"({s['targetable']/s['members']:.1%})")
    print(f"  Suppressed          : {s['suppressed']:>8,}")
    print(f"    account closed    : {s['account_closed']:>8,}")
    print(f"    fraud flagged     : {s['fraud_flagged']:>8,}")
    print(f"    no channel consent: {s['no_channel_consent']:>8,}")
    print(f"    frequency cap     : {n_capped:>8,}  (>= {FREQUENCY_CAP_30D} contacts in 30d)")
    print(f"  Dormant -> reactivation track: {s['dormant_reactivation']:,}")
    print(f"  Contact history: pop response rate {pop_rate:.1%} | "
          f"median contacts_30d {elig['contacts_30d'].median():.0f}")
    print(f"  Effective recovery rate: base {BASE_RECOVERY_RATE:.0%} -> "
          f"p10 {elig['effective_recovery_rate'].quantile(.1):.1%} / "
          f"median {elig['effective_recovery_rate'].median():.1%} / "
          f"p90 {elig['effective_recovery_rate'].quantile(.9):.1%}")
    print(f"  Consent: email {s['allow_email']:,} | push {s['allow_push']:,} | "
          f"sms {s['allow_sms']:,}")
    n_uncontactable = int((elig_out["recommended_channel"] == "").sum())
    print(f"  Members with NO contactable recommendation: {n_uncontactable:,}")
    print(f"  contact_eligibility.csv saved | {elapsed()}")

    # STEP 6 -- feature_descriptions.json
    print(f"\n{'='*60}")
    print("STEP 6: feature_descriptions.json")
    print(f"{'='*60}")

    dst_feat_desc = OUTPUT_DIR / "feature_descriptions.json"
    if feat_desc_src.exists() and feat_desc_src.resolve() != dst_feat_desc.resolve():
        shutil.copy2(feat_desc_src, dst_feat_desc)
        print(f"  Copied from models/feature_descriptions.json | {elapsed()}")
    elif dst_feat_desc.exists():
        print(f"  feature_descriptions.json already at target | {elapsed()}")
    else:
        print("  WARNING: models/feature_descriptions.json not found. "
              "Copy outputs/feature_descriptions.json to models/ first.")

    # FINAL SUMMARY
    print(f"\n{'='*70}")
    print("ALL STEPS COMPLETE")
    print(f"  observation_date = {OBS_DATE}")
    print(f"  Total elapsed    = {elapsed()}")
    print(f"{'='*70}")
    print("Files written:")
    expected = [
        # The five required submission files.
        "segment_assignments.csv",
        "state_assignments.csv",
        "transition_predictions.csv",
        "segment_profiles.json",
        "feature_descriptions.json",
        # Additional: consent and eligibility. Not part of the required
        # submission schema, but no recommendation should be acted on without it.
        "contact_eligibility.csv",
    ]
    all_ok = True
    for fname in expected:
        p = OUTPUT_DIR / fname
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  OK      {fname}  ({size_kb:.0f} KB)")
        else:
            print(f"  MISSING {fname}")
            all_ok = False

    if not all_ok:
        sys.exit(1)

    # Validation prints (same checks Kobie runs for scoring)
    print("\n--- VALIDATION ---")
    sa = pd.read_csv(OUTPUT_DIR / "segment_assignments.csv")
    tp = pd.read_csv(OUTPUT_DIR / "transition_predictions.csv")
    print(f"observation_date unique: {sa['observation_date'].unique().tolist()}")
    print(f"transition_predictions columns: {tp.columns.tolist()}")
    print("prob sum stats:")
    print(tp[prob_cols].sum(axis=1).describe().to_string())
    print("\nPIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
