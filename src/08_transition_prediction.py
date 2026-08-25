"""
src/08_transition_prediction.py
══════════════════════════════════════════════════════════════════════════════
PHASE 8 — Segment Transition Prediction
TBIE Pipeline | Kobie × PES University Hackathon

Target: seg_next (Layer 1 segment at T+1) — NOT state_next
This is the only objectively graded metric (Macro F1 on NovDec test).

═══════════════════════════════════════════════════════════════════════════════
WALK-FORWARD VALIDATION SPLIT
═══════════════════════════════════════════════════════════════════════════════
Train:    FebMar, MarApr, AprMay, MayJun,
          JunJul, JulAug, AugSep, SepOct  (8 pairs)
Validate: OctNov                              (1 pair)
Test:     NovDec                              (1 pair — report this F1)
Holdout:  Months 13-14 (Kobie — we never touch this)

NO data from validation or test windows was used
to set hyperparameters. Early stopping evaluated
on validation only.
═══════════════════════════════════════════════════════════════════════════════

Feature groups:
  1. 40 behavioural columns curated by Phase 6 (the K-Means feature set)
  2. 6 within-snapshot velocity features (spend_velocity, freq_velocity,
     app_velocity, recency_risk, engagement_score, spend_decline_flag)
  3. Month-over-month deltas + segment history (seg_prev, seg_changed) —
     see src/utils/lag_features.py. Ablate with --no-lag-features.
  4. 3 context columns: seg_curr, y_curr, month_num

Model selection — every choice is made on VALIDATION:
  - sample weighting: inverse-frequency, plus a boost for classes whose
    validation F1 falls below WEAK_F1_THRESHOLD (two-pass, keeps the better
    of the two on validation)
  - decision rule: argmax vs per-class threshold calibration
  - early stopping: validation mlogloss
The test split is read exactly once, to produce the reported number.

Reported alongside the headline metric:
  - majority-class and persistence (seg_next = seg_curr) baselines
  - 95% bootstrap confidence interval on the test macro F1

Run from TBIE_CODE root:
    python src/08_transition_prediction.py
    python src/08_transition_prediction.py --no-lag-features --tag noLag
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.utils.class_weight import compute_sample_weight

from utils.lag_features import (  # noqa: E402
    SEG_HIST_COLS,
    add_delta_features,
    add_segment_history,
    delta_col_names,
    lag_col_names,
    resolve_lag_base_cols,
)

_argp = argparse.ArgumentParser(
    description="TBIE Phase 8 — segment transition model training"
)
_argp.add_argument(
    "--no-lag-features", action="store_true",
    help="Ablation: train without month-over-month delta and segment-history "
         "features, to measure their contribution.",
)
_argp.add_argument(
    "--tag", type=str, default="",
    help="Suffix for report filenames, so ablation runs do not overwrite each other.",
)
ARGS     = _argp.parse_args()
USE_LAG  = not ARGS.no_lag_features
RUN_TAG  = f"_{ARGS.tag}" if ARGS.tag else ""

PIPE_START = time.time()
def elapsed():
    return f"{time.time() - PIPE_START:.1f}s"

print("=" * 70)
print("TBIE — PHASE 8: SEGMENT TRANSITION PREDICTION  [v2 — Improved]")
print("=" * 70)

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
SEGMENTS_DIR = ROOT / "segments"
STATES_DIR   = ROOT / "states"
MODELS_DIR   = ROOT / "models"
OUTPUTS_DIR  = ROOT / "outputs"
VAL_DIR      = ROOT / "validation"

for d in [MODELS_DIR, OUTPUTS_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# STEP 8.1 — LOAD CONFIG + DISCOVER ALL USABLE FEATURE COLUMNS
print(f"\n[{elapsed()}] STEP 8.1 — Loading config + feature column discovery...")

model_pkl   = joblib.load(SEGMENTS_DIR / "segment_model.pkl")
N_SEGMENTS  = model_pkl['k']

with open(SEGMENTS_DIR / "segment_definitions.json", encoding="utf-8") as f:
    seg_defs = json.load(f)
SEG_ID_TO_NAME = {int(k): v["name"] for k, v in seg_defs["segments"].items()}
SEGMENT_NAMES  = [SEG_ID_TO_NAME[i] for i in range(N_SEGMENTS)]

# Use the same 40 behavioral cols Phase 6 selected (memory-safe)
# We add 6 velocity features on top — total 49 features vs v1's 43
BASE_FEAT_COLS  = model_pkl['behavioral_feature_cols']

print(f"  Segments (k={N_SEGMENTS}): {SEGMENT_NAMES}")
print(f"  Behavioral feature cols: {len(BASE_FEAT_COLS)}  (Phase 6 curated)")

# STEP 8.2 — VELOCITY / TREND FEATURE ENGINEERING
#   These 6 derived features specifically target Lapse Risk detection.
#   They capture HOW FAST a member is declining, not just their current state.
VELOCITY_COLS = [
    "spend_velocity",     # spending 30d vs 90d average — < 1.0 means slowing down
    "freq_velocity",      # purchase frequency 30d vs 90d average
    "app_velocity",       # app opens 30d vs 90d average
    "recency_risk",       # recency × (1 - min(freq/10, 1)) — high = lapse incoming
    "engagement_score",   # total digital touches in 30d (app+email+push)
    "spend_decline_flag", # 1 if spending is at less than half 90d pace
]

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives 6 trend features from multi-window columns already in the parquet.
    Fully vectorized — operates on the full DataFrame at once.
    """
    eps  = 1e-6
    out  = df.copy()

    # spend_velocity: how does last-30d spending compare to 90d/3 (monthly avg)?
    out["spend_velocity"]     = out["spend_total_30d"]     / (out["spend_total_90d"]     / 3 + eps)
    # freq_velocity: purchase frequency trend
    out["freq_velocity"]      = out["purchase_count_30d"]  / (out["purchase_count_90d"]  / 3 + eps)
    # app_velocity: digital engagement trend
    out["app_velocity"]       = out["app_open_30d"]        / (out["app_open_90d"]        / 3 + eps)
    # recency_risk: long since last purchase AND low recent frequency = lapse signal
    out["recency_risk"]       = (out["recency_days"]
                                 * np.clip(1.0 - out["purchase_count_30d"] / 10.0, 0.0, 1.0))
    # engagement_score: total number of digital touchpoints in 30d
    out["engagement_score"]   = (out.get("app_open_30d",   pd.Series(0.0, index=df.index)).fillna(0)
                                + out.get("email_open_30d", pd.Series(0.0, index=df.index)).fillna(0)
                                + out.get("push_open_30d",  pd.Series(0.0, index=df.index)).fillna(0))
    # spend_decline_flag: binary flag — are they at less than 50% of their 90d pace?
    out["spend_decline_flag"] = (out["spend_velocity"] < 0.5).astype(np.float32)

    # Cap velocities at a reasonable upper bound (outlier protection)
    for vc in ["spend_velocity", "freq_velocity", "app_velocity"]:
        out[vc] = out[vc].clip(0, 10)

    return out

# STEP 8.2b — LAG / DELTA FEATURES (month-over-month change)
#
#   The target is a TRANSITION, but almost every feature above describes a
#   level: how much a member spent, how often they visited, how recently. The
#   six velocity features are ratios computed inside a single snapshot (30d vs
#   90d/3), so they never compare this month's snapshot to last month's.
#
#   These features close that gap by differencing consecutive monthly
#   snapshots, plus two segment-history features. seg_prev in particular lifts
#   the model from a first-order to a second-order Markov view: knowing a
#   member just arrived in a segment is very different from knowing they have
#   sat in it for months.
#
#   Ablate with --no-lag-features to measure the contribution.

LAG_BASE_COLS = resolve_lag_base_cols(BASE_FEAT_COLS)
DELTA_COLS    = delta_col_names(LAG_BASE_COLS)
LAG_COLS      = lag_col_names(LAG_BASE_COLS)


ALL_FEAT_COLS = BASE_FEAT_COLS + VELOCITY_COLS   # 40 + 6 = 46 features
if USE_LAG:
    ALL_FEAT_COLS = ALL_FEAT_COLS + LAG_COLS
X_COLS        = ALL_FEAT_COLS + ["seg_curr", "y_curr", "month_num"]

if USE_LAG:
    print(f"  Lag features enabled: {len(DELTA_COLS)} deltas + {len(SEG_HIST_COLS)} segment-history")
else:
    print("  Lag features DISABLED (--no-lag-features) — ablation run")

print(f"  Velocity features added: {VELOCITY_COLS}")
print(f"  Total X columns: {len(X_COLS)}  "
      f"({len(BASE_FEAT_COLS)} behavioral + {len(VELOCITY_COLS)} velocity + 3 context)")
print(f"  Memory estimate: {4_000_000 * len(X_COLS) * 4 / 1e9:.2f} GiB for X_train (float32)")

# STEP 8.3 — BUILD TRANSITION PAIRS (walk-forward)
print(f"\n[{elapsed()}] STEP 8.3 — Building month-over-month transition pairs...")
print("  Walk-forward split:")
print("    Train:    FebMar … SepOct  (8 pairs)")
print("    Validate: OctNov             (1 pair)")
print("    Test:     NovDec             (1 pair   report F1 on this)")

SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")

TRAIN_MONTHS = [2, 3, 4, 5, 6, 7, 8, 9]
VAL_MONTHS   = [10]
TEST_MONTHS  = [11]

train_rows, val_rows, test_rows = [], [], []

# state_name -> state_id, harvested from the Phase 7 outputs actually used for
# training. This exact map is saved in the model bundle so pipeline.py encodes
# y_curr identically at inference. Deriving it independently on either side
# (e.g. alphabetically) silently feeds the model a scrambled feature.
STATE_MAP: dict[str, int] = {}

for i, t_date in enumerate(SNAPSHOT_DATES[:-1]):
    t1_date  = SNAPSHOT_DATES[i + 1]
    t_str    = t_date.strftime("%Y_%m_%d")
    t1_str   = t1_date.strftime("%Y_%m_%d")
    t_month  = t_date.month

    # Skip January — cold-start (almost all are New & Uncertain)
    if t_month == 1:
        continue

    tm1_date = SNAPSHOT_DATES[i - 1] if i > 0 else None
    tm1_str  = tm1_date.strftime("%Y_%m_%d") if tm1_date is not None else None

    feat_path = FEATURES_DIR / f"features_{t_str}.parquet"
    seg_t     = SEGMENTS_DIR  / f"behavioral_segments_{t_str}.parquet"
    seg_t1    = SEGMENTS_DIR  / f"behavioral_segments_{t1_str}.parquet"
    state_t   = STATES_DIR    / f"lifecycle_states_{t_str}.parquet"

    required = [feat_path, seg_t, seg_t1, state_t]
    if USE_LAG:
        if tm1_str is None:
            print(f"  {t_str}: no prior month for lag features — skipping pair")
            continue
        feat_tm1 = FEATURES_DIR / f"features_{tm1_str}.parquet"
        seg_tm1  = SEGMENTS_DIR / f"behavioral_segments_{tm1_str}.parquet"
        required += [feat_tm1, seg_tm1]

    if not all(p.exists() for p in required):
        print(f"  {t_str}: missing file(s) — skipping pair")
        continue

    t0 = time.time()

    # Load features (all numeric cols) + add velocity features
    feat = (pd.read_parquet(feat_path, engine="pyarrow")
              .reset_index(drop=True))
    feat = add_velocity_features(feat)

    if USE_LAG:
        # Prior-month features, differenced against the current month.
        prev_feat = pd.read_parquet(
            feat_tm1, engine="pyarrow", columns=["member_id"] + LAG_BASE_COLS
        )
        feat = add_delta_features(feat, prev_feat, LAG_BASE_COLS)

        # Segment at T-1 — seg_prev/seg_changed are finalised after seg_curr
        # is merged in below.
        prev_seg = (pd.read_parquet(seg_tm1, engine="pyarrow")[["member_id", "segment_id"]]
                      .rename(columns={"segment_id": "seg_prev_raw"}))
        feat = feat.merge(prev_seg, on="member_id", how="left")

    keep = [c for c in ALL_FEAT_COLS if c not in SEG_HIST_COLS]
    if USE_LAG:
        keep = keep + ["seg_prev_raw"]
    feat = feat[["member_id"] + keep].fillna({c: 0 for c in keep if c != "seg_prev_raw"})

    # seg_curr (segment at T)
    sc = pd.read_parquet(seg_t, engine="pyarrow")[["member_id", "segment_id"]].rename(
             columns={"segment_id": "seg_curr"})

    # seg_next (target: segment at T+1)
    sn = pd.read_parquet(seg_t1, engine="pyarrow")[["member_id", "segment_id"]].rename(
             columns={"segment_id": "seg_next"})

    # y_curr (current lifecycle state from Phase 7)
    st_raw = pd.read_parquet(state_t, engine="pyarrow")[
        ["member_id", "state_id", "state_name"]]
    for nm, sid in st_raw[["state_name", "state_id"]].drop_duplicates().itertuples(index=False):
        prev = STATE_MAP.setdefault(nm, int(sid))
        if prev != int(sid):
            raise ValueError(
                f"Inconsistent state encoding for '{nm}': {prev} in an earlier "
                f"month vs {sid} in {t_str}. Phase 7 must emit stable state_ids."
            )
    st = st_raw[["member_id", "state_id"]].rename(columns={"state_id": "y_curr"})

    # Inner join on member_id
    pair = (feat.merge(sc, on="member_id")
                .merge(sn, on="member_id")
                .merge(st, on="member_id"))

    if USE_LAG:
        pair = add_segment_history(pair, pair["seg_prev_raw"].values)
        pair = pair.drop(columns=["seg_prev_raw"])

    pair["month_num"] = t_month
    pair["pair_str"]  = f"{t_date.strftime('%b')}{t1_date.strftime('%b')}"

    if t_month in TRAIN_MONTHS:
        train_rows.append(pair); split = "TRAIN"
    elif t_month in VAL_MONTHS:
        val_rows.append(pair);   split = "VAL  "
    else:
        test_rows.append(pair);  split = "TEST "

    t_e = time.time() - t0
    print(f"  [{split}] {t_date.strftime('%b')}{t1_date.strftime('%b')}: "
          f"{len(pair):,} pairs | {t_e:.1f}s")

df_train = pd.concat(train_rows, ignore_index=True)
df_val   = pd.concat(val_rows,   ignore_index=True)
df_test  = pd.concat(test_rows,  ignore_index=True)
print(f"\n  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# STEP 8.4 — BUILD X/y MATRICES
print(f"\n[{elapsed()}] STEP 8.4 — Building X/y matrices...")

X_tr = df_train[X_COLS].values.astype(np.float32)
y_tr = df_train["seg_next"].values.astype(int)

X_va = df_val[X_COLS].values.astype(np.float32)
y_va = df_val["seg_next"].values.astype(int)

X_te = df_test[X_COLS].values.astype(np.float32)
y_te = df_test["seg_next"].values.astype(int)

print(f"  X_train: {X_tr.shape}  |  Classes: {sorted(set(y_tr))}")
print(f"  X_val:   {X_va.shape}")
print(f"  X_test:  {X_te.shape}")

print("\n  Class distribution in TEST (NovDec):")
for cid in sorted(set(y_te)):
    cnt  = int((y_te == cid).sum())
    pct  = cnt / len(y_te) * 100
    flag = "  ️  < 500 examples" if cnt < 500 else ""
    print(f"    {cid} {SEG_ID_TO_NAME.get(cid,'?'):<25}: {cnt:>7,}  ({pct:.1f}%){flag}")

# STEP 8.5 — SAMPLE WEIGHTS
#   Inverse-frequency ('balanced') weights handle the class imbalance.
#
#   A previous version applied an extra 4x boost to a "Lapse Risk" segment.
#   That was a no-op: Lapse Risk is a lifecycle STATE (Layer 2), not a segment
#   (Layer 1). The lookup silently resolved to None and the boost never applied.
#   Weak classes are now identified from VALIDATION per-class F1 in STEP 8.6b
#   and boosted there — never from the test split.
print(f"\n[{elapsed()}] STEP 8.5 — Computing sample weights...")

WEAK_F1_THRESHOLD = 0.75   # validation F1 below this earns an extra boost
WEAK_CLASS_BOOST  = 2.0    # multiplier applied on top of 'balanced'


def build_sample_weights(y: np.ndarray, boost_classes=()) -> np.ndarray:
    """Inverse-frequency weights, optionally boosted for named class ids."""
    w = compute_sample_weight("balanced", y)
    if boost_classes:
        w = w * np.where(np.isin(y, list(boost_classes)), WEAK_CLASS_BOOST, 1.0)
    return w


base_weights = build_sample_weights(y_tr)
print("  Balanced (inverse-frequency) weights computed")
print(f"  Weight range: [{base_weights.min():.2f}, {base_weights.max():.2f}]")

# STEP 8.6 — XGBOOST TRAINING (early stopping on val mlogloss)
print(f"\n[{elapsed()}] STEP 8.6 — Training XGBoost...")
print(f"  Features: {len(X_COLS)} | Samples: {len(X_tr):,}")
print("  Early stopping: 30 rounds on val mlogloss")


def train_model(sample_weights: np.ndarray, label: str):
    """Fit one XGBoost model. Early stopping is evaluated on VALIDATION only."""
    m = xgb.XGBClassifier(
        n_estimators          = 1000,
        max_depth             = 7,          # deeper splits help the minority segments
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        colsample_bynode      = 0.8,        # additional column stochasticity
        min_child_weight      = 1,          # allow small-leaf splits for rare classes
        gamma                 = 0.05,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        max_delta_step        = 1,          # improves convergence for imbalanced classes
        objective             = "multi:softprob",   # full distribution — prob_S01..prob_S05
        num_class             = N_SEGMENTS,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 30,
        n_jobs                = -1,
        random_state          = 42,
        verbosity             = 0,
    )
    print(f"\n  [{label}] fitting...")
    m.fit(X_tr, y_tr, sample_weight=sample_weights,
          eval_set=[(X_va, y_va)], verbose=50)
    print(f"  [{label}] best_iteration={m.best_iteration}  val_mlogloss={m.best_score:.4f}")
    return m


# ── Pass 1: balanced weights only ────────────────────────────────────────────
model_p1 = train_model(base_weights, "pass 1 / balanced")
val_f1_p1 = f1_score(y_va, model_p1.predict(X_va), average="macro", zero_division=0)
print(f"  [pass 1 / balanced] val macro F1 = {val_f1_p1:.4f}")

# STEP 8.6b — IDENTIFY WEAK CLASSES ON VALIDATION, RETRAIN WITH BOOST
#   Which classes are hard is decided on the validation split only. The test
#   split is not consulted here or anywhere before the final report.
print(f"\n[{elapsed()}] STEP 8.6b — Weak-class boost (selected on validation)...")

per_class_val_p1 = f1_score(y_va, model_p1.predict(X_va), average=None, zero_division=0)
weak_classes = [c for c in range(N_SEGMENTS)
                if c < len(per_class_val_p1) and per_class_val_p1[c] < WEAK_F1_THRESHOLD]

for c in range(N_SEGMENTS):
    if c < len(per_class_val_p1):
        mark = "  <-- boosting" if c in weak_classes else ""
        print(f"    {c} {SEG_ID_TO_NAME.get(c,'?'):<25}: val F1={per_class_val_p1[c]:.3f}{mark}")

if weak_classes:
    print(f"  Retraining with {WEAK_CLASS_BOOST}x boost on {len(weak_classes)} weak class(es)")
    boosted_weights = build_sample_weights(y_tr, boost_classes=weak_classes)
    model_p2  = train_model(boosted_weights, "pass 2 / boosted")
    val_f1_p2 = f1_score(y_va, model_p2.predict(X_va), average="macro", zero_division=0)
    print(f"  [pass 2 / boosted] val macro F1 = {val_f1_p2:.4f}")

    # Keep whichever scores better ON VALIDATION.
    if val_f1_p2 > val_f1_p1:
        model, sample_weights, weighting_used = model_p2, boosted_weights, "balanced + weak-class boost"
    else:
        model, sample_weights, weighting_used = model_p1, base_weights, "balanced"
else:
    print(f"  No class below val F1 {WEAK_F1_THRESHOLD} — keeping balanced weights")
    model, sample_weights, weighting_used = model_p1, base_weights, "balanced"
    val_f1_p2 = float("nan")

print(f"\n  Weighting selected (on validation): {weighting_used}")
best_round = model.best_iteration
print(f"  Best iteration: {best_round}  |  Val mlogloss: {model.best_score:.4f}")

# STEP 8.7 — THRESHOLD CALIBRATION ON VALIDATION SET
#   For each class, finds the probability threshold that maximises binary F1
#   on the validation set. Applies scaled thresholds to test predictions.
#   This is the key fix for rare classes like Lapse Risk.
print(f"\n[{elapsed()}] STEP 8.7 — Threshold calibration on validation set...")

proba_va  = model.predict_proba(X_va)
proba_te  = model.predict_proba(X_te)

# Default argmax predictions
y_pred_va_default = np.argmax(proba_va, axis=1)
y_pred_te_default = np.argmax(proba_te, axis=1)

# Per-class threshold optimisation on validation
optimal_thresholds = np.full(N_SEGMENTS, 0.5)
print("\n  Per-class threshold sweep on validation (OctNov):")
for cls_id in range(N_SEGMENTS):
    y_bin    = (y_va == cls_id).astype(int)
    best_t, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.01, 0.95, 100):
        preds = (proba_va[:, cls_id] >= thresh).astype(int)
        f     = f1_score(y_bin, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, thresh
    optimal_thresholds[cls_id] = best_t
    val_f1_default = f1_score(y_bin, (y_pred_va_default == cls_id).astype(int), zero_division=0)
    print(f"    {cls_id} {SEG_ID_TO_NAME.get(cls_id,'?'):<25}: "
          f"threshold={best_t:.2f}  val_F1(default)={val_f1_default:.3f}  val_F1(tuned)={best_f1:.3f}")

# Apply threshold-calibrated prediction:
# Divide each class probability by its threshold  class with highest ratio wins
calibrated_proba  = proba_te / (optimal_thresholds + 1e-9)
y_pred_te_thresh  = np.argmax(calibrated_proba, axis=1)

calibrated_va     = proba_va / (optimal_thresholds + 1e-9)
y_pred_va_thresh  = np.argmax(calibrated_va, axis=1)

# STEP 8.8 — EVALUATION (the graded number)
print(f"\n[{elapsed()}] STEP 8.8 — Evaluation on TEST set (NovDec)...")

macro_f1_val_default   = f1_score(y_va, y_pred_va_default, average="macro", zero_division=0)
macro_f1_val_thresh    = f1_score(y_va, y_pred_va_thresh,  average="macro", zero_division=0)
macro_f1_test_default  = f1_score(y_te, y_pred_te_default, average="macro", zero_division=0)
macro_f1_test_thresh   = f1_score(y_te, y_pred_te_thresh,  average="macro", zero_division=0)

# Decision rule is chosen on VALIDATION, then applied once to test.
#   Selecting argmax-vs-calibrated by test score would be test-set peeking and
#   would inflate the reported number. Whatever the validation-selected rule
#   scores on test is what gets reported — better or worse.
use_calibrated  = macro_f1_val_thresh > macro_f1_val_default
method_used     = "threshold-calibrated" if use_calibrated else "argmax"
best_test_preds = y_pred_te_thresh if use_calibrated else y_pred_te_default
best_macro_f1   = macro_f1_test_thresh if use_calibrated else macro_f1_test_default

print(f"\n  {'='*65}")
print(f"  MACRO F1 — VAL (OctNov):  default={macro_f1_val_default:.4f}  calibrated={macro_f1_val_thresh:.4f}")
print(f"  MACRO F1 — TEST (NovDec): default={macro_f1_test_default:.4f}  calibrated={macro_f1_test_thresh:.4f}")
print("  ─────────────────────────────────────────────────────────────────")
print(f"  Decision rule selected on VALIDATION: {method_used}")
print(f"  REPORTED MACRO F1 (test, no test-set selection): {best_macro_f1:.4f}")
print(f"  {'='*65}")

print(f"\n  Per-class F1 — TEST (NovDec) — {method_used}:")
print(classification_report(
    y_te, best_test_preds,
    target_names=SEGMENT_NAMES,
    zero_division=0
))

# Flag low-F1 segments
per_class_f1 = f1_score(y_te, best_test_preds, average=None, zero_division=0)
low_f1 = [(SEGMENT_NAMES[i], float(per_class_f1[i]))
          for i in range(len(SEGMENT_NAMES)) if i < len(per_class_f1) and per_class_f1[i] < 0.50]
if low_f1:
    print("  ️  Segments still below 0.50 F1:")
    for name, f1 in low_f1:
        print(f"    {name}: F1={f1:.3f}")
else:
    print("   All segments have F1 ≥ 0.50")

# STEP 8.8b — BASELINES + BOOTSTRAP CONFIDENCE INTERVAL
#   A macro F1 in isolation says nothing. Two reference points:
#     majority     — always predict the most common segment in TRAIN
#     persistence  — predict that nobody moves (seg_next == seg_curr)
#   Persistence is the one that matters: seg_curr is the single strongest
#   feature, so beating it is the real evidence the model learned transitions.
print(f"\n[{elapsed()}] STEP 8.8b — Baselines + bootstrap CI...")

majority_class   = int(pd.Series(y_tr).value_counts().idxmax())
y_pred_majority  = np.full_like(y_te, majority_class)
y_pred_persist   = df_test["seg_curr"].values.astype(int)

f1_majority    = f1_score(y_te, y_pred_majority, average="macro", zero_division=0)
f1_persistence = f1_score(y_te, y_pred_persist,  average="macro", zero_division=0)

# Bootstrap the test macro F1 (percentile interval, fixed seed).
N_BOOT = 1000
rng    = np.random.default_rng(42)
n_te   = len(y_te)
boot   = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx     = rng.integers(0, n_te, n_te)
    boot[b] = f1_score(y_te[idx], best_test_preds[idx], average="macro", zero_division=0)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

lift_persist  = best_macro_f1 - f1_persistence
lift_majority = best_macro_f1 - f1_majority

print(f"\n  {'Model / baseline':<34} {'Macro F1':>10}  {'Lift':>8}")
print(f"  {'-'*56}")
print(f"  {'majority-class':<34} {f1_majority:>10.4f}  {'—':>8}")
print(f"  {'persistence (seg_next = seg_curr)':<34} {f1_persistence:>10.4f}  {'—':>8}")
print(f"  {'TBIE XGBoost':<34} {best_macro_f1:>10.4f}  {lift_persist:>+8.4f}")
print(f"  {'-'*56}")
print(f"  Test macro F1: {best_macro_f1:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  "
      f"({N_BOOT} bootstrap resamples)")
print(f"  Lift over persistence: {lift_persist:+.4f}   over majority: {lift_majority:+.4f}")
if lift_persist <= 0:
    print("  WARNING: model does not beat the persistence baseline.")

# STEP 8.9 — FEATURE IMPORTANCE
print(f"\n[{elapsed()}] STEP 8.9 — Top 20 feature importances...")

feat_imp = pd.Series(model.feature_importances_, index=X_COLS).sort_values(ascending=False)
print(f"\n  {'Rank':<5} {'Feature':<45} {'Importance':>10}")
print(f"  {'-'*65}")
for rank, (feat, imp) in enumerate(feat_imp.head(20).items(), 1):
    bar  = "█" * int(imp * 300)
    flag = "  NEW" if feat in VELOCITY_COLS else ""
    print(f"  {rank:>3}.  {feat:<45}  {imp:.4f}  {bar}{flag}")

# STEP 8.10 — SAVE OUTPUTS
print(f"\n[{elapsed()}] STEP 8.10 — Saving model + predictions...")

# Bundle schema is the contract with pipeline.py. Key names must match what
# pipeline.py reads, or inference dies with a KeyError on a fresh machine.
CLUSTER_TO_SID = {cid: f"S{cid + 1:02d}" for cid in range(N_SEGMENTS)}
SID_TO_NAME    = {CLUSTER_TO_SID[cid]: SEG_ID_TO_NAME[cid] for cid in range(N_SEGMENTS)}

bundle = {
    "model":              model,
    "feature_cols":       X_COLS,          # pipeline.py reads this name
    "velocity_cols":      VELOCITY_COLS,
    "cluster_to_sid":     CLUSTER_TO_SID,
    "sid_to_name":        SID_TO_NAME,
    "state_map":          STATE_MAP,       # exact encoding used to build y_curr
    "uses_lag_features":  USE_LAG,
    "lag_base_cols":      LAG_BASE_COLS if USE_LAG else [],
    "n_classes":          N_SEGMENTS,
    "random_seed":        42,
    "segment_names":      SEGMENT_NAMES,
    "optimal_thresholds": optimal_thresholds.tolist(),
    "decision_rule":      method_used,     # selected on validation
    "weighting":          weighting_used,  # selected on validation
    "macro_f1_val":       max(macro_f1_val_default, macro_f1_val_thresh),
    "macro_f1_test":      best_macro_f1,
}

# Fail loudly here rather than at inference time on someone else's machine.
REQUIRED_BY_PIPELINE = ["model", "feature_cols", "velocity_cols",
                        "cluster_to_sid", "sid_to_name", "state_map"]
missing_keys = [k for k in REQUIRED_BY_PIPELINE if k not in bundle]
assert not missing_keys, f"Bundle is missing keys pipeline.py requires: {missing_keys}"
assert len(STATE_MAP) == 10, f"Expected 10 lifecycle states, harvested {len(STATE_MAP)}"

MODEL_OUT = MODELS_DIR / f"segment_transition_model{RUN_TAG}.pkl"
joblib.dump(bundle, MODEL_OUT)
print(f"  {MODEL_OUT.name} saved")
print(f"  state_map ({len(STATE_MAP)} states): "
      f"{dict(sorted(STATE_MAP.items(), key=lambda x: x[1]))}")

# Test predictions parquet
test_out = df_test[["member_id", "pair_str", "seg_curr", "seg_next", "month_num"]].copy()
test_out["seg_pred"]      = best_test_preds
test_out["seg_curr_name"] = test_out["seg_curr"].map(SEG_ID_TO_NAME)
test_out["seg_next_name"] = test_out["seg_next"].map(SEG_ID_TO_NAME)
test_out["seg_pred_name"] = pd.Series(best_test_preds).map(SEG_ID_TO_NAME).values
test_out["correct"]       = (test_out["seg_next"] == test_out["seg_pred"]).astype(int)
test_out.to_parquet(OUTPUTS_DIR / f"phase8_predictions_nov_dec{RUN_TAG}.parquet", engine="pyarrow", index=False)

# Classification report text
report_str = classification_report(y_te, best_test_preds,
                                   target_names=SEGMENT_NAMES, zero_division=0)
(OUTPUTS_DIR / f"phase8_classification_report{RUN_TAG}.txt").write_text(
    f"PHASE 8 — CLASSIFICATION REPORT (NovDec Test)\n"
    f"Walk-forward: Train=FebOct | Val=OctNov | Test=NovDec\n"
    f"Features: {len(X_COLS)} ({len(BASE_FEAT_COLS)} raw + {len(VELOCITY_COLS)} velocity + 3 context)\n"
    f"Weighting (selected on val):    {weighting_used}\n"
    f"Decision rule (selected on val): {method_used}\n\n"
    f"Macro F1 (Val  default):    {macro_f1_val_default:.4f}\n"
    f"Macro F1 (Val  calibrated): {macro_f1_val_thresh:.4f}\n"
    f"Macro F1 (Test default):    {macro_f1_test_default:.4f}\n"
    f"Macro F1 (Test calibrated): {macro_f1_test_thresh:.4f}\n"
    f"REPORTED Macro F1 (Test):   {best_macro_f1:.4f}  "
    f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]\n\n"
    f"Baselines (test):\n"
    f"  majority-class:                {f1_majority:.4f}\n"
    f"  persistence (next = current):  {f1_persistence:.4f}\n"
    f"  lift over persistence:         {lift_persist:+.4f}\n\n"
    + report_str,
    encoding="utf-8"
)
print(f"  phase8_classification_report{RUN_TAG}.txt saved")

# STEP 8.11 — SUMMARY
summary_lines = [
    "PHASE 8 SUMMARY — SEGMENT TRANSITION PREDICTION",
    "=" * 60,
    f"Model:          XGBoost (n_classes={N_SEGMENTS}, best_iter={best_round})",
    f"Features:       {len(X_COLS)}  ({len(BASE_FEAT_COLS)} raw + {len(VELOCITY_COLS)} velocity + 3 context)",
    f"Train pairs:    {len(df_train):,} (8 month pairs, FebOct)",
    f"Val pairs:      {len(df_val):,} (OctNov)",
    f"Test pairs:     {len(df_test):,} (NovDec)",
    "",
    "Selected on VALIDATION (test never consulted for any choice):",
    f"  weighting:     {weighting_used}",
    f"  decision rule: {method_used}",
    "",
    f"MACRO F1 (Val  — default):    {macro_f1_val_default:.4f}",
    f"MACRO F1 (Val  — calibrated): {macro_f1_val_thresh:.4f}",
    f"MACRO F1 (Test — default):    {macro_f1_test_default:.4f}",
    f"MACRO F1 (Test — calibrated): {macro_f1_test_thresh:.4f}",
    f"REPORTED (Test):              {best_macro_f1:.4f}  "
    f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
    "",
    "Baselines (test):",
    f"  majority-class:               {f1_majority:.4f}",
    f"  persistence (next = current): {f1_persistence:.4f}",
    f"  LIFT over persistence:        {lift_persist:+.4f}",
    "",
    "Per-class F1 (Test):",
]
for i, name in enumerate(SEGMENT_NAMES):
    if i < len(per_class_f1):
        flag = " ️" if per_class_f1[i] < 0.50 else " "
        thr  = optimal_thresholds[i]
        summary_lines.append(f"  {name:<25}: F1={per_class_f1[i]:.3f}{flag}  threshold={thr:.2f}")

summary_lines += [
    "",
    "Files written:",
    f"  models/segment_transition_model{RUN_TAG}.pkl",
    f"  outputs/phase8_predictions_nov_dec{RUN_TAG}.parquet",
    f"  outputs/phase8_classification_report{RUN_TAG}.txt",
    f"Total elapsed: {elapsed()}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
(VAL_DIR / f"phase8_summary{RUN_TAG}.txt").write_text(summary_text, encoding="utf-8")

print(f"\n{'=' * 70}")
print(f"PHASE 8 COMPLETE — Reported Macro F1 (Test): {best_macro_f1:.4f}")
print(f"{'=' * 70}")
