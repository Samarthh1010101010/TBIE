"""
src/13_tune_hyperparams.py
══════════════════════════════════════════════════════════════════════════════
PHASE 13 — Hyperparameter Search
TBIE Pipeline | Kobie × PES University Hackathon

The Phase 8 hyperparameters were chosen by hand and never searched. This runs a
proper Bayesian (TPE) search over them.

Method
──────
  - Objective  : macro F1 on the VALIDATION split (Oct→Nov)
  - Search     : Optuna TPE, fixed seed, median pruner
  - Speed      : trials fit on a stratified subsample of the 4M training rows;
                 the winning configuration is then refitted on the full set
  - Test split : never read, at any point, by this script

Selecting on validation is legitimate — that is what a validation split is for.
The honest caveat is that many trials against one validation window invites
overfitting to that window, so the trial count is kept modest, the search space
is narrow around sensible values, and the final validation-vs-test gap is
reported so any overfitting is visible rather than hidden.

Run from TBIE_CODE root:
    python src/13_tune_hyperparams.py --trials 25
    python src/13_tune_hyperparams.py --trials 40 --subsample 2000000
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.pairs import build_pairs, to_model_matrix  # noqa: E402

p = argparse.ArgumentParser(description="TBIE Phase 13 — hyperparameter search")
p.add_argument("--model", type=str,
               default=str(ROOT / "models" / "segment_transition_model.pkl"),
               help="Baseline bundle — defines the feature set and the score to beat.")
p.add_argument("--trials", type=int, default=25)
p.add_argument("--subsample", type=int, default=1_500_000,
               help="Training rows per trial. The winner is refitted on all rows.")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-refit", action="store_true",
               help="Search only; do not refit the winner on the full training set.")
p.add_argument("--refit-from", type=str, default="",
               help="Skip the search and refit the best params from a previous "
                    "hyperparameter_search.json on the full training set. Use this "
                    "when a search has already run and only the fair, full-data "
                    "comparison against the baseline is still needed.")
args = p.parse_args()

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()
def elapsed() -> str:
    return f"{time.time() - T0:.1f}s"


print("=" * 70)
print("TBIE — PHASE 13: HYPERPARAMETER SEARCH")
print("=" * 70)

bundle         = joblib.load(args.model)
X_COLS         = bundle["feature_cols"]
VELOCITY_COLS  = bundle["velocity_cols"]
LAG_BASE_COLS  = bundle.get("lag_base_cols", [])
USE_LAG        = bool(bundle.get("uses_lag_features", bool(LAG_BASE_COLS)))
CLUSTER_TO_SID = bundle["cluster_to_sid"]
SID_TO_NAME    = bundle["sid_to_name"]
STATE_MAP      = bundle["state_map"]
N_CLASSES      = bundle.get("n_classes", len(CLUSTER_TO_SID))
SEGMENT_NAMES  = [SID_TO_NAME[CLUSTER_TO_SID[c]] for c in range(N_CLASSES)]
BASELINE_VAL   = bundle.get("macro_f1_val", float("nan"))

seg_model      = joblib.load(ROOT / "segments" / "segment_model.pkl")
BASE_FEAT_COLS = seg_model["behavioral_feature_cols"]

print(f"  baseline bundle   : {Path(args.model).name}")
print(f"  features          : {len(X_COLS)}  (lag: {'on' if USE_LAG else 'off'})")
print(f"  baseline val F1   : {BASELINE_VAL:.4f}  <- the score to beat")
print(f"  trials            : {args.trials}")
print(f"  rows per trial    : {args.subsample:,}")

SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
TRAIN_MONTHS   = [2, 3, 4, 5, 6, 7, 8, 9]

print(f"\n[{elapsed()}] Building training pairs...")
df_train = build_pairs(SNAPSHOT_DATES, TRAIN_MONTHS, ROOT, BASE_FEAT_COLS,
                       VELOCITY_COLS, USE_LAG, LAG_BASE_COLS, verbose=False)
print(f"[{elapsed()}] Building validation pairs...")
df_val = build_pairs(SNAPSHOT_DATES, [10], ROOT, BASE_FEAT_COLS,
                     VELOCITY_COLS, USE_LAG, LAG_BASE_COLS, verbose=False)

X_tr_full = to_model_matrix(df_train, X_COLS)
y_tr_full = df_train["seg_next"].values.astype(int)
X_va      = to_model_matrix(df_val, X_COLS)
y_va      = df_val["seg_next"].values.astype(int)
print(f"  train {X_tr_full.shape}   val {X_va.shape} | {elapsed()}")

# Stratified subsample for the search itself.
rng = np.random.default_rng(args.seed)
if args.subsample < len(X_tr_full):
    take = []
    for c in range(N_CLASSES):
        idx_c = np.where(y_tr_full == c)[0]
        n_c = max(1, int(round(len(idx_c) * args.subsample / len(y_tr_full))))
        take.append(rng.choice(idx_c, min(n_c, len(idx_c)), replace=False))
    sub = np.sort(np.concatenate(take))
    X_tr, y_tr = X_tr_full[sub], y_tr_full[sub]
else:
    X_tr, y_tr = X_tr_full, y_tr_full
print(f"  search subsample: {X_tr.shape[0]:,} rows "
      f"({X_tr.shape[0] / len(y_tr_full):.0%} of train)")

W_TR = compute_sample_weight("balanced", y_tr)
W_FULL = compute_sample_weight("balanced", y_tr_full)


def fit_eval(params: dict, X, y, w, n_estimators: int = 1200):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=args.seed,
        verbosity=0,
        **params,
    )
    model.fit(X, y, sample_weight=w, eval_set=[(X_va, y_va)], verbose=False)
    pred = model.predict(X_va)
    return model, f1_score(y_va, pred, average="macro", zero_division=0)


def objective(trial: optuna.Trial) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 5, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "max_delta_step":   trial.suggest_int("max_delta_step", 0, 5),
    }
    _, score = fit_eval(params, X_tr, y_tr, W_TR)
    trial.set_user_attr("val_macro_f1", score)
    return score


# ── Optional: skip the search and go straight to the fair comparison ─────────
if args.refit_from:
    import pathlib

    prior = json.loads(pathlib.Path(args.refit_from).read_text(encoding="utf-8"))
    BEST_PARAMS = prior["best_params"]
    print(f"\n[{elapsed()}] --refit-from: reusing best params from "
          f"{pathlib.Path(args.refit_from).name}")
    print(f"  best trial val F1 (on {prior['search_subsample']:,} rows): "
          f"{prior['best_val_macro_f1']:.4f}")
    for k, v in BEST_PARAMS.items():
        print(f"    {k:<20} {v}")

    print(f"\n[{elapsed()}] Refitting on the FULL training set "
          f"({len(y_tr_full):,} rows)...")
    best_model, full_val_f1 = fit_eval(BEST_PARAMS, X_tr_full, y_tr_full, W_FULL)

    print(f"\n{'=' * 70}")
    print("FAIR COMPARISON — both fitted on the full training set")
    print(f"{'=' * 70}")
    print(f"  baseline (hand-picked)  val macro F1: {BASELINE_VAL:.4f}")
    print(f"  tuned                   val macro F1: {full_val_f1:.4f}  "
          f"({full_val_f1 - BASELINE_VAL:+.4f})")

    prior["full_data_refit_val_macro_f1"] = float(full_val_f1)
    prior["beats_baseline"] = bool(full_val_f1 > BASELINE_VAL)
    pathlib.Path(args.refit_from).write_text(json.dumps(prior, indent=2), encoding="utf-8")

    if full_val_f1 <= BASELINE_VAL:
        print("\n  The tuned configuration does NOT beat the hand-picked one on")
        print("  validation, even with the subsample handicap removed. Keeping the")
        print("  existing model. A search that finds nothing is a real result: it")
        print("  means the original hyperparameters were already near-optimal.")
        sys.exit(0)

    tuned_path = MODELS_DIR / "segment_transition_model_tuned.pkl"
    joblib.dump({
        "model":             best_model,
        "feature_cols":      X_COLS,
        "velocity_cols":     VELOCITY_COLS,
        "cluster_to_sid":    CLUSTER_TO_SID,
        "sid_to_name":       SID_TO_NAME,
        "state_map":         STATE_MAP,
        "uses_lag_features": USE_LAG,
        "lag_base_cols":     LAG_BASE_COLS,
        "n_classes":         N_CLASSES,
        "random_seed":       args.seed,
        "segment_names":     SEGMENT_NAMES,
        "macro_f1_val":      float(full_val_f1),
        "tuned_params":      BEST_PARAMS,
        "weighting":         "balanced",
    }, tuned_path)
    print(f"\n  {tuned_path.name} saved (val macro F1 {full_val_f1:.4f})")
    print("\n  NOT promoted automatically — the test split has not been read for it.")
    sys.exit(0)


print(f"\n[{elapsed()}] Running {args.trials} TPE trials (objective: val macro F1)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)


def _cb(study_, trial_):
    best = study_.best_value
    print(f"  trial {trial_.number:>3}/{args.trials}: "
          f"val F1={trial_.value:.4f}   best={best:.4f} | {elapsed()}")


study.optimize(objective, n_trials=args.trials, callbacks=[_cb],
               show_progress_bar=False)

print(f"\n{'=' * 70}")
print("SEARCH COMPLETE")
print(f"{'=' * 70}")
print(f"  baseline val macro F1 : {BASELINE_VAL:.4f}")
print(f"  best trial val macro F1: {study.best_value:.4f}  "
      f"({study.best_value - BASELINE_VAL:+.4f})")
print("\n  Best parameters:")
for k, v in study.best_params.items():
    print(f"    {k:<20} {v}")

report = {
    "baseline_val_macro_f1": float(BASELINE_VAL),
    "best_val_macro_f1":     float(study.best_value),
    "improvement":           float(study.best_value - BASELINE_VAL),
    "best_params":           study.best_params,
    "n_trials":              args.trials,
    "search_subsample":      int(X_tr.shape[0]),
    "seed":                  args.seed,
    "note": ("Selected on validation only; the test split was never read by "
             "this script. Trials share one validation window, so a small part "
             "of the gain may be validation overfitting — compare the final "
             "val/test gap against the baseline's."),
    "trials": [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.value is not None
    ],
}
(OUTPUTS_DIR / "hyperparameter_search.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print("\n  hyperparameter_search.json saved")

# NOTE: do NOT gate on study.best_value here.
#
# Trials fit on a subsample (default 500K of 4M rows); the baseline was fitted
# on all of them. Comparing those two numbers directly is not like for like and
# systematically under-rates the search — a configuration reaching 0.8078 on 12%
# of the data can clear the baseline on 100%. An earlier version of this script
# exited at exactly this point and reported "search did not beat the baseline"
# without ever running the refit that makes the comparison valid.
#
# The refit below is the fair comparison, and it is the only one that decides.
if study.best_value <= BASELINE_VAL:
    print(f"\n  Best trial ({study.best_value:.4f}) is below the baseline "
          f"({BASELINE_VAL:.4f}), but the trials used "
          f"{X_tr.shape[0]:,} rows against the baseline's {len(y_tr_full):,}.")
    print("  Refitting on the full training set before drawing any conclusion.")

if args.no_refit:
    print("\n  --no-refit set; stopping before the full-data refit. No conclusion")
    print("  can be drawn about the baseline from subsample scores alone.")
    sys.exit(0)

print(f"\n[{elapsed()}] Refitting the winner on the FULL training set "
      f"({len(y_tr_full):,} rows)...")
best_model, full_val_f1 = fit_eval(study.best_params, X_tr_full, y_tr_full, W_FULL)
print(f"  full-data val macro F1: {full_val_f1:.4f}")

if full_val_f1 <= BASELINE_VAL:
    print("  Refit did NOT beat the baseline on validation. Keeping the existing")
    print("  model — the subsample search result did not survive full training.")
    sys.exit(0)

tuned_path = MODELS_DIR / "segment_transition_model_tuned.pkl"
joblib.dump({
    "model":             best_model,
    "feature_cols":      X_COLS,
    "velocity_cols":     VELOCITY_COLS,
    "cluster_to_sid":    CLUSTER_TO_SID,
    "sid_to_name":       SID_TO_NAME,
    "state_map":         STATE_MAP,
    "uses_lag_features": USE_LAG,
    "lag_base_cols":     LAG_BASE_COLS,
    "n_classes":         N_CLASSES,
    "random_seed":       args.seed,
    "segment_names":     SEGMENT_NAMES,
    "macro_f1_val":      float(full_val_f1),
    "tuned_params":      study.best_params,
    "weighting":         "balanced",
}, tuned_path)

print(f"\n  {tuned_path.name} saved (val macro F1 {full_val_f1:.4f})")
print("\n  NOT promoted automatically. To evaluate it on test and promote:")
print("    python src/14_evaluate_model.py --model models/segment_transition_model_tuned.pkl")
print(f"\n{'=' * 70}")
print(f"PHASE 13 COMPLETE | {elapsed()}")
print(f"{'=' * 70}")
