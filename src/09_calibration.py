"""
src/09_calibration.py
══════════════════════════════════════════════════════════════════════════════
PHASE 9 — Probability Calibration
TBIE Pipeline | Kobie × PES University Hackathon

The pipeline emits `prob_S01`…`prob_S05` and a `prediction_confidence` for every
member, and downstream targeting is meant to act on them. Nothing so far has
checked whether those numbers mean what they claim: when the model says 70%, is
it right 70% of the time?

Gradient-boosted trees trained with heavy class reweighting are usually poorly
calibrated — reweighting deliberately distorts the predicted distribution away
from the observed base rates in exchange for better minority-class recall. So
this is worth measuring rather than assuming.

What this does
──────────────
  1. Measures calibration of the raw model on validation and test:
     Brier score, expected calibration error (ECE), reliability curves.
  2. Fits isotonic regression per class ON VALIDATION ONLY.
  3. Re-measures on test with the calibrator applied.
  4. Confirms macro F1 is not damaged (isotonic is monotonic, so the argmax
     ranking within a class is preserved, but cross-class argmax can move).
  5. Saves the calibrator so serving can emit trustworthy probabilities.

Calibration is a prerequisite for src/10_cost_thresholds.py: expected-value
arithmetic on miscalibrated probabilities produces confidently wrong budgets.

Run from TBIE_CODE root:
    python src/09_calibration.py
    python src/09_calibration.py --model models/segment_transition_model_noLag.pkl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.calibration import (  # noqa: E402
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
    renormalise,
)
from utils.pairs import build_pairs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser(description="TBIE Phase 9 — probability calibration")
p.add_argument("--model", type=str,
               default=str(ROOT / "models" / "segment_transition_model.pkl"))
p.add_argument("--n-bins", type=int, default=15,
               help="Bins for the reliability curve / ECE.")
p.add_argument("--tag", type=str, default="")
args = p.parse_args()

TAG = f"_{args.tag}" if args.tag else ""

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR  = ROOT / "models"
VAL_DIR     = ROOT / "validation"
for d in (OUTPUTS_DIR, MODELS_DIR, VAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

T0 = time.time()
def elapsed() -> str:
    return f"{time.time() - T0:.1f}s"


print("=" * 70)
print("TBIE — PHASE 9: PROBABILITY CALIBRATION")
print("=" * 70)

# ── Load the trained bundle ──────────────────────────────────────────────────
bundle = joblib.load(args.model)
model         = bundle["model"]
X_COLS        = bundle["feature_cols"]
VELOCITY_COLS = bundle["velocity_cols"]
LAG_BASE_COLS = bundle.get("lag_base_cols", [])
USE_LAG       = bool(bundle.get("uses_lag_features", bool(LAG_BASE_COLS)))
CLUSTER_TO_SID = bundle["cluster_to_sid"]
SID_TO_NAME    = bundle["sid_to_name"]
N_CLASSES      = bundle.get("n_classes", len(CLUSTER_TO_SID))
SEGMENT_NAMES  = [SID_TO_NAME[CLUSTER_TO_SID[c]] for c in range(N_CLASSES)]

seg_model      = joblib.load(ROOT / "segments" / "segment_model.pkl")
BASE_FEAT_COLS = seg_model["behavioral_feature_cols"]

print(f"  model      : {Path(args.model).name}")
print(f"  features   : {len(X_COLS)}  (lag features: {'on' if USE_LAG else 'off'})")
print(f"  classes    : {SEGMENT_NAMES}")

# ── Rebuild the same walk-forward splits Phase 8 used ────────────────────────
SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
VAL_MONTHS, TEST_MONTHS = [10], [11]

print(f"\n[{elapsed()}] Building validation pairs (Oct->Nov)...")
df_val = build_pairs(SNAPSHOT_DATES, VAL_MONTHS, ROOT, BASE_FEAT_COLS,
                     VELOCITY_COLS, USE_LAG, LAG_BASE_COLS)
print(f"[{elapsed()}] Building test pairs (Nov->Dec)...")
df_test = build_pairs(SNAPSHOT_DATES, TEST_MONTHS, ROOT, BASE_FEAT_COLS,
                      VELOCITY_COLS, USE_LAG, LAG_BASE_COLS)

missing_val = [c for c in X_COLS if c not in df_val.columns]
if missing_val:
    raise KeyError(f"Rebuilt pairs are missing trained feature(s): {missing_val[:10]}")

X_va = df_val[X_COLS].values.astype(np.float32)
y_va = df_val["seg_next"].values.astype(int)
X_te = df_test[X_COLS].values.astype(np.float32)
y_te = df_test["seg_next"].values.astype(int)
print(f"  val: {X_va.shape}   test: {X_te.shape} | {elapsed()}")

proba_va = model.predict_proba(X_va)
proba_te = model.predict_proba(X_te)


# ── Metrics ──────────────────────────────────────────────────────────────────
# expected_calibration_error / maximum_calibration_error / reliability_curve /
# renormalise live in src/utils/calibration.py so they are testable and so the
# drift monitor can reuse them. See tests/test_economics_calibration.py.


def summarise(proba, y_true, label: str) -> dict:
    print(f"\n  {label}")
    print(f"    {'class':<25} {'Brier':>9} {'ECE':>9}  {'mean p':>8} {'base rate':>10}")
    print(f"    {'-' * 66}")
    per_class, briers, eces = {}, [], []
    for c in range(N_CLASSES):
        yb = (y_true == c).astype(int)
        pc = proba[:, c]
        brier = brier_score_loss(yb, pc)
        ece   = expected_calibration_error(yb, pc, args.n_bins)
        briers.append(brier)
        eces.append(ece)
        per_class[SEGMENT_NAMES[c]] = {
            "brier": round(float(brier), 6),
            "ece":   round(float(ece), 6),
            # ECE is population-weighted, so a badly miscalibrated but sparsely
            # populated region can hide inside a good-looking average. MCE
            # surfaces the worst well-populated bin.
            "mce":   round(float(maximum_calibration_error(yb, pc, args.n_bins)), 6),
            "mean_predicted": round(float(pc.mean()), 6),
            "base_rate":      round(float(yb.mean()), 6),
            "reliability":    reliability_curve(yb, pc, args.n_bins),
        }
        print(f"    {SEGMENT_NAMES[c]:<25} {brier:>9.5f} {ece:>9.5f}  "
              f"{pc.mean():>8.4f} {yb.mean():>10.4f}")

    ll = log_loss(y_true, np.clip(proba, 1e-9, 1), labels=list(range(N_CLASSES)))
    macro_f1 = f1_score(y_true, proba.argmax(1), average="macro", zero_division=0)
    print(f"    {'-' * 66}")
    print(f"    mean Brier={np.mean(briers):.5f}  mean ECE={np.mean(eces):.5f}  "
          f"log-loss={ll:.5f}  macro F1={macro_f1:.4f}")
    return {
        "per_class":  per_class,
        "mean_brier": float(np.mean(briers)),
        "mean_ece":   float(np.mean(eces)),
        "log_loss":   float(ll),
        "macro_f1":   float(macro_f1),
    }


print(f"\n[{elapsed()}] Calibration BEFORE (raw model)")
before_val  = summarise(proba_va, y_va, "VALIDATION (Oct->Nov)")
before_test = summarise(proba_te, y_te, "TEST (Nov->Dec)")

# ── Fit isotonic per class on VALIDATION only ────────────────────────────────
print(f"\n[{elapsed()}] Fitting per-class isotonic regression on VALIDATION...")
calibrators = []
for c in range(N_CLASSES):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(proba_va[:, c], (y_va == c).astype(float))
    calibrators.append(iso)
    print(f"    {SEGMENT_NAMES[c]:<25} fitted on {len(y_va):,} validation rows")


def apply_calibration(proba: np.ndarray) -> np.ndarray:
    """
    Per-class isotonic, then renormalise so each row sums to 1.

    Calibrators are fitted independently per class, so their outputs do not sum
    to 1. renormalise() also handles the row-collapses-to-zero case, which
    would otherwise produce NaN and propagate into a targeting decision.
    """
    out = np.column_stack([calibrators[c].predict(proba[:, c]) for c in range(N_CLASSES)])
    return renormalise(out, fallback=proba)


cal_va = apply_calibration(proba_va)
cal_te = apply_calibration(proba_te)

print(f"\n[{elapsed()}] Calibration AFTER (isotonic, fitted on validation)")
after_val  = summarise(cal_va, y_va, "VALIDATION (Oct->Nov) — in-sample for the calibrator")
after_test = summarise(cal_te, y_te, "TEST (Nov->Dec) — the honest measurement")

# ── Verdict ──────────────────────────────────────────────────────────────────
d_brier = after_test["mean_brier"] - before_test["mean_brier"]
d_ece   = after_test["mean_ece"]   - before_test["mean_ece"]
d_f1    = after_test["macro_f1"]   - before_test["macro_f1"]

print(f"\n{'=' * 70}")
print("CALIBRATION RESULT — TEST (Nov->Dec)")
print(f"{'=' * 70}")
print(f"  {'metric':<22} {'before':>12} {'after':>12} {'change':>12}")
print(f"  {'-' * 60}")
print(f"  {'mean Brier':<22} {before_test['mean_brier']:>12.5f} "
      f"{after_test['mean_brier']:>12.5f} {d_brier:>+12.5f}")
print(f"  {'mean ECE':<22} {before_test['mean_ece']:>12.5f} "
      f"{after_test['mean_ece']:>12.5f} {d_ece:>+12.5f}")
print(f"  {'log-loss':<22} {before_test['log_loss']:>12.5f} "
      f"{after_test['log_loss']:>12.5f} {after_test['log_loss'] - before_test['log_loss']:>+12.5f}")
print(f"  {'macro F1':<22} {before_test['macro_f1']:>12.4f} "
      f"{after_test['macro_f1']:>12.4f} {d_f1:>+12.4f}")
print(f"  {'-' * 60}")

improved = d_ece < 0 and d_brier <= 1e-6
if improved:
    print("  Isotonic calibration improves reliability on test. Recommended for")
    print("  any downstream use that treats the outputs as probabilities.")
else:
    print("  Isotonic calibration does NOT improve test reliability. The raw")
    print("  model is already well calibrated, or the validation window is not")
    print("  representative of the test window. Ship raw probabilities.")
if d_f1 < -0.005:
    print(f"  WARNING: macro F1 drops by {abs(d_f1):.4f} under calibration —")
    print("  calibrated probabilities and argmax accuracy are in tension here.")

# ── Persist ──────────────────────────────────────────────────────────────────
cal_path = MODELS_DIR / f"probability_calibrator{TAG}.pkl"
joblib.dump({
    "calibrators":    calibrators,
    "n_classes":      N_CLASSES,
    "segment_names":  SEGMENT_NAMES,
    "fitted_on":      "validation Oct->Nov",
    "source_model":   Path(args.model).name,
    "improves_test_reliability": bool(improved),
    "test_mean_ece_before": before_test["mean_ece"],
    "test_mean_ece_after":  after_test["mean_ece"],
}, cal_path)
print(f"\n  {cal_path.name} saved")

report = {
    "source_model":  Path(args.model).name,
    "n_bins":        args.n_bins,
    "before": {"validation": before_val, "test": before_test},
    "after":  {"validation": after_val,  "test": after_test},
    "delta_test": {
        "mean_brier": d_brier,
        "mean_ece":   d_ece,
        "macro_f1":   d_f1,
    },
    "recommendation": "use_calibrated" if improved else "use_raw",
}
report_path = OUTPUTS_DIR / f"calibration_report{TAG}.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"  {report_path.name} saved")

# Human-readable reliability table for the README / deck.
lines = ["# Probability Calibration — Reliability by Bin (TEST, Nov->Dec)", ""]
for c in range(N_CLASSES):
    name = SEGMENT_NAMES[c]
    lines += [f"## {name}", "",
              "| predicted range | n | mean predicted | observed rate | gap |",
              "|---|---:|---:|---:|---:|"]
    for row in after_test["per_class"][name]["reliability"]:
        gap = row["mean_predicted"] - row["observed_rate"]
        lines.append(
            f"| {row['bin_lo']:.2f}–{row['bin_hi']:.2f} | {row['n']:,} | "
            f"{row['mean_predicted']:.3f} | {row['observed_rate']:.3f} | {gap:+.3f} |"
        )
    lines.append("")
(OUTPUTS_DIR / f"calibration_reliability{TAG}.md").write_text("\n".join(lines), encoding="utf-8")
print(f"  calibration_reliability{TAG}.md saved")

print(f"\n{'=' * 70}")
print(f"PHASE 9 COMPLETE | {elapsed()}")
print(f"{'=' * 70}")
