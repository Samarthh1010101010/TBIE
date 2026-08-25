"""
src/10_cost_thresholds.py
══════════════════════════════════════════════════════════════════════════════
PHASE 10 — Cost-Sensitive Decision Thresholds
TBIE Pipeline | Kobie × PES University Hackathon

Phase 8 tunes per-class thresholds to maximise F1. F1 is a modelling metric: it
weights a false positive and a false negative equally. In a retention campaign
they are not equal at all — a false positive costs one touchpoint (a few
dollars), a false negative costs a share of a member's forward value (hundreds
to thousands). Optimising F1 therefore optimises the wrong thing.

This phase re-derives the operating point from expected value instead.

Decision rule
─────────────
Contact a member when the expected benefit exceeds the expected cost:

    P(adverse transition) x value_at_risk x recovery_rate  >  contact_cost

"Adverse" means the model predicts a member will land in a segment with lower
forward value than the one they occupy now. Segment values are estimated from
observed 180-day spend in the data, not assumed.

What this reports
─────────────────
  - the expected-value-optimal contact threshold
  - campaign profit, ROI and contact volume at that threshold
  - the same figures at the F1-optimal threshold, for comparison
  - a sweep so the operating point can be moved for a fixed budget

Honesty note
────────────
`recovery_rate` — the fraction of at-risk value that a successful contact
actually saves — cannot be estimated from this dataset. There is no experiment
in it: nobody was randomly withheld from contact. It is therefore an explicit
input with a conservative default, and the sweep shows how conclusions move
across the plausible range. Every currency figure here is conditional on it.

Run from TBIE_CODE root:
    python src/10_cost_thresholds.py
    python src/10_cost_thresholds.py --contact-cost 5 --recovery-rate 0.10
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
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.calibration import renormalise  # noqa: E402
from utils.economics import (  # noqa: E402
    adverse_probability,
    contact_by_expected_value,
    contact_by_probability,
    evaluate_campaign,
    segment_values,
    value_at_risk,
)
from utils.pairs import build_pairs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser(description="TBIE Phase 10 — cost-sensitive thresholds")
p.add_argument("--model", type=str,
               default=str(ROOT / "models" / "segment_transition_model.pkl"))
p.add_argument("--calibrator", type=str,
               default=str(ROOT / "models" / "probability_calibrator.pkl"),
               help="Isotonic calibrator from Phase 9. Expected-value maths on "
                    "miscalibrated probabilities is unreliable.")
p.add_argument("--contact-cost", type=float, default=5.0,
               help="Cost of one outreach touchpoint, in dollars.")
p.add_argument("--recovery-rate", type=float, default=0.10,
               help="Fraction of at-risk value a successful contact recovers. "
                    "NOT estimable from this dataset — see module docstring.")
p.add_argument("--horizon-days", type=int, default=180,
               help="Spend window used to value a segment.")
p.add_argument("--tag", type=str, default="")
args = p.parse_args()

TAG = f"_{args.tag}" if args.tag else ""
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()
def elapsed() -> str:
    return f"{time.time() - T0:.1f}s"


print("=" * 70)
print("TBIE — PHASE 10: COST-SENSITIVE DECISION THRESHOLDS")
print("=" * 70)
print(f"  contact cost   : ${args.contact_cost:,.2f} per touchpoint")
print(f"  recovery rate  : {args.recovery_rate:.0%}  (assumption — not measured)")
print(f"  value horizon  : {args.horizon_days}d observed spend")

# ── Load model ───────────────────────────────────────────────────────────────
bundle         = joblib.load(args.model)
model          = bundle["model"]
X_COLS         = bundle["feature_cols"]
VELOCITY_COLS  = bundle["velocity_cols"]
LAG_BASE_COLS  = bundle.get("lag_base_cols", [])
USE_LAG        = bool(bundle.get("uses_lag_features", bool(LAG_BASE_COLS)))
CLUSTER_TO_SID = bundle["cluster_to_sid"]
SID_TO_NAME    = bundle["sid_to_name"]
N_CLASSES      = bundle.get("n_classes", len(CLUSTER_TO_SID))
SEGMENT_NAMES  = [SID_TO_NAME[CLUSTER_TO_SID[c]] for c in range(N_CLASSES)]

seg_model      = joblib.load(ROOT / "segments" / "segment_model.pkl")
BASE_FEAT_COLS = seg_model["behavioral_feature_cols"]

# ── Build the test window ────────────────────────────────────────────────────
SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
print(f"\n[{elapsed()}] Building validation pairs (threshold selection)...")
df_val = build_pairs(SNAPSHOT_DATES, [10], ROOT, BASE_FEAT_COLS,
                     VELOCITY_COLS, USE_LAG, LAG_BASE_COLS)
print(f"[{elapsed()}] Building test pairs (evaluation)...")
df_test = build_pairs(SNAPSHOT_DATES, [11], ROOT, BASE_FEAT_COLS,
                      VELOCITY_COLS, USE_LAG, LAG_BASE_COLS)

SPEND_COL = f"spend_total_{args.horizon_days}d"
if SPEND_COL not in df_test.columns:
    raise KeyError(f"{SPEND_COL} not present in the built pairs — cannot value segments")

X_va, y_va = df_val[X_COLS].values.astype(np.float32), df_val["seg_next"].values.astype(int)
X_te, y_te = df_test[X_COLS].values.astype(np.float32), df_test["seg_next"].values.astype(int)

proba_va = model.predict_proba(X_va)
proba_te = model.predict_proba(X_te)

# Apply the Phase 9 calibrator when it was found to help on test.
cal_path = Path(args.calibrator)
if cal_path.exists():
    cal = joblib.load(cal_path)
    if cal.get("improves_test_reliability", False):
        def _apply(pr):
            out = np.column_stack([cal["calibrators"][c].predict(pr[:, c])
                                   for c in range(N_CLASSES)])
            return renormalise(out, fallback=pr)
        proba_va, proba_te = _apply(proba_va), _apply(proba_te)
        print("  Using CALIBRATED probabilities (Phase 9 isotonic)")
    else:
        print("  Phase 9 found calibration did not help — using raw probabilities")
else:
    print(f"  WARNING: no calibrator at {cal_path.name}; using raw probabilities. "
          f"Run src/09_calibration.py first for trustworthy expected values.")

# ── Value each segment from observed spend ───────────────────────────────────
print(f"\n[{elapsed()}] Estimating segment value from observed {args.horizon_days}d spend...")
seg_value = pd.Series(
    segment_values(df_test, "seg_curr", SPEND_COL, N_CLASSES),
    index=range(N_CLASSES),
)

print(f"  {'segment':<25} {'mean ' + SPEND_COL:>18}  {'members':>9}")
print(f"  {'-' * 58}")
for c in range(N_CLASSES):
    n = int((df_test['seg_curr'] == c).sum())
    print(f"  {SEGMENT_NAMES[c]:<25} {seg_value[c]:>18,.2f}  {n:>9,}")

VALUES = seg_value.values.astype(float)

# ── Define "adverse transition" and per-member value at risk ─────────────────
# A move is adverse when the destination segment is worth less than the origin.
# Expected value at risk = sum over destinations worth less than today of
#   P(destination) x (value_today - value_destination)
seg_curr_te = df_test["seg_curr"].values.astype(int)
seg_curr_va = df_val["seg_curr"].values.astype(int)


# value_at_risk / adverse_probability live in src/utils/economics.py so the
# serving API can quote the identical figure. See tests/test_economics_calibration.py.
var_te = value_at_risk(proba_te, seg_curr_te, VALUES)
adv_te = adverse_probability(proba_te, seg_curr_te, VALUES)
var_va = value_at_risk(proba_va, seg_curr_va, VALUES)
adv_va = adverse_probability(proba_va, seg_curr_va, VALUES)

# Ground truth: did the member actually move to a lower-value segment?
actually_adverse_te = VALUES[y_te] < VALUES[seg_curr_te]
actually_adverse_va = VALUES[y_va] < VALUES[seg_curr_va]
realised_loss_te = np.clip(VALUES[seg_curr_te] - VALUES[y_te], 0.0, None)

print(f"\n  Members whose segment value actually fell (test): "
      f"{actually_adverse_te.sum():,} ({actually_adverse_te.mean():.1%})")
print(f"  Total realised value drop (test): ${realised_loss_te.sum():,.0f}")


# ── Campaign economics ───────────────────────────────────────────────────────
#
# Two different decision statistics, and the distinction is the whole point:
#
#   probability-ranked : contact when P(adverse) >= t
#                        targets members most LIKELY to slip
#   value-ranked (EV)  : contact when P(adverse) x value_at_risk x recovery
#                        >= contact_cost x k
#                        targets members whose slipping COSTS most
#
# A member with a 90% chance of losing $20 is a worse use of a $5 touchpoint
# than one with a 15% chance of losing $900. Ranking on probability alone --
# which is what an F1-tuned threshold does -- cannot see that difference.

def campaign(contact_mask, actually_adverse, realised_loss, label_value: float):
    """Thin wrapper binding the CLI's cost assumptions to the shared scorer."""
    return evaluate_campaign(contact_mask, actually_adverse, realised_loss,
                             contact_cost=args.contact_cost,
                             recovery_rate=args.recovery_rate,
                             label=label_value)


def by_probability(adv, t):
    return contact_by_probability(adv, t)


def by_expected_value(var, k):
    """Contact when expected recoverable value clears k x the contact cost."""
    return contact_by_expected_value(var, args.recovery_rate, args.contact_cost, k)


realised_loss_va = np.clip(VALUES[seg_curr_va] - VALUES[y_va], 0.0, None)

# ── Select both operating points on VALIDATION, apply once to TEST ───────────
print(f"\n[{elapsed()}] Selecting operating points on VALIDATION...")

prob_grid = np.linspace(0.01, 0.99, 99)
k_grid    = np.concatenate([np.linspace(0.1, 5.0, 50), np.linspace(5.5, 40.0, 40)])

val_prob = [campaign(by_probability(adv_va, t), actually_adverse_va, realised_loss_va, t)
            for t in prob_grid]
val_ev   = [campaign(by_expected_value(var_va, k), actually_adverse_va, realised_loss_va, k)
            for k in k_grid]

best_prob_val = max(val_prob, key=lambda r: r["profit"])
best_ev_val   = max(val_ev,   key=lambda r: r["profit"])
PROB_THRESHOLD = best_prob_val["threshold"]
EV_K           = best_ev_val["threshold"]

print(f"  probability-ranked optimum : p >= {PROB_THRESHOLD:.2f}  "
      f"(val profit ${best_prob_val['profit']:,.0f})")
print(f"  value-ranked optimum       : EV >= {EV_K:.2f} x cost  "
      f"(val profit ${best_ev_val['profit']:,.0f})")

test_sweep = [campaign(by_expected_value(var_te, k), actually_adverse_te,
                       realised_loss_te, k) for k in k_grid]
ev_test   = campaign(by_expected_value(var_te, EV_K), actually_adverse_te,
                     realised_loss_te, EV_K)
prob_test = campaign(by_probability(adv_te, PROB_THRESHOLD), actually_adverse_te,
                     realised_loss_te, PROB_THRESHOLD)

# Reference point: the threshold that maximises F1 on the adverse-vs-not task,
# i.e. what you get by treating this as a pure classification problem.
f1_scores = [f1_score(actually_adverse_va, adv_va >= t, zero_division=0) for t in prob_grid]
F1_THRESHOLD = float(prob_grid[int(np.argmax(f1_scores))])
f1_test = campaign(by_probability(adv_te, F1_THRESHOLD), actually_adverse_te,
                   realised_loss_te, F1_THRESHOLD)

# Contact-everyone reference.
all_test = campaign(np.ones(len(adv_te), dtype=bool), actually_adverse_te,
                    realised_loss_te, 0.0)

print(f"\n{'=' * 70}")
print("CAMPAIGN ECONOMICS — TEST (Nov->Dec)")
print(f"{'=' * 70}")
print(f"  {'policy':<26} {'thr':>5} {'contacted':>10} {'cost':>12} "
      f"{'recovered':>12} {'profit':>12} {'ROI':>7}")
print(f"  {'-' * 88}")
for label, r in (("contact everyone", all_test),
                 ("F1-optimal (probability)", f1_test),
                 ("best probability threshold", prob_test),
                 ("EV-ranked (value-aware)", ev_test)):
    print(f"  {label:<26} {r['threshold']:>5.2f} {r['contacted']:>10,} "
          f"${r['cost']:>11,.0f} ${r['recovered']:>11,.0f} "
          f"${r['profit']:>11,.0f} {r['roi']:>6.2f}x")
print(f"  {'-' * 88}")

gain = ev_test["profit"] - f1_test["profit"]
print("\n  Ranking by expected value rather than by probability alone is worth")
print(f"  ${gain:,.0f} on this window ({ev_test['contacted']:,} contacts vs "
      f"{f1_test['contacted']:,} under the F1 threshold).")
print(f"  EV policy  precision {ev_test['precision']:.1%}  recall {ev_test['recall']:.1%}  "
      f"ROI {ev_test['roi']:.2f}x")
print(f"  F1 policy  precision {f1_test['precision']:.1%}  recall {f1_test['recall']:.1%}  "
      f"ROI {f1_test['roi']:.2f}x")
if gain <= 0:
    print("  NOTE: value-aware ranking did NOT beat the F1 threshold on this window.")

# ── Sensitivity to the recovery-rate assumption ──────────────────────────────
print("\n  Sensitivity — the recovery rate is an assumption, so vary it:")
print(f"    {'recovery':>9} {'EV thr':>7} {'contacted':>10} {'profit':>13} {'ROI':>7}")
print(f"    {'-' * 50}")
sensitivity = []
for rr in (0.02, 0.05, 0.10, 0.20, 0.30):
    saved = args.recovery_rate
    args.recovery_rate = rr
    sweep_v = [campaign(by_expected_value(var_va, k), actually_adverse_va,
                        realised_loss_va, k) for k in k_grid]
    thr = max(sweep_v, key=lambda r: r["profit"])["threshold"]
    r = campaign(by_expected_value(var_te, thr), actually_adverse_te,
                 realised_loss_te, thr)
    sensitivity.append({"recovery_rate": rr, **r})
    print(f"    {rr:>9.0%} {thr:>7.2f} {r['contacted']:>10,} "
          f"${r['profit']:>12,.0f} {r['roi']:>6.2f}x")
    args.recovery_rate = saved
print(f"    {'-' * 50}")
print("    Break-even recovery rate is where profit crosses zero. Below it, no")
print("    targeting policy pays for itself at this contact cost.")

# ── Persist ──────────────────────────────────────────────────────────────────
report = {
    "assumptions": {
        "contact_cost_usd":  args.contact_cost,
        "recovery_rate":     args.recovery_rate,
        "value_horizon_days": args.horizon_days,
        "recovery_rate_note": ("Not estimable from this dataset — no randomised "
                               "holdout exists. All currency figures are "
                               "conditional on this input."),
    },
    "segment_values_usd": {SEGMENT_NAMES[c]: float(VALUES[c]) for c in range(N_CLASSES)},
    "thresholds": {
        "ev_k_multiplier":    EV_K,
        "best_probability":   PROB_THRESHOLD,
        "f1_optimal":         F1_THRESHOLD,
    },
    "test": {
        "contact_everyone":     all_test,
        "f1_optimal":           f1_test,
        "best_probability":     prob_test,
        "ev_optimal":           ev_test,
        "profit_gain_from_ev":  gain,
    },
    "sensitivity_to_recovery_rate": sensitivity,
    "test_sweep": test_sweep,
    "source_model": Path(args.model).name,
}
out = OUTPUTS_DIR / f"cost_threshold_report{TAG}.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\n  {out.name} saved")

pd.DataFrame(test_sweep).to_csv(OUTPUTS_DIR / f"cost_threshold_sweep{TAG}.csv", index=False)
print(f"  cost_threshold_sweep{TAG}.csv saved")

print(f"\n{'=' * 70}")
print(f"PHASE 10 COMPLETE | {elapsed()}")
print(f"{'=' * 70}")
