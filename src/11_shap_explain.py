"""
src/11_shap_explain.py
══════════════════════════════════════════════════════════════════════════════
PHASE 11 — SHAP Explanations
TBIE Pipeline | Kobie × PES University Hackathon

The `supporting_evidence` column shipped in state_assignments.csv looks like an
explanation but is not one:

    "txn_decline:-33%,recency_days:38,purchases_30d:0"

Those are the member's input values, echoed back. They say what the member did;
they do not say what drove the model's prediction, and they carry no magnitude
or direction. A campaign manager reading it cannot tell whether recency or
frequency is doing the work.

SHAP values answer the actual question — how much did each feature push this
member's prediction toward this segment, relative to the population average —
and they are additive, so per-member contributions reconcile to the prediction.

What this produces
──────────────────
  outputs/shap_global_importance.csv    mean |SHAP| per feature per class
  outputs/shap_summary.md               readable global ranking for the README
  outputs/shap_top_drivers_sample.csv   per-member top-3 drivers (sample)
  models/shap_explainer_meta.json       feature order + base values for serving

TreeExplainer on XGBoost is exact (not sampled) and fast, but memory scales
with rows x features x classes, so global analysis runs on a subsample. Serving
explains one member at a time, where cost is irrelevant.

Requires shap:  pip install shap==0.52.0

Run from TBIE_CODE root:
    python src/11_shap_explain.py
    python src/11_shap_explain.py --sample 20000
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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.pairs import build_pairs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser(description="TBIE Phase 11 — SHAP explanations")
p.add_argument("--model", type=str,
               default=str(ROOT / "models" / "segment_transition_model.pkl"))
p.add_argument("--sample", type=int, default=25_000,
               help="Rows sampled for global SHAP. TreeExplainer is exact; this "
                    "only bounds memory.")
p.add_argument("--top-k", type=int, default=3, help="Drivers kept per member.")
p.add_argument("--tag", type=str, default="")
args = p.parse_args()

TAG = f"_{args.tag}" if args.tag else ""
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()
def elapsed() -> str:
    return f"{time.time() - T0:.1f}s"


print("=" * 70)
print("TBIE — PHASE 11: SHAP EXPLANATIONS")
print("=" * 70)

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

print(f"  model    : {Path(args.model).name}")
print(f"  features : {len(X_COLS)}")

SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
print(f"\n[{elapsed()}] Building test pairs (Nov->Dec)...")
df_test = build_pairs(SNAPSHOT_DATES, [11], ROOT, BASE_FEAT_COLS,
                      VELOCITY_COLS, USE_LAG, LAG_BASE_COLS)

n_sample = min(args.sample, len(df_test))
rng = np.random.default_rng(42)
idx = rng.choice(len(df_test), n_sample, replace=False)
sample = df_test.iloc[idx].reset_index(drop=True)
X = sample[X_COLS].values.astype(np.float32)
print(f"  sampled {n_sample:,} of {len(df_test):,} test rows (seed 42)")

print(f"\n[{elapsed()}] Computing SHAP values (xgboost native TreeSHAP, exact)...")
# xgboost implements TreeSHAP directly via pred_contribs. This is the same
# algorithm the `shap` package wraps, but without the extra dependency — and
# shap 0.48 cannot read xgboost 3.x multiclass models anyway, because
# base_score became a per-class vector and its loader still expects a scalar.
booster  = model.get_booster()
contribs = booster.predict(xgb.DMatrix(X, feature_names=list(X_COLS)),
                           pred_contribs=True)
contribs = np.asarray(contribs)

# Multiclass shape: (rows, classes, features + 1); the trailing column is bias.
if contribs.ndim == 2:                      # binary/regression fallback
    contribs = contribs[:, None, :]
sv = np.transpose(contribs[:, :, :-1], (0, 2, 1))     # -> rows x features x classes
base_values = contribs[0, :, -1].astype(float)        # bias per class

if sv.shape[1] != len(X_COLS):
    raise RuntimeError(
        f"SHAP feature axis is {sv.shape[1]} but the model has {len(X_COLS)} features"
    )
print(f"  SHAP array: {sv.shape}  (rows x features x classes) | {elapsed()}")

# TreeSHAP is additive: contributions + bias must reconstruct the raw margin.
margin = booster.predict(xgb.DMatrix(X, feature_names=list(X_COLS)),
                         output_margin=True)
recon  = sv.sum(axis=1) + base_values
max_err = float(np.abs(recon - margin).max())
print(f"  additivity check: max |sum(SHAP) + bias - margin| = {max_err:.2e}")
if max_err > 1e-2:
    raise RuntimeError(f"SHAP values do not reconstruct the margin (err {max_err})")

# ── Global importance ────────────────────────────────────────────────────────
print(f"\n[{elapsed()}] Global importance (mean |SHAP|)...")
mean_abs = np.abs(sv).mean(axis=0)              # (features, classes)
overall  = mean_abs.mean(axis=1)                # (features,)

imp = pd.DataFrame(mean_abs, index=X_COLS, columns=SEGMENT_NAMES)
imp.insert(0, "overall", overall)
imp = imp.sort_values("overall", ascending=False)
imp.index.name = "feature"
imp.to_csv(OUTPUTS_DIR / f"shap_global_importance{TAG}.csv")

print(f"\n  {'rank':>4}  {'feature':<32} {'mean |SHAP|':>12}")
print(f"  {'-' * 54}")
for rank, (feat, row) in enumerate(imp.head(20).iterrows(), 1):
    bar = "#" * max(1, int(row["overall"] / imp["overall"].max() * 28))
    print(f"  {rank:>4}. {feat:<32} {row['overall']:>12.5f}  {bar}")

# ── Per-member top drivers ───────────────────────────────────────────────────
print(f"\n[{elapsed()}] Extracting top-{args.top_k} drivers per member...")
pred_class = model.predict_proba(X).argmax(axis=1)
rows_idx   = np.arange(len(X))
sv_pred    = sv[rows_idx, :, pred_class]         # SHAP for the predicted class
order      = np.argsort(-np.abs(sv_pred), axis=1)[:, :args.top_k]

def format_evidence(drivers, max_k: int = 3) -> str:
    """
    Render drivers as a compact, signed, human-readable string.

    Unlike the old value-echo format, this states direction and magnitude:
        "recency_days=38 raises 0.420; purchase_count_30d=0 raises 0.310"
    """
    parts = []
    for d in drivers[:max_k]:
        direction = "raises" if d["shap"] > 0 else "lowers"
        parts.append(f"{d['feature']}={d['value']:.4g} {direction} {abs(d['shap']):.3f}")
    return "; ".join(parts)


feat_arr = np.array(X_COLS)
records  = []
for i in range(len(X)):
    drivers = [
        {
            "feature": feat_arr[j],
            "value":   float(X[i, j]),
            "shap":    float(sv_pred[i, j]),
        }
        for j in order[i]
    ]
    records.append({
        "member_id":         sample.loc[i, "member_id"],
        "predicted_segment": SID_TO_NAME[CLUSTER_TO_SID[int(pred_class[i])]],
        "evidence":          format_evidence(drivers),
        "drivers":           drivers,
    })

flat = pd.DataFrame([{
    "member_id":         r["member_id"],
    "predicted_segment": r["predicted_segment"],
    "evidence":          r["evidence"],
    **{f"driver{n+1}_feature": r["drivers"][n]["feature"] for n in range(len(r["drivers"]))},
    **{f"driver{n+1}_shap":    round(r["drivers"][n]["shap"], 5) for n in range(len(r["drivers"]))},
} for r in records])
flat.to_csv(OUTPUTS_DIR / f"shap_top_drivers_sample{TAG}.csv", index=False)
print(f"  shap_top_drivers_sample{TAG}.csv  ({len(flat):,} members)")
print("\n  Example explanations:")
for _, r in flat.head(3).iterrows():
    print(f"    {r['member_id']} -> {r['predicted_segment']}")
    print(f"      {r['evidence']}")

# ── Readable global summary ──────────────────────────────────────────────────
lines = [
    "# SHAP — Global Feature Importance",
    "",
    f"Model: `{Path(args.model).name}` · {len(X_COLS)} features · "
    f"{n_sample:,} sampled test rows (seed 42)",
    "",
    "Mean absolute SHAP value: the average magnitude by which each feature moves",
    "a prediction, in log-odds. Larger means the model leans on it more.",
    "",
    "| rank | feature | mean \\|SHAP\\| | strongest for |",
    "|---:|---|---:|---|",
]
for rank, (feat, row) in enumerate(imp.head(25).iterrows(), 1):
    strongest = row[SEGMENT_NAMES].astype(float).idxmax()
    lines.append(f"| {rank} | `{feat}` | {row['overall']:.5f} | {strongest} |")
lines += ["", "## Per-class top 5", ""]
for name in SEGMENT_NAMES:
    top5 = imp[name].astype(float).sort_values(ascending=False).head(5)
    lines.append(f"**{name}** — " + ", ".join(f"`{f}` ({v:.4f})" for f, v in top5.items()))
    lines.append("")
(OUTPUTS_DIR / f"shap_summary{TAG}.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n  shap_summary{TAG}.md saved")

(MODELS_DIR / f"shap_explainer_meta{TAG}.json").write_text(json.dumps({
    "feature_cols":   list(X_COLS),
    "segment_names":  SEGMENT_NAMES,
    "base_values":    base_values.tolist(),
    "source_model":   Path(args.model).name,
    "sample_rows":    int(n_sample),
    "top_k":          args.top_k,
}, indent=2), encoding="utf-8")
print(f"  shap_explainer_meta{TAG}.json saved")

print(f"\n{'=' * 70}")
print(f"PHASE 11 COMPLETE | {elapsed()}")
print(f"{'=' * 70}")
