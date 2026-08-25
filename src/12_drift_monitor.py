"""
src/12_drift_monitor.py
══════════════════════════════════════════════════════════════════════════════
PHASE 12 — Drift Monitoring
TBIE Pipeline | Kobie × PES University Hackathon

The K-Means centroids are frozen on December 2025 and never refitted. That is
what makes segment assignments reproducible, and it is also the model's main
expiry mechanism: as member behaviour moves, the December centroids describe
the population less and less well, and nothing in the pipeline notices.

MODEL_CARD.md promises a retraining trigger. This is it.

What it measures
────────────────
  1. Feature drift — Population Stability Index and a two-sample KS statistic
     per feature, current month vs the December 2025 reference.
  2. Cluster quality — Calinski-Harabasz on the current month against the
     score achieved at fit time. A sustained fall means the centroids no
     longer describe the data.
  3. Segment mix — how far the segment size distribution has moved, in
     percentage points and total variation distance.
  4. State mix — the same for the 10 lifecycle states.

PSI convention (standard in credit risk, where it comes from):
    < 0.10     no material shift
    0.10-0.25  moderate — worth watching
    > 0.25     significant — investigate and consider refitting

Exit code is 1 when any threshold is breached, so this can gate a scheduled
retrain in CI without extra glue.

Run from TBIE_CODE root:
    python src/12_drift_monitor.py --current 2025-12-01
    python src/12_drift_monitor.py --current 2025-12-01 --reference 2025-06-01
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
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.metrics import calinski_harabasz_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.state_rules import classify_states  # noqa: E402

p = argparse.ArgumentParser(description="TBIE Phase 12 — drift monitoring")
p.add_argument("--current", type=str, required=True,
               help="Observation date to check, YYYY-MM-DD.")
p.add_argument("--reference", type=str, default="2025-12-01",
               help="Reference date the frozen model was fitted on.")
p.add_argument("--psi-warn", type=float, default=0.10)
p.add_argument("--psi-alert", type=float, default=0.25)
p.add_argument("--ch-drop-alert", type=float, default=0.20,
               help="Fractional fall in Calinski-Harabasz that triggers an alert.")
p.add_argument("--segment-shift-alert", type=float, default=0.10,
               help="Total variation distance in segment mix that triggers an alert.")
p.add_argument("--fail-on-alert", action="store_true", default=True)
p.add_argument("--no-fail-on-alert", dest="fail_on_alert", action="store_false")
args = p.parse_args()

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()
def elapsed() -> str:
    return f"{time.time() - T0:.1f}s"


def feat_path(date_str: str) -> Path:
    return ROOT / "features" / f"features_{date_str.replace('-', '_')}.parquet"


print("=" * 70)
print("TBIE — PHASE 12: DRIFT MONITORING")
print("=" * 70)
print(f"  reference : {args.reference}   (frozen model fit window)")
print(f"  current   : {args.current}")

cur_path, ref_path = feat_path(args.current), feat_path(args.reference)
for path in (cur_path, ref_path):
    if not path.exists():
        print(f"FATAL: feature file not found: {path}")
        sys.exit(2)

seg_model = joblib.load(ROOT / "segments" / "segment_model.pkl")
FEAT_COLS = seg_model["behavioral_feature_cols"]

ref = pd.read_parquet(ref_path, engine="pyarrow")
cur = pd.read_parquet(cur_path, engine="pyarrow")
print(f"  reference rows: {len(ref):,}   current rows: {len(cur):,} | {elapsed()}")

alerts: list[str] = []
warnings_: list[str] = []


# ── 1. Feature drift ─────────────────────────────────────────────────────────
def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index over quantile bins of the reference sample.

    Bins come from the reference so the metric answers "has the current month
    moved relative to what the model was fitted on".
    """
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:          # near-constant feature; no meaningful bins
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    a_pct = np.histogram(actual,   bins=edges)[0] / len(actual)
    eps = 1e-6
    e_pct, a_pct = np.clip(e_pct, eps, None), np.clip(a_pct, eps, None)
    return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())


print(f"\n[{elapsed()}] 1. Feature drift (PSI + KS vs reference)...")
rng = np.random.default_rng(42)
ks_n = min(20_000, len(ref), len(cur))       # KS is O(n log n); subsample is plenty
ref_ks_idx = rng.choice(len(ref), ks_n, replace=False)
cur_ks_idx = rng.choice(len(cur), ks_n, replace=False)

rows = []
for col in FEAT_COLS:
    if col not in ref.columns or col not in cur.columns:
        continue
    e = ref[col].to_numpy(dtype=float)
    a = cur[col].to_numpy(dtype=float)
    val = psi(e, a)

    # ks_2samp propagates NaN and returns NaN for the whole test. recency_days
    # is null for never-purchased members, so dropping nulls here is required
    # or the most-drifted features come back with a blank statistic.
    e_ks = e[ref_ks_idx]
    a_ks = a[cur_ks_idx]
    e_ks = e_ks[np.isfinite(e_ks)]
    a_ks = a_ks[np.isfinite(a_ks)]
    if len(e_ks) and len(a_ks):
        ks_stat, ks_p = ks_2samp(e_ks, a_ks)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    rows.append({
        "feature":    col,
        "psi":        round(val, 5),
        "ks_stat":    round(float(ks_stat), 5),
        "ks_pvalue":  float(ks_p),
        "ks_n_ref":   int(len(e_ks)),
        "ks_n_cur":   int(len(a_ks)),
        "ref_mean":   round(float(np.nanmean(e)), 4),
        "cur_mean":   round(float(np.nanmean(a)), 4),
        "status":     "ALERT" if val > args.psi_alert
                      else ("WARN" if val > args.psi_warn else "ok"),
    })

drift = pd.DataFrame(rows).sort_values("psi", ascending=False)
drift.to_csv(OUTPUTS_DIR / "drift_feature_report.csv", index=False)

n_alert = int((drift["status"] == "ALERT").sum())
n_warn  = int((drift["status"] == "WARN").sum())
print(f"  {len(drift)} features checked: {n_alert} ALERT, {n_warn} WARN")
print(f"\n  {'feature':<32} {'PSI':>8} {'KS':>7}  {'ref mean':>12} {'cur mean':>12}  status")
print(f"  {'-' * 84}")
for _, r in drift.head(12).iterrows():
    print(f"  {r['feature']:<32} {r['psi']:>8.4f} {r['ks_stat']:>7.4f}  "
          f"{r['ref_mean']:>12,.3f} {r['cur_mean']:>12,.3f}  {r['status']}")

if n_alert:
    alerts.append(f"{n_alert} feature(s) exceed PSI {args.psi_alert}")
if n_warn:
    warnings_.append(f"{n_warn} feature(s) exceed PSI {args.psi_warn}")


# ── 2. Cluster quality under the frozen centroids ────────────────────────────
print(f"\n[{elapsed()}] 2. Cluster quality under frozen centroids...")
scaler, pca, centroids = seg_model["scaler"], seg_model["pca"], seg_model["centroids"]
cids = sorted(centroids.keys())
cmat = np.array([centroids[c] for c in cids])


def assign(df: pd.DataFrame):
    X = pca.transform(scaler.transform(df[FEAT_COLS].fillna(0).values))
    labels = np.array(cids)[cdist(X, cmat).argmin(axis=1)]
    return X, labels


ch_n = min(50_000, len(ref), len(cur))
ref_s = ref.sample(ch_n, random_state=42)
cur_s = cur.sample(ch_n, random_state=42)
Xr, lr = assign(ref_s)
Xc, lc = assign(cur_s)

ch_ref = calinski_harabasz_score(Xr, lr) if len(set(lr)) > 1 else float("nan")
ch_cur = calinski_harabasz_score(Xc, lc) if len(set(lc)) > 1 else float("nan")
ch_drop = (ch_ref - ch_cur) / ch_ref if ch_ref and np.isfinite(ch_ref) else 0.0

print(f"  Calinski-Harabasz  reference={ch_ref:,.0f}  current={ch_cur:,.0f}  "
      f"change={-ch_drop:+.1%}  (n={ch_n:,} each)")
if ch_drop > args.ch_drop_alert:
    alerts.append(f"Calinski-Harabasz fell {ch_drop:.1%} "
                  f"(threshold {args.ch_drop_alert:.0%}) — centroids may need refitting")


# ── 3. Segment mix ───────────────────────────────────────────────────────────
print(f"\n[{elapsed()}] 3. Segment mix shift...")
_, lr_full = assign(ref)
_, lc_full = assign(cur)
ref_mix = pd.Series(lr_full).value_counts(normalize=True).reindex(cids).fillna(0)
cur_mix = pd.Series(lc_full).value_counts(normalize=True).reindex(cids).fillna(0)
tvd = float(np.abs(ref_mix.values - cur_mix.values).sum() / 2)

print(f"  {'cluster':>8} {'reference':>11} {'current':>11} {'shift (pp)':>12}")
print(f"  {'-' * 46}")
for c in cids:
    shift = (cur_mix[c] - ref_mix[c]) * 100
    print(f"  {c:>8} {ref_mix[c]:>10.2%} {cur_mix[c]:>10.2%} {shift:>+11.2f}")
print(f"  total variation distance: {tvd:.4f}")
if tvd > args.segment_shift_alert:
    alerts.append(f"Segment mix moved TVD {tvd:.3f} "
                  f"(threshold {args.segment_shift_alert})")


# ── 4. State mix ─────────────────────────────────────────────────────────────
print(f"\n[{elapsed()}] 4. Lifecycle state mix shift...")
ref_states = pd.Series(classify_states(ref)).value_counts(normalize=True)
cur_states = pd.Series(classify_states(cur)).value_counts(normalize=True)
all_states = sorted(set(ref_states.index) | set(cur_states.index))
ref_states = ref_states.reindex(all_states).fillna(0)
cur_states = cur_states.reindex(all_states).fillna(0)
state_tvd = float(np.abs(ref_states.values - cur_states.values).sum() / 2)

print(f"  {'state':<22} {'reference':>11} {'current':>11} {'shift (pp)':>12}")
print(f"  {'-' * 60}")
for s in all_states:
    print(f"  {s:<22} {ref_states[s]:>10.2%} {cur_states[s]:>10.2%} "
          f"{(cur_states[s] - ref_states[s]) * 100:>+11.2f}")
print(f"  total variation distance: {state_tvd:.4f}")
if state_tvd > args.segment_shift_alert:
    warnings_.append(f"State mix moved TVD {state_tvd:.3f}")


# ── Verdict ──────────────────────────────────────────────────────────────────
report = {
    "reference_date": args.reference,
    "current_date":   args.current,
    "thresholds": {
        "psi_warn": args.psi_warn, "psi_alert": args.psi_alert,
        "ch_drop_alert": args.ch_drop_alert,
        "segment_shift_alert": args.segment_shift_alert,
    },
    "feature_drift": {
        "n_features": int(len(drift)),
        "n_alert": n_alert, "n_warn": n_warn,
        "max_psi_feature": str(drift.iloc[0]["feature"]) if len(drift) else None,
        "max_psi": float(drift.iloc[0]["psi"]) if len(drift) else 0.0,
    },
    "cluster_quality": {
        "calinski_harabasz_reference": float(ch_ref),
        "calinski_harabasz_current":   float(ch_cur),
        "fractional_drop":             float(ch_drop),
    },
    "segment_mix_tvd": tvd,
    "state_mix_tvd":   state_tvd,
    "alerts":   alerts,
    "warnings": warnings_,
    "verdict":  "RETRAIN_RECOMMENDED" if alerts else ("MONITOR" if warnings_ else "HEALTHY"),
}
(OUTPUTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\n{'=' * 70}")
print(f"VERDICT: {report['verdict']}")
print(f"{'=' * 70}")
for a in alerts:
    print(f"  ALERT  {a}")
for w in warnings_:
    print(f"  WARN   {w}")
if not alerts and not warnings_:
    print("  No material drift detected against the reference window.")
print(f"\n  drift_report.json + drift_feature_report.csv saved | {elapsed()}")

if alerts and args.fail_on_alert:
    sys.exit(1)
