"""
src/14_clustering_search.py
══════════════════════════════════════════════════════════════════════════════
PHASE 14 — Clustering Configuration Search (silhouette vs downstream F1)
TBIE Pipeline | Kobie × PES University Hackathon

The shipped segmentation scores silhouette 0.117, which looks poor. Two earlier
experiments (METHODOLOGY §2) raised it to 0.37 and 0.22 by transforming the
feature space, and both destroyed downstream macro F1 (0.69 and 0.61). The
conclusion drawn was that silhouette cannot be improved without cost.

That conclusion was based on two aggressive transformations. This searches the
space properly, including a lever those experiments never touched: PCA
dimensionality.

Why dimensionality matters here
───────────────────────────────
Silhouette is a distance-ratio metric, and distance ratios concentrate as
dimension grows — in high dimensions every point drifts towards equidistant, so
silhouette falls even when cluster structure is unchanged. The shipped model
clusters in 18 PCA components (85% variance). Clustering in fewer components
can raise silhouette substantially without changing the underlying behavioural
grouping at all.

Two stages
──────────
  Stage 1  sweep configurations, score cluster geometry only (fast)
  Stage 2  for the best candidates, retrain the transition model and measure
           validation macro F1 — because geometry is not the graded metric

Stage 2 is what makes this honest. A configuration that improves silhouette and
degrades F1 is not an improvement; the output is a Pareto table, not a winner.

Run from TBIE_CODE root:
    python src/14_clustering_search.py --stage 1
    python src/14_clustering_search.py --stage 2 --top 4
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

p = argparse.ArgumentParser(description="TBIE Phase 14 — clustering search")
p.add_argument("--stage", type=int, choices=[1, 2], default=1)
p.add_argument("--fit-n", type=int, default=80_000,
               help="Rows used to fit each candidate (speed).")
p.add_argument("--sil-n", type=int, default=20_000,
               help="Rows used for the silhouette computation (it is O(n^2)).")
p.add_argument("--top", type=int, default=4,
               help="Stage 2: how many stage-1 candidates to evaluate on F1.")
p.add_argument("--configs", type=str, default="",
               help="Stage 2: explicit candidates as 'transform:dims:k' separated "
                    "by commas, e.g. 'standard:4:5,standard:8:5'. Overrides --top. "
                    "Use this to isolate a single variable (dimensionality) rather "
                    "than taking whatever scored highest, which selects for "
                    "microcluster artifacts.")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--stage2-rows", type=int, default=600_000,
               help="Training rows per stage-2 candidate. Every candidate gets "
                   "the same budget, so the comparison stays fair; the absolute "
                   "F1 will sit below a full-data run.")
p.add_argument("--stage2-months", type=int, default=4,
               help="How many of the most recent training months to use in "
                    "stage 2. Fewer months means a faster comparison.")
args = p.parse_args()

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()
def elapsed() -> str:
    return f"{time.time() - T0:.1f}s"


print("=" * 70)
print("TBIE — PHASE 14: CLUSTERING CONFIGURATION SEARCH")
print("=" * 70)

seg_model = joblib.load(ROOT / "segments" / "segment_model.pkl")
FEAT_COLS = seg_model["behavioral_feature_cols"]

dec = pd.read_parquet(ROOT / "features" / "features_2025_12_01.parquet",
                      engine="pyarrow")
# Same fit population as Phase 6: complete features, some purchase history.
mask = (dec.get("feature_complete", pd.Series(1, index=dec.index)) == 1)
if "purchase_count_180d" in dec.columns:
    mask &= dec["purchase_count_180d"] > 0
fit_df = dec[mask][FEAT_COLS].fillna(0)
print(f"  fit population: {len(fit_df):,} rows x {len(FEAT_COLS)} features | {elapsed()}")

rng = np.random.default_rng(args.seed)
fit_idx = rng.choice(len(fit_df), min(args.fit_n, len(fit_df)), replace=False)
Xraw = fit_df.values[fit_idx]
print(f"  candidate fit subsample: {Xraw.shape[0]:,} rows")


# ── Transformations ──────────────────────────────────────────────────────────
def transform_none(X):
    return StandardScaler().fit_transform(X), "standard"


def transform_log1p(X):
    """
    log1p on non-negative heavy-tailed columns only, then standardise.

    A gentler version of the rejected Experiment 1: that one applied
    log1p + RobustScaler to everything, which compressed the spend variance the
    segmentation actually depends on.
    """
    Xc = X.copy()
    nonneg = (Xc >= 0).all(axis=0)
    # Skewness proxy: mean well above median implies a heavy right tail.
    med = np.median(Xc, axis=0)
    mean = Xc.mean(axis=0)
    heavy = nonneg & (mean > med * 1.5) & (mean > 0)
    Xc[:, heavy] = np.log1p(Xc[:, heavy])
    return StandardScaler().fit_transform(Xc), f"log1p({heavy.sum()} skewed cols)"


def transform_robust(X):
    return RobustScaler().fit_transform(X), "robust"


TRANSFORMS = {
    "standard": transform_none,
    "log1p_skewed": transform_log1p,
    "robust": transform_robust,
}

STAGE1_PATH = OUTPUTS_DIR / "clustering_search_stage1.csv"


def stage1() -> pd.DataFrame:
    rows = []
    sil_idx = rng.choice(len(Xraw), min(args.sil_n, len(Xraw)), replace=False)

    configs = []
    for tname in TRANSFORMS:
        for n_comp in [4, 6, 8, 10, 12, 18]:
            for k in [4, 5, 6, 7, 8]:
                configs.append((tname, n_comp, k))

    print(f"\n[{elapsed()}] Stage 1 — {len(configs)} configurations...")
    print(f"\n  {'transform':<16} {'dims':>5} {'k':>3} {'silhouette':>11} "
          f"{'DB':>8} {'CH':>10} {'min clust':>10}")
    print(f"  {'-' * 70}")

    for tname, n_comp, k in configs:
        Xs, _ = TRANSFORMS[tname](Xraw)
        if n_comp >= Xs.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=args.seed)
        Xp = pca.fit_transform(Xs)
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed, max_iter=300)
        lab = km.fit_predict(Xp)

        counts = np.bincount(lab, minlength=k)
        min_share = counts.min() / len(lab)

        sil = silhouette_score(Xp[sil_idx], lab[sil_idx])
        db = davies_bouldin_score(Xp, lab)
        ch = calinski_harabasz_score(Xp, lab)

        rows.append({
            "transform": tname, "n_components": n_comp, "k": k,
            "silhouette": round(float(sil), 4),
            "davies_bouldin": round(float(db), 4),
            "calinski_harabasz": round(float(ch), 1),
            "min_cluster_share": round(float(min_share), 5),
            "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        })
        flag = "  <- microcluster" if min_share < 0.01 else ""
        print(f"  {tname:<16} {n_comp:>5} {k:>3} {sil:>11.4f} {db:>8.3f} "
              f"{ch:>10,.0f} {min_share:>9.2%}{flag}")

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    df.to_csv(STAGE1_PATH, index=False)

    print(f"\n  {'=' * 68}")
    print("  TOP 10 BY SILHOUETTE")
    print(f"  {'=' * 68}")
    print(f"  {'transform':<16} {'dims':>5} {'k':>3} {'silhouette':>11} "
          f"{'DB':>8} {'min clust':>10} {'var':>7}")
    print(f"  {'-' * 68}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['transform']:<16} {int(r['n_components']):>5} {int(r['k']):>3} "
              f"{r['silhouette']:>11.4f} {r['davies_bouldin']:>8.3f} "
              f"{r['min_cluster_share']:>9.2%} {r['explained_variance']:>6.1%}")

    baseline = df[(df.transform == "standard") & (df.n_components == 18) & (df.k == 5)]
    if len(baseline):
        b = baseline.iloc[0]
        print(f"\n  Shipped configuration (standard / 18 dims / k=5): "
              f"silhouette {b['silhouette']:.4f}")
        best = df.iloc[0]
        print(f"  Best silhouette in sweep: {best['silhouette']:.4f} "
              f"({best['transform']} / {int(best['n_components'])} dims / k={int(best['k'])})")
        print(f"  Headroom: {best['silhouette'] - b['silhouette']:+.4f}")

    print("\n  Stage 1 measures GEOMETRY ONLY. A higher silhouette here is not")
    print("  an improvement until stage 2 shows downstream macro F1 survives.")
    print(f"  -> python src/14_clustering_search.py --stage 2 --top {args.top}")
    print(f"\n  {STAGE1_PATH.name} saved | {elapsed()}")
    return df


if args.stage == 1:
    stage1()
    print(f"\n{'=' * 70}")
    print(f"PHASE 14 STAGE 1 COMPLETE | {elapsed()}")
    print(f"{'=' * 70}")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — does the geometry gain survive contact with the graded metric?
# ══════════════════════════════════════════════════════════════════════════════
import xgboost as xgb  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from utils.velocity import add_velocity_features  # noqa: E402

if not STAGE1_PATH.exists():
    print("FATAL: run stage 1 first.")
    sys.exit(2)

stage1_df = pd.read_csv(STAGE1_PATH)


def _lookup(transform: str, dims: int, k: int) -> dict:
    row = stage1_df[(stage1_df["transform"] == transform)
                    & (stage1_df["n_components"] == dims)
                    & (stage1_df["k"] == k)]
    if not len(row):
        raise SystemExit(f"No stage-1 result for {transform}:{dims}:{k}")
    return row.iloc[0].to_dict()


if args.configs:
    candidates = []
    for spec in args.configs.split(","):
        t, d, k = spec.strip().split(":")
        candidates.append(_lookup(t, int(d), int(k)))
else:
    # Ranking by raw silhouette selects for microcluster artifacts: isolating
    # ~20 outliers into their own cluster inflates the score for everyone else.
    # 77 of 90 configurations in the sweep have a smallest cluster under 1%.
    viable = stage1_df[stage1_df["min_cluster_share"] >= 0.01]
    candidates = viable.head(args.top).to_dict("records")

# Always include the shipped configuration as the control.
if not any(c["transform"] == "standard" and c["n_components"] == 18 and c["k"] == 5
           for c in candidates):
    candidates.append(_lookup("standard", 18, 5))

print(f"\n[{elapsed()}] Stage 2 — retraining the transition model for "
      f"{len(candidates)} configuration(s)...")

MONTHS = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
# Stage 2 is a COMPARISON, not a production fit. Every candidate gets the same
# reduced budget, so the ranking is fair even though the absolute F1 sits below
# a full-data run. Retraining on all 4M rows per candidate took ~25 minutes each
# and answered the same question.
TRAIN_M = [2, 3, 4, 5, 6, 7, 8, 9][-args.stage2_months:]
VAL_M   = [10]
print(f"  stage-2 budget: train months {TRAIN_M}, "
      f"<= {args.stage2_rows:,} rows per candidate")

# Load every month's features once; each candidate re-labels them.
month_feats = {}
for d in MONTHS:
    fp = ROOT / "features" / f"features_{d.strftime('%Y_%m_%d')}.parquet"
    if fp.exists():
        month_feats[d] = pd.read_parquet(fp, engine="pyarrow")
print(f"  loaded {len(month_feats)} monthly feature files | {elapsed()}")

results = []
for cand in candidates:
    tname, n_comp, k = cand["transform"], int(cand["n_components"]), int(cand["k"])
    tag = f"{tname}/{n_comp}d/k={k}"
    print(f"\n  --- {tag} ---")

    # Refit the clustering on December, exactly as Phase 6 does.
    Xs, _ = TRANSFORMS[tname](fit_df.values)
    pca = PCA(n_components=n_comp, random_state=args.seed)
    Xp = pca.fit_transform(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed, max_iter=300)
    km.fit(Xp)
    centroids = km.cluster_centers_

    # Assign every month via nearest centroid (frozen, no refit).
    if tname == "standard":
        scaler = StandardScaler().fit(fit_df.values)
    elif tname == "robust":
        scaler = RobustScaler().fit(fit_df.values)
    else:
        scaler = None

    labels_by_month = {}
    for d, f in month_feats.items():
        Xm = f[FEAT_COLS].fillna(0).values
        if tname == "log1p_skewed":
            Xm_s, _ = TRANSFORMS[tname](Xm)
        else:
            Xm_s = scaler.transform(Xm)
        Xm_p = pca.transform(Xm_s)
        labels_by_month[d] = cdist(Xm_p, centroids).argmin(axis=1)

    # Build transition pairs against these labels.
    frames = []
    for i, d in enumerate(MONTHS[:-1]):
        d1 = MONTHS[i + 1]
        if d.month == 1 or d not in labels_by_month or d1 not in labels_by_month:
            continue
        if d.month not in TRAIN_M + VAL_M:
            continue
        f = add_velocity_features(month_feats[d])
        cols = FEAT_COLS + ["spend_velocity", "freq_velocity", "app_velocity",
                            "recency_risk", "engagement_score", "spend_decline_flag"]
        sub = f[cols].fillna(0).copy()
        sub["seg_curr"] = labels_by_month[d]
        sub["seg_next"] = labels_by_month[d1]
        sub["month_num"] = d.month
        sub["_split"] = "train" if d.month in TRAIN_M else "val"
        frames.append(sub)

    allp = pd.concat(frames, ignore_index=True)
    xcols = [c for c in allp.columns if c not in ("seg_next", "_split")]
    tr = allp[allp["_split"] == "train"]
    va = allp[allp["_split"] == "val"]

    if len(tr) > args.stage2_rows:
        tr = tr.sample(args.stage2_rows, random_state=args.seed)

    X_tr = tr[xcols].values.astype(np.float32)
    y_tr = tr["seg_next"].values.astype(int)
    X_va = va[xcols].values.astype(np.float32)
    y_va = va["seg_next"].values.astype(int)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        objective="multi:softprob", num_class=k, eval_metric="mlogloss",
        early_stopping_rounds=25, n_jobs=-1, random_state=args.seed, verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr),
              eval_set=[(X_va, y_va)], verbose=False)
    val_f1 = f1_score(y_va, model.predict(X_va), average="macro", zero_division=0)

    print(f"    silhouette {cand['silhouette']:.4f}   val macro F1 {val_f1:.4f}   "
          f"(k={k}, {len(y_tr):,} train rows)")
    results.append({
        "config": tag, "transform": tname, "n_components": n_comp, "k": k,
        "silhouette": float(cand["silhouette"]),
        "davies_bouldin": float(cand["davies_bouldin"]),
        "val_macro_f1": float(val_f1),
        "min_cluster_share": float(cand["min_cluster_share"]),
    })

res = pd.DataFrame(results).sort_values("silhouette", ascending=False)
res.to_csv(OUTPUTS_DIR / "clustering_search_stage2.csv", index=False)

print(f"\n{'=' * 74}")
print("SILHOUETTE vs DOWNSTREAM MACRO F1")
print(f"{'=' * 74}")
print(f"  {'configuration':<26} {'silhouette':>11} {'val macro F1':>13} {'min clust':>10}")
print(f"  {'-' * 66}")
for _, r in res.iterrows():
    mark = "  <- shipped" if r["config"] == "standard/18d/k=5" else ""
    print(f"  {r['config']:<26} {r['silhouette']:>11.4f} "
          f"{r['val_macro_f1']:>13.4f} {r['min_cluster_share']:>9.2%}{mark}")
print(f"  {'-' * 66}")
print("\n  NOTE: macro F1 is not directly comparable across different k — a")
print("  5-class and an 8-class problem have different difficulty. Compare")
print("  within the same k, and treat cross-k rows as indicative only.")
print(f"\n  clustering_search_stage2.csv saved | {elapsed()}")

(OUTPUTS_DIR / "clustering_search.json").write_text(
    json.dumps({"stage2": results}, indent=2), encoding="utf-8")

print(f"\n{'=' * 70}")
print(f"PHASE 14 STAGE 2 COMPLETE | {elapsed()}")
print(f"{'=' * 70}")
