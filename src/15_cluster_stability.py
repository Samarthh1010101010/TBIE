"""
Phase 15 — Cluster stability (Adjusted Rand Index).

Why this exists
---------------
Phase 8 predicts the segment a member will occupy next month. Those segment
labels are not observed — they are produced by the K-Means fit in Phase 6. So
the model is trained to predict the output of another model, and a reviewer is
entitled to ask whether that target is a stable structure or an artefact of one
particular fit.

Adjusted Rand Index answers it. ARI compares two partitions of the same points,
is invariant to label permutation, and is corrected for chance: 1.0 identical,
~0.0 no better than random agreement.

Three questions, three different runs
-------------------------------------
1. SEED       Refit on identical data with different random_state.
              Low ARI here means the partition is not even reproducible.
2. SUBSAMPLE  Refit on overlapping random subsamples, compare on shared members.
              Low ARI means the structure depends on which members you happened
              to draw.
3. TEMPORAL   Refit independently on each monthly snapshot, compare consecutive
              months on members present in both. Low ARI means the segment
              structure itself moves over time, which would undermine training a
              transition model on it.

Preprocessing is copied from src/06_segment_discovery.py so the numbers describe
the shipped clustering and not an approximation of it:
    behavioral cols from df.attrs, minus email_open_rate_30d
    fit_mask = feature_complete == 1 AND purchase_count_180d > 0
    fillna(0) -> StandardScaler -> PCA(n_components=0.85) -> KMeans(k=5, n_init=10)

Usage
-----
    python src/15_cluster_stability.py --features-dir C:\\TBIE-data\\TBIE_CODE\\features
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

K = 5
PCA_VAR = 0.85
N_INIT = 10
MAX_ITER = 500
DROP_COL = "email_open_rate_30d"

_t0 = time.time()


def elapsed() -> str:
    return f"{time.time() - _t0:6.1f}s"


def load_fit_matrix(path: Path):
    """Return (X_pca, member_ids, n_components) using Phase 6's exact recipe."""
    df = pd.read_parquet(path, engine="pyarrow")

    cols = df.attrs.get("behavioral_feature_cols", df.attrs.get("ml_feature_cols", []))
    cols = [c for c in cols if c != DROP_COL]
    if not cols:
        raise RuntimeError(f"no behavioral cols in attrs for {path.name}")

    mask = (df["feature_complete"] == 1) & (df["purchase_count_180d"] > 0)
    fit = df.loc[mask, cols].fillna(0)
    ids = df.loc[mask, "member_id"].to_numpy()

    X = StandardScaler().fit_transform(fit.to_numpy())
    pca = PCA(n_components=PCA_VAR, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, ids, pca.n_components_


def fit_labels(X, seed: int):
    return KMeans(n_clusters=K, n_init=N_INIT, random_state=seed,
                  max_iter=MAX_ITER).fit_predict(X)


def summarise(name, values):
    a = np.asarray(values, dtype=float)
    return {
        "comparison": name,
        "n_pairs": int(a.size),
        "mean_ari": round(float(a.mean()), 4),
        "min_ari": round(float(a.min()), 4),
        "max_ari": round(float(a.max()), 4),
        "std_ari": round(float(a.std()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--anchor", default="features_2025_12_01.parquet",
                    help="snapshot Phase 6 fits on")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 1, 7, 123, 2024, 31337])
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    ap.add_argument("--subsample-runs", type=int, default=6)
    ap.add_argument("--out", default="outputs/cluster_stability.json")
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    anchor = fdir / args.anchor
    report = {"config": {"k": K, "pca_variance": PCA_VAR, "n_init": N_INIT,
                         "anchor": args.anchor, "seeds": args.seeds,
                         "subsample_frac": args.subsample_frac}}

    # ---------- 1. SEED ----------
    print(f"[{elapsed()}] loading anchor snapshot {anchor.name} ...")
    X, ids, ncomp = load_fit_matrix(anchor)
    print(f"[{elapsed()}] fit population {X.shape[0]:,} x {X.shape[1]} "
          f"(PCA kept {ncomp} components)")
    report["config"]["fit_population"] = int(X.shape[0])
    report["config"]["pca_components"] = int(ncomp)

    print(f"[{elapsed()}] SEED: {len(args.seeds)} refits on identical data ...")
    seed_labels = {}
    for s in args.seeds:
        seed_labels[s] = fit_labels(X, s)
        print(f"[{elapsed()}]   seed {s:>6} done")

    seed_pairs = []
    seed_detail = []
    for a, b in itertools.combinations(args.seeds, 2):
        v = adjusted_rand_score(seed_labels[a], seed_labels[b])
        seed_pairs.append(v)
        seed_detail.append({"a": a, "b": b, "ari": round(float(v), 4)})
    report["seed"] = summarise("seed refit, identical data", seed_pairs)
    report["seed"]["pairs"] = seed_detail

    # ---------- 2. SUBSAMPLE ----------
    print(f"[{elapsed()}] SUBSAMPLE: {args.subsample_runs} refits on "
          f"{args.subsample_frac:.0%} draws ...")
    rng = np.random.default_rng(42)
    n = X.shape[0]
    take = int(n * args.subsample_frac)
    sub = []
    for r in range(args.subsample_runs):
        idx = rng.choice(n, size=take, replace=False)
        sub.append((idx, fit_labels(X[idx], 42)))
        print(f"[{elapsed()}]   draw {r + 1}/{args.subsample_runs} done")

    sub_pairs = []
    for (ia, la), (ib, lb) in itertools.combinations(sub, 2):
        # compare only on members drawn by both runs
        pa = pd.Series(la, index=ia)
        pb = pd.Series(lb, index=ib)
        common = pa.index.intersection(pb.index)
        if len(common) > 1000:
            sub_pairs.append(adjusted_rand_score(pa.loc[common], pb.loc[common]))
    report["subsample"] = summarise(
        f"{args.subsample_frac:.0%} subsample refit, shared members", sub_pairs)

    # ---------- 3. TEMPORAL ----------
    monthly = sorted(p for p in fdir.glob("features_2025_*_01.parquet"))
    print(f"[{elapsed()}] TEMPORAL: independent refit on {len(monthly)} monthly "
          f"snapshots ...")
    month_labels = {}
    for p in monthly:
        try:
            Xm, idm, _ = load_fit_matrix(p)
        except Exception as exc:               # cold-start months can be unusable
            print(f"[{elapsed()}]   {p.stem}: SKIPPED ({exc})")
            continue
        month_labels[p.stem] = pd.Series(fit_labels(Xm, 42), index=idm)
        print(f"[{elapsed()}]   {p.stem}: {Xm.shape[0]:,} members")

    keys = sorted(month_labels)
    consec, consec_detail = [], []
    for a, b in zip(keys, keys[1:]):
        pa, pb = month_labels[a], month_labels[b]
        common = pa.index.intersection(pb.index)
        if len(common) < 1000:
            continue
        v = adjusted_rand_score(pa.loc[common], pb.loc[common])
        consec.append(v)
        consec_detail.append({"from": a[-10:], "to": b[-10:],
                              "shared_members": int(len(common)),
                              "ari": round(float(v), 4)})
    if consec:
        report["temporal_consecutive"] = summarise(
            "independent refit, consecutive months", consec)
        report["temporal_consecutive"]["pairs"] = consec_detail

    # ---------- 3b. TEMPORAL, FROZEN REPRESENTATION ----------
    # The run above refits StandardScaler and PCA per month, so each month's
    # labels live in a different feature space. That conflates "the clustering
    # is unstable" with "the representation changed". Here the December scaler
    # and PCA are frozen and only K-Means is refit, which isolates the first.
    print(f"[{elapsed()}] TEMPORAL (frozen Dec scaler+PCA): refitting KMeans only ...")
    df_anchor = pd.read_parquet(anchor, engine="pyarrow")
    acols = df_anchor.attrs.get("behavioral_feature_cols",
                                df_anchor.attrs.get("ml_feature_cols", []))
    acols = [c for c in acols if c != DROP_COL]
    amask = (df_anchor["feature_complete"] == 1) & (df_anchor["purchase_count_180d"] > 0)
    frozen_scaler = StandardScaler().fit(df_anchor.loc[amask, acols].fillna(0).to_numpy())
    frozen_pca = PCA(n_components=PCA_VAR, random_state=42).fit(
        frozen_scaler.transform(df_anchor.loc[amask, acols].fillna(0).to_numpy()))

    frozen_labels = {}
    for p in monthly:
        dfm = pd.read_parquet(p, engine="pyarrow")
        mm = (dfm["feature_complete"] == 1) & (dfm["purchase_count_180d"] > 0)
        if mm.sum() < 1000:
            continue
        Xm = frozen_pca.transform(
            frozen_scaler.transform(dfm.loc[mm, acols].fillna(0).to_numpy()))
        frozen_labels[p.stem] = pd.Series(fit_labels(Xm, 42),
                                          index=dfm.loc[mm, "member_id"].to_numpy())
        print(f"[{elapsed()}]   {p.stem}: {Xm.shape[0]:,} members")

    fkeys = sorted(frozen_labels)
    fz, fz_detail = [], []
    for a, b in zip(fkeys, fkeys[1:]):
        pa, pb = frozen_labels[a], frozen_labels[b]
        common = pa.index.intersection(pb.index)
        if len(common) < 1000:
            continue
        v = adjusted_rand_score(pa.loc[common], pb.loc[common])
        fz.append(v)
        fz_detail.append({"from": a[-10:], "to": b[-10:], "ari": round(float(v), 4)})
    if fz:
        report["temporal_frozen_transform"] = summarise(
            "frozen Dec scaler+PCA, KMeans refit per month", fz)
        report["temporal_frozen_transform"]["pairs"] = fz_detail

    # anchor month vs every other month
    anchor_key = anchor.stem
    if anchor_key in month_labels:
        vs, vs_detail = [], []
        for m in keys:
            if m == anchor_key:
                continue
            pa, pb = month_labels[anchor_key], month_labels[m]
            common = pa.index.intersection(pb.index)
            if len(common) < 1000:
                continue
            v = adjusted_rand_score(pa.loc[common], pb.loc[common])
            vs.append(v)
            vs_detail.append({"month": m[-10:], "ari": round(float(v), 4)})
        if vs:
            report["temporal_vs_anchor"] = summarise(
                "independent refit, each month vs anchor", vs)
            report["temporal_vs_anchor"]["pairs"] = vs_detail

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("CLUSTER STABILITY — Adjusted Rand Index")
    print("=" * 62)
    for key in ("seed", "subsample", "temporal_consecutive", "temporal_frozen_transform", "temporal_vs_anchor"):
        if key in report:
            r = report[key]
            print(f"  {r['comparison']:<46}")
            print(f"    mean {r['mean_ari']:.4f} | min {r['min_ari']:.4f} "
                  f"| max {r['max_ari']:.4f} | n={r['n_pairs']}")
    print(f"\n  written to {out}")
    print(f"  total {elapsed()}")


if __name__ == "__main__":
    main()
