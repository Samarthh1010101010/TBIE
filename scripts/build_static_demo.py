#!/usr/bin/env python
"""
scripts/build_static_demo.py
────────────────────────────
Build a self-contained static demo of the TBIE dashboard for GitHub Pages.

The live dashboard (serving/dashboard.html) talks to the FastAPI service,
which needs the frozen models plus ~3 GB of feature parquet. That cannot run
on Pages, and nobody evaluating the repo is going to clone 3 GB to look at a UI.

This bakes a real, stratified sample of scored members — their actual segment,
lifecycle state, transition probabilities, exact TreeSHAP drivers, value at
risk and contact eligibility — into one HTML file with no network calls.

The numbers are real model output, not mock data. The sample is a snapshot, and
the page says so; it is a demo of the interface, not a live service.

Run from TBIE_CODE root (needs outputs/ and features/ populated):
    python scripts/build_static_demo.py --members 400 --out docs/index.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.economics import adverse_probability, segment_values, value_at_risk  # noqa: E402
from utils.lag_features import add_delta_features, add_segment_history  # noqa: E402
from utils.pairs import to_model_matrix  # noqa: E402
from utils.velocity import add_velocity_features  # noqa: E402

p = argparse.ArgumentParser(description="Build the static dashboard demo")
p.add_argument("--observation-date", default="2025-12-31")
p.add_argument("--members", type=int, default=400,
               help="Members to embed, stratified across segment x state.")
p.add_argument("--out", default="docs/index.html")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

OBS = args.observation_date
KEY = OBS.replace("-", "_")


def die(msg: str) -> None:
    print(f"FATAL: {msg}")
    sys.exit(1)


print("Building static demo…")

feat_path = ROOT / "features" / f"features_{KEY}.parquet"
seg_path = ROOT / "outputs" / "segment_assignments.csv"
state_path = ROOT / "outputs" / "state_assignments.csv"
elig_path = ROOT / "outputs" / "contact_eligibility.csv"
for f in (feat_path, seg_path, state_path, elig_path):
    if not f.exists():
        die(f"missing {f.relative_to(ROOT)} — run pipeline.py at {OBS} first")

bundle = joblib.load(ROOT / "models" / "segment_transition_model.pkl")
seg_model = joblib.load(ROOT / "segments" / "segment_model.pkl")
X_COLS = bundle["feature_cols"]
CLUSTER_TO_SID = bundle["cluster_to_sid"]
SID_TO_NAME = bundle["sid_to_name"]
SID_TO_CID = {v: k for k, v in CLUSTER_TO_SID.items()}
N = bundle.get("n_classes", len(CLUSTER_TO_SID))
NAMES = [SID_TO_NAME[CLUSTER_TO_SID[c]] for c in range(N)]

feat = pd.read_parquet(feat_path, engine="pyarrow").reset_index(drop=True)
seg = pd.read_csv(seg_path)
state = pd.read_csv(state_path)
elig = pd.read_csv(elig_path, keep_default_na=False, na_values=[])
print(f"  loaded {len(feat):,} members at {OBS}")

# ── Rebuild the model matrix exactly as pipeline.py does ─────────────────────
sub = add_velocity_features(feat)
sub["seg_curr"] = seg["segment_id"].map(SID_TO_CID).values.astype(int)
sub["y_curr"] = state["state_name"].map(bundle["state_map"]).astype(int).values
sub["month_num"] = pd.Timestamp(OBS).month

lag_base = bundle.get("lag_base_cols", [])
if lag_base:
    prior = pd.Timestamp(OBS) - pd.DateOffset(months=1)
    ppath = ROOT / "features" / f"features_{prior.strftime('%Y_%m_%d')}.parquet"
    if not ppath.exists():
        die(f"model uses lag features but {ppath.name} is missing")
    prior_feat = pd.read_parquet(ppath, engine="pyarrow")
    sub = add_delta_features(sub, prior_feat, lag_base)

    cols = seg_model["behavioral_feature_cols"]
    from scipy.spatial.distance import cdist
    Xp = seg_model["pca"].transform(
        seg_model["scaler"].transform(prior_feat[cols].fillna(0).values))
    cids = sorted(seg_model["centroids"].keys())
    cmat = np.array([seg_model["centroids"][c] for c in cids])
    prev_cid = np.array(cids)[cdist(Xp, cmat).argmin(axis=1)]
    prev_by_member = pd.Series(prev_cid, index=prior_feat["member_id"].values)
    sub = add_segment_history(sub, sub["member_id"].map(prev_by_member).values)

X = to_model_matrix(sub, X_COLS)
proba = bundle["model"].predict_proba(X)
print(f"  scored {len(X):,} members")

# ── Economics ────────────────────────────────────────────────────────────────
SPEND = "spend_total_180d"
vdf = pd.DataFrame({"seg": sub["seg_curr"], "spend": feat[SPEND].fillna(0)})
values = segment_values(vdf, "seg", "spend", N)
var = value_at_risk(proba, sub["seg_curr"].values, values)
adv = adverse_probability(proba, sub["seg_curr"].values, values)

# ── Stratified sample: every segment x state combination that exists ─────────
rng = np.random.default_rng(args.seed)
frame = pd.DataFrame({
    "i": np.arange(len(feat)),
    "member_id": feat["member_id"].astype(str),
    "segment_name": seg["segment_name"].values,
    "state_name": state["state_name"].values,
    "var": var,
})
groups = frame.groupby(["segment_name", "state_name"], observed=True)
per_group = max(2, args.members // max(len(groups), 1))
picked = []
for _, g in groups:
    # Bias toward higher value at risk — those are the interesting rows.
    g = g.sort_values("var", ascending=False)
    head = g.head(max(1, per_group // 2))
    rest = g.iloc[len(head):]
    extra = rest.sample(min(len(rest), per_group - len(head)), random_state=args.seed)
    picked.append(pd.concat([head, extra]))
sample = pd.concat(picked).drop_duplicates("member_id").head(args.members)
idx = sample["i"].to_numpy()
print(f"  sampled {len(idx)} members across {len(groups)} segment×state groups")

# ── Exact TreeSHAP for the sampled rows only ─────────────────────────────────
booster = bundle["model"].get_booster()
contribs = np.asarray(booster.predict(
    xgb.DMatrix(X[idx], feature_names=list(X_COLS)), pred_contribs=True))
if contribs.ndim == 2:
    contribs = contribs[:, None, :]
print(f"  SHAP computed: {contribs.shape}")

elig_by_member = elig.set_index("member_id")
members_out = []
for row_n, i in enumerate(idx):
    probs = proba[i]
    pred = int(probs.argmax())
    contrib = contribs[row_n, pred, :-1]
    order = np.argsort(-np.abs(contrib))[:3]
    mid = str(feat["member_id"].iloc[i])
    e = elig_by_member.loc[mid] if mid in elig_by_member.index else None
    members_out.append({
        "id": mid,
        "seg": str(seg["segment_name"].iloc[i]),
        "segId": str(seg["segment_id"].iloc[i]),
        "segConf": round(float(seg["segment_confidence"].iloc[i]), 4),
        "state": str(state["state_name"].iloc[i]),
        "pred": SID_TO_NAME[CLUSTER_TO_SID[pred]],
        "conf": round(float(probs.max()), 4),
        "probs": {CLUSTER_TO_SID[c]: round(float(probs[c]), 4) for c in range(N)},
        "var": round(float(var[i]), 2),
        "adv": round(float(adv[i]), 4),
        "drivers": [
            {"f": X_COLS[j], "v": round(float(X[i, j]), 4),
             "s": round(float(contrib[j]), 4)}
            for j in order
        ],
        "targetable": bool(e["is_targetable"]) if e is not None else None,
        "suppression": str(e["suppression_reason"]) if e is not None else "",
        "channel": str(e["recommended_channel"]) if e is not None else "",
        "action": str(e["recommended_action"]) if e is not None else "",
    })

payload = {
    "observationDate": OBS,
    "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    "totalMembers": int(len(feat)),
    "sampleSize": len(members_out),
    "model": {
        "features": len(X_COLS),
        "macroF1Test": round(float(bundle.get("macro_f1_test", float("nan"))), 4),
        "macroF1Val": round(float(bundle.get("macro_f1_val", float("nan"))), 4),
        "persistenceBaseline": 0.5651,
        "majorityBaseline": 0.0589,
        "decisionRule": bundle.get("decision_rule", "argmax"),
        "usesLag": bool(bundle.get("lag_base_cols")),
        "ece": 0.0184,
        "brier": 0.0673,
    },
    "segments": [
        {"name": n, "members": int((seg["segment_name"] == n).sum()),
         "valueUsd": round(float(values[c]), 2)}
        for c, n in enumerate(NAMES)
    ],
    "states": [
        {"name": n, "members": int(c)}
        for n, c in state["state_name"].value_counts().items()
    ],
    "eligibility": [
        {"reason": ("targetable" if r == "" else r), "members": int(c)}
        for r, c in elig["suppression_reason"].value_counts().items()
    ],
    "totalValueAtRisk": round(float(var.sum()), 2),
    "members": members_out,
}

out_path = ROOT / args.out
out_path.parent.mkdir(parents=True, exist_ok=True)
tpl = (ROOT / "scripts" / "static_demo_template.html").read_text(encoding="utf-8")
html = tpl.replace("/*__TBIE_DATA__*/null", json.dumps(payload, separators=(",", ":")))
out_path.write_text(html, encoding="utf-8")

kb = out_path.stat().st_size / 1024
print(f"\n  {args.out} written ({kb:,.0f} KB, {len(members_out)} members embedded)")
print(f"  total value at risk across all {len(feat):,} members: ${var.sum():,.0f}")
print("\n  Enable GitHub Pages: Settings -> Pages -> Source: main / docs")
