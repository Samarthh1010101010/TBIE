"""
serving/api.py
══════════════════════════════════════════════════════════════════════════════
TBIE Inference API

Turns the batch pipeline into a queryable service. The batch path answers
"score all 500,000 members overnight"; this answers "why is THIS member being
targeted, right now", which is the question a campaign manager actually asks.

Endpoints
─────────
  GET  /health                  liveness + what is loaded
  GET  /model                   model card summary: features, metrics, config
  GET  /segments                the five segments with sizes and descriptions
  GET  /states                  the ten lifecycle states and their rules
  GET  /members/{member_id}     full profile: segment, state, transition
                                probabilities, SHAP drivers, recommended action
  POST /predict                 score a raw feature payload (no stored member)
  GET  /cohort                  members matching segment/state filters, ranked
                                by value at risk

Design notes
────────────
Models load once at startup, never per request. Feature rows are served from
the precomputed snapshot for the configured observation date — this is an
inference service over a batch feature store, not a real-time feature pipeline,
and it says so rather than pretending otherwise.

Run:
    pip install fastapi uvicorn
    python -m uvicorn serving.api:app --reload --port 8000
    open http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.economics import (  # noqa: E402
    adverse_probability,
    segment_values,
    value_at_risk,
)
from utils.lag_features import add_delta_features, add_segment_history  # noqa: E402
from utils.pairs import to_model_matrix  # noqa: E402
from utils.state_rules import STATE_IDS, classify_states  # noqa: E402
from utils.velocity import add_velocity_features  # noqa: E402

OBS_DATE = os.environ.get("TBIE_OBSERVATION_DATE", "2025-12-31")

app = FastAPI(
    title="TBIE — Temporal Behavioural Intelligence Engine",
    description=(
        "Segment, lifecycle state and 30-day transition prediction for loyalty "
        "programme members, with SHAP attributions for every prediction."
    ),
    version="2.0.0",
)


# ── State loaded once at startup ─────────────────────────────────────────────
class Registry:
    ready: bool = False
    error: str | None = None
    features: pd.DataFrame | None = None
    prior_features: pd.DataFrame | None = None
    seg_model: dict | None = None
    bundle: dict | None = None
    explainer: Any = None
    segment_defs: dict | None = None
    state_defs: dict | None = None
    member_index: dict[str, int] | None = None
    X: np.ndarray | None = None
    proba: np.ndarray | None = None
    seg_ids: np.ndarray | None = None
    seg_names: np.ndarray | None = None
    seg_conf: np.ndarray | None = None
    states: np.ndarray | None = None
    seg_cluster: np.ndarray | None = None
    segment_value: np.ndarray | None = None
    var: np.ndarray | None = None
    adverse: np.ndarray | None = None


R = Registry()


def _date_key(d: str) -> str:
    return d.replace("-", "_")


def _assign_segments(feat: pd.DataFrame, seg_model: dict, cluster_to_sid, sid_to_name):
    from scipy.spatial.distance import cdist

    cols = seg_model["behavioral_feature_cols"]
    X = seg_model["pca"].transform(
        seg_model["scaler"].transform(feat[cols].fillna(0).values)
    )
    cids = sorted(seg_model["centroids"].keys())
    cmat = np.array([seg_model["centroids"][c] for c in cids])
    d = cdist(X, cmat)
    nearest = d.argmin(axis=1)
    nd = d.min(axis=1)
    d2 = d.copy()
    d2[np.arange(len(d)), nearest] = np.inf
    sd = d2.min(axis=1)
    cluster_ids = np.array(cids)[nearest]
    sids = np.array([cluster_to_sid[c] for c in cluster_ids])
    names = np.array([sid_to_name[s] for s in sids])
    conf = ((sd - nd) / (sd + nd + 1e-9)).clip(0, 1)
    return sids, names, conf.round(4), cluster_ids


@app.on_event("startup")
def load_everything() -> None:
    try:
        feat_path = ROOT / "features" / f"features_{_date_key(OBS_DATE)}.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"No feature snapshot for {OBS_DATE} at {feat_path}. "
                f"Run pipeline.py for that date, or set TBIE_OBSERVATION_DATE."
            )

        R.seg_model = joblib.load(ROOT / "segments" / "segment_model.pkl")
        R.bundle = joblib.load(ROOT / "models" / "segment_transition_model.pkl")

        with open(ROOT / "segments" / "segment_definitions.json", encoding="utf-8") as f:
            R.segment_defs = json.load(f)
        state_defs_path = ROOT / "models" / "state_definitions.json"
        if state_defs_path.exists():
            with open(state_defs_path, encoding="utf-8") as f:
                R.state_defs = json.load(f)

        feat = pd.read_parquet(feat_path, engine="pyarrow").reset_index(drop=True)
        R.features = feat

        cluster_to_sid = R.bundle["cluster_to_sid"]
        sid_to_name = R.bundle["sid_to_name"]
        state_map = R.bundle["state_map"]
        if state_map != STATE_IDS:
            raise RuntimeError(
                "Bundle state_map disagrees with state_rules.STATE_IDS — refusing "
                "to serve, y_curr would be encoded inconsistently."
            )

        sids, names, conf, cluster_ids = _assign_segments(
            feat, R.seg_model, cluster_to_sid, sid_to_name
        )
        R.seg_ids, R.seg_names, R.seg_conf = sids, names, conf
        R.states = classify_states(feat)

        # Build the model matrix exactly as pipeline.py does.
        X_COLS = R.bundle["feature_cols"]
        lag_base = R.bundle.get("lag_base_cols", [])
        sub = add_velocity_features(feat)
        sub["seg_curr"] = cluster_ids.astype(int)
        sub["y_curr"] = pd.Series(R.states).map(state_map).astype(int).values
        sub["month_num"] = pd.Timestamp(OBS_DATE).month

        if lag_base:
            prior = pd.Timestamp(OBS_DATE) - pd.DateOffset(months=1)
            ppath = ROOT / "features" / f"features_{prior.strftime('%Y_%m_%d')}.parquet"
            if not ppath.exists():
                raise FileNotFoundError(
                    f"Model uses lag features but the prior snapshot {ppath.name} "
                    f"is missing."
                )
            prior_feat = pd.read_parquet(ppath, engine="pyarrow")
            R.prior_features = prior_feat
            sub = add_delta_features(sub, prior_feat, lag_base)
            p_sids, _, _, p_cids = _assign_segments(
                prior_feat, R.seg_model, cluster_to_sid, sid_to_name
            )
            prev_by_member = pd.Series(p_cids, index=prior_feat["member_id"].values)
            sub = add_segment_history(
                sub, sub["member_id"].map(prev_by_member).values
            )

        # Same construction as training and pipeline.py: raises on an absent
        # feature, fills NaN with 0 so rows take the tree branches they did at
        # fit time.
        R.X = to_model_matrix(sub, X_COLS)
        R.proba = R.bundle["model"].predict_proba(R.X)
        R.member_index = {m: i for i, m in enumerate(feat["member_id"].astype(str))}
        R.seg_cluster = cluster_ids.astype(int)

        # Value each segment from observed spend in this snapshot, then price
        # each member's downside. Same functions the offline cost-threshold
        # analysis uses, so the API and the campaign maths agree.
        spend_col = "spend_total_180d"
        if spend_col in feat.columns:
            vdf = pd.DataFrame({"seg": R.seg_cluster, "spend": feat[spend_col].fillna(0)})
            R.segment_value = segment_values(vdf, "seg", "spend", len(cluster_to_sid))
            R.var = value_at_risk(R.proba, R.seg_cluster, R.segment_value)
            R.adverse = adverse_probability(R.proba, R.seg_cluster, R.segment_value)
            print(f"  value at risk: ${R.var.sum():,.0f} across {len(feat):,} members")

        # xgboost implements TreeSHAP natively via pred_contribs — no `shap`
        # dependency, and it works with 3.x multiclass models (the shap
        # package's loader does not).
        R.explainer = R.bundle["model"].get_booster()

        R.ready = True
        print(f"TBIE API ready — {len(feat):,} members at {OBS_DATE}")
    except Exception as exc:
        R.error = str(exc)
        R.ready = False
        print(f"TBIE API failed to start: {exc}")


def _require_ready() -> None:
    if not R.ready:
        raise HTTPException(503, detail=f"Service not ready: {R.error or 'loading'}")


# ── Schemas ──────────────────────────────────────────────────────────────────
class Driver(BaseModel):
    feature: str
    value: float
    shap: float = Field(..., description="Signed contribution, log-odds")
    direction: str


class MemberResponse(BaseModel):
    member_id: str
    observation_date: str
    segment_id: str
    segment_name: str
    segment_confidence: float
    lifecycle_state: str
    predicted_segment_id: str
    predicted_segment_name: str
    prediction_confidence: float
    transition_probabilities: dict[str, float]
    value_at_risk_usd: float | None = Field(
        None, description="Expected spend lost to a move into a lower-value segment"
    )
    adverse_probability: float | None = Field(
        None, description="Total probability of landing in any lower-value segment"
    )
    top_drivers: list[Driver] = []
    recommended_action: str | None = None


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature name -> value. Missing trained features are rejected."
    )


ACTIVATION = {
    "Lapse Risk": "Email + SMS win-back with bonus points, within 7 days",
    "Momentum Builder": "App push: tier upgrade progress, immediate",
    "Silent Accumulator": "App push: unspent points reminder, weekly",
    "Program Skeptic": "Email: re-permission with value proof, monthly",
    "Win-Back Target": "Email + SMS reactivation bonus, immediate",
    "Brand Advocate": "App: early access + referral bonus, immediate",
    "Value Maximizer": "Email: cross-category multiplier, mid-month",
    "Redemption Hunter": "Email + push: targeted flash promotion",
    "Plateau Cruiser": "Email: curated recommendation, monthly",
    "New & Uncertain": "App onboarding: first-purchase bonus, within 7 days",
}


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if R.ready else "unavailable",
        "error": R.error,
        "observation_date": OBS_DATE,
        "members_loaded": 0 if R.features is None else len(R.features),
        "shap_available": R.explainer is not None,
    }


@app.get("/model")
def model_info() -> dict:
    _require_ready()
    b = R.bundle
    return {
        "observation_date": OBS_DATE,
        "n_features": len(b["feature_cols"]),
        "uses_lag_features": bool(b.get("lag_base_cols")),
        "n_classes": b.get("n_classes"),
        "segments": b["sid_to_name"],
        "macro_f1_test": b.get("macro_f1_test"),
        "macro_f1_val": b.get("macro_f1_val"),
        "decision_rule": b.get("decision_rule"),
        "weighting": b.get("weighting"),
        "random_seed": b.get("random_seed"),
        "note": (
            "Macro F1 is measured on the Nov->Dec walk-forward test window. "
            "Compare it against the persistence baseline in MODEL_CARD.md, not "
            "against zero."
        ),
    }


@app.get("/segments")
def segments() -> list[dict]:
    _require_ready()
    counts = pd.Series(R.seg_names).value_counts()
    out = []
    for sid, name in R.bundle["sid_to_name"].items():
        out.append({
            "segment_id": sid,
            "segment_name": name,
            "members": int(counts.get(name, 0)),
            "share": float(counts.get(name, 0) / len(R.seg_names)),
        })
    return sorted(out, key=lambda r: r["segment_id"])


@app.get("/states")
def states() -> list[dict]:
    _require_ready()
    counts = pd.Series(R.states).value_counts()
    out = []
    for name, sid in sorted(STATE_IDS.items(), key=lambda kv: kv[1]):
        entry = {
            "state_id": sid,
            "state_name": name,
            "members": int(counts.get(name, 0)),
            "share": float(counts.get(name, 0) / len(R.states)),
            "recommended_action": ACTIVATION.get(name),
        }
        if R.state_defs and name in R.state_defs:
            entry["meaning"] = R.state_defs[name].get("meaning")
            entry["key_signals"] = R.state_defs[name].get("key_signals")
        out.append(entry)
    return out


def _drivers_for(row_idx: int, class_idx: int, top_k: int) -> list[Driver]:
    """Exact per-member TreeSHAP contributions for the predicted class."""
    if R.explainer is None or top_k == 0:
        return []
    import xgboost as xgb

    cols = R.bundle["feature_cols"]
    x = R.X[row_idx : row_idx + 1]
    contribs = np.asarray(
        R.explainer.predict(xgb.DMatrix(x, feature_names=list(cols)),
                            pred_contribs=True)
    )
    # (1, classes, features + 1) for multiclass; last column is the bias term.
    if contribs.ndim == 2:
        contribs = contribs[:, None, :]
    contrib = contribs[0, class_idx, :-1]
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [
        Driver(
            feature=cols[j],
            value=float(x[0, j]),
            shap=float(contrib[j]),
            direction="raises" if contrib[j] > 0 else "lowers",
        )
        for j in order
    ]


@app.get("/members/{member_id}", response_model=MemberResponse)
def member(member_id: str, top_k: int = Query(3, ge=0, le=10)) -> MemberResponse:
    _require_ready()
    i = R.member_index.get(str(member_id))
    if i is None:
        raise HTTPException(404, detail=f"Unknown member_id: {member_id}")

    cluster_to_sid = R.bundle["cluster_to_sid"]
    sid_to_name = R.bundle["sid_to_name"]
    probs = R.proba[i]
    pred = int(probs.argmax())
    pred_sid = cluster_to_sid[pred]
    state = str(R.states[i])

    return MemberResponse(
        member_id=str(member_id),
        observation_date=OBS_DATE,
        segment_id=str(R.seg_ids[i]),
        segment_name=str(R.seg_names[i]),
        segment_confidence=float(R.seg_conf[i]),
        lifecycle_state=state,
        predicted_segment_id=pred_sid,
        predicted_segment_name=sid_to_name[pred_sid],
        prediction_confidence=float(probs.max()),
        transition_probabilities={
            f"prob_{cluster_to_sid[c]}": float(round(probs[c], 4))
            for c in range(len(probs))
        },
        value_at_risk_usd=(None if R.var is None else round(float(R.var[i]), 2)),
        adverse_probability=(None if R.adverse is None else round(float(R.adverse[i]), 4)),
        top_drivers=_drivers_for(i, pred, top_k),
        recommended_action=ACTIVATION.get(state),
    )


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    _require_ready()
    cols = R.bundle["feature_cols"]
    missing = [c for c in cols if c not in req.features]
    if missing:
        raise HTTPException(
            422,
            detail={
                "error": "missing required features",
                "n_missing": len(missing),
                "missing": missing[:20],
                "hint": "GET /model lists the full feature contract.",
            },
        )
    x = np.array([[float(req.features[c]) for c in cols]], dtype=np.float32)
    probs = R.bundle["model"].predict_proba(x)[0]
    cluster_to_sid = R.bundle["cluster_to_sid"]
    pred = int(probs.argmax())
    return {
        "predicted_segment_id": cluster_to_sid[pred],
        "predicted_segment_name": R.bundle["sid_to_name"][cluster_to_sid[pred]],
        "prediction_confidence": float(probs.max()),
        "transition_probabilities": {
            f"prob_{cluster_to_sid[c]}": float(round(probs[c], 4))
            for c in range(len(probs))
        },
    }


@app.get("/cohort")
def cohort(
    segment_name: str | None = None,
    state_name: str | None = None,
    predicted_segment_name: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """
    Members matching the filters, ranked by how confidently the model expects
    them to move. This is the targeting list a campaign would actually pull.
    """
    _require_ready()
    mask = np.ones(len(R.seg_names), dtype=bool)
    if segment_name:
        mask &= R.seg_names == segment_name
    if state_name:
        mask &= R.states == state_name

    cluster_to_sid = R.bundle["cluster_to_sid"]
    sid_to_name = R.bundle["sid_to_name"]
    pred_idx = R.proba.argmax(axis=1)
    pred_names = np.array([sid_to_name[cluster_to_sid[c]] for c in pred_idx])
    if predicted_segment_name:
        mask &= pred_names == predicted_segment_name

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"matched": 0, "members": []}

    # Rank by expected value at risk, not by probability of moving.
    #
    # Probability ranking puts a member with a 95% chance of losing $20 above
    # one with a 20% chance of losing $900, which is backwards for a budget
    # holder. Where value is unavailable, fall back to probability.
    if R.var is not None:
        score = R.var[idx]
        ranked_by = "value_at_risk_usd"
    else:
        cur = R.seg_cluster[idx]
        score = 1.0 - R.proba[idx, cur]
        ranked_by = "probability_of_moving"

    order = idx[np.argsort(-score)][:limit]

    members = []
    for i in order:
        cur_c = int(R.seg_cluster[i])
        members.append({
            "member_id": str(R.features["member_id"].iloc[i]),
            "segment_name": str(R.seg_names[i]),
            "lifecycle_state": str(R.states[i]),
            "predicted_segment_name": str(pred_names[i]),
            "probability_of_moving": float(round(1.0 - R.proba[i, cur_c], 4)),
            "adverse_probability": (None if R.adverse is None
                                    else float(round(R.adverse[i], 4))),
            "value_at_risk_usd": (None if R.var is None
                                  else round(float(R.var[i]), 2)),
            "recommended_action": ACTIVATION.get(str(R.states[i])),
        })

    total_var = None if R.var is None else round(float(R.var[idx].sum()), 2)
    return {
        "matched": int(mask.sum()),
        "returned": len(members),
        "ranked_by": ranked_by,
        "cohort_value_at_risk_usd": total_var,
        "members": members,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    dash = Path(__file__).parent / "dashboard.html"
    if dash.exists():
        return dash.read_text(encoding="utf-8")
    return "<h1>TBIE API</h1><p>See <a href='/docs'>/docs</a>.</p>"
