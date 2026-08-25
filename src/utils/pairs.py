"""
src/utils/pairs.py
──────────────────
Build month-over-month transition pairs — the single implementation.

A "pair" is one member observed at month T with the label being their segment at
month T+1. Phase 8 trains on these; the calibration and cost-threshold analyses
need the identical construction, so it lives here rather than being copied into
each script.

Walk-forward discipline is the caller's responsibility: pass disjoint,
time-ordered month lists for train / validation / test. Nothing in this module
shuffles or leaks across months.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .lag_features import add_delta_features, add_segment_history
from .velocity import add_velocity_features


def month_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y_%m_%d")


def to_model_matrix(df: pd.DataFrame, x_cols: list[str]) -> np.ndarray:
    """
    Build the float32 model matrix, filling NaN with 0.

    The fill is the point. Training fills NaN with 0 before fitting, so the
    trees learned splits on a 0-valued feature. Passing NaN at inference makes
    XGBoost route those rows down its "missing" branch instead, which is a
    different path through the tree — a silent train/serve skew.

    On this dataset that affected `recency_days` (1,814 never-purchased
    members) and `months_since_last_tier_change` (95,328 members, ~19%, and the
    6th most important feature by SHAP).

    Raises if a trained feature is absent rather than silently zero-filling a
    whole column, which would degrade the model without any error.
    """
    absent = [c for c in x_cols if c not in df.columns]
    if absent:
        raise KeyError(
            f"{len(absent)} trained feature(s) missing at inference: "
            f"{absent[:10]}{' ...' if len(absent) > 10 else ''}"
        )
    return df.reindex(columns=x_cols).fillna(0).values.astype(np.float32)


def build_pair(
    t_date: pd.Timestamp,
    t1_date: pd.Timestamp,
    root: Path,
    base_feat_cols: list[str],
    velocity_cols: list[str],
    use_lag: bool = False,
    lag_base_cols: list[str] | None = None,
    tm1_date: pd.Timestamp | None = None,
    state_map_out: dict[str, int] | None = None,
) -> pd.DataFrame | None:
    """
    Assemble one (T -> T+1) transition pair.

    Returns None when any required file for this pair is absent, so callers can
    skip the month rather than fail the whole run.

    state_map_out : if given, the state_name -> state_id mapping observed in the
                    Phase 7 outputs is accumulated into this dict. Callers save
                    it in the model bundle so inference encodes y_curr the same
                    way. Raises if a state's id is inconsistent across months.
    """
    lag_base_cols = lag_base_cols or []

    features_dir = root / "features"
    segments_dir = root / "segments"
    states_dir   = root / "states"

    t_str, t1_str = month_key(t_date), month_key(t1_date)

    feat_path = features_dir / f"features_{t_str}.parquet"
    seg_t     = segments_dir / f"behavioral_segments_{t_str}.parquet"
    seg_t1    = segments_dir / f"behavioral_segments_{t1_str}.parquet"
    state_t   = states_dir   / f"lifecycle_states_{t_str}.parquet"
    required  = [feat_path, seg_t, seg_t1, state_t]

    feat_tm1 = seg_tm1 = None
    if use_lag:
        if tm1_date is None:
            return None
        tm1_str  = month_key(tm1_date)
        feat_tm1 = features_dir / f"features_{tm1_str}.parquet"
        seg_tm1  = segments_dir / f"behavioral_segments_{tm1_str}.parquet"
        required += [feat_tm1, seg_tm1]

    if not all(p.exists() for p in required):
        return None

    feat = pd.read_parquet(feat_path, engine="pyarrow").reset_index(drop=True)
    feat = add_velocity_features(feat)

    if use_lag:
        prev_feat = pd.read_parquet(
            feat_tm1, engine="pyarrow", columns=["member_id"] + lag_base_cols
        )
        feat = add_delta_features(feat, prev_feat, lag_base_cols)
        prev_seg = (pd.read_parquet(seg_tm1, engine="pyarrow")[["member_id", "segment_id"]]
                      .rename(columns={"segment_id": "seg_prev_raw"}))
        feat = feat.merge(prev_seg, on="member_id", how="left")

    keep = list(base_feat_cols) + list(velocity_cols)
    if use_lag:
        keep += [f"d_{c}" for c in lag_base_cols]
    feat = feat[["member_id"] + keep + (["seg_prev_raw"] if use_lag else [])]
    feat = feat.fillna({c: 0 for c in keep})

    sc = (pd.read_parquet(seg_t, engine="pyarrow")[["member_id", "segment_id"]]
            .rename(columns={"segment_id": "seg_curr"}))
    sn = (pd.read_parquet(seg_t1, engine="pyarrow")[["member_id", "segment_id"]]
            .rename(columns={"segment_id": "seg_next"}))

    st_raw = pd.read_parquet(state_t, engine="pyarrow")[
        ["member_id", "state_id", "state_name"]]
    if state_map_out is not None:
        for nm, sid in st_raw[["state_name", "state_id"]].drop_duplicates().itertuples(index=False):
            prev = state_map_out.setdefault(nm, int(sid))
            if prev != int(sid):
                raise ValueError(
                    f"Inconsistent state encoding for '{nm}': {prev} in an "
                    f"earlier month vs {sid} in {t_str}. Phase 7 must emit "
                    f"stable state_ids."
                )
    st = st_raw[["member_id", "state_id"]].rename(columns={"state_id": "y_curr"})

    pair = (feat.merge(sc, on="member_id")
                .merge(sn, on="member_id")
                .merge(st, on="member_id"))

    if use_lag:
        pair = add_segment_history(pair, pair["seg_prev_raw"].values)
        pair = pair.drop(columns=["seg_prev_raw"])

    pair["month_num"] = t_date.month
    pair["pair_str"]  = f"{t_date.strftime('%b')}->{t1_date.strftime('%b')}"
    return pair


def build_pairs(
    snapshot_dates,
    months: list[int],
    root: Path,
    base_feat_cols: list[str],
    velocity_cols: list[str],
    use_lag: bool = False,
    lag_base_cols: list[str] | None = None,
    skip_months: tuple[int, ...] = (1,),
    state_map_out: dict[str, int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Concatenate every available pair whose T-month is in `months`.

    skip_months defaults to January, which is cold-start: the panel has almost
    no behavioural history at the first snapshot.
    """
    frames = []
    for i, t_date in enumerate(snapshot_dates[:-1]):
        if t_date.month in skip_months or t_date.month not in months:
            continue
        pair = build_pair(
            t_date=t_date,
            t1_date=snapshot_dates[i + 1],
            root=root,
            base_feat_cols=base_feat_cols,
            velocity_cols=velocity_cols,
            use_lag=use_lag,
            lag_base_cols=lag_base_cols,
            tm1_date=snapshot_dates[i - 1] if i > 0 else None,
            state_map_out=state_map_out,
        )
        if pair is None:
            if verbose:
                print(f"  {month_key(t_date)}: missing file(s) — skipping pair")
            continue
        if verbose:
            print(f"  {pair['pair_str'].iloc[0]}: {len(pair):,} pairs")
        frames.append(pair)

    if not frames:
        raise RuntimeError(f"No transition pairs could be built for months={months}")
    return pd.concat(frames, ignore_index=True)
