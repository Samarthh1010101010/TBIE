"""
src/utils/calibration.py
────────────────────────
Calibration metrics — is a predicted 70% actually right 70% of the time?

Separated from src/09_calibration.py so the metrics are importable and testable,
and so drift monitoring can re-check calibration on a fresh month without
re-running the whole analysis.

Why this matters here specifically
──────────────────────────────────
The pipeline emits a probability distribution per member, and
src/utils/economics.py multiplies those probabilities by dollar values to decide
who to contact. That arithmetic is only meaningful if the probabilities mean
what they say. A model can have excellent macro F1 and still be badly
calibrated — F1 only cares about which class wins the argmax, not whether 0.9
means 0.9.

Class reweighting, which this pipeline uses, deliberately distorts predicted
probabilities away from observed base rates in exchange for minority-class
recall. So calibration is something to measure, never to assume.
"""

from __future__ import annotations

import numpy as np


def bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_index(prob: np.ndarray, n_bins: int) -> np.ndarray:
    # right=False with interior edges puts 1.0 in the final bin rather than
    # overflowing past it.
    return np.digitize(prob, bin_edges(n_bins)[1:-1], right=False)


def expected_calibration_error(y_true_binary: np.ndarray, prob: np.ndarray,
                               n_bins: int = 15) -> float:
    """
    Weighted mean gap between predicted confidence and observed frequency.

    0 is perfect. Each bin contributes in proportion to how many predictions
    fall in it, so a large gap in a nearly-empty bin does not dominate.
    """
    if len(prob) == 0:
        return 0.0
    idx = _bin_index(prob, n_bins)
    total = len(prob)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.sum() / total * abs(prob[m].mean() - y_true_binary[m].mean())
    return float(ece)


def maximum_calibration_error(y_true_binary: np.ndarray, prob: np.ndarray,
                              n_bins: int = 15, min_count: int = 50) -> float:
    """
    Worst gap in any sufficiently populated bin.

    ECE can hide a badly miscalibrated region if few predictions land there;
    this surfaces it. Bins below `min_count` are ignored as noise.
    """
    if len(prob) == 0:
        return 0.0
    idx = _bin_index(prob, n_bins)
    worst = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_count:
            continue
        worst = max(worst, abs(prob[m].mean() - y_true_binary[m].mean()))
    return float(worst)


def reliability_curve(y_true_binary: np.ndarray, prob: np.ndarray,
                      n_bins: int = 15) -> list[dict]:
    """
    Per-bin predicted vs observed rates — the reliability diagram, as data.

    Empty bins are omitted rather than emitted as zeros, which would draw a
    misleading line through regions the model never predicts.
    """
    edges = bin_edges(n_bins)
    idx = _bin_index(prob, n_bins)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin_lo":         float(edges[b]),
            "bin_hi":         float(edges[b + 1]),
            "n":              int(m.sum()),
            "mean_predicted": float(prob[m].mean()),
            "observed_rate":  float(y_true_binary[m].mean()),
            "gap":            float(prob[m].mean() - y_true_binary[m].mean()),
        })
    return rows


def renormalise(proba: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """
    Rescale rows to sum to 1 after per-class calibration.

    Per-class calibrators are fitted independently, so their outputs do not sum
    to 1. A row can also collapse to all-zeros if every class maps to 0; those
    rows fall back to the uncalibrated distribution rather than dividing by zero
    and producing NaN.
    """
    out = np.array(proba, dtype=float, copy=True)
    totals = out.sum(axis=1, keepdims=True)
    dead = totals[:, 0] <= 1e-12
    if dead.any():
        if fallback is None:
            out[dead] = 1.0 / out.shape[1]
        else:
            out[dead] = fallback[dead]
        totals = out.sum(axis=1, keepdims=True)
    return out / totals
