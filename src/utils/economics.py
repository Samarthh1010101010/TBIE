"""
src/utils/economics.py
──────────────────────
Campaign economics — turning predicted probabilities into targeting decisions.

Kept separate from src/10_cost_thresholds.py so the logic is importable and
testable rather than trapped inside a top-to-bottom analysis script, and so the
serving API can quote the same value-at-risk figure the offline analysis uses.

The core idea
─────────────
A classifier threshold tuned for F1 treats a false positive and a false negative
as equally bad. In a retention campaign they are wildly different: a false
positive wastes one touchpoint (a few dollars), a false negative forfeits a
share of a member's forward value (hundreds to thousands).

So the decision is not "is this member likely to slip" but "is the expected
recoverable value worth the cost of contacting them":

    P(move to a lower-value segment) x value lost x recovery_rate  >  contact_cost

A 90% chance of losing $20 is a worse use of $5 than a 15% chance of losing $900.
Probability-ranked targeting cannot see that difference; this can.

On recovery_rate
────────────────
The fraction of at-risk value a successful contact actually saves is currently
an explicit input everywhere it appears, never a hidden constant, and callers
are expected to report sensitivity across a range.

It is, however, REDUCIBLE rather than unknowable. engagement_events.parquet
records campaign exposure and response (`campaign_id`, `campaign_type`,
`campaign_response`) on 12.7M rows, which supports a modelled
P(responds | contacted) in place of a flat assumption. What is genuinely absent
is a randomised holdout, so causal lift still cannot be established from this
data alone. See AUDIT.md.
"""

from __future__ import annotations

import numpy as np


def segment_values(df, seg_col: str, spend_col: str, n_classes: int) -> np.ndarray:
    """
    Mean observed spend per segment — the value of being in each segment.

    Derived from the data rather than assumed. Segments a member never occupies
    in this window get 0.
    """
    means = df.groupby(seg_col)[spend_col].mean()
    return means.reindex(range(n_classes)).fillna(0.0).to_numpy(dtype=float)


def value_at_risk(proba: np.ndarray, seg_curr: np.ndarray,
                  values: np.ndarray) -> np.ndarray:
    """
    Expected value a member loses to an adverse move.

    Sum over destination segments worth less than the current one of
    P(destination) x (value_now - value_destination). Upward moves contribute
    nothing — this measures downside only, which is what a retention budget is
    protecting against.
    """
    current = values[seg_curr]
    drop = np.clip(current[:, None] - values[None, :], 0.0, None)
    return (proba * drop).sum(axis=1)


def adverse_probability(proba: np.ndarray, seg_curr: np.ndarray,
                        values: np.ndarray) -> np.ndarray:
    """Total probability of landing in any lower-value segment."""
    current = values[seg_curr]
    worse = values[None, :] < current[:, None]
    return (proba * worse).sum(axis=1)


def contact_by_probability(adverse: np.ndarray, threshold: float) -> np.ndarray:
    """Classic policy: contact everyone above a probability threshold."""
    return adverse >= threshold


def contact_by_expected_value(var: np.ndarray, recovery_rate: float,
                              contact_cost: float, k: float = 1.0) -> np.ndarray:
    """
    Expected-value policy: contact when recoverable value clears k x the cost.

    k = 1.0 is the break-even rule. Raising k demands a bigger margin and
    contacts fewer people, which is how a fixed budget is respected.
    """
    return (var * recovery_rate) >= (contact_cost * k)


def evaluate_campaign(contact: np.ndarray, actually_adverse: np.ndarray,
                      realised_loss: np.ndarray, contact_cost: float,
                      recovery_rate: float, label: float = 0.0) -> dict:
    """
    Score a contact policy against what actually happened.

    Benefit is credited only where a member genuinely went on to lose value.
    Contacting someone who was never going to slip recovers nothing — crediting
    it would make "contact everyone" look free, which is exactly the error this
    whole module exists to avoid.
    """
    n = int(contact.sum())
    cost = n * contact_cost
    recovered = float(realised_loss[contact & actually_adverse].sum() * recovery_rate)
    profit = recovered - cost
    n_adverse = int(actually_adverse.sum())
    return {
        "threshold":    float(label),
        "contacted":    n,
        "contact_rate": float(contact.mean()) if len(contact) else 0.0,
        "cost":         float(cost),
        "recovered":    recovered,
        "profit":       profit,
        "roi":          float(profit / cost) if cost > 0 else 0.0,
        "precision":    float(actually_adverse[contact].mean()) if n else 0.0,
        "recall":       float((contact & actually_adverse).sum() / n_adverse)
                        if n_adverse else 0.0,
    }


def break_even_recovery_rate(var: np.ndarray, actually_adverse: np.ndarray,
                             realised_loss: np.ndarray, contact_cost: float,
                             k: float = 1.0, lo: float = 1e-4,
                             hi: float = 1.0, iters: int = 40) -> float | None:
    """
    Smallest recovery rate at which the EV policy stops losing money.

    Bisection on a monotone quantity: raising the recovery rate can only raise
    recovered value. Returns None when even a 100% recovery rate loses money.
    """
    def profit_at(rate: float) -> float:
        contact = contact_by_expected_value(var, rate, contact_cost, k)
        return evaluate_campaign(contact, actually_adverse, realised_loss,
                                 contact_cost, rate)["profit"]

    if profit_at(hi) <= 0:
        return None
    if profit_at(lo) > 0:
        return lo

    for _ in range(iters):
        mid = (lo + hi) / 2
        if profit_at(mid) > 0:
            hi = mid
        else:
            lo = mid
    return hi
