"""
tests/test_economics_calibration.py
════════════════════════════════════════════════════════════════════════════
Tests for the decision and calibration maths.

These two modules turn model outputs into money and into claims about
trustworthiness, so their edge cases matter more than most:

  economics    — crediting recovered value for members who were never going to
                 churn would make "contact everyone" look profitable, which is
                 the exact error the module exists to prevent.
  calibration  — per-class calibrators can map a whole row to zero; dividing by
                 that sum yields NaN, which then propagates silently into a
                 targeting decision.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.calibration import (  # noqa: E402
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
    renormalise,
)
from utils.economics import (  # noqa: E402
    adverse_probability,
    break_even_recovery_rate,
    contact_by_expected_value,
    contact_by_probability,
    evaluate_campaign,
    segment_values,
    value_at_risk,
)

# Segment 0 is worth $1000, segment 1 $500, segment 2 $100.
VALUES = np.array([1000.0, 500.0, 100.0])


# ── Value at risk ────────────────────────────────────────────────────────────

class TestValueAtRisk:

    def test_downward_move_carries_risk(self):
        # Certain move from segment 0 ($1000) to segment 2 ($100) = $900 at risk.
        proba = np.array([[0.0, 0.0, 1.0]])
        assert value_at_risk(proba, np.array([0]), VALUES)[0] == pytest.approx(900.0)

    def test_upward_move_carries_no_risk(self):
        """Gaining value is not a retention problem."""
        proba = np.array([[1.0, 0.0, 0.0]])
        assert value_at_risk(proba, np.array([2]), VALUES)[0] == pytest.approx(0.0)

    def test_staying_put_carries_no_risk(self):
        proba = np.array([[0.0, 1.0, 0.0]])
        assert value_at_risk(proba, np.array([1]), VALUES)[0] == pytest.approx(0.0)

    def test_risk_is_probability_weighted(self):
        # 50% chance of dropping $900, 50% of staying.
        proba = np.array([[0.5, 0.0, 0.5]])
        assert value_at_risk(proba, np.array([0]), VALUES)[0] == pytest.approx(450.0)

    def test_only_lower_segments_count(self):
        # From segment 1: 0 is upward (ignored), 2 is a $400 drop.
        proba = np.array([[0.5, 0.0, 0.5]])
        assert value_at_risk(proba, np.array([1]), VALUES)[0] == pytest.approx(200.0)

    def test_adverse_probability_sums_lower_segments(self):
        proba = np.array([[0.2, 0.3, 0.5]])
        assert adverse_probability(proba, np.array([0]), VALUES)[0] == pytest.approx(0.8)
        assert adverse_probability(proba, np.array([2]), VALUES)[0] == pytest.approx(0.0)

    def test_segment_values_from_data(self):
        df = pd.DataFrame({"seg": [0, 0, 1, 1], "spend": [100.0, 200.0, 10.0, 30.0]})
        vals = segment_values(df, "seg", "spend", 3)
        assert vals[0] == pytest.approx(150.0)
        assert vals[1] == pytest.approx(20.0)
        assert vals[2] == pytest.approx(0.0)   # segment never occupied


# ── Policies ─────────────────────────────────────────────────────────────────

class TestPolicies:

    def test_probability_policy_is_a_threshold(self):
        adverse = np.array([0.1, 0.5, 0.9])
        assert contact_by_probability(adverse, 0.5).tolist() == [False, True, True]

    def test_expected_value_policy_uses_value_not_probability(self):
        """
        The whole point: a small chance of losing a lot beats a large chance of
        losing a little. Member A risks $20, member B risks $900.
        """
        var = np.array([20.0, 900.0])
        contact = contact_by_expected_value(var, recovery_rate=0.10,
                                            contact_cost=5.0, k=1.0)
        # A: 20 * 0.10 = $2 < $5 cost.  B: 900 * 0.10 = $90 > $5.
        assert contact.tolist() == [False, True]

    def test_raising_k_contacts_fewer(self):
        var = np.array([60.0, 200.0, 900.0])
        wide = contact_by_expected_value(var, 0.10, 5.0, k=1.0)
        tight = contact_by_expected_value(var, 0.10, 5.0, k=10.0)
        assert tight.sum() < wide.sum()

    def test_higher_recovery_rate_contacts_more(self):
        var = np.array([60.0, 200.0, 900.0])
        low = contact_by_expected_value(var, 0.02, 5.0)
        high = contact_by_expected_value(var, 0.50, 5.0)
        assert high.sum() >= low.sum()


# ── Campaign evaluation ──────────────────────────────────────────────────────

class TestCampaignEvaluation:

    def _case(self):
        contact          = np.array([True, True, False, False])
        actually_adverse = np.array([True, False, True, False])
        realised_loss    = np.array([1000.0, 0.0, 500.0, 0.0])
        return contact, actually_adverse, realised_loss

    def test_cost_scales_with_contacts(self):
        c, a, loss = self._case()
        r = evaluate_campaign(c, a, loss, contact_cost=5.0, recovery_rate=0.10)
        assert r["contacted"] == 2
        assert r["cost"] == pytest.approx(10.0)

    def test_no_credit_for_members_who_never_slipped(self):
        """
        Contacting someone who was not going to churn recovers nothing.
        Crediting it would make 'contact everyone' look free.
        """
        c, a, loss = self._case()
        r = evaluate_campaign(c, a, loss, contact_cost=5.0, recovery_rate=0.10)
        # Only member 0 was contacted AND actually slipped: 1000 * 0.10 = 100.
        assert r["recovered"] == pytest.approx(100.0)
        assert r["profit"] == pytest.approx(90.0)

    def test_no_credit_for_uncontacted_losses(self):
        """Member 2 slipped but was never contacted — no recovery."""
        c, a, loss = self._case()
        r = evaluate_campaign(c, a, loss, contact_cost=5.0, recovery_rate=0.10)
        assert r["recovered"] == pytest.approx(100.0)   # not 150.0

    def test_precision_and_recall(self):
        c, a, loss = self._case()
        r = evaluate_campaign(c, a, loss, contact_cost=5.0, recovery_rate=0.10)
        assert r["precision"] == pytest.approx(0.5)    # 1 of 2 contacted slipped
        assert r["recall"] == pytest.approx(0.5)       # caught 1 of 2 slippers

    def test_contacting_nobody_is_free_and_worthless(self):
        _, a, loss = self._case()
        r = evaluate_campaign(np.zeros(4, dtype=bool), a, loss, 5.0, 0.10)
        assert r["cost"] == 0.0
        assert r["recovered"] == 0.0
        assert r["roi"] == 0.0        # not a division-by-zero blow-up

    def test_profit_falls_as_contact_cost_rises(self):
        c, a, loss = self._case()
        cheap = evaluate_campaign(c, a, loss, 1.0, 0.10)["profit"]
        dear  = evaluate_campaign(c, a, loss, 100.0, 0.10)["profit"]
        assert dear < cheap

    def test_break_even_recovery_rate_is_found(self):
        var   = np.array([1000.0, 1000.0])
        adv   = np.array([True, True])
        loss  = np.array([1000.0, 1000.0])
        rate = break_even_recovery_rate(var, adv, loss, contact_cost=5.0)
        assert rate is not None
        assert 0.0 < rate < 1.0

    def test_break_even_is_none_when_never_profitable(self):
        """Nobody actually slipped, so no recovery rate can pay for contact."""
        var  = np.array([1000.0, 1000.0])
        adv  = np.array([False, False])
        loss = np.array([0.0, 0.0])
        assert break_even_recovery_rate(var, adv, loss, contact_cost=5.0) is None


# ── Calibration ──────────────────────────────────────────────────────────────

class TestCalibration:

    def test_perfectly_calibrated_has_zero_error(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, 200_000)
        y = (rng.uniform(0, 1, 200_000) < prob).astype(int)
        assert expected_calibration_error(y, prob, n_bins=10) < 0.01

    def test_overconfident_model_has_high_error(self):
        # Claims 0.99 but is right half the time.
        prob = np.full(10_000, 0.99)
        y = np.zeros(10_000, dtype=int)
        y[:5000] = 1
        assert expected_calibration_error(y, prob, n_bins=10) == pytest.approx(0.49, abs=0.02)

    def test_empty_input_is_zero_not_a_crash(self):
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_probability_of_one_lands_in_the_final_bin(self):
        """np.digitize can overflow past the last bin; it must not here."""
        rows = reliability_curve(np.array([1, 1]), np.array([1.0, 1.0]), n_bins=10)
        assert len(rows) == 1
        assert rows[0]["bin_hi"] == pytest.approx(1.0)

    def test_reliability_curve_omits_empty_bins(self):
        rows = reliability_curve(np.array([1, 0]), np.array([0.5, 0.5]), n_bins=10)
        assert len(rows) == 1
        assert rows[0]["n"] == 2

    def test_reliability_gap_direction(self):
        rows = reliability_curve(np.zeros(100, dtype=int), np.full(100, 0.8), n_bins=10)
        assert rows[0]["gap"] == pytest.approx(0.8)   # predicted high, observed 0

    def test_max_calibration_error_ignores_sparse_bins(self):
        prob = np.concatenate([np.full(1000, 0.5), np.full(5, 0.95)])
        y = np.concatenate([(np.arange(1000) % 2), np.zeros(5, dtype=int)])
        # The 0.95 bin is badly wrong but has only 5 members.
        assert maximum_calibration_error(y, prob, n_bins=10, min_count=50) < 0.1

    def test_renormalise_rows_sum_to_one(self):
        out = renormalise(np.array([[0.2, 0.2, 0.2], [1.0, 2.0, 1.0]]))
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_renormalise_handles_all_zero_row(self):
        """A collapsed row must not become NaN and poison downstream maths."""
        fallback = np.array([[0.3, 0.3, 0.4]])
        out = renormalise(np.array([[0.0, 0.0, 0.0]]), fallback=fallback)
        assert np.isfinite(out).all()
        assert np.allclose(out.sum(axis=1), 1.0)
        assert out[0].tolist() == pytest.approx(fallback[0].tolist())

    def test_renormalise_all_zero_without_fallback_is_uniform(self):
        out = renormalise(np.array([[0.0, 0.0, 0.0, 0.0]]))
        assert np.allclose(out, 0.25)
