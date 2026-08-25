"""
tests/test_contact_history.py
════════════════════════════════════════════════════════════════════════════
Tests for contact history, frequency capping and response propensity.

Three behaviours matter most here:

  - windows are computed strictly from events <= observation_date. Campaign
    history is the one place leakage could slip in unnoticed, because the
    pipeline's leakage guards cover transactions, not engagement.

  - per-member response rates are SHRUNK toward the population mean. A member
    with 2 touches and 2 responses does not have a 100% response rate; treating
    them as a certainty would send budget at noise.

  - the frequency cap only suppresses members who were otherwise targetable,
    so a closed account keeps the more informative "account_closed" reason.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.contact_history import (  # noqa: E402
    DEFAULT_FREQUENCY_CAP_30D,
    add_response_propensity,
    apply_frequency_cap,
    build_contact_history,
    effective_recovery_rate,
)

OBS = pd.Timestamp("2025-12-31")


def events(rows):
    """rows = [(member, days_before_obs, response)]"""
    return pd.DataFrame([
        {"member_id": m,
         "event_date": OBS - pd.Timedelta(days=d),
         "campaign_id": f"C{i}",
         "campaign_response": r}
        for i, (m, d, r) in enumerate(rows)
    ])


# ── Windows and leakage ──────────────────────────────────────────────────────

class TestContactHistory:

    def test_counts_fall_into_the_right_windows(self):
        h = build_contact_history(events([
            ("A", 5, "opened"), ("A", 20, "sent"),
            ("A", 60, "clicked"), ("A", 150, "none"),
        ]), OBS).set_index("member_id")
        assert h.loc["A", "contacts_30d"] == 2
        assert h.loc["A", "contacts_90d"] == 3
        assert h.loc["A", "contacts_180d"] == 4

    def test_future_events_are_excluded(self):
        """The one place engagement leakage could slip in."""
        h = build_contact_history(events([
            ("A", 5, "opened"), ("A", -10, "clicked"),   # -10 = 10 days AFTER obs
        ]), OBS).set_index("member_id")
        assert h.loc["A", "contacts_30d"] == 1
        assert h.loc["A", "contacts_180d"] == 1

    def test_window_boundary_is_inclusive(self):
        h = build_contact_history(events([("A", 30, "opened")]), OBS).set_index("member_id")
        assert h.loc["A", "contacts_30d"] == 1

    def test_non_campaign_rows_are_ignored(self):
        e = events([("A", 5, "opened")])
        e.loc[len(e)] = {"member_id": "A", "event_date": OBS - pd.Timedelta(days=1),
                         "campaign_id": None, "campaign_response": None}
        h = build_contact_history(e, OBS).set_index("member_id")
        assert h.loc["A", "contacts_30d"] == 1

    def test_opened_and_clicked_both_count_as_responses(self):
        h = build_contact_history(events([
            ("A", 5, "opened"), ("A", 6, "clicked"),
            ("A", 7, "sent"), ("A", 8, "none"),
        ]), OBS).set_index("member_id")
        assert h.loc["A", "responses_180d"] == 2
        assert h.loc["A", "clicks_180d"] == 1
        assert h.loc["A", "response_rate_raw"] == pytest.approx(0.5)

    def test_days_since_last_contact(self):
        h = build_contact_history(events([
            ("A", 5, "opened"), ("A", 40, "sent"),
        ]), OBS).set_index("member_id")
        assert h.loc["A", "days_since_last_contact"] == 5

    def test_empty_campaign_data_returns_empty_frame(self):
        e = events([("A", 5, "opened")])
        e["campaign_id"] = None
        assert len(build_contact_history(e, OBS)) == 0

    def test_missing_column_raises(self):
        e = events([("A", 5, "opened")]).drop(columns=["campaign_response"])
        with pytest.raises(KeyError, match="missing column"):
            build_contact_history(e, OBS)


# ── Response propensity ──────────────────────────────────────────────────────

class TestResponsePropensity:

    def test_thin_evidence_is_shrunk_toward_the_population(self):
        """
        A member with 2 touches and 2 responses is not a 100% responder.
        Failing to shrink here sends budget at noise.
        """
        rows = [("A", 5, "opened"), ("A", 6, "opened")]           # 2/2
        rows += [("B", d, "sent") for d in range(10, 110)]        # 0/100
        h = add_response_propensity(build_contact_history(events(rows), OBS)
                                    ).set_index("member_id")
        assert h.loc["A", "response_rate_raw"] == 1.0
        assert h.loc["A", "response_rate_smoothed"] < 0.5   # pulled hard to the mean

    def test_thick_evidence_is_trusted(self):
        rows = [("A", i % 180, "opened") for i in range(200)]     # 200/200
        rows += [("B", i % 180, "sent") for i in range(200)]      # 0/200
        h = add_response_propensity(build_contact_history(events(rows), OBS)
                                    ).set_index("member_id")
        assert h.loc["A", "response_rate_smoothed"] > 0.85
        assert h.loc["B", "response_rate_smoothed"] < 0.15

    def test_multiplier_is_relative_to_population(self):
        rows = [("A", i % 180, "opened") for i in range(200)]
        rows += [("B", i % 180, "sent") for i in range(200)]
        h = add_response_propensity(build_contact_history(events(rows), OBS)
                                    ).set_index("member_id")
        assert h.loc["A", "response_multiplier"] > 1.0
        assert h.loc["B", "response_multiplier"] < 1.0

    def test_population_rate_is_recorded(self):
        h = add_response_propensity(build_contact_history(events([
            ("A", 5, "opened"), ("B", 5, "sent"),
        ]), OBS))
        assert h.attrs["population_response_rate"] == pytest.approx(0.5)


# ── Frequency cap ────────────────────────────────────────────────────────────

class TestFrequencyCap:

    def _profile(self, contacts, targetable=True, reason=""):
        return pd.DataFrame([{
            "member_id": "M1", "contacts_30d": contacts,
            "is_targetable": targetable, "suppression_reason": reason,
        }])

    def test_over_cap_is_suppressed(self):
        out = apply_frequency_cap(self._profile(DEFAULT_FREQUENCY_CAP_30D)).iloc[0]
        assert not out["is_targetable"]
        assert out["suppression_reason"] == "frequency_cap"

    def test_under_cap_is_untouched(self):
        out = apply_frequency_cap(self._profile(DEFAULT_FREQUENCY_CAP_30D - 1)).iloc[0]
        assert out["is_targetable"]
        assert out["suppression_reason"] == ""

    def test_already_suppressed_keeps_the_more_informative_reason(self):
        """A closed account is closed, not merely over-contacted."""
        out = apply_frequency_cap(
            self._profile(99, targetable=False, reason="account_closed")).iloc[0]
        assert out["suppression_reason"] == "account_closed"

    def test_missing_history_column_raises(self):
        p = pd.DataFrame([{"member_id": "M1", "is_targetable": True,
                           "suppression_reason": ""}])
        with pytest.raises(KeyError, match="contacts_30d"):
            apply_frequency_cap(p)

    def test_never_contacted_is_not_capped(self):
        p = self._profile(np.nan)
        assert apply_frequency_cap(p).iloc[0]["is_targetable"]


# ── Effective recovery rate ──────────────────────────────────────────────────

class TestEffectiveRecoveryRate:

    def test_average_responder_gets_the_base_rate(self):
        assert effective_recovery_rate(0.10, [1.0])[0] == pytest.approx(0.10)

    def test_better_responders_get_more(self):
        assert effective_recovery_rate(0.10, [1.5])[0] == pytest.approx(0.15)

    def test_multiplier_is_clipped_both_ways(self):
        """
        Exposure was not randomised, so a 10x historical multiplier is not
        evidence of 10x recoverable value. Clipping keeps an observational
        signal from being read as a causal one.
        """
        assert effective_recovery_rate(0.10, [100.0])[0] == pytest.approx(0.25)
        assert effective_recovery_rate(0.10, [0.0])[0] == pytest.approx(0.025)

    def test_vectorised_over_many_members(self):
        out = effective_recovery_rate(0.10, [0.5, 1.0, 2.0])
        assert len(out) == 3
        assert out[1] == pytest.approx(0.10)
