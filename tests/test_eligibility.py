"""
tests/test_eligibility.py
════════════════════════════════════════════════════════════════════════════
Tests for contact eligibility and consent.

These matter more than the modelling tests. A wrong prediction costs a wasted
touchpoint; contacting someone who withdrew consent is a regulatory breach
(CAN-SPAM, GDPR Art. 21, TCPA). The behaviours pinned here are the ones that
were wrong before this module existed:

  - 222,259 members were recommended a channel they had opted out of
  -   9,898 closed accounts were receiving marketing recommendations
  -   2,455 fraud-flagged members were being targeted

The most important single assertion in this file is
`test_null_optin_is_not_consent`: opt-in is an affirmative act, and a missing
value must never be read as permission.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.eligibility import (  # noqa: E402
    CHANNELS,
    STATE_CHANNEL_PREFERENCE,
    build_contact_profile,
    choose_channel,
    recommend,
    summarise,
)


def members(**over):
    base = {
        "member_id": "M1",
        "account_status": "active",
        "fraud_flag": False,
        "opt_in_email": True,
        "opt_in_push": True,
        "opt_in_sms": True,
    }
    base.update(over)
    return pd.DataFrame([base])


def profile_of(**over):
    return build_contact_profile(members(**over)).iloc[0]


# ── Suppression ──────────────────────────────────────────────────────────────

class TestSuppression:

    def test_active_consenting_member_is_targetable(self):
        p = profile_of()
        assert p["is_targetable"]
        assert p["suppression_reason"] == ""

    def test_closed_account_is_suppressed(self):
        p = profile_of(account_status="closed")
        assert not p["is_targetable"]
        assert p["suppression_reason"] == "account_closed"

    def test_fraud_flag_is_suppressed(self):
        p = profile_of(fraud_flag=True)
        assert not p["is_targetable"]
        assert p["suppression_reason"] == "fraud_flagged"

    def test_no_consent_anywhere_is_suppressed(self):
        p = profile_of(opt_in_email=False, opt_in_push=False, opt_in_sms=False)
        assert not p["is_targetable"]
        assert p["suppression_reason"] == "no_channel_consent"

    def test_closed_outranks_other_reasons(self):
        """The most serious applicable reason is the one reported."""
        p = profile_of(account_status="closed", fraud_flag=True,
                       opt_in_email=False, opt_in_push=False, opt_in_sms=False)
        assert p["suppression_reason"] == "account_closed"

    def test_dormant_is_targetable_but_flagged(self):
        """Dormant is a different programme, not a block."""
        p = profile_of(account_status="dormant")
        assert p["is_targetable"]
        assert p["needs_reactivation_track"]

    def test_account_status_is_case_insensitive(self):
        assert not profile_of(account_status="CLOSED")["is_targetable"]
        assert not profile_of(account_status=" Closed ")["is_targetable"]

    # ── The assertion that matters most ──────────────────────────────────────
    def test_null_optin_is_not_consent(self):
        """
        A missing opt-in is an absence of permission, never permission. Reading
        null as True is how a compliance breach gets shipped.
        """
        p = profile_of(opt_in_email=None, opt_in_push=None, opt_in_sms=None)
        assert not p["allow_email"]
        assert not p["allow_push"]
        assert not p["allow_sms"]
        assert not p["is_targetable"]
        assert p["suppression_reason"] == "no_channel_consent"

    def test_null_fraud_flag_is_not_fraud(self):
        p = profile_of(fraud_flag=None)
        assert p["is_targetable"]

    def test_missing_column_raises(self):
        bad = members().drop(columns=["opt_in_sms"])
        with pytest.raises(KeyError, match="eligibility column"):
            build_contact_profile(bad)


# ── Channel selection ────────────────────────────────────────────────────────

class TestChannelChoice:

    def test_uses_the_preferred_channel_when_permitted(self):
        # Lapse Risk prefers email first.
        assert choose_channel("Lapse Risk", True, True, True) == "email"
        # Silent Accumulator prefers push first.
        assert choose_channel("Silent Accumulator", True, True, True) == "push"

    def test_falls_back_rather_than_dropping_the_member(self):
        """A Lapse Risk member who blocked email is still worth a push."""
        assert choose_channel("Lapse Risk", False, True, False) == "push"

    def test_respects_sms_opt_out(self):
        assert choose_channel("Win-Back Target", False, False, False) is None
        assert choose_channel("Win-Back Target", False, False, True) == "sms"

    def test_no_permitted_channel_returns_none(self):
        assert choose_channel("Program Skeptic", False, False, False) is None

    def test_unknown_state_still_resolves(self):
        assert choose_channel("Not A State", True, True, True) in CHANNELS

    def test_every_state_has_a_preference_order(self):
        for state, order in STATE_CHANNEL_PREFERENCE.items():
            assert set(order) == set(CHANNELS), state


# ── Recommendations ──────────────────────────────────────────────────────────

class TestRecommend:

    def test_suppressed_member_gets_do_not_contact(self):
        ch, action = recommend("Lapse Risk", profile_of(account_status="closed"))
        assert ch == ""
        assert action.startswith("DO NOT CONTACT")
        assert "closed" in action.lower()

    def test_recommendation_never_names_an_opted_out_channel(self):
        """The specific defect this module exists to prevent."""
        for state in STATE_CHANNEL_PREFERENCE:
            p = profile_of(opt_in_email=False, opt_in_sms=False, opt_in_push=True)
            ch, action = recommend(state, p)
            assert ch == "push", f"{state} picked {ch!r}"
            assert "email" not in action.lower()
            assert "sms" not in action.lower()

    def test_dormant_member_is_marked_reactivation(self):
        ch, action = recommend("Lapse Risk", profile_of(account_status="dormant"))
        assert ch == "email"
        assert "REACTIVATION TRACK" in action

    def test_action_names_the_chosen_channel(self):
        ch, action = recommend("Momentum Builder", profile_of())
        assert f"Channel: {ch}" in action

    def test_fraud_member_never_gets_a_channel(self):
        ch, action = recommend("Brand Advocate", profile_of(fraud_flag=True))
        assert ch == ""
        assert "Fraud" in action


# ── Summary ──────────────────────────────────────────────────────────────────

class TestSummary:

    def test_counts_add_up(self):
        df = pd.DataFrame([
            {"member_id": "A", "account_status": "active", "fraud_flag": False,
             "opt_in_email": True,  "opt_in_push": True,  "opt_in_sms": True},
            {"member_id": "B", "account_status": "closed", "fraud_flag": False,
             "opt_in_email": True,  "opt_in_push": True,  "opt_in_sms": True},
            {"member_id": "C", "account_status": "active", "fraud_flag": True,
             "opt_in_email": True,  "opt_in_push": True,  "opt_in_sms": True},
            {"member_id": "D", "account_status": "active", "fraud_flag": False,
             "opt_in_email": False, "opt_in_push": False, "opt_in_sms": False},
            {"member_id": "E", "account_status": "dormant", "fraud_flag": False,
             "opt_in_email": True,  "opt_in_push": False, "opt_in_sms": False},
        ])
        s = summarise(build_contact_profile(df))
        assert s["members"] == 5
        assert s["targetable"] == 2            # A and E
        assert s["suppressed"] == 3            # B, C, D
        assert s["account_closed"] == 1
        assert s["fraud_flagged"] == 1
        assert s["no_channel_consent"] == 1
        assert s["dormant_reactivation"] == 1
        assert s["targetable"] + s["suppressed"] == s["members"]
