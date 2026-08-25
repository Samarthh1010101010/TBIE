"""
src/utils/eligibility.py
────────────────────────
Contact eligibility and consent — who may actually be contacted, and how.

Why this exists
───────────────
Every earlier version of this pipeline emitted a `recommended_activation` naming
a channel, with no reference to whether the member had consented to that channel
or was even an open account. At 2025-12-31 that produced:

    222,259 members  recommended a channel they had opted OUT of
      9,898 members  closed accounts, still receiving recommendations
      7,512 members  dormant accounts
      2,455 members  fraud-flagged

`members.parquet` carries `opt_in_email`, `opt_in_push`, `opt_in_sms`,
`account_status` and `fraud_flag`. Nothing read them.

In a real loyalty programme this is not a data-quality issue, it is a
regulatory one — CAN-SPAM, GDPR Art. 21 (right to object), and TCPA for SMS.
A model output that instructs a marketer to SMS someone who opted out is worse
than no output: acting on it creates liability.

Design
──────
Suppression is a HARD gate applied after modelling, never a feature. The model
still scores everyone — a closed account's predicted transition is still
information — but the recommendation becomes an explicit
"do not contact, because X" rather than a channel.

Where a member has consented to some channels but not the one a state would
normally use, the recommendation falls back to a permitted channel rather than
being dropped. Only when NO channel is permitted is the member unreachable.
"""

from __future__ import annotations

import pandas as pd

# Account states that must never receive marketing contact.
BLOCKED_ACCOUNT_STATUS = {"closed"}

# Dormant is not blocked, but it is not business-as-usual either: these members
# belong in a reactivation programme with different economics and creative.
REACTIVATION_ACCOUNT_STATUS = {"dormant"}

CHANNELS = ("email", "push", "sms")

# Preferred channel order per lifecycle state, most appropriate first.
# Derived from the activation guidance: urgent/lapse states lead with the
# highest-attention channel, engagement states lead with in-app.
STATE_CHANNEL_PREFERENCE = {
    "Lapse Risk":         ("email", "sms", "push"),
    "Win-Back Target":    ("email", "sms", "push"),
    "Silent Accumulator": ("push", "email", "sms"),
    "Momentum Builder":   ("push", "email", "sms"),
    "Brand Advocate":     ("push", "email", "sms"),
    "New & Uncertain":    ("push", "email", "sms"),
    "Redemption Hunter":  ("email", "push", "sms"),
    "Value Maximizer":    ("email", "push", "sms"),
    "Plateau Cruiser":    ("email", "push", "sms"),
    "Program Skeptic":    ("email", "push", "sms"),
}

# Message guidance per state, independent of channel.
STATE_MESSAGE = {
    "Lapse Risk":         "personalised win-back, bonus points, within 7 days",
    "Win-Back Target":    "we-miss-you reactivation bonus, immediate",
    "Silent Accumulator": "unspent points reminder, weekly cadence",
    "Momentum Builder":   "tier upgrade progress, immediate",
    "Brand Advocate":     "early access + referral bonus, immediate",
    "New & Uncertain":    "onboarding journey, first-purchase bonus",
    "Redemption Hunter":  "targeted flash promotion around promo periods",
    "Value Maximizer":    "cross-category points multiplier, mid-month",
    "Plateau Cruiser":    "curated recommendation, monthly",
    "Program Skeptic":    "re-permission with value proof, monthly",
}

REQUIRED_MEMBER_COLS = [
    "member_id", "account_status", "fraud_flag",
    "opt_in_email", "opt_in_push", "opt_in_sms",
]


def build_contact_profile(members: pd.DataFrame) -> pd.DataFrame:
    """
    One row per member: may we contact them, on which channels, and if not why.

    Missing consent is treated as NO consent. Opt-in is an affirmative act; a
    null must never be read as permission.
    """
    missing = [c for c in REQUIRED_MEMBER_COLS if c not in members.columns]
    if missing:
        raise KeyError(f"members frame is missing eligibility column(s): {missing}")

    p = members[REQUIRED_MEMBER_COLS].copy()

    status = p["account_status"].astype(str).str.strip().str.lower()
    p["is_closed"]    = status.isin(BLOCKED_ACCOUNT_STATUS)
    p["is_dormant"]   = status.isin(REACTIVATION_ACCOUNT_STATUS)
    p["is_fraud"]     = p["fraud_flag"].fillna(False).astype(bool)

    for ch in CHANNELS:
        # fillna(False): absence of a recorded opt-in is not consent.
        p[f"allow_{ch}"] = p[f"opt_in_{ch}"].fillna(False).astype(bool)

    p["n_channels_allowed"] = p[[f"allow_{c}" for c in CHANNELS]].sum(axis=1)

    # Hard suppression, in order of severity — the reported reason is the most
    # serious one that applies.
    p["suppression_reason"] = ""
    p.loc[p["n_channels_allowed"] == 0, "suppression_reason"] = "no_channel_consent"
    p.loc[p["is_fraud"],  "suppression_reason"] = "fraud_flagged"
    p.loc[p["is_closed"], "suppression_reason"] = "account_closed"

    p["is_targetable"] = p["suppression_reason"] == ""
    p["needs_reactivation_track"] = p["is_dormant"] & p["is_targetable"]

    return p.drop(columns=["fraud_flag", "opt_in_email", "opt_in_push", "opt_in_sms"])


def choose_channel(state: str, allow_email: bool, allow_push: bool,
                   allow_sms: bool) -> str | None:
    """
    Best permitted channel for a state, or None if the member is unreachable.

    Falls back down the preference order rather than dropping the member: a
    Lapse Risk member who blocked email but allows push is still worth a push.
    """
    allowed = {"email": allow_email, "push": allow_push, "sms": allow_sms}
    for ch in STATE_CHANNEL_PREFERENCE.get(state, CHANNELS):
        if allowed.get(ch):
            return ch
    return None


def recommend(state: str, row) -> tuple[str, str]:
    """
    Return (channel, action_text) for one member.

    channel is "" when the member must not be contacted; action_text then
    explains why, so a downstream user never has to guess.
    """
    if not row["is_targetable"]:
        reason = {
            "account_closed":     "Account closed — suppress from all marketing",
            "fraud_flagged":      "Fraud-flagged — suppress pending investigation",
            "no_channel_consent": "No channel consent on file — not contactable",
        }.get(row["suppression_reason"], "Suppressed")
        return "", f"DO NOT CONTACT: {reason}"

    ch = choose_channel(state, row["allow_email"], row["allow_push"], row["allow_sms"])
    if ch is None:
        return "", "DO NOT CONTACT: no permitted channel for this state"

    msg = STATE_MESSAGE.get(state, "standard lifecycle message")
    prefix = "REACTIVATION TRACK | " if row.get("needs_reactivation_track") else ""
    return ch, f"{prefix}Channel: {ch} | {msg}"


def summarise(profile: pd.DataFrame) -> dict:
    """Counts for logging and for the audit trail."""
    n = len(profile)
    return {
        "members":             int(n),
        "targetable":          int(profile["is_targetable"].sum()),
        "suppressed":          int((~profile["is_targetable"]).sum()),
        "account_closed":      int((profile["suppression_reason"] == "account_closed").sum()),
        "fraud_flagged":       int((profile["suppression_reason"] == "fraud_flagged").sum()),
        "no_channel_consent":  int((profile["suppression_reason"] == "no_channel_consent").sum()),
        "dormant_reactivation": int(profile["needs_reactivation_track"].sum()),
        "allow_email":         int(profile["allow_email"].sum()),
        "allow_push":          int(profile["allow_push"].sum()),
        "allow_sms":           int(profile["allow_sms"].sum()),
    }
