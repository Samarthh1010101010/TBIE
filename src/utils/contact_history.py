"""
src/utils/contact_history.py
────────────────────────────
Contact history, frequency capping, and response propensity.

`engagement_events.parquet` records 12,759,405 campaign touches across 200
campaigns, each carrying `campaign_response` (sent / opened / clicked / none).
Nothing in the pipeline read it. Two consequences, both expensive:

  1. No frequency capping. The expected-value ranking will recommend the same
     high-value member every cycle, because their value at risk does not fall
     just because you contacted them last week. That is how a programme trains
     its best members to ignore it.

  2. The recovery rate — how much of a member's at-risk value a contact
     actually saves — was a flat assumed constant applied to everyone. Response
     history lets it vary by member.

On what this can and cannot establish
─────────────────────────────────────
Campaign exposure was not randomised: members were selected for campaigns by
some prior process, so responders differ from non-responders in ways beyond
responsiveness. That means this data supports **relative** propensity — who is
more likely to engage than whom — but NOT causal lift. Absolute lift needs a
randomised holdout, which does not exist here.

The recovery rate therefore stays an explicit assumed base; what this module
adds is a per-member multiplier around it. That is a real improvement over one
flat number for 500,000 people, and it is not a causal claim.

Leakage
───────
Every window is computed strictly from events dated <= observation_date. The
pipeline's leakage guards apply to transactions; this module enforces the same
rule for engagement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A response is an open or a click. "sent" means delivered with no engagement;
# "none" is an explicit non-response.
RESPONDED = {"opened", "clicked"}
ENGAGED_STRONG = {"clicked"}

# Contacts allowed per member per 30 days before further contact is suppressed.
# 6 is deliberately generous relative to the observed median of ~2/month; the
# intent is to catch over-contact, not to throttle normal cadence.
DEFAULT_FREQUENCY_CAP_30D = 6

# Bayesian shrinkage strength for per-member response rates. A member with 3
# touches should not be trusted over the population mean; one with 60 should.
# PRIOR_STRENGTH is expressed in pseudo-touches.
PRIOR_STRENGTH = 20.0


def build_contact_history(events: pd.DataFrame, obs_date: pd.Timestamp,
                          date_col: str = "event_date") -> pd.DataFrame:
    """
    Per-member campaign contact history as of `obs_date`.

    Returns one row per member seen in the campaign data. Members never
    contacted are absent; callers should left-join and fill.
    """
    required = {"member_id", date_col, "campaign_id", "campaign_response"}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"events frame is missing column(s): {sorted(missing)}")

    c = events[events["campaign_id"].notna()].copy()
    # Strictly no future data.
    c = c[c[date_col] <= obs_date]
    if c.empty:
        return pd.DataFrame(columns=[
            "member_id", "contacts_30d", "contacts_90d", "contacts_180d",
            "responses_180d", "clicks_180d", "days_since_last_contact",
            "response_rate_raw",
        ])

    resp = c["campaign_response"].astype(str).str.strip().str.lower()
    c["_responded"] = resp.isin(RESPONDED)
    c["_clicked"] = resp.isin(ENGAGED_STRONG)

    age = (obs_date - c[date_col]).dt.days
    c["_w30"] = age <= 30
    c["_w90"] = age <= 90
    c["_w180"] = age <= 180

    g = c.groupby("member_id")
    out = pd.DataFrame({
        "contacts_30d":   g["_w30"].sum(),
        "contacts_90d":   g["_w90"].sum(),
        "contacts_180d":  g["_w180"].sum(),
        "responses_180d": g.apply(lambda d: int((d["_w180"] & d["_responded"]).sum()),
                                  include_groups=False),
        "clicks_180d":    g.apply(lambda d: int((d["_w180"] & d["_clicked"]).sum()),
                                  include_groups=False),
        "days_since_last_contact": g[date_col].max().rsub(obs_date).dt.days,
    }).reset_index()

    out["response_rate_raw"] = np.where(
        out["contacts_180d"] > 0,
        out["responses_180d"] / out["contacts_180d"].replace(0, np.nan),
        np.nan,
    )
    return out


def add_response_propensity(history: pd.DataFrame,
                            prior_strength: float = PRIOR_STRENGTH) -> pd.DataFrame:
    """
    Smoothed per-member response rate, shrunk toward the population mean.

    A member with 2 touches and 2 responses does not have a 100% response rate;
    they have very little evidence. Empirical-Bayes shrinkage:

        rate = (responses + prior_strength * pop_rate) / (contacts + prior_strength)

    `response_multiplier` expresses the member relative to the population, which
    is the form the expected-value calculation consumes.
    """
    h = history.copy()
    total_resp = h["responses_180d"].sum()
    total_cont = h["contacts_180d"].sum()
    pop_rate = float(total_resp / total_cont) if total_cont else 0.0

    h["response_rate_smoothed"] = (
        (h["responses_180d"] + prior_strength * pop_rate)
        / (h["contacts_180d"] + prior_strength)
    )
    h["response_multiplier"] = (
        h["response_rate_smoothed"] / pop_rate if pop_rate else 1.0
    )
    h.attrs["population_response_rate"] = pop_rate
    return h


def apply_frequency_cap(profile: pd.DataFrame,
                        cap_30d: int = DEFAULT_FREQUENCY_CAP_30D) -> pd.DataFrame:
    """
    Suppress members already contacted `cap_30d` times in the last 30 days.

    Applied after consent and eligibility, and only to members who would
    otherwise be targetable — a closed account stays suppressed for being
    closed, which is the more informative reason.
    """
    p = profile.copy()
    if "contacts_30d" not in p.columns:
        raise KeyError("profile needs contacts_30d; join contact history first")

    over = p["contacts_30d"].fillna(0) >= cap_30d
    newly = over & p["is_targetable"]
    p.loc[newly, "suppression_reason"] = "frequency_cap"
    p.loc[newly, "is_targetable"] = False
    return p


def effective_recovery_rate(base_rate: float, response_multiplier,
                            lo: float = 0.25, hi: float = 2.5):
    """
    Scale the assumed base recovery rate by relative responsiveness.

    Clipped because the multiplier is an observational estimate, not a causal
    one: a member who historically opens twice as often is a better bet, but
    claiming they recover 10x the value would be reading far more into
    non-randomised exposure than it can support.
    """
    m = np.clip(np.asarray(response_multiplier, dtype=float), lo, hi)
    return base_rate * m
