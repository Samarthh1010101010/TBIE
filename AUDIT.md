# TBIE — Audit: What's Still Wrong, and What a Loyalty Marketer Would Ask For

Written after the engineering pass. The code is now correct and tested; this
document is about whether the **product** is right, which is a different
question. Read from the point of view of someone who owns a retention budget.

---

## Part 0 — Corrections to earlier claims in this repo

Three statements written into `METHODOLOGY.md`, `MODEL_CARD.md` and
`RESULTS.md` are **factually wrong**. They were written before the raw schema
was inspected properly.

| Claim made | Reality |
|---|---|
| "No historical A/B data was available to estimate campaign lift" | `engagement_events.parquet` has **12,759,405 rows (35.9%)** carrying `campaign_id`, `campaign_type` and `campaign_response` (`sent` / `opened` / `clicked` / `none`) |
| "There is no randomised holdout, so no counterfactual exists" | Not a randomised holdout, but campaign exposure and response **are** recorded, which supports matched-control estimation |
| "The dataset carries no protected attributes, so no subgroup analysis was performed" | `age_band`, `gender`, `region`, `urban_rural` are all present and **100% populated** |

The fairness audit has since been run — see Part 4. It is clean, which is worth
knowing, but the earlier statement was an assumption presented as a fact.

---

## Part 1 — The premise problem: this population does not churn

The project is framed around lapse prevention. The data barely contains lapse.

| Month | Median recency | p99 recency | Inactive >90d |
|---|---:|---:|---:|
| Feb 2025 | 6d | 29d | 0.00% |
| Jun 2025 | 6d | 49d | 0.05% |
| Sep 2025 | 7d | 70d | 0.39% |
| Dec 2025 | 3d | 38d | 0.08% |

At the submission date **98.4% of members purchased within 30 days** and only
**0.2% exceed the 60-day Win-Back gate**. Median transactions per member per
year is 30 — roughly one every twelve days, for essentially everybody.

Consequences:

- `Win-Back Target` = 485 members (0.1%), `Redemption Hunter` = 400 (0.1%).
  These are not rule bugs. There is nobody to find.
- `Lapse Risk` = 5,275 (1.1%). A real programme runs 20–40% at risk.
- `Momentum Builder` absorbs 173,346 (34.7%) because `spend_slope_30d > 5` is
  cleared by 35.5% of a population where the slope has σ = 64.
- Three states cover **73.6%** of members; the two most commercially actionable
  states cover **0.18%**.

**A marketer's reading:** the segmentation describes intensity of engagement
among the already-engaged. It does not identify who is leaving, because in this
data almost nobody leaves. Any "revenue at risk from attrition" narrative built
on it is describing a phenomenon the data does not contain.

**But the churn label does exist, unused** — see Part 2.

---

## Part 2 — Live defects in the product

These are not code bugs. The pipeline runs correctly and produces exactly what
it was told to produce. They are defects in what it was told to produce.

### 2.1 Marketing consent is ignored — 222,259 members (44%)

`members.parquet` carries `opt_in_email`, `opt_in_push`, `opt_in_sms`. Nothing
in the pipeline reads them. The `recommended_activation` strings assign channels
regardless.

| Recommendation | Members who have opted OUT of that channel |
|---|---:|
| Email-based action | **60,457** |
| SMS-based action | **4,523** |
| Push-based action | **159,252** |
| **Unique members with an uncontactable recommendation** | **222,259** |

In a real programme this is not a quality issue, it is a **compliance issue** —
CAN-SPAM, GDPR Art. 21, TCPA for SMS. A deliverable that tells a marketer to SMS
4,523 people who opted out is worse than useless; acting on it creates
liability. This is the single most serious finding in the audit.

### 2.2 Ineligible members are scored and targeted — ~19,865

| Population | Count | Should be |
|---|---:|---|
| `account_status == 'closed'` | 9,898 | suppressed entirely |
| `account_status == 'dormant'` | 7,512 | separate reactivation track, not BAU |
| `fraud_flag == True` | 2,455 | suppressed entirely |

Closed accounts are receiving segment assignments, lifecycle states,
transition predictions and recommended actions. Spend directed at them is
wasted by construction.

### 2.3 The actual churn label is unused

`account_status` (closed 2.0%, dormant 1.5%) and `account_close_date` are a
**direct, observed churn outcome**. The project predicts segment transitions as
a *proxy* for disengagement while a real label sits unused in the source data.

A supervised churn model on `account_close_date` would be more directly useful
to a retention team than 5-way segment transition prediction, and it would have
a real, if small, positive class.

---

## Part 3 — Data available and unused

Roughly half the source schema never reaches a feature.

| Field | Why a marketer wants it |
|---|---|
| `opt_in_email/push/sms` | Contactability. Nothing should be recommended without it. |
| `account_status`, `account_close_date` | Real churn label |
| `campaign_id`, `campaign_type`, `campaign_response` | Response propensity, contact history, fatigue, and the basis for incrementality |
| `coupon_used` | Discount dependence — who only ever buys on promotion |
| `credit_utilization` | Financial stress precedes spend decline; deliberately excluded |
| `tier_history` | Tier trajectory. `tier_ordinal` is the #1 SHAP feature yet its dynamics are unmodelled |
| `support_contact`, `support_resolution_status` | Service failure is a leading churn indicator |
| `points_clawed_back` | Returns and gaming |
| `enrollment_channel`, `acquisition_source` | Acquisition cohort quality |
| `age_band`, `gender`, `region`, `urban_rural` | Fairness auditing; regional targeting |
| `basket_size`, `merchant_category` | Category affinity for offer selection |

`campaign_response` is the most valuable of these. With 12.7M campaign rows the
project could estimate **who responds to contact**, not merely who is likely to
move — which is the question a budget holder actually asks.

---

## Part 4 — Fairness audit (now run)

Segment distribution across protected attributes, at 2025-12-31:

| Gender | Growth Builder | High-Tier | Program Skeptic | Silent Accum |
|---|---:|---:|---:|---:|
| F | 39.7% | 17.5% | 17.5% | 25.3% |
| M | 39.5% | 17.7% | 17.5% | 25.3% |
| O | 39.4% | 17.8% | 17.5% | 25.2% |
| unknown | 39.7% | 17.1% | 17.5% | 25.6% |

Age bands are equally flat (39.5–39.9% Growth Builder across all six bands).

**Result: no measurable disparate impact in segment assignment.** Worth stating
positively — but note this reflects synthetic data where demographics were
likely generated independently of behaviour. On real data this audit must be
re-run, not assumed to carry over.

---

## Part 5 — What a loyalty marketer would ask for next

Ordered by what actually changes a campaign.

### Tier 1 — blocking; would prevent me shipping this — **NOW IMPLEMENTED**

1. ~~**Eligibility and consent suppression.**~~ **Done.**
   `src/utils/eligibility.py` + `outputs/contact_eligibility.csv`. Consent
   violations went 222,259 → **0**. Closed and fraud-flagged accounts receive
   `DO NOT CONTACT` with a reason instead of a channel. Channel falls back down
   the state's preference order rather than dropping a reachable member.
2. ~~**Frequency capping and contact history.**~~ **Done.**
   `src/utils/contact_history.py`. Members with >= 6 campaign contacts in 30
   days are suppressed (87,405 at 2025-12-31). History is computed strictly
   from events dated <= observation_date.

Result: **319,271 of 500,000 members (63.9%) are actually targetable.** The
other 36% were previously being handed to a marketer as if they were.

3. **Response propensity** (was Tier 2) — **partially done.** Per-member
   response rates from `campaign_response`, shrunk toward the population mean,
   now scale the recovery rate (p10 8.3% / median 10.2% / p90 11.0% around a
   10% base). The spread is narrow because this population responds at 80.5%.
   A full propensity *model* — and the randomised holdout needed for causal
   lift — remain open.

### Tier 2 — changes the economics

3. **Response propensity, not just transition probability.** The current EV rule
   is `P(adverse move) × value × recovery_rate`, with recovery rate assumed.
   `campaign_response` allows a modelled `P(responds | contacted)`, replacing
   the single most important assumption in the business case.
4. **Real CLV instead of 180-day spend.** Margin, tenure and expected future
   value, not trailing revenue. Two members with equal spend are not equally
   valuable.
5. **Points liability.** `current_point_balance` and breakage are a balance-sheet
   item. Driving redemption has a cost the current model never sees.

### Tier 3 — depth

6. **Offer selection, not just channel.** Recommendations name a channel and a
   timing but never an offer depth. 10% vs 20% vs points multiplier is the
   decision with the actual margin consequence.
7. **Transition-specific value.** All transitions are currently equal to the
   model. High-Tier → Program Skeptic and Growth Builder → Silent Accumulator
   are worth very different amounts.
8. **Service-recovery trigger.** `support_contact` with an unresolved status is
   a strong, actionable, immediate churn signal that no state captures.
9. **A holdout group.** 5% of every campaign left uncontacted, permanently, is
   the only way this ever becomes causal rather than correlational.

---

## Part 6 — Technical items still open

| Item | Severity |
|---|---|
| Growth Builder ↔ Silent Accumulator confusion is 58% of all model error; the two centroids are the closest pair (3.33) and differ by magnitude, not kind | High — caps macro F1 at ~0.81 |
| `month_num` is the #2 SHAP feature; seasonality seen once will not transfer to months 13–14 | Medium |
| `Plateau Cruiser` is 117 members (0.02%) and inflates macro F1 with an F1 of 0.986 on 108 test rows | Medium — consider reporting macro F1 excluding it |
| Jan 2025 features are all-NaN for recency (cold start); Phase 8 skips January, but the file is misleading | Low |
| `.git` history is 229 MB of previously-committed data files | Low — needs `git filter-repo` |
| Dockerfile and Makefile are structurally validated but never executed | Low |

---

## Bottom line

The engineering is sound: single-sourced modules, 113 tests, honest metrics with
baselines and confidence intervals, calibration verified, drift monitored.

The **product** has three problems a loyalty marketer would raise immediately:
it recommends contacting 222,259 people who opted out, it targets 19,865 closed,
dormant or fraud-flagged accounts, and it is built around a churn narrative that
this population does not exhibit — while the real churn label sits unused in the
source data.

None of those are hard to fix. All of them are more valuable than another 0.005
of macro F1.
