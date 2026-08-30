# Model Card — TBIE Segment Transition Model

Temporal Behavioural Intelligence Engine (TBIE)
Kobie × PES University Hackathon 2025

---

## Model Details

| Field | Value |
|---|---|
| Model type | XGBoost multi-class classifier (`multi:softprob`) |
| Version | 2.0 |
| Task | Predict a member's behavioural segment 30 days ahead |
| Output | Probability distribution over 5 segments + argmax prediction |
| Features | 64 (40 behavioural + 6 velocity + 13 month-over-month deltas + `seg_prev`, `seg_changed`, `seg_curr`, `y_curr`, `month_num`) |
| Training data | 4.0M member-month transition pairs, Feb–Oct 2025 |
| Random seed | 42, fixed across K-Means, XGBoost, numpy, and all splits |
| Framework | xgboost 3.3.0, scikit-learn 1.9.0, Python 3.12 |
| License / access | Hackathon submission; not publicly released |

The model sits on top of two upstream components:

1. **Layer 1 — Segment discovery.** Frozen K-Means (k=5) fitted once on
   December 2025 features. Applied elsewhere by `scaler.transform()` →
   `pca.transform()` → nearest centroid. Never refitted at inference.
2. **Layer 2 — Lifecycle state.** A 10-state priority cascade
   (`src/utils/state_rules.py`). Its output feeds the transition model as the
   `y_curr` feature.

---

## Intended Use

**In scope.** Prioritising loyalty-programme outreach: identifying which
members are likely to migrate to a lower-value segment in the next 30 days, and
which lifecycle state they currently occupy, so that campaign budget can be
directed at the members whose behaviour is actually changing.

**Out of scope.**

- **Individual-level decisions with material consequences.** This model was
  built to rank a population for marketing contact. It is not validated for
  credit decisions, pricing, account restriction, or anything that
  meaningfully affects a member's standing.
- **Causal claims.** The model predicts what will happen, not what would happen
  under an intervention. The "recommended activation" strings in
  `segment_profiles.json` are hypotheses derived from segment behaviour, not
  effects measured against a control group. Campaign exposure and response
  ARE recorded (`campaign_id` / `campaign_response`, 12.7M rows), so response
  propensity is estimable — but no randomised holdout exists, so causal lift
  is not. An earlier revision wrongly stated no such data was available.
- **Horizons beyond ~30 days.** The target is defined at T+30d. It has not been
  evaluated at 60 or 90 days.
- **Populations other than the one it was trained on.** See Limitations.

---

## Evaluation

Walk-forward only. No random shuffling at any stage; every test row is strictly
later in time than every training row.

| Split | Window | Pairs |
|---|---|---|
| Train | Feb→Mar … Sep→Oct (8 pairs) | 4,000,000 |
| Validation | Oct→Nov | 500,000 |
| Test | Nov→Dec | 500,000 |
| Holdout | Months 13–14 | Never accessed |

**Every modelling choice was made on validation**: the sample-weighting scheme,
the argmax-vs-threshold-calibrated decision rule, and early stopping. The test
split is read exactly once, to produce the reported number. An earlier version
selected the decision rule by comparing test scores and reported the better of
the two, which inflated the headline figure; that is fixed.

### Headline results

| | Macro F1 (test, Nov→Dec) |
|---|---:|
| Majority-class baseline | 0.0589 |
| Persistence baseline (`seg_next = seg_curr`) | 0.5651 |
| **TBIE** | **0.8073**  95% CI [0.8035, 0.8100] |
| | **+0.2422 over persistence** |

Per class:

| Segment | F1 | Support |
|---|---:|---:|
| Growth Builder | 0.668 | 180,343 |
| High-Tier Accelerator | 0.896 | 79,487 |
| Program Skeptic | 0.801 | 86,337 |
| Silent Accumulator | 0.686 | 153,725 |
| Plateau Cruiser | 0.986 | 108 |

Growth Builder and Silent Accumulator set the ceiling on macro F1. Plateau
Cruiser's 0.986 rests on 108 rows and should be read as unstable, not strong.

Regenerate with `python src/08_transition_prediction.py`; full report in
`outputs/phase8_classification_report.txt`.

### Probability calibration

Measured, not assumed (`src/09_calibration.py`). Mean expected calibration
error on test is **0.0184** and mean Brier **0.0673** — when the model says 70%,
the observed rate is 71%. Isotonic recalibration fitted on validation made test
reliability *worse* (ECE 0.0184 → 0.0375), so raw probabilities ship.

This matters because §"Out of scope" permits ranking members for contact, and
`src/10_cost_thresholds.py` does expected-value arithmetic on these numbers.
That arithmetic is only meaningful because the probabilities were checked.

### Baselines

A macro F1 in isolation is not interpretable. Two references are reported
alongside it:

| Baseline | Definition | Why it matters |
|---|---|---|
| Majority class | Always predict the largest training segment | Floor |
| **Persistence** | Predict `seg_next = seg_curr` (nobody moves) | **The real bar** |

`seg_curr` is the single strongest feature by importance — segment membership
is highly autocorrelated month over month. Any honest claim about this model
must be stated as lift over persistence, not as a raw F1.

---

## Training Data

Three provided parquet files. No external data, pretrained embeddings, or
supplementary sources at any stage.

| Table | Rows |
|---|---|
| `members.parquet` | 500,000 |
| `transactions.parquet` | ~18M |
| `engagement_events.parquet` | ~35M |

### Known data quality issues and handling

| Issue | Scale | Resolution |
|---|---|---|
| Ghost member IDs | 88,717 | Excluded — both `MBR_GHOST_`-tagged and orphan IDs present in activity but absent from `members` |
| Mixed datetime formats | ~7.4% of date strings | Strict multi-format parser; unparseable values raise rather than coerce |
| Mixed-case categoricals | 24,738 rows | Normalised with `.strip().lower()` |
| Duplicate engagement events | 92,748 rows (0.26%) | Deduplicated on `[member_id, event_date, event_type]` |
| Session duration outliers | 18,663 rows > 4h, max 48h | Clipped at 14,400s |
| Leakage columns in `members` | 3 lifetime-total columns | Discarded and reconstructed as running sums up to the observation date |
| No `email_sent` events | Entire column absent | `email_open_30d` documented as a raw count, not a rate |
| Null-timestamp engagement events | 117 members | Inflated 30d counts; isolated into the S05 microcluster |

### Leakage controls

`assert_no_future_data` and `assert_pre_enrollment_removed`
(`src/utils/leakage_guard.py`) run on every snapshot build, not as spot checks.
A violation raises and stops the pipeline.

---

## Limitations

**Behavioural, not causal.** Covered above; repeated because it is the most
likely way this model gets misused.

**Silhouette of 0.117 on the segments.** Loyalty behaviour is a continuous
cloud, not density-separated groups — members differ in degree across many
dimensions, not in kind. 18 PCA components are needed for 85% of variance,
confirming no dominant behavioural axis.

Part of the low score is also a **dimensionality artifact**: distance ratios
concentrate as dimension grows, so silhouette falls in 18-D even when the
grouping is unchanged. A 90-configuration sweep
(`src/14_clustering_search.py`) measured the frontier rather than asserting it:

| Change | Silhouette | Val macro F1 |
|---|---:|---:|
| Shipped (18 components) | 0.1171 | **0.8021** |
| 4 components | **0.2791** | 0.7796 |
| log1p transform, 4 components | 0.2257 | 0.6097 |

Silhouette can be more than doubled, but never for free. Reducing
dimensionality costs 0.139 macro F1 per unit of silhouette gained; transforming
the feature space costs 1.766 — roughly 13× worse. **18 components ship because
macro F1 is the graded metric**, and that is now an operating point chosen from
a measured frontier rather than an assumption that no frontier exists.

Calinski-Harabasz (72,195 on the current run) remains the more informative
geometric measure at 18 components.

**State rules use fixed thresholds.** The 10-state cascade encodes business
logic, not learned boundaries. It is interpretable and auditable, which is why
it ships, but it will not adapt to seasonal shifts. A supervised or HMM
formulation is the natural successor and is scoped as future work.

**A microscopic segment.** Plateau Cruiser holds ~100 of 500,000 members
(0.02%). Its per-class F1 is high but rests on ~108 test examples and should be
read as unstable. It survives because removing it (k=6 → k=5 merge analysis)
cost more macro F1 elsewhere than it gained.

**Frozen centroids drift.** K-Means was fitted on December 2025. At later
observation dates, members are assigned to December centroids. Real behavioural
shifts in months 13–14 will show up as assignment drift, not as new segments.
There is no automatic refresh trigger; monitoring is scoped as future work.

**Out-of-distribution months — now quantified.** Training covers Feb→Oct
transitions, so December→January predictions sit outside that range. This was
previously a caveat; SHAP turns it into a measurement: **`month_num` is the
second most important feature in the model** (mean |SHAP| 0.714, behind only
`tier_ordinal` at 1.037). A substantial part of the model's confidence comes
from calendar seasonality it has observed exactly once. Treat month-13/14
predictions as extrapolation, and expect the drift monitor to fire.

**The submission date is already drifted.** `src/12_drift_monitor.py` comparing
2025-12-31 against the 2025-12-01 fit window returns RETRAIN_RECOMMENDED:
`recency_days` PSI 5.62, `purchase_count_7d` PSI 0.39 (1.13 → 0.54),
`spend_total_7d` PSI 0.27 (68.99 → 32.79). The 7-day windows halve across the
post-Christmas lull. Outputs at that date are valid but describe a population
the frozen centroids were not fitted on.

**Pre-enrolment rows.** The spine is a fixed 500K panel, so members appear at
observation dates before they enrolled. These carry negative `tenure_days` and
classify as New & Uncertain. This is deliberate (it preserves the 500,000-row
output contract) but means early-month state distributions are dominated by
members who are not yet active.

**Fairness evaluation — run, and clean.**

An earlier revision of this card stated that "the dataset carries no protected
attributes, so no subgroup analysis was performed". That was **wrong**.
`members.parquet` carries `age_band`, `gender`, `region` and `urban_rural`, all
100% populated. The audit has since been run.

Segment distribution at 2025-12-31:

| Gender | Growth Builder | High-Tier | Program Skeptic | Silent Accum |
|---|---:|---:|---:|---:|
| F | 39.7% | 17.5% | 17.5% | 25.3% |
| M | 39.5% | 17.7% | 17.5% | 25.3% |
| O | 39.4% | 17.8% | 17.5% | 25.2% |
| unknown | 39.7% | 17.1% | 17.5% | 25.6% |

Age bands are equally flat — Growth Builder ranges 39.5–39.9% across all six.
**No measurable disparate impact in segment assignment.**

Two caveats on that result. It reflects synthetic data in which demographics
appear to have been generated independently of behaviour, so it will not
necessarily carry over to real data — the audit must be re-run, not assumed.
And PLCC cardholder status still correlates strongly with segment (61.4% in
High-Tier Accelerator vs 13.7% in Silent Accumulator); credit access is not
evenly distributed in the real world, so segment-level treatment differences
can proxy for socioeconomic status even when demographics look balanced.

---

## Ethical Considerations

The model directs marketing spend toward members predicted to disengage. Two
failure modes are worth naming:

- **Under-service by prediction.** Members predicted to stay in a low-value
  segment receive less attention, which can make the prediction self-fulfilling.
  A holdout group that receives baseline treatment regardless of prediction is
  the standard mitigation and is not currently implemented.
- **Correlated proxies.** No protected attribute is used as a feature, but spend
  level, credit line, and channel preference can proxy for socioeconomic status.
  "Price-insensitive, send exclusivity not discounts" is a segmentation
  judgement that should be reviewed by someone accountable for it.

---

## Reproducibility

| Factor | Detail |
|---|---|
| Seed | 42, applied identically across K-Means, XGBoost, numpy, and all splits |
| Hardcoded dates | None in `pipeline.py`; all derived from `--observation_date` |
| External data | None |
| Package versions | Pinned in `requirements.txt`, verified against the working interpreter |
| Parquet engine | `pyarrow` specified explicitly on every read and write |
| Model freeze | K-Means centroids fitted once; never refitted at inference |
| Bundle contract | `scripts/check_model_contract.py`, enforced in CI |

Retrain with:

```bash
python src/07_lifecycle_states.py      # regenerate state labels
python src/08_transition_prediction.py # retrain and rewrite the bundle
```

Regenerating the model rewrites `models/segment_transition_model.pkl`. The
contract checker verifies the new bundle still satisfies what `pipeline.py`
reads, which is the failure that previously made the shipped model
unreproducible from the checked-in training script.

---

## Maintenance

**Retraining trigger.** No automatic trigger exists. Recommended signals:

- Calinski-Harabasz on the current month falls materially below 72,195
- Test macro F1 drops below the persistence baseline on a fresh month pair
- Segment size distribution shifts by more than ~10 percentage points

**Owner.** Kobie × PES University Hackathon 2025 team project — Samarth Hosalli
and Siddarth Reddy, both AI & ML, PES University.
