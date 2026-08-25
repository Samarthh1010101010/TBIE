# TBIE — Results

Every number here is reproducible from `data/train/` plus the two frozen model
files. The command that regenerates each one is listed beside it.

Observation date **2025-12-31** · walk-forward test window **Nov→Dec 2025** ·
seed **42** throughout.

---

## 1. Headline

| | Macro F1 (test, Nov→Dec) |
|---|---:|
| Majority-class baseline | 0.0589 |
| Persistence baseline (`seg_next = seg_curr`) | 0.5651 |
| **TBIE** | **0.8073** · 95% CI [0.8035, 0.8100] |
| | **+0.2422 over persistence** |

Segment membership is strongly autocorrelated month to month, so predicting
that nobody moves already scores 0.565. **The lift over persistence is the
number that means anything**; a bare macro F1 mostly measures autocorrelation.

Confidence interval from 1,000 bootstrap resamples of the test set.

```bash
python src/08_transition_prediction.py
```

### Per class

| Segment | F1 | Calibrated threshold | Support |
|---|---:|---:|---:|
| Growth Builder | 0.668 | 0.45 | 180,343 |
| High-Tier Accelerator | 0.896 | 0.56 | 79,487 |
| Program Skeptic | 0.801 | 0.38 | 86,337 |
| Silent Accumulator | 0.686 | 0.28 | 153,725 |
| Plateau Cruiser | 0.986 | 0.47 | 108 |

Growth Builder and Silent Accumulator set the ceiling on macro F1.
**Plateau Cruiser's 0.986 rests on 108 of 500,000 test rows and should be read
as unstable, not strong.**

### How the decision rule is chosen

| | Validation (Oct→Nov) | Test (Nov→Dec) |
|---|---:|---:|
| argmax | 0.8016 | 0.8168 |
| threshold-calibrated | **0.8083** ← selected | **0.8073** ← reported |

The rule is selected on validation and applied once to test. An earlier version
selected it by comparing **test** scores and reported the winner, which would
have returned 0.8168 here. The ~0.01 gap between those columns was the
inflation that produced the previously-reported 0.8138.

---

## 2. Feature ablation

Both configurations selected identically (weighting and decision rule on
validation), so the comparison is like for like.

| Configuration | Features | Val | Test | 95% CI |
|---|---:|---:|---:|---|
| Point-in-time only | 49 | 0.8060 | 0.8048 | [0.8009, 0.8075] |
| **+ month-over-month change** | **64** | **0.8083** | **0.8073** | [0.8035, 0.8100] |

The lag configuration wins on validation, so it ships. It leads on test too —
but by 0.0025, and **the confidence intervals overlap**. Honest statement: the
change features help directionally and are not statistically distinguishable
from the simpler model at 95% on a single test window.

```bash
python src/08_transition_prediction.py --no-lag-features --tag noLag
```

---

## 3. Clustering: the silhouette / accuracy frontier

Holding transform and k fixed, varying only PCA dimensionality:

| PCA dims | Variance | Silhouette | Davies-Bouldin | Val macro F1 |
|---:|---:|---:|---:|---:|
| **18 (shipped)** | 86.5% | 0.1171 | 1.740 | **0.8021** |
| 8 | 65.5% | 0.1810 | 1.330 | 0.7742 |
| 4 | 50.1% | 0.2791 | 0.953 | 0.7796 |
| 4 + log1p transform | 50.1% | 0.2257 | 1.332 | 0.6097 |

All four rows share one reduced training budget, so they compare with each
other; absolute F1 sits below a full-data run.

**Silhouette can be more than doubled, but not for free.** What the sweep adds
is the exchange rate:

| Lever | Silhouette gain | F1 cost | F1 cost per unit silhouette |
|---|---:|---:|---:|
| PCA 18 → 4 components | +0.162 | −0.0225 | **0.139** |
| log1p transform | +0.109 | −0.1924 | 1.766 |

Reducing dimensionality is roughly **13× more efficient** than transforming the
feature space. Prior experiments only tested the transform and concluded the
trade-off was prohibitive; it is prohibitive *for that lever*.

**Decision: keep 18 components.** Macro F1 is the graded metric. The value of
the sweep is that this is now an operating point chosen from a measured
frontier rather than an assertion.

Validity check: the log1p run scored 0.6097 where an earlier independent
implementation recorded 0.6113 — agreement to 0.0016.

```bash
python src/14_clustering_search.py --stage 1
python src/14_clustering_search.py --stage 2
```

---

## 4. Hyperparameter search — a null result

The Phase 8 hyperparameters were hand-picked and never searched, so a
12-trial Optuna TPE search ran over 10 of them. Objective: validation macro F1.
**The test split was never read by that script.**

| | Val macro F1 |
|---|---:|
| Hand-picked baseline (4M rows) | **0.8083** |
| Best tuned config (4M rows) | 0.8054 (**−0.0029**) |

**The search did not beat the hand-picked configuration.** The production model
is unchanged.

### Why it lost, and why the refit mattered

On the 500K-row search subsample the best trial reached 0.8078 — within 0.0005
of the baseline, and apparently closing. Refitted on all 4M rows it scored
**0.8054**: it got *worse* with eight times more data.

That is not noise. TPE compensated for the small subsample by choosing heavy
regularisation:

| Parameter | Hand-picked | Tuned |
|---|---:|---:|
| `max_depth` | 7 | 10 |
| `reg_lambda` | 1.0 | 9.13 |
| `gamma` | 0.05 | 0.48 |
| `colsample_bytree` | 0.8 | 0.51 |

Optimal regularisation strength scales with dataset size. A configuration tuned
on 12% of the data over-regularises on 100% of it, so subsample search results
do not transfer reliably — in either direction.

An earlier version of this script compared the subsample score directly against
the full-data baseline and exited before the refit, reporting "search did not
beat the baseline". It reached the right answer through an invalid comparison.
The gate now always refits before concluding (`--refit-from` reruns just that
step against a saved search).

**What this is worth saying:** a search that finds nothing is a real result. It
means the original hyperparameters were already near-optimal, and that the
accuracy gains available in this project came from fixing correctness bugs, not
from tuning.

```bash
python src/13_tune_hyperparams.py --trials 12 --subsample 500000
python src/13_tune_hyperparams.py --refit-from outputs/hyperparameter_search.json
```

---

## 5. Probability calibration

The pipeline emits `prob_S01`…`prob_S05` and the targeting logic multiplies
them by dollar values, so they were tested rather than assumed.

| | Test (Nov→Dec) |
|---|---:|
| Mean Brier | 0.0673 |
| Mean expected calibration error | **0.0184** |
| Log-loss | 0.5416 |

Reliability, Growth Builder — predicted vs what actually happened:

| Model says | Actually happened | Members |
|---:|---:|---:|
| 0.10 | 0.12 | 56,689 |
| 0.30 | 0.32 | 28,737 |
| 0.50 | 0.53 | 20,105 |
| 0.70 | 0.71 | 22,994 |
| 0.90 | 0.93 | 26,326 |

**Isotonic recalibration made test reliability worse** (ECE 0.0184 → 0.0375,
Brier 0.0673 → 0.0713, macro F1 −0.0054), so raw probabilities ship. Recorded
rather than discarded: the model is already well calibrated despite the class
reweighting, which is not the usual outcome.

```bash
python src/09_calibration.py
```

---

## 6. Targeting on money rather than F1

Contact when `P(adverse move) × value_at_risk × recovery` clears the contact
cost. Segment values come from observed 180-day spend, not assumption.

| Policy | Contacted | Cost | Recovered | Profit | ROI |
|---|---:|---:|---:|---:|---:|
| Contact everyone | 500,000 | $2,500,000 | $1,519,584 | −$980,416 | −0.39x |
| F1-optimal threshold | 91,209 | $456,045 | $1,069,272 | $613,227 | 1.34x |
| **Expected-value ranked** | 110,853 | $554,265 | $1,325,981 | **$771,716** | **1.39x** |

| | Precision | Recall |
|---|---:|---:|
| F1-optimal | 31.5% | 76.9% |
| Expected-value | 16.6% | 49.3% |

**The EV policy has worse precision and recall and makes $158,489 more.** A 90%
chance of losing $20 is a worse use of a $5 touchpoint than a 15% chance of
losing $900; classification metrics cannot see that difference.

Sensitivity to the recovery-rate assumption:

| Recovery rate | Contacted | Profit | ROI |
|---:|---:|---:|---:|
| 2% | 39,752 | −$20,075 | −0.10x |
| 5% | 86,952 | $194,353 | 0.45x |
| **10%** (default) | 110,853 | $771,716 | 1.39x |
| 20% | 155,827 | $2,134,031 | 2.74x |
| 30% | 179,634 | $3,542,477 | 3.94x |

> **The recovery rate cannot be estimated from this dataset.** There is no
> randomised holdout in it, so no counterfactual exists. Every currency figure
> above is conditional on that input. Break-even sits between 2% and 5%.

> **Correcting an earlier claim.** Previous revisions of this document stated
> that no A/B or campaign-response data existed. That was wrong.
> `engagement_events.parquet` carries `campaign_id`, `campaign_type` and
> `campaign_response` (sent / opened / clicked / none) on **12,759,405 rows
> (35.9%)**. It is not a randomised holdout, so it does not license causal
> claims on its own — but it does support modelling response propensity and a
> matched-control estimate, which would replace the assumed recovery rate with
> a measured one. See `AUDIT.md`.

```bash
python src/10_cost_thresholds.py --contact-cost 5 --recovery-rate 0.10
```

---

## 7. What the model actually uses

Exact TreeSHAP on 25,000 sampled test rows.

| Rank | Feature | mean \|SHAP\| |
|---:|---|---:|
| 1 | `tier_ordinal` | 1.037 |
| 2 | `month_num` | 0.714 |
| 3 | `spend_total_90d` | 0.405 |
| 4 | `tier_changes_count` | 0.393 |
| 5 | `spend_per_purchase_90d` | 0.216 |
| 6 | `months_since_last_tier_change` | 0.161 |
| 9 | `seg_curr` | 0.106 |

Two findings worth stating plainly:

- **Tier structure dominates, not recency.** An earlier revision listed
  `seg_curr` as the strongest predictor based on XGBoost gain importance. SHAP
  puts it ninth. Where gain and SHAP disagree, SHAP measures the contribution
  to actual predictions.
- **`month_num` at rank 2 is a limitation, not a feature.** The model leans
  heavily on calendar seasonality it has observed exactly once. This quantifies
  the month-13/14 extrapolation risk instead of leaving it as a caveat.

```bash
python src/11_shap_explain.py
```

---

## 8. Drift at the submission date

Comparing 2025-12-31 against the 2025-12-01 fit window:

| Feature | PSI | Dec 1 → Dec 31 |
|---|---:|---|
| `recency_days` | 5.62 | 6.07 → 9.08 |
| `months_since_last_tier_change` | 1.15 | 8.91 → 9.85 |
| `purchase_count_7d` | 0.39 | 1.13 → 0.54 |
| `spend_total_7d` | 0.27 | 68.99 → 32.79 |

**Verdict: RETRAIN_RECOMMENDED.** The 7-day windows halve because 31 December
sits in the post-Christmas lull. Outputs at that date remain valid but describe
a population the frozen centroids were not fitted on.

```bash
python src/12_drift_monitor.py --current 2025-12-31
```

---

## 9. Output at the submission date

| Segment | Members | Share | PLCC cardholders |
|---|---:|---:|---:|
| Growth Builder | 198,035 | 39.61% | 34.9% |
| High-Tier Accelerator | 87,688 | 17.54% | 61.4% |
| Program Skeptic | 87,505 | 17.50% | 25.7% |
| Silent Accumulator | 126,655 | 25.33% | 13.7% |
| Plateau Cruiser | 117 | 0.02% | 29.9% |

Segment assignment confidence — normalised margin between the nearest and
second-nearest centroid, spanning the full [0, 1] range:

| | |
|---|---:|
| Mean | 0.132 |
| Max | 0.873 |

A low mean is the honest reading of segments that genuinely overlap, and is
consistent with silhouette 0.117. The previous formula was bounded to
[0.5, 1.0], so an uninformative assignment printed as "0.50 confidence" — which
looked like a coin flip but was actually the floor.

```bash
python pipeline.py --data_dir ./data/train/ \
  --observation_date 2025-12-31 --output_dir ./outputs/
```

---

---

## 10. Contact eligibility, consent and frequency (`contact_eligibility.csv`)

A sixth output. The five required submission files are unchanged.

Earlier revisions emitted a `recommended_activation` naming a channel with no
reference to consent, account status, or how recently the member had been
contacted. At 2025-12-31 that meant **222,259 members were pointed at a channel
they had opted out of** and ~20,000 closed or fraud-flagged accounts were being
targeted. In a real programme that is a CAN-SPAM / GDPR Art. 21 / TCPA exposure,
not a data-quality nit.

| | Members | Share |
|---|---:|---:|
| **Targetable** | **319,271** | **63.9%** |
| Frequency-capped (>= 6 contacts in 30d) | 87,405 | 17.5% |
| No consent on any channel | 81,026 | 16.2% |
| Account closed | 9,898 | 2.0% |
| Fraud-flagged | 2,400 | 0.5% |
| *Dormant → reactivation track* | *6,312* | *1.3%* |

**Consent violations after the fix: 0** (from 222,259).

Channel mix of valid recommendations: email 198,523 · push 191,607 · sms 16,546.

### Design notes

- **Suppression is a hard gate applied after modelling, never a feature.** A
  closed account is still scored — its predicted transition is still
  information — but the recommendation becomes `DO NOT CONTACT: Account closed`.
- **Channel falls back rather than dropping the member.** A Lapse Risk member
  who blocked email but allows push still gets a push.
- **A null opt-in is not consent.** Opt-in is an affirmative act;
  `test_null_optin_is_not_consent` pins this.
- **Frequency capping** uses `campaign_id` history computed strictly from events
  dated <= observation_date.

### Per-member recovery rate

The expected-value model previously applied one assumed recovery rate to all
500,000 members. It is now scaled by observed relative responsiveness, using
per-member response rates shrunk toward the population mean (empirical Bayes,
prior strength 20 pseudo-touches).

| | Effective recovery rate |
|---|---:|
| Assumed base | 10.0% |
| p10 member | 8.3% |
| median member | 10.2% |
| p90 member | 11.0% |

**The spread is narrow, and that is a finding, not a bug.** The population
response rate in this data is 80.5%, so there is little room to differentiate;
shrinkage correctly pulls thin evidence to the mean. On real data with a 20-30%
response rate the spread would be far wider.

The multiplier is clipped to [0.25x, 2.5x]. Campaign exposure was **not**
randomised, so this is relative propensity, not causal lift — a member who
historically opens twice as often is a better bet, but reading 10x recoverable
value out of non-randomised exposure would be unsupportable.

```bash
python pipeline.py --data_dir ./data/train/   --observation_date 2025-12-31 --output_dir ./outputs/
```

---

## 11. Engineering

| | |
|---|---:|
| Tests | 113 passing |
| Tracked repo size | 15 MB (was 271 MB) |
| Lint | ruff clean |
| CI | tests · lint · model-bundle contract · shared-module imports |

Four train/serve skews were found and fixed, all from the same root cause —
logic living in two files that drifted apart:

| Skew | Impact |
|---|---|
| `y_curr` encoding: priority order in training, alphabetical at inference | Feature scrambled at serve time |
| `recency_risk`: NaN filled with 0 in training, 999 at inference | 1,814 never-purchased members given the opposite signal |
| NaN passed to XGBoost at inference, filled with 0 in training | 95,328 members (19%) routed down the "missing" tree branch |
| State cascade thresholds drifted between the two copies | Value Maximizer required 0.21 at serve vs 0.10 in training |

Six shared concerns are now single-sourced in `src/utils/`, and CI fails if any
stops importing standalone.

Two further defects that changed results:

- **`tenure_days` was absent from the feature files**, so the priority-1 state
  rule (`tenure < 90`) never fired. `New & Uncertain` appeared in 1 of 12
  training months and 0 in the rest.
- **Ghost members were never excluded.** Detection matched only an
  `MBR_GHOST_` id prefix; the actual ghosts are 88,717 orphan ids present in
  activity but absent from `members.parquet`. The prefix rule matched zero.

---

## 12. Are the segments real? — cluster stability (ARI)

Phase 8 predicts a label that Phase 6 invented. If that label is an artefact of
one particular K-Means fit, the whole transition model is predicting noise. This
section measures it instead of assuming either way.

Adjusted Rand Index compares two partitions of the same members, ignores label
permutation, and is corrected for chance: 1.0 identical, ~0.0 no better than
random.

```bash
python src/15_cluster_stability.py --features-dir <features>
```

| Test | What it varies | mean ARI | min | max |
|---|---|---:|---:|---:|
| Seed | `random_state`, identical data | 0.8477 | 0.5620 | 0.9992 |
| Subsample | 80% random draws, shared members | 0.8615 | 0.6229 | 0.9956 |
| Temporal (frozen Dec scaler+PCA) | month, K-Means refit only | 0.3588 | 0.1071 | 0.6615 |
| Temporal (full refit per month) | month, scaler+PCA+K-Means | 0.2748 | 0.2032 | 0.3328 |

### The shipped seed is the minority solution

The seed mean hides the actual finding. Every pair is either ~0.57 or ~0.98,
and the split is not random:

| Pair | ARI |
|---|---:|
| 42 vs 1 / 7 / 123 / 2024 / 31337 | 0.562 – 0.571 |
| every pair among 1, 7, 123, 2024, 31337 | 0.972 – 0.999 |

Five of six seeds converge on one partition. **Seed 42 — the one that ships —
converges somewhere else**, and `n_init=10` does not rescue it because all ten
initialisations inside that seed find the same basin.

This is the expected behaviour of a weakly separated space. Silhouette is 0.1196:
the segments genuinely overlap, so several near-equivalent 5-way cuts exist and
which one is reached is close to arbitrary. The production segmentation is *a*
valid cut, not *the* cut.

Not changed, deliberately: re-seeding would move every shipped artefact — segment
ids, profiles, the frozen transition model, and every number in this document.
Recording which solution was reached is more useful than quietly swapping it.

### Structure survives resampling; boundaries do not survive time

Subsampling at 0.86 says the broad shape does not depend on which members were
drawn. Temporal at 0.36 says something different: **re-deriving the segments in a
different month produces materially different segments.**

Two thirds of that is real. The first temporal run refit scaler and PCA per month,
so each month's labels lived in a different feature space — that conflates unstable
clustering with a moved representation. Freezing December's scaler and PCA and
refitting only K-Means raises the mean from 0.2748 to 0.3588. The remainder is the
clustering itself.

**This does not break the production pipeline**, which fits K-Means exactly once on
December and scores every other month through frozen centroids, so no member is
ever assigned by a re-derived partition. What it does mean is that the segment
definitions are a **December-specific construct, not a durable property of the
population**. Discovery re-run in March would have produced a different set of five
segments, given different names.

The honest consequence for Section 1: the transition model's 0.8073 macro F1 is
predicting movement inside a fixed, arbitrary-but-frozen coordinate system. That
is a legitimate task — the coordinate system is the product — but it is not the
same claim as predicting a natural, recoverable structure.

### What would move these numbers

- k=5 was floored by the brief. Silhouette 0.1196 and DB 1.77 both say five
  Voronoi cells is a coarse description of this space.
- A soft assignment (GMM) would report the overlap instead of hiding it behind a
  hard label; Section 9's mean assignment confidence of 0.132 is already saying so.
- Consensus clustering across seeds would produce a partition that does not depend
  on landing in one basin.
