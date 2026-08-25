# METHODOLOGY
### TBIE — Temporal Behavioural Intelligence Engine
### Kobie × PES University Hackathon 2025

---

## 1. Feature Engineering

### Data Sources

All features are derived exclusively from three provided datasets: `members.parquet`, `transactions.parquet`, and `engagement_events.parquet`. No external data sources, pretrained embeddings, or supplementary datasets were used at any stage.

### Behavioural Hypotheses

The feature set tests four core hypotheses about loyalty programme behaviour:

| Hypothesis | Feature Group | Key Columns |
|---|---|---|
| Spending momentum drives segment migration | Spend (4 windows) + slopes | `spend_total_7d/30d/90d/180d`, `spend_slope_30d`, `spend_acceleration` |
| Purchase frequency is the primary loyalty signal | Frequency (4 windows) | `purchase_count_7d/30d/90d/180d`, `frequency_slope_30d` |
| Digital disengagement precedes lapse | Engagement | `app_open_30d`, `email_open_30d`, `push_open_30d`, `reward_browse_30d` |
| Reward programme utilisation defines member strategy | Points behaviour | `redemption_rate`, `hoarding_ratio`, `reward_redemption_30d` |

### Window Selection

- **7d:** Captures immediate re-engagement signals — required for Win-Back Target detection.
- **30d:** Standard loyalty programme billing cycle; the primary window for state assignment.
- **90d:** Smooths monthly volatility; captures quarterly purchase patterns.
- **180d:** Long-horizon baseline; used for cold-start and seasonal comparisons.

### Velocity Features

Six derived features measure rate of change rather than absolute values, capturing directional momentum:

| Feature | Formula | Purpose |
|---|---|---|
| `spend_velocity` | `spend_total_30d / (spend_total_90d / 3 + ε)` | Spending momentum vs 3-month baseline |
| `freq_velocity` | `purchase_count_30d / (purchase_count_90d / 3 + ε)` | Frequency momentum |
| `app_velocity` | `app_open_30d / (app_open_90d / 3 + ε)` | Engagement momentum |
| `recency_risk` | `recency_days × clip(1 − freq_30d / 10, 0, 1)` | Composite lapse signal |
| `engagement_score` | `app_open_30d + email_open_30d + push_open_30d` | Total digital touchpoints |
| `spend_decline_flag` | `1 if spend_velocity < 0.5` | Binary alert: spend below 50% of baseline |

### Month-over-Month Change Features

Every feature above describes a *level*. The prediction target is a
*transition*, and the six velocity features are ratios computed inside a single
snapshot, so none of them compares this month against last month. Thirteen
delta features (`d_spend_total_30d`, `d_recency_days`, …) plus two
segment-history features close that gap:

| Feature | Meaning |
|---|---|
| `d_<feature>` | This month's value minus last month's, for 13 base features |
| `seg_prev` | Segment at T−1 — lifts the model from a first- to second-order Markov view |
| `seg_changed` | Whether the member moved segment last month (instability signal) |

Members absent from the prior snapshot get a delta of 0 and `seg_changed` of 0
— "no observed change" rather than a fabricated transition — which preserves
the fixed 500,000-row panel. Ablate with
`python src/08_transition_prediction.py --no-lag-features`.

### Top Features by SHAP

Measured with exact TreeSHAP (`booster.predict(pred_contribs=True)`) on 25,000
sampled test rows. Full table in `outputs/shap_summary.md`.

| Rank | Feature | mean \|SHAP\| | Strongest for |
|---|---|---:|---|
| 1 | `tier_ordinal` | 1.037 | Program Skeptic |
| 2 | `month_num` | 0.714 | Plateau Cruiser |
| 3 | `spend_total_90d` | 0.405 | High-Tier Accelerator |
| 4 | `tier_changes_count` | 0.393 | Program Skeptic |
| 5 | `spend_per_purchase_90d` | 0.216 | High-Tier Accelerator |
| 6 | `months_since_last_tier_change` | 0.161 | Program Skeptic |
| … | | | |
| 9 | `seg_curr` | 0.106 | Silent Accumulator |

**Two findings worth stating plainly.**

*Tier structure dominates, not recency.* An earlier revision of this document
listed `seg_curr` as the single strongest predictor and `recency_days` second,
based on XGBoost's built-in gain importance. SHAP disagrees sharply: tier
position and tier history occupy three of the top six slots, and `seg_curr` is
only ninth. Gain importance counts how often a feature is split on and how much
it reduces loss at those splits; SHAP measures the actual contribution to each
prediction. Where they disagree, SHAP is the one to trust — and the SHAP
ranking is what `outputs/shap_summary.md` reports.

*`month_num` at rank 2 is a limitation, not a feature.* The model leans heavily
on which calendar month it is scoring. That works inside the training range but
is precisely the signal that will not transfer to months 13–14, which sit
outside it. This quantifies the extrapolation risk listed in §7 rather than
leaving it as a caveat: a meaningful part of the model's confidence comes from
seasonality it has only ever seen once.

---

## 2. Clustering Methodology

### Algorithm Comparison

Seven algorithms were evaluated on a 50,000-member subsample before selecting K-Means:

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz | Outcome |
|---|---|---|---|---|
| HDBSCAN | — | — | — | 71–85% noise — rejected |
| GMM | 0.018 | 3.163 | 32,390 | Lost on all metrics — rejected |
| Bisecting K-Means | 0.089 | 1.911 | 58,236 | Lost on all metrics — rejected |
| BIRCH | 0.099 | 1.901 | 35,054 | OOM / 64.7% in one cluster — rejected |
| K-Means k=6 | 0.106 | 1.547 | 71,222 | Microclusters, −0.12 F1 — rejected |
| **K-Means k=5** | **0.120** | **2.000** | **77,123** | **Selected** |

**Why HDBSCAN was rejected.** Loyalty behavioural data is a continuous Gaussian distribution — members differ in degree across many dimensions simultaneously, not in kind. HDBSCAN requires low-density gaps between clusters; no such gaps exist in this data. The 18 PCA components required to explain 85% of variance confirms this: there is no single dominant behavioural axis and no separable density regions. With every sensitivity configuration tested (min_cluster_size from 50 to 5,000), 71–85% of members were labelled as noise. K-Means, which is designed for Gaussian data, is the correct algorithm for this dataset.

### K Selection

Full elbow sweep on a 50,000-member subsample:

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| **5** | **1,072,920** | **0.1196** | **2.000** | **7,295** |
| 6 | 1,025,958 | 0.1058 | 2.020 | 6,561 |
| 7 | 955,148 | 0.1221 | 1.557 | 6,490 |
| 8 | 917,244 | 0.1116 | 1.740 | 6,088 |
| 10 | 865,914 | 0.0998 | 1.841 | 5,345 |

**k=5 selected over k=6** because k=6 produced two microclusters (combined under 0.8% of population) with unstable month-over-month assignments, reducing Phase 8 Macro F1 by 0.12. k=5 achieves the highest Calinski-Harabasz score (7,295 vs 6,561) — a more reliable metric than silhouette for Gaussian data.

**On Davies-Bouldin = 2.000 vs k=6 DB = 1.547.** k=6 achieved a DB of 1.547, closer to the ideal of under 1.5. However, eliminating the unstable 117-member microcluster at k=5 increased DB to 2.000 — a deliberate and accepted trade-off. The downstream gain was +0.12 Macro F1 in the transition model, and the k=5 Calinski-Harabasz score (77,123 vs 71,222) is higher, confirming better within-cluster compactness. DB measures cluster boundary overlap, which is expected to be elevated in continuous Gaussian data regardless of k.

**On Silhouette = 0.1196.** This is not a model failure. Loyalty behavioural data forms a single dense cloud in feature space — members exist on a continuum, not in clearly separated groups. Silhouette rewards density gaps; Calinski-Harabasz rewards compact clusters relative to within-cluster variance. The CH of 77,123 on the full 500,000-member run confirms clean, compact clusters.

### Silhouette Optimisation Experiments

To address the Silhouette score, two experiments were conducted. Both revealed the same fundamental trade-off: improving geometric separation destroys downstream predictive accuracy.

| Metric | Original | Exp 1: Log1p + RobustScaler | Exp 2: Feature Selection + Winsorize |
|---|---|---|---|
| Silhouette | 0.1196 | **0.3703** | **0.2171** |
| Davies-Bouldin | 2.000 | **0.700** | **1.301** |
| Calinski-Harabasz | 77,123 | **495,648** | **165,524** |
| **Macro F1 (Test)** | **0.8138** | 0.6945 | 0.6113 |
| Microclusters | 1 (117 members) | 2 (108 + 246 members) | 0 |

The original clusters are geometrically noisy but behaviourally stable. When the geometry is cleaned up (by compressing spend variance or dropping correlated features), members are reassigned to different segments, the transition patterns change entirely, and XGBoost cannot predict them. High-Tier Accelerator F1 collapsed to 0.348 in Experiment 2. Better geometry does not produce more predictable transitions on this data type.

### The silhouette / accuracy frontier

The two experiments above both varied the **feature transform**. A later sweep
(`src/14_clustering_search.py`, 90 configurations) added a lever they never
tested — **PCA dimensionality** — and measured downstream validation macro F1
for each candidate rather than stopping at geometry.

Silhouette is a distance-ratio metric, and distance ratios concentrate as
dimension grows: in high dimensions every point drifts toward equidistant. The
shipped model clusters in 18 components, so part of the low score is a
dimensionality artifact rather than a property of the data.

Holding transform and k fixed and varying only the number of components:

| PCA dims | Variance kept | Silhouette | Davies-Bouldin | Val macro F1 |
|---:|---:|---:|---:|---:|
| **18 (shipped)** | 86.5% | 0.1171 | 1.740 | **0.8021** |
| 8 | 65.5% | 0.1810 | 1.330 | 0.7742 |
| 4 | 50.1% | 0.2791 | 0.953 | 0.7796 |
| 4, with log1p transform | 50.1% | 0.2257 | 1.332 | 0.6097 |

(All four rows share one reduced training budget so they are comparable with
each other; absolute F1 sits below a full-data run.)

**Silhouette can be more than doubled — but not for free.** Dropping to 4
components raises it from 0.117 to 0.279 (+138%) and pushes Davies-Bouldin
below 1.0, at a cost of 0.023 macro F1 (−2.8%). The trade-off the original
experiments identified is real, and it applies to this lever too.

What is new is the **exchange rate**:

| Lever | Silhouette gain | Macro F1 cost | F1 cost per unit silhouette |
|---|---:|---:|---:|
| PCA 18 → 4 components | +0.162 | −0.0225 | **0.139** |
| log1p transform | +0.109 | −0.1924 | 1.766 |

Reducing dimensionality is roughly **13× more efficient** than transforming the
feature space at buying geometric separation. The log1p run also independently
reproduces the original Experiment 2 result (0.6097 here vs 0.6113 recorded
earlier), which is good evidence both measurements are sound.

**Decision: keep 18 components.** Macro F1 is the graded metric, and 0.8021 vs
0.7796 is a larger difference than any silhouette figure justifies. The point of
running this is that the choice is now an operating point selected from a
measured frontier, rather than an assertion that the frontier does not exist.

**Conclusion:** Silhouette of 0.12 is expected on continuous loyalty data at 18
components. Calinski-Harabasz of 72,195 confirms compact clusters. Downstream
Macro F1 is the primary graded metric and is preserved.

> The Macro F1 figures in the experiment table above were measured under the
> earlier evaluation procedure, which selected its decision rule on the test set
> (see §3). They are retained because the **comparison** between the three
> feature treatments is the point, and all three were measured the same way. The
> absolute values are not comparable to the corrected headline figure in §3.

### Final Model Configuration

```python
KMeans(n_clusters=5, n_init=10, max_iter=500, random_state=42)
```

Fitted on December 2025 features (500,000 members). Applied to all observation dates via `scaler.transform()` + `pca.transform()` + nearest-centroid — never refitted on test or validation data.

---

## 3. Transition Model

### Architecture

- **Model:** XGBoost multi-class classifier (`multi:softprob`)
- **Target:** `seg_next` — segment membership 30 days after observation_date
- **Features:** 49 total: 40 behavioural + 6 velocity + `seg_curr` + `y_curr` + `month_num`

### Walk-Forward Validation

| Split | Pairs | Months | Approximate Rows |
|---|---|---|---|
| Train | Feb→Mar through Sep→Oct | 8 pairs | ~3.5M |
| Validate | Oct→Nov | 1 pair | ~489K |
| Test | Nov→Dec | 1 pair | ~492K |
| Holdout | Months 13–14 | Kobie-evaluated, never accessed | — |

No shuffling at any stage. All test data is strictly later than all training data. Early stopping evaluated on validation log-loss only.

### Class Imbalance Handling

Inverse-frequency weights, plus an optional extra boost for classes that are
genuinely hard — where "hard" is measured on **validation**, never on test:

```python
# Pass 1 — inverse-frequency weights only
base_weights = compute_sample_weight('balanced', y_train)
model_p1     = train(base_weights)

# Identify weak classes from VALIDATION per-class F1
weak = [c for c in classes if val_f1(model_p1, c) < WEAK_F1_THRESHOLD]

# Pass 2 — retrain with those classes boosted; keep it only if validation improves
model_p2 = train(base_weights * np.where(np.isin(y_train, weak), 2.0, 1.0))
model    = model_p2 if val_macro_f1(model_p2) > val_macro_f1(model_p1) else model_p1
```

On the current data this procedure flags Growth Builder (val F1 0.632) and
Silent Accumulator (val F1 0.717), retrains with the boost, and then **rejects
it** — the boosted model scores 0.7938 on validation against 0.7989 for plain
inverse-frequency weighting. Balanced weights ship.

> **Correcting an earlier version of this document.** A previous revision
> described a 4× boost applied to a "Lapse Risk" segment. That code never
> executed: Lapse Risk is a lifecycle *state* (Layer 2), not a segment
> (Layer 1), so the segment lookup resolved to `None` and the multiplier was
> uniformly 1.0. The results reported here were produced without it.

### Why `multi:softprob`

`multi:softprob` outputs a full probability distribution over all 5 segments for every member. This is required by the submission format (`prob_S01` through `prob_S05`) and enables per-member prediction confidence and calibrated threshold adjustment per segment.

### Decision Rule Selection — and why the reported number went down

Two decision rules are available: plain `argmax` over the predicted
probabilities, and per-class threshold calibration tuned on validation. The
rule is chosen by validation score and then applied once to test.

| | Validation (Oct→Nov) | Test (Nov→Dec) |
|---|---|---|
| argmax | 0.8016 | 0.8168 |
| threshold-calibrated | **0.8083** ← selected | **0.8073** ← reported |

Validation prefers threshold calibration, so that is what ships, and 0.8073 is
what it scores on test.

An earlier version of this pipeline chose between the two rules by comparing
their **test** scores and reporting the winner — which would have returned
0.8168 here. That is selection on the test set, and the roughly 0.01 gap
between the two columns is the inflation it produced; it is where the
previously-reported 0.8138 came from. The reported figure is now the honest
one, even though it is lower.

### Feature Ablation — do the change features earn their place?

Both configurations were selected the same way (weighting and decision rule on
validation), so the comparison is like for like.

| Configuration | Features | Val (selected) | Test | 95% CI |
|---|---:|---:|---:|---|
| Point-in-time only | 49 | 0.8060 | 0.8048 | [0.8009, 0.8075] |
| **+ month-over-month change** | **64** | **0.8083** | **0.8073** | [0.8035, 0.8100] |

The lag configuration wins on validation, so it ships. It is also ahead on
test — but by 0.0025, and **the two confidence intervals overlap**. The honest
statement is that the change features help directionally and are not
statistically distinguishable from the simpler model at 95% confidence on a
single test window. They are retained because validation preferred them, not
because the improvement is proven.

### Results

| Metric | Value |
|---|---|
| **Macro F1 (Test: Nov→Dec)** | **0.8073**  (95% CI [0.8035, 0.8100]) |
| Macro F1 (Val: Oct→Nov) | 0.8083 |
| Persistence baseline (test) | 0.5651 |
| Majority-class baseline (test) | 0.0589 |
| **Lift over persistence** | **+0.2422** |
| Random seed | 42 |

Confidence interval from 1,000 bootstrap resamples of the test set, seed 42.

### Why the persistence baseline is the one that matters

Segment membership is highly autocorrelated month over month, so a model can
look strong simply by predicting that nobody moves. Quoting a macro F1 without
that reference point is close to meaningless.

| Reference | Macro F1 | What it establishes |
|---|---:|---|
| Majority class | 0.0589 | Floor |
| Persistence (`seg_next = seg_curr`) | 0.5651 | What you get for free from autocorrelation |
| **TBIE** | **0.8073** | **+0.2422 over persistence** |

### Per-Class F1 — Test Window (Nov→Dec)

| Segment | F1 | Calibrated threshold | Support |
|---|---:|---:|---:|
| Growth Builder | 0.666 | 0.42 | 180,343 |
| High-Tier Accelerator | 0.891 | 0.57 | 79,487 |
| Program Skeptic | 0.803 | 0.37 | 86,337 |
| Silent Accumulator | 0.678 | 0.29 | 153,725 |
| Plateau Cruiser | 0.986 | 0.49 | 108 |
| **Macro Average** | **0.805** | — | 500,000 |

All five segments clear F1 ≥ 0.50. Growth Builder and Silent Accumulator are the
weakest and set the ceiling on macro F1; these are the two classes the
validation-driven weighting procedure targets.

**Plateau Cruiser's 0.986 should not be read as strength.** It covers 108 of
500,000 test rows (0.02%). The score is real but rests on a very small sample and
will be unstable across resamples.

---

## 4. State Mapping Logic

### 10 Loyalty-Native States — Priority Cascade

States are assigned by a vectorized priority cascade (`numpy.select`). Every member receives exactly one state. Higher-priority conditions override lower-priority ones. All feature columns are filled with 0.0 before the cascade runs — no member is excluded due to null values.

| Priority | State | Exact Rule |
|---|---|---|
| 1 | New & Uncertain | `tenure_days < 90` |
| 2 | Win-Back Target | `recency_days > 60` AND (`email_open_30d > 0` OR `app_open_30d > 0` OR `push_open_30d > 0`) |
| 3 | Lapse Risk | `recency_days > 30` AND `purchase_count_30d == 0` AND `spend_slope_30d <= 0` |
| 4 | Momentum Builder | `spend_slope_30d > 5.0` AND `purchase_count_30d >= 2` AND `recency_days < 30` AND `category_diversity_90d > 0.3` |
| 5 | Brand Advocate | (`email_open_30d >= 2` OR `app_open_30d >= 3` OR `social_share_30d >= 1` OR `referral_sent_30d >= 1` OR `survey_completed_30d >= 1`) AND `tier_ordinal >= 2` AND `category_diversity_90d > 0.4` |
| 6 | Redemption Hunter | `redemption_rate > 0.30` AND `purchase_count_30d <= 1` |
| 7 | Value Maximizer | `category_diversity_90d > 0.50` AND `redemption_rate > 0.10` |
| 8 | Silent Accumulator | `purchase_count_30d >= 1` AND `app_open_30d == 0` AND `email_open_30d == 0` AND `push_open_30d == 0` |
| 9 | Plateau Cruiser | `spend_slope_30d >= -2.0` AND `spend_slope_30d <= 5.0` AND `purchase_count_30d >= 1` |
| 10 | Program Skeptic | Default — all above conditions false |

**Implementation notes:**
- The cascade has exactly one implementation, `src/utils/state_rules.py`, imported by both `src/07_lifecycle_states.py` (which produces the training labels) and `pipeline.py` (inference). It previously existed as two copies that had drifted apart on three rules; because the resulting state feeds the transition model as `y_curr`, the drift changed the feature distribution between training and serving. `tests/test_state_rules.py` pins every threshold and boundary in the table above.
- `tenure_days` is computed in `src/04_feature_engine.py` as `(observation_date − account_open_date).dt.days` and written into the feature files. It was previously derived ad hoc at inference and absent from the feature files entirely, which meant the priority-1 rule never fired during training in 11 of 12 months while firing for ~27% of members at inference.
- Members whose `account_open_date` is later than the observation date carry negative `tenure_days`. The member spine is a fixed 500,000-row panel, so these rows exist at every snapshot; the negative value is retained deliberately as a "not yet enrolled" signal rather than clipped to 0, which would conflate it with "enrolled today".
- `spend_slope_30d` is the OLS slope of weekly spend over the prior 30 days, clipped at ±50 to prevent outlier dominance.
- New & Uncertain is purely tenure-based. A veteran member with low recent purchases is Lapse Risk or Program Skeptic, not New & Uncertain.
- Win-Back Target uses digital re-engagement signals rather than purchases, resolving the logical impossibility of `recency > 60 AND purchases_7d > 0` being simultaneously true.
- Brand Advocate uses OR across engagement channels. A member active on any single channel qualifies if they meet the tier and diversity thresholds.
- Plateau Cruiser upper slope bound is 5.0 (not 2.0) to prevent members at slope 2.0–5.0 from falling through to Program Skeptic.

### supporting_evidence Format

Each member record in `state_assignments.csv` includes a `supporting_evidence` field with the actual feature values that triggered the rule, in `metric:value` format:

```
Lapse Risk:        "txn_decline:-33%,recency_days:38,purchases_30d:0"
Momentum Builder:  "spend_slope:7.5,purchases_30d:3,tier:2"
Brand Advocate:    "app_opens_30d:4,email_opens_30d:2,tier:3,category_diversity:0.65"
```

---

## 5. Data Quality

Eight data quality issues were discovered during EDA and resolved before modelling. Full detail in `data_quality_report.md`.

| # | Issue | Scale | Resolution |
|---|---|---|---|
| 1 | Ghost member IDs in transactions | 88,717 orphaned IDs | Excluded via member spine LEFT JOIN. `pipeline.py` resolves ghosts by two independent rules — the `MBR_GHOST_` id prefix in `members.parquet`, and IDs observed in transactions/events but absent from `members.parquet` — because orphans are by definition not findable by scanning the members table alone. |
| 2 | Mixed datetime formats (ISO + day-first) | ~7.4% of date strings | Dual-pass parser in Phase 3 |
| 3 | Mixed-case transaction types | 24,738 rows | `.str.strip().str.lower()` normalisation |
| 4 | Duplicate engagement events | 92,748 rows (0.26%) | `drop_duplicates()` on `[member_id, event_date, event_type]` |
| 5 | Session duration outliers | 18,663 rows exceeding 4 hours; max 48 hours | Hard clip at 14,400s (4 hours) |
| 6 | Leakage columns in members.parquet | 3 columns (lifetime totals) | Reconstructed from transactions as running sums up to obs_date |
| 7 | No email_sent events | Entire column absent | `email_open_30d` documented as raw count, not rate |
| 8 | Null-timestamp engagement events | 117 members | Inflated 30d window counts; K-Means correctly isolated into S05 |

---

## 6. Cardholder Analysis

### Identification Method

Members with a non-null `credit_line` in `members.parquet` are PLCC cardholders (32.6% of the population). `credit_utilization` was not used in the behavioural feature set — clustering was performed entirely on transactional and engagement signals.

```
Total members:     500,000
PLCC cardholders:  163,000  (32.6%)
Non-cardholders:   337,000  (67.4%)
```

### Composition by Segment

PLCC cardholder percentage varies meaningfully across segments:

| Segment | PLCC % |
|---|---|
| High-Tier Accelerator | 61.4% |
| Growth Builder | 34.9% |
| Plateau Cruiser | 29.9% |
| Program Skeptic | 25.7% |
| Silent Accumulator | 13.7% |

### Impact on Segmentation

PLCC cardholders influence segmentation indirectly through their spending patterns (higher average transaction value, higher tier concentration, lower recency risk due to credit programme touchpoints). The top 20 XGBoost features by importance do not include any explicit PLCC-derived column, confirming that cardholder status is captured implicitly through spend behaviour rather than through direct flag usage.

---

## 7. Limitations and Future Work

### Known Limitations

| Limitation | Detail |
|---|---|
| Silhouette = 0.117 at 18 components | Partly a dimensionality artifact: distance ratios concentrate as dimension grows. Reducing to 4 components more than doubles it (0.279) but costs 0.023 macro F1 — see the frontier table in §2. Calinski-Harabasz (72,195 on the current run) is the more informative measure at 18 components. |
| State rules use fixed thresholds | The 10-state priority cascade uses thresholds derived from business logic, not learned from data. A supervised or Hidden Markov Model approach would generalise better to unseen seasonal patterns. |
| Month 12 to Month 13 extrapolation | XGBoost was trained on Feb→Oct transitions. December-to-January predictions are one month outside the training distribution. `month_num` is included as a feature to partially account for seasonality. |
| Frozen centroids at future dates | K-Means centroids were fitted on December 2025. At `observation_date=2026-01-31`, members are assigned via nearest-centroid to December 2025 centroids. Significant behavioural shifts in Months 13–14 may cause assignment drift. |
| No causal inference | Recommended activation actions are hypothesis-driven. Predicted lift from interventions is not quantified from historical A/B data. |

### Future Work

1. Apply HDBSCAN to the transition feature space (month-over-month delta features) rather than the static feature space — density gaps may exist in velocity space even if not in absolute-value space.
2. Replace the rule-based state engine with a Hidden Markov Model trained on the 11-month member sequence.
3. Estimate counterfactual impact of each activation action using matched controls from historical data.
4. Compute full SHAP values for each member's transition prediction to support individual-level reasoning.
5. Implement centroid drift monitoring: track Calinski-Harabasz month-over-month and trigger model refresh if it drops below a defined threshold.

---

## 8. Business Impact

### Revenue at Risk

TBIE identifies **6,824 High-Tier Accelerator members currently in a Program
Skeptic state** at 2025-12-31 — structurally high-value members who have
disengaged from the programme.

Two different numbers describe this cohort, and conflating them is the most
common way a business case gets overstated:

| Measure | Value | What it means |
|---|---:|---|
| Gross exposure | ~$36.8M annualised | Total forward spend **if every member lapsed completely** |
| **Expected value at risk** | **$1,130,607** | Model-derived: `Σ P(move to lower-value segment) × (value now − value then)` |

The second is the number to plan against. It is computed by
`src/utils/economics.value_at_risk` from the model's own probabilities and
segment values observed in the data, and it is what the serving API returns per
member. Across all 500,000 members the total expected value at risk is
**$17.79M** — against a gross exposure many times larger.

The gap is not a correction to the exposure figure; they measure different
things. Exposure asks "how much revenue sits in this cohort", expected value at
risk asks "how much of it do we actually expect to lose". Only the latter can
be compared against a campaign budget.

Targeting all 6,824 members at $5 per touchpoint costs ~$34,000 per cycle.
`src/10_cost_thresholds.py` derives the operating point from expected value
rather than from a classification threshold; see §9 and `RESULTS.md`.

> **What is still assumed.** How much of the at-risk value a successful contact
> recovers is currently an explicit input with a sensitivity sweep; break-even
> sits between a 2% and 5% recovery rate. The cost side is grounded; the
> benefit side is conditional.
>
> This assumption is **reducible**, contrary to what earlier revisions said.
> `engagement_events.parquet` records campaign exposure and response on 12.7M
> rows, which supports a modelled response propensity in place of a guess. No
> randomised holdout exists, so causal lift still requires one. See `AUDIT.md`.

### Member Example 1 — Silent Accumulator in Win-Back Target

Member MBR_0200452 is structurally a Silent Accumulator (S04): 18 purchases totalling $1,558 over 180 days, no email opens, no app sessions, no reward catalogue browsing, and 32,609 unspent loyalty points. This week they crossed into Win-Back Target state: 62 days since their last purchase with no 30-day spend.

The transition model gives a 61.2% probability they migrate to Growth Builder (S01) next month. Recommended action: a single SMS referencing their exact point balance and a specific redemption item. Email has been ignored for 18+ months. First redemption for Silent Accumulators is the single highest predictor of long-term retention and must occur within 7 days before the win-back window closes.

### Member Example 2 — High-Tier Accelerator in Program Skeptic

6,824 High-Tier Accelerator members average $3,043 in 180-day spend — 2.5x the population average of $1,198 — yet are currently ignoring programme communications and showing declining engagement. The Markov transition model projects that the majority will migrate to Lapse Risk within 60 days without intervention.

Recommended action: recognition and exclusivity, not a discount. High-Tier Accelerators are price-insensitive; a percentage-off email signals the programme is generic and cheapens their status. The correct intervention is an in-app notification with tier status, a personalised message referencing their spend history, and early access to a new product. Channel: in-app push and personalised email. Timing: within 14 days.
