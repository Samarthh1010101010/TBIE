# Phase 4 — Feature Engine: Complete Findings

**Status: COMPLETE** — 9 feature families engineered for all 12 snapshot months. Outputs written to `features/`

---

## What Phase 4 Does

Consumes monthly snapshots (500,000 rows × 97 cols each) and engineers **9 behavioral feature families**, producing feature parquets of 500,000 rows × 118 cols. It applies outlier clipping bounds extracted from the Phase 1 audit and stores an explicit allowlist of behavioral features for downstream clustering.

---

## Inputs Consumed

| Input | Source | Used For |
|-------|--------|----------|
| `snapshots/snapshot_YYYY_MM_DD.parquet` | Phase 3 Output | Base behavioral metrics (500,000 rows × 97 cols) |
| `data/raw/transactions.parquet` | Raw data | Weekly-bucket resampling for trend slopes (17,778,971 rows) |
| `validation/clip_bounds.json` | Phase 1 Output | Outlier clipping limits (loaded dynamically, not hardcoded) |

---

## Feature Families Engineered

### 1. Spend
* **`avg_order_value_30d`** = `spend_total_30d / purchase_count_30d` (0-filled where count = 0)
* **`spend_per_purchase_90d`** = `spend_total_90d / purchase_count_90d` (0-filled where count = 0)

### 2. Frequency & 3. Recency
* Pass-through counts (`purchase_count_7d/30d/90d/180d`, `return_count_30d/90d`) and `recency_days` from Phase 3 snapshots. `recency_days` is null for 34.5% of cohort (never-purchased members) — intentional sentinel.

### 4. Loyalty — Points (Leakage-Safe)
* **`redemption_rate`** = `points_redeemed_lifetime / points_earned_lifetime` (clamped to [0.0, 1.0])
* **`hoarding_ratio`** = `current_point_balance_reconstructed / points_earned_lifetime` (clipped to [0.0, 10.0])

Both `points_earned_lifetime` and `points_redeemed_lifetime` are **reconstructed as-of each observation date** by Phase 3 (running sum of transaction-level `points_earned` and `points_clawed_back` up to `obs_date`). The raw leakage columns from `members.parquet` (`lifetime_points_earned`, `current_point_balance`) are never used.

**Numerical leakage verification** (Jan vs Dec snapshot comparison, 500,000 members):

| Column | Identical Rate (Jan == Dec) | Verdict |
|---|---:|---|
| `points_earned_lifetime` | **1.4192%** | ✅ LEAKAGE ABSENT — Jan mean = 0.0, Dec mean = 17,196.36 |
| `points_redeemed_lifetime` | **27.6330%** | ✅ LEAKAGE ABSENT — all identical cases are `0 == 0` (never-redeemers); zero non-zero collisions |

### 5. Engagement
* **`email_open_rate_30d`** = `email_opens_30d / email_sent_30d` (clipped to [0.0, 1.0])
* **`email_click_rate_30d`** = `email_clicks_30d / email_opens_30d` (clipped to [0.0, 1.0])
* **`browse_to_purchase_ratio_30d`** = `browse_sessions_30d / purchase_count_30d` (clipped to [0.0, 10.0])
* **`browsed_but_never_purchased_30d`** = 1 if `browse_sessions_30d > 0` and `purchase_count_30d == 0`, else 0

### 6. Trend & 7. Acceleration
* **`spend_slope_30d` / `frequency_slope_30d`**: OLS linear trend fitted on weekly-resampled buckets within the 30-day window per member. Returns exactly `0.0` (never NaN) when < 2 weekly data points exist.
* **`spend_acceleration`**: `spend_slope_30d(t) - spend_slope_30d(t-1)`. Set to `0.0` for January (no prior month). Computed chronologically — each month references the previously computed feature parquet.

### 8. Diversity
* **`category_diversity_90d`** = `unique_categories_90d / purchase_count_90d` (clamped to [0.0, 1.0])
* **`channel_diversity_90d`** = `unique_channels_90d / purchase_count_90d` (clamped to [0.0, 1.0])
* Inputs (`unique_categories_90d`, `unique_channels_90d`) are computed in Phase 3 via `.nunique()`.

### 9. Tier Trajectory
* **`tier_ordinal`**: `base` = 0, `silver` = 1, `gold` = 2, `platinum` = 3
* Pass-through: `tier_changes_count`, `months_since_last_tier_change` (null for 70.6% with no tier change history — intentional sentinel).

---

## Outlier Clipping Bounds (from Phase 1 Audit)

All clips loaded dynamically from `validation/clip_bounds.json` — no hardcoded thresholds:

| Column | Clip Bound | Source |
|--------|-----------|--------|
| `transaction_amount` | upper = $205.50 | Phase 1 audit |
| `session_duration_sec` | upper = 14,400s (4 hrs) | Phase 1 audit, Decision 006 |
| `hoarding_ratio` | [0.0, 10.0] | Decision 011 |
| `browse_to_purchase_ratio_30d` | [0.0, 10.0] | Decision 011 |

---

## Behavioral Feature Allowlist (Decision 008)

An explicit list of **34 numeric features** is attached to each output DataFrame as `df.attrs['behavioral_feature_cols']` (alias: `df.attrs['ml_feature_cols']`). This is the definitive Phase 6 input — Phase 6 must load this list, not infer columns from the schema.

Excludes: `member_id`, `observation_date`, `last_transaction_date`, `account_open_date`, `current_tier`, `tier_trajectory_direction`, `feature_complete`, `current_point_balance_reconstructed`, `points_clawed_back_lifetime`.

```
# Spend (6)
spend_total_7d, spend_total_30d, spend_total_90d, spend_total_180d,
avg_order_value_30d, spend_per_purchase_90d,

# Frequency (6)
purchase_count_7d, purchase_count_30d, purchase_count_90d, purchase_count_180d,
return_count_30d, return_count_90d,

# Recency (1)
recency_days,

# Loyalty (4)
redemption_rate, hoarding_ratio, points_earned_lifetime, points_redeemed_lifetime,

# Engagement (7)
email_open_rate_30d, email_click_rate_30d,
browse_to_purchase_ratio_30d, browsed_but_never_purchased_30d,
reward_browse_30d, reward_redemption_30d,
total_session_sec_30d,

# Trend (2)
spend_slope_30d, frequency_slope_30d,

# Acceleration (1)
spend_acceleration,

# Diversity (4)
category_diversity_90d, channel_diversity_90d,
unique_categories_90d, unique_channels_90d,

# Tier trajectory (3)
tier_ordinal, tier_changes_count, months_since_last_tier_change
```


---

## Output File Summary

| File | Rows | Cols | Size |
|------|-----:|-----:|-----:|
| `features/features_2025_01_01.parquet` | 500,000 | 118 | 4.2 MB |
| `features/features_2025_02_01.parquet` | 500,000 | 118 | 40.4 MB |
| `features/features_2025_03_01.parquet` | 500,000 | 118 | 45.2 MB |
| `features/features_2025_04_01.parquet` | 500,000 | 118 | 51.0 MB |
| `features/features_2025_05_01.parquet` | 500,000 | 118 | 53.5 MB |
| `features/features_2025_06_01.parquet` | 500,000 | 118 | 56.4 MB |
| `features/features_2025_07_01.parquet` | 500,000 | 118 | 57.0 MB |
| `features/features_2025_08_01.parquet` | 500,000 | 118 | 57.7 MB |
| `features/features_2025_09_01.parquet` | 500,000 | 118 | 59.2 MB |
| `features/features_2025_10_01.parquet` | 500,000 | 118 | 61.1 MB |
| `features/features_2025_11_01.parquet` | 500,000 | 118 | 62.9 MB |
| `features/features_2025_12_01.parquet` | 500,000 | 118 | 66.7 MB |

Total size: ~616 MB (excluded from git — rebuild with `python src/04_feature_engine.py`)

---

## Exit Criteria — All PASS

| Check | Result |
|-------|--------|
| All 9 feature families implemented and populated | ✅ PASSED |
| Slopes return 0.0 (never NaN) for sparse records | ✅ PASSED (0 NaN slopes in any month) |
| Acceleration computed chronologically | ✅ PASSED (each month uses prior month's slope) |
| January `spend_acceleration` = 0.0 for all 500,000 members | ✅ PASSED |
| Outlier bounds loaded from `clip_bounds.json` (not hardcoded) | ✅ PASSED |
| Explicit `behavioral_feature_cols` list stored in `df.attrs` | ✅ PASSED (34 features) |
| `points_earned_lifetime` leakage check: identical rate | ✅ PASSED — **1.4192%** (leakage absent) |
| `points_redeemed_lifetime` leakage check: identical rate | ✅ PASSED — **27.6330%**, all identical = 0==0 (leakage absent) |
| All 12 feature parquets written with `engine='pyarrow'` | ✅ PASSED |
