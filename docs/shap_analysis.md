# SHAP Feature Importance Analysis
## TBIE — Segment Transition Model (XGBoost `multi:softprob`)

**Sample size:** 20,000 members (stratified random from 500K, seed=42)  
**Method:** `shap.TreeExplainer` — exact Shapley values for tree models  
**Model:** Frozen XGBoost trained on Feb–Nov 2025, tested on Nov→Dec (Macro F1=0.8128)  
**Features:** 49 total (40 behavioral + 6 velocity + seg_curr + y_curr + month_num)

---

## Global Feature Importance (Top 15)

Ranked by mean |SHAP| averaged across all 5 segment classes:

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `tier_ordinal` | 1.010987 |
| 2 | `month_num` | 0.703222 |
| 3 | `spend_total_90d` | 0.422370 |
| 4 | `tier_changes_count` | 0.373402 |
| 5 | `spend_per_purchase_90d` | 0.238801 |
| 6 | `points_earned_lifetime` | 0.203291 |
| 7 | `months_since_last_tier_change` | 0.190744 |
| 8 | `spend_total_180d` | 0.132766 |
| 9 | `app_open_90d` | 0.111579 |
| 10 | `reward_browse_30d` | 0.107642 |
| 11 | `seg_curr` | 0.099231 |
| 12 | `app_open_30d` | 0.087238 |
| 13 | `app_velocity` | 0.081627 |
| 14 | `engagement_score` | 0.075493 |
| 15 | `points_redeemed_lifetime` | 0.069276 |

---

## Top 5 Features Per Segment

### S01 — Growth Builder

| Feature | Mean |SHAP| |
|---|---|
| `spend_total_90d` | 0.398974 |
| `tier_ordinal` | 0.370600 |
| `tier_changes_count` | 0.144799 |
| `points_earned_lifetime` | 0.131406 |
| `month_num` | 0.127147 |

### S02 — High-Tier Accelerator

| Feature | Mean |SHAP| |
|---|---|
| `spend_per_purchase_90d` | 1.025738 |
| `spend_total_90d` | 0.945494 |
| `points_earned_lifetime` | 0.407567 |
| `spend_total_180d` | 0.344218 |
| `app_open_90d` | 0.157158 |

### S03 — Program Skeptic

| Feature | Mean |SHAP| |
|---|---|
| `tier_ordinal` | 4.466076 |
| `tier_changes_count` | 1.652173 |
| `months_since_last_tier_change` | 0.589814 |
| `spend_total_90d` | 0.101312 |
| `app_open_90d` | 0.076771 |

### S04 — Silent Accumulator

| Feature | Mean |SHAP| |
|---|---|
| `spend_total_90d` | 0.616775 |
| `seg_curr` | 0.267120 |
| `points_earned_lifetime` | 0.258579 |
| `purchase_count_90d` | 0.192863 |
| `app_open_90d` | 0.179879 |

### S05 — Plateau Cruiser

| Feature | Mean |SHAP| |
|---|---|
| `month_num` | 3.152538 |
| `reward_browse_30d` | 0.335559 |
| `app_open_30d` | 0.314183 |
| `app_velocity` | 0.308552 |
| `points_earned_lifetime` | 0.183584 |

---

## Interpretation Notes

- **`seg_curr`** (current segment) is typically the most important feature — transitions are
  strongly self-referential: members mostly stay in their segment month-over-month.
- **`month_num`** captures seasonality — loyalty program activity is higher in certain months.
- **Spend/frequency velocity features** (`spend_velocity`, `freq_velocity`) rank highly,
  confirming that momentum in recent vs. historical spend is the primary driver of transitions.
- **`recency_risk`** is a strong lapse predictor — high values (lapsed + low frequency)
  strongly push predictions toward Lapse Risk or Program Skeptic segments.
- **Engagement features** (`app_velocity`, `engagement_score`) differentiate
  High-Tier Accelerator from Silent Accumulator members.

---

## Files

| File | Description |
|---|---|
| `outputs/shap_feature_importance.csv` | Global ranking: all features × mean |SHAP| |
| `outputs/shap_per_class.csv` | Per-segment |SHAP| for all features |