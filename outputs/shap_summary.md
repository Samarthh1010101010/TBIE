# SHAP — Global Feature Importance

Model: `segment_transition_model.pkl` · 64 features · 25,000 sampled test rows (seed 42)

Mean absolute SHAP value: the average magnitude by which each feature moves
a prediction, in log-odds. Larger means the model leans on it more.

| rank | feature | mean \|SHAP\| | strongest for |
|---:|---|---:|---|
| 1 | `tier_ordinal` | 1.03719 | Program Skeptic |
| 2 | `month_num` | 0.71413 | Plateau Cruiser |
| 3 | `spend_total_90d` | 0.40462 | High-Tier Accelerator |
| 4 | `tier_changes_count` | 0.39274 | Program Skeptic |
| 5 | `spend_per_purchase_90d` | 0.21594 | High-Tier Accelerator |
| 6 | `months_since_last_tier_change` | 0.16078 | Program Skeptic |
| 7 | `points_earned_lifetime` | 0.15591 | High-Tier Accelerator |
| 8 | `spend_total_180d` | 0.10941 | High-Tier Accelerator |
| 9 | `seg_curr` | 0.10590 | Silent Accumulator |
| 10 | `reward_browse_30d` | 0.09843 | Plateau Cruiser |
| 11 | `app_open_90d` | 0.09072 | High-Tier Accelerator |
| 12 | `d_app_open_30d` | 0.07413 | Plateau Cruiser |
| 13 | `spend_total_30d` | 0.07140 | High-Tier Accelerator |
| 14 | `engagement_score` | 0.06520 | Silent Accumulator |
| 15 | `app_open_30d` | 0.06207 | Plateau Cruiser |
| 16 | `points_redeemed_lifetime` | 0.05975 | Plateau Cruiser |
| 17 | `total_session_sec_30d` | 0.05501 | High-Tier Accelerator |
| 18 | `app_velocity` | 0.05363 | Plateau Cruiser |
| 19 | `d_spend_total_30d` | 0.05296 | Plateau Cruiser |
| 20 | `purchase_count_30d` | 0.04937 | Silent Accumulator |
| 21 | `purchase_count_180d` | 0.04321 | Plateau Cruiser |
| 22 | `d_purchase_count_30d` | 0.04018 | Silent Accumulator |
| 23 | `email_click_30d` | 0.03795 | High-Tier Accelerator |
| 24 | `y_curr` | 0.03578 | High-Tier Accelerator |
| 25 | `avg_order_value_30d` | 0.03553 | High-Tier Accelerator |

## Per-class top 5

**Growth Builder** — `tier_ordinal` (0.4237), `spend_total_90d` (0.3239), `tier_changes_count` (0.1675), `month_num` (0.1497), `points_earned_lifetime` (0.1431)

**High-Tier Accelerator** — `spend_total_90d` (1.1015), `spend_per_purchase_90d` (0.9133), `spend_total_180d` (0.2806), `points_earned_lifetime` (0.2772), `spend_total_30d` (0.1625)

**Program Skeptic** — `tier_ordinal` (4.5165), `tier_changes_count` (1.7339), `months_since_last_tier_change` (0.5311), `spend_total_90d` (0.0869), `month_num` (0.0735)

**Silent Accumulator** — `spend_total_90d` (0.4678), `seg_curr` (0.3186), `points_earned_lifetime` (0.1775), `purchase_count_30d` (0.1405), `tier_ordinal` (0.1317)

**Plateau Cruiser** — `month_num` (3.1996), `reward_browse_30d` (0.2989), `d_app_open_30d` (0.2714), `app_open_30d` (0.2163), `app_velocity` (0.2027)
