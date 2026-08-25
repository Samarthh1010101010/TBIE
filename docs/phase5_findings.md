# Phase 5 — Feature Validator: Complete Findings

**Status: COMPLETE** — All 12 monthly snapshots validated and passed ✅. Outputs written to `validation/`

---

## What Phase 5 Does

Performs automated quality assurance checking over all 12 monthly feature parquet files. It evaluates the engineered features across 5 checks: Missingness, Ranges/Bounds, Business Sanity, Collinearity, and Cross-snapshot Stability (drift).

---

## Inputs Consumed

| Input | Source | Used For |
|-------|--------|----------|
| `features/features_2025_MM_01.parquet` | Phase 4 Output | Feature quality audit (12 files, 500,000 rows × 118 cols each) |

---

## Validation Findings

### 5.1 Missingness (Null rates < 5%)

All 118 feature columns pass the < 5% null threshold across all 12 months, with two columns explicitly excluded from the missingness gate by design:

| Column | Null Rate | Reason Excluded |
|--------|----------:|-----------------|
| `recency_days` | ~34.5% of cohort | Null = never-purchased member. Sentinel by design; imputed before clustering. |
| `months_since_last_tier_change` | ~70.6% of cohort | Null = no tier change ever recorded. Sentinel by design; imputed before clustering. |

All other columns: **0 missingness flags across all 12 months.**

### 5.2 Range Violations (Zero Violations)

All ratio and ordinal columns stay strictly within domain bounds post-clipping across all 500,000 members and all 12 months:

| Column | Enforced Bound | Violations |
|--------|---------------|------------|
| `redemption_rate` | [0.0, 1.0] | 0 |
| `email_open_rate_30d` | [0.0, 1.0] | 0 |
| `email_click_rate_30d` | [0.0, 1.0] | 0 |
| `category_diversity_90d` | [0.0, 1.0] | 0 |
| `channel_diversity_90d` | [0.0, 1.0] | 0 |
| `hoarding_ratio` | [0.0, 10.0] | 0 |
| `browse_to_purchase_ratio_30d` | [0.0, 10.0] | 0 |

### 5.3 Business Sanity (Zero Sanity Issues)

All 12 snapshots pass. Verified across all 500,000 members:

- `purchase_count_30d == 0` ↔ `spend_total_30d == 0` for all members (zero join bugs).
- `recency_days` is null only for members with zero purchases across all windows — no case of a member having `purchase_count_180d > 0` with `recency_days` null (0 inconsistent rows found).
- Zero-activity members carry default values of `0.0` for all count/spend columns, consistent with LEFT JOIN spine design.

### 5.4 Collinearity Check (Informational — Not a Failure)

Structurally collinear pairs identified: correlation |r| > 0.90 in ≥ 8 of 12 months. These arise from window-overlap design and points arithmetic — they are **expected** and are not dropped at this stage.

| Feature Pair | Avg r | Months Correlated |
|---|---:|---:|
| `points_earned_lifetime` ↔ `current_point_balance_reconstructed` | 0.997 | 12 of 12 |
| `points_clawed_back_lifetime` ↔ `points_redeemed_lifetime` | 1.000 | 12 of 12 |
| `spend_total_90d` ↔ `spend_total_180d` | 0.964 | ≥ 8 of 12 |
| `app_opens_30d` ↔ `reward_browse_30d` | ~0.970 | ≥ 8 of 12 |
| `spend_total_*d` ↔ `points_earned_*d` (all windows) | high | ≥ 8 of 12 |

Collinear pair counts by month:

| Month | Collinear Pairs |
|-------|---------------:|
| 2025-01 | 0 |
| 2025-02 | 130 |
| 2025-03 | 72 |
| 2025-04 | 72 |
| 2025-05 | 64 |
| 2025-06 | 51 |
| 2025-07 | 44 |
| 2025-08 | 37 |
| 2025-09 | 34 |
| 2025-10 | 33 |
| 2025-11 | 27 |
| 2025-12 | 30 |

> January shows 0 collinear pairs because almost all members have zero activity on day 1 — no correlation can be computed from constant columns.
>
> **Recommendation for Phase 6:** Do not drop these columns manually. Apply `StandardScaler` → `PCA` → `UMAP` → `HDBSCAN`. PCA absorbs the collinearity into its first few principal components and reduces dimensionality without information loss.

### 5.5 Cross-Snapshot Stability (PSI Drift)

Population Stability Index (PSI) computed month-over-month for all behavioral features. Full results in `validation/cross_snapshot_stability.json` (58.8 KB).

- **Jan → Feb transition**: Large PSI observed across all features. **Expected** — January 2025 is the programme launch date (obs date = 2025-01-01), so the entire cohort has zero historical activity in January. February is the first month with populated rolling lookbacks, producing an artefactual distribution shift.
- **March → December**: PSI < 0.10 for all stable features, indicating consistent population distributions. Pipeline ramp-up is complete by March.

---

## Leakage Verification — `points_earned_lifetime` and `points_redeemed_lifetime`

A post-hoc numerical audit was run to confirm that `points_earned_lifetime` and `points_redeemed_lifetime` are correctly reconstructed **as-of each observation date** (from transaction-level sums up to `obs_date`) rather than copied from the raw `members.parquet` leakage columns.

**Methodology:** Compare January 2025 snapshot values against December 2025 snapshot values for all 500,000 members. If leakage were present (raw full-dataset totals used), Jan and Dec would be identical for most members.

### `points_earned_lifetime`

| Metric | Value |
|---|---|
| Members compared | 500,000 |
| Identical (Jan == Dec) | 7,096 |
| **Identical rate** | **1.4192%** |
| Different (Jan != Dec) | 492,904 (98.5808%) |
| Jan zeros | 500,000 (100.00%) |
| Dec zeros | 7,096 (1.42%) |
| Jan mean / max | 0.0 / 0.0 |
| Dec mean / median / max | 17,196.36 / 12,148.0 / 89,671.0 |

**Identical rate = 1.4192% — below 10%. LEAKAGE ABSENT.**

The 7,096 identical cases are members who earned exactly 0 points in both January and December (never transacted during 2025). The 100% January zeros are correct: the observation date is 2025-01-01 (programme launch day), so no transactions precede it.

### `points_redeemed_lifetime`

| Metric | Value |
|---|---|
| Members compared | 500,000 |
| Identical (Jan == Dec) | 138,165 |
| **Identical rate** | **27.6330%** |
| Different (Jan != Dec) | 361,835 (72.3670%) |
| Jan zeros | 500,000 (100.00%) |
| Dec zeros | 138,165 (27.63%) |
| Jan mean / max | 0.0 / 0.0 |
| Dec mean / median / max | 732.74 / 482.0 / 16,905.0 |

**Identical rate = 27.6330% — but LEAKAGE ABSENT.** All 138,165 "identical" cases are `0 == 0` matches: members who never redeemed points at any point in 2025. Zero members have a non-zero January value matching a non-zero December value. The 10-member sample confirms this structure:

| member_id | Jan redeemed | Dec redeemed | identical |
|---|---:|---:|---|
| MBR_0000000 | 0.0 | 188.0 | False |
| MBR_0000001 | 0.0 | 617.0 | False |
| MBR_0000002 | 0.0 | 139.0 | False |
| MBR_0000003 | 0.0 | 285.0 | False |
| MBR_0000004 | 0.0 | 310.0 | False |
| MBR_0000005 | 0.0 | 474.0 | False |
| MBR_0000006 | 0.0 | 720.0 | False |
| MBR_0000007 | 0.0 | 401.0 | False |
| MBR_0000008 | 0.0 | 417.0 | False |
| MBR_0000009 | 0.0 | 463.0 | False |

**Conclusion: Both lifetime points columns are correctly built as running sums up to `obs_date`. No fix required.**

---

## Global Validation Summary Table

| Month Snapshot | Status | Missingness Flags | Range Violations | Sanity Issues | Collinear Pairs |
|-------|:---:|:---:|:---:|:---:|:---:|
| **2025-01** | ✅ | 0 | 0 | 0 | 0 |
| **2025-02** | ✅ | 0 | 0 | 0 | 130 |
| **2025-03** | ✅ | 0 | 0 | 0 | 72 |
| **2025-04** | ✅ | 0 | 0 | 0 | 72 |
| **2025-05** | ✅ | 0 | 0 | 0 | 64 |
| **2025-06** | ✅ | 0 | 0 | 0 | 51 |
| **2025-07** | ✅ | 0 | 0 | 0 | 44 |
| **2025-08** | ✅ | 0 | 0 | 0 | 37 |
| **2025-09** | ✅ | 0 | 0 | 0 | 34 |
| **2025-10** | ✅ | 0 | 0 | 0 | 33 |
| **2025-11** | ✅ | 0 | 0 | 0 | 27 |
| **2025-12** | ✅ | 0 | 0 | 0 | 30 |

---

## Output Files

| File | Size | Type | Purpose |
|------|-----:|:---:|---------:|
| `validation/feature_validation_summary.md` | 1.5 KB | Markdown | Global status summary table |
| `validation/cross_snapshot_stability.json` | 58.8 KB | JSON | MoM drift, median jumps, structural collinearity details |
| `validation/feature_validation_report_2025_MM.md` | 349 B – 7.5 KB | Markdown | 12 monthly per-snapshot audit files |
