# Phase 3 — Monthly Snapshot Builder: Complete Findings

**Status: COMPLETE** — All monthly snapshots built and schema-aligned. Outputs written to `snapshots/`

---

## What Phase 3 Does

Produces 12 monthly snapshots (`snapshot_2025_01_01.parquet` through `snapshot_2025_12_01.parquet`) from `2025-01-01` to `2025-12-01`. Each snapshot represents the state of all **500,000 members** as of that observation date (500,000 rows × 97 cols), aggregating transaction and engagement data over rolling lookback windows (7d, 30d, 90d, 180d) with strict time-censoring to prevent future data leakage.

---

## Inputs Consumed

| Input | Source | Used For |
|-------|--------|----------|
| `spine/member_spine.parquet` | Phase 2 Output | Left join anchor (500K rows) |
| `validation/ghost_cohort.csv` | Phase 1 Output | Excluding guest checkouts (88,717 IDs) |
| `data/raw/transactions.parquet` | Raw data | Transaction-level rolling aggregations |
| `data/raw/engagement_events.parquet` | Raw data | Digital behavior rolling aggregations |
| `data/raw/members.parquet` | Raw data | Parsing `tier_history` JSON structures |

---

## Key Steps & Design Decisions

### 1. Unified Base Table Loading & Cleaning
Base tables are loaded and cleaned **once** rather than reloaded per snapshot.
* Normalize categoricals (e.g. `transaction_type`, `channel`, `merchant_category`) using case folding and whitespace stripping.
* Exclude the 88,717 ghost cohort member IDs (`MBR_GHOST_*`).
* Drop exact duplicate engagement event rows (`member_id`, `event_date`, `event_type`) (Decision 002).
* Clip `session_duration_sec` outlier values at 14,400s (4 hours) (Decision 006).

### 2. Leakage-Censored Tier Reconstruction
* Customer tier is reconstructed historically by parsing `tier_history` JSON lists. The tier status as of `observation_date` is determined by filtering transitions prior to the snapshot date (Decision 009), ensuring `tier_current` (which represents the state at the end of the year) does not leak back in time.

### 3. Spend/Count Features (Purchase-Only)
* Spend and frequency rolling aggregations are filtered exclusively on `transaction_type == 'purchase'` to capture core customer buying behavior, leaving return/exchange logic out of the main behavioral segment metrics (Decision 007).

### 4. Diversity Columns Integration
* Features tracking unique category counts (`unique_categories_{w}`) and unique channel counts (`unique_channels_{w}`) are computed using `.nunique()` in Phase 3 so that Phase 4 can derive category and channel diversity features without re-scanning transactions.

### 5. Lifetime Points — Leakage-Safe Reconstruction

The raw `members.parquet` file contains two leakage columns (`lifetime_points_earned`, `current_point_balance`) representing the member's **full-dataset totals** as of data extraction — never the as-of-date state. These are never read in Phase 3.

Instead, lifetime points are reconstructed directly from `txns_t` (transactions filtered to `transaction_date <= obs_date`) using transaction-level sums:

```python
# points_earned_lifetime: running sum up to obs_date only
lifetime_earned = txns_t.groupby("member_id")["points_earned"].sum()

# points_redeemed_lifetime: aliased from points_clawed_back running sum
lifetime_clawback = txns_t.groupby("member_id")["points_clawed_back"].sum()

# current_point_balance_reconstructed: derived, clipped at 0
balance = (points_earned_lifetime - points_redeemed_lifetime).clip(lower=0)
```

**Numerical leakage verification** (Jan vs Dec snapshots, 500,000 members):

| Column | Identical Rate (Jan == Dec) | Verdict |
|---|---:|---|
| `points_earned_lifetime` | **1.4192%** | ✅ LEAKAGE ABSENT |
| `points_redeemed_lifetime` | **27.6330%** | ✅ LEAKAGE ABSENT (all identical = `0 == 0`) |

January values: mean = 0.0 for both columns (correct — programme launch date, no prior transactions).  
December values: `points_earned_lifetime` mean = 17,196.36, median = 12,148.0, max = 89,671.0; `points_redeemed_lifetime` mean = 732.74, median = 482.0, max = 16,905.0.

---

## Snapshot Output Summary (Per-Month Runs)

All snapshots consist of exactly **500,000 rows** (reconciled against the spine size) and **97 columns** (perfectly aligned schema across all months). Runtime measured from end-to-end including parquet writes:

| Month | Rows | Active 30d Members | Active 30d % | Runtime |
|-------|------:|-------------------:|------------:|--------:|
| **2025-01-01** | 500,000 | 0 | 0.0% | ~30s |
| **2025-02-01** | 500,000 | 327,543 | 65.5% | ~60s |
| **2025-03-01** | 500,000 | 360,143 | 72.0% | ~65s |
| **2025-04-01** | 500,000 | 383,830 | 76.8% | ~68s |
| **2025-05-01** | 500,000 | 400,012 | 80.0% | ~70s |
| **2025-06-01** | 500,000 | 417,453 | 83.5% | ~72s |
| **2025-07-01** | 500,000 | 431,774 | 86.4% | ~74s |
| **2025-08-01** | 500,000 | 447,593 | 89.5% | ~76s |
| **2025-09-01** | 500,000 | 464,593 | 92.9% | ~78s |
| **2025-10-01** | 500,000 | 477,143 | 95.4% | ~80s |
| **2025-11-01** | 500,000 | 485,249 | 97.1% | ~82s |
| **2025-12-01** | 500,000 | 493,124 | 98.6% | ~85s |

---

## Exit Criteria — All PASS

| Check | Result |
|-------|--------|
| Row count matches spine size on all 12 snapshots | ✅ PASSED (exactly 500,000 rows each) |
| Schema column count consistent across all months | ✅ PASSED (97 cols on every snapshot) |
| `assert_no_future_data` leakage assertion passed | ✅ PASSED (fires on every snapshot call, 0 violations) |
| Pre-enrollment activity removed | ✅ PASSED (`transaction_date >= account_open_date`) |
| Ghost cohort excluded (88,717 IDs) | ✅ PASSED (all `MBR_GHOST_*` excluded from both tables) |
| Vectorized aggregations — no `.apply()` | ✅ PASSED (groupby + pivot_table only) |
| Lifetime points reconstructed from transactions | ✅ PASSED (verified: `points_earned_lifetime` identical rate = **1.4192%**) |
| Raw leakage columns (`lifetime_points_earned`, `current_point_balance`) never read | ✅ PASSED |

---

## What Phase 4 Needs from Phase 3

Phase 4 (feature engine) reads directly from:
- `snapshots/snapshot_YYYY_MM_DD.parquet` — to extract the rolling aggregate columns.
- `validation/clip_bounds.json` — to read outlier boundaries.
- `data/raw/transactions.parquet` — to calculate trend slopes.
