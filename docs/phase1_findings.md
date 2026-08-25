# Phase 1 — Raw Data Validation: Complete Findings

**Status: COMPLETE** — All 7 output files written to `validation/`

---

## Data Loaded

| File | Rows | Cols |
|------|-----:|-----:|
| members.parquet | 500,000 | 28 |
| transactions.parquet | 17,778,971 | 17 |
| engagement_events.parquet | 35,530,781 | 10 |

---

## Step 1.1 — Schema (members.parquet — 28 cols)

| Column | dtype | Nulls |
|--------|-------|------:|
| account_open_date | datetime64[us] | 0 |
| account_close_date | datetime64[us] | 490,102 (98%) |
| credit_line | float64 | 337,169 (67%) |
| credit_utilization | float64 | 337,169 (67%) |
| tier_history | object (JSON) | 0 |
| lifetime_points_earned | int64 | 0 — LEAKAGE, never use directly |
| current_point_balance | int64 | 0 — LEAKAGE, never use directly |

Categoricals: account_status (active/dormant/closed), portfolio (BrandA/B/C),
tier_current (base/silver/gold/platinum), gender (M/F/O/unknown),
age_band (18-24 to 65+), urban_rural (urban/suburban/rural),
acquisition_source (organic/event/referral/social/paid_search/direct_mail)

## Step 1.1 — Schema (transactions.parquet — 17 cols)

| Column | dtype | Note |
|--------|-------|------|
| transaction_date | object | Mixed ISO + day-first strings — parser REQUIRED |
| transaction_amount | float64 | Includes negatives (returns) |
| transaction_type | object | Mixed case: 'purchase','Purchase','PURCHASE' |
| All other columns | various | 0 nulls anywhere in transactions |

## Step 1.1 — Schema (engagement_events.parquet — 10 cols)

| Column | dtype | Nulls |
|--------|-------|------:|
| event_date | object | 0 — mixed formats |
| session_duration_sec | float64 | 47.5% (non-web events) |
| pages_viewed | float64 | 47.5% (non-web events) |
| campaign_id/type/response | object | 64.1% (non-campaign events) |
| support_resolution_status | object | 96.6% (support events only) |

---

## Step 1.2 — Row Count Verification

| Table | Actual | Expected | Diff | Status |
|-------|-------:|--------:|-----:|--------|
| members | 500,000 | 500,000 | 0.0% | OK |
| transactions | 17,778,971 | 18,000,000 | 1.2% | OK |
| engagement | 35,530,781 | 35,000,000 | 1.5% | OK |

---

## Step 1.3 — member_id Format

| Table | Pattern | % | Deviants |
|-------|---------|--:|---------|
| members | MBR_####### | 100% | 0 |
| transactions | MBR_####### | 99.44% | 28 sample rows use MBR_GHOST_##### |
| engagement | MBR_####### | 100% | 0 |

---

## Step 1.4 — Null Audit (critical columns only)

| Column | Table | Null % |
|--------|-------|-------:|
| account_close_date | members | 98.0% |
| account_expire_date | members | 95.0% |
| credit_line | members | 67.4% |
| session_duration_sec | engagement | 47.5% |
| campaign_id/type/response | engagement | 64.1% |
| support_resolution_status | engagement | 96.6% |
| member_id | transactions | 0% — OK |
| member_id | engagement | 0% — OK |
| account_open_date | members | 0% — OK |
| tier_history | members | 0% — OK |

---

## Step 1.5 — Datetime Format (CONFIRMED MIXED)

| Table | Column | ISO hits | Day-first hits | Mixed? |
|-------|--------|----------|----------------|--------|
| transactions | transaction_date | 463/500 | 37/500 | YES — dual parser REQUIRED |
| engagement | event_date | 475/500 | 25/500 | YES — dual parser REQUIRED |

---

## Step 1.6 — tier_history Structure

- Null values: 0 (100% populated)
- JSON parse success: 200/200 (0 failures)
- Structure: list of dicts with keys ["tier", "date"]
- Example: [{"tier": "base", "date": "2024-08-08"}, {"tier": "gold", "date": "2025-01-15"}]
- Tier values: base -> silver -> gold -> platinum

---

## Step 1.7 — Duplicate Engagement Rows

| Metric | Value |
|--------|------:|
| Total rows | 35,530,781 |
| Duplicate keys (count > 1) | 46,374 |
| Total rows in duplicates | 92,748 |
| Duplicate rate | 0.261% |

Dedup key: [member_id, event_date, event_type]
Action: drop_duplicates() in Phase 3 load (decisions.md #002)

---

## Step 1.8 — session_duration_sec Outliers

| Stat | Value |
|------|------:|
| Median | 82s |
| 95th pct | 454s |
| 99th pct | 933s |
| Max | 172,797s (48 hours!) |
| Rows > 4 hrs | 18,663 |

CLIP BOUND LOCKED: 14,400s (4 hours) → saved to validation/clip_bounds.json

---

## Step 1.9 — Ghost Cohort Reconciliation

| Metric | Count |
|--------|------:|
| Known members | 500,000 |
| Unique IDs in transactions | 588,540 |
| Unique IDs in engagement | 497,957 |
| Ghost cohort size | 88,717 |
| Ghost in transactions only | 88,717 |
| Ghost in engagement | 0 |

All ghost IDs use MBR_GHOST_##### prefix.
Saved: validation/ghost_cohort.csv (88,717 IDs)

---

## Step 1.10 — Transaction Amount Distribution

| Stat | Value |
|------|------:|
| Mean | $56.06 |
| Median | $49.58 |
| Min | -$1,176.14 |
| Max | $8,283.41 |
| 1st pct | -$82.36 |
| 99th pct | $205.50 |
| Negative amounts (returns) | 937,809 rows |

No lower clip. Upper soft cap at $205.50 (99th pct).

---

## Step 1.11 — Transaction Type Distribution

| Type (raw) | Count | % |
|-----------|------:|--:|
| 'purchase' | 16,460,106 | 92.58% |
| 'return' | 936,920 | 5.27% |
| 'exchange' | 355,286 | 2.00% |
| 'Purchase' | 12,382 | 0.07% |
| 'PURCHASE' | 12,356 | 0.07% |
| 'Return' | 694 | 0.00% |
| 'RETURN' | 690 | 0.00% |

After strip().lower(): purchase=16,484,844 / return=938,304 / exchange=355,823

---

## Output Files (all written to validation/)

| File | Size |
|------|-----:|
| raw_data_audit.md | 19.8 KB |
| ghost_cohort.csv | 1,473 KB |
| ghost_cohort_summary.json | 0.3 KB |
| clip_bounds.json | 0.5 KB |
| tier_history_schema.md | 0.9 KB |
| date_parse_check.csv | 0.2 KB |
| duplicate_check.csv | 23.6 KB |

---

## Exit Criteria — All 11 PASS

- [PASS] Schema inspected for all 3 tables
- [PASS] Row/column counts verified
- [PASS] member_id format consistency confirmed
- [PASS] Null audit on all key columns
- [PASS] Datetime formats detected (dtype-first then regex sample)
- [PASS] tier_history null vs parse failure separation documented
- [PASS] Duplicate engagement rows quantified
- [PASS] session_duration_sec clip bound confirmed (14,400s)
- [PASS] Ghost cohort reconciled — 88,717 (one authoritative number)
- [PASS] Transaction amount bounds measured
- [PASS] Transaction type distribution + mixed-case confirmed

---

## Next Step

Run Phase 2:
    python src/02_build_spine.py
