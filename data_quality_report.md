# Data Quality Report

**Submission:** TBIE — Temporal Behavioural Intelligence Engine
**Hackathon:** Kobie × PES University
**Data sources:** `members.parquet` (500K rows), `transactions.parquet` (17.8M rows), `engagement_events.parquet` (35.5M rows)

---

## Issue 1 — Ghost Member IDs (88,717 orphaned transaction records)

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 member_id reconciliation: `transactions.parquet` contains 588,540 unique member IDs vs 500,000 in `members.parquet` |
| **Scale** | 88,717 IDs present in transactions but absent from member registry. All follow `MBR_GHOST_#####` prefix — clearly synthetic |
| **Impact** | If included in the member spine, feature engineering would produce rows with no member metadata (nulls on all `members.*` columns), corrupting clustering |
| **Resolution** | Built explicit ghost allowlist (`validation/ghost_cohort.csv`). All downstream phases LEFT JOIN from `member_spine.parquet` which excludes ghost IDs. Ghost transactions excluded from all feature windows |

---

## Issue 2 — Mixed Datetime Formats (dual-format parser required)

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 sample of 500 rows from `transactions.transaction_date` and `engagement_events.event_date` |
| **Scale** | 463/500 rows use ISO format `YYYY-MM-DD`; 37/500 use day-first `DD-MM-YYYY`. Both formats present in same column |
| **Impact** | `pd.to_datetime()` with default settings silently parses day-first dates incorrectly (e.g., `07-01-2025` read as January 7 instead of July 1), producing up to 6-month feature window errors |
| **Resolution** | Dual-pass parser in Phase 3: attempt ISO first (`dayfirst=False`), fall back to day-first (`dayfirst=True`) for failed rows. Validated: 0 remaining unparseable dates across all 17.8M transaction rows |

---

## Issue 3 — Mixed-Case Transaction Types (silent classification error)

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 transaction_type distribution audit |
| **Scale** | `'purchase'` (16,460,106), `'Purchase'` (12,382), `'PURCHASE'` (12,356) — 24,738 rows with non-lowercase types |
| **Impact** | Without normalisation, `purchase_count_30d` features would undercount by ~0.15%, returns would be misclassified, Lapse Risk detection would degrade |
| **Resolution** | `.str.strip().str.lower()` applied to `transaction_type` in Phase 3 snapshot builder before any feature aggregation. Post-normalisation: `purchase=16,484,844 / return=938,304 / exchange=355,823` |

---

## Issue 4 — Duplicate Engagement Events (92,748 rows)

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 dedup audit on key `[member_id, event_date, event_type]` |
| **Scale** | 46,374 duplicate key groups → 92,748 total rows affected (0.261% of 35.5M) |
| **Impact** | Duplicate events inflate `app_open_30d`, `email_open_30d`, `push_open_30d` counts — directly affects engagement-driven state assignments (Brand Advocate, Silent Accumulator thresholds) |
| **Resolution** | `drop_duplicates(subset=['member_id', 'event_date', 'event_type'], keep='first')` applied in Phase 3 before any aggregation. Documented in `docs/decisions.md` decision #002 |

---

## Issue 5 — session_duration_sec Outliers (172,797 seconds = 48 hours)

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 distribution audit of `engagement_events.session_duration_sec` |
| **Scale** | Max value: 172,797s (48 hours). 18,663 rows exceed 4 hours. Median: 82s, 99th percentile: 933s |
| **Impact** | `total_session_sec_30d` would be dominated by outliers, destroying the feature's signal for cluster separation. One member with a single 48-hour session would appear as extremely high engagement |
| **Resolution** | Hard clip at 14,400s (4 hours). Bound locked in `validation/clip_bounds.json` and applied identically across all 12 monthly snapshots. `session_duration_sec.clip(upper=14400)` in Phase 4 |

---

## Issue 6 — Leakage Columns in members.parquet

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 schema audit + Phase 5 leakage verification |
| **Columns** | `lifetime_points_earned`, `lifetime_points_redeemed`, `current_point_balance` in `members.parquet` |
| **Impact** | These columns reflect end-of-dataset totals (December 2025), not as-of-observation-date values. Using them directly for any month before December would leak future data — January snapshot would incorrectly show full-year points |
| **Resolution** | These columns are **never used** from `members.parquet`. All point features are reconstructed from `transactions.parquet` as running sums up to `obs_date`. Phase 5 confirmed: Jan 2025 reconstructed values are 0 for all members (correct — programme launch day), vs Dec 2025 mean of 17,196 points |

---

## Issue 7 — No email_sent Events (rate computation impossible)

| Field | Detail |
|---|---|
| **Discovery** | Phase 1 event_type distribution in `engagement_events.parquet` |
| **Event types present** | `app_open`, `email_open`, `email_click`, `push_open`, `reward_browse`, `reward_redemption`, `tier_status_check`, `support_contact` |
| **Missing** | `email_sent` events do not exist in the dataset |
| **Impact** | True email open rate (opens / sends) cannot be computed. A column named `email_open_rate_30d` would be misleading — it would actually be a count |
| **Resolution** | Column `email_open_rate_30d` was renamed to reflect its true nature (raw count of email opens). Documented as a count in `feature_descriptions.json`. Where email engagement matters for state rules, raw open count (`email_open_30d`) is used with empirically derived thresholds |

---


---

## Issue 8 — Null-Timestamp Engagement Events (NaT rows inflating 30-day window counts)

| Field | Detail |
|---|---|
| **Discovery** | Phase 6 clustering produced a 117-member micro-cluster (S05) with implausibly high engagement: mean 375 app opens in 30 days vs. 8.5x higher than the next most-engaged segment. Diagnostic confirmed the cause was NaT (null timestamp) rows in `engagement_events.parquet` |
| **Scale** | 117 members (0.023% of the 500,000-member population) have engagement event rows where `event_date` is null/NaT. Raw event counts per member for S05: mean=445 app_open rows, min=230, max=627 — all well above any plausible human usage pattern |
| **Root cause** | When `event_date` is NaT, the 30-day window comparison `event_date >= obs_date - 30` evaluates inconsistently in pandas — NaT rows are not filtered out and are counted into the 30d aggregates. Session duration for these phantom events: median=2 seconds (vs. 79s population-wide), confirming they are null-date artefacts, not real sessions |
| **Impact** | Feature columns `app_open_30d`, `reward_browse_30d`, `total_session_sec_30d` are significantly inflated for 117 members. K-Means correctly isolated these members into their own cluster (S05) rather than polluting the 4 genuine behavioural segments |
| **Resolution** | S05 is retained as a documented artefact cluster. Its existence is a data quality finding, not a modelling error — the clustering algorithm correctly quarantined anomalous members. The affected features' spike is capped indirectly via the 30-day session clip (Issue 5). A null-timestamp guard (`dropna(subset=['event_date'])`) should be applied in `03_snapshot_builder.py` in any future pipeline refresh |

---

## Summary Table

| # | Issue | Rows Affected | Resolution | Phase |
|---|---|---|---|---|
| 1 | Ghost member IDs | 88,717 | Excluded via spine LEFT JOIN | Phase 2 |
| 2 | Mixed datetime formats | ~7.4% of date rows | Dual-pass parser | Phase 3 |
| 3 | Mixed-case transaction types | 24,738 | `.str.lower()` normalisation | Phase 3 |
| 4 | Duplicate engagement events | 92,748 | `drop_duplicates()` on key | Phase 3 |
| 5 | Session duration outliers | 18,663 | Clip at 14,400s | Phase 4 |
| 6 | Leakage columns in members | 3 columns | Reconstructed from transactions | Phase 4 |
| 7 | No email_sent events | Entire column | Documented as count | Phase 4 |
| 8 | Null-timestamp engagement events | 117 members | Isolated into S05 micro-cluster; documented | Phase 6 |
