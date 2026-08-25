# TBIE — Locked Decisions Log

This file records all design and data-handling decisions made during implementation.
Each entry cites the Phase and Step where the decision was made.

---

## Decision 001 — Ghost / Guest Cohort Definition
**Phase:** 1, Step 1.9
**Decision:** Member IDs present in `transactions` or `engagement_events` but absent from `members.parquet` are designated as ghost/guest members. They are excluded from all member-level modeling and are saved separately in `spine/guest_cohort.parquet`.
**Rationale:** These IDs almost certainly represent guest checkouts. The "exactly 1 transaction" hypothesis is validated against the actual data (see `validation/ghost_cohort_summary.json`).

---

## Decision 002 — Duplicate Engagement Row Handling
**Phase:** 1, Step 1.7 → Phase 3
**Decision:** Exact duplicate engagement rows (same `member_id`, `event_date`, `event_type`) are dropped before any aggregation. `drop_duplicates(subset=['member_id', 'event_date', 'event_type'])` is applied once during base-table cleaning in Phase 3 (`load_clean_base_tables()`), not per snapshot.
**Rationale:** Duplicate rows inflate event counts. The 0.13% duplicate rate confirmed in Phase 1 is non-trivial at 35M-row scale. Dropping is safer than keeping — if an event genuinely occurred twice in the same second with the same type, the second occurrence is still captured in other engagement signals.

---

## Decision 003 — Datetime Parsing Strategy
**Phase:** 1, Step 1.5 → Shared utility
**Decision:** All datetime columns are parsed using a strict dual-format parser (see `src/utils/datetime_parser.py`):
1. ISO first: `%Y-%m-%d %H:%M:%S`
2. Day-first fallback: `%d/%m/%Y %H:%M:%S`
3. ISO date-only: `%Y-%m-%d`
4. Day-first date-only: `%d/%m/%Y`
Bare `pd.to_datetime()` with default inference is NEVER used. If the column already arrives as `datetime64`, it is returned unchanged.
**Rationale:** Mixed formats are confirmed present in `transaction_date` and `event_date`. Pandas default inference silently misparses day-first values when day ≤ 12.

---

## Decision 004 — Zero-Activity Member Handling
**Phase:** 2, Step 2.3
**Decision:** Members present in `members.parquet` with zero transactions and zero engagement events are KEPT in the spine and in every snapshot. They receive 0.0 for all count/spend features and NaN for recency (which signals "never purchased").
**Rationale:** Spec requirement. These members are expected to exist (e.g., recently enrolled but not yet active). Dropping them would bias any downstream churn or segmentation model.

---

## Decision 005 — member_spine.parquet vs enrolled_members.parquet
**Phase:** 2
**Decision:** `member_spine.parquet` and `enrolled_members.parquet` are intentionally **identical files** at this stage (Phases 1–5). Both contain the same two columns: `member_id` and `account_open_date`. They are written separately per the spec's naming convention. Future phases may differentiate them (e.g., enrolled_members could be filtered to a specific enrollment window), but as of Phase 2 they are aliases.
**Rationale:** The spec names both files. To avoid confusion downstream, we document here that semantic divergence between the two has not yet occurred.

---

## Decision 006 — Session Duration Clip Bound
**Phase:** 1, Step 1.8 → Phase 3
**Decision:** `session_duration_sec` is clipped at **14,400 seconds (4 hours)** for all feature computations. The actual observed 99th percentile and distribution bounds are saved to `validation/clip_bounds.json` and are consumed by Phase 4's outlier clipping function.
**Rationale:** Values above 4 hours (confirmed: max ~134,927 seconds ≈ 37.5 hours) represent sessions left open or failed timeouts, not genuine engagement. Clipping at 4 hours removes a negligible fraction of rows while eliminating numerically extreme values that would distort means and slopes.

---

## Decision 007 — Purchase-Only Spend & Count Features
**Phase:** 3
**Decision:** `spend_total_*` and `purchase_count_*` features count ONLY rows where `transaction_type == 'purchase'` (after string normalization). Returns and exchanges are NOT included in these aggregations.
**Rationale:** Returns have negative amounts and exchanges have zero points_earned — both contaminate spend-based behavioral features. If return behavior needs analysis, a separate `return_count_*` feature family should be added rather than widening the purchase features.

---

## Decision 008 — Behavioral Feature Allowlist for Clustering
**Phase:** 4
**Decision:** An explicit `BEHAVIORAL_FEATURE_COLS` allowlist is defined in Phase 4. This list excludes raw IDs (`member_id`), raw dates (`observation_date`, `last_transaction_date`, `account_open_date`), intermediate computation columns, and any raw string fields. Only the features in this allowlist are passed to PCA/HDBSCAN in Phase 6.
**Rationale:** An exclusion-based approach is fragile — new columns added to snapshots would silently enter the model. An allowlist makes the clustering input surface explicit and auditable.

---

## Decision 009 — Leakage-Safe Tier History
**Phase:** 3/4
**Decision:** The `tier_history` JSON field is parsed at each observation date to extract the member's tier as-of that date. Only tier transitions with a date ≤ `observation_date` are considered. The `tier_current` field from `members.parquet` is never used in features.
**Rationale:** `tier_current` reflects the member's tier as of data extraction (Dec 2025), not as of any historical snapshot. Using it would introduce severe forward-looking leakage.

---

## Decision 010 — Engagement Deduplication Key
**Phase:** 1, Step 1.7
**Decision:** Engagement row deduplication uses the composite key: `['member_id', 'event_date', 'event_type']`. The `campaign_id` column is NOT included in the deduplication key because it is null for non-campaign events and would prevent deduplication of genuine duplicates lacking campaign context.
**Rationale:** The confirmed duplicate in the sample data (MBR_0000004, app_open, 2025-11-23 17:34:04) has no campaign_id — including it would have failed to catch this exact pattern.

---

## Decision 011 — Structural Collinearity
**Phase:** 5
**Decision:** Feature pairs with |r| > 0.90 in a single snapshot are flagged. Feature pairs that appear highly correlated across the majority of the 12 monthly snapshots are additionally marked as "structurally collinear" in the validation summary. These pairs must be reviewed and one member dropped before Phase 6 clustering.
**Rationale:** Collinear features inflate the effective weight of a behavioral dimension without adding information, biasing PCA and distorting HDBSCAN distance calculations.

---

## Decision 012 — January Exclusion from HDBSCAN Fit Population
**Phase:** 6
**Decision:** The January 2025 snapshot is excluded from the population used to train the HDBSCAN clustering model.
**Rationale:** The January snapshot is overwhelmingly zero-filled (73.71% all-zero across behavioral features) because there is no prior-year lookback history. Fitting on January would distort the clustering model by introducing a large artificial "all-zero" cluster that represents a lack of historical history rather than a real behavioral segment.

---

## Decision 013 — Dynamic Cold-Start Identification
**Phase:** 6
**Decision:** The cold-start population is identified dynamically per snapshot using the `feature_complete` flag (or checks on whether `purchase_count_180d == 0`), rather than relying on a static count (such as the 172,500 figure from early in the year).
**Rationale:** The cold-start population shrinks month-over-month (from 100% in January, to 35.43% in February, and down to 1.42% in December) as members accumulate transaction history. Treating cold-start members as a fixed cohort of 172,500 would be incorrect; the model must evaluate cold-start status dynamically.

---

## Decision 014 — Structural Window Truncation and December-Only Fit Cadence
**Phase:** 6
**Decision:** The rolling lookback windows (90-day and 180-day) are structurally truncated for snapshots prior to May 2025 (90-day) and July 2025 (180-day) due to the dataset beginning on January 1, 2025. This measurement artifact causes high PSI drift in the early months. To avoid clustering on immature features, the segmentation model is fit once on the fully mature December 2025 snapshot, where all members have a complete 180-day lookback window.
**Rationale:** The 90-day window is only 100% complete starting in May 2025, and the 180-day window is only 100% complete starting in July 2025. Running clustering on early snapshots would cluster on under-developed signal. The December snapshot contains fully mature, differentiated feature vectors (explaining 85% variance requires 12 components in December vs 9 in February), making it the optimal population for discovering stable behavioral segments.

---

## Decision 015 — Pre-Enrollment Activity Filtering
**Phase:** 3
**Decision:** All transactions and engagement events that occurred chronologically prior to the member's `account_open_date` are filtered out during snapshot construction.
**Rationale:** Members cannot have valid program-based transactions or engagement events prior to their account enrollment date. Keeping these records would represent chronological inconsistency, distort lookback aggregates, and potentially introduce structural noise or data prep leakage.

---

## Decision 016 — Strict Datetime Parse Exception Strategy
**Phase:** 1 / 3 / 4
**Decision:** The datetime parser tries a sequential list of formats (`%Y-%m-%d %H:%M:%S`, `%d/%m/%Y %H:%M:%S`, `%Y-%m-%d`, `%d/%m/%Y`). If any non-null value fails all formats, it raises a strict `ValueError` instead of silently coercing to NaT.
**Rationale:** Silently coercing parsing failures to NaT masks malformed input data, leading to silent data loss or hard-to-debug downstream errors. Raising an exception forces immediate visibility of date format inconsistencies.

---

## Decision 017 — Categorical Conversion to Category Dtype
**Phase:** 3
**Decision:** Raw string categorical columns (`channel`, `transaction_type`, `merchant_category`, `event_type`, `event_channel`) are normalized and then explicitly cast to the pandas `category` dtype *once* during base table loading.
**Rationale:** At 18M transactions and 35M engagement events scale, repeating string operations and groupbys across all 12 snapshots causes CPU/RAM exhaustion. Converting to the `category` dtype ensures pandas uses integer codes under the hood, dramatically increasing the performance of groupbys, joins, and filters.

---

## Decision 018 — Structural Missingness Exclusions
**Phase:** 5
**Decision:** Features that are designed to be sparse (specifically `recency_days` for never-purchased members, and `months_since_last_tier_change` for members with no tier change history) are explicitly excluded from the 5% null-rate threshold check in Phase 5 validation.
**Rationale:** A standard missingness filter would flag these columns as "failed coverage" due to their high null rate. However, since the NaNs represent valid structural states (sentinels indicating a non-event or lack of transaction history), they must be allowed to remain sparse without failing validation.

---

## Decision 019 — Weekly Bucketing for Slope Computations
**Phase:** 4
**Decision:** When fitting linear trend slopes (`spend_slope_30d`, `frequency_slope_30d`), daily logs are resampled into weekly Monday-based buckets per member before fitting OLS regression. If a member has less than 2 weekly data points in their resampled window, the slope is default-assigned to `0.0`.
**Rationale:** Fitting linear regression models on daily transactional logs is computationally expensive and highly noisy. Aggregating to weekly buckets stabilizes the time series trend, and assigning a 0.0 default prevents NaN values from propagating into the feature parquets.

---

## Decision 020 — Numpy Version Restriction
**Phase:** Environment Setup
**Decision:** The `numpy` version in `requirements.txt` is capped at `numpy>=1.24,<2.0` to avoid compatibility issues.
**Rationale:** HDBSCAN (which is targeted for Phase 6 clustering) has known breaking compatibility issues with Numpy 2.x releases. Locking the major version to `<2.0` ensures local environment stability without necessitating breaking environment rebuilds or custom patches during Phase 6 execution.

---

## Decision 021 — Engagement Event Type Restrictions
**Phase:** 3
**Decision:** Engagement aggregation counts are filtered and calculated only for the 14 real event types confirmed in Phase 1 (`app_open`, `email_open`, `email_click`, `push_open`, `push_dismiss`, `point_balance_check`, `reward_browse`, `reward_redemption`, `tier_status_check`, `support_contact`, `social_share`, `referral_sent`, `survey_completed`, `profile_update`). Hypothesized event types (such as `email_sent` or `browse_session` that do not exist in the raw dataset) are explicitly ignored.
**Rationale:** In retail analytics, adding hypothetical event categories that are not supported by the underlying logging schema would result in empty columns (zero-filled) or parsing bugs. Filtering to the 14 verified event types keeps the feature space clean and aligned with the physical data.



