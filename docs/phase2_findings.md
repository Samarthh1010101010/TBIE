# Phase 2 — Member Spine Builder: Complete Findings

**Status: COMPLETE** — All 5 exit criteria passed. Outputs written to `spine/` and `validation/`

---

## What Phase 2 Does

Builds the single canonical member spine that all downstream phases
(snapshots, features, clustering) anchor to. Every snapshot row in Phases 3–6
traces back to a `member_id` that exists in this spine.

---

## Inputs Consumed

| Input | Source | Used For |
|-------|--------|----------|
| `data/raw/members.parquet` | Raw data | Spine definition |
| `validation/ghost_cohort.csv` | Phase 1 output | Exclusion list |
| `data/raw/transactions.parquet` | Raw data (member_id col only) | Zero-activity check |
| `data/raw/engagement_events.parquet` | Raw data (member_id col only) | Zero-activity check |

---

## Step 2.1 — Load & Parse members.parquet

| Check | Result |
|-------|--------|
| Rows loaded | 500,000 |
| Columns | 28 |
| `account_open_date` dtype | datetime64[us] — already parsed ✅ |
| Null `account_open_date` | 0 ✅ |
| Date range | 2022-12-23 → 2025-12-16 |
| Duplicate `member_id` | 0 — uniqueness asserted ✅ |

The `account_open_date` column arrived as `datetime64[us]` from Parquet — the dual-format parser passed through unchanged (fast path triggered).

---

## Step 2.2 — Ghost Cohort Exclusion

| Metric | Value |
|--------|------:|
| Ghost IDs loaded from Phase 1 | 88,717 |
| Ghost IDs found IN members.parquet | 0 |
| Overlap assertion | ✅ PASSED |

Zero overlap confirmed by assertion — ghost IDs (`MBR_GHOST_*`) are structurally separate from real member IDs (`MBR_*`). The assertion would raise a hard error if any ghost ID appeared in the spine.

---

## Step 2.3 — Zero-Activity Member Identification

| Metric | Count | % of Spine |
|--------|------:|----------:|
| Spine members total | 500,000 | 100% |
| Members with ≥1 transaction (excl. ghosts) | 499,823 | 99.96% |
| Members with ≥1 engagement event (excl. ghosts) | 497,957 | 99.59% |
| Members with ANY activity (union) | 499,931 | 99.99% |
| **Zero-activity members** | **69** | **0.01%** |

**Decision**: All 69 zero-activity members are KEPT in the spine.
They receive 0.0 for all count/spend features and NaN for recency in
every snapshot. Dropping them would bias downstream churn/segmentation models.
See `docs/decisions.md` Decision 004.

---

## Step 2.4 — Guest Cohort Parquet

The 88,717 ghost member IDs are saved with their full transaction history
to `spine/guest_cohort.parquet` for reference. They are excluded from all
member-level modeling but preserved for potential guest analysis.

| Metric | Value |
|--------|------:|
| Ghost rows saved | 88,717 |
| Unique ghost IDs | 88,717 |
| Each ghost: transactions only | 0 engagement events for any ghost |

---

## Step 2.5 — Spine Alias Note

`member_spine.parquet` and `enrolled_members.parquet` are **intentionally
identical** at this phase. Both contain exactly two columns:

```
member_id          (string, e.g. MBR_0000000)
account_open_date  (datetime64[us])
```

Future phases may differentiate them. As of Phase 2 they are aliases.
See `docs/decisions.md` Decision 005.

---

## Output Files

| File | Size | Rows | Purpose |
|------|-----:|-----:|---------|
| `spine/member_spine.parquet` | 3.2 MB | 500,000 | Canonical spine for Phase 3+ |
| `spine/enrolled_members.parquet` | 3.2 MB | 500,000 | Alias of above |
| `spine/guest_cohort.parquet` | 3.2 MB | 88,717 | Ghost members, excluded from modeling |
| `validation/spine_summary.json` | 0.3 KB | — | Reconciliation counts for audit trail |

---

## spine_summary.json (persisted)

```json
{
  "total_members": 500000,
  "ghost_members": 88717,
  "zero_activity_members": 69,
  "members_with_any_activity": 499931,
  "members_with_transactions": 499823,
  "members_with_engagement": 497957,
  "null_account_open_date": 0
}
```

---

## Exit Criteria — All 5 PASS

| Check | Result |
|-------|--------|
| One canonical spine, no duplicate member_ids | ✅ Asserted (hard error if violated) |
| Zero overlap between spine and ghost cohort | ✅ Asserted (hard error if violated) |
| Zero-activity members present in spine | ✅ 69 members confirmed and kept |
| guest_cohort.parquet saved separately | ✅ 88,717 rows |
| spine_summary.json persisted | ✅ Written to validation/ |

---

## What Phase 3 Needs from Phase 2

Phase 3 (snapshot builder) reads directly from:
- `spine/member_spine.parquet` — the LEFT JOIN anchor for all 12 snapshots
- `validation/ghost_cohort.csv` — to filter ghost IDs from transactions + engagement
- `validation/clip_bounds.json` — for session duration clipping

**No data outside these files is needed. Phase 2 is fully self-contained.**

---

## Next Step

```bash
python src/03_snapshot_builder.py
```

> Builds 12 monthly snapshots (2025-01-01 through 2025-12-01).
> Expected runtime: 30–60 minutes on this dataset.
> Outputs written to `snapshots/snapshot_YYYY_MM_DD.parquet` (12 files).
