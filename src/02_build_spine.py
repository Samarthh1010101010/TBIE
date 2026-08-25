"""
src/02_build_spine.py
══════════════════════════════════════════════════════════════════════════════
PHASE 2 — Member Spine Builder

Builds the canonical member spine:
  1. Load members.parquet, parse account_open_date with the shared utility
  2. Assert member_id uniqueness
  3. Exclude ghost cohort (IDs from Phase 1 validation/ghost_cohort.csv)
  4. Assert zero overlap between spine and ghost IDs
  5. Identify and log zero-activity members (kept in spine — never dropped)
  6. Write member_spine.parquet and enrolled_members.parquet (intentional aliases)
  7. Write guest_cohort.parquet
  8. Write spine_summary.json (persistence of reconciliation counts)

Run from TBIE/ root:
    python src/02_build_spine.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Path setup
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "raw"
SPINE_DIR = ROOT / "spine"
VAL_DIR   = ROOT / "validation"

SPINE_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from utils.datetime_parser import parse_mixed_datetime


def main():
    print("\n" + "═"*70)
    print("TBIE — PHASE 2: MEMBER SPINE BUILDER")
    print("═"*70 + "\n")

    # Check prerequisite outputs from Phase 1
    ghost_csv = VAL_DIR / "ghost_cohort.csv"
    if not ghost_csv.exists():
        print(f"  ️  Ghost cohort file not found: {ghost_csv}")
        print("  Run src/01_validate_raw.py first.")
        sys.exit(1)

    # Step 2.1 — Load members.parquet
    print("Loading members.parquet…")
    members_df = pd.read_parquet(DATA_DIR / "members.parquet")
    print(f"  Loaded: {members_df.shape[0]:,} rows × {members_df.shape[1]} columns")

    # Step 2.1 — Parse account_open_date
    print("\nParsing account_open_date…")
    members_df["account_open_date"] = parse_mixed_datetime(
        members_df["account_open_date"], col_name="account_open_date"
    )
    print(f"  account_open_date dtype: {members_df['account_open_date'].dtype}")
    print(f"  Range: {members_df['account_open_date'].min()}  {members_df['account_open_date'].max()}")
    null_aod = members_df["account_open_date"].isnull().sum()
    if null_aod > 0:
        print(f"  ️  {null_aod:,} rows have null account_open_date — flagged for special handling")
    else:
        print("   No null account_open_date values")

    # Step 2.1 — Assert member_id uniqueness
    print("\nAsserting member_id uniqueness…")
    n_dupes = members_df["member_id"].duplicated().sum()
    assert n_dupes == 0, (
        f"Duplicate member_id in members.parquet — {n_dupes} duplicates found. "
        f"Investigate before proceeding."
    )
    print(f"   member_id is unique ({members_df['member_id'].nunique():,} unique IDs)")

    # Step 2.2 — Build canonical spine
    print("\nBuilding canonical spine (member_id + account_open_date)…")
    spine = members_df[["member_id", "account_open_date"]].copy()
    print(f"  Spine shape: {spine.shape}")

    # Step 2.2 — Load ghost cohort and exclude from spine
    print("\nLoading ghost cohort…")
    ghost_ids = set(pd.read_csv(ghost_csv)["member_id"].tolist())
    print(f"  Ghost cohort size (from Phase 1): {len(ghost_ids):,}")

    # Assert zero overlap: ghost IDs should NOT be in members.parquet by definition
    overlap = set(spine["member_id"]) & ghost_ids
    assert len(overlap) == 0, (
        f"CRITICAL: {len(overlap)} ghost IDs found in members.parquet! "
        f"Ghost IDs must be IDs that are NOT in members.parquet. "
        f"Sample overlapping IDs: {list(overlap)[:5]}"
    )
    print("   Zero overlap between spine and ghost cohort (asserted)")

    # Step 2.3 — Identify zero-activity members
    print("\nIdentifying zero-activity members…")
    txns_df = pd.read_parquet(DATA_DIR / "transactions.parquet",
                               columns=["member_id"])
    engagement_df = pd.read_parquet(DATA_DIR / "engagement_events.parquet",
                                    columns=["member_id"])

    members_with_txns = set(txns_df["member_id"].unique())
    members_with_eng  = set(engagement_df["member_id"].unique())
    # Exclude ghost IDs from activity sets before determining zero-activity
    known_members_with_txns = members_with_txns - ghost_ids
    known_members_with_eng  = members_with_eng - ghost_ids
    members_with_activity = known_members_with_txns | known_members_with_eng

    spine_ids           = set(spine["member_id"])
    zero_activity_ids   = spine_ids - members_with_activity
    n_zero_activity     = len(zero_activity_ids)
    pct_zero            = n_zero_activity / len(spine) * 100

    print(f"  Spine members: {len(spine_ids):,}")
    print(f"  Members with ≥1 transaction (excl. ghosts): {len(known_members_with_txns):,}")
    print(f"  Members with ≥1 engagement event (excl. ghosts): {len(known_members_with_eng):,}")
    print(f"  Members with ANY activity: {len(members_with_activity):,}")
    print(f"  Zero-activity members: {n_zero_activity:,} ({pct_zero:.2f}%)")
    print("   Zero-activity members KEPT in spine (as per spec)")

    # Step 2.2 — Save guest_cohort.parquet
    print("\nBuilding guest cohort parquet…")
    # Load ghost transaction details
    txns_full = pd.read_parquet(DATA_DIR / "transactions.parquet")
    guest_cohort_txns = txns_full[txns_full["member_id"].isin(ghost_ids)].copy()
    guest_cohort_txns.to_parquet(SPINE_DIR / "guest_cohort.parquet", index=False)
    print(f"  Saved: spine/guest_cohort.parquet ({len(guest_cohort_txns):,} rows, "
          f"{guest_cohort_txns['member_id'].nunique():,} unique ghost IDs)")

    # Step 2.4 — Write spine outputs
    print("\nWriting spine outputs…")
    spine.to_parquet(SPINE_DIR / "member_spine.parquet", index=False)
    spine.to_parquet(SPINE_DIR / "enrolled_members.parquet", index=False)

    print(f"  Saved: spine/member_spine.parquet      ({len(spine):,} rows)")
    print(f"  Saved: spine/enrolled_members.parquet  ({len(spine):,} rows)")
    print("\n  NOTE (Decision 005): member_spine.parquet and enrolled_members.parquet")
    print("  are intentionally IDENTICAL at this phase. See docs/decisions.md #005.")

    # Step 2.5 — Write spine_summary.json
    print("\nWriting spine_summary.json…")
    spine_summary = {
        "total_members": int(len(spine)),
        "ghost_members": int(len(ghost_ids)),
        "zero_activity_members": int(n_zero_activity),
        "members_with_any_activity": int(len(members_with_activity)),
        "members_with_transactions": int(len(known_members_with_txns)),
        "members_with_engagement": int(len(known_members_with_eng)),
        "null_account_open_date": int(null_aod),
        "spine_files": [
            "spine/member_spine.parquet",
            "spine/enrolled_members.parquet",
        ],
        "note_aliases": (
            "member_spine.parquet and enrolled_members.parquet are identical "
            "at Phase 2. See docs/decisions.md Decision 005."
        ),
    }

    with open(VAL_DIR / "spine_summary.json", "w") as f:
        json.dump(spine_summary, f, indent=2)
    print("  Saved: validation/spine_summary.json")

    # Exit criteria summary
    print("\n" + "─"*60)
    print("PHASE 2 EXIT CRITERIA:")
    print("   One canonical member_id spine, no duplicates (asserted)")
    print("   Zero overlap between spine and ghost cohort (asserted)")
    print(f"   Zero-activity members confirmed present: {n_zero_activity:,}")
    print("   guest_cohort.parquet separately saved")
    print("   spine_summary.json persisted for downstream consumption")
    print("\nPHASE 2 COMPLETE.")


if __name__ == "__main__":
    main()
