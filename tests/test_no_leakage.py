"""
tests/test_no_leakage.py
════════════════════════════════════════════════════════════════════════════
Unit tests for leakage guards and datetime parser.

Run from TBIE/ root:
    python -m pytest tests/ -v

Or individually:
    python tests/test_no_leakage.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.datetime_parser import parse_mixed_datetime
from utils.leakage_guard import assert_no_future_data, assert_pre_enrollment_removed

# ════════════════════════════════════════════════════════════════════════════
# DATETIME PARSER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestDatetimeParser:

    def test_iso_datetime_parsed_correctly(self):
        s = pd.Series(["2025-03-14 15:30:22", "2025-01-01 00:00:00"])
        result = parse_mixed_datetime(s, "test_col")
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0] == pd.Timestamp("2025-03-14 15:30:22")
        assert result.iloc[1] == pd.Timestamp("2025-01-01 00:00:00")

    def test_dayfirst_datetime_parsed_correctly(self):
        s = pd.Series(["14/03/2025 15:30:22", "01/01/2025 00:00:00"])
        result = parse_mixed_datetime(s, "test_col")
        assert result.iloc[0] == pd.Timestamp("2025-03-14 15:30:22")
        assert result.iloc[1] == pd.Timestamp("2025-01-01 00:00:00")

    def test_mixed_formats_in_same_series(self):
        """Day 13+ disambiguates dayfirst vs month-first."""
        s = pd.Series([
            "2025-03-14 15:30:22",   # ISO
            "14/03/2025 15:30:22",   # day-first
            "2025-01-01 00:00:00",   # ISO
        ])
        result = parse_mixed_datetime(s, "test_col")
        assert result.iloc[0] == pd.Timestamp("2025-03-14 15:30:22")
        assert result.iloc[1] == pd.Timestamp("2025-03-14 15:30:22")
        assert result.iloc[2] == pd.Timestamp("2025-01-01 00:00:00")

    def test_iso_date_only(self):
        s = pd.Series(["2025-06-15", "2024-12-31"])
        result = parse_mixed_datetime(s, "test_col")
        assert result.iloc[0] == pd.Timestamp("2025-06-15")
        assert result.iloc[1] == pd.Timestamp("2024-12-31")

    def test_dayfirst_date_only(self):
        s = pd.Series(["15/06/2025", "31/12/2024"])
        result = parse_mixed_datetime(s, "test_col")
        assert result.iloc[0] == pd.Timestamp("2025-06-15")
        assert result.iloc[1] == pd.Timestamp("2024-12-31")

    def test_already_datetime64_returned_unchanged(self):
        s = pd.Series(pd.to_datetime(["2025-03-14", "2025-06-15"]))
        result = parse_mixed_datetime(s, "test_col")
        assert pd.api.types.is_datetime64_any_dtype(result)
        pd.testing.assert_series_equal(result, s)

    def test_null_values_preserved_as_nat(self):
        s = pd.Series(["2025-01-01 00:00:00", None, "2025-06-15 10:00:00"])
        result = parse_mixed_datetime(s, "test_col")
        assert pd.isna(result.iloc[1])
        assert result.iloc[0] == pd.Timestamp("2025-01-01 00:00:00")

    def test_unparseable_nonnull_raises_valueerror(self):
        s = pd.Series(["not-a-date", "2025-01-01 00:00:00"])
        with pytest.raises(ValueError, match="failed ALL"):
            parse_mixed_datetime(s, "bad_col")

    def test_all_null_series_returns_nat(self):
        s = pd.Series([None, None, None])
        result = parse_mixed_datetime(s, "test_col")
        assert result.isna().all()

    def test_empty_series(self):
        s = pd.Series([], dtype=object)
        result = parse_mixed_datetime(s, "test_col")
        assert len(result) == 0


# ════════════════════════════════════════════════════════════════════════════
# LEAKAGE GUARD TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestLeakageGuard:

    def _make_txn_df(self, dates, member_ids=None):
        if member_ids is None:
            member_ids = [f"M{i}" for i in range(len(dates))]
        return pd.DataFrame({
            "member_id":       member_ids,
            "transaction_date": pd.to_datetime(dates),
        })

    def test_no_future_data_passes_clean_data(self):
        obs = pd.Timestamp("2025-06-01")
        df  = self._make_txn_df(["2025-01-01", "2025-05-31"])
        # Should not raise
        assert_no_future_data(df, "transaction_date", obs, "test")

    def test_no_future_data_raises_on_future_row(self):
        obs = pd.Timestamp("2025-06-01")
        df  = self._make_txn_df(["2025-01-01", "2025-06-02"])  # 2nd row is future
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            assert_no_future_data(df, "transaction_date", obs, "test")

    def test_no_future_data_on_exact_obs_date_passes(self):
        obs = pd.Timestamp("2025-06-01")
        df  = self._make_txn_df(["2025-06-01", "2025-06-01"])  # exactly obs_date = OK
        assert_no_future_data(df, "transaction_date", obs, "test")

    def test_no_future_data_missing_column_raises_keyerror(self):
        df = pd.DataFrame({"wrong_col": [1, 2]})
        with pytest.raises(KeyError):
            assert_no_future_data(df, "transaction_date", pd.Timestamp("2025-01-01"), "test")

    def test_pre_enrollment_removed_passes_clean_data(self):
        df = pd.DataFrame({
            "member_id":        ["M1", "M2"],
            "transaction_date": pd.to_datetime(["2025-03-01", "2025-06-01"]),
            "account_open_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        })
        assert_pre_enrollment_removed(df, "transaction_date", "account_open_date", "test")

    def test_pre_enrollment_removed_raises_on_violation(self):
        df = pd.DataFrame({
            "member_id":        ["M1"],
            "transaction_date": pd.to_datetime(["2024-12-01"]),   # before enrollment
            "account_open_date": pd.to_datetime(["2025-01-01"]),
        })
        with pytest.raises(AssertionError, match="PRE-ENROLLMENT"):
            assert_pre_enrollment_removed(df, "transaction_date", "account_open_date", "test")

    def test_pre_enrollment_on_same_date_passes(self):
        df = pd.DataFrame({
            "member_id":        ["M1"],
            "transaction_date": pd.to_datetime(["2025-01-01"]),   # same as open date = OK
            "account_open_date": pd.to_datetime(["2025-01-01"]),
        })
        assert_pre_enrollment_removed(df, "transaction_date", "account_open_date", "test")


# ════════════════════════════════════════════════════════════════════════════
# SNAPSHOT INTEGRITY CHECKS (if snapshots exist)
# ════════════════════════════════════════════════════════════════════════════

class TestSnapshotIntegrity:

    def _get_snapshot_paths(self):
        snap_dir = ROOT / "snapshots"
        return sorted(snap_dir.glob("snapshot_*.parquet")) if snap_dir.exists() else []

    def test_snapshots_have_no_future_transactions(self):
        """
        For each snapshot, verify all transaction-derived dates ≤ observation_date.
        """
        paths = self._get_snapshot_paths()
        if not paths:
            pytest.skip("No snapshots found — run Phase 3 first")

        for path in paths:
            df = pd.read_parquet(path)
            obs_str    = path.stem.replace("snapshot_", "").replace("_", "-")
            obs_date   = pd.Timestamp(obs_str)

            # last_transaction_date must be ≤ observation_date
            if "last_transaction_date" in df.columns:
                valid_rows = df[df["last_transaction_date"].notna()]
                future     = valid_rows[valid_rows["last_transaction_date"] > obs_date]
                assert len(future) == 0, (
                    f"LEAKAGE in {path.name}: "
                    f"{len(future)} rows have last_transaction_date > {obs_date}"
                )

    def test_snapshot_row_count_equals_spine(self):
        """All snapshots should have the same row count as the member spine."""
        spine_path = ROOT / "spine" / "member_spine.parquet"
        if not spine_path.exists():
            pytest.skip("Spine not found")

        spine      = pd.read_parquet(spine_path)
        spine_n    = len(spine)
        paths      = self._get_snapshot_paths()

        if not paths:
            pytest.skip("No snapshots found")

        for path in paths:
            df = pd.read_parquet(path)
            assert len(df) == spine_n, (
                f"{path.name}: expected {spine_n} rows (spine), got {len(df)}"
            )

    def test_no_apply_in_pipeline_files(self):
        """
        Heuristic check: no .apply() calls on large tables in Phase 3/4 scripts.
        This is a code review guard, not a runtime check.
        """
        scripts_to_check = [
            ROOT / "src" / "03_snapshot_builder.py",
            ROOT / "src" / "04_feature_engine.py",
        ]
        for script in scripts_to_check:
            if not script.exists():
                continue
            content = script.read_text(encoding="utf-8")
            # Count .apply( calls that are NOT in a comment
            lines = content.splitlines()
            apply_in_code = [
                l for l in lines
                if ".apply(" in l and not l.strip().startswith("#")
            ]
            # The spec allows .apply() on small frames — we just count and warn
            if apply_in_code:
                print(f"\n  INFO: {script.name} has {len(apply_in_code)} .apply() call(s):")
                for l in apply_in_code[:5]:
                    print(f"    {l.strip()}")


if __name__ == "__main__":
    # Run as script (without pytest)
    import traceback

    tests = [
        TestDatetimeParser,
        TestLeakageGuard,
        TestSnapshotIntegrity,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_class in tests:
        instance = test_class()
        for method_name in dir(instance):
            if not method_name.startswith("test_"):
                continue
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✅ {test_class.__name__}.{method_name}")
                passed += 1
            except pytest.skip.Exception as e:
                print(f"  ⏭️  {test_class.__name__}.{method_name} — SKIPPED: {e}")
                skipped += 1
            except Exception:
                print(f"  ❌ {test_class.__name__}.{method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'─'*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
