"""
src/utils/leakage_guard.py
──────────────────────────
Leakage assertion helpers for the TBIE pipeline.

These functions are called on EVERY snapshot build, not just spot-checked.
A failure here means the pipeline has produced forward-looking data and
must be fixed before the snapshot is used downstream.
"""

import pandas as pd


def assert_no_future_data(
    df: pd.DataFrame,
    date_col: str,
    observation_date: pd.Timestamp,
    context: str = "",
) -> None:
    """
    Asserts that no row in `df` has `date_col` > `observation_date`.

    Parameters
    ----------
    df               : DataFrame to check
    date_col         : name of the datetime column to test
    observation_date : the snapshot cutoff date
    context          : human-readable label for error messages (e.g. phase + table)

    Raises
    ------
    AssertionError if any row violates the constraint.
    """
    if date_col not in df.columns:
        raise KeyError(
            f"[leakage_guard] Column '{date_col}' not found in dataframe. "
            f"Context: {context}"
        )

    violations = df[df[date_col] > observation_date]
    if len(violations) > 0:
        raise AssertionError(
            f"[leakage_guard] LEAKAGE DETECTED in '{context}':\n"
            f"  {len(violations)} rows have {date_col} > observation_date "
            f"({observation_date}).\n"
            f"  Max violating date: {violations[date_col].max()}\n"
            f"  Sample member IDs: {violations['member_id'].head(5).tolist() if 'member_id' in violations.columns else 'N/A'}\n"
            f"Fix the upstream filter before proceeding."
        )


def assert_pre_enrollment_removed(
    df: pd.DataFrame,
    date_col: str,
    enrollment_col: str,
    context: str = "",
) -> None:
    """
    Asserts that no row in `df` has `date_col` < `enrollment_col`.
    Used to verify that pre-enrollment transactions/events have been removed.

    Parameters
    ----------
    df              : DataFrame to check
    date_col        : activity date column (e.g. 'transaction_date')
    enrollment_col  : member enrollment/open date column (e.g. 'account_open_date')
    context         : human-readable label for error messages

    Raises
    ------
    AssertionError if any row violates the constraint.
    """
    if date_col not in df.columns:
        raise KeyError(f"[leakage_guard] Column '{date_col}' not found. Context: {context}")
    if enrollment_col not in df.columns:
        raise KeyError(f"[leakage_guard] Column '{enrollment_col}' not found. Context: {context}")

    violations = df[df[date_col] < df[enrollment_col]]
    if len(violations) > 0:
        raise AssertionError(
            f"[leakage_guard] PRE-ENROLLMENT ACTIVITY DETECTED in '{context}':\n"
            f"  {len(violations)} rows have {date_col} < {enrollment_col}.\n"
            f"  Pre-enrollment activity was not filtered correctly.\n"
            f"  Sample member IDs: {violations['member_id'].head(5).tolist() if 'member_id' in violations.columns else 'N/A'}\n"
            f"Fix the upstream filter before proceeding."
        )
