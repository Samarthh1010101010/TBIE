"""
src/utils/datetime_parser.py
────────────────────────────
Robust dual-format datetime parser for the TBIE pipeline.

Rules (from locked spec):
  1. If the column is already datetime64, return it unchanged.
  2. Try ISO format: %Y-%m-%d %H:%M:%S
  3. Try day-first format: %d/%m/%Y %H:%M:%S
  4. Try ISO date-only: %Y-%m-%d
  5. Try day-first date-only: %d/%m/%Y
  6. Raise an explicit error for any non-null value that fails ALL formats.
     Never silently coerce or drop failures.
"""

import pandas as pd

# ── Ordered format list ──────────────────────────────────────────────────────
_FORMATS = [
    "%Y-%m-%d %H:%M:%S",   # ISO datetime  (dominant format)
    "%d/%m/%Y %H:%M:%S",   # day-first datetime (minority, confirmed in data)
    "%Y-%m-%d",             # ISO date-only (members table dates)
    "%d/%m/%Y",             # day-first date-only
]


def parse_mixed_datetime(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Strict multi-format parser. Tries each format in order, accumulating
    successfully parsed values. Raises ValueError if any non-null value
    fails every format.

    Parameters
    ----------
    series   : pd.Series  — raw datetime column (object/string or datetime64)
    col_name : str        — column name used in error messages only

    Returns
    -------
    pd.Series with dtype datetime64[ns]
    """

    # ── Fast path: already datetime64 ─────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # ── Initialise output series as NaT ───────────────────────────────────
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining_mask = series.notna()  # tracks rows not yet successfully parsed

    for fmt in _FORMATS:
        if not remaining_mask.any():
            break  # all non-null values resolved

        subset = series[remaining_mask]
        parsed = pd.to_datetime(subset, format=fmt, errors="coerce")

        # Rows successfully parsed under this format
        success_mask = parsed.notna()
        if success_mask.any():
            global_idx = remaining_mask[remaining_mask].index[success_mask]
            result.loc[global_idx] = parsed[success_mask].values
            # Remove successfully parsed rows from remaining work
            remaining_mask.loc[global_idx] = False

    # ── Check for unresolvable non-null values ─────────────────────────────
    final_failures = remaining_mask  # still True → not parsed by any format
    if final_failures.any():
        n = final_failures.sum()
        sample = series[final_failures].head(10).tolist()
        raise ValueError(
            f"[datetime_parser] {n} value(s) in column '{col_name}' failed "
            f"ALL {len(_FORMATS)} date formats.\n"
            f"Formats tried: {_FORMATS}\n"
            f"Sample failures (up to 10): {sample}\n"
            f"Inspect manually — do NOT silently coerce or drop these rows."
        )

    return result


def parse_date_column(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Convenience alias — same as parse_mixed_datetime.
    Use this when you want to be explicit that the column contains dates
    (with or without a time component).
    """
    return parse_mixed_datetime(series, col_name=col_name)
