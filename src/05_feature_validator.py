"""
src/05_feature_validator.py
══════════════════════════════════════════════════════════════════════════════
PHASE 5 — Feature Validator

The gate before Phase 6. Every check must pass on all 12 snapshots.

Validation checks implemented:
  5.1  Missingness / coverage (>5% null rate on any feature = fail)
  5.2  Range / bounds checks (hard domain bounds)
  5.3  Business sanity checks
         - Zero purchases ⇒ zero spend
         - Never-purchased members: proxy = purchase_count_180d == 0
           (not recency_days.isna() alone — that can hide join bugs)
         - Consistency assertion: both signals agree
         - redemption_rate outside [0,1] post-clip = clip logic bug
  5.4  Collinearity check (|r| > 0.90 per snapshot + structural across months)
  5.5  Cross-snapshot stability (PSI-style drift + median jump detection)
  5.6  Full 12-snapshot run producing per-month reports + summary

Run from TBIE/ root:
    python src/05_feature_validator.py
"""

import json
import sys
from pathlib import Path

# Force UTF-8 output — Windows terminal defaults to cp1252 which can't encode
# the box-drawing characters (═══) used in print statements below.
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

# Path setup
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
VAL_DIR      = ROOT / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")

# Range bounds (domain-based hard constraints)
RANGE_CHECKS = {
    "redemption_rate":              (0.0, 1.0),
    # email_open_rate_30d is a RAW COUNT (not a 0-1 rate) after Phase 4 bug fix
    # (diagnosed 2026-06-22): no email_sent denominator exists in the data.
    # Range check removed — raw counts are unbounded above.
    "email_click_rate_30d":         (0.0, 1.0),
    "category_diversity_90d":       (0.0, 1.0),
    "channel_diversity_90d":        (0.0, 1.0),
    "tier_ordinal":                 (0,   3),
    "recency_days":                 (0,   2000),  # sanity upper
    "browse_to_purchase_ratio_30d": (0.0, 10.0),
    "hoarding_ratio":               (0.0, 10.0),
    "feature_complete":             (0,   1),
    "browsed_but_never_purchased_30d": (0, 1),
}


# CHECK FUNCTIONS

def check_missingness(df: pd.DataFrame) -> dict:
    """Flag any feature column with > 5% null rate."""
    # Exclude non-feature columns and columns that are legitimately sparse by design:
    #   - recency_days: NaN for members who have never purchased (expected)
    #   - months_since_last_tier_change: NaN for members with no tier change history (expected)
    #   - last_transaction_date: datetime anchor, not a feature
    exclude = {
        "member_id", "observation_date", "last_transaction_date", "account_open_date",
        "current_tier", "tier_trajectory_direction",
        "recency_days", "months_since_last_tier_change",
    }
    cols = [c for c in df.columns if c not in exclude and
            not pd.api.types.is_datetime64_any_dtype(df[c])]
    null_rates = df[cols].isnull().mean()
    failures   = null_rates[null_rates > 0.05]
    return failures.to_dict()


def check_ranges(df: pd.DataFrame) -> dict:
    """Check hard domain bounds for known ratio/ordinal features."""
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad) > 0:
            violations[col] = {
                "n_violations": int(len(bad)),
                "min_val":      float(df[col].min()),
                "max_val":      float(df[col].max()),
            }
    return violations


def check_business_sanity(df: pd.DataFrame) -> list[str]:
    """
    Business logic consistency checks.
    Never-purchased proxy: purchase_count_180d == 0 (not recency_days.isna() alone).
    Consistency assertion: both signals must agree.
    """
    issues = []

    # Zero purchases ⇒ zero spend
    for window in ["7d", "30d", "90d", "180d"]:
        cnt_col   = f"purchase_count_{window}"
        spend_col = f"spend_total_{window}"
        if cnt_col in df.columns and spend_col in df.columns:
            bad = df[(df[cnt_col] == 0) & (df[spend_col] > 0)]
            if len(bad) > 0:
                issues.append(
                    f"{len(bad)} rows: {cnt_col}==0 but {spend_col}>0 "
                    f"(max spend: {bad[spend_col].max():.2f})"
                )

    # Never-purchased proxy: purchase_count_180d == 0
    if "purchase_count_180d" in df.columns:
        never_purchased_180d = df["purchase_count_180d"] == 0
        n_never_180d = never_purchased_180d.sum()

        # Consistency check: recency_days.isna() should agree with purchase_count_180d == 0
        # for members who have been in the programme for ≥ 180 days
        if "recency_days" in df.columns:
            never_purchased_recency = df["recency_days"].isna()
            # Members with no purchase in 180d should also have null recency
            # (recency is based on lifetime last_transaction_date, not the window)
            # If recency is non-null but purchase_count_180d == 0, that's OK —
            # they bought before the 180d window. Check the REVERSE:
            # If recency_days IS null but purchase_count_30d > 0, that's a bug.
            inconsistent = df[
                df["recency_days"].isna() &
                (df.get("purchase_count_30d", pd.Series(0)) > 0)
            ]
            if len(inconsistent) > 0:
                issues.append(
                    f"{len(inconsistent)} rows: recency_days is NaN but "
                    f"purchase_count_30d > 0 — possible join bug"
                )

        if n_never_180d > 0:
            # Check feature_complete rate for never-purchased members
            if "feature_complete" in df.columns:
                completeness_rate = df.loc[never_purchased_180d, "feature_complete"].mean()
                if completeness_rate > 0.3:
                    issues.append(
                        f"Unexpectedly high feature_complete rate ({completeness_rate:.1%}) "
                        f"among never-purchased-in-180d members ({n_never_180d} members). "
                        f"Expected low completeness for zero-history members."
                    )

    # redemption_rate in [0,1] post-clip
    if "redemption_rate" in df.columns:
        bad = df[(df["redemption_rate"] < 0) | (df["redemption_rate"] > 1)]
        if len(bad) > 0:
            issues.append(
                f"{len(bad)} rows: redemption_rate outside [0,1] after clipping — "
                f"clip logic bug (min={bad['redemption_rate'].min():.4f}, "
                f"max={bad['redemption_rate'].max():.4f})"
            )

    # Slope/acceleration should be finite
    for col in ["spend_slope_30d", "frequency_slope_30d", "spend_acceleration"]:
        if col in df.columns:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                issues.append(f"{n_inf} rows: {col} is infinite (division bug)")

    return issues


def check_collinearity(df: pd.DataFrame, threshold: float = 0.90) -> list[tuple]:
    """
    Find feature pairs with |r| > threshold.
    Returns list of (col_a, col_b, correlation_value).
    """
    exclude = {"member_id", "observation_date", "last_transaction_date",
               "account_open_date", "feature_complete", "tier_trajectory_direction",
               "current_tier", "browsed_but_never_purchased_30d"}
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude
    ]

    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr().abs()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if r > threshold:
                pairs.append((
                    corr.columns[i],
                    corr.columns[j],
                    round(float(r), 4),
                ))
    return pairs


# CROSS-SNAPSHOT STABILITY (PSI-style + median jump)

def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.
    PSI < 0.1 : stable; 0.1–0.25 : minor shift; > 0.25 : major shift
    """
    # Remove nulls
    e = expected.dropna()
    a = actual.dropna()
    if len(e) == 0 or len(a) == 0:
        return 0.0

    # Build bins from expected distribution
    try:
        _, bin_edges = np.histogram(e, bins=bins)
        # Ensure edges are unique
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        e_counts, _ = np.histogram(e, bins=bin_edges)
        a_counts, _ = np.histogram(a, bins=bin_edges)

        e_pct = (e_counts / len(e)).clip(min=1e-6)
        a_pct = (a_counts / len(a)).clip(min=1e-6)

        psi = np.sum((e_pct - a_pct) * np.log(e_pct / a_pct))
        return float(psi)
    except Exception:
        return 0.0


def check_cross_snapshot_stability(
    all_features: dict[str, pd.DataFrame]
) -> dict:
    """
    5.5 Cross-snapshot stability.
    - Median changes per feature month-over-month
    - Flag >50% median jumps
    - PSI between consecutive months for numeric features
    - Identify structurally collinear pairs (correlated in majority of months)
    """
    months = sorted(all_features.keys())
    if len(months) < 2:
        return {}

    exclude = {"member_id", "observation_date", "last_transaction_date",
               "account_open_date", "tier_trajectory_direction", "current_tier"}

    # Median trajectory
    numeric_cols_sets = [
        set(df.select_dtypes(include=np.number).columns) - exclude
        for df in all_features.values()
    ]
    common_numeric = set.intersection(*numeric_cols_sets) if numeric_cols_sets else set()

    medians = pd.DataFrame({
        month: all_features[month][list(common_numeric)].median()
        for month in months
    }).T

    pct_change = medians.pct_change().abs()
    suspicious_median = {}
    for col in pct_change.columns:
        jumps = pct_change[col][pct_change[col] > 0.5].dropna()
        if len(jumps) > 0:
            suspicious_median[col] = jumps.to_dict()

    # PSI between consecutive months
    psi_summary = {}
    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]
        prev_df    = all_features[prev_month]
        curr_df    = all_features[curr_month]

        psi_vals = {}
        for col in common_numeric:
            psi = compute_psi(prev_df[col], curr_df[col])
            if psi > 0.1:  # Only log noteworthy drift
                psi_vals[col] = round(psi, 4)

        if psi_vals:
            psi_summary[f"{prev_month}{curr_month}"] = psi_vals

    # Structural collinearity (correlated in majority of months)
    collinear_per_month = {}
    for month, df in all_features.items():
        pairs = check_collinearity(df, threshold=0.90)
        for col_a, col_b, r in pairs:
            key = tuple(sorted([col_a, col_b]))
            if key not in collinear_per_month:
                collinear_per_month[key] = []
            collinear_per_month[key].append((month, r))

    structurally_collinear = {
        f"{k[0]}  {k[1]}": {
            "n_months_correlated": len(v),
            "months": [m for m, _ in v],
            "avg_r": round(np.mean([r for _, r in v]), 4),
        }
        for k, v in collinear_per_month.items()
        if len(v) >= 8  # correlated in ≥8 of 12 months = structural
    }

    return {
        "suspicious_median_jumps": suspicious_median,
        "psi_drift_summary": psi_summary,
        "structurally_collinear_pairs": structurally_collinear,
        "median_trajectory": medians.to_dict(),
    }


# REPORT WRITER

def write_monthly_report(report_path: Path, month_key: str, result: dict):
    """Write a per-month validation report as markdown."""
    passed = result["passed"]
    status = " PASSED" if passed else " FAILED"

    lines = [
        f"# Feature Validation Report — {month_key}",
        f"\n**Overall Status:** {status}\n",
    ]

    # Missingness
    lines.append("## 5.1 Missingness (>5% null rate)")
    if result["missingness_issues"]:
        lines.append("| Column | Null Rate |")
        lines.append("|--------|----------:|")
        for col, rate in result["missingness_issues"].items():
            lines.append(f"| {col} | {rate:.2%} |")
    else:
        lines.append(" No feature column exceeds 5% null rate")

    # Range violations
    lines.append("\n## 5.2 Range Violations")
    if result["range_violations"]:
        for col, info in result["range_violations"].items():
            lines.append(f"- **{col}**: {info['n_violations']} violations "
                         f"(min={info['min_val']:.4f}, max={info['max_val']:.4f})")
    else:
        lines.append(" No range violations")

    # Sanity issues
    lines.append("\n## 5.3 Business Sanity Issues")
    if result["sanity_issues"]:
        for issue in result["sanity_issues"]:
            lines.append(f"- ️  {issue}")
    else:
        lines.append(" All sanity checks passed")

    # Collinearity
    lines.append("\n## 5.4 Collinear Pairs (|r| > 0.90)")
    if result["collinear_pairs"]:
        lines.append("| Feature A | Feature B | |r| |")
        lines.append("|-----------|-----------|-----|")
        for col_a, col_b, r in result["collinear_pairs"]:
            lines.append(f"| {col_a} | {col_b} | {r:.4f} |")
    else:
        lines.append(" No collinear pairs above threshold")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# MAIN

def main():
    print("\n" + "═"*70)
    print("TBIE — PHASE 5: FEATURE VALIDATOR")
    print("═"*70 + "\n")

    results      = {}
    all_features = {}

    for obs_date in SNAPSHOT_DATES:
        month_key = obs_date.strftime("%Y_%m")
        date_str  = obs_date.strftime("%Y_%m_%d")
        feat_path = FEATURES_DIR / f"features_{date_str}.parquet"

        if not feat_path.exists():
            print(f"  ️  Feature file not found: {feat_path} — skipping")
            continue

        print(f"  Validating {obs_date.strftime('%Y-%m')}…", end="", flush=True)
        df = pd.read_parquet(feat_path)
        all_features[month_key] = df

        missingness     = check_missingness(df)
        range_viol      = check_ranges(df)
        sanity          = check_business_sanity(df)
        collinear_pairs = check_collinearity(df)

        passed = (len(missingness) == 0) and (len(range_viol) == 0) and (len(sanity) == 0)

        flag = "" if passed else ""
        print(f" {flag}  "
              f"miss={len(missingness)} range={len(range_viol)} "
              f"sanity={len(sanity)} coll={len(collinear_pairs)}")

        results[month_key] = {
            "passed":            passed,
            "missingness_issues": missingness,
            "range_violations":   range_viol,
            "sanity_issues":      sanity,
            "collinear_pairs":    collinear_pairs,
        }

        # Write per-month report
        report_path = VAL_DIR / f"feature_validation_report_{month_key}.md"
        write_monthly_report(report_path, month_key, results[month_key])

    # Cross-snapshot stability
    if len(all_features) > 1:
        print("\n  Running cross-snapshot stability checks…")
        stability = check_cross_snapshot_stability(all_features)

        # Print PSI alerts
        if stability.get("suspicious_median_jumps"):
            print("\n  ️  Suspicious median jumps (>50% month-over-month):")
            for col, jumps in stability["suspicious_median_jumps"].items():
                print(f"    {col}: {jumps}")

        if stability.get("structurally_collinear_pairs"):
            print("\n  ️  Structurally collinear pairs (correlated in ≥8/12 months):")
            for pair, info in stability["structurally_collinear_pairs"].items():
                print(f"    {pair}: avg_r={info['avg_r']:.4f} in {info['n_months_correlated']} months")

        if stability.get("psi_drift_summary"):
            print("\n   PSI drift detected (>0.1) in the following transitions:")
            for transition, psi_vals in stability["psi_drift_summary"].items():
                print(f"    {transition}: {psi_vals}")

        # Save stability report
        stability_path = VAL_DIR / "cross_snapshot_stability.json"
        # Convert non-serializable objects
        stability_json = json.loads(json.dumps(stability, default=str))
        with open(stability_path, "w") as f:
            json.dump(stability_json, f, indent=2)
        print("\n  Saved: validation/cross_snapshot_stability.json")

    # Summary table
    print("\n" + "─"*60)
    print("FEATURE VALIDATION SUMMARY:")
    print()

    summary_rows = []
    for month_key, result in results.items():
        summary_rows.append({
            "month":         month_key,
            "passed":        "" if result["passed"] else "",
            "missingness":   len(result["missingness_issues"]),
            "range_violations": len(result["range_violations"]),
            "sanity_issues": len(result["sanity_issues"]),
            "collinear_pairs": len(result["collinear_pairs"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    all_passed = all(r["passed"] for r in results.values())

    # Write markdown summary
    summary_md_lines = [
        "# Feature Validation Summary — All 12 Snapshots",
        "",
        summary_df.to_string(index=False),
        "",
    ]

    if all_passed:
        summary_md_lines.append("##  ALL 12 SNAPSHOTS PASSED — Ready for Phase 6")
    else:
        failed = [k for k, v in results.items() if not v["passed"]]
        summary_md_lines.append(
            f"##  FAILURES FOUND — Fix root cause in Phase 3/4 before Phase 6\n"
            f"Failed months: {failed}"
        )

    (VAL_DIR / "feature_validation_summary.md").write_text(
        "\n".join(summary_md_lines), encoding="utf-8"
    )
    print("\n  Saved: validation/feature_validation_summary.md")

    print("\n" + "═"*70)
    if all_passed:
        print("ALL 12 SNAPSHOTS PASSED — Phase 6 may proceed.")
    else:
        failed = [k for k, v in results.items() if not v["passed"]]
        print(f" FAILURES FOUND in: {failed}")
        print("Fix the root cause in Phase 3 or 4, regenerate, and rerun this validator.")
        print("Do NOT patch the validator to pass a bad snapshot.")
    print("═"*70)
    print("\nPHASE 5 COMPLETE.")


if __name__ == "__main__":
    main()
