#!/usr/bin/env python
"""
scripts/check_model_contract.py
───────────────────────────────
Verify the shipped model bundles satisfy the contract pipeline.py depends on.

This exists because the training script and the inference script once disagreed
on key names: src/08 wrote 'x_cols' while pipeline.py read 'feature_cols'. The
committed model happened to have the right keys because it was produced by a
script that was never checked in, so the repo could not actually rebuild the
model it shipped, and nobody noticed until inference was run on a fresh clone.

Exits non-zero on any violation. Wired into CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.state_rules import STATE_IDS  # noqa: E402

TRANSITION_PKL = ROOT / "models"   / "segment_transition_model.pkl"
SEGMENT_PKL    = ROOT / "segments" / "segment_model.pkl"

# Keys pipeline.py reads off the transition bundle.
TRANSITION_KEYS = [
    "model",
    "feature_cols",
    "velocity_cols",
    "cluster_to_sid",
    "sid_to_name",
    "state_map",
]

# Keys pipeline.py reads off the segment bundle.
SEGMENT_KEYS = ["scaler", "pca", "centroids", "behavioral_feature_cols", "k"]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    for path in (TRANSITION_PKL, SEGMENT_PKL):
        if not path.exists():
            print(f"SKIP: {path.relative_to(ROOT)} not present "
                  f"(expected on a fresh clone without model artifacts)")
            return 0

    print(f"Checking {TRANSITION_PKL.relative_to(ROOT)}")
    tb = joblib.load(TRANSITION_PKL)
    for key in TRANSITION_KEYS:
        check(key in tb, f"transition bundle missing key '{key}'")

    print(f"Checking {SEGMENT_PKL.relative_to(ROOT)}")
    sb = joblib.load(SEGMENT_PKL)
    for key in SEGMENT_KEYS:
        check(key in sb, f"segment bundle missing key '{key}'")

    if "state_map" in tb:
        check(
            tb["state_map"] == STATE_IDS,
            "transition bundle state_map disagrees with state_rules.STATE_IDS — "
            "y_curr would be encoded differently at train and serve time.\n"
            f"    bundle     : {dict(sorted(tb['state_map'].items(), key=lambda x: x[1]))}\n"
            f"    state_rules: {dict(sorted(STATE_IDS.items(), key=lambda x: x[1]))}",
        )

    if "feature_cols" in tb and "velocity_cols" in tb:
        check(
            all(c in tb["feature_cols"] for c in tb["velocity_cols"]),
            "velocity_cols are not all present in feature_cols",
        )
        for ctx in ("seg_curr", "y_curr", "month_num"):
            check(ctx in tb["feature_cols"], f"feature_cols missing context column '{ctx}'")

    if "cluster_to_sid" in tb and "sid_to_name" in tb:
        check(
            set(tb["cluster_to_sid"].values()) == set(tb["sid_to_name"].keys()),
            "cluster_to_sid values do not match sid_to_name keys",
        )

    if "feature_cols" in tb and "model" in tb:
        n_model = getattr(tb["model"], "n_features_in_", None)
        if n_model is not None:
            check(
                n_model == len(tb["feature_cols"]),
                f"model expects {n_model} features but feature_cols has "
                f"{len(tb['feature_cols'])}",
            )

    if failures:
        print("\nFAILED — model bundle does not satisfy the inference contract:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nOK — both bundles satisfy the contract pipeline.py expects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
