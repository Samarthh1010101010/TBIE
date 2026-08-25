"""
tests/test_state_rules.py
════════════════════════════════════════════════════════════════════════════
Regression tests for the canonical 10-state cascade.

These exist because the cascade was previously implemented twice (Phase 7 and
pipeline.py) and the copies drifted: Value Maximizer required redemption_rate
>= 0.21 at inference versus > 0.10 in training, Momentum Builder flipped > to
>=, and Plateau Cruiser grew an undocumented recency clause. Nothing caught it
because the rules had no tests at all.

Each test pins one rule to the thresholds documented in METHODOLOGY.md §4.

Run from TBIE_CODE root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.state_rules import (  # noqa: E402
    STATE_IDS,
    STATE_ORDER,
    VALID_STATES,
    build_conditions,
    classify_states,
    state_ids_from_names,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

# A member who matches no rule and therefore lands on the catch-all.
NEUTRAL = {
    "tenure_days":            5000.0,   # long-tenured, not New
    "recency_days":           10.0,     # recent, not lapsed
    "purchase_count_30d":     0.0,      # no purchases -> not Plateau/Silent
    # Below the Plateau window (-2) and below the Momentum threshold (+5), so
    # the neutral member trips neither. A large positive slope would silently
    # satisfy Momentum Builder whenever a test raises frequency and diversity.
    "spend_slope_30d":        -10.0,
    "redemption_rate":        0.0,
    "app_open_30d":           1.0,      # non-zero -> not Silent Accumulator
    "email_open_30d":         0.0,
    "push_open_30d":          0.0,
    "social_share_30d":       0.0,
    "referral_sent_30d":      0.0,
    "survey_completed_30d":   0.0,
    "tier_ordinal":           0.0,
    "category_diversity_90d": 0.0,
}


def member(**overrides) -> pd.DataFrame:
    """One-row frame: a neutral member with specific fields overridden."""
    row = dict(NEUTRAL)
    row.update(overrides)
    return pd.DataFrame([row])


def state_of(**overrides) -> str:
    return classify_states(member(**overrides))[0]


# ── Baseline ─────────────────────────────────────────────────────────────────

class TestCatchAll:

    def test_neutral_member_is_program_skeptic(self):
        """The NEUTRAL fixture must match no rule, or every test below is void."""
        assert state_of() == "Program Skeptic"

    def test_every_row_receives_a_state(self):
        df = pd.DataFrame([NEUTRAL] * 50)
        out = classify_states(df)
        assert len(out) == 50
        assert set(out) <= VALID_STATES

    def test_no_nulls_produced_on_all_nan_input(self):
        df = pd.DataFrame([{k: np.nan for k in NEUTRAL}])
        out = classify_states(df)
        assert out[0] in VALID_STATES

    def test_missing_columns_are_tolerated(self):
        """Sparse engagement columns are absent from some feature files."""
        df = pd.DataFrame([{"tenure_days": 5000.0, "recency_days": 10.0}])
        out = classify_states(df)
        assert out[0] in VALID_STATES


# ── Priority-ordered rules (METHODOLOGY.md §4) ───────────────────────────────

class TestStateRules:

    # P1 — New & Uncertain: tenure_days < 90
    def test_new_and_uncertain_fires_below_90_days(self):
        assert state_of(tenure_days=89.0) == "New & Uncertain"

    def test_new_and_uncertain_boundary_is_exclusive(self):
        assert state_of(tenure_days=90.0) != "New & Uncertain"

    def test_negative_tenure_is_new_and_uncertain(self):
        """Pre-enrolment rows carry negative tenure and must not crash."""
        assert state_of(tenure_days=-30.0) == "New & Uncertain"

    def test_new_and_uncertain_outranks_every_other_rule(self):
        # Also satisfies Win-Back Target, but P1 wins.
        assert state_of(tenure_days=10.0, recency_days=90.0,
                        app_open_30d=5.0) == "New & Uncertain"

    # P2 — Win-Back Target: recency > 60 AND any digital re-engagement
    def test_win_back_target_requires_reengagement_signal(self):
        assert state_of(recency_days=61.0, app_open_30d=1.0) == "Win-Back Target"

    def test_win_back_target_via_email_or_push(self):
        assert state_of(recency_days=61.0, app_open_30d=0.0,
                        email_open_30d=1.0) == "Win-Back Target"
        assert state_of(recency_days=61.0, app_open_30d=0.0,
                        push_open_30d=1.0) == "Win-Back Target"

    def test_lapsed_without_signal_is_not_win_back(self):
        assert state_of(recency_days=61.0, app_open_30d=0.0) != "Win-Back Target"

    def test_win_back_boundary_is_exclusive(self):
        assert state_of(recency_days=60.0, app_open_30d=1.0) != "Win-Back Target"

    # P3 — Lapse Risk: recency > 30 AND no 30d purchases AND slope <= 0
    def test_lapse_risk(self):
        assert state_of(recency_days=45.0, purchase_count_30d=0.0,
                        spend_slope_30d=-1.0, app_open_30d=0.0) == "Lapse Risk"

    def test_lapse_risk_requires_non_positive_slope(self):
        assert state_of(recency_days=45.0, purchase_count_30d=0.0,
                        spend_slope_30d=1.0, app_open_30d=0.0) != "Lapse Risk"

    def test_lapse_risk_requires_zero_purchases(self):
        assert state_of(recency_days=45.0, purchase_count_30d=1.0,
                        spend_slope_30d=-1.0, app_open_30d=0.0) != "Lapse Risk"

    # P4 — Momentum Builder: slope > 5 AND freq >= 2 AND recency < 30 AND div > 0.3
    def test_momentum_builder(self):
        assert state_of(spend_slope_30d=7.5, purchase_count_30d=3.0,
                        recency_days=5.0,
                        category_diversity_90d=0.45) == "Momentum Builder"

    def test_momentum_builder_slope_boundary_is_strict(self):
        """slope > 5.0, not >= 5.0. pipeline.py had this wrong."""
        assert state_of(spend_slope_30d=5.0, purchase_count_30d=3.0,
                        recency_days=5.0,
                        category_diversity_90d=0.45) != "Momentum Builder"

    def test_momentum_builder_requires_two_purchases(self):
        assert state_of(spend_slope_30d=7.5, purchase_count_30d=1.0,
                        recency_days=5.0,
                        category_diversity_90d=0.45) != "Momentum Builder"

    # P5 — Brand Advocate: engagement OR-clause AND tier >= 2 AND div > 0.4
    def test_brand_advocate_via_email(self):
        assert state_of(email_open_30d=2.0, tier_ordinal=2.0,
                        category_diversity_90d=0.65) == "Brand Advocate"

    def test_brand_advocate_via_single_referral(self):
        assert state_of(referral_sent_30d=1.0, tier_ordinal=3.0,
                        category_diversity_90d=0.65) == "Brand Advocate"

    def test_brand_advocate_requires_tier_two(self):
        assert state_of(email_open_30d=2.0, tier_ordinal=1.0,
                        category_diversity_90d=0.65) != "Brand Advocate"

    def test_brand_advocate_diversity_threshold_is_point_four(self):
        assert state_of(email_open_30d=2.0, tier_ordinal=2.0,
                        category_diversity_90d=0.40) != "Brand Advocate"

    # P6 — Redemption Hunter: redemption_rate > 0.30 AND freq <= 1
    def test_redemption_hunter(self):
        assert state_of(redemption_rate=0.5,
                        purchase_count_30d=1.0) == "Redemption Hunter"

    def test_redemption_hunter_boundary_is_strict(self):
        assert state_of(redemption_rate=0.30,
                        purchase_count_30d=1.0) != "Redemption Hunter"

    def test_redemption_hunter_requires_low_frequency(self):
        assert state_of(redemption_rate=0.5,
                        purchase_count_30d=2.0) != "Redemption Hunter"

    # P7 — Value Maximizer: diversity > 0.50 AND redemption_rate > 0.10
    def test_value_maximizer(self):
        assert state_of(category_diversity_90d=0.6, redemption_rate=0.2,
                        purchase_count_30d=2.0) == "Value Maximizer"

    def test_value_maximizer_redeem_threshold_is_point_one_not_point_two_one(self):
        """
        The documented threshold is 0.10. pipeline.py used redeem_min * 0.7
        = 0.21, which silently reclassified members between train and serve.
        """
        assert state_of(category_diversity_90d=0.6, redemption_rate=0.15,
                        purchase_count_30d=2.0) == "Value Maximizer"

    def test_value_maximizer_diversity_boundary_is_strict(self):
        assert state_of(category_diversity_90d=0.50, redemption_rate=0.2,
                        purchase_count_30d=2.0) != "Value Maximizer"

    # P8 — Silent Accumulator: freq >= 1 AND zero app/email/push
    def test_silent_accumulator(self):
        assert state_of(purchase_count_30d=3.0, app_open_30d=0.0,
                        email_open_30d=0.0,
                        push_open_30d=0.0) == "Silent Accumulator"

    def test_any_digital_touch_disqualifies_silent_accumulator(self):
        assert state_of(purchase_count_30d=3.0, app_open_30d=1.0,
                        email_open_30d=0.0,
                        push_open_30d=0.0) != "Silent Accumulator"

    # P9 — Plateau Cruiser: slope in [-2, +5] AND freq >= 1
    def test_plateau_cruiser(self):
        assert state_of(spend_slope_30d=1.0,
                        purchase_count_30d=2.0) == "Plateau Cruiser"

    def test_plateau_cruiser_includes_slope_boundaries(self):
        assert state_of(spend_slope_30d=-2.0,
                        purchase_count_30d=2.0) == "Plateau Cruiser"
        assert state_of(spend_slope_30d=5.0,
                        purchase_count_30d=2.0) == "Plateau Cruiser"

    def test_plateau_cruiser_has_no_recency_clause(self):
        """
        pipeline.py added `recency <= 30`, which training never applied. A
        member 45 days out with a flat slope is still a Plateau Cruiser --
        Lapse Risk does not claim them because they purchased in the window.
        """
        # app_open_30d stays non-zero so Silent Accumulator (P8) does not claim
        # them first -- this test is about the absence of a recency clause.
        assert state_of(spend_slope_30d=1.0, purchase_count_30d=2.0,
                        recency_days=45.0) == "Plateau Cruiser"

    def test_plateau_cruiser_slope_outside_window(self):
        assert state_of(spend_slope_30d=-3.0,
                        purchase_count_30d=2.0) != "Plateau Cruiser"


# ── Priority ordering ────────────────────────────────────────────────────────

class TestPriorityOrder:

    def test_lapse_risk_outranks_silent_accumulator(self):
        # Zero digital engagement, but no purchases and lapsed -> Lapse Risk.
        assert state_of(recency_days=45.0, purchase_count_30d=0.0,
                        spend_slope_30d=-1.0, app_open_30d=0.0) == "Lapse Risk"

    def test_brand_advocate_outranks_value_maximizer(self):
        assert state_of(email_open_30d=2.0, tier_ordinal=2.0,
                        category_diversity_90d=0.65, redemption_rate=0.2,
                        purchase_count_30d=2.0) == "Brand Advocate"

    def test_redemption_hunter_outranks_value_maximizer(self):
        assert state_of(redemption_rate=0.5, purchase_count_30d=1.0,
                        category_diversity_90d=0.65) == "Redemption Hunter"

    def test_conditions_arity_matches_labels(self):
        conds = build_conditions(member())
        assert len(conds) == len(STATE_ORDER) - 1


# ── Encoding ─────────────────────────────────────────────────────────────────

class TestStateEncoding:

    def test_priority_encoding_is_one_indexed_and_ordered(self):
        assert STATE_IDS["New & Uncertain"] == 1
        assert STATE_IDS["Program Skeptic"] == 10
        assert sorted(STATE_IDS.values()) == list(range(1, 11))

    def test_encoding_covers_every_state(self):
        assert set(STATE_IDS) == VALID_STATES
        assert len(STATE_IDS) == 10

    def test_state_ids_from_names_roundtrip(self):
        names = np.array(STATE_ORDER)
        ids   = state_ids_from_names(names)
        assert list(ids) == [STATE_IDS[n] for n in STATE_ORDER]

    def test_unknown_state_name_raises(self):
        with pytest.raises(ValueError, match="Unknown state"):
            state_ids_from_names(np.array(["Not A State"]))

    def test_encoding_matches_committed_training_labels(self):
        """
        The state_id column in states/*.parquet is what y_curr was built from.
        If STATE_IDS drifts from it, the model is served a scrambled feature.
        """
        state_dir = ROOT / "states"
        paths = sorted(state_dir.glob("lifecycle_states_*.parquet")) if state_dir.exists() else []
        if not paths:
            pytest.skip("No state files — run Phase 7 first")

        for path in paths:
            df = pd.read_parquet(path, columns=["state_name", "state_id"],
                                 engine="pyarrow")[["state_name", "state_id"]].drop_duplicates()
            for name, sid in df.itertuples(index=False):
                assert STATE_IDS[name] == sid, (
                    f"{path.name}: '{name}' is {sid} on disk but "
                    f"{STATE_IDS[name]} in state_rules.STATE_IDS"
                )
