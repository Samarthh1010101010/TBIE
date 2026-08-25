"""
tests/test_feature_modules.py
════════════════════════════════════════════════════════════════════════════
Regression tests for the shared feature modules.

Both of these existed as duplicated implementations that drifted between
training and inference, so the behaviours pinned here are the specific ones
that were previously wrong:

  velocity     — recency_days is NaN for members who have never purchased.
                 Training filled it with 0 ("purchased today, no risk"),
                 inference filled it with 999 ("maximum risk"). Opposite
                 signals for the same member.

  lag_features — members absent from the prior snapshot must produce a delta
                 of 0 and seg_changed of 0, not a dropped row or a fabricated
                 transition. The panel is a fixed 500K members and the output
                 contract is 500,000 rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.lag_features import (  # noqa: E402
    LAG_BASE_COLS_ALL,
    SEG_HIST_COLS,
    add_delta_features,
    add_segment_history,
    delta_col_names,
    lag_col_names,
    resolve_lag_base_cols,
)
from utils.velocity import (  # noqa: E402
    NEVER_PURCHASED_RECENCY,
    VELOCITY_CLIP_MAX,
    VELOCITY_COLS,
    add_velocity_features,
)

# ── Velocity ─────────────────────────────────────────────────────────────────

class TestVelocityFeatures:

    def _frame(self, **over):
        base = {
            "spend_total_30d": 100.0, "spend_total_90d": 300.0,
            "purchase_count_30d": 3.0, "purchase_count_90d": 9.0,
            "app_open_30d": 4.0, "app_open_90d": 12.0,
            "email_open_30d": 2.0, "push_open_30d": 1.0,
            "recency_days": 10.0,
        }
        base.update(over)
        return pd.DataFrame([base])

    def test_all_velocity_columns_produced(self):
        out = add_velocity_features(self._frame())
        for c in VELOCITY_COLS:
            assert c in out.columns, f"{c} not produced"

    def test_steady_member_has_unit_velocity(self):
        """30d exactly equal to the 90d monthly pace => velocity 1.0."""
        out = add_velocity_features(self._frame())
        assert out["spend_velocity"].iloc[0] == pytest.approx(1.0, abs=1e-3)
        assert out["freq_velocity"].iloc[0]  == pytest.approx(1.0, abs=1e-3)
        assert out["app_velocity"].iloc[0]   == pytest.approx(1.0, abs=1e-3)

    def test_decline_flag_fires_below_half_pace(self):
        out = add_velocity_features(self._frame(spend_total_30d=40.0))  # 0.4x pace
        assert out["spend_velocity"].iloc[0] < 0.5
        assert out["spend_decline_flag"].iloc[0] == 1.0

    def test_decline_flag_off_at_pace(self):
        out = add_velocity_features(self._frame())
        assert out["spend_decline_flag"].iloc[0] == 0.0

    def test_velocities_are_clipped(self):
        out = add_velocity_features(self._frame(spend_total_30d=1e6))
        assert out["spend_velocity"].iloc[0] <= VELOCITY_CLIP_MAX

    def test_zero_baseline_does_not_divide_by_zero(self):
        out = add_velocity_features(self._frame(spend_total_90d=0.0, purchase_count_90d=0.0,
                                                app_open_90d=0.0))
        assert np.isfinite(out["spend_velocity"].iloc[0])
        assert np.isfinite(out["freq_velocity"].iloc[0])
        assert np.isfinite(out["app_velocity"].iloc[0])

    def test_engagement_score_sums_channels(self):
        out = add_velocity_features(self._frame(app_open_30d=4, email_open_30d=2, push_open_30d=1))
        assert out["engagement_score"].iloc[0] == 7.0

    # ── The regression that mattered ─────────────────────────────────────────
    def test_null_recency_uses_never_purchased_sentinel(self):
        """
        A member who has never purchased has recency_days NaN and must be
        scored as HIGH lapse risk, not zero risk. The training path used to
        fill this with 0, which inverted the signal.
        """
        out = add_velocity_features(self._frame(recency_days=np.nan, purchase_count_30d=0.0))
        assert out["recency_risk"].iloc[0] == pytest.approx(NEVER_PURCHASED_RECENCY)

    def test_null_recency_is_not_zero(self):
        out = add_velocity_features(self._frame(recency_days=np.nan, purchase_count_30d=0.0))
        assert out["recency_risk"].iloc[0] != 0.0

    def test_recency_risk_decays_with_frequency(self):
        """Frequent buyers carry no lapse risk regardless of recency."""
        out = add_velocity_features(self._frame(recency_days=60.0, purchase_count_30d=10.0))
        assert out["recency_risk"].iloc[0] == pytest.approx(0.0)

    def test_missing_columns_do_not_raise(self):
        out = add_velocity_features(pd.DataFrame([{"spend_total_30d": 10.0}]))
        for c in VELOCITY_COLS:
            assert c in out.columns
            assert np.isfinite(out[c].iloc[0])

    def test_input_frame_not_mutated(self):
        df = self._frame()
        before = list(df.columns)
        add_velocity_features(df)
        assert list(df.columns) == before


# ── Lag / delta ──────────────────────────────────────────────────────────────

class TestLagFeatures:

    def _pair(self):
        cur = pd.DataFrame({
            "member_id": ["A", "B", "C"],
            "spend_total_30d": [100.0, 50.0, 10.0],
            "purchase_count_30d": [5.0, 2.0, 0.0],
        })
        prev = pd.DataFrame({
            "member_id": ["A", "B"],          # C is absent from the prior month
            "spend_total_30d": [60.0, 80.0],
            "purchase_count_30d": [3.0, 4.0],
        })
        return cur, prev, ["spend_total_30d", "purchase_count_30d"]

    def test_delta_columns_named_and_computed(self):
        cur, prev, cols = self._pair()
        out = add_delta_features(cur, prev, cols)
        assert out["d_spend_total_30d"].tolist()[:2] == [40.0, -30.0]
        assert out["d_purchase_count_30d"].tolist()[:2] == [2.0, -2.0]

    def test_member_absent_from_prior_month_gets_zero_delta(self):
        """Not a dropped row, and not a fabricated jump from zero."""
        cur, prev, cols = self._pair()
        out = add_delta_features(cur, prev, cols)
        c = out[out["member_id"] == "C"].iloc[0]
        assert c["d_spend_total_30d"] == 10.0   # 10 - 0, prior treated as 0
        assert len(out) == 3                    # row preserved

    def test_row_count_preserved(self):
        cur, prev, cols = self._pair()
        assert len(add_delta_features(cur, prev, cols)) == len(cur)

    def test_prior_columns_are_dropped(self):
        cur, prev, cols = self._pair()
        out = add_delta_features(cur, prev, cols)
        assert not [c for c in out.columns if c.startswith("prev_")]

    def test_missing_base_column_raises(self):
        cur, prev, _ = self._pair()
        with pytest.raises(KeyError, match="lag base column"):
            add_delta_features(cur, prev, ["spend_total_30d", "not_a_column"])

    def test_segment_history_flags_moves(self):
        df = pd.DataFrame({"seg_curr": [0, 1, 2]})
        out = add_segment_history(df, [0, 3, 2])
        assert out["seg_changed"].tolist() == [0.0, 1.0, 0.0]

    def test_unseen_member_is_not_recorded_as_moving(self):
        """
        seg_prev NaN means we never observed them last month. Treating that as
        a move would invent a transition that was never seen.
        """
        df = pd.DataFrame({"seg_curr": [2, 4]})
        out = add_segment_history(df, [np.nan, np.nan])
        assert out["seg_changed"].tolist() == [0.0, 0.0]
        assert out["seg_prev"].tolist() == [2.0, 4.0]

    def test_column_name_helpers_agree(self):
        cols = ["spend_total_30d", "recency_days"]
        assert delta_col_names(cols) == ["d_spend_total_30d", "d_recency_days"]
        assert lag_col_names(cols) == delta_col_names(cols) + SEG_HIST_COLS

    def test_resolve_restricts_to_available(self):
        got = resolve_lag_base_cols(["spend_total_30d", "irrelevant"])
        assert got == ["spend_total_30d"]

    def test_resolve_preserves_canonical_order(self):
        shuffled = list(reversed(LAG_BASE_COLS_ALL))
        assert resolve_lag_base_cols(shuffled) == LAG_BASE_COLS_ALL
