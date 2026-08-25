"""
src/07_lifecycle_states.py
══════════════════════════════════════════════════════════════════════════════
PHASE 7 — Lifecycle State Classification Engine
TBIE Pipeline | Kobie × PES University Hackathon

Two-layer model:
  Layer 1 (Phase 6) Structural Segments  — durable behavioral archetypes
  Layer 2 (Phase 7) Lifecycle States     — current behavioral posture (transient)

Key engineering decisions:
  - numpy.select() vectorization: 0 Python loops over rows
    (df.apply on 6M rows = 45 min; numpy.select = <2s)
  - rule_fired: human-readable string of actual values that triggered the rule
  - Priority order: first matching rule wins, catch-all = Program Skeptic
  - Outputs merged with Phase 6 segment assignments for cross-analysis

Files written:
  states/lifecycle_states_YYYY_MM_DD.parquet  ×12
  outputs/state_definitions.json              (State Passport — Action 5)
  outputs/segment_state_cross_table.csv       (Action 4)
  outputs/state_transition_matrix.csv         (Markov chain)
  validation/phase7_summary.txt

Python: c:\\tbie_venv\\Scripts\\python.exe
Run from TBIE_CODE root: python src/07_lifecycle_states.py
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.state_rules import (
    STATE_IDS,
    build_conditions,
    state_ids_from_names,
)
from utils.state_rules import (
    classify_states as canonical_classify_states,
)

PIPE_START = time.time()
def elapsed():
    return f"{time.time() - PIPE_START:.1f}s"

print("=" * 70)
print("TBIE — PHASE 7: LIFECYCLE STATE CLASSIFICATION ENGINE")
print("=" * 70)

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
SEGMENTS_DIR = ROOT / "segments"
STATES_DIR   = ROOT / "states"
OUTPUTS_DIR  = ROOT / "outputs"
VAL_DIR      = ROOT / "validation"

for d in [STATES_DIR, OUTPUTS_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATES = pd.date_range("2025-01-01", "2025-12-01", freq="MS")

# STEP 7.1 — SEGMENT ID  NAME MAP  (from Phase 6 segment_definitions.json)
print(f"\n[{elapsed()}] STEP 7.1 — Loading Phase 6 segment definitions...")

with open(SEGMENTS_DIR / "segment_definitions.json", encoding="utf-8") as f:
    seg_defs = json.load(f)

SEG_ID_TO_NAME = {
    int(k): v["name"]
    for k, v in seg_defs.get("segments", {}).items()
}
print(f"  Segment map: {SEG_ID_TO_NAME}")

# STEP 7.2 — STATE PASSPORT  (Action 5 — embedded here, written to JSON)
print(f"\n[{elapsed()}] STEP 7.2 — Writing State Passport (state_definitions.json)...")

STATE_PASSPORT = {
    # NOTE: an "Unactivated" entry (state_id 0) used to sit here. No condition
    # in the cascade ever produced it, it was excluded from the exported
    # state_definitions.json, and the brief specifies exactly ten states.
    # Members who never purchased carry recency_days == 999 and fall through to
    # Lapse Risk or Program Skeptic, which is the documented behaviour.
    "New & Uncertain": {
        "state_id": 1,
        "meaning": (
            "Member is newly enrolled or has insufficient transaction history to reliably "
            "classify. Too few data points to determine behavioral archetype."
        ),
        "key_signals": ["tenure_days < 90"],
        "thresholds": {"tenure_days_max": 90},
        "recommended_action": "Send onboarding journey — guide first meaningful purchase",
        "urgency": "MEDIUM",
        "window_days": 90,
    },
    "Win-Back Target": {
        "state_id": 2,
        "meaning": (
            "Member lapsed 60+ days ago but is showing early re-engagement signals — "
            "they opened an email, visited the app, or made a small purchase. "
            "The window for recovery is now."
        ),
        "key_signals": ["recency_days > 60", "email_open_30d > 0 OR app_open_30d > 0 OR push_open_30d > 0"],
        "thresholds": {"recency_days_min": 60},
        "recommended_action": "Send personalised win-back offer referencing last purchase category",
        "urgency": "HIGH",
        "window_days": 30,
    },
    "Lapse Risk": {
        "state_id": 3,
        "meaning": (
            "Member is still active but engagement is clearly declining — fewer transactions, "
            "growing gaps between visits. Without intervention within 30 days, they will fully lapse."
        ),
        "key_signals": ["recency_days > 30", "purchase_count_30d == 0", "spend_slope_30d <= 0"],
        "thresholds": {"recency_days_min": 30, "purchase_count_30d_max": 0, "spend_slope_max": 0.0},
        "recommended_action": "Send urgency-framed win-back offer — highlight expiring points balance",
        "urgency": "HIGH",
        "window_days": 30,
    },
    "Momentum Builder": {
        "state_id": 4,
        "meaning": (
            "Member is on an upward trajectory — purchase frequency and spend are actively "
            "increasing. Most likely to move up a tier in the next 60 days."
        ),
        "key_signals": [
            "spend_slope_30d > 5.0",
            "purchase_count_30d >= 2",
            "recency_days < 30",
            "category_diversity_90d > 0.3"
        ],
        "thresholds": {
            "spend_slope_min": 5.0,
            "purchase_count_30d_min": 2,
            "recency_days_max": 30,
            "diversity_min": 0.3
        },
        "recommended_action": "Send tier upgrade nudge — show proximity to next tier",
        "urgency": "LOW",
        "window_days": 30,
    },
    "Brand Advocate": {
        "state_id": 5,
        "meaning": (
            "Member is deeply engaged across every channel — they open emails, use the app, "
            "hold a high loyalty tier, and shop across multiple categories."
        ),
        "key_signals": [
            "email_open_30d >= 2 OR app_open_30d >= 3 OR social OR referral OR survey",
            "tier_ordinal >= 2",
            "category_diversity_90d > 0.4",
        ],
        "thresholds": {
            "email_open_30d_min": 2, "app_open_30d_min": 3,
            "tier_ordinal_min": 2, "diversity_min": 0.4,
        },
        "recommended_action": "Send referral incentive — double points for friend referrals",
        "urgency": "LOW",
        "window_days": 30,
    },
    "Redemption Hunter": {
        "state_id": 6,
        "meaning": (
            "Member is primarily motivated by point redemption. They redeem a high fraction "
            "of earned points, often spiking activity around promotional events."
        ),
        "key_signals": ["redemption_rate > 0.30", "purchase_count_30d <= 1"],
        "thresholds": {"redemption_rate_min": 0.30, "purchase_count_30d_max": 1},
        "recommended_action": "Send promo expiry alert — create urgency around point balance",
        "urgency": "MEDIUM",
        "window_days": 30,
    },
    "Value Maximizer": {
        "state_id": 7,
        "meaning": (
            "Member balances high redemption with broad engagement. "
            "They shop across categories to optimize earn rates and utilize rewards effectively."
        ),
        "key_signals": ["category_diversity_90d > 0.50", "redemption_rate > 0.10"],
        "thresholds": {"diversity_min": 0.50, "redemption_rate_min": 0.10},
        "recommended_action": "Highlight partner network — show new categories to earn points",
        "urgency": "LOW",
        "window_days": 90,
    },
    "Silent Accumulator": {
        "state_id": 8,
        "meaning": (
            "Member is purchasing regularly but completely invisible in digital channels — "
            "no app opens, no email opens, no push notifications clicked. "
            "They earn points but never engage with loyalty communications."
        ),
        "key_signals": [
            "purchase_count_30d >= 1",
            "app_open_30d == 0",
            "email_open_30d == 0",
            "push_open_30d == 0",
        ],
        "thresholds": {
            "purchase_count_30d_min": 1,
            "app_open_30d_max": 0,
            "email_open_30d_max": 0,
            "push_open_30d_max": 0,
        },
        "recommended_action": "Send app download incentive — bonus points for first in-app purchase",
        "urgency": "MEDIUM",
        "window_days": 30,
    },
    "Plateau Cruiser": {
        "state_id": 9,
        "meaning": (
            "Member is stable and habitual — consistent purchasing with no growth or decline "
            "signal. Comfortable where they are and unlikely to move up a tier without a nudge."
        ),
        "key_signals": ["-2.0 <= spend_slope_30d <= 5.0", "purchase_count_30d >= 1"],
        "thresholds": {
            "spend_slope_min": -2.0,
            "spend_slope_max": 5.0,
            "purchase_count_30d_min": 1,
        },
        "recommended_action": "Send personalised offer based on most-purchased category",
        "urgency": "LOW",
        "window_days": 30,
    },
    "Program Skeptic": {
        "state_id": 10,
        "meaning": (
            "Member is enrolled but shows minimal engagement with the programme — "
            "low app use, ignores communications, and minimal profile completion. "
            "The loyalty programme is not influencing their behaviour."
        ),
        "key_signals": ["catch-all — no other state rule fired"],
        "thresholds": {},
        "recommended_action": "Send reactivation email — focus on programme value proposition",
        "urgency": "MEDIUM",
        "window_days": 30,
    },
}

state_def_path = OUTPUTS_DIR / "state_definitions.json"
with open(state_def_path, "w", encoding="utf-8") as f:
    json.dump(STATE_PASSPORT, f, indent=2, ensure_ascii=False)
print(f"  state_definitions.json written ({state_def_path.stat().st_size / 1024:.1f} KB)")

STATE_ID_MAP = {name: d["state_id"] for name, d in STATE_PASSPORT.items()}

# The passport and the canonical rule module must agree on the encoding, or the
# state_id column written here will not match the state_map saved in the model
# bundle — which is exactly how y_curr got scrambled between train and serve.
assert STATE_ID_MAP == STATE_IDS, (
    f"State id encoding mismatch.\n"
    f"  STATE_PASSPORT: {dict(sorted(STATE_ID_MAP.items(), key=lambda x: x[1]))}\n"
    f"  state_rules   : {dict(sorted(STATE_IDS.items(), key=lambda x: x[1]))}"
)

# STEP 7.3 — classify_states_vectorized()
def classify_states_vectorized(df: pd.DataFrame):
    """
    Assigns each member to exactly one of 10 loyalty lifecycle states.

    Engineering notes:
      - numpy.select() applies all conditions in one C-level pass (no Python loops)
      - rule_fired built with vectorized pandas string ops on exclusive masks
      - All column accesses use fillna(0) to guard against NaN comparison failures
      - DataFrame must be reset_index(drop=True) before calling (caller's responsibility)

    Priority (first match wins):
      1  New & Uncertain    tenure_days < 90
      2  Win-Back Target    recency > 60 AND re-engagement signal (email/app/push)
      3  Lapse Risk         recency > 30 AND spend_slope <= 0
      4  Momentum Builder   slope > 5.0 AND freq30 >= 2 AND recency < 30 AND diversity > 0.3
      5  Brand Advocate     tier >= 2 AND high engagement (email/app/social/ref/survey) AND diversity > 0.4
      6  Redemption Hunter  redemption_rate > 0.30 AND freq30 <= 1
      7  Value Maximizer    diversity > 0.50 AND redemption_rate > 0.10
      8  Silent Accumulator freq30 >= 1 AND app=0 AND email=0 AND push=0
      9  Plateau Cruiser    slope in [-2, +5] AND freq30 >= 1
     10  Program Skeptic    catch-all

    Returns
    -------
    state_names : np.ndarray[str]    — state label per row
    state_ids   : np.ndarray[int]    — state integer id per row
    rule_fired  : pd.Series[str]     — human-readable rule string per row
    """
    idx = df.index

    # Rule logic lives in src/utils/state_rules.py — the single source of truth
    # shared with pipeline.py. Do not reimplement the conditions here; the two
    # copies drifted last time and silently changed the label distribution
    # between training and inference.
    conditions = build_conditions(df)
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = conditions

    state_names = canonical_classify_states(df)
    state_ids   = state_ids_from_names(state_names)

    # rule_fired strings
    # Exclusive masks — each fires only if no higher-priority condition fired
    m0 = c0
    m1 = ~m0 & c1
    m2 = ~m0 & ~m1 & c2
    m3 = ~m0 & ~m1 & ~m2 & c3
    m4 = ~m0 & ~m1 & ~m2 & ~m3 & c4
    m5 = ~m0 & ~m1 & ~m2 & ~m3 & ~m4 & c5
    m6 = ~m0 & ~m1 & ~m2 & ~m3 & ~m4 & ~m5 & c6
    m7 = ~m0 & ~m1 & ~m2 & ~m3 & ~m4 & ~m5 & ~m6 & c7
    m8 = ~m0 & ~m1 & ~m2 & ~m3 & ~m4 & ~m5 & ~m6 & ~m7 & c8

    rule_fired = pd.Series("No strong behavioural signal detected this month; the member is enrolled but not actively engaging with the programme.", index=idx, dtype=object)

    def fmtcol(name, decimals=None, fill=0.0):
        """Return formatted pandas Series for a column (full-length, caller subsets)."""
        s = df[name].fillna(fill) if name in df.columns else pd.Series(fill, index=idx)
        return s.round(decimals).astype(str) if decimals is not None else s.astype(int).astype(str)

    # New & Uncertain
    if m0.any():
        rule_fired[m0] = (
            "Member enrolled " + fmtcol("tenure_days")[m0]
            + " days ago — too early to establish a reliable behavioural pattern. Onboarding journey applies."
        )

    # Win-Back Target
    if m1.any():
        rule_fired[m1] = (
            "Member has been inactive for " + fmtcol("recency_days", 1)[m1]
            + " days (60+ day lapse) but showed a digital re-engagement signal this month — email open, app visit, or push click detected."
        )

    # Lapse Risk
    if m2.any():
        rule_fired[m2] = (
            "Member has not purchased in " + fmtcol("recency_days", 1)[m2]
            + " days and spend trend is declining (slope: " + fmtcol("spend_slope_30d", 2)[m2]
            + "). Without intervention within 30 days, full lapse is likely."
        )

    # Momentum Builder
    if m3.any():
        rule_fired[m3] = (
            "Member is on an upward trajectory — spend rising at +" + fmtcol("spend_slope_30d", 1)[m3]
            + " per week, with " + fmtcol("purchase_count_30d")[m3]
            + " purchases in the last 30 days across diverse categories. Most likely to reach the next tier within 60 days."
        )

    # Brand Advocate
    if m4.any():
        rule_fired[m4] = (
            "Member is deeply engaged across multiple programme touchpoints — holds tier level "
            + fmtcol("tier_ordinal")[m4]
            + " (Gold or Platinum) and shops across " + fmtcol("category_diversity_90d", 2)[m4]
            + " normalised category breadth. High email and app activity confirm programme affinity."
        )

    # Redemption Hunter
    if m5.any():
        rule_fired[m5] = (
            "Member redeems " + fmtcol("redemption_rate", 1)[m5]
            + "% of earned points and made only " + fmtcol("purchase_count_30d")[m5]
            + " purchase(s) this month — behaviour is driven by redemption events rather than regular spending."
        )

    # Value Maximizer
    if m6.any():
        rule_fired[m6] = (
            "Member shops strategically across categories (diversity: " + fmtcol("category_diversity_90d", 2)[m6]
            + ") and redeems " + fmtcol("redemption_rate", 1)[m6]
            + "% of earned points — optimising both earn rate and reward value."
        )

    # Silent Accumulator
    if m7.any():
        rule_fired[m7] = (
            "Member made " + fmtcol("purchase_count_30d")[m7]
            + " purchase(s) this month but opened zero emails, zero app sessions, and zero push notifications. "
            "They earn points consistently but ignore all programme communications."
        )

    # Plateau Cruiser
    if m8.any():
        rule_fired[m8] = (
            "Member is in a stable routine — " + fmtcol("purchase_count_30d")[m8]
            + " purchase(s) this month with a flat spend trend (slope: " + fmtcol("spend_slope_30d", 1)[m8]
            + "). No growth or decline detected; behaviour has been consistent for multiple months."
        )

    return state_names, state_ids, rule_fired


# STEP 7.4 — APPLY TO ALL 12 MONTHS
print(f"\n[{elapsed()}] STEP 7.4 — Classifying lifecycle states for all 12 months...")
print("  Vectorized numpy.select() — no df.apply()\n")

monthly_summary = []

for obs_date in SNAPSHOT_DATES:
    t_s       = time.time()
    date_str  = obs_date.strftime("%Y_%m_%d")
    month_str = obs_date.strftime("%Y_%m")

    feat_path = FEATURES_DIR / f"features_{date_str}.parquet"
    if not feat_path.exists():
        print(f"  ️  {date_str}: feature file not found — skipping")
        continue

    # Load + reset index (required for vectorized string mask alignment)
    feat = pd.read_parquet(feat_path, engine="pyarrow").reset_index(drop=True)

    # Load segment assignments for this month
    seg_path = SEGMENTS_DIR / f"behavioral_segments_{date_str}.parquet"
    if seg_path.exists():
        segs = pd.read_parquet(seg_path, engine="pyarrow")[["member_id", "segment_id"]]
        segs["segment_name"] = segs["segment_id"].map(SEG_ID_TO_NAME).fillna("Unknown")
    else:
        segs = pd.DataFrame({
            "member_id":    feat["member_id"],
            "segment_id":   -1,
            "segment_name": "Unknown",
        })

    # Classify (vectorized)
    state_names, state_ids, rule_fired = classify_states_vectorized(feat)

    # Build output row
    out = pd.DataFrame({
        "member_id":        feat["member_id"],
        "observation_date": obs_date,
        "observation_month": month_str,
        "state_name":       state_names,
        "state_id":         state_ids,
        "rule_fired":       rule_fired.values,
    })

    # Merge Phase 6 segment labels
    out = out.merge(segs, on="member_id", how="left")
    out["segment_id"]   = out["segment_id"].fillna(-1).astype(int)
    out["segment_name"] = out["segment_name"].fillna("Unknown")

    # Reorder columns to match spec output format
    out = out[[
        "member_id", "segment_id", "segment_name",
        "state_id", "state_name", "rule_fired",
        "observation_date", "observation_month",
    ]]

    # Write parquet
    out_path = STATES_DIR / f"lifecycle_states_{date_str}.parquet"
    out.to_parquet(out_path, engine="pyarrow", index=False)

    state_dist = out["state_name"].value_counts().to_dict()
    t_e        = time.time() - t_s

    print(
        f"  {obs_date.strftime('%Y-%m')}: {len(out):,} members | {t_e:.1f}s | "
        f"LapseRisk={state_dist.get('Lapse Risk', 0):,}  "
        f"SilentAcc={state_dist.get('Silent Accumulator', 0):,}  "
        f"Momentum={state_dist.get('Momentum Builder', 0):,}  "
        f"Skeptic={state_dist.get('Program Skeptic', 0):,}"
    )

    monthly_summary.append({
        "month":     month_str,
        "n_members": len(out),
        **{name: state_dist.get(name, 0) for name in STATE_PASSPORT.keys()},
    })

print(f"\n   All months classified. Written to: {STATES_DIR}")

# STEP 7.5 — SEGMENT × STATE CROSS-TABLE (Action 4 — December 2025)
print(f"\n[{elapsed()}] STEP 7.5 — Building Segment × State cross-table (December 2025)...")

dec_path = STATES_DIR / "lifecycle_states_2025_12_01.parquet"
if dec_path.exists():
    dec = pd.read_parquet(dec_path, engine="pyarrow")
    dec = dec[dec["segment_name"] != "Unknown"]   # exclude unmerged rows

    cross = pd.crosstab(
        dec["segment_name"],
        dec["state_name"],
        margins=True,
        margins_name="TOTAL",
    )
    cross_path = OUTPUTS_DIR / "segment_state_cross_table.csv"
    cross.to_csv(cross_path)

    print("\n  Segment × State Cross-Table (December 2025):")
    print(cross.to_string())
    print(f"\n   segment_state_cross_table.csv written ({cross_path.stat().st_size/1024:.1f} KB)")
    print("\n  KEY INSIGHT: A 'Silent Accumulator' segment member can be in Lapse Risk,")
    print("  Momentum Builder, or Program Skeptic state — Segment ≠ State.")
else:
    print("  ️  December state parquet not found — skipping cross-table")

# STEP 7.6 — STATE TRANSITION MATRIX (Month-over-Month Markov Chain)
print(f"\n[{elapsed()}] STEP 7.6 — Building state transition matrix (Markov chain)...")

all_months = []
for obs_date in SNAPSHOT_DATES:
    date_str = obs_date.strftime("%Y_%m_%d")
    p = STATES_DIR / f"lifecycle_states_{date_str}.parquet"
    if p.exists():
        df_m = pd.read_parquet(p, engine="pyarrow")[["member_id", "state_name", "observation_date"]]
        df_m["observation_date"] = pd.to_datetime(df_m["observation_date"])
        all_months.append(df_m)

if len(all_months) >= 2:
    all_states = pd.concat(all_months, ignore_index=True)
    all_states = all_states.sort_values(["member_id", "observation_date"]).reset_index(drop=True)

    # Use groupby().shift() — vectorized, no cross-join memory spike
    all_states["state_next"] = all_states.groupby("member_id")["state_name"].shift(-1)
    all_states["date_next"]  = all_states.groupby("member_id")["observation_date"].shift(-1)

    pairs = all_states.dropna(subset=["state_next"]).copy()
    day_diff = (pd.to_datetime(pairs["date_next"]) - pd.to_datetime(pairs["observation_date"])).dt.days
    pairs = pairs[day_diff.between(28, 35)]   # consecutive month pairs only

    # Transition count matrix
    trans_counts = pd.crosstab(pairs["state_name"], pairs["state_next"])

    # Normalize to probability (row-wise)
    trans_probs = trans_counts.div(trans_counts.sum(axis=1), axis=0).round(4)

    trans_path = OUTPUTS_DIR / "state_transition_matrix.csv"
    trans_probs.to_csv(trans_path)
    print(f"   state_transition_matrix.csv written  ({trans_path.stat().st_size/1024:.1f} KB)")
    print(f"  Matrix: {trans_probs.shape[0]} states × {trans_probs.shape[1]} states")

    print("\n  State Retention Rates (probability of staying in same state):")
    for state in sorted(trans_probs.index):
        if state in trans_probs.columns:
            ret = trans_probs.loc[state, state]
            bar = "█" * int(ret * 20)
            print(f"    {state:<25}: {ret:.3f} ({ret*100:4.1f}%)  {bar}")

    print("\n  Top 5 most likely transitions (from  to):")
    # Melt to long, exclude self-transitions, sort by probability
    melted = trans_probs.reset_index().melt(
        id_vars=["state_name"], var_name="state_next", value_name="prob"
    )
    melted = melted[melted["state_name"] != melted["state_next"]]
    top5 = melted.nlargest(5, "prob")
    for _, row in top5.iterrows():
        print(f"    {row['state_name']:<25}  {row['state_next']:<25}: {row['prob']:.3f}")
else:
    print("  ️  Not enough monthly files for transition matrix (need ≥2)")

# STEP 7.7 — SAMPLE OUTPUT (5 members — matches spec output format)
print(f"\n[{elapsed()}] STEP 7.7 — Sample output (5 members from December)...")

if dec_path.exists():
    dec_sample = pd.read_parquet(dec_path, engine="pyarrow")
    # Show one representative example per state (if exists)
    states_to_show = ["Lapse Risk", "Silent Accumulator", "Momentum Builder", "Brand Advocate", "New & Uncertain"]
    print()
    for state in states_to_show:
        row = dec_sample[dec_sample["state_name"] == state].head(1)
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print("  {")
        print(f"    'member_id':        '{r['member_id']}',")
        print(f"    'segment_name':     '{r['segment_name']}',")
        print(f"    'state_name':       '{r['state_name']}',")
        print(f"    'rule_fired':       '{r['rule_fired']}',")
        print(f"    'observation_month': '{r['observation_month']}'")
        print("  }")

# STEP 7.8 — PHASE 7 SUMMARY
print(f"\n[{elapsed()}] STEP 7.8 — Writing Phase 7 summary...")

dec_row = monthly_summary[-1] if monthly_summary else {}
n_members = dec_row.get("n_members", 0)

summary_lines = [
    "PHASE 7 SUMMARY — LIFECYCLE STATE CLASSIFICATION",
    "=" * 60,
    "Algorithm:        numpy.select() vectorized (no df.apply)",
    f"States defined:   {len(STATE_PASSPORT)}",
    f"Months processed: {len(monthly_summary)}",
    "",
    "December 2025 State Distribution:",
]
for name in STATE_PASSPORT.keys():
    cnt = dec_row.get(name, 0)
    pct = cnt / n_members * 100 if n_members > 0 else 0
    urgency = STATE_PASSPORT[name]["urgency"]
    summary_lines.append(f"  {name:<25}: {cnt:>7,}  ({pct:>5.1f}%)  [{urgency}]")

summary_lines += [
    "",
    "Files written:",
    f"  states/lifecycle_states_*.parquet      x{len(monthly_summary)}",
    "  outputs/state_definitions.json",
    "  outputs/segment_state_cross_table.csv",
    "  outputs/state_transition_matrix.csv",
    "",
    f"Total elapsed: {elapsed()}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

(VAL_DIR / "phase7_summary.txt").write_text(summary_text, encoding="utf-8")

print(f"\n{'=' * 70}")
print("PHASE 7 COMPLETE")
print(f"{'=' * 70}")
