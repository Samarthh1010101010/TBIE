# TBIE — Temporal Behavioural Intelligence Engine
### Kobie × PES University Hackathon 2025

Predicts which loyalty segment each of 500,000 members will occupy 30 days from
now, and what behavioural state they are in today — so that retention spend goes
to the members whose behaviour is actually changing.

| | Macro F1 (Test: Nov→Dec) |
|---|---:|
| Majority-class baseline | 0.0589 |
| Persistence baseline (`seg_next = seg_curr`) | 0.5651 |
| **TBIE** | **0.8073**  95% CI [0.8035, 0.8100] |
| | **+0.2422 over persistence** |

Segment membership is strongly autocorrelated month over month, so predicting
that nobody moves already scores 0.565. The lift over that baseline — not the
raw F1 — is the number that means something.

**5 segments · 10 lifecycle states · 500,000 members · 18M transactions**
Strict walk-forward validation; every modelling choice made on validation, test
read exactly once.

### ▶ [Open the live dashboard](https://siddarth10reddy.github.io/TBIE/)

A static snapshot — real predictions, SHAP attributions and contact eligibility
for a stratified sample of scored members. No server needed. Run
`python -m uvicorn serving.api:app` for the live service over all 500,000.

Full benchmarks, ablations and the reproducing command for every number:
**[RESULTS.md](RESULTS.md)** · Intended use and limitations:
**[MODEL_CARD.md](MODEL_CARD.md)** · Method: **[METHODOLOGY.md](METHODOLOGY.md)** ·
What's still wrong: **[AUDIT.md](AUDIT.md)**

---

## What it does

| | |
|---|---|
| **Segment** | Which of 5 long-term behavioural archetypes a member belongs to |
| **State** | Which of 10 short-term lifecycle states they occupy today |
| **Transition** | Probability distribution over the segment they'll occupy in 30 days |
| **Explain** | Per-member SHAP attributions — what drove *this* prediction |
| **Target** | Expected-value-optimal contact list, not F1-optimal |
| **Monitor** | Drift detection against the frozen fit window, with a retrain trigger |
| **Suppress** | Consent, account status, fraud and frequency capping — who may actually be contacted |
| **Serve** | REST API + dashboard over the whole thing |

```bash
# Batch scoring
python pipeline.py --data_dir ./data/train/ --observation_date 2025-12-31 --output_dir ./outputs/

# Interactive API + dashboard
python -m uvicorn serving.api:app --port 8000   # then open http://localhost:8000
```

---

## Architecture

A two-layer inference pipeline. Layer 1 groups members into long-term behavioural archetypes; Layer 2 classifies each member's current short-term posture within that archetype.

| Layer | Component | Method | Output |
|---|---|---|---|
| Layer 1 | Segment Discovery | K-Means k=5, frozen centroids, 18 PCA components | 5 behavioural segments |
| Layer 2 | State Classification | 10-state priority cascade (vectorised `numpy.select`) | 1 of 10 Loyalty-Native States |
| Transition | Segment Prediction | XGBoost `multi:softprob`, 64 features | Probability distribution over segments at t+30d |

The 64 features are 40 behavioural + 6 within-snapshot velocity + 13
month-over-month deltas + `seg_prev`, `seg_changed`, `seg_curr`, `y_curr`,
`month_num`. Layer 2's output feeds the transition model as `y_curr`, so the
cascade has exactly one implementation
([src/utils/state_rules.py](src/utils/state_rules.py)) shared by training,
batch inference and the API.

---

## Beyond the model

### Probabilities you can act on

The pipeline emits `prob_S01`…`prob_S05`, and downstream targeting acts on
them, so they were tested rather than assumed (`src/09_calibration.py`).

| | Test (Nov→Dec) |
|---|---:|
| Mean Brier score | 0.0673 |
| Mean expected calibration error | **0.0184** |

Reliability on Growth Builder — predicted vs what actually happened:

| model says | actually happened | members |
|---:|---:|---:|
| 0.30 | 0.32 | 28,737 |
| 0.50 | 0.53 | 20,105 |
| 0.70 | 0.71 | 22,994 |
| 0.90 | 0.93 | 26,326 |

Isotonic recalibration was fitted on validation and **made test reliability
worse** (ECE 0.0184 → 0.0375), so the raw probabilities ship. A negative result,
recorded rather than discarded.

### Targeting on money, not on F1

`src/10_cost_thresholds.py` replaces the F1-optimal threshold with an
expected-value rule: contact when `P(adverse move) × value_at_risk × recovery`
clears the contact cost. Segment values come from observed 180-day spend.

| Policy | Contacted | Profit | ROI | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Contact everyone | 500,000 | −$980,416 | −0.39x | — | — |
| F1-optimal threshold | 91,209 | $613,227 | 1.34x | 31.5% | 76.9% |
| **Expected-value ranked** | 110,853 | **$771,716** | **1.39x** | 16.6% | 49.3% |

The EV policy has *worse* precision and recall and makes **$158K more**, because
a 90% chance of losing $20 is a worse use of a $5 touchpoint than a 15% chance
of losing $900. Classification metrics cannot see that difference.

> Currency figures are conditional on an assumed 10% base recovery rate. That
> base is now scaled per member by observed campaign responsiveness
> (p10 8.3% / median 10.2% / p90 11.0%), but no randomised holdout exists, so
> absolute causal lift remains unestablished. The script sweeps the assumption;
> break-even sits between 2% and 5%.

### Who may actually be contacted

Consent, account status, fraud and contact frequency are enforced before any
recommendation is made (`outputs/contact_eligibility.csv`).

| | Members | Share |
|---|---:|---:|
| **Targetable** | **319,271** | **63.9%** |
| Frequency-capped (≥6 contacts/30d) | 87,405 | 17.5% |
| No consent on any channel | 81,026 | 16.2% |
| Account closed | 9,898 | 2.0% |
| Fraud-flagged | 2,400 | 0.5% |

Before this layer existed, **222,259 members were being pointed at a channel they
had opted out of**, and ~20,000 closed or fraud-flagged accounts were being
targeted. Consent violations are now **0**.

### Explanations that explain

`supporting_evidence` used to echo the member's own inputs back
(`recency_days:38`). Exact TreeSHAP now reports direction and magnitude:

```
tier_ordinal=0 raises 0.784; tier_changes_count=0 raises 0.294;
months_since_last_tier_change=0 raises 0.133
```

### Drift monitoring

`src/12_drift_monitor.py` compares any month against the frozen fit window and
exits non-zero past threshold, so it can gate a scheduled retrain. Run against
the submission date it returns **RETRAIN_RECOMMENDED**:

| Feature | PSI | Dec 1 → Dec 31 |
|---|---:|---|
| `recency_days` | 5.62 | 6.07 → 9.08 |
| `purchase_count_7d` | 0.39 | 1.13 → 0.54 |
| `spend_total_7d` | 0.27 | 68.99 → 32.79 |

The 7-day windows halve because 31 December sits in the post-Christmas lull.
Outputs at that date are still valid, but they describe a population the frozen
centroids were not fitted on.

---

## Requirements

- Python 3.12 (Python 3.14+ breaks numpy/scikit-learn serialisation — do not use)
- RAM: 16 GB minimum (feature matrix ~4 GB; pipeline peak ~6 GB)
- CPU: 4+ cores recommended
- GPU: not required

---

## Setup

```bash
cd TBIE_CODE

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / Mac

pip install -r requirements.txt
```

Place the three provided data files in `data/train/`:

```
TBIE_CODE/
└── data/
    └── train/
        ├── members.parquet
        ├── transactions.parquet
        └── engagement_events.parquet
```

---

## Running the Pipeline

```bash
python pipeline.py \
  --data_dir ./data/train/ \
  --observation_date 2025-12-31 \
  --output_dir ./outputs/
```

The pipeline verifies its dependencies at startup and exits with a clear message if any are missing. It does **not** install anything — a scoring harness should not have its environment mutated as a side effect of running a prediction. Install with `pip install -r requirements.txt`, or use the provided `Dockerfile`.

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `--data_dir` | Yes | Directory containing the three raw parquet files |
| `--observation_date` | Yes | Snapshot date in YYYY-MM-DD format |
| `--output_dir` | Yes | Output directory (created if it does not exist) |
| `--k` | No (default: 5) | Number of segments — must match frozen segment_model.pkl |

### Running at a Different Date

```bash
python pipeline.py \
  --data_dir ./data/train/ \
  --observation_date 2026-01-31 \
  --output_dir ./outputs/jan2026/
```

The pipeline uses frozen model weights and never retrains at inference time. Any observation date within the dataset range produces valid, reproducible outputs.

### Runtime

Measured on 16 GB RAM, 8-core CPU, no GPU:

| Condition | Duration |
|---|---|
| Features pre-cached from prior run | 10 – 15 min |
| Full build from raw data (first run) | 20 – 25 min |

Actual submission run (2025-12-31): 340 seconds end-to-end.

> **Windows encoding note:** If you see encoding errors on print output, prefix the command with `set PYTHONUTF8=1 &&` or run `python -X utf8 pipeline.py ...`

---

## Output Files

Five files are written to `--output_dir`:

| File | Rows | Schema |
|---|---|---|
| `segment_assignments.csv` | 500,000 | `member_id, observation_date, segment_id, segment_name, segment_confidence` |
| `state_assignments.csv` | 500,000 | `member_id, observation_date, state_name, state_confidence, supporting_evidence` |
| `transition_predictions.csv` | 500,000 | `member_id, current_segment_id, predicted_segment_id, prediction_confidence, prob_S01…prob_S05` |
| `segment_profiles.json` | 5 entries | Per-segment: description, size, key_characteristics, cardholder_composition, common_states, recommended_activation |
| `feature_descriptions.json` | 51 entries | Per-feature: name, description, source_tables, temporal_window, computation |

### Segment Reference

| ID | Name | Count (Dec 2025) | Share |
|---|---|---:|---:|
| S01 | Growth Builder | 198,035 | 39.61% |
| S02 | High-Tier Accelerator | 87,688 | 17.54% |
| S03 | Program Skeptic | 87,505 | 17.50% |
| S04 | Silent Accumulator | 126,655 | 25.33% |
| S05 | Plateau Cruiser | 117 | 0.02% |

### State Reference

Ten Loyalty-Native States, matching the problem statement exactly:

`Brand Advocate`, `Lapse Risk`, `Momentum Builder`, `New & Uncertain`, `Plateau Cruiser`, `Program Skeptic`, `Redemption Hunter`, `Silent Accumulator`, `Value Maximizer`, `Win-Back Target`

---

## Reproducibility

| Factor | Detail |
|---|---|
| Random seed | 42 — set identically across K-Means, XGBoost, numpy, and all train/test splits |
| Hardcoded dates | None in `pipeline.py` — all dates derived from `--observation_date` |
| External data | None — all features derived exclusively from the three provided parquet files |
| Package versions | Pinned in `requirements.txt`, verified against the working interpreter |
| Parquet engine | `pyarrow` specified explicitly on every read and write |
| Model freeze | K-Means centroids fitted once on Dec 2025; never refitted at inference |
| Bundle contract | `scripts/check_model_contract.py`, enforced in CI |

Two runs on the same machine with the same input files produce byte-identical output files.

### Retraining

```bash
python src/07_lifecycle_states.py       # regenerate lifecycle state labels
python src/08_transition_prediction.py  # retrain and rewrite the model bundle
```

`src/08` writes `models/segment_transition_model.pkl` in exactly the schema
`pipeline.py` reads, and asserts as much before saving. `check_model_contract.py`
re-verifies it independently — including that the `state_map` encoding in the
bundle matches `src/utils/state_rules.STATE_IDS`, so `y_curr` cannot be encoded
one way during training and another way at inference.

---

## Project Structure

```
TBIE_CODE/
├── pipeline.py                  -- single-command inference runner
├── requirements.txt             -- pinned, verified against the working interpreter
├── pyproject.toml               -- ruff + pytest config
├── Dockerfile                   -- reproducible inference image (Python 3.12)
├── Makefile                     -- make check / predict / train / analysis / serve
├── README.md
├── RESULTS.md                   -- every benchmark, with the command that reproduces it
├── METHODOLOGY.md               -- features, clustering, transition model, state rules
├── MODEL_CARD.md                -- intended use, limitations, ethical considerations
├── data_quality_report.md
├── src/
│   ├── 01_validate_raw.py …  08_transition_prediction.py   -- build + train
│   ├── 09_calibration.py        -- probability reliability + isotonic fit
│   ├── 10_cost_thresholds.py    -- expected-value operating point
│   ├── 11_shap_explain.py       -- global + per-member attributions
│   ├── 12_drift_monitor.py      -- PSI/KS drift, retrain trigger
│   ├── 13_tune_hyperparams.py   -- Optuna search (validation only)
│   ├── 14_clustering_search.py  -- silhouette vs downstream F1 sweep
│   └── utils/                   -- ONE implementation of each shared concern
│       ├── state_rules.py       -- the 10-state cascade (phase 7 + pipeline + API)
│       ├── velocity.py          -- velocity features (phase 8 + pipeline + API)
│       ├── lag_features.py      -- month-over-month deltas + segment history
│       ├── pairs.py             -- transition pairs + model-matrix construction
│       ├── economics.py         -- value at risk, contact policies, campaign P&L
│       ├── calibration.py       -- ECE / MCE / reliability curves
│       ├── datetime_parser.py   -- strict dual-format parser
│       └── leakage_guard.py     -- assertions run on every snapshot build
├── serving/
│   ├── api.py                   -- FastAPI inference service
│   └── dashboard.html           -- self-contained demo UI
├── scripts/
│   └── check_model_contract.py  -- verifies bundles match what pipeline.py reads
├── tests/                       -- rules, features, parser, leakage guards
├── tools/                       -- presentation/deck tooling (not the inference path)
├── .github/workflows/ci.yml     -- tests, lint, model-bundle contract
├── data/train/                  -- input parquet files (not tracked)
├── segments/                    -- frozen K-Means model + segment definitions
├── models/                      -- frozen XGBoost bundle + state definitions
├── docs/                        -- phase findings, algorithm selection notes
├── features/  snapshots/  states/   -- generated intermediates (not tracked)
├── validation/                  -- phase summary logs (generated)
└── outputs/                     -- generated output files (not tracked)
```

Only the two frozen `.pkl` model files and the JSON definitions are tracked as
binaries. Everything under `features/`, `snapshots/`, `states/` and `outputs/`
is derived from `data/train/` plus those models, and is regenerated by running
the pipeline.

---

## Key Design Decisions

1. **K-Means over HDBSCAN.** HDBSCAN classified 71–85% of members as noise across all tested configurations. Loyalty behavioural data is Gaussian, not density-separated. Full rationale in `docs/algorithm_selection.md`.

2. **k=5 over k=6.** k=6 produced a microcluster of under 0.8% of the population with unstable month-to-month assignments, reducing Macro F1 by 0.12. k=5 achieves the highest Calinski-Harabasz score (77,123 vs 71,222).

3. **Walk-forward validation only.** No random shuffling at any stage. Train = Feb through Sep, Val = Oct→Nov, Test = Nov→Dec. Future data is never seen during training.

4. **Frozen centroids at inference.** Guarantees identical segment assignments when Kobie re-runs the pipeline at 2026-01-31 with Months 13–14 data.

5. **No external data.** All 46 features are derived exclusively from the three provided parquet files.
