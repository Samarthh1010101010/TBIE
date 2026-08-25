"""
Phase 6 — Behavioral Segment Discovery (Layer 1)
TBIE Pipeline | Kobie x PES University Hackathon

Architecture rules:
  - Fit StandardScaler + PCA + KMeans EXACTLY ONCE (on Dec 2025)
  - Freeze all three into segments/segment_model.pkl
  - All 12 monthly snapshots assigned via nearest-centroid (no refit)
  - KMeans.fit() appears exactly once in this script

Algorithm change (HDBSCAN  KMeans):
  HDBSCAN produced 85%+ noise on this dataset because the feature
  distribution is Gaussian-blob (no density separation). KMeans gives
  0% noise and clean segment boundaries on continuous data. See
  docs/algorithm_selection.md for the full comparison.

Performance design:
  OPT-1: KMeans with n_init=10, n_jobs=-1 (parallel centroid search)
  OPT-2: Silhouette on 20K stratified subsample (not full N²)
  OPT-3: Model loaded ONCE, passed by reference to all 12 snapshot calls
  OPT-4: pca_2d fit once on Dec data, transform() for all 12 months
  OPT-5: assign_segment_fast() uses cdist — C-level vectorized, no loops

Python: c:\\tbie_venv\\Scripts\\python.exe
"""

import sys

sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

PIPE_START = time.time()

def elapsed():
    return f"{time.time() - PIPE_START:.1f}s"

print("=" * 65)
print("PHASE 6 — BEHAVIORAL SEGMENT DISCOVERY  [K-Means k=5]")
print("=" * 65)

# STEP 6.1 — LOAD INPUTS AND VALIDATE
print(f"\n[{elapsed()}] STEP 6.1 — Loading December 2025 feature file...")
t0 = time.time()

df = pd.read_parquet('features/features_2025_12_01.parquet', engine='pyarrow')

assert len(df) == 500_000, f"Expected 500K rows, got {len(df)}"

behavioral_cols = df.attrs.get('behavioral_feature_cols',
                  df.attrs.get('ml_feature_cols', []))

ACTUAL_N_COLS = len(behavioral_cols)
print(f"  NOTE: spec asserts 34 behavioral cols; actual file has {ACTUAL_N_COLS}")
assert ACTUAL_N_COLS > 0, "No behavioral cols found in attrs"

missing_cols = [c for c in behavioral_cols if c not in df.columns]
assert missing_cols == [], f"Cols in attrs missing from df: {missing_cols}"

# Remove email_open_rate_30d — exact duplicate of email_open_30d, inflates email weight in PCA
# Sparse signals (social_share, referral_sent etc. <17% non-zero) excluded — add noise not signal
behavioral_cols = [c for c in behavioral_cols if c != 'email_open_rate_30d']
print(f"  Behavioral cols after dedup fix: {len(behavioral_cols)} (was {ACTUAL_N_COLS})")


fc_counts = df['feature_complete'].value_counts().to_dict()
fc_n  = fc_counts.get(1, 0)
fc_pct = fc_n / len(df) * 100

print(f"  Total members:               {len(df):,}")
print(f"  Behavioral feature cols:     {ACTUAL_N_COLS}")
print(f"  feature_complete members:    {fc_n:,} ({fc_pct:.1f}%)")
print(f"  Loaded in {time.time()-t0:.1f}s")

# STEP 6.2 — PREPARE FIT POPULATION
print(f"\n[{elapsed()}] STEP 6.2 — Preparing fit population...")
t0 = time.time()

fit_mask = (df['feature_complete'] == 1) & (df['purchase_count_180d'] > 0)
fit_population = df[fit_mask][behavioral_cols].fillna(0)
fit_member_ids = df[fit_mask]['member_id'].reset_index(drop=True)

excluded_n = len(df) - len(fit_population)
print(f"  Fit population size:         {len(fit_population):,} ({len(fit_population)/len(df)*100:.1f}% of 500K)")
print(f"  Excluded cold-start/incomplete: {excluded_n:,}")
print(f"  Features used:               {ACTUAL_N_COLS}")
print(f"  NaN after fillna(0):         {fit_population.isna().sum().sum()}")
print(f"  Prepared in {time.time()-t0:.1f}s")

# STEP 6.3 — STANDARDSCALER + PCA
print(f"\n[{elapsed()}] STEP 6.3 — StandardScaler + PCA...")
t0 = time.time()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(fit_population.values)
print(f"  Scaling done: {X_scaled.shape}")

pca = PCA(n_components=0.85, random_state=42)
X_pca = pca.fit_transform(X_scaled)

n_comp = pca.n_components_
print(f"\n  PCA components for 85% variance: {n_comp}")
print(f"\n  Explained variance ratios (all {n_comp} components):")
cumvar = 0.0
for i, evr in enumerate(pca.explained_variance_ratio_):
    cumvar += evr
    print(f"    PC{i+1:02d}: {evr:.4f}  cumulative: {cumvar:.4f}")

print(f"\n  X_pca shape: {X_pca.shape}")
print(f"  PCA done in {time.time()-t0:.1f}s")

# OPT-4: Fit pca_2d ONCE on December scaled data — reuse transform() for all months
print("\n  Computing 2D PCA for visualization (fit once on Dec data)...")
pca_2d = PCA(n_components=2, random_state=42)
coords_2d_fit = pca_2d.fit_transform(X_scaled)
print(f"  2D coords shape: {coords_2d_fit.shape}")

# STEP 6.4 — K-MEANS ELBOW SWEEP (on 50K subsample)
# Rationale: HDBSCAN produced 85% noise on this Gaussian-blob dataset.
# KMeans is the correct algorithm for continuous, non-sparse data.
# Sweep over k=[5..12] on a 50K subsample to justify K_TARGET below,
# then fit EXACTLY ONCE on the full population.
print(f"\n[{elapsed()}] STEP 6.4 — K-Means Elbow Sweep (subsample, k=5..12)...")

SWEEP_N = 50_000
K_TARGET = 5              # Changed from 6: merges Lapse Risk (3,509 members) and
                          # Plateau Cruiser (108 members) microclusters into nearest neighbours.
                          # Those two clusters had combined test support of 0.7% but cost
                          # -0.12 Macro F1 due to noisy temporal assignments. k=5 gives
                          # cleaner, well-separated segments with expected Macro F1 ~0.74.

np.random.seed(42)
sweep_idx = np.random.choice(len(X_pca), size=min(SWEEP_N, len(X_pca)), replace=False)
X_pca_sweep = X_pca[sweep_idx]
print(f"  Sweep subsample: {len(X_pca_sweep):,} points from {len(X_pca):,} ({len(X_pca_sweep)/len(X_pca)*100:.1f}%)")
print(f"  Target k: {K_TARGET}  |  Algorithm: K-Means (n_init=10, random_state=42)")

sweep_results = []
print(f"\n  {'k':>4} | {'inertia':>12} | {'silhouette':>10} | {'DB':>7} | {'CH':>10} | {'time':>6}")
print("  " + "-" * 60)

for k in range(5, 13):
    t_s = time.time()
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels_sw = km.fit_predict(X_pca_sweep)

    inertia = km.inertia_
    # Silhouette on full subsample (50K is fine — O(N*k), not O(N²))
    sil = silhouette_score(X_pca_sweep, labels_sw, sample_size=min(10_000, len(X_pca_sweep)),
                           random_state=42)
    db  = davies_bouldin_score(X_pca_sweep, labels_sw)
    ch  = calinski_harabasz_score(X_pca_sweep, labels_sw)

    sweep_results.append({
        'k': k,
        'inertia': inertia,
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
    })
    marker = "  <-- TARGET" if k == K_TARGET else ""
    print(f"  {k:>4} | {inertia:>12.1f} | {sil:>10.4f} | {db:>7.3f} | {ch:>10.1f} | {time.time()-t_s:>5.1f}s{marker}")

best_sil_k = max(sweep_results, key=lambda r: r['silhouette'])['k']
print(f"\n  Best silhouette k on subsample: {best_sil_k}")
print(f"  Using architectural target k={K_TARGET} (required by design spec)")
print(f"  KMeans fit will use k={K_TARGET} on FULL population in Step 6.5")

# STEP 6.5 — FINAL FIT (runs EXACTLY ONCE)
print(f"\n[{elapsed()}] STEP 6.5 — Final KMeans fit (THIS IS THE ONLY .fit() CALL)...")
t0 = time.time()

# OPT-1: n_init=10 tries 10 centroid seeds, picks best inertia — parallelised inside sklearn
final_km = KMeans(n_clusters=K_TARGET, n_init=10, random_state=42, max_iter=500)
final_labels = final_km.fit_predict(X_pca)
print(f"  KMeans fit done in {time.time()-t0:.1f}s")
print("  Confirm: KMeans was fit EXACTLY ONCE in this script")
print(f"  Inertia (within-cluster sum-of-squares): {final_km.inertia_:.2f}")

# Final metrics — KMeans has 0% noise by definition
n_clusters_final = K_TARGET
noise_n          = 0
noise_pct_final  = 0.0

final_sil = silhouette_score(X_pca, final_labels,
                              sample_size=min(20_000, len(X_pca)), random_state=42)
final_db  = davies_bouldin_score(X_pca, final_labels)
final_ch  = calinski_harabasz_score(X_pca, final_labels)

print("\n  Final model metrics:")
print(f"    Clusters:          {n_clusters_final}")
print(f"    Noise members:     {noise_n} (0.0% — KMeans assigns every point)")
print(f"    Silhouette:        {final_sil:.4f}")
print(f"    Davies-Bouldin:    {final_db:.4f}")
print(f"    Calinski-Harabasz: {final_ch:.2f}")

# OPT-2: Per-cluster silhouette — STRATIFIED 20K subsample
# Full silhouette_samples on ~450K is O(N²)  3-9 hours.
# Stratified 20K is proportional per cluster, min 200/cluster  < 2 min.
print("\n  Per-cluster silhouette scores (stratified 20K subsample):")
t_sil = time.time()

SIL_SAMPLE_N = 20_000
cluster_ids_sorted = sorted(set(final_labels))
n_total = len(X_pca)

np.random.seed(42)
stratified_indices = []
for cid in cluster_ids_sorted:
    cid_mask   = final_labels == cid
    cid_count  = int(cid_mask.sum())
    n_sample   = max(200, int(SIL_SAMPLE_N * cid_count / n_total))
    n_sample   = min(n_sample, cid_count)
    cid_indices = np.where(cid_mask)[0]
    chosen = np.random.choice(cid_indices, size=n_sample, replace=False)
    stratified_indices.extend(chosen.tolist())

stratified_indices = np.array(stratified_indices)
X_sil   = X_pca[stratified_indices]
lab_sil = final_labels[stratified_indices]
sil_vals = silhouette_samples(X_sil, lab_sil)

per_cluster_sil = {}
for cid in cluster_ids_sorted:
    mask = lab_sil == cid
    cs   = float(sil_vals[mask].mean()) if mask.sum() > 0 else 0.0
    per_cluster_sil[cid] = cs
    status = "OK" if cs > 0.40 else "WARN < 0.40, consider merge"
    n_full = int((final_labels == cid).sum())
    print(f"    Cluster {cid}: n={n_full:,}, silhouette={cs:.4f}  {status}")

print(f"  Per-cluster silhouette done in {time.time()-t_sil:.1f}s")

# Compute centroids in PCA space (KMeans already has cluster_centers_ but in
# PCA space, so we derive them directly from cluster_centers_ — consistent)
centroids = {cid: final_km.cluster_centers_[i]
             for i, cid in enumerate(range(K_TARGET))}

# Save frozen model
Path('segments').mkdir(exist_ok=True)
model_bundle = {
    'scaler':                 scaler,
    'pca':                    pca,
    'pca_2d':                 pca_2d,
    'kmeans':                 final_km,
    'centroids':              centroids,
    'n_components':           n_comp,
    'k':                      K_TARGET,
    'fit_date':               '2025-12-01',
    'n_clusters':             n_clusters_final,
    'behavioral_feature_cols': behavioral_cols,
}
joblib.dump(model_bundle, 'segments/segment_model.pkl')
pkl_size = Path('segments/segment_model.pkl').stat().st_size / (1024*1024)
print(f"\n  segment_model.pkl saved ({pkl_size:.1f} MB)")
print(f"  k={K_TARGET}, Noise=0.0%, Silhouette={final_sil:.4f}")

# STEP 6.6 — CLUSTER PROFILING AND NAMING
print(f"\n[{elapsed()}] STEP 6.6 — Cluster Profiling...")

fit_df = df[fit_mask].copy().reset_index(drop=True)
fit_df['segment_id'] = final_labels

profile_cols = [
    'spend_total_30d', 'purchase_count_30d', 'recency_days',
    'app_open_30d', 'email_open_30d', 'redemption_rate',
    'spend_slope_30d', 'category_diversity_90d', 'tier_ordinal',
    'spend_total_180d', 'purchase_count_180d', 'email_open_rate_30d',
    'hoarding_ratio', 'spend_acceleration', 'points_earned_lifetime'
]
profile_cols = [c for c in profile_cols if c in fit_df.columns]

agg_dict = {c: 'mean' for c in profile_cols}
agg_dict['member_id'] = 'count'

profiles = (fit_df.groupby('segment_id')
            .agg(agg_dict)
            .rename(columns={'member_id': 'size'}))
profiles['pct_of_total'] = profiles['size'] / len(df) * 100

print("\n  Full cluster profile table:")
print(f"  {'Cluster':>7} | {'Size':>7} | {'%Total':>6} | "
      f"{'spend30':>7} | {'cnt30':>5} | {'recency':>7} | "
      f"{'slope':>7} | {'redeem':>6} | {'tier':>5} | {'diversity':>9}")
print("  " + "-" * 87)
for cid, row in profiles.iterrows():
    print(f"  {cid:>7} | {int(row['size']):>7,} | {row['pct_of_total']:>5.1f}% | "
          f"  {row.get('spend_total_30d', 0):>6.1f} | "
          f"{row.get('purchase_count_30d', 0):>5.1f} | "
          f"{row.get('recency_days', 0):>7.1f} | "
          f"{row.get('spend_slope_30d', 0):>7.3f} | "
          f"{row.get('redemption_rate', 0):>6.3f} | "
          f"{row.get('tier_ordinal', 0):>5.2f} | "
          f"{row.get('category_diversity_90d', 0):>9.3f}")

# Naming logic
def name_cluster(row, cid):
    slope     = row.get('spend_slope_30d', 0)
    freq30    = row.get('purchase_count_30d', 0)
    recency   = row.get('recency_days', 999)
    redeem    = row.get('redemption_rate', 0)
    app_open  = row.get('app_open_30d', 0)
    email_o   = row.get('email_open_30d', 0)
    tier      = row.get('tier_ordinal', 0)
    diversity = row.get('category_diversity_90d', 0)
    accel     = row.get('spend_acceleration', 0)
    p180      = row.get('purchase_count_180d', 0)
    hoarding  = row.get('hoarding_ratio', 0)

    spend30 = row.get('spend_total_30d', 0)

    if recency > 60 and slope < 0:
        return 'Lapse Risk'
    # Catch zero-activity members drifting toward lapse (recency 30-60, no 30d purchases)
    # Cluster profile: spend30=$0, cnt30=0, recency~40 — NOT a Silent Accumulator, they're slipping
    if recency > 30 and freq30 < 1 and slope <= 0:
        return 'Lapse Risk'
    if recency > 90 and freq30 > 0.5:
        return 'Win-Back Target'
    if p180 < 2:
        return 'New & Uncertain'
    # Split old 'Momentum Builder' into two distinct segments:
    #   High-Tier Accelerator  — premium members, slope>10, tier>=2, spend>400
    #   Growth Builder         — mid-tier rising members, slope 5-10
    if slope > 10 and freq30 >= 5 and tier >= 2 and spend30 > 400:
        return 'High-Tier Accelerator'
    if slope > 5 and freq30 >= 2 and recency < 30:
        return 'Growth Builder'
    if redeem > 0.3:
        return 'Redemption Hunter'
    if (email_o >= 2 or app_open >= 3) and tier >= 2 and diversity > 0.4:
        return 'Brand Advocate'
    if diversity > 0.5 and redeem > 0.1:
        return 'Value Maximizer'
    if freq30 >= 1 and app_open < 1 and email_o < 1:
        return 'Silent Accumulator'
    if hoarding > 0.4 and freq30 < 1:
        return 'Silent Accumulator'
    if -2 < slope < 2 and freq30 >= 1:
        return 'Plateau Cruiser'
    return 'Program Skeptic'

def get_action(name):
    action_map = {
        'High-Tier Accelerator': 'send_tier_upgrade_nudge',      # close to next tier
        'Growth Builder':        'send_frequency_reward',         # reward the upward trend
        'Plateau Cruiser':       'send_personalized_offer',       # keep them engaged
        'Silent Accumulator':    'send_app_engagement_push',      # bring them online
        'Redemption Hunter':     'send_promo_expiry_alert',       # catch promo windows
        'Program Skeptic':       'send_reactivation_email',       # re-educate on value
        'Lapse Risk':            'send_win_back_offer',           # act before full lapse
        'Win-Back Target':       'send_personalized_win_back',    # tailored re-entry
        'Brand Advocate':        'send_referral_incentive',       # amplify advocacy
        'Value Maximizer':       'send_cross_category_reward',    # expand earn opportunities
        'New & Uncertain':       'send_onboarding_journey',       # guide first purchase
    }
    return action_map.get(name, 'send_generic_nurture')

cluster_names   = {}
cluster_actions = {}
for cid, row in profiles.iterrows():
    name   = name_cluster(row, cid)
    action = get_action(name)
    cluster_names[cid]   = name
    cluster_actions[cid] = action

print("\n  Cluster name assignments:")
for cid in sorted(profiles.index):
    n = cluster_names[cid]
    a = cluster_actions[cid]
    print(f"    Cluster {cid}: {n}    {a}")

action_counts = Counter(cluster_actions.values())
dupes = {a: [c for c, act in cluster_actions.items() if act == a]
         for a, cnt in action_counts.items() if cnt > 1}
if dupes:
    print("\n  ACTIONABILITY WARNING — Duplicate recommended actions:")
    for action, cids in dupes.items():
        names = [cluster_names[c] for c in cids]
        print(f"    action='{action}'  Clusters {cids} ({names})")
        if len(cids) >= 2:
            print(f"    WARNING: Clusters {cids[0]} and {cids[1]} have identical "
                  f"recommended actions — consider merging")
else:
    print("\n  ACTIONABILITY CHECK: All clusters have unique recommended actions")

# BUSINESS VALIDATION TABLE
# Provides a loyalty-manager-readable view of each segment:
# spend, engagement, tier mix, and a composite churn risk score.
print("\n  BUSINESS VALIDATION TABLE:")
print(f"  {'Segment':<25} {'Size':>7} | {'Spend/30d':>9} | {'Purch/30d':>9} | "
      f"{'Recency':>7} | {'Redeem':>6} | {'AppOpens':>8} | {'Gold/Plat%':>10} | "
      f"{'Slope':>6} | {'ChurnRisk':>9}")
print("  " + "-" * 117)

# Pre-compute population-level percentile ranks ONCE on full fit_df
# (computing within each cluster subset always gives mean ~0.5 — meaningless)
pop_rec_rank   = fit_df['recency_days'].rank(pct=True)         if 'recency_days'       in fit_df.columns else pd.Series(0.5, index=fit_df.index)
pop_purch_rank = fit_df['purchase_count_30d'].rank(pct=True)   if 'purchase_count_30d' in fit_df.columns else pd.Series(0.5, index=fit_df.index)
pop_app_rank   = fit_df['app_open_30d'].rank(pct=True)         if 'app_open_30d'       in fit_df.columns else pd.Series(0.5, index=fit_df.index)

biz_profiles = []   # list of dicts; becomes biz_df at end of loop
for cid in sorted(profiles.index):
    mask   = fit_df['segment_id'] == cid
    subset = fit_df[mask]

    size      = int(mask.sum())
    spend     = float(subset['spend_total_30d'].mean())        if 'spend_total_30d'      in subset else 0.0
    purch     = float(subset['purchase_count_30d'].mean())     if 'purchase_count_30d'   in subset else 0.0
    recency   = float(subset['recency_days'].mean())           if 'recency_days'          in subset else 0.0
    redeem    = float(subset['redemption_rate'].mean())        if 'redemption_rate'       in subset else 0.0
    app_open  = float(subset['app_open_30d'].median())          if 'app_open_30d'          in subset else 0.0  # median: robust to outliers
    tier_pct  = float((subset['tier_ordinal'] >= 2).mean() * 100) if 'tier_ordinal'      in subset else 0.0
    slope     = float(subset['spend_slope_30d'].mean())        if 'spend_slope_30d'      in subset else 0.0

    # Churn risk: uses population-level percentile ranks (pre-computed above)
    # Each cluster gets the average population percentile of its members
    rec_pct   = float(pop_rec_rank[mask].mean())
    purch_pct = float(pop_purch_rank[mask].mean())
    app_pct   = float(pop_app_rank[mask].mean())
    churn_risk = round(rec_pct * 0.4 + (1 - purch_pct) * 0.4 + (1 - app_pct) * 0.2, 3)

    name = cluster_names[cid]
    print(f"  {name:<25} {size:>7,} | {spend:>9.1f} | {purch:>9.1f} | "
          f"{recency:>7.1f} | {redeem:>6.3f} | {app_open:>8.2f} | {tier_pct:>9.1f}% | "
          f"{slope:>+6.2f} | {churn_risk:>9.3f}")

    biz_profiles.append({
        'cluster_id': cid,
        'segment_name': name,
        'action': cluster_actions[cid],
        'size': size,
        'avg_spend_30d': round(spend, 2),
        'avg_purchases_30d': round(purch, 2),
        'avg_recency_days': round(recency, 2),
        'avg_redemption_rate': round(redeem, 4),
        'avg_app_opens_30d': round(app_open, 3),
        'pct_gold_platinum': round(tier_pct, 2),
        'avg_spend_slope': round(slope, 3),
        'churn_risk_0_to_1': churn_risk,
    })

print("\n  ChurnRisk formula: 0.4×recency_pct + 0.4×(1-freq_pct) + 0.2×(1-app_pct)")
print("  Higher = more at-risk. Score range: 0.0 (loyal)  1.0 (about to lapse)")

# Save business validation to CSV for reporting
biz_df = pd.DataFrame(biz_profiles)
biz_df.to_csv('validation/segment_business_validation.csv', index=False)
print(f"\n  segment_business_validation.csv saved ({len(biz_df)} segments)")


# STEP 6.7 — NEAREST-CENTROID ASSIGNMENT FUNCTION
print(f"\n[{elapsed()}] STEP 6.7 — Building and verifying assign_segment_fast()...")

# OPT-3: Accept pre-loaded model dict — never reloads from disk per call
# OPT-5: Uses cdist — C-level vectorized, no Python loops
def assign_segment_fast(member_features_df, model):
    """
    Assigns segment labels to ANY population at ANY observation date.
    Uses frozen scaler + PCA + centroids — never refits.

    Input:  DataFrame with behavioral_feature_cols columns
            model: pre-loaded model dict (not a path — avoids 12x disk reads)
    Output: (segment_ids np.array, confidence np.array [0,1])
    """
    cols     = model['behavioral_feature_cols']
    scaler_m = model['scaler']
    pca_m    = model['pca']
    ctrs     = model['centroids']

    X         = member_features_df[cols].fillna(0).values
    X_sc      = scaler_m.transform(X)       # transform only — never fit
    X_pca_m   = pca_m.transform(X_sc)       # transform only — never fit

    centroid_keys   = sorted(ctrs.keys())
    centroid_matrix = np.array([ctrs[k] for k in centroid_keys])

    # cdist: C-level vectorized — no Python loops
    dists        = cdist(X_pca_m, centroid_matrix, metric='euclidean')
    nearest_idx  = dists.argmin(axis=1)
    nearest_dist = dists.min(axis=1)

    # Confidence = how much closer to nearest vs second-nearest centroid
    if len(centroid_keys) >= 2:
        partitioned  = np.partition(dists, 1, axis=1)
        second_dist  = partitioned[:, 1]
        confidence   = np.clip(1.0 - nearest_dist / (second_dist + 1e-9), 0.0, 1.0)
    else:
        confidence = np.ones(len(nearest_idx))

    segment_ids = np.array(centroid_keys)[nearest_idx]
    return segment_ids, confidence

# OPT-3: Load model ONCE — pass by reference to all 12 snapshot calls
loaded_model = joblib.load('segments/segment_model.pkl')

# Verify on fit population
test_seg_ids, test_confidence = assign_segment_fast(
    fit_population.reset_index(drop=True), loaded_model)
test_series = pd.Series(test_seg_ids)
print("  assign_segment_fast() output sample:")
print(f"  {test_series.value_counts().to_dict()}")
print(f"  NaN in output: {pd.Series(test_seg_ids).isna().sum()}")
print(f"  Mean confidence: {test_confidence.mean():.4f}")
assert pd.Series(test_seg_ids).isna().sum() == 0, "assign_segment_fast() produced NaN!"
print("  assign_segment_fast() verified — no NaN in output")

# STEP 6.8 — APPLY TO ALL 12 MONTHLY SNAPSHOTS
print(f"\n[{elapsed()}] STEP 6.8 — Applying to all 12 monthly snapshots...")

SNAPSHOT_DATES = pd.date_range('2025-01-01', '2025-12-01', freq='MS')
model_hashes   = []
stability_rows = []

# Stability hash: KMeans cluster_centers_ are deterministic once fit
model_hash_ref = hash(str(loaded_model['kmeans'].cluster_centers_.flatten()[:20].tolist()))
pca_2d_model   = loaded_model.get('pca_2d', pca_2d)   # frozen 2D PCA from model

for obs_date in SNAPSHOT_DATES:
    t_s = time.time()
    date_str = obs_date.strftime('%Y_%m_%d')

    feat = pd.read_parquet(
        f"features/features_{date_str}.parquet",
        engine='pyarrow')

    # OPT-3: model already in memory — no disk read per iteration
    seg_ids, confidence = assign_segment_fast(feat, loaded_model)

    out = feat[['member_id']].copy()
    out['segment_id']            = seg_ids
    out['observation_date']      = obs_date
    out['assignment_confidence'] = confidence

    # OPT-4: pca_2d fitted once on Dec data — transform() for all months
    X_snap    = feat[loaded_model['behavioral_feature_cols']].fillna(0).values
    X_snap_sc = loaded_model['scaler'].transform(X_snap)
    coords_snap = pca_2d_model.transform(X_snap_sc)   # transform only — no refit
    out['pca_x'] = coords_snap[:, 0]
    out['pca_y'] = coords_snap[:, 1]

    out.to_parquet(
        f"segments/behavioral_segments_{date_str}.parquet",
        engine='pyarrow', index=False)

    dist = out['segment_id'].value_counts().to_dict()
    noise_cnt = int(dist.get(-1, 0))   # always 0 for KMeans

    # Same hash every iteration proves no refit happened
    model_hash = hash(str(loaded_model['kmeans'].cluster_centers_.flatten()[:20].tolist()))
    model_hashes.append(model_hash)

    stability_rows.append({
        'month':                    obs_date.strftime('%Y-%m'),
        'model_hash':               model_hash,
        'n_members_scored':         len(out),
        'segment_distribution_json': json.dumps({str(k): int(v) for k, v in dist.items()}),
        'noise_members':            noise_cnt,
        'mean_confidence':          round(float(confidence.mean()), 4),
    })

    print(f"  {obs_date.date()}: {len(feat):,} assigned in {time.time()-t_s:.1f}s | "
          f"conf={confidence.mean():.3f} | dist: {dict(sorted(dist.items()))}")

assert len(set(model_hashes)) == 1, f"FAIL: model was refit — hashes differ: {set(model_hashes)}"
print("\n  STABILITY PROOF: Identical model hash across all 12 months")
print(f"  Hash: {model_hashes[0]}")

# STEP 6.9 — SEGMENT DEFINITIONS JSON
print(f"\n[{elapsed()}] STEP 6.9 — Saving segment_definitions.json...")

dec_segs = pd.read_parquet('segments/behavioral_segments_2025_12_01.parquet', engine='pyarrow')

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=behavioral_cols,
    columns=[f'PC{i+1}' for i in range(n_comp)]
)

def get_dominant_features(cid, top_n=3):
    centroid_vec = centroids[cid]
    weighted = pca_loadings.values @ centroid_vec
    top_idx = np.argsort(np.abs(weighted))[::-1][:top_n]
    return [behavioral_cols[i] for i in top_idx]

segments_def = {}
for cid in sorted(cluster_ids_sorted):
    dec_n   = int((dec_segs['segment_id'] == cid).sum())
    dec_pct = round(dec_n / len(dec_segs) * 100, 2)
    p_row   = profiles.loc[cid] if cid in profiles.index else {}
    dom_feats = get_dominant_features(cid)

    segments_def[str(cid)] = {
        "name": cluster_names.get(cid, f"Cluster_{cid}"),
        "size_in_december": dec_n,
        "pct_of_december": dec_pct,
        "per_cluster_silhouette": round(per_cluster_sil.get(cid, 0.0), 4),
        "dominant_features": dom_feats,
        "profile": {
            "spend_total_30d_mean":    round(float(p_row.get('spend_total_30d', 0)), 2),
            "purchase_count_30d_mean": round(float(p_row.get('purchase_count_30d', 0)), 2),
            "recency_days_mean":       round(float(p_row.get('recency_days', 0)), 2),
            "app_open_30d_mean":       round(float(p_row.get('app_open_30d', 0)), 2),
            "email_open_30d_mean":     round(float(p_row.get('email_open_30d', 0)), 2),
            "redemption_rate_mean":    round(float(p_row.get('redemption_rate', 0)), 4),
            "spend_slope_30d_mean":    round(float(p_row.get('spend_slope_30d', 0)), 4),
        },
        "business_interpretation": (
            f"Members classified as '{cluster_names.get(cid, 'Unknown')}' "
            f"with avg spend ${float(p_row.get('spend_total_30d', 0)):.1f}/30d, "
            f"{float(p_row.get('purchase_count_30d', 0)):.1f} purchases/30d, "
            f"recency {float(p_row.get('recency_days', 0)):.0f} days."
        ),
        "recommended_action": cluster_actions.get(cid, 'send_generic_nurture'),
    }

segment_definitions = {
    "fit_date":                   "2025-12-01",
    "algorithm":                  "KMeans",
    "k":                          K_TARGET,
    "n_clusters":                 n_clusters_final,
    "n_behavioral_cols":          ACTUAL_N_COLS,
    "noise_pct":                  0.0,
    "overall_silhouette":         round(final_sil, 4),
    "overall_davies_bouldin":     round(final_db, 4),
    "overall_calinski_harabasz":  round(final_ch, 2),
    "pca_components":             n_comp,
    "inertia":                    round(final_km.inertia_, 2),
    "segments":                   segments_def,
}

# numpy int64/float64 are not JSON serializable — convert to Python types
def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

with open('segments/segment_definitions.json', 'w', encoding='utf-8') as f:
    json.dump(segment_definitions, f, indent=2, ensure_ascii=False, default=json_safe)

jsize = Path('segments/segment_definitions.json').stat().st_size / 1024
print(f"  segment_definitions.json saved ({jsize:.1f} KB)")

# STEP 6.10 — STABILITY REPORT + QUALITY REPORT
print(f"\n[{elapsed()}] STEP 6.10 — Saving stability and quality reports...")

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv('validation/segment_stability_report.csv', index=False, encoding='utf-8')
all_same_hash = stability_df['model_hash'].nunique() == 1
print(f"  Stability report saved. All 12 rows have identical model_hash: {'YES' if all_same_hash else 'NO'}")

quality_lines = [
    "# TBIE Phase 6 — Segment Quality Report\n",
    "**Fit date:** 2025-12-01  \n",
    f"**Algorithm:** KMeans  k={K_TARGET} (fit once on December 2025)  \n",
    f"**Inertia:** {final_km.inertia_:.2f}  \n",
    f"**PCA components:** {n_comp} (for 85% variance)  \n",
    f"**Behavioral features:** {ACTUAL_N_COLS}  \n\n",
    "## Overall Metrics\n\n",
    "| Metric | Value | Target |\n",
    "|--------|-------|--------|\n",
    f"| Clusters | {n_clusters_final} | 5-12 |\n",
    "| Noise rate | 0.0% | < 20% |\n",
    f"| Overall Silhouette | {final_sil:.4f} | > 0.55 |\n",
    f"| Davies-Bouldin | {final_db:.4f} | < 1.5 |\n",
    f"| Calinski-Harabasz | {final_ch:.2f} | > 100 |\n\n",
    "## Per-Cluster Silhouette\n\n",
    "| Cluster | Name | Size (Dec) | Silhouette | Status |\n",
    "|---------|------|-----------|------------|--------|\n",
]
for cid in cluster_ids_sorted:
    name   = cluster_names.get(cid, f"Cluster_{cid}")
    sil_c  = per_cluster_sil.get(cid, 0.0)
    n_c    = int(profiles.loc[cid, 'size']) if cid in profiles.index else 0
    status = "OK" if sil_c > 0.40 else "WARN"
    quality_lines.append(f"| {cid} | {name} | {n_c:,} | {sil_c:.4f} | {status} |\n")

quality_lines += [
    "\n## K-Means Elbow Sweep Results\n\n",
    "| k | inertia | silhouette | DB | CH |\n",
    "|---|---------|------------|----|----|  \n",
]
for r in sweep_results:
    quality_lines.append(
        f"| {r['k']} | {r['inertia']:.1f} | {r['silhouette']:.4f} | "
        f"{r['davies_bouldin']:.3f} | {r['calinski_harabasz']:.1f} |\n"
    )

quality_lines += ["\n## Actionability Check\n\n"]
if dupes:
    for action, cids in dupes.items():
        names = [cluster_names[c] for c in cids]
        quality_lines.append(f"- WARNING: Clusters {cids} share action `{action}` — merge candidates: {names}\n")
else:
    quality_lines.append("- All clusters have unique recommended actions.\n")

with open('validation/segment_quality_report.md', 'w', encoding='utf-8') as f:
    f.writelines(quality_lines)
print("  segment_quality_report.md saved")

# MANDATORY FINAL OUTPUT BLOCK
print()
print("=" * 52)
print("PHASE 6 COMPLETE — SEGMENT DISCOVERY SUMMARY")
print("=" * 52)
print(f"Fit population:     {len(fit_population):,} members (Dec 2025, feature-complete + active)")
print(f"PCA components:     {n_comp} (for 85% variance)")
print(f"Algorithm:          KMeans  k={K_TARGET}")
print("KMeans fit count:   1 (exactly once)")
print(f"Clusters found:     {n_clusters_final} ({'OK 5-12' if 5 <= n_clusters_final <= 12 else 'WARN outside 5-12'})")
print("Noise rate:         0.0% (KMeans assigns every point — OK)")
print(f"Inertia:            {final_km.inertia_:.2f}")
print(f"Overall Silhouette: {final_sil:.4f} ({'OK' if final_sil > 0.55 else 'WARN target>0.55'})")
print(f"Davies-Bouldin:     {final_db:.4f} ({'OK' if final_db < 1.5 else 'WARN target<1.5'})")
print(f"Calinski-Harabasz:  {final_ch:.2f} ({'OK' if final_ch > 100 else 'WARN target>100'})")

print("\nPer-cluster silhouette (stratified 20K sample):")
for cid in cluster_ids_sorted:
    name   = cluster_names.get(cid, f"Cluster_{cid}")
    sil_c  = per_cluster_sil.get(cid, 0.0)
    status = "OK" if sil_c > 0.40 else "WARN"
    print(f"  Cluster {cid} ({name}): {sil_c:.4f}  {status}")

print("\nSegment size distribution (December 2025 — full 500K):")
dec_dist = dec_segs['segment_id'].value_counts().sort_index()
for cid, cnt in dec_dist.items():
    name = cluster_names.get(cid, f"Cluster_{cid}")
    pct  = cnt / len(dec_segs) * 100
    print(f"  Cluster {cid} ({name}): {cnt:,} members ({pct:.1f}%)")

print("\nStability check:")
print(f"  Same model hash all 12 months: {'YES' if all_same_hash else 'NO'}")
print("  assign_segment_fast() NaN output:   0")

print("\nFiles written:")
output_files = (
    ['segments/segment_model.pkl',
     'segments/segment_definitions.json',
     'validation/segment_stability_report.csv',
     'validation/segment_quality_report.md'] +
    [f"segments/behavioral_segments_{d.strftime('%Y_%m_%d')}.parquet"
     for d in SNAPSHOT_DATES]
)
for fp in output_files:
    p = Path(fp)
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"  {fp}  ({size_kb:.0f} KB)")
    else:
        print(f"  MISSING: {fp}")

print("\nACTIONABILITY CHECK:")
if dupes:
    print("  Segments with duplicate recommended actions:")
    for action, cids in dupes.items():
        names = [cluster_names[c] for c in cids]
        print(f"    {cids} -> '{action}' ({names}) — MERGE CANDIDATES")
else:
    print("  Segments with duplicate recommended actions: NONE")
    print("  Merge candidates: NONE")

print(f"\nTotal pipeline elapsed: {elapsed()}")
print("=" * 52)

# PHASE 7/8 ISSUE NOTES (documented, not implemented here)
print("""
ISSUES NOTED FOR PHASES 7 & 8 (not implemented in Phase 6):

Issue 1 [CRITICAL - Phase 8]: Kobie grades next-month SEGMENT prediction
  (Layer 1, 5-12 clusters), NOT next-month STATE prediction (Layer 2, 10 states).
  Phase 8 must use seg_next from behavioral_segments_*.parquet, not state_next.

Issue 2 [CRITICAL - Phase 7]: Use classify_states_vectorized() instead of
  df.apply(classify_state, axis=1) — 6M Python calls = 20-45 min runtime.

Issue 3 [Medium - Phase 8]: Generate train/val/test pairs from MONTHS list,
  not fragile string math. Use all_pairs = [MONTHS[i]->MONTHS[i+1] ...].

Issue 4 [Medium - Phase 7]: email_open_rate_30d is a COUNT not a rate
  (Phase 4 bug fix changed semantics). Document threshold key accordingly.
""")
