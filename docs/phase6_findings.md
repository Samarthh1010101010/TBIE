# TBIE Phase 6 — Segment Quality Report

**Fit date:** 2025-12-01  
**Algorithm:** KMeans  k=5 (fit once on December 2025)  
**Inertia:** 10480487.19  
**PCA components:** 18 (for 85% variance)  
**Behavioral features:** 41  

## Overall Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Clusters | 5 | 5-12 |
| Noise rate | 0.0% | < 20% |
| Overall Silhouette | 0.1196 | > 0.55 |
| Davies-Bouldin | 1.7678 | < 1.5 |
| Calinski-Harabasz | 77123.64 | > 100 |

## Per-Cluster Silhouette

| Cluster | Name | Size (Dec) | Silhouette | Status |
|---------|------|-----------|------------|--------|
| 0 | Growth Builder | 180,343 | 0.0958 | WARN |
| 1 | High-Tier Accelerator | 79,487 | 0.1591 | WARN |
| 2 | Program Skeptic | 84,227 | 0.1400 | WARN |
| 3 | Silent Accumulator | 148,717 | 0.1217 | WARN |
| 4 | Plateau Cruiser | 108 | 0.6944 | OK |

## K-Means Elbow Sweep Results

| k | inertia | silhouette | DB | CH |
|---|---------|------------|----|----|  
| 5 | 1072920.0 | 0.1196 | 2.000 | 7295.3 |
| 6 | 1025958.0 | 0.1058 | 2.020 | 6560.9 |
| 7 | 955148.4 | 0.1221 | 1.557 | 6490.4 |
| 8 | 917243.6 | 0.1116 | 1.740 | 6088.1 |
| 9 | 886657.8 | 0.0996 | 1.830 | 5726.3 |
| 10 | 865914.1 | 0.0998 | 1.841 | 5344.9 |
| 11 | 847946.2 | 0.0960 | 1.918 | 5018.2 |
| 12 | 832645.5 | 0.0902 | 1.960 | 4729.2 |

## Actionability Check

- All clusters have unique recommended actions.
