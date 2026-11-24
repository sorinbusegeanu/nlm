# E34 inference benchmark

All rows refer to the same deterministic 4,000-response workload (seed 20261123) used by E31/E32 and the local RTX 5070 Ti. Existing checkpoint and group reconstruction timings are reused; the new candidate rank/features, CatBoost scoring, score adjustment and per-method Hungarian passes are measured in this run from the exact cached workload probabilities.

The E31 row uses E32's measured 40-checkpoint full-pipeline replay of E31's exact frozen roster, which includes all three groupers requested here. The original E31 benchmark's own checkpoint and end-to-end measurements are retained in `artifacts/analysis/e31/inference_benchmark.json` and cross-referenced in the new benchmark JSON.

| Experiment | Checkpoints | Checkpoint inference wall time | Candidate rescoring time | Grouping and remaining overhead | Total local wall time | Estimated dual-T4 Kaggle time |
|---|---:|---:|---:|---:|---:|---:|
| E31 | 40 | 1533.52 s | 0.00 s | 53.54 s | 1587.06 s | 815.31 s (13.59 min) |
| E33-F | 40 | 1533.52 s | 4.39 s | 53.61 s | 1591.51 s | 819.77 s (13.66 min) |
| E32 | 60 | 1877.29 s | 0.00 s | 53.96 s | 1931.25 s | 985.46 s (16.42 min) |
| E34-F | 60 | 1877.29 s | 6.44 s | 54.03 s | 1937.77 s | 991.98 s (16.53 min) |

## Candidate classifier timings

| Classifier | Roster | Reliability/rank features | Candidate feature construction | CatBoost inference | Score adjustment | Total rescoring |
|---|---:|---:|---:|---:|---:|---:|
| E33-F | 40 | 0.047 s | 4.313 s | 0.026 s | 0.0001 s | 4.386 s |
| E34-F | 60 | 0.071 s | 6.340 s | 0.033 s | 0.0000 s | 6.443 s |

## Grouping and assignment timings

The existing E32 grouping measurements include EmbeddingGemma loading/inference/cleanup and E19-F-A, E19-F-C, E19-D reconstruction. The old assignment durations are removed from that stage total and the newly timed assignment durations for the rescored outputs are added.

| Classifier | E19-F-A Hungarian | E19-F-C Hungarian | E19-D Hungarian |
|---|---:|---:|---:|
| E33-F | 0.037 s | 0.042 s | 0.031 s |
| E34-F | 0.038 s | 0.043 s | 0.032 s |

The dual-T4 figures are E28 scheduler estimates, not direct Kaggle measurements. Candidate rescoring is added after the checkpoint workers complete. The measured GPU inference stages themselves were not repeated for this report; their saved per-checkpoint loading, inference and cleanup records remain in the E31/E32 benchmark artifacts. This avoids redundant base-model inference while preserving the exact workload and checkpoint timing measurements.

Detailed machine-readable timings: `artifacts/analysis/e34/inference_benchmark.json`.
