# E32 inference benchmark

## Per-checkpoint measurements for the 20 additions

Each checkpoint was timed over the same 4,000 responses selected for E31, on the local NVIDIA GeForce RTX 5070 Ti (16,303 MiB reported device memory). GPU batch sizes were found with the existing OOM-reducing benchmark. CPU-only CatBoost and SVC checkpoints have no GPU batch size or VRAM measurement; they are scored as one full 4,000-row CPU workload by the existing inference path.

| Checkpoint | Model | Stable batch | Peak allocated MiB | Peak reserved MiB | Load s | Inference s | Cleanup s | Total s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N0048 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0104 | ultrabert380 | 8 | 1633.7 | 2022.0 | 0.9 | 37.4 | 0.2 | 40.8 |
| N0035 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.6 | 0.0 | 4.6 |
| N0007 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0029 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0036 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0101 | qwen17 | 64 | 15003.5 | 15296.0 | 1.2 | 72.5 | 0.2 | 79.8 |
| N0083 | svc | N/A (CPU) | N/A | N/A | N/A | 4.8 | 0.0 | 4.8 |
| N0046 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0038 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.9 | 0.0 | 4.9 |
| N0093 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.9 | 0.0 | 4.9 |
| N0039 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0070 | qwen06 | 128 | 15151.7 | 15452.0 | 0.9 | 41.2 | 0.2 | 46.3 |
| N0121 | ultrabert380 | 8 | 1633.7 | 2022.0 | 0.7 | 37.3 | 0.2 | 39.5 |
| N0085 | svc | N/A (CPU) | N/A | N/A | N/A | 4.8 | 0.0 | 4.8 |
| N0032 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.8 | 0.0 | 4.8 |
| N0005 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.7 | 0.0 | 4.7 |
| N0034 | catboost_stylometry | N/A (CPU) | N/A | N/A | N/A | 4.6 | 0.0 | 4.6 |
| N0115 | ultrabert380 | 8 | 1633.7 | 2022.0 | 0.7 | 37.3 | 0.2 | 39.4 |
| N0116 | ultrabert380 | 8 | 1633.7 | 2022.0 | 0.7 | 37.4 | 0.2 | 39.6 |

## Full workload timing

Both passes used the same frozen 4,000-response workload, one local GPU, the same E19-F-A, E19-F-C and E19-D grouping implementations, and Hungarian assignment. The 60-checkpoint pass uses the frozen E32 router and roster.

| Roster | Local end-to-end wall s | Checkpoint inference wall s | Sum checkpoint total s | Sum pure inference s | Grouping + assignment + remaining overhead s |
|---:|---:|---:|---:|---:|---:|
| 40 | 1587.1 | 1533.5 | 1525.4 | 1459.5 | 61.6 |
| 60 | 1931.2 | 1877.3 | 1865.1 | 1785.7 | 66.2 |

The 60-checkpoint local pass took **344.2 seconds (21.7%) longer** than the 40-checkpoint pass on the same sample. The E32 pass includes all 60 model probabilities, grouping, and Hungarian assignment.

## E28 two-GPU Kaggle runtime estimate

E28’s longest-runtime-first scheduler distributes checkpoint inference across two GPU workers. Its predicted checkpoint-worker durations use the local per-checkpoint timings; grouping and assignment are added once after the workers finish. These are scheduling estimates, not measured Kaggle T4 runtimes. GPU throughput on Kaggle may differ.

| Roster | Worker 1 s | Worker 2 s | Estimated checkpoint wall s | Grouping and assignment s | Estimated total s | Estimated total min |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 762.2 | 763.3 | 763.3 | 52.0 | 815.3 | 13.6 |
| 60 | 933.2 | 931.9 | 933.2 | 52.2 | 985.5 | 16.4 |

## Deployment batch parameters

The frozen `e32_manifest.json` contains a per-checkpoint `maximum_stable_batch_size` for all 60 roster members, plus the 4,000-row workload hash. Neural checkpoint batch sizes are used as the configured starting points for Kaggle; inference retains automatic OOM batch reduction. For CPU CatBoost/SVC entries, the manifest uses `1` as a placeholder because their inference functions are not GPU-batched; those rows are explicitly marked as CPU-only in this report.

## Reproducibility and limits

- Workload seed: 20261123; row-ID and text hashes are stored in `artifacts/analysis/e32/inference_benchmark.json`.
- Per-checkpoint probability arrays and per-row predictions for the 40- and 60-checkpoint passes are saved as `workload_probabilities_{40,60}.npz` and `workload_predictions_{40,60}.parquet`.
- Local timings include dynamic OOM retries observed for several models. A reported stable batch is the largest size accepted by the benchmark on this exact workload and local GPU.
- The Kaggle estimate applies the E28 two-worker scheduling policy to local RTX 5070 Ti durations; it is an estimate and does not substitute for a T4 run.
