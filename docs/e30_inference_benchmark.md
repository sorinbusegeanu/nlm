# E30 Inference Benchmark

Measured on **NVIDIA GeForce RTX 5070 Ti** (15840 MiB VRAM), one model at a time, using 4,000 deterministic rows sampled across 50 V0-derived scenarios. The effective batch column is the successful batch reported by the inference path; CUDA OOM fallback remained enabled.

| Checkpoint | Model | Effective batch | Peak allocated MiB | Peak reserved MiB | Load s | Pure inference s | Cleanup s | Total s | Rows/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N0110 | granite4_h_micro | 32 | 10892.6 | 14862.0 | 2.0 | 183.7 | 0.2 | 190.3 | 21.8 |
| N0054 | embeddinggemma2 | 128 | 7049.9 | 8216.0 | 2.2 | 21.8 | 0.2 | 25.5 | 183.9 |
| N0008 | catboost_stylometry | n/a | n/a | n/a | n/a | 4.9 | 0.0 | 4.9 | 811.8 |
| N0127 | modernbert_large | 256 | 7660.1 | 9286.0 | 0.4 | 51.1 | 0.2 | 53.1 | 78.3 |
| N0082 | svc | n/a | n/a | n/a | n/a | 5.1 | 0.0 | 5.1 | 780.3 |
| N0065 | ministral3_3b | 64 | 11847.9 | 14054.0 | 2.4 | 127.0 | 0.2 | 131.1 | 31.5 |
| N0124 | ultrabert380 | 256 | 6616.4 | 8806.0 | 0.3 | 51.2 | 0.2 | 53.1 | 78.1 |
| N0117 | ultrabert380 | 256 | 6616.4 | 8806.0 | 0.4 | 51.2 | 0.2 | 53.2 | 78.1 |
| N0133 | deberta_v3_large | 64 | 7010.7 | 9420.0 | 1.0 | 56.6 | 0.2 | 59.6 | 70.6 |
| N0097 | granite4_h_micro | 32 | 9847.8 | 12046.0 | 1.5 | 139.3 | 0.2 | 142.6 | 28.7 |
| N0120 | ultrabert380 | 256 | 6616.4 | 8806.0 | 0.3 | 51.2 | 0.2 | 53.1 | 78.2 |
| N0033 | catboost_stylometry | n/a | n/a | n/a | n/a | 4.8 | 0.0 | 4.8 | 829.7 |
| N0059 | granite4_350m | 256 | 5716.1 | 6466.0 | 0.8 | 22.0 | 0.2 | 24.5 | 181.9 |
| N0113 | ultrabert380 | 1024 | 12229.3 | 15054.0 | 0.5 | 27.6 | 0.2 | 29.4 | 145.1 |
| N0071 | qwen06 | 64 | 4625.4 | 5256.0 | 1.0 | 39.6 | 0.2 | 42.3 | 100.9 |
| N0030 | catboost_stylometry | n/a | n/a | n/a | n/a | 4.9 | 0.0 | 4.9 | 813.6 |
| N0006 | catboost_stylometry | n/a | n/a | n/a | n/a | 4.9 | 0.0 | 4.9 | 814.4 |

**Actual end-to-end wall time, 17 checkpoints sequentially:** 885.5s (14.8 min).
**Sum of pure inference:** 847.1s.
**Sum of individual checkpoint totals:** 882.3s.
**Wall minus pure inference:** 38.5s.
**Wall minus checkpoint total sum:** 3.3s.

## Two-GPU Kaggle estimate

The E28 longest-runtime-first scheduler assigns the 17 checkpoints to two workers. For another 4,000-row workload, the local-measured estimate is **443.5s**; workers are estimated at 438.8s and 443.5s.

| GPU worker | Checkpoints | Estimated seconds |
|---:|---|---:|
| 0 | N0110, N0133, N0117, N0127, N0071, N0054, N0082, N0008, N0006 | 438.8 |
| 1 | N0097, N0065, N0124, N0120, N0113, N0059, N0030, N0033 | 443.5 |

This is a scheduler estimate extrapolated from the local RTX 5070 Ti timings, not a measured T4 runtime. The public Kaggle `test.csv` currently exposes only 3 sample rows; the hidden scored workload size is not visible locally. Kaggle T4 throughput and OOM batch reductions therefore remain uncertain. The deployment notebook keeps smaller-batch fallback enabled.
