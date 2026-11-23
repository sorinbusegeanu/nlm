# E30 Inference Benchmark

Measured locally on **NVIDIA GeForce RTX 5070 Ti** (15840 MiB total VRAM), one checkpoint at a time, with a deterministic 4,000-row sample drawn across all 50 `v0_group` scenarios. The probe ceiling was 1024 and production OOM halving determined the first stable batch. This records a safe measured batch under that probe policy, not a continuous search over every possible batch size.

Pure neural inference time covers model forward passes and excludes tokenization. Checkpoint total covers `predict_candidate` model load, preprocessing, inference and cleanup. Frozen-backbone extraction APIs combine backbone load and forward timing; those entries are marked. SVC and CatBoost stylometry run on CPU, so no GPU batch or VRAM applies.

| Checkpoint | Model | Batch | Peak allocated MiB | Peak reserved MiB | Load s | Pure inference s | Cleanup s | Total s | Rows/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N0110 | granite4_h_micro | 32.00 | 14721.10 | 15408.00 | 3.36 | 179.09 | 0.19 | 193.69 | 22.33 |
| N0054 | embeddinggemma2 | 128.00 | 15276.10 | 15426.00 | 2.23 | 21.75 | 0.20 | 26.15 | 183.91 |
| N0008 | catboost_stylometry | n/a | n/a | n/a | n/a | 4.95 | 0.00 | 4.95 | 807.41 |
| N0127 | modernbert_large | 256.00 | 15079.50 | 15430.00 | 0.81 | 51.21 | 0.21 | 54.36 | 78.10 |
| N0082 | svc | n/a | n/a | n/a | n/a | 4.83 | 0.00 | 4.83 | 828.44 |
| N0065 | ministral3_3b | 64.00 | 14488.60 | 15476.00 | 4.95 | 126.97 | 0.20 | 134.45 | 31.50 |
| N0124 | ultrabert380 | 256.00 | 15212.70 | 15264.00 | 0.75 | 51.28 | 0.20 | 54.24 | 78.00 |
| N0117 | ultrabert380 | 256.00 | 15212.70 | 15264.00 | 0.72 | 51.32 | 0.20 | 54.27 | 77.94 |
| N0133 | deberta_v3_large | 64.00 | 15028.80 | 15434.00 | 1.37 | 56.92 | 0.20 | 61.32 | 70.28 |
| N0097 | granite4_h_micro | 32.00 | 14646.60 | 15408.00 | 1.48 | 139.49 | 0.19 | 144.84 | 28.68 |
| N0120 | ultrabert380 | 256.00 | 15212.70 | 15264.00 | 0.74 | 51.26 | 0.20 | 54.29 | 78.04 |
| N0033 | catboost_stylometry | n/a | n/a | n/a | n/a | 4.68 | 0.00 | 4.68 | 855.11 |
| N0059 | granite4_350m | 256.00 | 14093.70 | 15284.00 | 1.56 | 21.99 | 0.19 | 27.67 | 181.92 |
| N0113 | ultrabert380 | 1024.00 | 12227.70 | 15062.00 | 1.97 | 27.74 | 0.21 | 31.01 | 144.19 |
| N0071 | qwen06 | 64.00 | 15158.70 | 15464.00 | 4.69 | 39.66 | 0.19 | 49.86 | 100.85 |
| N0063 | llama32_3b | 64.00 | 14442.10 | 15376.00 | 4.05 | 117.43 | 0.20 | 124.42 | 34.06 |
| N0131 | electra_large | 256.00 | 14776.30 | 15244.00 | 0.64 | 38.85 | 0.20 | 42.55 | 102.96 |
| N0102 | smollm3_3b | 128.00 | 14674.10 | 15388.00 | 3.87 | 115.22 | 0.21 | 121.89 | 34.72 |
| N0130 | minicpm5_2b | 256.00 | 13654.30 | 15218.00 | 2.46 | 94.15 | 0.21 | 99.12 | 42.49 |
| N0129 | hunyuan_1_8b | 128.00 | 14128.20 | 14990.00 | 2.82 | 80.08 | 0.21 | 89.55 | 49.95 |

**Total measured pure inference:** 1278.9s.
**Sum of per-checkpoint totals:** 1378.1s.
**End-to-end wall time:** 1895.8s.
Wall minus pure inference: 616.9s; wall minus checkpoint totals: 517.6s.

## Two-GPU Kaggle estimate

E28's longest-estimated-runtime-first scheduler assigns the 20 checkpoints to two balanced workers. At a 4,000-row workload the local-measurement estimate is **694.2s** (GPU work estimates 694.2s and 683.9s). This is an RTX 5070 Ti extrapolation, not a Kaggle T4 measurement; dynamic OOM batch reduction remains enabled.
