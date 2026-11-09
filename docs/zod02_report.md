# ZOD02 — Multi-Specialist Cascade

Five-fold prompt-group evaluation on frozen V0 (540 rows, 30 groups, 18 labels). Existing ranked checkpoint-label candidates are re-ranked within each fitting fold with Beta(1,1) precision smoothing and minimum support 10. Overlap checkpoints are excluded.

Each candidate rule can assign only still-unassigned rows whose checkpoint top-1 predicts that label. A rule is retained only when applying it improves fitting-fold balanced accuracy relative to the current cascade with the V1-all → V0 UltraBERT fallback on unassigned rows. The frozen ZOD01 comparison applies its existing 18 fixed rules on the same held-out folds.

V0 was used earlier to select the checkpoints and ZOD01 specialists; results are development diagnostics, not an independent generalization estimate.

## Five-fold out-of-fold comparison

| Cascade | Correct | Accuracy | Balanced accuracy | Δ balanced accuracy vs ZOD01 |
|---|---:|---:|---:|---:|
| ZOD01 | 331/540 | 61.30% | 61.30% | — |
| ZOD02 | 283/540 | 52.41% | 52.41% | -8.89% |

ZOD02 corrected 47 examples ZOD01 missed and regressed 95 examples ZOD01 got right.

## Per-label OOF recall

| Label | ZOD01 recall | ZOD02 recall | Δ |
|---|---:|---:|---:|
| claude-fable-5 | 60.0% | 30.0% | -30.0% |
| claude-opus-4-6 | 96.7% | 90.0% | -6.7% |
| claude-opus-4-8 | 50.0% | 46.7% | -3.3% |
| claude-opus-5 | 56.7% | 80.0% | +23.3% |
| claude-sonnet-5 | 83.3% | 53.3% | -30.0% |
| gemini-3.1-pro-preview | 86.7% | 80.0% | -6.7% |
| gemini-3.6-flash | 83.3% | 63.3% | -20.0% |
| gemini-3.7-flash | 50.0% | 23.3% | -26.7% |
| gemini-3.8-flash | 56.7% | 63.3% | +6.7% |
| glm-5.3 | 33.3% | 10.0% | -23.3% |
| glm-5.3-flash | 40.0% | 40.0% | +0.0% |
| gpt-5.5 | 83.3% | 56.7% | -26.7% |
| gpt-5.6-luna | 50.0% | 50.0% | +0.0% |
| gpt-5.6-sol | 36.7% | 23.3% | -13.3% |
| gpt-5.6-terra | 56.7% | 33.3% | -23.3% |
| gpt-6-astra | 83.3% | 83.3% | +0.0% |
| kimi-k3 | 20.0% | 43.3% | +23.3% |
| qwen3.8-max | 76.7% | 73.3% | -3.3% |

## Full-V0 deployment rule list

The following 38 rules were selected by running the fitting procedure on all 540 V0 examples. This list is used for the Kaggle package and is exploratory; OOF metrics above use fold-specific rule lists.

| Order | Label | Checkpoint | Smoothed precision | Fit support | Assigned on full V0 | Incremental TP | Incremental FP | Δ fit balanced accuracy | Recall after rule |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | gpt-6-astra | `qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0` | 92.3% | 24 | 24 | 23 | 1 | +0.926% | 83.3% |
| 2 | claude-opus-4-6 | `gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0` | 83.3% | 28 | 28 | 24 | 4 | +0.185% | 96.7% |
| 3 | gemini-3.1-pro-preview | `gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0` | 82.8% | 27 | 27 | 23 | 4 | +0.185% | 96.7% |
| 4 | gpt-6-astra | `qwen06_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf24b2061_code667e2d0` | 81.2% | 30 | 7 | 3 | 4 | +0.185% | 86.7% |
| 5 | claude-opus-5 | `gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0` | 80.8% | 24 | 24 | 20 | 4 | +0.185% | 73.3% |
| 6 | claude-opus-4-8 | `gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0` | 77.8% | 16 | 16 | 13 | 3 | +0.185% | 56.7% |
| 7 | claude-opus-5 | `granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0` | 76.9% | 24 | 6 | 3 | 3 | +0.185% | 76.7% |
| 8 | qwen3.8-max | `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0` | 71.0% | 29 | 29 | 21 | 8 | +0.370% | 76.7% |
| 9 | gemini-3.8-flash | `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0` | 70.6% | 15 | 14 | 11 | 3 | +0.741% | 60.0% |
| 10 | gpt-5.5 | `ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0` | 68.2% | 20 | 18 | 13 | 5 | +0.370% | 80.0% |
| 11 | claude-opus-5 | `qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfg5d2d686c_code667e2d0` | 64.5% | 29 | 7 | 2 | 5 | +0.185% | 83.3% |
| 12 | gpt-5.6-luna | `qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0` | 62.5% | 14 | 14 | 9 | 5 | +0.926% | 50.0% |
| 13 | claude-opus-5 | `smollm3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg58d8423d_code667e2d0` | 61.9% | 19 | 6 | 2 | 4 | +0.370% | 90.0% |
| 14 | gemini-3.6-flash | `svc_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg50e77a40_code1b9bfc2` | 60.9% | 21 | 19 | 13 | 6 | +0.370% | 83.3% |
| 15 | glm-5.3-flash | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg3519c09b_code667e2d0` | 58.3% | 10 | 10 | 6 | 4 | +0.185% | 40.0% |
| 16 | gpt-5.6-sol | `qwen06_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf24b2061_code667e2d0` | 58.3% | 10 | 10 | 6 | 4 | +0.370% | 40.0% |
| 17 | gemini-3.7-flash | `svc_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg21d4b524_code1b9bfc2` | 57.9% | 17 | 14 | 9 | 5 | +0.741% | 46.7% |
| 18 | kimi-k3 | `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2` | 53.1% | 30 | 23 | 14 | 9 | +0.185% | 46.7% |
| 19 | gpt-5.6-luna | `gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0` | 52.6% | 17 | 7 | 4 | 3 | +0.370% | 56.7% |
| 20 | gpt-5.5 | `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2` | 52.6% | 17 | 7 | 3 | 4 | +0.185% | 70.0% |
| 21 | gemini-3.8-flash | `embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0` | 50.0% | 32 | 13 | 7 | 6 | +0.185% | 73.3% |
| 22 | glm-5.3-flash | `smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0` | 50.0% | 30 | 18 | 10 | 8 | +1.111% | 63.3% |
| 23 | kimi-k3 | `granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0` | 50.0% | 28 | 7 | 2 | 5 | +0.185% | 53.3% |
| 24 | gemini-3.6-flash | `ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0` | 50.0% | 28 | 11 | 8 | 3 | +0.370% | 80.0% |
| 25 | qwen3.8-max | `qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfg5d2d686c_code667e2d0` | 50.0% | 26 | 5 | 2 | 3 | +0.185% | 83.3% |
| 26 | gemini-3.7-flash | `embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0` | 50.0% | 20 | 6 | 4 | 2 | +0.185% | 50.0% |
| 27 | gpt-5.6-terra | `embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0` | 50.0% | 14 | 8 | 6 | 2 | +0.185% | 53.3% |
| 28 | gpt-5.6-luna | `granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0` | 48.5% | 31 | 9 | 6 | 3 | +0.185% | 63.3% |
| 29 | gpt-5.5 | `qwen06_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg4df85576_code667e2d0` | 48.4% | 29 | 5 | 4 | 1 | +0.370% | 76.7% |
| 30 | gpt-5.6-sol | `qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0` | 47.2% | 34 | 6 | 3 | 3 | +0.185% | 36.7% |
| 31 | gpt-5.6-sol | `qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0` | 45.8% | 22 | 3 | 1 | 2 | +0.185% | 40.0% |
| 32 | gemini-3.8-flash | `embeddinggemma2_v1_datada64f007_split147ce64e_seed2026_cfgdfb26ed6_code667e2d0` | 43.3% | 28 | 6 | 3 | 3 | +0.185% | 76.7% |
| 33 | gpt-5.6-terra | `smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0` | 43.2% | 35 | 6 | 4 | 2 | +0.185% | 53.3% |
| 34 | glm-5.3-flash | `catboost_stylometry_v1.5_data490fe052_split215fd0d0_seed2026_cfgdb877fc5_code1b9bfc2` | 42.1% | 17 | 3 | 1 | 2 | +0.185% | 66.7% |
| 35 | gpt-5.6-terra | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfge6177d6b_code667e2d0` | 40.0% | 23 | 6 | 3 | 3 | +0.185% | 56.7% |
| 36 | gpt-5.6-terra | `qwen06_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf24b2061_code667e2d0` | 38.6% | 42 | 4 | 4 | 0 | +0.185% | 60.0% |
| 37 | claude-fable-5 | `ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0` | 37.0% | 44 | 23 | 12 | 11 | +0.741% | 63.3% |
| 38 | gpt-5.6-sol | `granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0` | 36.8% | 17 | 2 | 1 | 1 | +0.185% | 43.3% |

Full-V0 fit balanced accuracy: 67.78%; this is a selection score, not an independent estimate.
Validation split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`. Artifacts are under `artifacts/analysis/zod02/`.
