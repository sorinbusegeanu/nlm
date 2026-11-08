# UltraBERT Four-Checkpoint Ensemble Experiments

## Method and caveat

All ensemble results are five-fold prompt-group out-of-fold (OOF) diagnostics on the frozen V0 set (540 rows, 18 classes, 30 prompt groups). Every fold's calibration, fit weights, subset selection, and conditional-weight penalty used only its four fitting folds; folds were assigned with GroupKFold. Checkpoint and roster choices already used this V0 set, so these are development-set diagnostics, not independent generalization estimates. The four checkpoints' saved validation probabilities were reused; no model training or checkpoint loading occurred.

Split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`. Validation ID SHA-256: `d3f39a2003dfdd9e99e5d662465066bd60368f8e43864b6ed10a733edbe74f9d`. Class order is canonical sorted V0 order.

## A. Baseline checkpoints

| ID | Experiment | Training path | Balanced accuracy | Top-3 | MRR | Macro F1 | Log loss | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | `ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0` | V1 | 51.1% | 84.8% | 0.6940 | 51.8% | 2.7764 | 0.7932 |
| B | `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2` | V1-all | 52.2% | 85.4% | 0.6956 | 52.4% | 1.9111 | 0.6988 |
| C | `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0` | V1 → V0 | 53.1% | 83.7% | 0.6970 | 53.6% | 13.4997 | 0.9237 |
| D | `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff` | V1-all → V0 | 55.2% | 86.3% | 0.7166 | 54.2% | 2.5716 | 0.7288 |

Manifests and saved probability Parquets:

- **A** manifest `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0_manifest.json`; predictions `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0_validation.parquet`; checkpoint `artifacts/runs/ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0/checkpoints/ultrabert380_foldvalidation`.
- **B** manifest `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2_manifest.json`; predictions `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2_validation.parquet`; checkpoint `artifacts/runs/ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2/checkpoints/ultrabert380_foldvalidation`.
- **C** manifest `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_manifest.json`; predictions `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_validation.parquet`; checkpoint `artifacts/runs/ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0/checkpoints/ultrabert380_continued`.
- **D** manifest `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_manifest.json`; predictions `/home/zodrak/nlm/artifacts/predictions/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_validation.parquet`; checkpoint `artifacts/runs/ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff/checkpoints/ultrabert380_continued`.

## B. Experiment results

Metrics below are calculated from pooled OOF predictions; Δ columns are percentage-point differences from standalone D (55.2% balanced accuracy). Corrected/regressed counts compare OOF top-1 predictions with D's saved prediction for those same rows.

### Experiment 1 — B + D

| Fusion | Balanced acc | Δ vs D | Top-3 | MRR | Macro F1 | Log loss | Brier | Corrected / regressed vs D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E1_BD_equal` | 56.7% | +1.5 pp | 88.0% | 0.7265 | 56.3% | 1.6213 | 0.6220 | 28 / 20 |
| `E1_BD_weighted` | 55.2% | +0.0 pp | 87.4% | 0.7193 | 55.1% | 1.6128 | 0.6230 | 41 / 41 |
| `E1_BD_rank` | 55.6% | +0.4 pp | 86.5% | 0.7215 | 56.0% | 1.7234 | 0.7194 | 29 / 27 |

### Experiment 2 — C + D

| Fusion | Balanced acc | Δ vs D | Top-3 | MRR | Macro F1 | Log loss | Brier | Corrected / regressed vs D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E2_CD_equal` | 54.1% | -1.1 pp | 88.0% | 0.7164 | 54.5% | 2.4148 | 0.6769 | 66 / 72 |
| `E2_CD_weighted` | 54.8% | -0.4 pp | 88.1% | 0.7202 | 53.8% | 2.3037 | 0.6746 | 12 / 14 |
| `E2_CD_rank` | 54.8% | -0.4 pp | 87.2% | 0.7198 | 54.8% | 1.7224 | 0.7157 | 30 / 32 |
| `E2_CD_temp_equal` | 55.2% | +0.0 pp | 87.0% | 0.7225 | 54.2% | 1.5294 | 0.5879 | 21 / 21 |
| `E2_CD_temp_weighted` | 56.7% | +1.5 pp | 87.0% | 0.7299 | 55.7% | 1.5245 | 0.5819 | 31 / 23 |
| `E2_C_temp_only` | 53.1% | -2.0 pp | 83.7% | 0.6970 | 53.6% | 2.9854 | 0.6210 | 70 / 81 |

### Experiment 3 — A + B + C + D

| Fusion | Balanced acc | Δ vs D | Top-3 | MRR | Macro F1 | Log loss | Brier | Corrected / regressed vs D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E3_ABCD_equal` | 55.0% | -0.2 pp | 89.4% | 0.7223 | 55.4% | 1.5418 | 0.6176 | 54 / 55 |
| `E3_ABCD_weighted` | 55.9% | +0.7 pp | 89.1% | 0.7285 | 56.2% | 1.4891 | 0.6039 | 41 / 37 |
| `E3_ABCD_rank` | 55.4% | +0.2 pp | 88.9% | 0.7267 | 56.0% | 1.7138 | 0.7170 | 44 / 43 |
| `E3_forward_weighted` | 55.2% | +0.0 pp | 87.4% | 0.7193 | 55.1% | 1.6128 | 0.6230 | 41 / 41 |

### Experiment 4 — class/family weights

| Fusion | Balanced acc | Δ vs D | Top-3 | MRR | Macro F1 | Log loss | Brier | Corrected / regressed vs D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E4_best_global_cv` | 55.2% | +0.0 pp | 87.4% | 0.7212 | 54.7% | 1.5649 | 0.5990 | 30 / 30 |
| `E4_family_regularized` | 55.7% | +0.6 pp | 88.9% | 0.7260 | 55.8% | 1.4937 | 0.6049 | 42 / 39 |
| `E4_class_regularized` | 55.0% | -0.2 pp | 88.9% | 0.7243 | 54.9% | 1.4810 | 0.6027 | 45 / 46 |

### Fold-selected subsets, weights and calibration

Global blend weights sum to one and were fit with `fit_weights` (multiclass log loss plus L2 shrinkage toward equal weights, λ=0.05). Rank fusion averages reciprocal class ranks and normalizes scores. For C+D calibration, only C is temperature-scaled; each temperature is selected from a fixed grid using inner prompt-group OOF loss within the outer fitting fold.

C-only calibration: log loss 13.4997 → 2.9854; Brier 0.9237 → 0.6210; top-1 balanced accuracy 53.1% → 53.1%. Fold temperatures: `[32.0, 32.0, 32.0, 32.0, 32.0]`. The same fold-fitted temperatures are then used in the C+D blends.

Per-class top-1 recall changes relative to D (OOF). This shows the pair effects for the named recoveries and all other labels:

| Model label | D | B + D equal | Δ vs D | C + D equal | Δ vs D | C + D temp-weighted | Δ vs D | A + B + C + D weighted | Δ vs D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-fable-5 | 53.3% | 43.3% | -10.0 pp | 30.0% | -23.3 pp | 53.3% | +0.0 pp | 43.3% | -10.0 pp |
| claude-opus-4-6 | 86.7% | 83.3% | -3.3 pp | 73.3% | -13.3 pp | 86.7% | +0.0 pp | 73.3% | -13.3 pp |
| claude-opus-4-8 | 53.3% | 56.7% | +3.3 pp | 60.0% | +6.7 pp | 60.0% | +6.7 pp | 63.3% | +10.0 pp |
| claude-opus-5 | 66.7% | 70.0% | +3.3 pp | 70.0% | +3.3 pp | 66.7% | +0.0 pp | 70.0% | +3.3 pp |
| claude-sonnet-5 | 86.7% | 80.0% | -6.7 pp | 56.7% | -30.0 pp | 86.7% | +0.0 pp | 76.7% | -10.0 pp |
| gemini-3.1-pro-preview | 90.0% | 86.7% | -3.3 pp | 83.3% | -6.7 pp | 90.0% | +0.0 pp | 80.0% | -10.0 pp |
| gemini-3.6-flash | 76.7% | 70.0% | -6.7 pp | 50.0% | -26.7 pp | 66.7% | -10.0 pp | 66.7% | -10.0 pp |
| gemini-3.7-flash | 26.7% | 36.7% | +10.0 pp | 70.0% | +43.3 pp | 46.7% | +20.0 pp | 40.0% | +13.3 pp |
| gemini-3.8-flash | 46.7% | 50.0% | +3.3 pp | 40.0% | -6.7 pp | 40.0% | -6.7 pp | 53.3% | +6.7 pp |
| glm-5.3 | 23.3% | 16.7% | -6.7 pp | 26.7% | +3.3 pp | 13.3% | -10.0 pp | 20.0% | -3.3 pp |
| glm-5.3-flash | 30.0% | 36.7% | +6.7 pp | 46.7% | +16.7 pp | 43.3% | +13.3 pp | 40.0% | +10.0 pp |
| gpt-5.5 | 76.7% | 73.3% | -3.3 pp | 50.0% | -26.7 pp | 73.3% | -3.3 pp | 50.0% | -26.7 pp |
| gpt-5.6-luna | 36.7% | 36.7% | +0.0 pp | 40.0% | +3.3 pp | 40.0% | +3.3 pp | 36.7% | +0.0 pp |
| gpt-5.6-sol | 36.7% | 36.7% | +0.0 pp | 36.7% | +0.0 pp | 30.0% | -6.7 pp | 46.7% | +10.0 pp |
| gpt-5.6-terra | 46.7% | 60.0% | +13.3 pp | 53.3% | +6.7 pp | 53.3% | +6.7 pp | 56.7% | +10.0 pp |
| gpt-6-astra | 63.3% | 70.0% | +6.7 pp | 76.7% | +13.3 pp | 80.0% | +16.7 pp | 73.3% | +10.0 pp |
| kimi-k3 | 26.7% | 43.3% | +16.7 pp | 36.7% | +10.0 pp | 26.7% | +0.0 pp | 46.7% | +20.0 pp |
| qwen3.8-max | 66.7% | 70.0% | +3.3 pp | 73.3% | +6.7 pp | 63.3% | -3.3 pp | 70.0% | +3.3 pp |
**E1_BD_weighted:**
- fold 1: members B, D; weights B=0.585, D=0.415.
- fold 2: members B, D; weights B=0.614, D=0.386.
- fold 3: members B, D; weights B=0.621, D=0.379.
- fold 4: members B, D; weights B=0.605, D=0.395.
- fold 5: members B, D; weights B=0.598, D=0.402.
  Fold mean ± SD: B=0.605±0.013, D=0.395±0.013.
**E2_CD_weighted:**
- fold 1: members C, D; weights C=0.176, D=0.824.
- fold 2: members C, D; weights C=0.205, D=0.795.
- fold 3: members C, D; weights C=0.178, D=0.822.
- fold 4: members C, D; weights C=0.204, D=0.796.
- fold 5: members C, D; weights C=0.206, D=0.794.
  Fold mean ± SD: C=0.194±0.014, D=0.806±0.014.
**E2_CD_temp_weighted:**
- fold 1: members C, D; weights C=0.566, D=0.434; T(C)=32.
- fold 2: members C, D; weights C=0.581, D=0.419; T(C)=32.
- fold 3: members C, D; weights C=0.584, D=0.416; T(C)=32.
- fold 4: members C, D; weights C=0.567, D=0.433; T(C)=32.
- fold 5: members C, D; weights C=0.586, D=0.414; T(C)=32.
  Fold mean ± SD: C=0.577±0.009, D=0.423±0.009.
**E3_ABCD_weighted:**
- fold 1: members A, B, C, D; weights A=0.211, B=0.435, C=0.034, D=0.320.
- fold 2: members A, B, C, D; weights A=0.114, B=0.485, C=0.100, D=0.301.
- fold 3: members A, B, C, D; weights A=0.177, B=0.458, C=0.069, D=0.296.
- fold 4: members A, B, C, D; weights A=0.201, B=0.440, C=0.058, D=0.301.
- fold 5: members A, B, C, D; weights A=0.184, B=0.442, C=0.084, D=0.290.
  Fold mean ± SD: A=0.178±0.034, B=0.452±0.018, C=0.069±0.023, D=0.302±0.010.
**E3_forward_weighted:**
- fold 1: members D, B; weights D=0.415, B=0.585.
- fold 2: members D, B; weights D=0.386, B=0.614.
- fold 3: members D, B; weights D=0.379, B=0.621.
- fold 4: members D, B; weights D=0.395, B=0.605.
- fold 5: members D, B; weights D=0.402, B=0.598.
  Fold mean ± SD: D=0.395±0.013, B=0.605±0.013.
**E4_best_global_cv:**
- fold 1: members C, D; weights C=0.566, D=0.434; T(C)=32; inner-CV choice=CD_temp_weighted.
- fold 2: members A, B, C, D; weights A=0.114, B=0.485, C=0.100, D=0.301; inner-CV choice=ABCD_weighted.
- fold 3: members C, D; weights C=0.584, D=0.416; T(C)=32; inner-CV choice=CD_temp_weighted.
- fold 4: members C, D; weights C=0.567, D=0.433; T(C)=32; inner-CV choice=CD_temp_weighted.
- fold 5: members A, B, C, D; weights A=0.184, B=0.442, C=0.084, D=0.290; inner-CV choice=ABCD_weighted.
  Fold mean ± SD for members C,D: C=0.572±0.008, D=0.428±0.008.
  Fold mean ± SD for members A,B,C,D: A=0.149±0.035, B=0.463±0.021, C=0.092±0.008, D=0.296±0.006.

Family-specific checkpoint weights by fold (the six family weights are independently normalized across A–D):
- Fold 1, shrinkage λ=1.0: Claude: A=0.122, B=0.449, C=0.011, D=0.417; GLM: A=0.269, B=0.428, C=0.000, D=0.303; GPT: A=0.259, B=0.448, C=0.015, D=0.278; Gemini: A=0.156, B=0.420, C=0.127, D=0.297; Kimi: A=0.140, B=0.610, C=0.000, D=0.250; Qwen: A=0.297, B=0.310, C=0.071, D=0.322.
- Fold 2, shrinkage λ=1.0: Claude: A=0.120, B=0.467, C=0.053, D=0.360; GLM: A=0.169, B=0.470, C=0.106, D=0.254; GPT: A=0.132, B=0.563, C=0.057, D=0.248; Gemini: A=0.051, B=0.462, C=0.196, D=0.291; Kimi: A=0.000, B=0.641, C=0.105, D=0.254; Qwen: A=0.175, B=0.335, C=0.120, D=0.370.
- Fold 3, shrinkage λ=1.0: Claude: A=0.137, B=0.470, C=0.047, D=0.346; GLM: A=0.232, B=0.423, C=0.114, D=0.231; GPT: A=0.237, B=0.521, C=0.019, D=0.224; Gemini: A=0.105, B=0.442, C=0.120, D=0.332; Kimi: A=0.124, B=0.571, C=0.000, D=0.305; Qwen: A=0.169, B=0.260, C=0.206, D=0.365.
- Fold 4, shrinkage λ=1.0: Claude: A=0.163, B=0.442, C=0.007, D=0.388; GLM: A=0.166, B=0.427, C=0.147, D=0.260; GPT: A=0.328, B=0.473, C=0.000, D=0.199; Gemini: A=0.101, B=0.438, C=0.146, D=0.315; Kimi: A=0.144, B=0.593, C=0.000, D=0.263; Qwen: A=0.251, B=0.316, C=0.054, D=0.379.
- Fold 5, shrinkage λ=1.0: Claude: A=0.078, B=0.479, C=0.048, D=0.396; GLM: A=0.154, B=0.424, C=0.160, D=0.261; GPT: A=0.279, B=0.464, C=0.033, D=0.224; Gemini: A=0.158, B=0.406, C=0.174, D=0.262; Kimi: A=0.073, B=0.619, C=0.000, D=0.308; Qwen: A=0.245, B=0.283, C=0.131, D=0.341.

Per-class checkpoint weights vary across labels. Across all fold × label weights, mean ± SD by checkpoint:
- A=0.175±0.108, B=0.457±0.125, C=0.070±0.075, D=0.299±0.100. Full per-label matrices are in `fold_weights.json`.

Forward-selection step traces (inner OOF, calculated independently within each outer training fold):
- Fold 1: D (55.8%) → B (56.9%).
- Fold 2: D (55.6%) → B (57.9%).
- Fold 3: D (54.4%) → B (55.3%).
- Fold 4: D (55.3%) → B (56.0%).
- Fold 5: D (54.9%) → B (57.2%).

## C. Per-class top-1 recall

Best pooled-OOF method among evaluated candidates: `E1_BD_equal`. This descriptive selection is itself based on the same OOF comparison and is exploratory. Positive or negative deltas in the last column are relative to checkpoint D.

| Model label | A | B | C | D | Best ensemble |
|---|---:|---:|---:|---:|---:|
| claude-fable-5 | 26.7% | 30.0% | 23.3% | 53.3% | **43.3% (-10.0 pp)** |
| claude-opus-4-6 | 73.3% | 56.7% | 73.3% | 86.7% | **83.3% (-3.3 pp)** |
| claude-opus-4-8 | 50.0% | 70.0% | 60.0% | 53.3% | **56.7% (+3.3 pp)** |
| claude-opus-5 | 63.3% | 66.7% | 70.0% | 66.7% | **70.0% (+3.3 pp)** |
| claude-sonnet-5 | 60.0% | 53.3% | 53.3% | 86.7% | **80.0% (-6.7 pp)** |
| gemini-3.1-pro-preview | 73.3% | 73.3% | 80.0% | 90.0% | **86.7% (-3.3 pp)** |
| gemini-3.6-flash | 46.7% | 70.0% | 43.3% | 76.7% | **70.0% (-6.7 pp)** |
| gemini-3.7-flash | 46.7% | 33.3% | 73.3% | 26.7% | **36.7% (+10.0 pp)** |
| gemini-3.8-flash | 56.7% | 63.3% | 36.7% | 46.7% | **50.0% (+3.3 pp)** |
| glm-5.3 | 26.7% | 23.3% | 30.0% | 23.3% | **16.7% (-6.7 pp)** |
| glm-5.3-flash | 33.3% | 53.3% | 50.0% | 30.0% | **36.7% (+6.7 pp)** |
| gpt-5.5 | 40.0% | 30.0% | 50.0% | 76.7% | **73.3% (-3.3 pp)** |
| gpt-5.6-luna | 36.7% | 16.7% | 40.0% | 36.7% | **36.7% (+0.0 pp)** |
| gpt-5.6-sol | 50.0% | 46.7% | 36.7% | 36.7% | **36.7% (+0.0 pp)** |
| gpt-5.6-terra | 53.3% | 66.7% | 53.3% | 46.7% | **60.0% (+13.3 pp)** |
| gpt-6-astra | 76.7% | 70.0% | 76.7% | 63.3% | **70.0% (+6.7 pp)** |
| kimi-k3 | 43.3% | 53.3% | 36.7% | 26.7% | **43.3% (+16.7 pp)** |
| qwen3.8-max | 63.3% | 63.3% | 70.0% | 66.7% | **70.0% (+3.3 pp)** |

Per-class top-1 precision for the same baselines and selected descriptive ensemble:

| Model label | A | B | C | D | Best ensemble |
|---|---:|---:|---:|---:|---:|
| claude-fable-5 | 47.1% | 23.7% | 26.9% | 43.2% | **33.3%** |
| claude-opus-4-6 | 75.9% | 85.0% | 78.6% | 72.2% | **71.4%** |
| claude-opus-4-8 | 57.7% | 56.8% | 54.5% | 84.2% | **89.5%** |
| claude-opus-5 | 76.0% | 71.4% | 75.0% | 76.9% | **75.0%** |
| claude-sonnet-5 | 58.1% | 76.2% | 61.5% | 59.1% | **63.2%** |
| gemini-3.1-pro-preview | 91.7% | 88.0% | 96.0% | 77.1% | **76.5%** |
| gemini-3.6-flash | 50.0% | 51.2% | 54.2% | 53.5% | **56.8%** |
| gemini-3.7-flash | 35.9% | 47.6% | 40.0% | 44.4% | **47.8%** |
| gemini-3.8-flash | 58.6% | 55.9% | 73.3% | 58.3% | **57.7%** |
| glm-5.3 | 24.2% | 24.1% | 26.5% | 28.0% | **23.8%** |
| glm-5.3-flash | 25.0% | 32.7% | 42.9% | 34.6% | **29.7%** |
| gpt-5.5 | 50.0% | 52.9% | 53.6% | 41.8% | **52.4%** |
| gpt-5.6-luna | 39.3% | 33.3% | 40.0% | 34.4% | **42.3%** |
| gpt-5.6-sol | 45.5% | 40.0% | 37.9% | 55.0% | **50.0%** |
| gpt-5.6-terra | 44.4% | 41.7% | 40.0% | 43.8% | **43.9%** |
| gpt-6-astra | 92.0% | 95.5% | 92.0% | 86.4% | **91.3%** |
| kimi-k3 | 30.2% | 53.3% | 36.7% | 57.1% | **68.4%** |
| qwen3.8-max | 63.3% | 63.3% | 72.4% | 62.5% | **70.0%** |

For `E1_BD_equal`, OOF examples corrected relative to D: 28; regressed: 20.

Per-class precision and recall for every strategy are saved in `results.json`.

## D. Complementarity

| Checkpoint pair | Shared errors | Left-only correct | Right-only correct | Pair oracle coverage |
|---|---:|---:|---:|---:|
| A + B | 193 | 65 | 71 | 64.3% |
| A + C | 219 | 34 | 45 | 59.4% |
| A + D | 185 | 57 | 79 | 65.7% |
| B + C | 180 | 73 | 78 | 66.7% |
| B + D | 180 | 62 | 78 | 66.7% |
| C + D | 172 | 70 | 81 | 68.1% |

Four-checkpoint oracle coverage (theoretical, picks a correct checkpoint with hindsight): **76.5%**. This is not an achievable ensemble score. Pairwise coverage is likewise an upper bound.

Family confusion movement for the selected OOF method versus D (rows = true family; columns = predicted family):

**D**

| True \ Predicted | Claude | Gemini | GLM | GPT | Kimi | Qwen |
|---|---:|---:|---:|---:|---:|---:|
| Claude | 130 | 1 | 15 | 0 | 2 | 2 |
| Gemini | 1 | 117 | 0 | 0 | 0 | 2 |
| GLM | 27 | 0 | 31 | 1 | 1 | 0 |
| GPT | 0 | 0 | 1 | 141 | 2 | 6 |
| Kimi | 2 | 0 | 4 | 14 | 8 | 2 |
| Qwen | 2 | 2 | 0 | 5 | 1 | 20 |

**E1_BD_equal**

| True \ Predicted | Claude | Gemini | GLM | GPT | Kimi | Qwen |
|---|---:|---:|---:|---:|---:|---:|
| Claude | 129 | 1 | 17 | 0 | 2 | 1 |
| Gemini | 1 | 117 | 0 | 0 | 0 | 2 |
| GLM | 23 | 0 | 36 | 1 | 0 | 0 |
| GPT | 1 | 0 | 1 | 138 | 4 | 6 |
| Kimi | 2 | 0 | 4 | 11 | 13 | 0 |
| Qwen | 3 | 2 | 0 | 4 | 0 | 21 |

Change in family confusion counts (selected method minus D):

| True \ Predicted | Claude | Gemini | GLM | GPT | Kimi | Qwen |
|---|---:|---:|---:|---:|---:|---:|
| Claude | -1 | +0 | +2 | +0 | +0 | -1 |
| Gemini | +0 | +0 | +0 | +0 | +0 | +0 |
| GLM | -4 | +0 | +5 | +0 | -1 | +0 |
| GPT | +1 | +0 | +0 | -3 | +2 | +0 |
| Kimi | +0 | +0 | +0 | -3 | +5 | -2 |
| Qwen | +1 | +0 | +0 | -1 | -1 | +1 |

## E. Comparison and interpretation

| Rank | Strategy | Grouped OOF balanced accuracy | Top-3 | MRR | Macro F1 | Log loss | Brier |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `E1_BD_equal` | 56.7% | 88.0% | 0.7265 | 56.3% | 1.6213 | 0.6220 |
| 2 | `E2_CD_temp_weighted` | 56.7% | 87.0% | 0.7299 | 55.7% | 1.5245 | 0.5819 |
| 3 | `E3_ABCD_weighted` | 55.9% | 89.1% | 0.7285 | 56.2% | 1.4891 | 0.6039 |
| 4 | `E4_family_regularized` | 55.7% | 88.9% | 0.7260 | 55.8% | 1.4937 | 0.6049 |
| 5 | `E1_BD_rank` | 55.6% | 86.5% | 0.7215 | 56.0% | 1.7234 | 0.7194 |
| 6 | `E3_ABCD_rank` | 55.4% | 88.9% | 0.7267 | 56.0% | 1.7138 | 0.7170 |
| 7 | `E4_best_global_cv` | 55.2% | 87.4% | 0.7212 | 54.7% | 1.5649 | 0.5990 |
| 8 | `E1_BD_weighted` | 55.2% | 87.4% | 0.7193 | 55.1% | 1.6128 | 0.6230 |
| 9 | `E3_forward_weighted` | 55.2% | 87.4% | 0.7193 | 55.1% | 1.6128 | 0.6230 |
| 10 | `E2_CD_temp_equal` | 55.2% | 87.0% | 0.7225 | 54.2% | 1.5294 | 0.5879 |
| 11 | `E3_ABCD_equal` | 55.0% | 89.4% | 0.7223 | 55.4% | 1.5418 | 0.6176 |
| 12 | `E4_class_regularized` | 55.0% | 88.9% | 0.7243 | 54.9% | 1.4810 | 0.6027 |
| 13 | `E2_CD_weighted` | 54.8% | 88.1% | 0.7202 | 53.8% | 2.3037 | 0.6746 |
| 14 | `E2_CD_rank` | 54.8% | 87.2% | 0.7198 | 54.8% | 1.7224 | 0.7157 |
| 15 | `E2_CD_equal` | 54.1% | 88.0% | 0.7164 | 54.5% | 2.4148 | 0.6769 |
| 16 | `E2_C_temp_only` | 53.1% | 83.7% | 0.6970 | 53.6% | 2.9854 | 0.6210 |

Family weights use a small inner-fold search over shrinkage penalties `[0.05, 0.25, 1.0]`; class-specific weights use `[0.25, 1.0, 4.0]`. Both optimize normalized multiclass scores, so false positives are penalized as well as missed true labels. Per-class weights have 18 separate 4-checkpoint simplexes with only 30 rows per target label; treat those as high-variance diagnostics. A family with one class (Kimi or Qwen) is statistically equivalent to class-specific weights for that class.

The top accuracy ties at 56.7%: B+D equal (higher top-3) and C+D temperature-calibrated weighted (higher MRR and better log loss/Brier). Family-specific weighting scores 55.7%, below the strongest global-weight strategy at 56.7%; it only beats the inner-fold-selected global strategy (55.2%).

## F. Conclusions

1. **Can two checkpoints beat the best standalone?** Yes on grouped OOF diagnostics; best pair score = 56.7% versus D 55.2%.
2. **Do four checkpoints outperform two?** No by the best grouped OOF scores; pooled OOF strategy ranking is selection-biased and should be treated as exploratory.
3. **Does family-specific weighting outperform the strongest global weighting?** No. It scores 55.7%, versus 56.7% for the strongest global-weight experiment. It does beat the per-fold inner-selected global strategy (55.2%).
4. **Which labels benefit/deteriorate?** See the per-class table; `results.json` records the full precision/recall and each strategy's class-level movement.
5. **Candidate for a future multi-architecture ensemble:** `E1_BD_equal` is a reasonable candidate because it ties for top accuracy with two members and has the stronger top-3 score. `E2_CD_temp_weighted` ties on accuracy and has better MRR/log loss/Brier, but depends on calibrating C. Confirm candidates on an untouched evaluation set before replacing the current multi-model ensemble.

Artifacts: OOF predictions `artifacts/analysis/ultrabert_four_checkpoint/oof_predictions.parquet`; fold assignments and weights are saved beside `results.json`. Checkpoint candidates and weights do not modify `docs/ensembles.md` or the active roster.
