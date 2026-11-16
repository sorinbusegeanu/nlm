# Accuracy-filtered Trust List 2 ensemble search

Trust List: `docs/trust-list-2-v025.csv` (`trust-list-2-v025`), SHA-256 `c93ae4cc0612ed36b7c126d6d50a5b50456ecfafd4e835369272076442e8cf9b`.
V0 contains 2,699 rows. ‘Per-label accuracy’ is the Trust List recall for that label: correct predictions divided by actual examples of that label.
Eligibility required a published full-V0 row, an eligible rule, an available checkpoint and no training/validation overlap.
Filtering and ensemble optimization both use full V0, so all scores and selections are development diagnostics and are selection-optimistic for Kaggle.

## Filtering

Eligible checkpoints before filtering: **78**. Retained pool: **26**. Threshold: best per-label recall minus **5.0%** (inclusive).

| Label | Best per-label accuracy | Cutoff | Retained checkpoints (accuracy) |
|---|---:|---:|---|
| claude-fable-5 | 53.7% | 48.7% | N0105 (53.7%), N0058 (53.0%), N0034 (49.7%) |
| claude-opus-4-6 | 85.3% | 80.3% | N0062 (85.3%), N0105 (83.3%), N0057 (82.7%), N0054 (81.3%), N0070 (80.7%) |
| claude-opus-4-8 | 81.3% | 76.3% | N0030 (81.3%) |
| claude-opus-5 | 84.7% | 79.7% | N0069 (84.7%) |
| claude-sonnet-5 | 75.3% | 70.3% | N0105 (75.3%) |
| gemini-3.1-pro-preview | 92.0% | 87.0% | N0062 (92.0%) |
| gemini-3.6-flash | 90.0% | 85.0% | N0081 (90.0%) |
| gemini-3.7-flash | 77.3% | 72.3% | N0065 (77.3%) |
| gemini-3.8-flash | 92.7% | 87.7% | N0070 (92.7%), N0075 (88.7%) |
| glm-5.3 | 48.0% | 43.0% | N0075 (48.0%), N0057 (45.3%), N0058 (44.0%), N0067 (44.0%) |
| glm-5.3-flash | 62.7% | 57.7% | N0061 (62.7%) |
| gpt-5.5 | 72.7% | 67.7% | N0105 (72.7%), N0081 (70.7%) |
| gpt-5.6-luna | 74.7% | 69.7% | N0106 (74.7%), N0057 (70.0%), N0074 (70.0%) |
| gpt-5.6-sol | 60.0% | 55.0% | N0062 (60.0%), N0077 (58.7%), N0075 (55.3%) |
| gpt-5.6-terra | 69.3% | 64.3% | N0038 (69.3%), N0030 (68.7%), N0079 (67.3%), N0081 (67.3%), N0076 (65.3%) |
| gpt-6-astra | 85.3% | 80.3% | N0081 (85.3%) |
| kimi-k3 | 61.3% | 56.3% | N0067 (61.3%), N0063 (58.0%), N0048 (56.7%), N0058 (56.7%) |
| qwen3.8-max | 78.0% | 73.0% | N0063 (78.0%), N0060 (76.7%), N0035 (75.3%), N0042 (75.3%), N0081 (75.3%), N0037 (74.7%), N0079 (74.0%) |

Candidate pool (26): `N0030, N0034, N0035, N0037, N0038, N0042, N0048, N0054, N0057, N0058, N0060, N0061, N0062, N0063, N0065, N0067, N0069, N0070, N0074, N0075, N0076, N0077, N0079, N0081, N0105, N0106`.

## Search coverage

Exact combinations for sizes 5–10: **10,952,370**. Unique combinations scored by the search: **25,501** (0.23% coverage). The full journal includes **32,086** unique scores, including singleton-to-four-member beam nodes and the EQUAL9 reference.
Method: deterministic beam search (width 256) with up to 12 local-search seeds per size and 5 rounds of one-checkpoint swaps. `evaluated_combinations.jsonl` and `search_state.json` make the run resumable.

## Best 10 overall

| Rank | Size | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 6 | N0030, N0037, N0048, N0070, N0081, N0105 | 63.8322% | 63.8385% | 1.011813 | -1.558 pp | 154 | 196 |
| 2 | 10 | N0030, N0034, N0037, N0048, N0061, N0070, N0074, N0077, N0081, N0105 | 63.7952% | 63.8014% | 1.028239 | -1.595 pp | 144 | 187 |
| 3 | 10 | N0030, N0034, N0058, N0061, N0065, N0070, N0074, N0077, N0081, N0105 | 63.7591% | 63.7644% | 1.028202 | -1.631 pp | 147 | 191 |
| 4 | 10 | N0030, N0034, N0037, N0058, N0061, N0065, N0070, N0077, N0081, N0105 | 63.7216% | 63.7273% | 1.025671 | -1.668 pp | 150 | 195 |
| 5 | 8 | N0030, N0034, N0037, N0048, N0070, N0077, N0081, N0105 | 63.6861% | 63.6903% | 1.034698 | -1.704 pp | 153 | 199 |
| 6 | 10 | N0030, N0034, N0037, N0057, N0058, N0070, N0076, N0077, N0081, N0105 | 63.6498% | 63.6532% | 1.020240 | -1.740 pp | 156 | 203 |
| 7 | 9 | N0030, N0034, N0037, N0048, N0058, N0070, N0077, N0081, N0105 | 63.6488% | 63.6532% | 1.026558 | -1.741 pp | 133 | 180 |
| 8 | 10 | N0030, N0034, N0035, N0037, N0058, N0061, N0070, N0076, N0081, N0105 | 63.6480% | 63.6532% | 1.018956 | -1.742 pp | 139 | 186 |
| 9 | 10 | N0030, N0034, N0037, N0048, N0058, N0061, N0065, N0070, N0081, N0105 | 63.6475% | 63.6532% | 1.010059 | -1.742 pp | 127 | 174 |
| 10 | 10 | N0030, N0034, N0057, N0058, N0061, N0065, N0070, N0077, N0081, N0105 | 63.6473% | 63.6532% | 1.028256 | -1.742 pp | 146 | 193 |

## Best 3 for each ensemble size

### 5 checkpoints

| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | N0030, N0058, N0070, N0074, N0081 | 63.6450% | 63.6532% | 1.015838 | -1.745 pp | 168 | 215 |
| 2 | N0030, N0061, N0070, N0081, N0105 | 63.4581% | 63.4680% | 1.011219 | -1.932 pp | 166 | 218 |
| 3 | N0030, N0034, N0061, N0081, N0105 | 63.3144% | 63.3197% | 1.029018 | -2.075 pp | 174 | 230 |

### 6 checkpoints

| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | N0030, N0037, N0048, N0070, N0081, N0105 | 63.8322% | 63.8385% | 1.011813 | -1.558 pp | 154 | 196 |
| 2 | N0030, N0057, N0058, N0065, N0070, N0081 | 63.5337% | 63.5421% | 1.019987 | -1.856 pp | 156 | 206 |
| 3 | N0030, N0034, N0057, N0061, N0081, N0105 | 63.4606% | 63.4680% | 1.027498 | -1.929 pp | 168 | 220 |

### 7 checkpoints

| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | N0030, N0037, N0058, N0070, N0074, N0081, N0105 | 63.5735% | 63.5791% | 1.002471 | -1.816 pp | 164 | 213 |
| 2 | N0030, N0057, N0058, N0065, N0070, N0074, N0081 | 63.5339% | 63.5421% | 1.018263 | -1.856 pp | 160 | 210 |
| 3 | N0030, N0037, N0048, N0061, N0070, N0081, N0105 | 63.4979% | 63.5050% | 1.008394 | -1.892 pp | 134 | 185 |

### 8 checkpoints

| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | N0030, N0034, N0037, N0048, N0070, N0077, N0081, N0105 | 63.6861% | 63.6903% | 1.034698 | -1.704 pp | 153 | 199 |
| 2 | N0030, N0037, N0057, N0058, N0070, N0076, N0081, N0105 | 63.6120% | 63.6162% | 1.000620 | -1.778 pp | 154 | 202 |
| 3 | N0030, N0034, N0037, N0058, N0070, N0074, N0081, N0105 | 63.6115% | 63.6162% | 1.006320 | -1.778 pp | 155 | 203 |

### 9 checkpoints

| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | N0030, N0034, N0037, N0048, N0058, N0070, N0077, N0081, N0105 | 63.6488% | 63.6532% | 1.026558 | -1.741 pp | 133 | 180 |
| 2 | N0030, N0034, N0037, N0057, N0070, N0076, N0077, N0081, N0105 | 63.6122% | 63.6162% | 1.028166 | -1.778 pp | 171 | 219 |
| 3 | N0030, N0034, N0037, N0058, N0070, N0076, N0077, N0081, N0105 | 63.5759% | 63.5791% | 1.021807 | -1.814 pp | 159 | 208 |

### 10 checkpoints

| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | N0030, N0034, N0037, N0048, N0061, N0070, N0074, N0077, N0081, N0105 | 63.7952% | 63.8014% | 1.028239 | -1.595 pp | 144 | 187 |
| 2 | N0030, N0034, N0058, N0061, N0065, N0070, N0074, N0077, N0081, N0105 | 63.7591% | 63.7644% | 1.028202 | -1.631 pp | 147 | 191 |
| 3 | N0030, N0034, N0037, N0058, N0061, N0065, N0070, N0077, N0081, N0105 | 63.7216% | 63.7273% | 1.025671 | -1.668 pp | 150 | 195 |

## Per-label recall and precision

The following table compares EQUAL9 with the best overall searched ensemble. Full per-label precision and recall for the best ten and each size's top three are in `per_label_metrics.csv` and `evaluation_metrics.json`.

| Label | EQUAL9 recall / precision | Best-search recall / precision | Δ recall / precision |
|---|---:|---:|---:|
| claude-fable-5 | 52.3% / 40.6% | 47.0% / 36.6% | -5.4 / -4.0 pp |
| claude-opus-4-6 | 80.0% / 85.7% | 81.3% / 83.6% | +1.3 / -2.2 pp |
| claude-opus-4-8 | 76.7% / 75.7% | 68.0% / 78.5% | -8.7 / +2.8 pp |
| claude-opus-5 | 78.7% / 90.8% | 78.7% / 79.2% | +0.0 / -11.6 pp |
| claude-sonnet-5 | 73.3% / 68.8% | 74.7% / 72.3% | +1.3 / +3.5 pp |
| gemini-3.1-pro-preview | 78.0% / 95.9% | 80.7% / 95.3% | +2.7 / -0.6 pp |
| gemini-3.6-flash | 80.7% / 63.7% | 80.7% / 67.2% | +0.0 / +3.5 pp |
| gemini-3.7-flash | 22.0% / 66.0% | 20.0% / 65.2% | -2.0 / -0.8 pp |
| gemini-3.8-flash | 83.3% / 50.0% | 85.3% / 49.2% | +2.0 / -0.8 pp |
| glm-5.3 | 45.3% / 39.8% | 36.0% / 32.9% | -9.3 / -6.8 pp |
| glm-5.3-flash | 48.7% / 52.1% | 39.3% / 41.8% | -9.3 / -10.3 pp |
| gpt-5.5 | 58.0% / 65.9% | 64.7% / 62.2% | +6.7 / -3.7 pp |
| gpt-5.6-luna | 61.3% / 71.9% | 52.0% / 79.6% | -9.3 / +7.7 pp |
| gpt-5.6-sol | 54.7% / 64.1% | 56.7% / 55.9% | +2.0 / -8.1 pp |
| gpt-5.6-terra | 77.3% / 56.0% | 73.3% / 57.9% | -4.0 / +1.9 pp |
| gpt-6-astra | 72.0% / 90.0% | 78.7% / 88.7% | +6.7 / -1.3 pp |
| kimi-k3 | 53.3% / 67.8% | 51.3% / 61.1% | -2.0 / -6.7 pp |
| qwen3.8-max | 81.3% / 72.2% | 80.7% / 78.1% | -0.7 / +5.9 pp |

## Prediction disagreement among the best three

Pairwise top-1 disagreement on full V0 for each size's best three. The full disagreement matrices and prediction artifacts are saved under the analysis output directory.

| Size | Ensemble pair | Rows different | Disagreement |
|---:|---|---:|---:|
| 5 | #1 vs #2 | 409 | 15.15% |
| 5 | #1 vs #3 | 587 | 21.75% |
| 5 | #2 vs #3 | 436 | 16.15% |
| 6 | #1 vs #2 | 492 | 18.23% |
| 6 | #1 vs #3 | 485 | 17.97% |
| 6 | #2 vs #3 | 517 | 19.16% |
| 7 | #1 vs #2 | 296 | 10.97% |
| 7 | #1 vs #3 | 334 | 12.37% |
| 7 | #2 vs #3 | 466 | 17.27% |
| 8 | #1 vs #2 | 360 | 13.34% |
| 8 | #1 vs #3 | 299 | 11.08% |
| 8 | #2 vs #3 | 283 | 10.49% |
| 9 | #1 vs #2 | 289 | 10.71% |
| 9 | #1 vs #3 | 236 | 8.74% |
| 9 | #2 vs #3 | 154 | 5.71% |
| 10 | #1 vs #2 | 259 | 9.60% |
| 10 | #1 vs #3 | 241 | 8.93% |
| 10 | #2 vs #3 | 182 | 6.74% |

## Kaggle candidates

The highest V0-scoring searched ensemble is `N0030, N0037, N0048, N0070, N0081, N0105` (63.8322% balanced accuracy). The current EQUAL9 baseline is 65.3898%.
Use the best-ranked configurations as candidates for later testing, while treating differences on this same V0 set as optimistic. Do not interpret V0 balanced accuracy as a reliable Kaggle score.

## Reproducibility artifacts

- `experiment_configs.json`: dataset/version, eligibility/filtering rules, exact selected checkpoint lists and equal weights, search parameters, cache paths and hashes.
- `evaluation_metrics.json`: baseline, best ten, best three per size and per-label metrics.
- `ensemble_predictions.parquet`: row-aligned predictions and probability vectors for baseline, best ten and per-size top three.
- `per_label_metrics.csv`: per-label precision/recall for all reported ensembles.
- `evaluated_combinations.jsonl`: resumable score journal.
- `search_state.json`: last completed beam size and search state.
