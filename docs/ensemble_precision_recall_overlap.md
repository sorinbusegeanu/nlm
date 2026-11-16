# Precision, recall and overlap-based ensemble optimization

Trust List 2: `trust-list-2-v025` (`docs/trust-list-2-v025.csv`), SHA-256 `c93ae4cc0612ed36b7c126d6d50a5b50456ecfafd4e835369272076442e8cf9b`. V0: 2,699 rows, 18 classes, 150 prompt groups.
All prediction matrices were loaded from existing full-V0 artifacts. Row IDs and true labels were checked against V0; prompt IDs were also checked when present, and otherwise attached from V0 by exact row-ID alignment. The full-V0 diagnostic selects and scores on the same rows; only the prompt-group cross-fitted results are held out within this analysis. Checkpoint development/selection may still have used V0, so neither scope is an independent Kaggle estimate.

## Candidate selection

Eligible checkpoint count before filtering: **78**. Filtered candidate union: **41**. The candidate set includes each label's tied best-F1, best-precision and best-recall checkpoints, all candidates within 5 percentage points of best F1, and every EQUAL9 member.

Per-checkpoint counts and metrics are in `per_label_checkpoint_metrics.csv`. The exact top-1 prediction overlap of all 1–3 member combinations per label is in `label_overlap_combinations.csv`; columns include shared/unique TP and FP plus incremental contributions.

## Ensemble comparison

| Scope | Strategy | Accuracy | Balanced accuracy | Log loss | Corrected vs EQUAL9 | Introduced vs EQUAL9 | Disagreements |
|---|---|---:|---:|---:|---:|---:|---:|
| Prompt-group OOF | equal9 | 65.3946% | 65.3898% | 1.132076 | 0 | 0 | 0 |
| Prompt-group OOF | equal_selected | 64.8018% | 64.7962% | 1.076578 | 111 | 127 | 344 |
| Prompt-group OOF | per_label_f1 | 63.2086% | 63.2021% | 1.119479 | 222 | 281 | 707 |
| Prompt-group OOF | complementarity | 62.5787% | 62.5722% | 1.175583 | 237 | 313 | 796 |
| Full-V0 diagnostic | equal9 | 65.3946% | 65.3898% | 1.132076 | 0 | 0 | 0 |
| Full-V0 diagnostic | equal_selected | 65.0611% | 65.0562% | 1.074353 | 120 | 129 | 350 |
| Full-V0 diagnostic | per_label_f1 | 65.5428% | 65.5364% | 1.028626 | 265 | 261 | 723 |
| Full-V0 diagnostic | complementarity | 64.3201% | 64.3137% | 1.088983 | 264 | 293 | 800 |

EQUAL9 full-V0 reference: **65.3898% balanced accuracy**, 65.3946% accuracy, log loss 1.132076.

## Best label-specific rules and overlap

The following are the best full-V0 label rules by union F1 (diagnostic only). Use `label_overlap_combinations.csv` for all combinations and fold-specific selections. Strategy-specific per-label precision, recall and F1 for both scopes are in `per_label_ensemble_metrics.csv`.

| Label | Checkpoints | TP | FP | Precision | Recall | F1 | Shared TP/FP | Unique TP/FP | Incremental TP/FP by rule |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| claude-fable-5 | N0003, N0006, N0009 | 82 | 126 | 39.4% | 55.0% | 45.9% | 57/63 | 25/63 | N0009 +61/+77; N0006 +16/+38; N0003 +5/+11 |
| claude-opus-4-6 | N0034, N0048, N0055 | 130 | 44 | 74.7% | 86.7% | 80.2% | 101/9 | 29/35 | N0055 +115/+29; N0034 +13/+15; N0048 +2/+0 |
| claude-opus-4-8 | N0034, N0081 | 106 | 39 | 73.1% | 70.7% | 71.9% | 51/2 | 55/37 | N0081 +97/+28; N0034 +9/+11 |
| claude-opus-5 | N0030, N0055, N0061 | 128 | 39 | 76.6% | 85.3% | 80.8% | 97/10 | 31/29 | N0030 +109/+26; N0055 +14/+11; N0061 +5/+2 |
| claude-sonnet-5 | N0061, N0076 | 108 | 55 | 66.3% | 72.0% | 69.0% | 66/13 | 42/42 | N0076 +97/+41; N0061 +11/+14 |
| gemini-3.1-pro-preview | N0058, N0081, N0105 | 135 | 14 | 90.6% | 90.0% | 90.3% | 114/4 | 21/10 | N0105 +120/+10; N0081 +10/+1; N0058 +5/+3 |
| gemini-3.6-flash | N0105 | 109 | 46 | 70.3% | 72.7% | 71.5% | 0/0 | 109/46 | N0105 +109/+46 |
| gemini-3.7-flash | N0030, N0055, N0056 | 126 | 181 | 41.0% | 84.0% | 55.1% | 64/69 | 62/112 | N0030 +82/+104; N0056 +27/+49; N0055 +17/+28 |
| gemini-3.8-flash | N0030, N0048 | 120 | 114 | 51.3% | 80.0% | 62.5% | 62/33 | 58/81 | N0048 +94/+77; N0030 +26/+37 |
| glm-5.3 | N0009, N0030, N0080 | 98 | 177 | 35.6% | 65.3% | 46.1% | 39/27 | 59/150 | N0030 +56/+90; N0080 +25/+64; N0009 +17/+23 |
| glm-5.3-flash | N0022, N0030, N0058 | 82 | 75 | 52.2% | 54.7% | 53.4% | 22/6 | 60/69 | N0022 +58/+30; N0058 +22/+41; N0030 +2/+4 |
| gpt-5.5 | N0055, N0081 | 117 | 104 | 52.9% | 78.0% | 63.1% | 69/20 | 48/84 | N0081 +106/+82; N0055 +11/+22 |
| gpt-5.6-luna | N0009, N0071 | 110 | 87 | 55.8% | 73.3% | 63.4% | 38/6 | 72/81 | N0071 +97/+64; N0009 +13/+23 |
| gpt-5.6-sol | N0009, N0061, N0081 | 102 | 131 | 43.8% | 68.0% | 53.3% | 40/21 | 62/110 | N0081 +78/+88; N0009 +16/+29; N0061 +8/+14 |
| gpt-5.6-terra | N0081 | 101 | 90 | 52.9% | 67.3% | 59.2% | 0/0 | 101/90 | N0081 +101/+90 |
| gpt-6-astra | N0048, N0081 | 132 | 29 | 82.0% | 88.0% | 84.9% | 80/4 | 52/25 | N0081 +128/+24; N0048 +4/+5 |
| kimi-k3 | N0057, N0058 | 98 | 117 | 45.6% | 65.3% | 53.7% | 59/27 | 39/90 | N0058 +85/+96; N0057 +13/+21 |
| qwen3.8-max | N0081, N0105 | 121 | 41 | 74.7% | 80.7% | 77.6% | 74/5 | 47/36 | N0081 +113/+29; N0105 +8/+12 |

## Per-label cross-fitted precision, recall and F1

Each cell is precision / recall / F1 from pooled held-out prompt groups. Full-V0 diagnostic per-label metrics are in `per_label_ensemble_metrics.csv`.

| Label | EQUAL9 | Equal selected | Per-label F1 | Complementarity |
|---|---|---|---|---|
| claude-fable-5 | 40.6% / 52.3% / 45.7% | 35.9% / 49.7% / 41.7% | 42.5% / 45.6% / 44.0% | 39.6% / 45.0% / 42.1% |
| claude-opus-4-6 | 85.7% / 80.0% / 82.8% | 83.8% / 82.7% / 83.2% | 81.0% / 79.3% / 80.1% | 77.5% / 78.0% / 77.7% |
| claude-opus-4-8 | 75.7% / 76.7% / 76.2% | 81.3% / 66.7% / 73.3% | 65.4% / 68.0% / 66.7% | 67.5% / 68.0% / 67.8% |
| claude-opus-5 | 90.8% / 78.7% / 84.3% | 87.2% / 77.3% / 82.0% | 85.2% / 76.7% / 80.7% | 78.3% / 74.7% / 76.5% |
| claude-sonnet-5 | 68.8% / 73.3% / 71.0% | 68.5% / 74.0% / 71.2% | 62.7% / 67.3% / 65.0% | 66.7% / 66.7% / 66.7% |
| gemini-3.1-pro-preview | 95.9% / 78.0% / 86.0% | 93.8% / 81.3% / 87.1% | 94.6% / 81.3% / 87.5% | 91.7% / 80.7% / 85.8% |
| gemini-3.6-flash | 63.7% / 80.7% / 71.2% | 67.0% / 78.7% / 72.4% | 66.5% / 72.7% / 69.4% | 66.5% / 74.0% / 70.0% |
| gemini-3.7-flash | 66.0% / 22.0% / 33.0% | 61.2% / 20.0% / 30.2% | 50.8% / 44.7% / 47.5% | 52.0% / 44.0% / 47.7% |
| gemini-3.8-flash | 50.0% / 83.3% / 62.5% | 48.8% / 84.7% / 62.0% | 55.1% / 68.0% / 60.9% | 54.2% / 68.7% / 60.6% |
| glm-5.3 | 39.8% / 45.3% / 42.4% | 38.3% / 39.3% / 38.8% | 41.7% / 46.7% / 44.0% | 37.3% / 50.0% / 42.7% |
| glm-5.3-flash | 52.1% / 48.7% / 50.3% | 47.0% / 47.3% / 47.2% | 51.1% / 46.7% / 48.8% | 58.2% / 35.3% / 44.0% |
| gpt-5.5 | 65.9% / 58.0% / 61.7% | 65.3% / 65.3% / 65.3% | 60.9% / 65.3% / 63.0% | 62.1% / 66.7% / 64.3% |
| gpt-5.6-luna | 71.9% / 61.3% / 66.2% | 74.4% / 60.0% / 66.4% | 73.6% / 52.0% / 60.9% | 66.4% / 55.3% / 60.4% |
| gpt-5.6-sol | 64.1% / 54.7% / 59.0% | 60.0% / 50.0% / 54.5% | 51.4% / 48.0% / 49.7% | 49.0% / 46.7% / 47.8% |
| gpt-5.6-terra | 56.0% / 77.3% / 65.0% | 55.6% / 72.7% / 63.0% | 50.5% / 68.7% / 58.2% | 52.8% / 62.0% / 57.1% |
| gpt-6-astra | 90.0% / 72.0% / 80.0% | 90.8% / 79.3% / 84.7% | 87.0% / 80.0% / 83.3% | 84.1% / 81.3% / 82.7% |
| kimi-k3 | 67.8% / 53.3% / 59.7% | 65.9% / 55.3% / 60.1% | 59.8% / 48.7% / 53.7% | 60.3% / 52.7% / 56.2% |
| qwen3.8-max | 72.2% / 81.3% / 76.5% | 77.8% / 82.0% / 79.9% | 76.0% / 78.0% / 77.0% | 78.8% / 76.7% / 77.7% |

## Recommendation

EQUAL9 remains the strongest of these strategies on prompt-group OOF balanced accuracy. None of the new strategies beats its 65.3898% OOF reference, so this experiment does not recommend replacing EQUAL9 for Kaggle. Full-V0 per-label F1 weighting looks better only on the same rows used to select candidates and weights; treat that as selection optimism, not evidence of a generalization gain.

## Reproducibility files

- `experiment_config.json`: trust-list/V0 hashes, checkpoint sources and method definitions.
- `checkpoint_probabilities.npz` and `prediction_cache_manifest.json`: cached aligned input matrices and provenance.
- `per_label_checkpoint_metrics.csv`: precision/recall/F1/TP/FP for every candidate and label.
- `per_label_ensemble_metrics.csv`: per-label precision/recall/F1 for each strategy in both evaluation scopes.
- `label_overlap_combinations.csv`: exact 1–3 checkpoint overlap and incremental contribution tables.
- `fold_selected_rules.json` and `selected_weights.json`: cross-fit rules and learned weights.
- `row_predictions.parquet` and `metrics.json`: OOF/full diagnostic outputs and metrics.
