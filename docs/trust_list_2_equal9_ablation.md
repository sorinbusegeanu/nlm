# Trust List 2 EQUAL9 ablation and extension

Cache-only evaluation over the full 2,699-row V0 dataset. EQUAL9-minus-N0056 is the fixed reference. No training, inference, Kaggle work, or production changes were performed.

## Results

| Experiment | Checkpoints | Balanced accuracy | Δ vs EQUAL9 | Corrected | Introduced | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| EQUAL9-minus-N0056 (baseline) | 9 | 65.3898% | — | — | — | 1.132076 |
| EQUAL10-Qwen | 10 | 65.1680% | -0.222 pp | 44 | 50 | 1.115821 |
| EQUAL8-minus-N0044 | 8 | 65.1307% | -0.259 pp | 13 | 20 | 1.091810 |
| EQUAL8-minus-N0048 | 8 | 64.7609% | -0.629 pp | 81 | 98 | 1.155086 |
| EQUAL8-minus-N0072 | 8 | 64.6495% | -0.740 pp | 23 | 43 | 1.112683 |
| EQUAL8-minus-N0030 | 8 | 64.4268% | -0.963 pp | 26 | 52 | 1.149955 |
| EQUAL8-minus-N0071 | 8 | 64.3142% | -1.076 pp | 85 | 114 | 1.171949 |
| EQUAL8-minus-N0061 | 8 | 64.2816% | -1.108 pp | 44 | 74 | 1.153497 |
| EQUAL8-minus-N0058 | 8 | 64.2767% | -1.113 pp | 82 | 112 | 1.164126 |
| EQUAL8-minus-N0005 | 8 | 64.2414% | -1.148 pp | 25 | 56 | 1.122381 |
| EQUAL8-minus-N0009 | 8 | 64.1298% | -1.260 pp | 24 | 58 | 1.120361 |
| EQUAL10-Filtered | 10 | 63.7974% | -1.592 pp | 67 | 110 | 1.110203 |

## Per-label changes for the three best experiments

Accuracy deltas are percentage points relative to EQUAL9-minus-N0056.

### EQUAL10-Qwen (65.1680%)

| Label | Baseline accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|
| claude-fable-5 | 52.35% | 53.69% | +1.34 pp |
| claude-opus-4-6 | 80.00% | 82.00% | +2.00 pp |
| claude-opus-4-8 | 76.67% | 76.00% | -0.67 pp |
| claude-opus-5 | 78.67% | 78.00% | -0.67 pp |
| claude-sonnet-5 | 73.33% | 72.67% | -0.67 pp |
| gemini-3.1-pro-preview | 78.00% | 78.00% | +0.00 pp |
| gemini-3.6-flash | 80.67% | 82.67% | +2.00 pp |
| gemini-3.7-flash | 22.00% | 16.67% | -5.33 pp |
| gemini-3.8-flash | 83.33% | 84.67% | +1.33 pp |
| glm-5.3 | 45.33% | 44.00% | -1.33 pp |
| glm-5.3-flash | 48.67% | 48.67% | +0.00 pp |
| gpt-5.5 | 58.00% | 57.33% | -0.67 pp |
| gpt-5.6-luna | 61.33% | 63.33% | +2.00 pp |
| gpt-5.6-sol | 54.67% | 50.67% | -4.00 pp |
| gpt-5.6-terra | 77.33% | 76.00% | -1.33 pp |
| gpt-6-astra | 72.00% | 73.33% | +1.33 pp |
| kimi-k3 | 53.33% | 54.67% | +1.33 pp |
| qwen3.8-max | 81.33% | 80.67% | -0.67 pp |

### EQUAL8-minus-N0044 (65.1307%)

| Label | Baseline accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|
| claude-fable-5 | 52.35% | 53.02% | +0.67 pp |
| claude-opus-4-6 | 80.00% | 79.33% | -0.67 pp |
| claude-opus-4-8 | 76.67% | 76.67% | +0.00 pp |
| claude-opus-5 | 78.67% | 78.67% | +0.00 pp |
| claude-sonnet-5 | 73.33% | 73.33% | +0.00 pp |
| gemini-3.1-pro-preview | 78.00% | 78.00% | +0.00 pp |
| gemini-3.6-flash | 80.67% | 81.33% | +0.67 pp |
| gemini-3.7-flash | 22.00% | 21.33% | -0.67 pp |
| gemini-3.8-flash | 83.33% | 84.67% | +1.33 pp |
| glm-5.3 | 45.33% | 44.00% | -1.33 pp |
| glm-5.3-flash | 48.67% | 46.67% | -2.00 pp |
| gpt-5.5 | 58.00% | 57.33% | -0.67 pp |
| gpt-5.6-luna | 61.33% | 60.67% | -0.67 pp |
| gpt-5.6-sol | 54.67% | 54.00% | -0.67 pp |
| gpt-5.6-terra | 77.33% | 78.00% | +0.67 pp |
| gpt-6-astra | 72.00% | 70.00% | -2.00 pp |
| kimi-k3 | 53.33% | 53.33% | +0.00 pp |
| qwen3.8-max | 81.33% | 82.00% | +0.67 pp |

### EQUAL8-minus-N0048 (64.7609%)

| Label | Baseline accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|
| claude-fable-5 | 52.35% | 54.36% | +2.01 pp |
| claude-opus-4-6 | 80.00% | 83.33% | +3.33 pp |
| claude-opus-4-8 | 76.67% | 72.67% | -4.00 pp |
| claude-opus-5 | 78.67% | 79.33% | +0.67 pp |
| claude-sonnet-5 | 73.33% | 73.33% | +0.00 pp |
| gemini-3.1-pro-preview | 78.00% | 83.33% | +5.33 pp |
| gemini-3.6-flash | 80.67% | 78.67% | -2.00 pp |
| gemini-3.7-flash | 22.00% | 16.00% | -6.00 pp |
| gemini-3.8-flash | 83.33% | 85.33% | +2.00 pp |
| glm-5.3 | 45.33% | 41.33% | -4.00 pp |
| glm-5.3-flash | 48.67% | 43.33% | -5.33 pp |
| gpt-5.5 | 58.00% | 62.00% | +4.00 pp |
| gpt-5.6-luna | 61.33% | 67.33% | +6.00 pp |
| gpt-5.6-sol | 54.67% | 50.00% | -4.67 pp |
| gpt-5.6-terra | 77.33% | 70.67% | -6.67 pp |
| gpt-6-astra | 72.00% | 76.00% | +4.00 pp |
| kimi-k3 | 53.33% | 46.00% | -7.33 pp |
| qwen3.8-max | 81.33% | 82.67% | +1.33 pp |

## Conclusions

- Best of the 11 tested variants: **EQUAL10-Qwen**, 65.1680% balanced accuracy (-0.222 pp vs the EQUAL9 baseline).
- None of the 11 tested variants exceeds the EQUAL9 baseline at 65.3898%.
- Removing a checkpoint from EQUAL9 improves balanced accuracy for: none.
- EQUAL10-Qwen changes the baseline by -0.222 pp; EQUAL10-Filtered changes it by -1.592 pp.

## Inputs

- Baseline members, in order: `N0071, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048`; each has weight `1/9`.
- EQUAL8 ablations give each remaining checkpoint weight `1/8`; extensions give every member weight `1/10`.
- Baseline row predictions match the saved prior EQUAL9 artifact exactly: True.
- Every cache was checked for full V0 ID and prompt alignment, canonical 18-label ordering, finite nonnegative probabilities, and normalized rows. Cache paths, hashes, checkpoint IDs and exact weights are recorded in the JSON artifact.
- Row-level probabilities and predictions for the baseline and 11 evaluations are in `artifacts/analysis/trust2_equal9_ablation/v0_ensemble_predictions.parquet`.
