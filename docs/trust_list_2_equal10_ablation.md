# Trust List 2 EQUAL10 checkpoint ablation and extension

Cache-only evaluation on all 2,699 V0 responses. No training, inference, weight optimization, Kaggle work, or production changes were performed.

## Results

| Experiment | Checkpoints | Balanced accuracy | Δ vs EQUAL10 | Corrected | Introduced | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| EQUAL9-minus-N0056 | 9 | 65.3898% | +1.149 pp | 91 | 60 | 1.132076 |
| EQUAL11-Qwen | 11 | 64.3515% | +0.111 pp | 38 | 35 | 1.097558 |
| EQUAL9-minus-N0058 | 9 | 64.2757% | +0.035 pp | 73 | 72 | 1.132709 |
| EQUAL10 | 10 | 64.2409% | — | — | — | 1.110030 |
| EQUAL9-minus-N0044 | 9 | 64.1670% | -0.074 pp | 19 | 21 | 1.073450 |
| EQUAL11-Filtered | 11 | 64.0932% | -0.148 pp | 58 | 62 | 1.091842 |
| EQUAL9-minus-N0048 | 9 | 64.0557% | -0.185 pp | 73 | 78 | 1.126419 |
| EQUAL9-minus-N0061 | 9 | 63.9083% | -0.333 pp | 41 | 50 | 1.124726 |
| EQUAL12-Combined | 12 | 63.7972% | -0.444 pp | 61 | 73 | 1.082589 |
| EQUAL9-minus-N0072 | 9 | 63.7223% | -0.519 pp | 23 | 37 | 1.092204 |
| EQUAL9-minus-N0009 | 9 | 63.7211% | -0.520 pp | 25 | 39 | 1.099412 |
| EQUAL9-minus-N0030 | 9 | 63.6841% | -0.557 pp | 32 | 47 | 1.124175 |
| EQUAL9-minus-N0005 | 9 | 63.4994% | -0.741 pp | 28 | 48 | 1.101251 |
| EQUAL9-minus-N0071 | 9 | 63.3505% | -0.890 pp | 80 | 104 | 1.142586 |

## Per-label changes for the three best experiments

Accuracy deltas are percentage points relative to EQUAL10.

### EQUAL9-minus-N0056 (65.3898%)

| Label | EQUAL10 accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|
| claude-fable-5 | 50.34% | 52.35% | +2.01 pp |
| claude-opus-4-6 | 78.67% | 80.00% | +1.33 pp |
| claude-opus-4-8 | 71.33% | 76.67% | +5.33 pp |
| claude-opus-5 | 78.00% | 78.67% | +0.67 pp |
| claude-sonnet-5 | 72.00% | 73.33% | +1.33 pp |
| gemini-3.1-pro-preview | 78.67% | 78.00% | -0.67 pp |
| gemini-3.6-flash | 80.00% | 80.67% | +0.67 pp |
| gemini-3.7-flash | 22.00% | 22.00% | +0.00 pp |
| gemini-3.8-flash | 80.00% | 83.33% | +3.33 pp |
| glm-5.3 | 41.33% | 45.33% | +4.00 pp |
| glm-5.3-flash | 44.00% | 48.67% | +4.67 pp |
| gpt-5.5 | 56.00% | 58.00% | +2.00 pp |
| gpt-5.6-luna | 62.00% | 61.33% | -0.67 pp |
| gpt-5.6-sol | 54.00% | 54.67% | +0.67 pp |
| gpt-5.6-terra | 77.33% | 77.33% | +0.00 pp |
| gpt-6-astra | 73.33% | 72.00% | -1.33 pp |
| kimi-k3 | 55.33% | 53.33% | -2.00 pp |
| qwen3.8-max | 82.00% | 81.33% | -0.67 pp |

### EQUAL11-Qwen (64.3515%)

| Label | EQUAL10 accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|
| claude-fable-5 | 50.34% | 48.99% | -1.34 pp |
| claude-opus-4-6 | 78.67% | 80.67% | +2.00 pp |
| claude-opus-4-8 | 71.33% | 70.67% | -0.67 pp |
| claude-opus-5 | 78.00% | 78.67% | +0.67 pp |
| claude-sonnet-5 | 72.00% | 73.33% | +1.33 pp |
| gemini-3.1-pro-preview | 78.67% | 78.00% | -0.67 pp |
| gemini-3.6-flash | 80.00% | 80.67% | +0.67 pp |
| gemini-3.7-flash | 22.00% | 18.67% | -3.33 pp |
| gemini-3.8-flash | 80.00% | 82.67% | +2.67 pp |
| glm-5.3 | 41.33% | 40.67% | -0.67 pp |
| glm-5.3-flash | 44.00% | 44.00% | +0.00 pp |
| gpt-5.5 | 56.00% | 59.33% | +3.33 pp |
| gpt-5.6-luna | 62.00% | 64.00% | +2.00 pp |
| gpt-5.6-sol | 54.00% | 51.33% | -2.67 pp |
| gpt-5.6-terra | 77.33% | 74.00% | -3.33 pp |
| gpt-6-astra | 73.33% | 76.00% | +2.67 pp |
| kimi-k3 | 55.33% | 54.67% | -0.67 pp |
| qwen3.8-max | 82.00% | 82.00% | +0.00 pp |

### EQUAL9-minus-N0058 (64.2757%)

| Label | EQUAL10 accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|
| claude-fable-5 | 50.34% | 44.30% | -6.04 pp |
| claude-opus-4-6 | 78.67% | 77.33% | -1.33 pp |
| claude-opus-4-8 | 71.33% | 74.00% | +2.67 pp |
| claude-opus-5 | 78.00% | 79.33% | +1.33 pp |
| claude-sonnet-5 | 72.00% | 73.33% | +1.33 pp |
| gemini-3.1-pro-preview | 78.67% | 78.67% | +0.00 pp |
| gemini-3.6-flash | 80.00% | 82.00% | +2.00 pp |
| gemini-3.7-flash | 22.00% | 24.67% | +2.67 pp |
| gemini-3.8-flash | 80.00% | 82.00% | +2.00 pp |
| glm-5.3 | 41.33% | 40.67% | -0.67 pp |
| glm-5.3-flash | 44.00% | 48.00% | +4.00 pp |
| gpt-5.5 | 56.00% | 53.33% | -2.67 pp |
| gpt-5.6-luna | 62.00% | 61.33% | -0.67 pp |
| gpt-5.6-sol | 54.00% | 52.00% | -2.00 pp |
| gpt-5.6-terra | 77.33% | 78.00% | +0.67 pp |
| gpt-6-astra | 73.33% | 74.00% | +0.67 pp |
| kimi-k3 | 55.33% | 52.00% | -3.33 pp |
| qwen3.8-max | 82.00% | 82.00% | +0.00 pp |

## Conclusions

1. **Leave-one-out:** removing N0056 and N0058 improves balanced accuracy over EQUAL10. Removing N0056 and N0058 also raises overall accuracy.
2. N0044 and N0009 both contribute positively to equal-weight blending: removing N0044 lowers balanced accuracy by 0.074 pp, and removing N0009 lowers it by 0.520 pp. Their near-zero optimized weights therefore do not imply they are unhelpful in an equal blend.
3. EQUAL11-Qwen changes balanced accuracy by +0.111 pp; EQUAL11-Filtered changes it by -0.148 pp.
4. Best overall: **EQUAL9-minus-N0056**, 65.3898% balanced accuracy, +1.149 pp versus EQUAL10.
5. Best tested ensemble size: **9 checkpoints** (EQUAL9-minus-N0056, 65.3898%). Tested-size winners: 9: EQUAL9-minus-N0056 65.3898%, 10: EQUAL10 64.2409%, 11: EQUAL11-Qwen 64.3515%, 12: EQUAL12-Combined 63.7972%.

## Cache and method details

- EQUAL10 baseline reproduces the previous artifact exactly at the row-prediction level; 1,734/2,699 correct, accuracy 64.2460%, balanced accuracy 64.2409%.
- Base checkpoint order: `N0071, N0056, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048`. EQUAL9 variants give each retained checkpoint weight `1/9`; extensions use `1/11` or `1/12` for every included checkpoint.
- All cache IDs, labels, prompt IDs, finite values, and probability row sums were checked. Cached source paths, SHA256 hashes, checkpoint IDs and exact per-experiment weights are recorded in `artifacts/analysis/trust2_equal10_ablation/evaluation_metrics.json`.
- Row-level probabilities and predictions for EQUAL10 and all 13 evaluated variants are in `artifacts/analysis/trust2_equal10_ablation/v0_ensemble_predictions.parquet`.
