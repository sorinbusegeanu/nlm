# Trust List 2 ten-checkpoint weight comparison

Cache-only evaluation over all 2,699 V0 rows. No inference, classifier training, Kaggle work, or production ensemble changes were performed. The V0-optimized weights use V0 labels; only the cross-fitted OOF result is treated as the primary generalization comparison.

## Comparison

| Strategy | V0 accuracy | V0 balanced accuracy | Correct | Δ vs B10 | Corrected | Introduced | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| B10 | 63.9126% | 63.9073% | 1,725/2,699 | — | — | — | 1.086108 |
| EQUAL10 | 64.2460% | 64.2409% | 1,734/2,699 | +0.334 pp | 35 | 26 | 1.110030 |
| OPTIMIZED10 OOF | 62.3564% | 62.3510% | 1,683/2,699 | -1.556 pp | 40 | 82 | 1.034462 |
| OPTIMIZED10 full-V0 fit | 62.2823% | 62.2769% | 1,681/2,699 | -1.630 pp | 37 | 81 | 1.028173 |

## Weights

B10 preserves the original WEIGHTED5 relative weights: each original weight is halved, and each of the five added checkpoints receives 0.1. EQUAL10 assigns 0.1 to every checkpoint.

| Checkpoint | B10 | EQUAL10 | OPT OOF mean | OPT OOF SD | OPT OOF min–max | Full-V0 fit |
|---|---:|---:|---:|---:|---:|---:|
| N0071 | 0.140596 | 0.100000 | 0.212319 | 0.011069 | 0.197009–0.223767 | 0.212147 |
| N0056 | 0.116236 | 0.100000 | 0.141718 | 0.007210 | 0.131656–0.152316 | 0.141345 |
| N0005 | 0.091676 | 0.100000 | 0.059253 | 0.014912 | 0.045229–0.088088 | 0.058272 |
| N0058 | 0.117273 | 0.100000 | 0.156164 | 0.005217 | 0.152047–0.166191 | 0.156122 |
| N0072 | 0.034219 | 0.100000 | 0.006557 | 0.004543 | 0.000000–0.013349 | 0.006058 |
| N0030 | 0.100000 | 0.100000 | 0.210415 | 0.012714 | 0.195764–0.233574 | 0.210873 |
| N0061 | 0.100000 | 0.100000 | 0.137626 | 0.011167 | 0.118083–0.151637 | 0.138334 |
| N0044 | 0.100000 | 0.100000 | 0.000000 | 0.000000 | 0.000000–0.000000 | 0.000000 |
| N0009 | 0.100000 | 0.100000 | 0.000094 | 0.000189 | 0.000000–0.000472 | 0.000000 |
| N0048 | 0.100000 | 0.100000 | 0.075853 | 0.013674 | 0.052063–0.089673 | 0.076849 |

### Fold-specific OPTIMIZED10 OOF weights

| Checkpoint | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| N0071 | 0.197009 | 0.219387 | 0.220517 | 0.223767 | 0.200916 |
| N0056 | 0.152316 | 0.139423 | 0.147042 | 0.131656 | 0.138151 |
| N0005 | 0.088088 | 0.056676 | 0.052068 | 0.045229 | 0.054205 |
| N0058 | 0.155432 | 0.152052 | 0.166191 | 0.152047 | 0.155100 |
| N0072 | 0.003434 | 0.000000 | 0.007735 | 0.013349 | 0.008268 |
| N0030 | 0.233574 | 0.202945 | 0.195764 | 0.209253 | 0.210536 |
| N0061 | 0.118083 | 0.139844 | 0.135110 | 0.151637 | 0.143458 |
| N0044 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| N0009 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000472 |
| N0048 | 0.052063 | 0.089673 | 0.075574 | 0.073061 | 0.088893 |

## Per-label accuracy

| Label | B10 | EQUAL10 | OPTIMIZED10 OOF | Full-V0 fit |
|---|---:|---:|---:|---:|
| claude-fable-5 | 49.66% | 50.34% | 47.65% | 47.65% |
| claude-opus-4-6 | 77.33% | 78.67% | 77.33% | 77.33% |
| claude-opus-4-8 | 70.00% | 71.33% | 68.00% | 69.33% |
| claude-opus-5 | 78.00% | 78.00% | 75.33% | 74.67% |
| claude-sonnet-5 | 72.00% | 72.00% | 74.00% | 72.00% |
| gemini-3.1-pro-preview | 78.00% | 78.67% | 76.67% | 77.33% |
| gemini-3.6-flash | 80.67% | 80.00% | 78.67% | 78.67% |
| gemini-3.7-flash | 23.33% | 22.00% | 24.67% | 22.67% |
| gemini-3.8-flash | 80.00% | 80.00% | 80.67% | 80.67% |
| glm-5.3 | 40.67% | 41.33% | 37.33% | 38.00% |
| glm-5.3-flash | 39.33% | 44.00% | 29.33% | 31.33% |
| gpt-5.5 | 57.33% | 56.00% | 60.00% | 60.00% |
| gpt-5.6-luna | 60.67% | 62.00% | 62.00% | 61.33% |
| gpt-5.6-sol | 56.00% | 54.00% | 52.00% | 52.00% |
| gpt-5.6-terra | 75.33% | 77.33% | 73.33% | 73.33% |
| gpt-6-astra | 75.33% | 73.33% | 72.00% | 71.33% |
| kimi-k3 | 55.33% | 55.33% | 52.67% | 52.67% |
| qwen3.8-max | 81.33% | 82.00% | 80.67% | 80.67% |

## Error overlap

Pairwise and three-way intersections count rows misclassified by each listed strategy.

| Error set intersection | Rows |
|---|---:|
| B10 ∩ EQUAL10 | 939 |
| B10 ∩ OPTIMIZED10 OOF | 934 |
| EQUAL10 ∩ OPTIMIZED10 OOF | 905 |
| B10 ∩ EQUAL10 ∩ OPTIMIZED10 OOF | 904 |

## Conclusions

1. EQUAL10 improves on B10 by +0.334 pp balanced accuracy.
2. Cross-fitted OPTIMIZED10 does not improve on B10 by -1.556 pp. This is the primary out-of-fold comparison.
3. Fold-weight stability: mean per-checkpoint standard deviation 0.0081; the largest fold range is 0.0429 (N0005).
4. Highest mean OOF weights: N0071 (0.212), N0030 (0.210), N0058 (0.156).
5. Best accuracy is EQUAL10 (64.2409%); best generalization evidence is OPTIMIZED10 OOF because its fold weights are learned without the held-out prompt groups.

OPTIMIZED10 full-V0 fit metrics are in-sample: weights were fit using all V0 labels and are not a generalization estimate.

## Inputs and reproducibility

- Checkpoint order: `N0071, N0056, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048`.
- Prompt-group folds: `GroupKFold(n_splits=5, shuffle=True, random_state=2026)`; each prompt group is assigned wholly to one held-out fold.
- Weight fitting uses `fit_weights` / `cross_fitted_weights` from `src/ensemble_factory.py`, log loss with `l2=0.05`, simplex constraints, and five fold-specific training complements.
- Extended evaluation input artifact SHA256: `f65e9b3ecf8e60eb04089348f126cc23fe6375a16273dc397b05e8a4515c0869`.
- Per-checkpoint cache paths and SHA256 hashes, exact row order, canonical label order, fold assignments, weight vectors, and metrics are recorded in `artifacts/analysis/trust2_ten_checkpoint_weights/evaluation_metrics.json`.
- Row-level probabilities and predictions for all four strategies are in `artifacts/analysis/trust2_ten_checkpoint_weights/v0_ensemble_predictions.parquet`.
