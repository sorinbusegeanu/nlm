# November 15 ensemble hypotheses

Cache-only comparison on all 2,699 official V0 rows. No model was loaded; no inference, model training, or Kaggle submission was run.

ID resolution used [trust-list-2-v022](../docs/trust-list-2-v022.csv) because `2026-11-15T06-10_export.csv` was not found in the local workspace. Prediction files were aligned by row ID, true label, and the canonical 18-label order. Cache files and hashes are recorded in `artifacts/analysis/ensemble_hypotheses_nov15/metadata.json`.

Full-V0 results are exploratory: V0 was used for checkpoint selection and previous trust-list/ensemble development. Cross-fitted optimized scores use five GroupKFold folds with prompt IDs kept intact; report them separately from the full-V0 fixed-weight scores.

## Historical baselines

| Ensemble | Members | V0 accuracy | V0 balanced accuracy | Kaggle public |
|---|---:|---:|---:|---:|
| HIST-EQUAL9-v013 | 9 | 65.3946% | 65.3898% | 0.567 |
| HIST-EQUAL10-Qwen-v013 | 10 | 65.1723% | 65.1680% | 0.567 |
| HIST-WEIGHTED5-v013 | 5 | 60.3927% | 60.3888% | 0.564 |
| HIST-EQUAL10 | 10 | 64.2460% | 64.2409% | not submitted |

Historical EQUAL10 is the reference for the scenario deltas (64.2409% balanced accuracy). EQUAL9 and EQUAL10-Qwen had public scores of 0.567; WEIGHTED5 had 0.564. Those public scores are treated as practically equivalent.

## Fixed-weight scenarios

Ranked by full-V0 balanced accuracy. Corrected/errors-lost compare each row’s predictions against historical EQUAL10.

| Rank | Scenario | Method | Members | Accuracy | Balanced acc. | Δ bal. acc. | Corrected | Correct lost | Log loss |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | H09 | EQUAL | 10 | 65.5057% | 65.5004% | +1.260 pp | 98 | 64 | 1.10515 |
| 2 | H09 | FAMILY-EQUAL | 10 | 65.4687% | 65.4631% | +1.222 pp | 113 | 80 | 1.07308 |
| 3 | H10 | EQUAL | 10 | 65.0982% | 65.0922% | +0.851 pp | 103 | 80 | 1.10135 |
| 4 | H08 | FAMILY-EQUAL | 8 | 64.2460% | 64.2411% | +0.000 pp | 103 | 103 | 1.03733 |
| 5 | H10 | FAMILY-EQUAL | 10 | 64.2090% | 64.2036% | -0.037 pp | 87 | 88 | 1.07010 |
| 6 | H07 | FAMILY-EQUAL | 8 | 64.0978% | 64.0937% | -0.147 pp | 142 | 146 | 1.03112 |
| 7 | H04 | FAMILY-EQUAL | 8 | 63.9867% | 63.9826% | -0.258 pp | 97 | 104 | 1.02581 |
| 8 | H02 | EQUAL | 6 | 63.8755% | 63.8710% | -0.370 pp | 138 | 148 | 1.03001 |
| 9 | H02 | FAMILY-EQUAL | 6 | 63.8755% | 63.8710% | -0.370 pp | 138 | 148 | 1.03001 |
| 10 | H08 | EQUAL | 8 | 63.8014% | 63.7949% | -0.446 pp | 94 | 106 | 1.02307 |
| 11 | H07 | EQUAL | 8 | 63.6532% | 63.6495% | -0.591 pp | 158 | 174 | 1.01970 |
| 12 | H05 | FAMILY-EQUAL | 9 | 63.6162% | 63.6115% | -0.629 pp | 104 | 121 | 1.02579 |
| 13 | H04 | EQUAL | 8 | 63.5791% | 63.5739% | -0.667 pp | 91 | 109 | 1.01834 |
| 14 | H06 | FAMILY-EQUAL | 10 | 63.4680% | 63.4628% | -0.778 pp | 97 | 118 | 1.02717 |
| 15 | H03 | FAMILY-EQUAL | 7 | 63.4309% | 63.4270% | -0.814 pp | 103 | 125 | 1.02983 |
| 16 | H03 | EQUAL | 7 | 63.4309% | 63.4263% | -0.815 pp | 108 | 130 | 1.02046 |
| 17 | H06 | EQUAL | 10 | 63.4309% | 63.4251% | -0.816 pp | 102 | 124 | 1.02257 |
| 18 | H05 | EQUAL | 9 | 62.9122% | 62.9068% | -1.334 pp | 93 | 129 | 1.01891 |
| 19 | H01 | EQUAL | 5 | 62.8381% | 62.8340% | -1.407 pp | 140 | 178 | 1.04583 |
| 20 | H01 | FAMILY-EQUAL | 5 | 62.8381% | 62.8340% | -1.407 pp | 140 | 178 | 1.04583 |

### Progressive additions H01–H06

Accuracy change from each preceding scenario (H02−H01 through H06−H05), separately by fixed weighting method:

| Step | Checkpoint added | Equal accuracy Δ | Family-equal accuracy Δ | Equal balanced-accuracy Δ | Family-equal balanced-accuracy Δ |
|---|---|---:|---:|---:|---:|
| H01 → H02 | N0030 | +1.037 pp | +1.037 pp | +1.037 pp | +1.037 pp |
| H02 → H03 | N0056 | -0.445 pp | -0.445 pp | -0.445 pp | -0.444 pp |
| H03 → H04 | N0061 | +0.148 pp | +0.556 pp | +0.148 pp | +0.556 pp |
| H04 → H05 | N0057 | -0.667 pp | -0.371 pp | -0.667 pp | -0.371 pp |
| H05 → H06 | N0060 | +0.519 pp | -0.148 pp | +0.518 pp | -0.149 pp |

### H09–H10 versus historical EQUAL10

| Scenario | Method | Accuracy Δ | Balanced accuracy Δ | Corrected | Correct lost |
|---|---|---:|---:|---:|---:|
| H09 | EQUAL | +1.260 pp | +1.260 pp | 98 | 64 |
| H09 | FAMILY-EQUAL | +1.223 pp | +1.222 pp | 113 | 80 |
| H10 | EQUAL | +0.852 pp | +0.851 pp | 103 | 80 |
| H10 | FAMILY-EQUAL | -0.037 pp | -0.037 pp | 87 | 88 |

## Cross-fitted optimized weights

The three highest-ranked distinct checkpoint sets among the 20 fixed-weight scenario/method results were selected. Nonnegative weights were refit with the existing `fit_weights` optimizer (`l2=0.05`) on the other four GroupKFold prompt folds, then applied to each held-out fold. These are out-of-fold predictions, not full-fit resubstitution scores.

| Scenario | Best fixed method | Members | OOF accuracy | OOF balanced acc. | OOF log loss | Weights by fold |
|---|---|---:|---:|---:|---:|---|
| H09 | EQUAL | 10 | 63.4309% | 63.4246% | 1.00936 | `artifacts/analysis/ensemble_hypotheses_nov15/optimized_weights.json` |
| H10 | EQUAL | 10 | 63.1715% | 63.1646% | 1.00591 | `artifacts/analysis/ensemble_hypotheses_nov15/optimized_weights.json` |
| H08 | FAMILY-EQUAL | 8 | 63.3197% | 63.3125% | 1.00580 | `artifacts/analysis/ensemble_hypotheses_nov15/optimized_weights.json` |

## Focus labels

Precision and recall are from the fixed-weight configurations and the three cross-fitted optimized configurations. Deltas compare with historical EQUAL10. Full per-label metrics for every method, including all H01–H10 variants, are in `artifacts/analysis/ensemble_hypotheses_nov15/per_label_metrics.csv`.

| Configuration | Label | Precision | Recall | Δ recall | Δ correct | Support |
|---|---|---:|---:|---:|---:|---:|
| H09-EQUAL | claude-fable-5 | 41.1% | 51.0% | +0.7 pp | +1 | 149 |
| H09-EQUAL | gemini-3.7-flash | 70.0% | 14.0% | -8.0 pp | -12 | 150 |
| H09-EQUAL | glm-5.3 | 38.9% | 43.3% | +2.0 pp | +3 | 150 |
| H09-EQUAL | glm-5.3-flash | 51.0% | 51.3% | +7.3 pp | +11 | 150 |
| H09-EQUAL | gpt-5.6-sol | 63.3% | 54.0% | +0.0 pp | +0 | 150 |
| H09-FAMILY-EQUAL | claude-fable-5 | 40.3% | 50.3% | +0.0 pp | +0 | 149 |
| H09-FAMILY-EQUAL | gemini-3.7-flash | 75.9% | 14.7% | -7.3 pp | -11 | 150 |
| H09-FAMILY-EQUAL | glm-5.3 | 34.7% | 44.0% | +2.7 pp | +4 | 150 |
| H09-FAMILY-EQUAL | glm-5.3-flash | 49.6% | 46.7% | +2.7 pp | +4 | 150 |
| H09-FAMILY-EQUAL | gpt-5.6-sol | 61.4% | 52.0% | -2.0 pp | -3 | 150 |
| H10-EQUAL | claude-fable-5 | 40.3% | 49.0% | -1.3 pp | -2 | 149 |
| H10-EQUAL | gemini-3.7-flash | 67.9% | 12.7% | -9.3 pp | -14 | 150 |
| H10-EQUAL | glm-5.3 | 34.7% | 40.7% | -0.7 pp | -1 | 150 |
| H10-EQUAL | glm-5.3-flash | 48.3% | 46.0% | +2.0 pp | +3 | 150 |
| H10-EQUAL | gpt-5.6-sol | 66.1% | 52.0% | -2.0 pp | -3 | 150 |
| HIST-EQUAL10 | claude-fable-5 | 39.9% | 50.3% | +0.0 pp | +0 | 149 |
| HIST-EQUAL10 | gemini-3.7-flash | 58.9% | 22.0% | +0.0 pp | +0 | 150 |
| HIST-EQUAL10 | glm-5.3 | 33.5% | 41.3% | +0.0 pp | +0 | 150 |
| HIST-EQUAL10 | glm-5.3-flash | 45.8% | 44.0% | +0.0 pp | +0 | 150 |
| HIST-EQUAL10 | gpt-5.6-sol | 62.8% | 54.0% | +0.0 pp | +0 | 150 |
| H09-OPTIMIZED-CV | claude-fable-5 | 35.8% | 46.3% | -4.0 pp | -6 | 149 |
| H09-OPTIMIZED-CV | gemini-3.7-flash | 66.7% | 9.3% | -12.7 pp | -19 | 150 |
| H09-OPTIMIZED-CV | glm-5.3 | 28.7% | 33.3% | -8.0 pp | -12 | 150 |
| H09-OPTIMIZED-CV | glm-5.3-flash | 43.4% | 39.3% | -4.7 pp | -7 | 150 |
| H09-OPTIMIZED-CV | gpt-5.6-sol | 58.9% | 57.3% | +3.3 pp | +5 | 150 |
| H10-OPTIMIZED-CV | claude-fable-5 | 36.1% | 44.3% | -6.0 pp | -9 | 149 |
| H10-OPTIMIZED-CV | gemini-3.7-flash | 62.5% | 10.0% | -12.0 pp | -18 | 150 |
| H10-OPTIMIZED-CV | glm-5.3 | 29.9% | 35.3% | -6.0 pp | -9 | 150 |
| H10-OPTIMIZED-CV | glm-5.3-flash | 40.6% | 34.7% | -9.3 pp | -14 | 150 |
| H10-OPTIMIZED-CV | gpt-5.6-sol | 60.4% | 56.0% | +2.0 pp | +3 | 150 |
| H08-OPTIMIZED-CV | claude-fable-5 | 35.7% | 43.6% | -6.7 pp | -10 | 149 |
| H08-OPTIMIZED-CV | gemini-3.7-flash | 67.9% | 12.7% | -9.3 pp | -14 | 150 |
| H08-OPTIMIZED-CV | glm-5.3 | 30.2% | 36.7% | -4.7 pp | -7 | 150 |
| H08-OPTIMIZED-CV | glm-5.3-flash | 40.0% | 32.0% | -12.0 pp | -18 | 150 |
| H08-OPTIMIZED-CV | gpt-5.6-sol | 60.4% | 56.0% | +2.0 pp | +3 | 150 |

## Specialist check: N0065

N0065 alone recalls `gemini-3.7-flash` at 77.3%; N0081 alone recalls it at 0.7%. H08's equal and family-equal blends recall it at 24.7% and 24.7%. The ensemble dilutes most of N0065's specialist recall.

## Recommendation

1. **H09-EQUAL** — 10 checkpoints, 65.50% full-V0 balanced accuracy; +0.11 pp vs EQUAL9, +0.33 pp vs EQUAL10-Qwen, +5.11 pp vs WEIGHTED5; cross-fitted optimized result 63.42% (-2.08 pp vs fixed).
2. **H09-FAMILY-EQUAL** — 10 checkpoints, 65.46% full-V0 balanced accuracy; +0.07 pp vs EQUAL9, +0.30 pp vs EQUAL10-Qwen, +5.07 pp vs WEIGHTED5; cross-fitted optimized result 63.42% (-2.04 pp vs fixed).
3. **H10-EQUAL** — 10 checkpoints, 65.09% full-V0 balanced accuracy; -0.30 pp vs EQUAL9, -0.08 pp vs EQUAL10-Qwen, +4.70 pp vs WEIGHTED5; cross-fitted optimized result 63.16% (-1.93 pp vs fixed).

Recommendations prioritize balanced accuracy, then cross-fitted behavior and label-specific regressions. Validate on a genuinely unseen set or Kaggle before treating small V0 differences as gains.
