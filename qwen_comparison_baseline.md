# Qwen prediction comparison baseline

## Scope

Compared the three most recent existing Qwen validation prediction files, each covering the same 540 rows from split `147ce64e`:

| Prediction file | Interpretation | Accuracy | Errors |
|---|---|---:|---:|
| `qwen06_best_epoch03_v0_from3789_code8422c8c_validation.parquet` | Qwen3-0.6B, epoch 3 checkpoint | 36.11% | 345 |
| `qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfg5d2d686c_code667e2d0_validation.parquet` | Qwen3-1.7B | 31.11% | 372 |
| `qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg42943789_code8422c8c_validation.parquet` | Qwen3-0.6B run's final checkpoint | 32.22% | 366 |

The three files have identical validation IDs and labels. Probabilities are normalized in all three files. I found five Qwen validation files in total; the two older files are additional Qwen3-0.6B runs, scoring 34.44% and 30.19%. There is not yet a Qwen3-4B validation prediction file; its current training run has not completed. This comparison therefore does not measure a three-size (0.6B/1.7B/4B) ensemble. The selected three artifacts include two checkpoints from the 0.6B run.

## Error overlap

“Both wrong” counts rows where both models miss the true label. Error Jaccard is the intersection of their error sets divided by their union. The conditional column reports the fraction of the stronger model's errors also missed by the other model.

| Pair | Both wrong | Error Jaccard | Stronger model's errors shared | Rows where only one is wrong |
|---|---:|---:|---:|---:|
| 0.6B epoch 3 + 1.7B | 278 | 63.3% | 80.6% | 161 |
| 0.6B epoch 3 + 0.6B final | 293 | 70.1% | 84.9% | 125 |
| 1.7B + 0.6B final | 289 | 64.4% | 78.9% | 160 |

Error overlap is high. The 0.6B epoch 3 and final checkpoints are especially redundant, as expected for checkpoints from the same training run.

## Equal-weight probability ensembles

Probabilities were averaged by class and the top class selected. Gain is measured against the strongest individual member of that combination.

| Combination | Ensemble accuracy | Strongest member | Gain vs. strongest member |
|---|---:|---:|---:|
| 0.6B epoch 3 + 1.7B | 34.81% | 36.11% | -1.30 pp |
| 0.6B epoch 3 + 0.6B final | 32.96% | 36.11% | -3.15 pp |
| 1.7B + 0.6B final | 34.07% | 32.22% | +1.85 pp |
| All three | 34.44% | 36.11% | -1.67 pp |

The 1.7B + 0.6B-final pair improves over either member of that pair, but still trails the epoch 3 0.6B checkpoint. Adding all three does not improve on the best single model. For the trio, all three are wrong on 243 rows; at least one is wrong on 466 rows.

## Conclusion

For these artifacts, error overlap is substantial and equal-weight ensembling provides no gain over the best checkpoint. The 0.6B epoch 3 checkpoint is the strongest result at 36.11%. The one pairwise gain is modest and does not beat it. Revisit this comparison after Qwen3-4B has a completed validation prediction file; then compare one selected checkpoint per model size.

## Reproducibility

- Dataset: V0 validation split, 540 rows; split hash `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`.
- Prediction columns: `prob::<class>`; ensemble is their arithmetic mean across members.
- Metrics are computed directly from each parquet's `y_true` and probability columns.
