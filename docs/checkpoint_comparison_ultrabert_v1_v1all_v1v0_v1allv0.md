# UltraBERT checkpoint comparison: V1, V1-all, and V0 continuations

## Scope and method

This report compares four saved UltraBERT checkpoints on the same frozen V0 validation rows. Predictions are loaded from the run artifacts when available; missing predictions are generated from the saved checkpoint. IDs, labels, prompt IDs, class order, and probability rows are validated before scoring.

- Validation rows: **540**; classes: **18**; prompt groups: **30**.
- Split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`.
- Validation ID SHA-256: `d3f39a2003dfdd9e99e5d662465066bd60368f8e43864b6ed10a733edbe74f9d`.
- Metrics: accuracy, balanced accuracy, macro F1, top-3 accuracy, mean reciprocal rank (MRR), log loss, and multiclass Brier score.
- Bootstrap: paired prompt-group bootstrap with 2,000 replicates and seed 2026; intervals describe validation sampling uncertainty, not generalization guarantees.

## Checkpoints

| Stage | Experiment | Training sequence | Checkpoint | Selected epoch | Run-recorded selection accuracy | Learning rate | Decay | Validation predictions |
|---|---|---|---|---:|---:|---:|---:|---|
| V1 | `ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0` | V1 | `ultrabert380_foldvalidation` | — | — | 8e-06 | — | `ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0_validation.parquet` |
| V1-all | `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2` | V1-all | `ultrabert380_foldvalidation` | 2 | 52.4% | 8e-06 | 0.7 | `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2_validation.parquet` |
| V1 → V0 | `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0` | V1 → V0 | `best_checkpoint` | 5 | 53.1% | 8e-06 | 0.7 | `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_validation.parquet` |
| V1-all → V0 | `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff` | V1-all → V0 | `best_checkpoint` | 2 | 55.4% | 8e-06 | 0.7 | `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_validation.parquet` |

## Overall metrics

Accuracy is row-level top-1 accuracy. Balanced accuracy is the macro-average recall; because the frozen set has equal class support, it should closely match accuracy.

The checkpoint table's run-recorded selection accuracy is copied from the training manifest. The metric table is recalculated from each persisted validation probability file, giving one consistent inference-time comparison. The V1-all and V1-all → V0 records differ from their persisted predictions by one validation row; the recalculated values are used in the comparisons below.

| Checkpoint | Accuracy | Balanced accuracy | Macro F1 | Top-3 accuracy | MRR | Log loss ↓ | Brier ↓ | Δ balanced accuracy vs V1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 51.1% | 51.1% | 51.8% | 84.8% | 0.6940 | 2.7764 | 0.7932 | +0.0 pp |
| V1-all | 52.2% | 52.2% | 52.4% | 85.4% | 0.6956 | 1.9111 | 0.6988 | +1.1 pp |
| V1 → V0 | 53.1% | 53.1% | 53.6% | 83.7% | 0.6970 | 13.4997 | 0.9237 | +2.0 pp |
| V1-all → V0 | 55.2% | 55.2% | 54.2% | 86.3% | 0.7166 | 2.5716 | 0.7288 | +4.1 pp |

## Paired changes from each base checkpoint

Corrected/regressed counts are rows where the V0 continuation becomes correct or wrong relative to its base checkpoint.

| Comparison | Δ accuracy | Δ balanced accuracy | 95% prompt-bootstrap CI for Δ balanced accuracy | Corrected | Regressed |
|---|---:|---:|---:|---:|---:|
| V1 base → V0 continuation | +2.0 pp | +2.0 pp | [-0.9 pp, +5.4 pp] | 45 | 34 |
| V1-all base → V0 continuation | +3.0 pp | +3.0 pp | [-1.3 pp, +7.2 pp] | 78 | 62 |

## Per-class top-1 recall

| Target model | V1 | V1-all | V1 → V0 | V1-all → V0 | Best checkpoint |
|---|---:|---:|---:|---:|---:|
| Claude Fable 5 | 26.7% | 30.0% | 23.3% | 53.3% | V1-all → V0 |
| Claude Opus 4 6 | 73.3% | 56.7% | 73.3% | 86.7% | V1-all → V0 |
| Claude Opus 4.8 | 50.0% | 70.0% | 60.0% | 53.3% | V1-all |
| Claude Opus 5 | 63.3% | 66.7% | 70.0% | 66.7% | V1 → V0 |
| Claude Sonnet 5 | 60.0% | 53.3% | 53.3% | 86.7% | V1-all → V0 |
| Gemini 3.1 Pro Preview | 73.3% | 73.3% | 80.0% | 90.0% | V1-all → V0 |
| Gemini 3.6 Flash | 46.7% | 70.0% | 43.3% | 76.7% | V1-all → V0 |
| Gemini 3.7 Flash | 46.7% | 33.3% | 73.3% | 26.7% | V1 → V0 |
| Gemini 3.8 Flash | 56.7% | 63.3% | 36.7% | 46.7% | V1-all |
| GLM 5.3 | 26.7% | 23.3% | 30.0% | 23.3% | V1 → V0 |
| Glm 5.3 Flash | 33.3% | 53.3% | 50.0% | 30.0% | V1-all |
| Gpt 5.5 | 40.0% | 30.0% | 50.0% | 76.7% | V1-all → V0 |
| GPT-5.6 Luna | 36.7% | 16.7% | 40.0% | 36.7% | V1 → V0 |
| GPT-5.6 Sol | 50.0% | 46.7% | 36.7% | 36.7% | V1 |
| Gpt 5.6 Terra | 53.3% | 66.7% | 53.3% | 46.7% | V1-all |
| Gpt 6 Astra | 76.7% | 70.0% | 76.7% | 63.3% | V1 |
| Kimi K3 | 43.3% | 53.3% | 36.7% | 26.7% | V1-all |
| Qwen3.8 Max | 63.3% | 63.3% | 70.0% | 66.7% | V1 → V0 |

## Pairwise prediction and error overlap

Agreement compares the two models' predicted labels. 'Both wrong' counts rows both miss; 'either correct' is the oracle coverage if a selector could choose the right model per row.

| Pair | Prediction agreement | Both wrong | Either correct | Left corrected by right | Left regressed by right | Δ balanced accuracy, left → right |
|---|---:|---:|---:|---:|---:|---:|
| V1 vs V1-all | 60.7% | 193 | 347 | 71 | 65 | +1.1 pp |
| V1 vs V1 → V0 | 77.0% | 219 | 321 | 45 | 34 | +2.0 pp |
| V1 vs V1-all → V0 | 61.1% | 185 | 355 | 79 | 57 | +4.1 pp |
| V1-all vs V1 → V0 | 57.8% | 180 | 360 | 78 | 73 | +0.9 pp |
| V1-all vs V1-all → V0 | 63.0% | 180 | 360 | 78 | 62 | +3.0 pp |
| V1 → V0 vs V1-all → V0 | 60.0% | 172 | 368 | 81 | 70 | +2.0 pp |

## Reading the result

- Highest balanced accuracy on this validation set: **V1-all → V0** (55.2%).
- V1 → V0 changed balanced accuracy by +2.0 pp (95% prompt-bootstrap CI [-0.9 pp, +5.4 pp]); V1-all → V0 changed it by +3.0 pp (95% CI [-1.3 pp, +7.2 pp]). Both intervals include zero, so these validation comparisons do not establish a reliable improvement under prompt resampling.
- V1 → V0 has much worse probability scores than V1 (log loss 13.50 vs 2.78; Brier 0.924 vs 0.793), indicating severe overconfidence in its saved probabilities.
- V1-all → V0 has the best top-1 and top-3 scores, but its log loss and Brier score are worse than V1-all (2.57 vs 1.91; 0.729 vs 0.699); recalibration should be evaluated separately before relying on its confidence values.
- Validation comparisons are development evidence: these same V0 rows were used for checkpoint selection.
- A positive bootstrap interval wholly above zero supports an improvement under prompt resampling; an interval spanning zero leaves the direction uncertain at this sample size.
