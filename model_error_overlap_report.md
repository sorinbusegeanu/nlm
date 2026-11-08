# Model prediction error overlap and ensemble baseline

**Scope:** completed V0 validation predictions available in `artifacts/predictions/` when this report was prepared. Each included individual prediction file covers the same 540 validation rows with the same split hash (`147ce64e`). Accuracy is ordinary accuracy; the validation set is balanced, so it equals balanced accuracy here.

## Individual prediction files

Seven individual-model artifacts are available, covering four model families. Qwen 0.6B has four runs/checkpoint files, so it appears repeatedly; those are not four independent model families.

| Short name | Prediction artifact | Accuracy | Errors |
|---|---|---:|---:|
| SVC | `svc_v0_data40a22ce8_split147ce64e_seed2026_cfg6c3be430_code365dbc7_validation.parquet` | 27.22% | 393 |
| UltraBERT | `ultrabert380_v0_data40a22ce8_split147ce64e_seed2026_cfg5a387d29_code365dbc7_validation.parquet` | 34.44% | 354 |
| Q06-E3 | `qwen06_best_epoch03_v0_from3789_code8422c8c_validation.parquet` | **36.11%** | 345 |
| Q17 | `qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfg5d2d686c_code667e2d0_validation.parquet` | 31.11% | 372 |
| Q06-A | `qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg42943789_code8422c8c_validation.parquet` | 32.22% | 366 |
| Q06-B | `qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg5cb8998d_code365dbc7_validation.parquet` | 34.44% | 354 |
| Q06-cont4 | `qwen06_v0_continue4_fromc313b204_seed2026_cfgcc195d58_code8422c8c_validation.parquet` | 30.19% | 377 |

There is no completed Qwen4, Llama, Gemma, Ministral, SmolLM3, Granite, or CatBoost validation prediction artifact in this inventory yet. The running model queue may add them later.

## Pairwise error overlap

Cells show error-set Jaccard: `both wrong / wrong in either model`. Higher means more of the models' mistakes are shared. All seven files were aligned by row ID and checked against the same labels.

|  | SVC | UltraBERT | Q06-E3 | Q17 | Q06-A | Q06-B | Q06-cont4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SVC** | 100% | 60.3% | 59.7% | 61.1% | 60.8% | 60.6% | 62.4% |
| **UltraBERT** | 60.3% | 100% | 60.0% | 61.7% | 58.6% | 62.0% | 58.2% |
| **Q06-E3** | 59.7% | 60.0% | 100% | 63.3% | 70.1% | 70.1% | 69.1% |
| **Q17** | 61.1% | 61.7% | 63.3% | 100% | 64.4% | 62.8% | 63.9% |
| **Q06-A** | 60.8% | 58.6% | 70.1% | 64.4% | 100% | 72.7% | 77.3% |
| **Q06-B** | 60.6% | 62.0% | 70.1% | 62.8% | 72.7% | 100% | 74.9% |
| **Q06-cont4** | 62.4% | 58.2% | 69.1% | 63.9% | 77.3% | 74.9% | 100% |

Overlap is substantial across all families (58.2–63.3% for comparisons across the representative families). It is higher between repeated Qwen 0.6B runs (69.1–77.3%), so combining multiple Qwen 0.6B checkpoints mostly repeats errors.

## Equal-weight pair results

The following are arithmetic means of the saved class probabilities, evaluated on the shared validation rows. Gain is relative to the stronger individual member.

| Pair | Pair accuracy | Stronger member | Gain |
|---|---:|---:|---:|
| UltraBERT + Q06-E3 | **38.15%** | 36.11% | **+2.04 pp** |
| SVC + Q06-E3 | 37.59% | 36.11% | +1.48 pp |
| SVC + UltraBERT | 35.56% | 34.44% | +1.11 pp |
| Q06-E3 + Q17 | 34.81% | 36.11% | -1.30 pp |
| UltraBERT + Q17 | 32.59% | 34.44% | -1.85 pp |
| SVC + Q17 | 31.11% | 31.11% | 0.00 pp |

The best pair is UltraBERT + the Qwen 0.6B epoch 3 checkpoint. Their error Jaccard is 60.0%; they make different errors on 175 rows. This is enough complementarity for a small measured gain despite substantial overlap. Adding SVC to that pair gives 37.78%, below the pair. Averaging all four representative models (SVC, UltraBERT, Q06-E3, Q17) gives 36.11%, equal to the best single model.

The two existing ensemble reports are consistent with this result: the ensemble using Q06-E3 reports UltraBERT + Q06-E3 at 38.15%; its train-fitted weighted blend reports 37.22%. The older ensemble built around Q06-B reports 36.48% for its best ensemble. The validation numbers are evaluation results, not a basis for changing ensemble weights.

## Implications

- Prefer one Qwen 0.6B checkpoint, currently Q06-E3, rather than averaging several Qwen 0.6B runs; their errors overlap heavily.
- The strongest current candidate is the equal-weight UltraBERT + Q06-E3 pair. Its gain is modest (11 additional correct rows out of 540, about 2.04 percentage points).
- Qwen 1.7B has lower accuracy and does not improve the Q06-E3 blend in this run.
- Wait for completed predictions from the newly queued model families, then extend the same row-aligned overlap report. Keep any weight fitting on training predictions and use validation only for the final comparison.

## Method

- Predicted class: argmax over each parquet's `prob::<class>` columns.
- Error set: validation rows where predicted class differs from `y_true`.
- Pair overlap: Jaccard similarity of the two error sets.
- Equal-weight ensemble: arithmetic mean of member probabilities for each class, then argmax.
- No validation-based weight tuning was performed for this report.
