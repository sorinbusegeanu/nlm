# Trust List 2 W6/W7 fixed-weight V0 evaluation

These are exploratory full-V0 results. The checkpoints were selected on V0 and earlier ensemble optimization also used V0.
No inference was run: all seven probability matrices were loaded from saved V0 caches, checked for exact response-ID alignment, and aligned to the canonical 18-label order.

## Comparison

| Ensemble | V0 accuracy | V0 balanced accuracy | Correct | Δ accuracy vs WEIGHTED5 | Δ balanced accuracy vs WEIGHTED5 | Corrected | Introduced errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| WEIGHTED5 | 60.3927% | 60.3888% | 1,630/2,699 | — | — | — | — |
| W6-Qwen | 61.2820% | 61.2786% | 1,654/2,699 | +0.889 pp | +0.890 pp | 59 | 35 |
| W6-Filtered | 61.2820% | 61.2784% | 1,654/2,699 | +0.889 pp | +0.890 pp | 64 | 40 |
| W7-Combined | 61.9489% | 61.9456% | 1,672/2,699 | +1.556 pp | +1.557 pp | 95 | 53 |

**Best by V0 balanced accuracy:** W7-Combined (61.9456%).

## Exact baseline and fixed formulas

Baseline: successful Trust List 2 v013 Kaggle WEIGHTED5, using the same five checkpoint probability caches and saved WEIGHTED5 weights:

| Sequence | Model | Weight | Cached V0 prediction source | Max length / processing note |
|---|---|---:|---|---|
| N0071 | ultrabert380 | 0.2811918866145629 | `artifacts/predictions/ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgb9e735ff_code81bce4a_validation.parquet` | max_length=581; inference_batch_size=8; calibration_temperature=1.0 |
| N0056 | granite4_h_micro | 0.2324715095557317 | `artifacts/analysis/trust_list_2_v0/granite4_h_micro_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg768a0d47_coded9e8831.parquet` | max_length=538; inference_batch_size=8; calibration_temperature=0.6332207877222997 |
| N0005 | catboost_stylometry | 0.1833526736593965 | `artifacts/analysis/trust_list_2_v0/catboost_stylometry_catboost_v1_a5_s2029_datada64f007_split94737c59_seed2029_cfg263b702b_code667e2d0.parquet` | max_length=None; inference_batch_size=8; calibration_temperature=0.49994443572162695; CatBoost temperature calibration |
| N0058 | ministral3_3b | 0.2345455729093069 | `artifacts/analysis/trust_list_2_v0/ministral3_3b_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfg4619b410_code279cc81.parquet` | max_length=476; inference_batch_size=8; calibration_temperature=0.668458265948126 |
| N0072 | svc | 0.0684383572610019 | `artifacts/predictions/svc_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfg66dc1893_code81bce4a_validation.parquet` | max_length=None; inference_batch_size=8; calibration_temperature=1.0; SVC char+word feature views |

Additional cached checkpoints:

- N0074 (`qwen06`, V2.14): `artifacts/predictions/qwen06_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgba46767d_code81bce4a_validation.parquet`.
- N0076 (`ultrabert380`, V2.14-prediction-filtered-80pct): `artifacts/predictions/ultrabert380_v2.14-prediction-filtered-80pct_valv0_dataee63cf16_splitab1d2edc_seed2026_cfgf9f6cf6c_code9c2e334_validation.parquet`.

Fixed probability formulas:

- W6-Qwen = `0.85 × WEIGHTED5 + 0.15 × N0074`.
- W6-Filtered = `0.90 × WEIGHTED5 + 0.10 × N0076`.
- W7-Combined = `0.75 × WEIGHTED5 + 0.15 × N0074 + 0.10 × N0076`.

## Per-label accuracy changes

Accuracy is correct predictions divided by V0 support for that true label. Deltas are percentage points versus WEIGHTED5.

| Label | V0 support | WEIGHTED5 | W6-Qwen Δ | W6-Filtered Δ | W7-Combined Δ |
|---|---:|---:|---:|---:|---:|
| claude-fable-5 | 149 | 49.7% | +2.7 pp | +2.0 pp | +3.4 pp |
| claude-opus-4-6 | 150 | 74.7% | +3.3 pp | +2.7 pp | +5.3 pp |
| claude-opus-4-8 | 150 | 59.3% | +1.3 pp | +3.3 pp | +5.3 pp |
| claude-opus-5 | 150 | 74.0% | -0.7 pp | -2.0 pp | -1.3 pp |
| claude-sonnet-5 | 150 | 74.0% | -0.7 pp | -0.7 pp | -1.3 pp |
| gemini-3.1-pro-preview | 150 | 74.0% | +0.7 pp | +0.7 pp | +0.7 pp |
| gemini-3.6-flash | 150 | 76.7% | +1.3 pp | +0.0 pp | +2.0 pp |
| gemini-3.7-flash | 150 | 16.0% | -2.0 pp | -4.0 pp | -5.3 pp |
| gemini-3.8-flash | 150 | 85.3% | +0.7 pp | -0.7 pp | +0.7 pp |
| glm-5.3 | 150 | 36.7% | +1.3 pp | +3.3 pp | +1.3 pp |
| glm-5.3-flash | 150 | 28.0% | +1.3 pp | -2.0 pp | -2.7 pp |
| gpt-5.5 | 150 | 63.3% | +1.3 pp | +1.3 pp | +4.0 pp |
| gpt-5.6-luna | 150 | 61.3% | +2.7 pp | +2.0 pp | +4.0 pp |
| gpt-5.6-sol | 150 | 50.7% | +0.7 pp | +1.3 pp | +0.0 pp |
| gpt-5.6-terra | 150 | 69.3% | +0.0 pp | +6.0 pp | +5.3 pp |
| gpt-6-astra | 150 | 70.0% | +2.0 pp | +2.7 pp | +5.3 pp |
| kimi-k3 | 150 | 48.7% | +2.7 pp | +3.3 pp | +4.0 pp |
| qwen3.8-max | 150 | 75.3% | -2.7 pp | -3.3 pp | -2.7 pp |

### Per-experiment improvements and regressions

- **W6-Qwen:** improved claude-fable-5, claude-opus-4-6, claude-opus-4-8, gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.8-flash, glm-5.3, glm-5.3-flash, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-6-astra, kimi-k3; regressed claude-opus-5, claude-sonnet-5, gemini-3.7-flash, qwen3.8-max; unchanged gpt-5.6-terra.
- **W6-Filtered:** improved claude-fable-5, claude-opus-4-6, claude-opus-4-8, gemini-3.1-pro-preview, glm-5.3, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3; regressed claude-opus-5, claude-sonnet-5, gemini-3.7-flash, gemini-3.8-flash, glm-5.3-flash, qwen3.8-max; unchanged gemini-3.6-flash.
- **W7-Combined:** improved claude-fable-5, claude-opus-4-6, claude-opus-4-8, gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.8-flash, glm-5.3, gpt-5.5, gpt-5.6-luna, gpt-5.6-terra, gpt-6-astra, kimi-k3; regressed claude-opus-5, claude-sonnet-5, gemini-3.7-flash, glm-5.3-flash, qwen3.8-max; unchanged gpt-5.6-sol.

## Artifacts and interpretation

- Row-level predictions and all 18 probability columns: `artifacts/analysis/trust2_w6_w7/v0_ensemble_predictions.parquet`.
- Per-label accuracy table: `artifacts/analysis/trust2_w6_w7/per_label_accuracy.csv`.
- Machine-readable metrics, formulas, cache paths and hashes: `artifacts/analysis/trust2_w6_w7/evaluation_metrics.json`.
- Kaggle public score: WEIGHTED5 `0.564` as previously recorded; new blends are not scored here.
- Baseline artifacts and production ensemble selection were left unchanged.
