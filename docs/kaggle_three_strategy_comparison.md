# Trust List 2 v025: C3, equal-selected, and C1 Kaggle submissions

## Frozen configurations

All checkpoints were resolved against `docs/trust-list-2-v025.csv`. Test inference uses the saved Kaggle offline inference path and the canonical 18-label order. No training, production-ensemble change, or queue change is part of this run.

**EQUAL9 fallback** (equal probability weight, `1/9` each): N0071, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048.

**C3 high-precision cascade:** EQUAL9 fallback, after ordered rules N0071 → `gpt-6-astra`, N0105 → `gemini-3.1-pro-preview`, N0048 → `claude-opus-4-6`. The full-V0 rules have supports 100, 130, and 88 and precisions 93.00%, 92.31%, and 90.91%, respectively. They meet the documented 10-row minimum support and ≥90% precision thresholds.

**equal-selected:** equal probability weight, `1/20` each: N0003, N0005, N0006, N0009, N0022, N0030, N0034, N0044, N0048, N0055, N0056, N0057, N0058, N0061, N0071, N0072, N0076, N0080, N0081, N0105.

**C1 precision cascade:** EQUAL9 fallback; ordered by descending full-V0 precision, with one selected rule per label:

| Priority | Checkpoint | Label |
|---:|---|---|
| 1 | N0056 | `gemini-3.1-pro-preview` |
| 2 | N0105 | `gpt-6-astra` |
| 3 | N0048 | `claude-opus-4-6` |
| 4 | N0076 | `claude-opus-5` |
| 5 | N0105 | `qwen3.8-max` |
| 6 | N0034 | `claude-opus-4-8` |
| 7 | N0081 | `gpt-5.6-luna` |
| 8 | N0064 | `gemini-3.6-flash` |
| 9 | N0061 | `claude-sonnet-5` |
| 10 | N0022 | `glm-5.3-flash` |
| 11 | N0055 | `gpt-5.5` |
| 12 | N0057 | `gpt-5.6-sol` |
| 13 | N0054 | `kimi-k3` |
| 14 | N0074 | `gemini-3.7-flash` |
| 15 | N0059 | `gpt-5.6-terra` |
| 16 | N0030 | `gemini-3.8-flash` |
| 17 | N0009 | `glm-5.3` |
| 18 | N0003 | `claude-fable-5` |

## V0 validation

Full-V0 metrics are diagnostic because the final rules/checkpoint selections use V0. Cross-fit values are the five-fold prompt-group results already recorded in the source analyses.

| Strategy | Cross-fit accuracy | Cross-fit balanced accuracy | Full-V0 diagnostic accuracy | Full-V0 diagnostic balanced accuracy |
|---|---:|---:|---:|---:|
| EQUAL9 reference | 65.3946% | 65.3898% | 65.3946% | 65.3898% |
| C3 high-precision cascade | 65.5057% | 65.5006% | 65.8021% | 65.7972% |
| equal-selected | 64.8018% | 64.7962% | 65.0611% | 65.0562% |
| C1 precision cascade | 64.8018% | 64.8006% | 66.1727% | 66.1718% |

The rule order and 20-member list were checked against their source JSON, and the recomputed full-V0 accuracy and balanced accuracy match the recorded diagnostics. Details are saved in `artifacts/kaggle/three_strategy_comparison/full_v0_reproduction.json`. One Kaggle inference run scored the 24-checkpoint union in 1,618.38 seconds and generated all three strategy outputs. The test IDs and order are `00353fb1c541`, `005f5a9a6139`, `0062a53d5487`; all three strategies predict `glm-5.3`, `gpt-5.5`, and `gemini-3.1-pro-preview`, respectively. All output hashes match (`0cc980a510ebb3513dbeb259ca79dd818cd412b119c547ca40a2da13409feaaa`), including the prior EQUAL9/C3 test output.

## Kaggle submissions

All three requested variants were already submitted, so no duplicate submissions were created. The current Kaggle API status is:

| Strategy | Submission description | Submission ID | Status | Public score returned | Difference vs v025 EQUAL9 |
|---|---|---:|---|---:|---:|
| C3 high-precision cascade | `C3 high-precision cascade Trust List 2 v025` | 58446640 | Complete | **0.569** | **+0.002** |
| equal-selected | `Equal-selected equal-weight ensemble Trust List 2 v025` | 58446713 | Complete | Not returned | N/A |
| C1 precision cascade | `C1 highest-precision cascade Trust List 2 v025` | 58446716 | Complete | Not returned | N/A |

The comparison baseline is the completed EQUAL9 v025 submission `58444305`, which returned 0.567. The C3 and equal-selected/C1 Parquet outputs all have the same SHA-256 (`0cc980a510ebb3513dbeb259ca79dd818cd412b119c547ca40a2da13409feaaa`) and identical predictions for the three test IDs. I also downloaded the completed EQUAL9 v025 kernel output; it has the same SHA-256 and predictions. Kaggle nevertheless reports 0.569 for C3 versus 0.567 for EQUAL9, and no score for equal-selected or C1. The recorded +0.002 cannot be attributed to the cascade because all submitted prediction files are byte-identical. This report records returned scores only and does not infer scores for equal-selected/C1.

The three submissions were produced by completed kernel versions: C3 from `zodrak/equal9-v025-plus-c3-cascade` v2, equal-selected from `zodrak/equal-selected-trust-list-2-v025` v1, and C1 from `zodrak/c1-precision-trust-list-2-v025` v1. Each kernel's `submission.parquet` was downloaded and checked for exact test-ID order, unique IDs, the required `id,model` schema, non-null values, and valid competition labels.

## Artifacts

- Kaggle checkpoint package: `artifacts/kaggle/three-strategy-checkpoints/`
- Kaggle inference notebook: `artifacts/kaggle/three-strategy-kernel/trust_v025_three_strategies.ipynb`
- Test outputs, V0 reproduction check, and submission manifest are saved under `artifacts/kaggle/three_strategy_comparison/`.
