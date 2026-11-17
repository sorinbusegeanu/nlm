# ORACLE02 — Learned stacking ensemble

A supervised meta-classifier over the saved V0 probability vectors from Trust List 2 v025. Base checkpoints were not loaded or retrained.

## Validation and exposure limitation

The 2,699 V0 rows are scored with the existing five prompt-group folds. Hyperparameter and blend weight selection use three inner prompt-group folds restricted to each outer training partition. Feature scalers are fitted inside each inner/final training fit. The 78-checkpoint and EQUAL9 rosters are fixed. The base checkpoints and Trust List roster were selected/developed using V0, so outer-fold stacking does not remove that exposure; results are diagnostics, not independent Kaggle estimates.

C3 is the existing EQUAL9 + three-rule cascade (`N0071:gpt-6-astra`, `N0105:gemini-3.1-pro-preview`, `N0048:claude-opus-4-6`) with EQUAL9 fallback. Its stored artifact provides OOF hard labels only, so its log loss is reported as N/A.

## Baselines

| Baseline | Accuracy | Balanced accuracy | Log loss | Correct / 2,699 |
|---|---:|---:|---:|---:|
| EQUAL9 | 65.3946% | 65.3898% | 1.132076 | 1,765 |
| C3 | 65.5057% | 65.5006% | N/A | 1,768 |

## Cross-fitted held-out comparison

| Checkpoints | Meta-classifier | Accuracy | Balanced accuracy | Log loss | Δ BA vs EQUAL9 | Δ BA vs C3 | Corrected / introduced vs EQUAL9 | vs C3 | Disagreements vs EQUAL9 / C3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | logistic | 64.9129% | 64.9060% | 1.179628 | -0.484 pp | -0.595 pp | 250 / 263 | 251 / 267 | 778 / 778 |
| 9 | catboost | 65.9133% | 65.9063% | 1.315677 | +0.517 pp | +0.406 pp | 176 / 162 | 176 / 165 | 480 / 483 |
| 9 | catboost_equal9_blend | 66.6173% | 66.6120% | 1.229250 | +1.222 pp | +1.111 pp | 103 / 70 | 107 / 77 | 259 / 273 |
| 78 | logistic | 66.4320% | 66.4258% | 1.371964 | +1.036 pp | +0.925 pp | 290 / 262 | 289 / 264 | 792 / 786 |
| 78 | catboost | 66.0986% | 66.0882% | 1.316055 | +0.698 pp | +0.588 pp | 222 / 203 | 221 / 205 | 621 / 611 |
| 78 | catboost_equal9_blend | 67.1730% | 67.1670% | 1.199166 | +1.777 pp | +1.666 pp | 108 / 60 | 111 / 66 | 238 / 242 |

## Outer-fold training versus held-out results

Training columns are the mean apparent scores on each outer fold's training rows after inner-fold selection. Held-out columns pool each row's single outer-fold prediction.

| Checkpoints | Strategy | Train accuracy | Train balanced accuracy | Train log loss | Held-out accuracy | Held-out balanced accuracy | Held-out log loss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 9 | logistic | 89.4406% ± 0.457% | 89.4391% ± 0.456% | 0.367118 | 64.9129% | 64.9060% | 1.179628 |
| 9 | catboost | 69.2757% ± 0.327% | 69.2702% ± 0.325% | 1.266307 | 65.9133% | 65.9063% | 1.315677 |
| 9 | catboost_equal9_blend | 68.2290% ± 0.463% | 68.2243% ± 0.463% | 1.202540 | 66.6173% | 66.6120% | 1.229250 |
| 78 | logistic | 100.0000% ± 0.000% | 100.0000% ± 0.000% | 0.020916 | 66.4320% | 66.4258% | 1.371964 |
| 78 | catboost | 70.0538% ± 0.610% | 70.0436% ± 0.607% | 1.257101 | 66.0986% | 66.0882% | 1.316055 |
| 78 | catboost_equal9_blend | 68.3588% ± 0.832% | 68.3530% ± 0.831% | 1.173114 | 67.1730% | 67.1670% | 1.199166 |

## Per-label recall

P/R/F1, TP, FP and FN for every label and all six held-out experiments are in `artifacts/analysis/oracle02/per_label_metrics.csv`. Training-fold per-label metrics are recorded in `metrics.json`.

## Nested selection settings

- Logistic regression: L2 multinomial, C grid `[0.1, 1.0, 10.0]`, max_iter 500, inner pooled balanced accuracy with log-loss tie-break.
- CatBoost: CPU, one thread, depth 3–4, 120 iterations, L2 leaf regularization 10 or 20, learning rate 0.05, feature subsampling rsm=0.25, Bernoulli subsample 0.85; selected on inner prompt-group OOF balanced accuracy with log-loss tie-break.
- Blend: `alpha * CatBoost + (1-alpha) * EQUAL9`, alpha selected from `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` using inner-fold held-out predictions only.
- The feature matrix includes per-checkpoint 18-class probabilities, top-1 one-hot, max probability, top-1/top-2 margin, entropy, class agreement counts, and EQUAL9 probabilities. Logistic scaling is fit within training folds.

## Artifacts

- `run_manifest.json`, `feature_manifest.json`: input hashes, exact rosters, feature definitions and fold/configuration provenance.
- `folds/<roster>/outer_<fold>.json/.npz`: inner selection, fold predictions, training predictions, and resumable state.
- `folds/<roster>/models_outer_<fold>/`: fitted scaler/logistic and CatBoost models for each outer fold.
- `row_predictions.parquet`: ID-aligned full-OOF probabilities and labels for all six experiments plus EQUAL9 and C3.
- `metrics.json`, `per_label_metrics.csv`: overall, per-fold, training, held-out and per-label scores.

All artifacts are derived from cached V0 predictions. No base-model inference, base-model training or Kaggle submission was performed.
