# E31-Confusion — Three-Model Confusion Diagnosis

This is the completed confusion-diagnosis study. The active E31 checkpoint expansion is documented separately in [docs/e31.md](e31.md) and its current progress is listed in the experiment catalogue.

## Results at a glance

| Analysis | Result | Net change / interpretation |
|---|---:|---|
| Fixed-roster E13-B router, five-fold OOF | 63.90% balanced accuracy | 1,725 / 2,699 correct |
| Historical E13-B v037 saved OOF | 64.65% balanced accuracy | Historical folds used fold-specific rosters |
| Dedicated 3-label classifier | 57.68% accuracy and balanced accuracy | 259 / 449 correct when forced to choose among the three target labels |
| E19-D reconstructed-group Hungarian, full V0 | 75.65% balanced accuracy | +317 net-correct overall; +36 among target labels vs fixed-roster independent scores |
| E19-D reconstructed-group Hungarian, 50 incomplete scenarios | 70.78% mean whole-dataset BA | +1,707 net-correct scenario-row outcomes; target-label net +197 |
| True-prompt Hungarian, 50 incomplete scenarios | 71.28% mean whole-dataset BA | +1,828 net-correct scenario-row outcomes; diagnostic upper bound |

The group-constrained results are the strongest signal in this experiment. E19-D improved whole-dataset balanced accuracy in all 50 incomplete scenarios and raised the target labels' mean balanced accuracy by 4.57 percentage points. These are repeated evaluations of reused V0 responses, not 50 independent validation sets.

## Protocol and inputs

- Base probabilities: cached Trust List 2 v037 probabilities; no base model inference or training.
- Roster: the exact 15-checkpoint E13-B deployment roster, fixed in every outer fold: `N0110, N0054, N0008, N0127, N0082, N0065, N0124, N0117, N0133, N0097, N0120, N0033, N0059, N0113, N0071`.
- Outer folds: the five saved E13-B prompt-group folds. For each held fold, E13 reliability statistics and its CatBoost router were fit using only the other four folds; training correctness records used inner prompt-group OOF feature construction.
- Classifier features: E13's existing checkpoint correctness feature frame and family-max routing code. The three-label classifier uses the fixed roster's 15 × 18 cached class probabilities.
- Grouping: frozen E28 external groupers E19-F-C, E19-F-A and E19-D. Groups were reconstructed from only the rows in the current full-V0 or 500-row scenario batch before Hungarian assignment. True prompt groups were used only for the oracle diagnostic.
- V0 support: 150 rows each for `glm-5.3` and `glm-5.3-flash`; 149 for `claude-fable-5`.
- Execution: cached probabilities and features; CPU analysis, 24-thread environment; runtime 416.5 seconds.

The fixed-roster OOF result is not expected to reproduce historical E13-B exactly: the saved E13-B OOF used a separately selected 15-checkpoint roster in each fold (37 checkpoint IDs across its five folds). The fixed deployment roster has lower OOF balanced accuracy here: 63.90% versus the historical 64.65% (−0.74 pp). This follows the requested fixed-roster protocol and is the correct baseline for the comparisons below.

| Fold | Fixed-roster BA | Saved E13-B BA |
|---:|---:|---:|
| 1 | 62.59% | 63.33% |
| 2 | 62.22% | 63.89% |
| 3 | 67.41% | 66.30% |
| 4 | 60.93% | 63.70% |
| 5 | 66.39% | 66.02% |

## 1. Class separability

The dedicated classifier was a `StandardScaler` plus balanced multinomial logistic regression (`C=1`, `max_iter=2000`). It was trained only on the three target labels and evaluated out of fold by prompt. Its probabilities choose among the three labels only; this is a diagnostic of their separability, not a full 18-label replacement.

| Metric | Dedicated 3-class OOF |
|---|---:|
| Accuracy | 57.68% (259 / 449) |
| Balanced accuracy | 57.68% |
| Mean rank of the correct target label | 1.58 / 3 |
| Correct label ranked first / in top 2 | 259 / 449; 377 / 449 |
| Pairwise AUC: GLM-5.3 vs GLM-5.3-Flash | 0.777 |
| Pairwise AUC: GLM-5.3 vs Claude Fable 5 | 0.756 |
| Pairwise AUC: GLM-5.3-Flash vs Claude Fable 5 | 0.844 |

Three-class confusion matrix (rows and columns in the order GLM-5.3, GLM-5.3-Flash, Claude Fable 5):

| True \\ Predicted | GLM-5.3 | GLM-5.3-Flash | Claude Fable 5 |
|---|---:|---:|---:|
| GLM-5.3 | 78 | 35 | 37 |
| GLM-5.3-Flash | 32 | 95 | 23 |
| Claude Fable 5 | 38 | 25 | 86 |

The historical E13-B full 18-class predictions correctly classified 178 / 449 target-label rows (39.64% when predictions outside these three labels count as misses). The specialist classifier correctly classified 259 / 449, a gain of **81 correct target rows** in this forced three-way task. E13-B's full 18-class correct-label score ranked the true target label first for 169 / 449 rows, top three for 385, and top five for 434. The data therefore contain useful pairwise discrimination, but a deployed specialist needs a reliable gate so it does not override the other 15 labels.

## 2. Label-quality screen

For each outer training partition, suspicious rows were flagged only within that training partition if either (a) an inner prompt-group OOF three-class classifier disagreed with the row's assigned label, or (b) at least 9 of 15 cached checkpoint top-1 predictions disagreed. Cross-label duplicate evidence was checked using `data_factory._normal_text()` and character 3–5-gram TF-IDF cosine similarity ≥0.95. No exact or near-duplicate rows were found. Flags indicate disagreement, **not verified label errors**.

| Fold | Target training rows | Flagged | Removed per label: GLM / Flash / Claude Fable | Rows left for training |
|---:|---:|---:|---:|---:|
| 1 | 359 | 301 | 104 / 103 / 94 | 58 |
| 2 | 359 | 299 | 106 / 98 / 95 | 60 |
| 3 | 359 | 305 | 105 / 104 / 96 | 54 |
| 4 | 359 | 304 | 103 / 103 / 98 | 55 |
| 5 | 360 | 298 | 103 / 98 / 97 | 62 |

Across the training-fold appearances, the vote-disagreement rule flagged 1,424 row-folds (356 distinct rows); inner-CV classifier disagreement flagged 786 row-folds (281 distinct rows). The union contains 387 distinct rows across the 449 target examples. No duplicate-based flags were generated. For the random-removal control, the same number was removed from each label as the flagged variant, using a fixed fold-specific seed.

| Training set | Held-out accuracy / BA | Corrected vs all | Introduced vs all | Net correct vs all |
|---|---:|---:|---:|---:|
| A. All target training examples | 57.68% / 57.68% | — | — | — |
| B. Remove suspicious rows | 53.01% / 53.04% | 66 | 87 | −21 |
| C. Remove equal per-label random count | 46.99% / 47.01% | 51 | 99 | −48 |

Removing flagged rows was 6.01 pp better than removing the same number at random, but still reduced accuracy by 4.68 pp versus using all rows. The filter removes roughly 83–85% of the target training examples in each fold, leaving only 54–62 total rows. This does **not** establish label noise; it is compatible with a smaller training set losing useful variation. Do not delete these rows based on this screen.

## 3. Group constraints

All decoders used the identical fixed-roster held-out 18-class score matrices. Counts in the table are corrected / introduced / net-correct predictions versus independent fixed-roster classification.

### Full V0 (2,699 rows)

| Decoder | Whole accuracy | Whole BA | Whole corrected / introduced / net | Target accuracy | Target BA | Target corrected / introduced / net |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 63.91% | 63.90% | — | 37.64% | 37.65% | — |
| E19-F-C + Hungarian | 72.47% | 72.46% | 341 / 110 / **+231** | 43.88% | 43.88% | 65 / 37 / **+28** |
| E19-F-A + Hungarian | 75.40% | 75.39% | 451 / 141 / **+310** | 45.88% | 45.89% | 86 / 49 / **+37** |
| E19-D + Hungarian | 75.66% | 75.65% | 451 / 134 / **+317** | 45.66% | 45.66% | 85 / 49 / **+36** |
| True prompt groups + Hungarian | 76.77% | 76.76% | 499 / 152 / **+347** | 45.66% | 45.66% | 94 / 58 / **+36** |

Target-label recall on full V0:

| Decoder | GLM-5.3 | GLM-5.3-Flash | Claude Fable 5 |
|---|---:|---:|---:|
| Independent | 27.33% | 43.33% | 42.28% |
| E19-F-C | 37.33% | 48.67% | 45.64% |
| E19-F-A | 38.67% | 50.00% | 48.99% |
| E19-D | **40.00%** | 48.00% | **48.99%** |
| True prompt groups | 40.67% | **49.33%** | 46.98% |

Target confusion matrices below include a fourth prediction column for any of the other 15 labels. Rows are GLM-5.3, GLM-5.3-Flash, Claude Fable 5; columns are those same three labels plus `other 15`.

| Decoder | GLM-5.3 row | GLM-5.3-Flash row | Claude Fable 5 row |
|---|---|---|---|
| Independent | 41, 44, 32, 33 | 27, 65, 25, 33 | 18, 24, 63, 44 |
| E19-F-C | 56, 38, 26, 30 | 37, 73, 22, 18 | 26, 26, 68, 29 |
| E19-F-A | 58, 40, 30, 22 | 34, 75, 22, 19 | 28, 23, 73, 25 |
| E19-D | 60, 41, 28, 21 | 36, 72, 24, 18 | 29, 21, 73, 26 |
| True prompt groups | 61, 41, 31, 17 | 32, 74, 24, 20 | 32, 21, 70, 26 |

### Fifty incomplete V0 scenarios

Each scenario contains 500 rows. The table reports scenario mean and standard deviation, minimum, 10th percentile and maximum. Changes are summed across 25,000 row-scenario outcomes; because responses recur in different scenarios, these are **not unique-response counts**.

| Decoder | Whole accuracy mean | Whole BA mean ± SD | Min / P10 / max BA | Scenarios better / worse | Whole corrected / introduced / net | Target BA mean ± SD | Target min / P10 / max BA | Target net-correct outcomes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Independent | 63.82% | 63.93% ± 2.12 pp | 59.66 / 60.82 / 68.05% | — | — | 38.47% ± 4.58 pp | 28.45 / 31.30 / 48.05% | — |
| E19-F-C | 64.16% | 64.27% ± 2.13 pp | 59.86 / 61.19 / 68.34% | 39 / 10 | 192 / 107 / **+85** | 38.49% ± 4.47 pp | 27.17 / 32.61 / 45.58% | **+1** |
| E19-F-A | 63.85% | 63.96% ± 2.11 pp | 59.65 / 60.74 / 67.80% | 24 / 26 | 234 / 228 / **+6** | 38.68% ± 4.56 pp | 29.74 / 31.75 / 46.81% | **+9** |
| E19-D | **70.65%** | **70.78% ± 2.02 pp** | **66.95 / 67.50 / 75.16%** | **50 / 0** | 2,526 / 819 / **+1,707** | **43.04% ± 5.87 pp** | **29.53 / 36.61 / 58.51%** | **+197** |
| True prompt groups | 71.14% | 71.28% ± 2.07 pp | 67.13 / 68.05 / 75.43% | 50 / 0 | 2,707 / 879 / **+1,828** | 43.11% ± 6.33 pp | 29.53 / 35.75 / 60.71% | **+199** |

Mean target recall over the 50 scenarios:

| Decoder | GLM-5.3 | GLM-5.3-Flash | Claude Fable 5 |
|---|---:|---:|---:|
| Independent | 27.99% | 42.82% | 44.59% |
| E19-F-C | 28.85% | 42.58% | 44.04% |
| E19-F-A | 29.23% | 42.56% | 44.25% |
| E19-D | **37.72%** | **46.71%** | 44.68% |
| True prompt groups | 37.90% | 46.86% | 44.58% |

E19-D's target-label gains are concentrated in GLM-5.3 and GLM-5.3-Flash; Claude Fable 5 changes little across incomplete scenarios. Its 50-scenario target net gain (+197) is close to the true-group diagnostic (+199), while the whole-dataset gap is larger: 70.78% versus 71.28% mean balanced accuracy.

## Hypothesis assessment

1. **The labels are separable in checkpoint evidence — supported.** The three-way classifier reached 57.68% OOF balanced accuracy, pairwise AUCs ranged from 0.756 to 0.844, and it beat E13-B's forced target-label accuracy by 81 correct rows. This supports a gated specialist or reranker, not a standalone three-class deployment.
2. **Training labels contain detectable noise — not established.** Disagreement is common, but no cross-label exact or near duplicates were found. Removing flagged rows hurt versus using all examples; the difference from random removal is not proof that flagged labels are wrong.
3. **Prompt-group constraints resolve some confusion — strongly supported on this V0 diagnostic.** E19-D improves 50/50 incomplete scenarios, with +197 target-label correct outcomes summed over repeated scenario rows and +1,707 whole-dataset correct outcomes. The oracle adds only two target-label correct outcomes across those repeated scenarios, but remains about 0.50 pp higher in whole-dataset mean BA.

## Recommended next action

Prioritize an **externally evaluated, confidence-gated group assignment** using the frozen E19-D grouping path, with E13-B independent scores as fallback. The next validation must use genuinely new labeled prompt groups and must not use V0 for selecting the gate or threshold. In parallel, the dedicated three-way classifier is a promising signal for a target-label reranker; evaluate it as a gated correction on external held-out prompts before deployment.

Do not filter the dataset based on the disagreement screen. It did not verify attribution mistakes, and aggressive removal reduced three-class accuracy.

## Limitations

- V0 is reused for router folds, model/roster development, and diagnosis. Although this run fits its router statistics and model inside each outer training partition, the fixed 15-checkpoint roster was selected for the historical E13-B deployment using V0. These scores are not independent Kaggle generalization estimates.
- The three-class classifier sees only the three true target classes during training and is evaluated on target-label rows. Its metrics do not include competition examples from the other 15 classes; a gate is required.
- Disagreement flags identify hard or inconsistent examples, not ground-truth label errors. The source data do not provide authoritative provenance for adjudicating those labels.
- `v0_group` consists of repeated V0 responses. Scenario statistics characterize sensitivity to incomplete group composition; they are correlated and cannot be treated as 50 independent test sets.
- E19-D reconstructed groups are model-derived and imperfect. The true-prompt Hungarian results are diagnostic upper bounds and cannot be used at inference.

## Reproducibility and artifacts

Run with:

```bash
OPENBLAS_NUM_THREADS=24 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
  python -u scripts/run_e31_confusion.py
```

The reproducible implementation is [src/e31_confusion.py](../src/e31_confusion.py). Results, fold routers, fixed-roster score matrices, three-class fold models and OOF probabilities, row-level suspicion flags, full-V0 predictions, and per-scenario outputs are saved under [artifacts/analysis/e31_confusion/](../artifacts/analysis/e31_confusion/). Input hashes and the experiment configuration are in `metrics.json` and `configuration.json`.

