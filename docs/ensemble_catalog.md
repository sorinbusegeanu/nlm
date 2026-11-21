# Dynamic ensemble catalog

The [active E-series catalog](../configs/ensemble_catalog_v2.json) reads the latest Trust List 2 and scores cached probabilities on all 2,699 V0 rows using five prompt-group folds. It saves predictions, resolved checkpoints, rules, and input hashes in a new output directory. The same JSON documents the later E14-E21 experiments under `experiments`; the dashboard’s **Ensemble catalog** tab loads and displays both the executable strategies and these research runs, including live E21 queue status.

```bash
python -m src.ensemble_catalog --list
python -m src.ensemble_catalog
python -m src.ensemble_catalog --strategy E08
```

| ID | Ensemble | Selection rule |
|---|---|---|
| E01 | Top-10 Mean | Equal mean of the ten highest-balanced-accuracy checkpoints. |
| E02 | Family-Champion Mean | Equal mean of up to ten best distinct-family champions. |
| E03 | Label-Coverage Blend | Greedily add ten checkpoints to maximize the mean best F1 across labels, then blend equally. |
| E04 | Greedy Mean-10 | Add checkpoints while they improve the fitting balanced accuracy of the equal mean, capped at ten. |
| E05 | Precision-First Cascade | Highest precision per label and precision-first order, with E01 as fallback. |
| E06 | High-Precision Recall Cascade | Highest recall per label among rules with at least 90% precision, with E01 as fallback. |
| E07 | Residual F1 Cascade | Highest-precision rules followed by complementary F1 rules on unclaimed rows, with E01 as fallback. |
| E08 | Shared-Roster Precision Blend | Capped LC3/LC4 replacement. Select at most ten checkpoints for all labels and blend probabilities per label by precision. The full-V0 deployment roster and weights are saved separately from the held-out fold rosters. |
| E09 | Label F1 Mean-5 | Blend the five best supported rules per label, weighted by F1. |
| E10 | Top-Two Specialist | Switch E01's top label to its runner-up when a checkpoint rule has positive net gain and sufficient fitting support for that directed label pair. |
| E11 | Confusion-Aware Evidence Ensemble | Select up to 15 checkpoints for complementary top-1 coverage. Score every label using checkpoint confusion likelihoods, a uniform prior, and a penalty for correlated errors. |
| E12-1 | Error-Aware Specialist Arbitration | Choose from the top two candidates using 15 checkpoints. |
| E12-2 | Error-Aware Specialist Arbitration | Choose from the top three candidates using 15 checkpoints. |
| E12-3 | Error-Aware Specialist Arbitration | Choose from the top three candidates using 20 checkpoints. |
| E13-A | Learned Reliability Router · checkpoint | CatBoost estimates each selected checkpoint's top-1 correctness; choose the highest-reliability checkpoint from a 15-checkpoint fold-local roster. |
| E13-B | Learned Reliability Router · family max | Aggregate reliability by predicted label, taking each family's maximum and averaging family scores across a 15-checkpoint roster. |
| E13-C | Learned Reliability Router · family max · 20 checkpoints | Same label-level family-max routing with a 20-checkpoint roster. |

E01–E10 use at most 10 checkpoints in each fold and deployment roster. E11 and E12-1/E12-2 use at most 15, and E12-3 uses at most 20. E05–E07 and E10 select rules only from the top ten fitting balanced-accuracy checkpoints, which also supply their fallback. E09 selects a shared pool of ten by label coverage before choosing its five F1 rules per label. Eligible checkpoints compete across the whole Trust List 2 pool during roster selection. E02 uses family membership only to limit each family to one champion. Rule-based strategies require at least ten fitting predictions per rule. The checkpoint count is the number of distinct checkpoints in the full-V0 deployment roster, including cascade and specialist checkpoints as well as fallback members. Fold rosters are saved separately. V0 was previously used for checkpoint selection, so these scores are development results.

E11 starts with the highest fitting balanced-accuracy checkpoint. It adds checkpoints by the count of previously uncovered fitting rows where their top-1 prediction is correct. Coverage ties use standalone balanced accuracy, accuracy, and ascending sequence ID. Selection stops at 15 or when no checkpoint adds coverage.

Each E11 confusion matrix records true labels in rows and predicted labels in columns. Additive smoothing uses `(count + alpha) / (true-label support + 18 * alpha)`. Evidence weight is `1 / (1 + gamma * sum of positive error correlations with other selected checkpoints)`. Correlations involving a constant error vector are zero. E11 sums the weighted log likelihoods for each possible true label and chooses the highest score in canonical label order.

Within each outer fitting partition, three inner prompt-group folds select alpha from `[0.5, 1, 2, 5]` and gamma from `[0, 0.5, 1, 2]`. Each inner fit selects its own roster and estimates its own confusion matrices and correlations. Pooled inner accuracy selects the parameters, followed by balanced accuracy, larger alpha, and larger gamma on ties. Outer held-out labels are used only for reporting. The full-V0 deployment fit repeats that inner selection and saves the canonical labels, prior, ordered checkpoint IDs, smoothed matrices, counts, correlation weights, and parameters.

E11 results include fold-local top-1 oracle coverage, per-label precision/recall/FP/FN, and corrected/lost counts against E08 when both are evaluated together. `predictions.parquet` stores held-out predictions under `E11` and the separate full-V0 fitted predictions under `deployment::E11`. `predict_confusion_ensemble` in `src/confusion_ensemble.py` replays the saved deployment parameters on checkpoint probabilities without fitting. All selection and fitting use CPU and cached probabilities.

E12 selects complementary checkpoints and fits a symmetric pairwise logistic classifier within each fitting fold. Three nested prompt-group folds select regularization. The catalog saves a full-V0 deployment model separately from its five held-out scores. See [E12 evaluation](e12.md) for historical reference scores and the initial experiment details.

E13 selects its 15- or 20-checkpoint roster from fitting-fold labels, then trains a CPU CatBoost checkpoint-correctness router using four prompt-group cross-fit partitions. Each training feature record excludes its prompt group from roster selection and reliability/confusion statistics; early-stopping features use a separate group-disjoint split. E13-A chooses the checkpoint with highest estimated correctness. E13-B/C take the maximum score within each family predicting a label and average those family scores. All three are scored in the five outer prompt-group folds. Run `python -m src.e13_reliability_router` for the three configurations or pass `--strategy E13-B` to run one. Models and reproducibility artifacts are saved under `artifacts/analysis/e13/`.
