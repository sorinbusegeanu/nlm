# Trust List 2 v013 Top-Five Submission Experiments

The checkpoint list is frozen from the v013 5-checkpoint greedy subset experiment. Selection order is preserved across all three strategies.

| Strategy | Five checkpoint IDs, in selection order | Full-V0 accuracy | Fixed-five grouped OOF accuracy | Kaggle submission |
|---|---|---:|---:|---|
| TRUST5 | N0071, N0056, N0005, N0058, N0072 | 63.62% (1,717/2,699) | 63.36% (1,710/2,699) | `submission_trust5.parquet` |
| EQUAL5 | N0071, N0056, N0005, N0058, N0072 | 61.69% (1,665/2,699) | 61.69% (1,665/2,699) | `submission_equal5.parquet` |
| WEIGHTED5 | N0071, N0056, N0005, N0058, N0072 | 60.39% (1,630/2,699) | 60.54% (1,634/2,699) | `submission_weighted5.parquet` |

Balanced accuracy was respectively 63.61%, 61.69%, and 60.39% on full V0; fixed-five OOF balanced accuracy was 63.35%, 61.69%, and 60.54%.

## TRUST5 rule

For each eligible checkpoint-label prediction in v013, the score contribution is `precision × prediction_count / (prediction_count + 30)`. Contributions from checkpoints in the same model family are divided by the number of selected checkpoints in that family, then summed by predicted class. The highest score wins; canonical class order breaks exact ties. If no selected checkpoint produces a vote with an eligible v013 rule, fallback is N0071, the selected checkpoint with highest recorded overall V0 accuracy (56.24%). No full-V0 row required fallback. The replay exactly matched the saved v013 k05 predictions.

WEIGHTED5 uses `src.ensemble_factory.fit_weights` with nonnegative weights summing to one, multiclass log loss, and L2 regularization (`l2=0.05`) toward equal weights. Weights fit on all V0 for the Kaggle test predictions:

| Checkpoint | Weight |
|---|---:|
| N0071 | 0.281192 |
| N0056 | 0.232472 |
| N0005 | 0.183353 |
| N0058 | 0.234546 |
| N0072 | 0.068438 |

## OOF reference and interpretation

The existing v013 subset-selection OOF result was also rechecked from its saved prediction artifact: 1,679/2,699 = **62.21%**. That historical procedure refit rule statistics and selected checkpoint subsets separately within each fold, so members could differ by fold. To respect the frozen-five requirement, the table reports new grouped OOF scores with the same five checkpoints in every fold; only trust statistics or WEIGHTED5 weights are fit on each fold’s other prompt groups. All V0 results remain development diagnostics because V0 contributed to checkpoint and trust-rule selection.

The five checkpoints are N0071 UltraBERT (`v2.14`), N0056 Granite H Micro (`v1_all`), N0005 CatBoost + Stylometry (`catboost_v1_a5_s2029`), N0058 Ministral 3 3B (`v1_all`), and N0072 SVC (`v2.14`). All five are present in the v013 audited 68-checkpoint roster; the audit reports zero V0 training-ID and normalized-response overlap. N0005’s original training partition hash was not directly verified, so its audit instead scans the complete V1 source dataset, which also found zero V0 overlap. Excluded IDs N0028, N0029, N0033, N0041, and N0043 are absent.

Test inference used the saved checkpoint manifests and settings, loaded each model sequentially, and cached one aligned probability matrix per checkpoint. The GPU was occupied by another active UltraBERT training process, so inference ran on CPU; Granite used Transformers’ reference PyTorch Mamba/causal-convolution functions instead of the CUDA extensions. This did not change the checkpoint or inference parameters, but CPU reference kernels may produce small numerical differences from GPU kernels. All three strategies consumed the same five cached test probability matrices. The local `data/raw/test.csv` contains three rows.

## Files

- [TRUST5 submission](../artifacts/kaggle/trust2_v013_top5/submission_trust5.parquet)
- [EQUAL5 submission](../artifacts/kaggle/trust2_v013_top5/submission_equal5.parquet)
- [WEIGHTED5 submission](../artifacts/kaggle/trust2_v013_top5/submission_weighted5.parquet)
- [Experiment manifest](../artifacts/kaggle/trust2_v013_top5/experiment_manifest.json)
- Per-checkpoint test probabilities: `artifacts/kaggle/trust2_v013_top5/checkpoint_predictions/`
- V0 and fixed-five OOF predictions: `artifacts/kaggle/trust2_v013_top5/v0_evaluation_predictions.parquet`

## Kaggle submissions

All three variants use the same five checkpoints. Kaggle requires the submitted output to be named exactly `submission.parquet`, so each strategy was run as the primary output of a completed kernel version.

| Strategy | Kernel version | Submission ID | Kaggle status | Public score |
|---|---:|---:|---|---:|
| TRUST5 | 3 | 58307956 | Complete | 0.551 |
| EQUAL5 | 6 | 58312421 | Pending | — |
| WEIGHTED5 | 7 | 58312651 | Pending | — |

The EQUAL5 and WEIGHTED5 files were validated against the same Kaggle-run test IDs and valid label set before submission. Their visible test predictions happen to be identical, though their probability blending rules differ. The full five-checkpoint inference run used the same frozen roster and saved WEIGHTED5 coefficients; the kernel log records the active strategy for versions 6 and 7.
