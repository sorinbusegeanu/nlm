# Trust List 2 Greedy Checkpoint Subset Selection

Trust List 2 trust-list-2-v013, full V0 (2,699 rows), 68 checkpoints, 1,216 eligible rules; validation hash `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.

Greedy selection starts from the best single checkpoint by balanced accuracy, then adds the candidate with the largest balanced-accuracy gain. Family divisors are recomputed for every tested subset. Ties go to the lower Trust List sequence ID. This procedure is greedy and does not guarantee a globally optimal subset.

## Trust-list cleanup

Trust List 2 v013 was created from v012 after removing N0028, N0029, N0033, N0041 and N0043. Their V1.5 training partition contained 2,159 V0 rows, matching both response IDs and normalized response text. The other 68 v012 checkpoints were checked against their complete local training sources; no additional V0 ID or normalized-text overlap was found. The V1 CatBoost ablation runs were traced to the V1 source by matching the manifest dataset hash, and that complete source had zero V0 overlap. The full 73-checkpoint audit and the removed rows are preserved under `artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v012/`.

## Full-V0 greedy selection

Reference full-roster ensemble: 1,658/2,699 correct, accuracy 61.4302%, balanced accuracy 61.4216%. N0122 UltraBERT: 1,518/2,699, accuracy 56.2431%, balanced accuracy 56.2384%.

| Checkpoints | Selected IDs (family) | Accuracy | Balanced accuracy | Correct | Δ Acc / BAcc vs full roster (pp) | Δ Acc / BAcc vs N0122 (pp) | Checkpoint payload (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | N0071 (ultrabert), N0056 (granite), N0005 (catboost), N0058 (mistral), N0072 (svc) | 63.6162% | 63.6087% | 1,717 | +2.186 / +2.187 | +7.373 / +7.370 | 3,459.9 |
| 6 | N0071 (ultrabert), N0056 (granite), N0005 (catboost), N0058 (mistral), N0072 (svc), N0030 (gemma) | 64.0978% | 64.0895% | 1,730 | +2.668 / +2.668 | +7.855 / +7.851 | 3,886.4 |
| 7 | N0071 (ultrabert), N0056 (granite), N0005 (catboost), N0058 (mistral), N0072 (svc), N0030 (gemma), N0061 (smollm) | 64.8018% | 64.7927% | 1,749 | +3.372 / +3.371 | +8.559 / +8.554 | 4,207.4 |
| 8 | N0071 (ultrabert), N0056 (granite), N0005 (catboost), N0058 (mistral), N0072 (svc), N0030 (gemma), N0061 (smollm), N0044 (svc) | 65.0611% | 65.0539% | 1,756 | +3.631 / +3.632 | +8.818 / +8.816 | 4,966.4 |
| 9 | N0071 (ultrabert), N0056 (granite), N0005 (catboost), N0058 (mistral), N0072 (svc), N0030 (gemma), N0061 (smollm), N0044 (svc), N0009 (catboost) | 65.4316% | 65.4246% | 1,766 | +4.001 / +4.003 | +9.189 / +9.186 | 4,980.6 |
| 10 | N0071 (ultrabert), N0056 (granite), N0005 (catboost), N0058 (mistral), N0072 (svc), N0030 (gemma), N0061 (smollm), N0044 (svc), N0009 (catboost), N0048 (ultrabert) | 65.5428% | 65.5352% | 1,769 | +4.113 / +4.114 | +9.300 / +9.297 | 7,868.0 |

Accuracy deltas are also available in `full_v0_subset_metrics.csv`. The selected IDs and individual checkpoint payloads are in `full_v0_selected_checkpoints.csv`.

## Training-overlap audit

The full-V0 greedy scores are selection-set scores. OOF selection fits reliability statistics and subset order only on fold training prompts, but checkpoint roster membership and prior checkpoint/trust-list selection used full V0; OOF is therefore a diagnostic, not a fully independent estimate. Training-data audit found no V0 ID or normalized-response overlap among the 68 active checkpoints. It hash-verified 37/68 recorded partitions; 31 runs lacked a directly verifiable local training partition, but their complete source datasets were scanned where locally available.

Checkpoint-level audit: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/checkpoint_training_overlap_audit.csv`.

## Five-fold prompt-group OOF results

Each fold recomputed per-label precision/support and the minimum support rule using only its training prompts. Greedy selection used those fitting rows; its held-out predictions were pooled across the five folds. Fold-fit full-roster reference: accuracy 61.3190%, balanced accuracy 61.3105%. N0122 reference: accuracy 56.2431%, balanced accuracy 56.2384%.

| Checkpoints | OOF accuracy | OOF balanced accuracy | Correct | Δ Acc / BAcc vs fold-fit full roster (pp) | Δ Acc / BAcc vs N0122 (pp) | Mean selected payload per fold (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 62.2082% | 62.2003% | 1,679 | +0.889 / +0.890 | +5.965 / +5.962 | 5,141.5 |
| 6 | 62.3935% | 62.3855% | 1,684 | +1.074 / +1.075 | +6.150 / +6.147 | 5,688.1 |
| 7 | 62.3194% | 62.3102% | 1,682 | +1.000 / +1.000 | +6.076 / +6.072 | 6,172.3 |
| 8 | 62.3935% | 62.3833% | 1,684 | +1.074 / +1.073 | +6.150 / +6.145 | 6,683.0 |
| 9 | 62.4305% | 62.4213% | 1,685 | +1.112 / +1.111 | +6.187 / +6.183 | 8,281.4 |
| 10 | 62.4676% | 62.4589% | 1,686 | +1.149 / +1.148 | +6.225 / +6.220 | 8,793.6 |

Fold-specific selected checkpoint IDs are saved in `fold_selected_checkpoints.json`; OOF predictions are in `oof_predictions/`.

## Inference-cost proxy and limitations

Checkpoint payload MiB is the sum of actual files under each selected Trust List checkpoint path. It estimates model payload and loading/storage burden; it is not a measured latency or throughput estimate. Repeated checkpoints from one architecture are counted separately because they are separate saved checkpoints.

The full-V0 greedy scores are selection-set scores. OOF selection fits reliability statistics and subset order only on fold training prompts, but checkpoint roster membership and prior checkpoint/trust-list selection used full V0; OOF is therefore a diagnostic, not a fully independent estimate. Training-data audit found no V0 ID or normalized-response overlap among the 68 active checkpoints. It hash-verified 37/68 recorded partitions; 31 runs lacked a directly verifiable local training partition, but their complete source datasets were scanned where locally available.

## Artifacts

- Selected lists and complete metrics: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/selected_checkpoints.json`
- Full-V0 predictions: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/full_v0_predictions`
- OOF predictions: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/oof_predictions`
- Full-V0 metrics: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/full_v0_subset_metrics.csv`
- OOF metrics: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/prompt_group_oof_subset_metrics.csv`
- Greedy selection trace: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/full_v0_greedy_steps.csv`
- Manifest/data training-overlap audit: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/checkpoint_training_overlap_audit.csv`
