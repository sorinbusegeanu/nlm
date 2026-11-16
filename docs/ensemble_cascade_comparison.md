# Trust List 2 v025 cascade comparison

Cache-only, prompt-group cross-fitted comparison on the 2,699 official V0 rows. No model training, model loading, new inference, production-ensemble changes, or Kaggle submissions were performed.

Trust List 2: `trust-list-2-v025`; `docs/trust-list-2-v025.csv`; SHA-256 `c93ae4cc0612ed36b7c126d6d50a5b50456ecfafd4e835369272076442e8cf9b`. The 78-checkpoint probability cache is verified against this list and the ordered V0 IDs. The existing five GroupKFold prompt partitions were reused.

The cross-fitted scores are the primary comparison. Checkpoints were previously selected using V0, so even held-out prompt folds do not remove checkpoint-selection exposure. Full-V0-selected cascades are optimistic diagnostics only.

EQUAL9 baseline: **65.3898% balanced accuracy**. Rule selection uses only each fold's other four partitions. A rule requires at least 10 matching rows in its fitting partition. Rules fire in listed order; unmatched rows use EQUAL9.

## Overall results

| Scope | Configuration | Rules | Accuracy | Balanced accuracy | Δ BA vs EQUAL9 | Corrected | Introduced | Rows differing | Fallback rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v0_diagnostic | C6_greedy_swaps | 22 | 69.0997% | 69.0987% | +3.709 pp | 253 | 153 | 529 | 1001 |
| full_v0_diagnostic | C1_precision | 18 | 66.1727% | 66.1718% | +0.782 pp | 118 | 97 | 288 | 1241 |
| full_v0_diagnostic | C4_precision | 36 | 65.8392% | 65.8357% | +0.446 pp | 290 | 278 | 754 | 140 |
| full_v0_diagnostic | C3_precision | 3 | 65.8021% | 65.7972% | +0.407 pp | 17 | 6 | 28 | 2381 |
| full_v0_diagnostic | C3_f1 | 3 | 65.8021% | 65.7972% | +0.407 pp | 17 | 6 | 28 | 2381 |
| full_v0_diagnostic | C3_precision_recall | 3 | 65.8021% | 65.7972% | +0.407 pp | 17 | 6 | 28 | 2381 |
| full_v0_diagnostic | C1_f1 | 18 | 65.7651% | 65.7644% | +0.375 pp | 116 | 106 | 297 | 1241 |
| full_v0_diagnostic | C1_precision_recall | 18 | 65.7651% | 65.7644% | +0.375 pp | 116 | 106 | 297 | 1241 |
| full_v0_diagnostic | C2_precision | 18 | 65.7280% | 65.7231% | +0.333 pp | 227 | 218 | 601 | 402 |
| full_v0_diagnostic | C5_precision | 35 | 65.7280% | 65.7216% | +0.332 pp | 270 | 261 | 742 | 139 |
| full_v0_diagnostic | weighted_per_label_f1 | — | 65.5428% | 65.5364% | +0.147 pp | 265 | 261 | 723 | — |
| full_v0_diagnostic | C5_precision_recall | 35 | 65.2834% | 65.2777% | -0.112 pp | 264 | 267 | 726 | 159 |
| full_v0_diagnostic | C5_f1 | 35 | 65.2093% | 65.2036% | -0.186 pp | 266 | 271 | 737 | 157 |
| full_v0_diagnostic | C2_f1 | 18 | 65.0982% | 65.0935% | -0.296 pp | 223 | 231 | 610 | 402 |
| full_v0_diagnostic | C2_precision_recall | 18 | 65.0982% | 65.0935% | -0.296 pp | 223 | 231 | 610 | 402 |
| full_v0_diagnostic | C4_f1 | 36 | 65.0611% | 65.0574% | -0.332 pp | 287 | 296 | 778 | 109 |
| full_v0_diagnostic | C4_precision_recall | 36 | 65.0611% | 65.0574% | -0.332 pp | 287 | 296 | 778 | 109 |
| full_v0_diagnostic | weighted_equal_selected | — | 65.0611% | 65.0562% | -0.334 pp | 120 | 129 | 350 | — |
| full_v0_diagnostic | weighted_complementarity | — | 64.3201% | 64.3137% | -1.076 pp | 264 | 293 | 800 | — |
| full_v0_diagnostic | precision_weighted_vote | — | 62.3564% | 62.3480% | -3.042 pp | 162 | 244 | 588 | — |
| full_v0_diagnostic | majority_vote | — | 61.1338% | 61.1283% | -4.261 pp | 134 | 249 | 564 | — |
| prompt_group_oof | C3_precision | 2.8 avg | 65.5057% | 65.5006% | +0.111 pp | 18 | 15 | 43 | 2375 |
| prompt_group_oof | C3_f1 | 2.8 avg | 65.5057% | 65.5006% | +0.111 pp | 18 | 15 | 43 | 2375 |
| prompt_group_oof | C3_precision_recall | 2.8 avg | 65.5057% | 65.5006% | +0.111 pp | 18 | 15 | 43 | 2375 |
| prompt_group_oof | C1_precision | 18.0 avg | 64.8018% | 64.8006% | -0.589 pp | 103 | 119 | 310 | 1232 |
| prompt_group_oof | weighted_equal_selected | — | 64.8018% | 64.7962% | -0.594 pp | 111 | 127 | 344 | — |
| prompt_group_oof | C6_greedy_swaps | 19.4 avg | 64.7647% | 64.7621% | -0.628 pp | 185 | 202 | 516 | 1129 |
| prompt_group_oof | C1_precision_recall | 18.0 avg | 64.5795% | 64.5784% | -0.811 pp | 104 | 126 | 320 | 1232 |
| prompt_group_oof | C1_f1 | 18.0 avg | 64.5054% | 64.5046% | -0.885 pp | 105 | 129 | 323 | 1232 |
| prompt_group_oof | C2_precision | 18.0 avg | 64.4313% | 64.4263% | -0.963 pp | 203 | 229 | 593 | 429 |
| prompt_group_oof | C2_f1 | 18.0 avg | 64.2460% | 64.2411% | -1.149 pp | 204 | 235 | 600 | 429 |
| prompt_group_oof | C2_precision_recall | 18.0 avg | 64.1719% | 64.1670% | -1.223 pp | 203 | 236 | 601 | 429 |
| prompt_group_oof | C5_precision | 34.4 avg | 63.2827% | 63.2769% | -2.113 pp | 233 | 290 | 739 | 170 |
| prompt_group_oof | C5_f1 | 34.4 avg | 63.2827% | 63.2764% | -2.113 pp | 240 | 297 | 758 | 149 |
| prompt_group_oof | weighted_per_label_f1 | — | 63.2086% | 63.2021% | -2.188 pp | 222 | 281 | 707 | — |
| prompt_group_oof | C5_precision_recall | 34.4 avg | 62.9863% | 62.9806% | -2.409 pp | 235 | 300 | 760 | 148 |
| prompt_group_oof | weighted_complementarity | — | 62.5787% | 62.5722% | -2.818 pp | 237 | 313 | 796 | — |
| prompt_group_oof | C4_precision | 36.0 avg | 62.0971% | 62.0927% | -3.297 pp | 223 | 312 | 752 | 151 |
| prompt_group_oof | precision_weighted_vote | — | 62.0600% | 62.0515% | -3.338 pp | 160 | 250 | 594 | — |
| prompt_group_oof | C4_f1 | 36.0 avg | 61.6525% | 61.6480% | -3.742 pp | 223 | 324 | 746 | 181 |
| prompt_group_oof | C4_precision_recall | 36.0 avg | 61.3931% | 61.3888% | -4.001 pp | 213 | 321 | 743 | 149 |
| prompt_group_oof | majority_vote | — | 61.1338% | 61.1283% | -4.261 pp | 134 | 249 | 563 | — |

`weighted_equal_selected`, `weighted_per_label_f1`, and `weighted_complementarity` are the previously reported probability ensembles, reused from the same v025 cache and fold assignments. Family-normalized majority and precision-weighted votes were also refit using the v025 predictions, with fit-fold-only per-label support and precision, then applied to held-out rows.

## Best three distinct cross-fitted cascades

Ranked by prompt-group OOF balanced accuracy. Ordering variants with identical OOF predictions are collapsed so the list shows distinct cascades.

1. **C3_precision** — 65.5006% balanced accuracy (+0.111 pp vs EQUAL9).
   - Fold 1: N0105:gemini-3.1-pro-preview → N0048:claude-opus-4-6 → N0106:gpt-6-astra.
   - Fold 2: N0071:gpt-6-astra → N0105:gemini-3.1-pro-preview → N0048:claude-opus-4-6.
   - Fold 3: N0048:claude-opus-4-6 → N0071:gpt-6-astra → N0105:gemini-3.1-pro-preview.
   - Fold 4: N0105:gemini-3.1-pro-preview → N0075:claude-opus-5 → N0106:gpt-6-astra.
   - Fold 5: N0105:gemini-3.1-pro-preview → N0071:gpt-6-astra.
2. **C1_precision** — 64.8006% balanced accuracy (-0.589 pp vs EQUAL9).
   - Fold 1: N0081:gemini-3.1-pro-preview → N0105:gpt-6-astra → N0048:claude-opus-4-6 → N0071:claude-opus-5 → N0105:qwen3.8-max → N0034:claude-opus-4-8 → N0081:gpt-5.6-luna → N0064:gemini-3.6-flash → N0048:claude-sonnet-5 → N0055:gpt-5.5 → N0022:glm-5.3-flash → N0057:gpt-5.6-sol → N0040:kimi-k3 → N0058:gpt-5.6-terra → N0065:gemini-3.8-flash → N0009:glm-5.3 → N0080:gemini-3.7-flash → N0003:claude-fable-5.
   - Fold 2: N0105:gpt-6-astra → N0056:gemini-3.1-pro-preview → N0048:claude-opus-4-6 → N0076:claude-opus-5 → N0055:claude-opus-4-8 → N0077:qwen3.8-max → N0081:gpt-5.6-luna → N0034:claude-sonnet-5 → N0064:gemini-3.6-flash → N0057:gpt-5.6-sol → N0022:glm-5.3-flash → N0054:kimi-k3 → N0055:gpt-5.5 → N0058:gpt-5.6-terra → N0030:gemini-3.8-flash → N0011:glm-5.3 → N0048:gemini-3.7-flash → N0003:claude-fable-5.
   - Fold 3: N0056:gemini-3.1-pro-preview → N0105:gpt-6-astra → N0048:claude-opus-4-6 → N0068:claude-opus-5 → N0034:claude-opus-4-8 → N0076:qwen3.8-max → N0081:gpt-5.6-luna → N0064:gemini-3.6-flash → N0055:claude-sonnet-5 → N0042:kimi-k3 → N0057:gpt-5.6-sol → N0057:gpt-5.5 → N0022:glm-5.3-flash → N0080:gemini-3.7-flash → N0038:gemini-3.8-flash → N0009:glm-5.3 → N0059:gpt-5.6-terra → N0003:claude-fable-5.
   - Fold 4: N0056:gemini-3.1-pro-preview → N0061:claude-opus-5 → N0105:gpt-6-astra → N0048:claude-opus-4-6 → N0076:qwen3.8-max → N0034:claude-opus-4-8 → N0061:claude-sonnet-5 → N0064:gemini-3.6-flash → N0081:gpt-5.6-luna → N0022:glm-5.3-flash → N0055:gpt-5.5 → N0040:kimi-k3 → N0057:gpt-5.6-sol → N0059:gpt-5.6-terra → N0030:gemini-3.8-flash → N0053:gemini-3.7-flash → N0009:glm-5.3 → N0003:claude-fable-5.
   - Fold 5: N0058:gemini-3.1-pro-preview → N0105:gpt-6-astra → N0081:claude-opus-5 → N0048:claude-opus-4-6 → N0081:gpt-5.6-luna → N0070:qwen3.8-max → N0034:claude-opus-4-8 → N0061:claude-sonnet-5 → N0064:gemini-3.6-flash → N0074:gemini-3.7-flash → N0054:kimi-k3 → N0022:glm-5.3-flash → N0057:gpt-5.6-sol → N0055:gpt-5.5 → N0059:gpt-5.6-terra → N0009:glm-5.3 → N0048:gemini-3.8-flash → N0009:claude-fable-5.
3. **C6_greedy_swaps** — 64.7621% balanced accuracy (-0.628 pp vs EQUAL9).
   - Fold 1: N0081:gpt-6-astra → N0055:gpt-5.5 → N0081:gemini-3.1-pro-preview → N0009:glm-5.3 → N0005:glm-5.3-flash → N0003:claude-fable-5 → N0105:gemini-3.6-flash → N0034:claude-opus-4-6 → N0005:gemini-3.7-flash → N0105:qwen3.8-max → N0007:claude-fable-5 → N0058:gpt-5.6-luna → N0061:claude-opus-5 → N0003:glm-5.3 → N0074:gemini-3.1-pro-preview → N0055:gemini-3.7-flash → N0009:gpt-5.6-luna → N0038:claude-opus-5 → N0081:claude-opus-4-8.
   - Fold 2: N0022:glm-5.3-flash → N0034:claude-opus-4-6 → N0081:gpt-6-astra → N0081:gemini-3.1-pro-preview → N0105:gemini-3.6-flash → N0055:gemini-3.7-flash → N0009:glm-5.3 → N0081:gpt-5.6-luna → N0038:claude-opus-5 → N0009:claude-fable-5 → N0081:claude-opus-4-8 → N0005:gemini-3.7-flash → N0005:glm-5.3 → N0003:claude-fable-5 → N0030:gemini-3.1-pro-preview → N0034:claude-sonnet-5 → N0060:gpt-5.6-luna.
   - Fold 3: N0105:gemini-3.1-pro-preview → N0022:glm-5.3-flash → N0009:glm-5.3 → N0106:claude-fable-5 → N0034:claude-opus-4-6 → N0009:gpt-5.6-luna → N0081:gpt-6-astra → N0105:gemini-3.6-flash → N0005:gemini-3.7-flash → N0055:gemini-3.7-flash → N0055:gpt-5.5 → N0074:claude-opus-5 → N0081:claude-opus-4-8 → N0055:claude-sonnet-5 → N0005:glm-5.3 → N0006:claude-fable-5 → N0059:claude-opus-5 → N0076:claude-sonnet-5 → N0066:gemini-3.1-pro-preview → N0048:gemini-3.6-flash → N0064:qwen3.8-max → N0081:gpt-5.6-luna.
   - Fold 4: N0022:glm-5.3-flash → N0081:gpt-6-astra → N0081:gemini-3.1-pro-preview → N0055:gpt-5.5 → N0009:glm-5.3 → N0014:claude-fable-5 → N0053:gemini-3.7-flash → N0105:gemini-3.1-pro-preview → N0005:glm-5.3 → N0070:gemini-3.6-flash → N0038:claude-opus-5 → N0081:claude-opus-4-8 → N0080:claude-opus-4-6 → N0055:gemini-3.6-flash → N0078:claude-opus-5 → N0003:claude-fable-5 → N0030:glm-5.3-flash → N0003:gpt-5.6-sol → N0071:gpt-5.6-luna → N0009:gpt-5.6-luna → N0080:gemini-3.7-flash.
   - Fold 5: N0081:gpt-6-astra → N0105:gemini-3.1-pro-preview → N0009:glm-5.3-flash → N0009:claude-fable-5 → N0081:gpt-5.6-luna → N0001:glm-5.3 → N0061:gemini-3.7-flash → N0034:claude-opus-4-6 → N0055:gemini-3.6-flash → N0007:gemini-3.7-flash → N0055:gpt-5.5 → N0105:qwen3.8-max → N0009:glm-5.3 → N0058:gpt-6-astra → N0076:gemini-3.6-flash → N0001:glm-5.3-flash → N0037:claude-opus-5 → N0048:claude-opus-4-6.

## Per-label OOF metrics for EQUAL9 and the three leading distinct cascades

Cells show precision / recall / F1. Per-label values for every ordering and weighted comparator are in `per_label_metrics.csv`.

| Label | EQUAL9 P/R/F1 | C3_precision | C1_precision | C6_greedy_swaps |
|---|---:|---:|---:|---:|
| claude-fable-5 | 0.406 / 0.523 / 0.457 | 0.403 / 0.517 / 0.453 | 0.444 / 0.617 / 0.517 | 0.406 / 0.577 / 0.476 |
| claude-opus-4-6 | 0.857 / 0.800 / 0.828 | 0.857 / 0.800 / 0.828 | 0.879 / 0.773 / 0.823 | 0.801 / 0.807 / 0.804 |
| claude-opus-4-8 | 0.757 / 0.767 / 0.762 | 0.760 / 0.760 / 0.760 | 0.736 / 0.727 / 0.732 | 0.732 / 0.673 / 0.701 |
| claude-opus-5 | 0.908 / 0.787 / 0.843 | 0.855 / 0.787 / 0.819 | 0.854 / 0.780 / 0.815 | 0.780 / 0.733 / 0.756 |
| claude-sonnet-5 | 0.688 / 0.733 / 0.710 | 0.688 / 0.733 / 0.710 | 0.658 / 0.707 / 0.682 | 0.713 / 0.647 / 0.678 |
| gemini-3.1-pro-preview | 0.959 / 0.780 / 0.860 | 0.914 / 0.853 / 0.883 | 0.918 / 0.820 / 0.866 | 0.904 / 0.820 / 0.860 |
| gemini-3.6-flash | 0.637 / 0.807 / 0.712 | 0.684 / 0.793 / 0.735 | 0.669 / 0.700 / 0.684 | 0.663 / 0.787 / 0.720 |
| gemini-3.7-flash | 0.660 / 0.220 / 0.330 | 0.673 / 0.220 / 0.332 | 0.538 / 0.280 / 0.368 | 0.478 / 0.500 / 0.489 |
| gemini-3.8-flash | 0.500 / 0.833 / 0.625 | 0.500 / 0.833 / 0.625 | 0.496 / 0.800 / 0.612 | 0.606 / 0.647 / 0.626 |
| glm-5.3 | 0.398 / 0.453 / 0.424 | 0.405 / 0.453 / 0.428 | 0.463 / 0.460 / 0.462 | 0.491 / 0.540 / 0.514 |
| glm-5.3-flash | 0.521 / 0.487 / 0.503 | 0.514 / 0.473 / 0.493 | 0.583 / 0.587 / 0.585 | 0.529 / 0.540 / 0.535 |
| gpt-5.5 | 0.659 / 0.580 / 0.617 | 0.659 / 0.580 / 0.617 | 0.613 / 0.633 / 0.623 | 0.625 / 0.633 / 0.629 |
| gpt-5.6-luna | 0.719 / 0.613 / 0.662 | 0.710 / 0.587 / 0.642 | 0.691 / 0.627 / 0.657 | 0.608 / 0.620 / 0.614 |
| gpt-5.6-sol | 0.641 / 0.547 / 0.590 | 0.659 / 0.540 / 0.593 | 0.610 / 0.500 / 0.549 | 0.647 / 0.440 / 0.524 |
| gpt-5.6-terra | 0.560 / 0.773 / 0.650 | 0.562 / 0.760 / 0.646 | 0.541 / 0.700 / 0.610 | 0.586 / 0.633 / 0.609 |
| gpt-6-astra | 0.900 / 0.720 / 0.800 | 0.844 / 0.760 / 0.800 | 0.922 / 0.713 / 0.805 | 0.798 / 0.867 / 0.831 |
| kimi-k3 | 0.678 / 0.533 / 0.597 | 0.684 / 0.533 / 0.599 | 0.605 / 0.480 / 0.535 | 0.717 / 0.440 / 0.545 |
| qwen3.8-max | 0.722 / 0.813 / 0.765 | 0.725 / 0.807 / 0.763 | 0.713 / 0.760 / 0.735 | 0.764 / 0.753 / 0.758 |

## Full-V0 diagnostics

These configurations select rules and score them on the same V0 rows; do not treat them as held-out performance. Exact selected sequences are in `selected_rules.json`.

## Incremental rule contributions

Held-out rule handling, unique true positives, additional false positives and incremental precision/recall/F1 are in `oof_rule_contributions.csv`; full-V0 diagnostics are in `full_v0_rule_contributions.csv`. Incremental recall is measured against still-unclaimed true rows of the rule's label. The final EQUAL9 fallback row reports remaining examples.

## Rule metrics and overlap

`checkpoint_rule_metrics.csv` contains full-V0 precision, recall, F1, TP, FP and support for every checkpoint-label pair (18 labels × 78 checkpoints). `same_label_pairwise_overlap.csv` gives exact shared and unique TP/FP counts for every same-label rule pair. `rule_row_membership.npz` stores the row-level firing matrix, V0 IDs and true labels so overlaps can be reconstructed exactly.

## Rule stability

`rule_stability.csv` reports how often each exact checkpoint-label rule is selected across the five fit folds and its mean cascade position. C3 labels without a ≥90% precision rule fall back to EQUAL9; fold-wise and full-V0 diagnostic label lists are in `c3_fallback_labels.json`.

## Per-label metrics

Full precision, recall, F1, support and correct counts for all cascade, voting and previously reported probability-ensemble configurations are in `per_label_metrics.csv`.

## Reproducibility

- `experiment_config.json`: trust/cache hashes, fold source, selection thresholds and strategy definitions.
- `fold_assignments.csv`: prompt-group fold for every V0 row.
- `selected_rules.json`: fold-specific and full-V0 rule orders.
- `cascade_predictions.parquet`: row-aligned OOF/full predictions plus prior weighted ensemble predictions.
- `metrics.csv`, `per_label_metrics.csv`, and contribution/overlap files: machine-readable outputs.
- `analysis_state.json`: completion and input hashes.
