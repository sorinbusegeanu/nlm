# ORACLE03 — Closing the Oracle Accuracy Gap

All base probability matrices are cached Trust List 2 v025 predictions for the same 2,699 V0 rows. No base checkpoints were inferred or trained. Text embeddings use the local frozen EmbeddingGemma2 snapshot; stylometry is deterministic and row-local. Supervised text-stack and specialist fitting used the existing five outer prompt-group folds and inner prompt-group splits.

## Exposure limitation

These are development-set diagnostics, not independent Kaggle estimates. Base checkpoints, roster selection, and prior checkpoint selection have seen V0. Outer cross-fitting prevents the ORACLE03 meta-classifiers from directly fitting their held-out labels; it cannot remove base-checkpoint or roster exposure. The public benchmark is 0.581; V0 results alone do not establish a likely leaderboard gain.

## 1. Candidate coverage

Oracle coverage means the true label occurs among at least one member's top-1 labels (upper bound). The candidate-ranker coverage below asks whether that true label survives each deployable ranking's top-k shortlist. Vote-count ties are resolved by canonical label order; probability ties use stable canonical order.

| Roster | Ranking | Top 1 | Top 2 | Top 3 | Top 5 | Correct label unsupported by every member | Supported labels omitted at top 3 |
|---|---|---:|---:|---:|---:|---:|---:|
| EQUAL9 | top1_vote_count | 63.88% | 79.03% | 86.55% | 91.11% | 228 | 135 |
| EQUAL9 | summed_checkpoint_probability | 65.39% | 83.07% | 90.29% | 97.07% | 228 | 95 |
| EQUAL9 | ensemble_probability | 65.39% | 83.07% | 90.29% | 97.07% | 228 | 95 |
| H09 | top1_vote_count | 64.91% | 80.14% | 87.29% | 91.96% | 202 | 141 |
| H09 | summed_checkpoint_probability | 65.51% | 83.48% | 90.89% | 97.18% | 202 | 106 |
| H09 | ensemble_probability | 65.51% | 83.48% | 90.89% | 97.18% | 202 | 106 |
| ALL78 | top1_vote_count | 55.54% | 75.36% | 85.99% | 95.55% | 36 | 342 |
| ALL78 | summed_checkpoint_probability | 61.39% | 79.70% | 89.18% | 96.44% | 36 | 256 |
| ALL78 | ensemble_probability | 61.39% | 79.70% | 89.18% | 96.44% | 36 | 256 |
| ALL78 | ORACLE02_oof_probability | 67.17% | 84.22% | 91.14% | 97.52% | 36 | 206 |

For each roster, the oracle upper bound is the share whose true label was a top-1 proposal by at least one member. This table shows how much of that bound remains in the shortlist (coverage divided by oracle coverage); probabilities can nominate labels absent from all top-1 votes.

| Roster | Oracle upper bound | Ranking | Top 2 of oracle | Top 3 of oracle | Supported examples excluded from top 3 |
|---|---:|---|---:|---:|---:|
| EQUAL9 | 91.55% | top1_vote_count | 86.32% | 94.54% | 135 |
| EQUAL9 | 91.55% | summed_checkpoint_probability | 89.88% | 96.16% | 95 |
| EQUAL9 | 91.55% | ensemble_probability | 89.88% | 96.16% | 95 |
| H09 | 92.52% | top1_vote_count | 86.62% | 94.35% | 141 |
| H09 | 92.52% | summed_checkpoint_probability | 89.35% | 95.75% | 106 |
| H09 | 92.52% | ensemble_probability | 89.35% | 95.75% | 106 |
| ALL78 | 98.67% | top1_vote_count | 76.38% | 87.16% | 342 |
| ALL78 | 98.67% | summed_checkpoint_probability | 80.77% | 90.39% | 256 |
| ALL78 | 98.67% | ensemble_probability | 80.77% | 90.39% | 256 |
| ALL78 | 98.67% | ORACLE02_oof_probability | 85.35% | 92.26% | 206 |

### Per-label shortlist coverage at top 3

Counts are correct-label rows covered by each top-3 list; the oracle column counts rows where at least one member predicted the true label as top-1.

| Label | EQUAL9 oracle | EQUAL9 vote | EQUAL9 prob | H09 oracle | H09 vote | H09 prob | ALL78 oracle | ALL78 vote | ALL78 prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-fable-5 | 136 | 132 | 131 | 136 | 131 | 131 | 148 | 124 | 126 |
| claude-opus-4-6 | 143 | 141 | 142 | 145 | 143 | 141 | 150 | 139 | 141 |
| claude-opus-4-8 | 141 | 138 | 137 | 143 | 138 | 138 | 149 | 124 | 128 |
| claude-opus-5 | 135 | 131 | 127 | 135 | 130 | 126 | 148 | 134 | 132 |
| claude-sonnet-5 | 136 | 128 | 126 | 136 | 127 | 126 | 144 | 124 | 124 |
| gemini-3.1-pro-preview | 143 | 141 | 140 | 144 | 142 | 141 | 149 | 144 | 141 |
| gemini-3.6-flash | 147 | 145 | 148 | 149 | 147 | 148 | 150 | 146 | 147 |
| gemini-3.7-flash | 126 | 123 | 141 | 126 | 120 | 141 | 149 | 137 | 144 |
| gemini-3.8-flash | 146 | 145 | 147 | 146 | 145 | 148 | 149 | 147 | 146 |
| glm-5.3 | 134 | 117 | 136 | 135 | 120 | 139 | 149 | 107 | 131 |
| glm-5.3-flash | 138 | 122 | 138 | 140 | 123 | 140 | 148 | 118 | 132 |
| gpt-5.5 | 133 | 126 | 129 | 136 | 131 | 129 | 148 | 128 | 134 |
| gpt-5.6-luna | 141 | 129 | 134 | 142 | 128 | 135 | 150 | 125 | 127 |
| gpt-5.6-sol | 128 | 118 | 131 | 133 | 120 | 134 | 147 | 114 | 128 |
| gpt-5.6-terra | 143 | 135 | 143 | 143 | 136 | 144 | 148 | 132 | 141 |
| gpt-6-astra | 140 | 128 | 137 | 143 | 136 | 141 | 148 | 136 | 140 |
| kimi-k3 | 120 | 102 | 109 | 123 | 104 | 112 | 142 | 102 | 105 |
| qwen3.8-max | 141 | 137 | 141 | 142 | 137 | 139 | 147 | 140 | 140 |

### Candidate-set size and major confusions

For vote lists, effective candidate count is limited by the number of distinct top-1 labels proposed in that row; probability lists always contain k labels.

| Roster | Distinct top-1 labels proposed | Rows |
|---|---:|---:|
| EQUAL9 | 1 | 252 |
| EQUAL9 | 2 | 581 |
| EQUAL9 | 3 | 765 |
| EQUAL9 | 4 | 646 |
| EQUAL9 | 5 | 341 |
| EQUAL9 | 6 | 102 |
| EQUAL9 | 7 | 8 |
| EQUAL9 | 8 | 4 |
| EQUAL9 | 9 | 0 |
| H09 | 1 | 246 |
| H09 | 2 | 555 |
| H09 | 3 | 739 |
| H09 | 4 | 645 |
| H09 | 5 | 361 |
| H09 | 6 | 123 |
| H09 | 7 | 25 |
| H09 | 8 | 5 |
| H09 | 9 | 0 |
| H09 | 10 | 0 |
| ALL78 | 1 | 23 |
| ALL78 | 2 | 93 |
| ALL78 | 3 | 277 |
| ALL78 | 4 | 374 |
| ALL78 | 5 | 413 |
| ALL78 | 6 | 508 |
| ALL78 | 7 | 430 |
| ALL78 | 8 | 275 |
| ALL78 | 9 | 167 |
| ALL78 | 10 | 86 |
| ALL78 | 11 | 34 |
| ALL78 | 12 | 14 |
| ALL78 | 13 | 4 |
| ALL78 | 14 | 1 |
| ALL78 | 15 | 0 |
| ALL78 | 16 | 0 |
| ALL78 | 17 | 0 |
| ALL78 | 18 | 0 |
| ALL78 | 19 | 0 |
| ALL78 | 20 | 0 |
| ALL78 | 21 | 0 |
| ALL78 | 22 | 0 |
| ALL78 | 23 | 0 |
| ALL78 | 24 | 0 |
| ALL78 | 25 | 0 |
| ALL78 | 26 | 0 |
| ALL78 | 27 | 0 |
| ALL78 | 28 | 0 |
| ALL78 | 29 | 0 |
| ALL78 | 30 | 0 |
| ALL78 | 31 | 0 |
| ALL78 | 32 | 0 |
| ALL78 | 33 | 0 |
| ALL78 | 34 | 0 |
| ALL78 | 35 | 0 |
| ALL78 | 36 | 0 |
| ALL78 | 37 | 0 |
| ALL78 | 38 | 0 |
| ALL78 | 39 | 0 |
| ALL78 | 40 | 0 |
| ALL78 | 41 | 0 |
| ALL78 | 42 | 0 |
| ALL78 | 43 | 0 |
| ALL78 | 44 | 0 |
| ALL78 | 45 | 0 |
| ALL78 | 46 | 0 |
| ALL78 | 47 | 0 |
| ALL78 | 48 | 0 |
| ALL78 | 49 | 0 |
| ALL78 | 50 | 0 |
| ALL78 | 51 | 0 |
| ALL78 | 52 | 0 |
| ALL78 | 53 | 0 |
| ALL78 | 54 | 0 |
| ALL78 | 55 | 0 |
| ALL78 | 56 | 0 |
| ALL78 | 57 | 0 |
| ALL78 | 58 | 0 |
| ALL78 | 59 | 0 |
| ALL78 | 60 | 0 |
| ALL78 | 61 | 0 |
| ALL78 | 62 | 0 |
| ALL78 | 63 | 0 |
| ALL78 | 64 | 0 |
| ALL78 | 65 | 0 |
| ALL78 | 66 | 0 |
| ALL78 | 67 | 0 |
| ALL78 | 68 | 0 |
| ALL78 | 69 | 0 |
| ALL78 | 70 | 0 |
| ALL78 | 71 | 0 |
| ALL78 | 72 | 0 |
| ALL78 | 73 | 0 |
| ALL78 | 74 | 0 |
| ALL78 | 75 | 0 |
| ALL78 | 76 | 0 |
| ALL78 | 77 | 0 |
| ALL78 | 78 | 0 |

Most frequent directed member top-1 errors (true label → predicted label, counted across checkpoint predictions):

| True label | Predicted label | Checkpoint error count |
|---|---|---:|
| gemini-3.7-flash | gemini-3.8-flash | 4772 |
| glm-5.3 | claude-fable-5 | 2347 |
| gpt-5.6-sol | gpt-5.6-terra | 2324 |
| gemini-3.8-flash | gemini-3.7-flash | 2265 |
| gemini-3.6-flash | gemini-3.7-flash | 2211 |
| gpt-5.5 | qwen3.8-max | 2117 |
| gemini-3.6-flash | gemini-3.8-flash | 1974 |
| glm-5.3-flash | glm-5.3 | 1966 |
| glm-5.3 | glm-5.3-flash | 1911 |
| gpt-5.6-luna | gpt-5.6-terra | 1899 |
| gemini-3.1-pro-preview | gemini-3.6-flash | 1794 |
| gemini-3.7-flash | gemini-3.6-flash | 1779 |
| glm-5.3-flash | claude-fable-5 | 1772 |
| gpt-5.6-terra | qwen3.8-max | 1566 |
| claude-fable-5 | glm-5.3 | 1557 |

See `candidate_coverage.json` for per-label results, correct-label vote-support histograms, minority support, and confusion pairs. See `candidate_coverage.csv` for row counts and rates.

## 2. Text-augmented ORACLE02

A is the saved ORACLE02 all78 CatBoost+EQUAL9 cross-fitted baseline (67.167% BA). B–D use nested fitting: the CatBoost configuration and EQUAL9 blend weight are selected using inner prompt-group OOF predictions within each outer training partition. Text-only logistic models use the same nested splits. Their blend weight is selected against fixed EQUAL9 inner-fold probabilities, then applied to ORACLE02 on the held-out fold; this avoids using globally cross-fitted ORACLE02 training rows that could include the current outer fold, but it is a conservative proxy rather than a fully refit ORACLE02 inner stacker. Stylometry is baseline length/case/punctuation/function words plus L,V,C,P,W,F,R,X,M,U feature groups. Embeddings are frozen EmbeddingGemma2 768-dimensional text features. The base-probability matrix is row-aligned and unchanged across variants.

| Strategy | Accuracy | Balanced accuracy | Log loss | Δ BA vs ORACLE02 | Corrected / introduced vs ORACLE02 |
|---|---:|---:|---:|---:|---:|
| ORACLE02_A | 67.173% | 67.167% | 1.19917 | +0.000 pp | 0 / 0 |
| stack_stylometry_catboost | 64.209% | 64.191% | 1.40962 | -2.976 pp | 125 / 205 |
| stack_stylometry_blend | 66.321% | 66.314% | 1.24028 | -0.853 pp | 32 / 55 |
| stack_embedding_catboost | 64.579% | 64.562% | 1.41475 | -2.605 pp | 133 / 203 |
| stack_embedding_blend | 66.617% | 66.611% | 1.25519 | -0.556 pp | 42 / 57 |
| stack_stylometry_embedding_catboost | 63.690% | 63.672% | 1.41465 | -3.495 pp | 117 / 211 |
| stack_stylometry_embedding_blend | 66.024% | 66.019% | 1.24202 | -1.148 pp | 30 / 61 |
| text_only_stylometry_logistic | 31.604% | 31.598% | 2.33276 | -35.569 pp | 165 / 1125 |
| text_only_stylometry_blend | 67.173% | 67.167% | 1.19917 | +0.000 pp | 0 / 0 |
| text_only_embedding_logistic | 17.155% | 17.152% | 4.24172 | -50.015 pp | 112 / 1462 |
| text_only_embedding_blend | 67.173% | 67.167% | 1.19917 | +0.000 pp | 0 / 0 |
| text_only_stylometry_embedding_logistic | 27.343% | 27.339% | 4.93807 | -39.828 pp | 144 / 1219 |
| text_only_stylometry_embedding_blend | 67.173% | 67.167% | 1.19917 | +0.000 pp | 0 / 0 |

### Error movement against the reference systems

Corrected means the strategy is correct and the named reference is wrong; introduced means the reverse.

| Strategy | EQUAL9 corrected / introduced | H09 corrected / introduced | ORACLE02 corrected / introduced |
|---|---:|---:|---:|
| stack_stylometry_catboost | 213 / 245 | 192 / 227 | 125 / 205 |
| stack_stylometry_blend | 82 / 57 | 82 / 60 | 32 / 55 |
| stack_embedding_catboost | 220 / 242 | 196 / 221 | 133 / 203 |
| stack_embedding_blend | 101 / 68 | 89 / 59 | 42 / 57 |
| stack_stylometry_embedding_catboost | 199 / 245 | 183 / 232 | 117 / 211 |
| stack_stylometry_embedding_blend | 75 / 58 | 78 / 64 | 30 / 61 |
| text_only_stylometry_logistic | 175 / 1087 | 174 / 1089 | 165 / 1125 |
| text_only_stylometry_blend | 108 / 60 | 104 / 59 | 0 / 0 |
| text_only_embedding_logistic | 117 / 1419 | 121 / 1426 | 112 / 1462 |
| text_only_embedding_blend | 108 / 60 | 104 / 59 | 0 / 0 |
| text_only_stylometry_embedding_logistic | 161 / 1188 | 161 / 1191 | 144 / 1219 |
| text_only_stylometry_embedding_blend | 108 / 60 | 104 / 59 | 0 / 0 |
| rerank_top2 | 211 / 154 | 210 / 156 | 123 / 114 |
| rerank_top3 | 266 / 673 | 272 / 682 | 232 / 687 |
| selective_override | 202 / 147 | 199 / 147 | 112 / 105 |

Fold-specific model configurations, blend weights, train/held-out metrics, and specialist thresholds are in `metrics.json`. Per-label P/R/F1 for every strategy is in `per_label_metrics.csv`; ID-aligned probabilities are in `row_predictions.parquet`.

## 3. Pairwise and family specialists

Specialists are L2 logistic classifiers trained only on examples belonging to their confusion group. Candidate reranking is permitted only when at least two group labels occur in the baseline top-k. Selective override thresholds are chosen from inner-fold OOF predictions by overall balanced accuracy and are applied only when the baseline top-1 is in that group.

| Strategy | Accuracy | Balanced accuracy | Log loss | Δ BA vs ORACLE02 | Corrected / introduced vs ORACLE02 |
|---|---:|---:|---:|---:|---:|
| rerank_top2 | 67.506% | 67.500% | 1.19450 | +0.333 pp | 123 / 114 |
| rerank_top3 | 50.315% | 50.320% | 1.25439 | -16.847 pp | 232 / 687 |
| selective_override | 67.432% | 67.427% | 2.53072 | +0.260 pp | 112 / 105 |

| Specialist group |
|---|
| claude-fable-5, glm-5.3 |
| gemini-3.6-flash, gemini-3.7-flash |
| gemini-3.7-flash, gemini-3.8-flash |
| glm-5.3, glm-5.3-flash |
| gpt-5.5, qwen3.8-max |
| gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra |

| Specialist group | Top-2 conflicts: eligible; correct/wrong; corrected/introduced | Top-3 conflicts: eligible; correct/wrong; corrected/introduced | Selective trigger: rows; correct/incorrect; corrected/introduced |
|---|---:|---:|---:|
| gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra | 333; 217/116; 23/25 | 597; 280/317; 43/174 | 340; 225/115; 14/22 |
| gemini-3.7-flash, gemini-3.8-flash | 259; 165/94; 48/29 | 421; 193/228; 70/133 | 242; 152/90; 41/22 |
| glm-5.3, glm-5.3-flash | 212; 103/109; 18/28 | 530; 171/359; 61/179 | 252; 111/141; 16/25 |
| claude-fable-5, glm-5.3 | 144; 68/76; 22/13 | 523; 169/354; 67/177 | 299; 140/159; 30/23 |
| gpt-5.5, qwen3.8-max | 105; 78/27; 6/7 | 191; 112/79; 12/33 | 297; 212/85; 7/8 |
| gemini-3.6-flash, gemini-3.7-flash | 93; 61/32; 6/12 | 499; 219/280; 91/200 | 214; 144/70; 6/13 |

Group rows are aggregated over their five held-out folds. A conflict row contains at least two labels from that specialist group in the ORACLE02 top-k list.
Fold-level trigger counts, conditional performance, selected thresholds, and specialist support are recorded in `metrics.json`.

| Fold | ORACLE02 BA | Top-2 rerank BA | Change | Rows reranked |
|---:|---:|---:|---:|---:|
| 1 | 65.000% | 65.556% | +0.556 pp | 44 |
| 2 | 66.667% | 66.852% | +0.185 pp | 65 |
| 3 | 69.630% | 69.074% | -0.556 pp | 51 |
| 4 | 65.185% | 65.370% | +0.185 pp | 54 |
| 5 | 69.361% | 70.677% | +1.315 pp | 77 |

## Ranked comparison and recommendations

Strategies are ranked by pooled outer-fold balanced accuracy; this ranking is descriptive because all base models and roster selection were exposed to V0.

1. **rerank_top2** — BA 67.500%, accuracy 67.506%, log loss 1.19450.
2. **selective_override** — BA 67.427%, accuracy 67.432%, log loss 2.53072.
3. **ORACLE02_A** — BA 67.167%, accuracy 67.173%, log loss 1.19917.
4. **text_only_embedding_blend** — BA 67.167%, accuracy 67.173%, log loss 1.19917.
5. **text_only_stylometry_blend** — BA 67.167%, accuracy 67.173%, log loss 1.19917.
6. **text_only_stylometry_embedding_blend** — BA 67.167%, accuracy 67.173%, log loss 1.19917.
7. **stack_embedding_blend** — BA 66.611%, accuracy 66.617%, log loss 1.25519.
8. **stack_stylometry_blend** — BA 66.314%, accuracy 66.321%, log loss 1.24028.
9. **stack_stylometry_embedding_blend** — BA 66.019%, accuracy 66.024%, log loss 1.24202.
10. **C3** — BA 65.501%, accuracy 65.506%, log loss 12.43299.
11. **H09** — BA 65.500%, accuracy 65.506%, log loss 1.10515.
12. **EQUAL9** — BA 65.390%, accuracy 65.395%, log loss 1.13208.
13. **stack_embedding_catboost** — BA 64.562%, accuracy 64.579%, log loss 1.41475.
14. **stack_stylometry_catboost** — BA 64.191%, accuracy 64.209%, log loss 1.40962.
15. **stack_stylometry_embedding_catboost** — BA 63.672%, accuracy 63.690%, log loss 1.41465.
16. **ALL78** — BA 61.390%, accuracy 61.393%, log loss 1.24802.
17. **rerank_top3** — BA 50.320%, accuracy 50.315%, log loss 1.25439.
18. **text_only_stylometry_logistic** — BA 31.598%, accuracy 31.604%, log loss 2.33276.
19. **text_only_stylometry_embedding_logistic** — BA 27.339%, accuracy 27.343%, log loss 4.93807.
20. **text_only_embedding_logistic** — BA 17.152%, accuracy 17.155%, log loss 4.24172.

1. **Best measured strategy:** rerank_top2 at 67.500% BA (+0.333 pp versus ORACLE02). Treat it as a development candidate because the base roster has V0 exposure.
2. **Text features:** text_only_stylometry_blend is the strongest text blend (+0.000 pp versus ORACLE02); outer-fold BAs: 65.00%, 66.67%, 69.63%, 65.19%, 69.36%.
3. **Specialists and Kaggle:** rerank_top2 is the strongest specialist integration (+0.333 pp; 123 corrected and 114 introduced versus ORACLE02). Preserve the current 0.581 public-score candidate until a locked submission comparison demonstrates an improvement.

## Explicit answers

1. **Oracle coverage:** EQUAL9 top-1 oracle support is 91.55%; probability top-2/top-3 retain 89.88%/96.16% of it. ALL78 support is 98.67%; top-2/top-3 retain 80.77%/90.39%.
2. **Largest measured confusion opportunities:** the leading member error is `gemini-3.7-flash → gemini-3.8-flash` (4,772 checkpoint predictions). By held-out top-2 specialist net corrections, `gemini-3.7-flash, gemini-3.8-flash` is strongest (48 corrected versus 29 introduced across folds).
3. **Text features:** no text-augmented stack beats ORACLE02. The best CatBoost+text blend is stack_embedding_blend; its balanced accuracy is below the ORACLE02 baseline.
4. **Specialist quality:** top-2 reranking improves pooled OOF balanced accuracy by +0.333 pp; top-3 is harmful because it introduces 687 errors versus 232 corrections.
5. **Best combination:** top-2 specialist reranking (67.500% BA), ahead of ORACLE02 in 4/5 folds.
6. **Remaining gap:** the ALL78 98.67% top-1 oracle coverage is 31.17 pp above the best measured balanced accuracy. Candidate-supported confusion rows remain, but current reranking features resolve them unreliably.
7. **Kaggle readiness:** do not submit this based on the current evidence. The 0.581 public benchmark is not directly comparable with V0, the checkpoints/roster were exposed to V0, and top-2's 0.333 pp gain reverses in one of five folds.

## Three recommendations

1. Keep ORACLE02 as the baseline; stylometry and embeddings add no measured OOF gain.
2. Carry top-2 reranking forward as an experimental candidate only; do not use top-3 reranking.
3. Require an unseen labeled evaluation or controlled Kaggle test before replacing the 0.581 public-score reference.

## Artifacts

- `candidate_coverage.csv/.json`: top-k shortlist coverage, oracle-support counts, per-label distributions and confusions.
- `text_features.npz`, `text_embeddings.npz`: exact ID-aligned text feature arrays.
- `row_predictions.parquet`, `per_label_metrics.csv`, `metrics.json`: OOF predictions and scores.
- `run_manifest.json`: source hashes, methods, roster and fold provenance.
