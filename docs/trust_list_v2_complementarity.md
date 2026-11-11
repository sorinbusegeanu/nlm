# Trust-list v2 checkpoint complementarity

Snapshot: **trust-list-2-v001**, generated `2026-11-11T08:10:48+00:00`. Validation: full V0, 2,699 rows; SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.

## Fixed trust-list order

Rules are replayed in the saved SeqID order. Each eligible checkpoint-label rule claims only still-unclaimed rows whose checkpoint top-1 prediction equals the rule label. Unsupported/ineligible rules are skipped. This is a same-V0 development diagnostic: V0 supplied the ranking metrics and was used for checkpoint selection.

| Measure | Result |
|---|---:|
| Checkpoints / rules | 65 / 1,170 |
| Eligible rules | 1,164 |
| Rules claiming at least one row | 56 |
| Eligible rules claiming zero rows | 1,108 |
| Rows claimed / left unclaimed | 2,699 / 0 |
| Correct / incorrect assignments | 2,392 / 307 |
| Accuracy | 88.63% |
| Last SeqID assigning any row | 87 |

| SeqID cutoff | Claimed | Correct | Incorrect | Remaining | Accuracy on claimed |
|---:|---:|---:|---:|---:|---:|
| 10 | 738 | 698 | 40 | 1,961 | 94.58% |
| 20 | 1,313 | 1,210 | 103 | 1,386 | 92.16% |
| 50 | 2,664 | 2,375 | 289 | 35 | 89.15% |
| 87 | 2,699 | 2,392 | 307 | 0 | 88.63% |

### Global versus incremental precision examples

| SeqID | Checkpoint | Label | Global precision / support | Newly claimed | Incremental TP / FP | Incremental precision |
|---:|---|---|---:|---:|---:|---:|
| 1 | N0086 | gpt-6-astra | 98.6% / 145 | 145 | 143 / 2 | 98.6% |
| 3 | N0072 | gpt-6-astra | 98.6% / 138 | 5 | 3 / 2 | 60.0% |
| 4 | N0097 | gemini-3.1-pro-preview | 97.1% / 104 | 6 | 3 / 3 | 50.0% |
| 11 | N0096 | gemini-3.1-pro-preview | 94.5% / 109 | 3 | 0 / 3 | 0.0% |
| 15 | N0104 | gemini-3.1-pro-preview | 93.1% / 116 | 2 | 0 / 2 | 0.0% |
| 19 | N0081 | gemini-3.1-pro-preview | 92.4% / 157 | 9 | 0 / 9 | 0.0% |

### Candidate checkpoints versus fixed-cascade errors

This substitution diagnostic asks how many current cascade mistakes each checkpoint gets right, and how many current cascade-correct rows it would lose if used alone.

| Checkpoint | Standalone accuracy | Current cascade errors it predicts correctly | Current correct rows it would lose | Net change if substituted |
|---|---:|---:|---:|---:|
| N0104 | 48.61% | 96 | 1176 | -1080 |
| N0103 | 49.39% | 76 | 1135 | -1059 |

## Greedy rule-selection diagnostics

Both methods use the full V0 labels to choose rules and evaluate them, require at least 10 newly claimed rows per selected rule, and use `N0081` as the fallback for rows left unclaimed. Residual-precision selection stops when no rule meets residual support; net-gain selection also requires positive TP − FP. Scores are exploratory and in-sample.

| Greedy objective | Rules | Rule-claimed | Fallback rows correct | Correct total | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Residual precision | 37 | 2,670 | 12 | 2,460/2,699 | 91.14% | 91.14% |
| Residual TP − FP | 21 | 2,600 | 21 | 2,434/2,699 | 90.18% | 90.18% |

Best standalone checkpoint in this snapshot: `N0081`, accuracy 88.59% (2391/2699). The fixed trust-list replay is 1 example above it on these same rows; this is not an independent estimate. The greedy results also compare with this checkpoint only because `N0081` was selected as their fallback.

### New V2 checkpoints

N0103 and N0104 are included using saved full-V0 prediction Parquets. Under the fixed order, N0104's highest-ranked Gemini 3.1 rule claims only two residual rows and both are wrong; N0103's first rule is after full coverage. In greedy diagnostics, N0104 contributes one 11-row rule per objective (7/11 Terra under precision selection; 6/11 Sol under net-gain selection); N0103 is not selected. These are ordering-specific observations, not an exhaustive estimate of their routing potential.

## Interpretation and limits

- The global precision is measured over all predictions, but the cascade sees only the subset left after earlier rules. These can have sharply different precision.
- In the fixed order, 1,108 eligible rules add no rows. Most cascade coverage and correctness comes from a small number of early checkpoints; later checkpoints mostly overlap already-claimed predictions.
- Greedy results are selected and scored on the same 2,699 V0 rows. The apparent gains over the fixed order are subject to substantial selection optimism and must not be treated as expected Kaggle accuracy.
- No checkpoint was loaded, no inference was run, and the active trust list was not modified.

Rule-by-rule audit and greedy selection traces are saved under `artifacts/analysis/trust_list_2_v0/`.
