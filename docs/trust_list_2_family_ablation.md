# Trust List 2 Leave-One-Family-Out Analysis

Trust snapshot `trust-list-2-v012`: 73 active checkpoints, 1306 eligible checkpoint-label rules, and 2,699 full-V0 rows (SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`).
The baseline uses all nine families. Each ablation removes the named family, then recomputes the divisor from the eligible checkpoints remaining in each family. Per-vote weighting stays `precision × prediction_count / (prediction_count + 30)`, and ties follow canonical label order.

## Baseline and family ablations

Difference columns are ablation minus baseline in percentage points. `Measured contribution` is baseline balanced accuracy minus ablated balanced accuracy; positive values mean the full ensemble benefits from that family.

| Rank | Removed family | Removed checkpoints | Accuracy | Balanced accuracy | Δ accuracy (pp) | Δ balanced accuracy (pp) | Errors corrected | Correct lost | Net correct Δ | Measured contribution (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **All families baseline** | 0 | 0.6425 | 0.6424 | 0.000 | 0.000 | — | — | 0 | — |
| 1 | smollm | 2 | 0.6269 | 0.6268 | -1.556 | -1.556 | 49 | 91 | -42 | +1.556 |
| 2 | qwen | 6 | 0.6321 | 0.6320 | -1.037 | -1.037 | 20 | 48 | -28 | +1.037 |
| 3 | ultrabert | 12 | 0.6428 | 0.6428 | +0.037 | +0.038 | 35 | 34 | +1 | -0.038 |
| 4 | gemma | 6 | 0.6439 | 0.6439 | +0.148 | +0.148 | 33 | 29 | +4 | -0.148 |
| 5 | catboost | 34 | 0.6443 | 0.6442 | +0.185 | +0.185 | 33 | 28 | +5 | -0.185 |
| 6 | svc | 5 | 0.6443 | 0.6442 | +0.185 | +0.185 | 46 | 41 | +5 | -0.185 |
| 7 | mistral | 2 | 0.6480 | 0.6479 | +0.556 | +0.555 | 71 | 56 | +15 | -0.555 |
| 8 | granite | 4 | 0.6506 | 0.6505 | +0.815 | +0.816 | 54 | 32 | +22 | -0.816 |
| 9 | llama | 2 | 0.6606 | 0.6606 | +1.815 | +1.817 | 95 | 46 | +49 | -1.817 |

## Per-label changes in correct predictions

Each value is the number correct after removing a family minus the baseline count.

| label                  |   catboost |   gemma |   granite |   llama |   mistral |   qwen |   smollm |   svc |   ultrabert |
|:-----------------------|-----------:|--------:|----------:|--------:|----------:|-------:|---------:|------:|------------:|
| claude-fable-5         |          0 |       1 |         3 |      10 |        -1 |      2 |       -1 |     0 |           3 |
| claude-opus-4-6        |          0 |       2 |         2 |       1 |         2 |      1 |       -1 |    -3 |           0 |
| claude-opus-4-8        |          2 |      -1 |         1 |       5 |         6 |     -2 |        0 |     2 |          -1 |
| claude-opus-5          |          0 |       0 |         3 |       3 |        -1 |      1 |       -3 |     0 |           0 |
| claude-sonnet-5        |          0 |      -2 |        -4 |      -3 |        -4 |     -2 |       -4 |     0 |          -1 |
| gemini-3.1-pro-preview |         -1 |      -2 |         1 |       1 |         3 |      0 |       -3 |    -2 |           1 |
| gemini-3.6-flash       |          2 |       4 |         2 |       1 |        -1 |     -2 |       -5 |    -3 |          -1 |
| gemini-3.7-flash       |          2 |      -1 |         2 |       4 |         2 |      1 |       -3 |     1 |           2 |
| gemini-3.8-flash       |         -3 |      -3 |        -1 |      -2 |         2 |     -1 |       -2 |    -3 |          -3 |
| glm-5.3                |          1 |      -2 |         3 |      -5 |        -7 |     -4 |       10 |     0 |          -1 |
| glm-5.3-flash          |         -1 |       3 |         7 |       8 |        11 |     -4 |       -9 |     6 |           9 |
| gpt-5.5                |         -1 |       1 |        -2 |       5 |         3 |     -7 |       -9 |     0 |          -3 |
| gpt-5.6-luna           |          1 |       3 |         1 |       2 |         2 |     -5 |        0 |     2 |           0 |
| gpt-5.6-sol            |          3 |       0 |         1 |       5 |        -1 |     -2 |        1 |     5 |          -1 |
| gpt-5.6-terra          |          0 |       1 |         4 |       8 |        -2 |     -1 |       -3 |     1 |           1 |
| gpt-6-astra            |          0 |       0 |        -2 |       1 |         0 |     -2 |       -2 |    -2 |          -1 |
| kimi-k3                |          1 |       0 |         2 |       2 |         0 |      0 |       -6 |     1 |          -2 |
| qwen3.8-max            |         -1 |       0 |        -1 |       3 |         1 |     -1 |       -2 |     0 |          -1 |

## Artifacts

- Summary and machine-readable metrics: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_family_ablation/summary.json`
- Family metrics and contribution ranking: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_family_ablation/family_ablation_metrics.csv`
- Per-label changes: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_family_ablation/per_label_correct_changes.csv` and `/home/zodrak/nlm/artifacts/analysis/trust_list_2_family_ablation/per_label_correct_changes_wide.csv`
- Baseline and nine ablation predictions: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_family_ablation/predictions`
- Confusion matrices: `/home/zodrak/nlm/artifacts/analysis/trust_list_2_family_ablation/confusion_matrices`

## Scope

This is a diagnostic on the same V0 set used to select Trust List 2 checkpoint rules, so it is not an independent estimate of generalization. The analysis reads saved full-V0 predictions only; it does not load models, train, or run inference.
