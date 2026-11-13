# Trust List 2 Per-Label Weighted Voting

Trust snapshot: `trust-list-2-v012` (generated 2026-11-12T15:29:15+00:00).
Validation: full V0, 2,699 rows, hash `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.

## Overall comparison

| Method | Accuracy | Balanced accuracy | Macro F1 | Correct / rows |
|---|---:|---:|---:|---:|
| Weighted vote | 0.6425 | 0.6424 | 0.6310 | 1,734 / 2,699 |
| N0122 UltraBERT standalone | 0.5624 | 0.5624 | 0.5573 | 1,518 / 2,699 |

## Vote roster and normalization

73 checkpoints supplied at least one eligible checkpoint-label vote (1306 eligible label rules). A checkpoint votes once per response only when its top-1 predicted label has an eligible row in the trust list. The raw vote is `precision × prediction_count / (prediction_count + 30)`. Each raw vote is divided by the number of included checkpoints in its model family; this gives each family a mean contribution instead of letting a family gain weight by checkpoint count. Scores are summed by label; the normalized `vote_share::` columns in the prediction artifact are score shares, not calibrated probabilities. Zero-vote rows: 0.

| model_family   |   eligible_checkpoints |
|:---------------|-----------------------:|
| catboost       |                     34 |
| gemma          |                      6 |
| granite        |                      4 |
| llama          |                      2 |
| mistral        |                      2 |
| qwen           |                      6 |
| smollm         |                      2 |
| svc            |                      5 |
| ultrabert      |                     12 |

Per response, eligible checkpoint voter count ranged from 69 to 73 (median 73, mean 72.99).

|   eligible_checkpoint_votes |   validation_rows |   row_percent |
|----------------------------:|------------------:|--------------:|
|                     69.0000 |            3.0000 |        0.0011 |
|                     70.0000 |            1.0000 |        0.0004 |
|                     71.0000 |            2.0000 |        0.0007 |
|                     72.0000 |           18.0000 |        0.0067 |
|                     73.0000 |         2675.0000 |        0.9911 |

## Per-label precision and recall

| label                  |   precision |   recall |   N0122_precision |   N0122_recall |   predicted |   support |
|:-----------------------|------------:|---------:|------------------:|---------------:|------------:|----------:|
| claude-fable-5         |      0.4096 |   0.4564 |            0.3009 |         0.4362 |         166 |       149 |
| claude-opus-4-6        |      0.7074 |   0.8867 |            0.7986 |         0.7400 |         188 |       150 |
| claude-opus-4-8        |      0.7328 |   0.6400 |            0.6761 |         0.6400 |         131 |       150 |
| claude-opus-5          |      0.7558 |   0.8667 |            0.8879 |         0.6867 |         172 |       150 |
| claude-sonnet-5        |      0.6178 |   0.7867 |            0.6581 |         0.6800 |         191 |       150 |
| gemini-3.1-pro-preview |      0.8690 |   0.8400 |            0.9545 |         0.7000 |         145 |       150 |
| gemini-3.6-flash       |      0.6905 |   0.7733 |            0.5979 |         0.7533 |         168 |       150 |
| gemini-3.7-flash       |      0.6023 |   0.3533 |            0.4242 |         0.0933 |          88 |       150 |
| gemini-3.8-flash       |      0.5411 |   0.7467 |            0.4513 |         0.8333 |         207 |       150 |
| glm-5.3                |      0.4632 |   0.2933 |            0.3243 |         0.2400 |          95 |       150 |
| glm-5.3-flash          |      0.5000 |   0.4733 |            0.3200 |         0.2667 |         142 |       150 |
| gpt-5.5                |      0.6357 |   0.5933 |            0.5330 |         0.6467 |         140 |       150 |
| gpt-5.6-luna           |      0.6541 |   0.5800 |            0.6025 |         0.6467 |         133 |       150 |
| gpt-5.6-sol            |      0.6974 |   0.3533 |            0.4747 |         0.5000 |          76 |       150 |
| gpt-5.6-terra          |      0.5789 |   0.6600 |            0.4947 |         0.6267 |         171 |       150 |
| gpt-6-astra            |      0.7473 |   0.9067 |            0.9300 |         0.6200 |         182 |       150 |
| kimi-k3                |      0.7400 |   0.4933 |            0.3563 |         0.3800 |         100 |       150 |
| qwen3.8-max            |      0.6324 |   0.8600 |            0.7037 |         0.6333 |         204 |       150 |

## Confusion matrix

Rows are true labels; columns are predicted labels.

|                        |   claude-fable-5 |   claude-opus-4-6 |   claude-opus-4-8 |   claude-opus-5 |   claude-sonnet-5 |   gemini-3.1-pro-preview |   gemini-3.6-flash |   gemini-3.7-flash |   gemini-3.8-flash |   glm-5.3 |   glm-5.3-flash |   gpt-5.5 |   gpt-5.6-luna |   gpt-5.6-sol |   gpt-5.6-terra |   gpt-6-astra |   kimi-k3 |   qwen3.8-max |
|:-----------------------|-----------------:|------------------:|------------------:|----------------:|------------------:|-------------------------:|-------------------:|-------------------:|-------------------:|----------:|----------------:|----------:|---------------:|--------------:|----------------:|--------------:|----------:|--------------:|
| claude-fable-5         |               68 |                12 |                 9 |              12 |                12 |                        0 |                  0 |                  0 |                  2 |        16 |              16 |         0 |              0 |             0 |               0 |             0 |         1 |             1 |
| claude-opus-4-6        |                3 |               133 |                 3 |               0 |                 7 |                        0 |                  0 |                  0 |                  3 |         1 |               0 |         0 |              0 |             0 |               0 |             0 |         0 |             0 |
| claude-opus-4-8        |               14 |                11 |                96 |               3 |                19 |                        0 |                  0 |                  0 |                  0 |         2 |               5 |         0 |              0 |             0 |               0 |             0 |         0 |             0 |
| claude-opus-5          |                4 |                 0 |                 3 |             130 |                 4 |                        0 |                  0 |                  0 |                  0 |         1 |               6 |         0 |              0 |             0 |               0 |             1 |         0 |             1 |
| claude-sonnet-5        |                5 |                 8 |                 7 |               1 |               118 |                        0 |                  0 |                  0 |                  1 |         4 |               1 |         1 |              0 |             0 |               0 |             0 |         2 |             2 |
| gemini-3.1-pro-preview |                0 |                 1 |                 0 |               0 |                 0 |                      126 |                 15 |                  5 |                  3 |         0 |               0 |         0 |              0 |             0 |               0 |             0 |         0 |             0 |
| gemini-3.6-flash       |                0 |                 0 |                 0 |               0 |                 1 |                       10 |                116 |                 10 |                 13 |         0 |               0 |         0 |              0 |             0 |               0 |             0 |         0 |             0 |
| gemini-3.7-flash       |                0 |                 1 |                 0 |               0 |                 0 |                        7 |                 22 |                 53 |                 67 |         0 |               0 |         0 |              0 |             0 |               0 |             0 |         0 |             0 |
| gemini-3.8-flash       |                0 |                 0 |                 0 |               0 |                 1 |                        2 |                 14 |                 20 |                112 |         0 |               0 |         0 |              0 |             0 |               0 |             0 |         0 |             1 |
| glm-5.3                |               38 |                10 |                 3 |              11 |                 9 |                        0 |                  0 |                  0 |                  1 |        44 |              32 |         0 |              0 |             0 |               0 |             0 |         2 |             0 |
| glm-5.3-flash          |               23 |                 7 |                 4 |               6 |                12 |                        0 |                  0 |                  0 |                  0 |        21 |              71 |         1 |              0 |             0 |               0 |             2 |         1 |             2 |
| gpt-5.5                |                1 |                 0 |                 1 |               2 |                 1 |                        0 |                  0 |                  0 |                  0 |         1 |               1 |        89 |              2 |             5 |              14 |             0 |         9 |            24 |
| gpt-5.6-luna           |                1 |                 0 |                 1 |               1 |                 0 |                        0 |                  0 |                  0 |                  0 |         1 |               0 |         5 |             87 |             6 |              18 |            20 |         1 |             9 |
| gpt-5.6-sol            |                0 |                 0 |                 0 |               0 |                 0 |                        0 |                  0 |                  0 |                  2 |         0 |               0 |         7 |             25 |            53 |              33 |            13 |         4 |            13 |
| gpt-5.6-terra          |                0 |                 0 |                 0 |               0 |                 0 |                        0 |                  0 |                  0 |                  1 |         0 |               0 |        13 |             11 |             9 |              99 |             6 |         4 |             7 |
| gpt-6-astra            |                0 |                 1 |                 0 |               0 |                 1 |                        0 |                  0 |                  0 |                  0 |         0 |               0 |         1 |              5 |             2 |               2 |           136 |         0 |             2 |
| kimi-k3                |                8 |                 4 |                 3 |               6 |                 5 |                        0 |                  0 |                  0 |                  0 |         4 |              10 |        18 |              0 |             0 |               3 |             2 |        74 |            13 |
| qwen3.8-max            |                1 |                 0 |                 1 |               0 |                 1 |                        0 |                  1 |                  0 |                  2 |         0 |               0 |         5 |              3 |             1 |               2 |             2 |         2 |           129 |

## Artifacts

- Weighted-vote full-V0 predictions and per-label scores: `artifacts/analysis/trust_list_2_weighted_vote/predictions.parquet`
- N0122 baseline predictions: `artifacts/analysis/trust_list_2_weighted_vote/n0122_predictions.parquet`
- Per-checkpoint, per-response contribution audit: `artifacts/analysis/trust_list_2_weighted_vote/checkpoint_contributions.parquet`
- Per-response voter-count distribution: `artifacts/analysis/trust_list_2_weighted_vote/vote_count_distribution.csv`
- Machine-readable metrics: `artifacts/analysis/trust_list_2_weighted_vote/metrics.json`
- Confusion CSV: `artifacts/analysis/trust_list_2_weighted_vote/confusion_matrix.csv`

## Limitation

Trust List 2 precision values and the checkpoints were evaluated/selected using this same full V0 set; these are in-sample development diagnostics, not independent generalization estimates.
