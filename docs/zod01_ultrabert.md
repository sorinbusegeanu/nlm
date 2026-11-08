# zod01 — Sequential Precision-Ranked UltraBERT Ensemble

## Method and validation

Five-fold GroupKFold by prompt ID evaluates 540 frozen V0 examples (30 prompt groups, 18 labels). Each fold learns its rule statistics and ordering from the other four folds only. Checkpoints were previously selected using V0, so these are development diagnostics, not an independent generalization estimate.

All 72 checkpoint-label rules are ranked by Beta-smoothed precision `(TP+1)/(prediction_count+1+1)`; tie-breaks are raw precision, prediction support, checkpoint order `A, B, C, D`, and canonical label order. A rule is applied only when its fitting-fold prediction count is at least 10; unsupported rules remain in the audit sequence but are skipped. Remaining held-out rows receive checkpoint D's top-1 prediction.

Split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`; validation ID SHA-256: `d3f39a2003dfdd9e99e5d662465066bd60368f8e43864b6ed10a733edbe74f9d`.

## Overall result

| Rows | Correct | Accuracy | Balanced accuracy | Macro F1 | Labels recall ≥80% | Labels recall ≥90% |
|---:|---:|---:|---:|---:|---:|---:|
| 540 | 290 | 53.7% | 53.7% | 51.2% | 4 | 2 |

## Per-label precision and recall

| Label | TP | FP | FN | Predicted | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| claude-fable-5 | 12 | 18 | 18 | 30 | 40.0% | 40.0% |
| claude-opus-4-6 | 28 | 10 | 2 | 38 | 73.7% | 93.3% **≥90%** |
| claude-opus-4-8 | 16 | 9 | 14 | 25 | 64.0% | 53.3% |
| claude-opus-5 | 21 | 15 | 9 | 36 | 58.3% | 70.0% |
| claude-sonnet-5 | 25 | 17 | 5 | 42 | 59.5% | 83.3% **≥80%** |
| gemini-3.1-pro-preview | 27 | 8 | 3 | 35 | 77.1% | 90.0% **≥90%** |
| gemini-3.6-flash | 20 | 19 | 10 | 39 | 51.3% | 66.7% |
| gemini-3.7-flash | 5 | 5 | 25 | 10 | 50.0% | 16.7% |
| gemini-3.8-flash | 18 | 18 | 12 | 36 | 50.0% | 60.0% |
| glm-5.3 | 2 | 5 | 28 | 7 | 28.6% | 6.7% |
| glm-5.3-flash | 14 | 20 | 16 | 34 | 41.2% | 46.7% |
| gpt-5.5 | 13 | 17 | 17 | 30 | 43.3% | 43.3% |
| gpt-5.6-luna | 6 | 13 | 24 | 19 | 31.6% | 20.0% |
| gpt-5.6-sol | 12 | 17 | 18 | 29 | 41.4% | 40.0% |
| gpt-5.6-terra | 11 | 17 | 19 | 28 | 39.3% | 36.7% |
| gpt-6-astra | 25 | 6 | 5 | 31 | 80.6% | 83.3% **≥80%** |
| kimi-k3 | 13 | 13 | 17 | 26 | 50.0% | 43.3% |
| qwen3.8-max | 22 | 23 | 8 | 45 | 48.9% | 73.3% |

## Sequential rule trace

Counts are from each rule applied to that fold's held-out rows; 'remaining' is after the rule. Support and precision used for ordering come only from the fitting portion. Rule ranks can differ by fold, so sequences are shown separately.

### Fold 1

| Rank | Rule | Fit support | Smoothed precision | Assigned | Correct | Incorrect | Remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C → gemini-3.1-pro-preview | 20 | 90.9% | 5 | 5 | 0 | 103 |
| 2 | B → gpt-6-astra | 17 | 89.5% | 5 | 5 | 0 | 98 |
| 3 | A → gemini-3.1-pro-preview | 20 | 86.4% | 0 | 0 | 0 | 98 |
| 4 | A → gpt-6-astra | 19 | 85.7% | 1 | 1 | 0 | 97 |
| 5 | C → gpt-6-astra | 19 | 85.7% | 0 | 0 | 0 | 97 |
| 6 | D → gpt-6-astra | 18 | 85.0% | 1 | 0 | 1 | 96 |
| 7 | B → gemini-3.1-pro-preview | 21 | 82.6% | 1 | 1 | 0 | 95 |
| 8 | B → claude-opus-4-6 | 14 | 81.2% | 6 | 5 | 1 | 89 |
| 9 | D → claude-opus-5 | 19 | 81.0% | 7 | 4 | 3 | 82 |
| 10 | A → claude-opus-5 | 20 | 77.3% | 0 | 0 | 0 | 82 |
| 11 | C → claude-opus-4-6 | 20 | 77.3% | 2 | 1 | 1 | 80 |
| 12 | D → claude-opus-4-6 | 27 | 75.9% | 3 | 0 | 3 | 77 |
| 13 | D → claude-opus-4-8 | 14 | 75.0% | 5 | 5 | 0 | 72 |
| 14 | A → claude-opus-4-6 | 22 | 75.0% | 0 | 0 | 0 | 72 |
| 15 | D → gemini-3.1-pro-preview | 28 | 73.3% | 1 | 0 | 1 | 71 |
| 16 | C → claude-opus-5 | 23 | 72.0% | 0 | 0 | 0 | 71 |
| 17 | C → qwen3.8-max | 24 | 69.2% | 5 | 4 | 1 | 66 |
| 18 | B → claude-opus-5 | 23 | 68.0% | 1 | 1 | 0 | 65 |
| 19 | C → gemini-3.8-flash | 13 | 66.7% | 2 | 2 | 0 | 63 |
| 20 | B → claude-sonnet-5 | 16 | 66.7% | 5 | 5 | 0 | 58 |
| 21 | B → qwen3.8-max | 24 | 65.4% | 3 | 0 | 3 | 55 |
| 22 | D → qwen3.8-max | 26 | 64.3% | 2 | 0 | 2 | 53 |
| 23 | A → qwen3.8-max | 25 | 63.0% | 0 | 0 | 0 | 53 |
| 24 | A → gemini-3.8-flash | 23 | 60.0% | 4 | 1 | 3 | 49 |
| 25 | B → gemini-3.8-flash | 27 | 58.6% | 2 | 0 | 2 | 47 |
| 26 | B → kimi-k3 | 22 | 58.3% | 7 | 3 | 4 | 40 |
| 27 | D → kimi-k3 | 12 | 57.1% | 1 | 0 | 1 | 39 |
| 28 | D → claude-sonnet-5 | 37 | 56.4% | 0 | 0 | 0 | 39 |
| 29 | B → gemini-3.7-flash | 16 | 55.6% | 4 | 0 | 4 | 35 |
| 30 | D → gpt-5.6-sol | 16 | 55.6% | 1 | 1 | 0 | 34 |
| 31 | B → claude-opus-4-8 | 29 | 54.8% | 1 | 0 | 1 | 33 |
| 32 | C → gemini-3.6-flash | 17 | 52.6% | 4 | 3 | 1 | 29 |
| 33 | D → gemini-3.8-flash | 19 | 52.4% | 0 | 0 | 0 | 29 |
| 34 | A → claude-opus-4-8 | 21 | 52.2% | 0 | 0 | 0 | 29 |
| 35 | C → claude-sonnet-5 | 21 | 52.2% | 0 | 0 | 0 | 29 |
| 36 | B → gemini-3.6-flash | 33 | 51.4% | 0 | 0 | 0 | 29 |
| 37 | D → gemini-3.6-flash | 35 | 51.4% | 0 | 0 | 0 | 29 |
| 38 | A → claude-sonnet-5 | 26 | 50.0% | 0 | 0 | 0 | 29 |
| 39 | C → gpt-5.5 | 22 | 50.0% | 4 | 2 | 2 | 25 |
| 40 | B → gpt-5.5 | 14 | 50.0% | 0 | 0 | 0 | 25 |
| 41 | C → claude-opus-4-8 | 29 | 48.4% | 0 | 0 | 0 | 25 |
| 42 | A → gpt-5.6-sol | 27 | 48.3% | 4 | 1 | 3 | 21 |
| 43 | A → claude-fable-5 | 15 | 47.1% | 1 | 0 | 1 | 20 |
| 44 | D → gemini-3.7-flash | 15 | 47.1% | 0 | 0 | 0 | 20 |
| 45 | D → gpt-5.6-terra | 26 | 46.4% | 2 | 1 | 1 | 18 |
| 46 | A → gemini-3.6-flash | 22 | 45.8% | 0 | 0 | 0 | 18 |
| 47 | A → gpt-5.6-terra | 29 | 45.2% | 2 | 1 | 1 | 16 |
| 48 | A → gpt-5.5 | 18 | 45.0% | 0 | 0 | 0 | 16 |
| 49 | A → gpt-5.6-luna | 23 | 44.0% | 4 | 1 | 3 | 12 |
| 50 | C → gpt-5.6-luna | 23 | 44.0% | 0 | 0 | 0 | 12 |
| 51 | B → gpt-5.6-sol | 33 | 42.9% | 0 | 0 | 0 | 12 |
| 52 | D → claude-fable-5 | 33 | 42.9% | 3 | 2 | 1 | 9 |
| 53 | B → gpt-5.6-terra | 38 | 42.5% | 0 | 0 | 0 | 9 |
| 54 | D → gpt-5.5 | 43 | 42.2% | 0 | 0 | 0 | 9 |
| 55 | D → gpt-5.6-luna | 22 | 41.7% | 1 | 0 | 1 | 8 |
| 56 | B → gpt-5.6-luna | 10 | 41.7% | 0 | 0 | 0 | 8 |
| 57 | C → kimi-k3 | 20 | 40.9% | 1 | 1 | 0 | 7 |
| 58 | C → gpt-5.6-sol | 25 | 40.7% | 0 | 0 | 0 | 7 |
| 59 | C → glm-5.3-flash | 30 | 40.6% | 4 | 2 | 2 | 3 |
| 60 | C → gpt-5.6-terra | 33 | 40.0% | 0 | 0 | 0 | 3 |
| 61 | C → gemini-3.7-flash | 46 | 39.6% | 0 | 0 | 0 | 3 |
| 62 | A → gemini-3.7-flash | 32 | 38.2% | 0 | 0 | 0 | 3 |
| 63 | D → glm-5.3-flash | 23 | 36.0% | 1 | 1 | 0 | 2 |
| 64 | A → kimi-k3 | 30 | 34.4% | 0 | 0 | 0 | 2 |
| 65 | B → glm-5.3-flash | 42 | 34.1% | 0 | 0 | 0 | 2 |
| 66 | D → glm-5.3 | 19 | 33.3% | 2 | 1 | 1 | 0 |
| 67 | A → glm-5.3 | 28 | 30.0% | 0 | 0 | 0 | 0 |
| 68 | C → claude-fable-5 | 22 | 29.2% | 0 | 0 | 0 | 0 |
| 69 | A → glm-5.3-flash | 32 | 26.5% | 0 | 0 | 0 | 0 |
| 70 | C → glm-5.3 | 25 | 25.9% | 0 | 0 | 0 | 0 |
| 71 | B → glm-5.3 | 18 | 25.0% | 0 | 0 | 0 | 0 |
| 72 | B → claude-fable-5 | 35 | 24.3% | 0 | 0 | 0 | 0 |
| fallback | D fallback | — | — | 0 | 0 | 0 | 0 |

### Fold 2

| Rank | Rule | Fit support | Smoothed precision | Assigned | Correct | Incorrect | Remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B → gpt-6-astra | 20 | 90.9% | 2 | 2 | 0 | 106 |
| 2 | C → gemini-3.1-pro-preview | 20 | 90.9% | 5 | 5 | 0 | 101 |
| 3 | B → gemini-3.1-pro-preview | 18 | 90.0% | 2 | 0 | 2 | 99 |
| 4 | A → gpt-6-astra | 23 | 88.0% | 0 | 0 | 0 | 99 |
| 5 | C → gpt-6-astra | 23 | 88.0% | 0 | 0 | 0 | 99 |
| 6 | A → gemini-3.1-pro-preview | 20 | 86.4% | 0 | 0 | 0 | 99 |
| 7 | D → claude-opus-4-8 | 15 | 82.4% | 4 | 3 | 1 | 95 |
| 8 | D → gpt-6-astra | 20 | 81.8% | 0 | 0 | 0 | 95 |
| 9 | B → claude-opus-4-6 | 17 | 78.9% | 3 | 3 | 0 | 92 |
| 10 | D → gemini-3.1-pro-preview | 27 | 75.9% | 2 | 1 | 1 | 90 |
| 11 | B → claude-sonnet-5 | 18 | 75.0% | 3 | 2 | 1 | 87 |
| 12 | D → claude-opus-5 | 21 | 73.9% | 5 | 4 | 1 | 82 |
| 13 | C → claude-opus-4-6 | 23 | 72.0% | 2 | 2 | 0 | 80 |
| 14 | C → qwen3.8-max | 23 | 72.0% | 6 | 4 | 2 | 74 |
| 15 | C → gemini-3.8-flash | 15 | 70.6% | 0 | 0 | 0 | 74 |
| 16 | A → claude-opus-5 | 21 | 69.6% | 0 | 0 | 0 | 74 |
| 17 | B → qwen3.8-max | 21 | 69.6% | 4 | 1 | 3 | 70 |
| 18 | A → claude-opus-4-6 | 24 | 69.2% | 0 | 0 | 0 | 70 |
| 19 | C → claude-opus-5 | 24 | 69.2% | 0 | 0 | 0 | 70 |
| 20 | A → qwen3.8-max | 20 | 68.2% | 1 | 0 | 1 | 69 |
| 21 | D → kimi-k3 | 10 | 66.7% | 2 | 1 | 1 | 67 |
| 22 | D → claude-opus-4-6 | 30 | 65.6% | 1 | 1 | 0 | 66 |
| 23 | B → claude-opus-5 | 24 | 65.4% | 0 | 0 | 0 | 66 |
| 24 | D → qwen3.8-max | 24 | 65.4% | 1 | 0 | 1 | 65 |
| 25 | C → claude-sonnet-5 | 23 | 64.0% | 0 | 0 | 0 | 65 |
| 26 | A → claude-opus-4-8 | 20 | 59.1% | 2 | 0 | 2 | 63 |
| 27 | D → claude-sonnet-5 | 39 | 58.5% | 1 | 0 | 1 | 62 |
| 28 | A → claude-sonnet-5 | 29 | 58.1% | 0 | 0 | 0 | 62 |
| 29 | B → claude-opus-4-8 | 31 | 57.6% | 0 | 0 | 0 | 62 |
| 30 | D → gemini-3.8-flash | 19 | 57.1% | 4 | 2 | 2 | 58 |
| 31 | B → gemini-3.8-flash | 30 | 56.2% | 0 | 0 | 0 | 58 |
| 32 | C → gpt-5.5 | 23 | 56.0% | 4 | 1 | 3 | 54 |
| 33 | D → gpt-5.6-sol | 16 | 55.6% | 4 | 2 | 2 | 50 |
| 34 | A → gemini-3.8-flash | 25 | 55.6% | 1 | 1 | 0 | 49 |
| 35 | C → gemini-3.6-flash | 22 | 54.2% | 1 | 1 | 0 | 48 |
| 36 | C → claude-opus-4-8 | 26 | 53.6% | 0 | 0 | 0 | 48 |
| 37 | B → gpt-5.5 | 13 | 53.3% | 2 | 1 | 1 | 46 |
| 38 | B → gemini-3.6-flash | 32 | 52.9% | 6 | 3 | 3 | 40 |
| 39 | D → gemini-3.6-flash | 34 | 52.8% | 2 | 1 | 1 | 38 |
| 40 | B → kimi-k3 | 23 | 52.0% | 3 | 0 | 3 | 35 |
| 41 | A → gemini-3.6-flash | 24 | 50.0% | 0 | 0 | 0 | 35 |
| 42 | A → gpt-5.5 | 22 | 50.0% | 0 | 0 | 0 | 35 |
| 43 | C → gpt-5.6-luna | 20 | 45.5% | 8 | 2 | 6 | 27 |
| 44 | B → gemini-3.7-flash | 16 | 44.4% | 2 | 2 | 0 | 25 |
| 45 | C → gpt-5.6-terra | 32 | 44.1% | 4 | 1 | 3 | 21 |
| 46 | C → kimi-k3 | 23 | 44.0% | 0 | 0 | 0 | 21 |
| 47 | C → gemini-3.7-flash | 39 | 43.9% | 0 | 0 | 0 | 21 |
| 48 | C → glm-5.3-flash | 28 | 43.3% | 6 | 2 | 4 | 15 |
| 49 | D → gpt-5.6-terra | 28 | 43.3% | 0 | 0 | 0 | 15 |
| 50 | B → gpt-5.6-sol | 26 | 42.9% | 1 | 0 | 1 | 14 |
| 51 | B → gpt-5.6-luna | 12 | 42.9% | 0 | 0 | 0 | 14 |
| 52 | D → gpt-5.5 | 45 | 42.6% | 0 | 0 | 0 | 14 |
| 53 | D → claude-fable-5 | 29 | 41.9% | 7 | 3 | 4 | 7 |
| 54 | B → gpt-5.6-terra | 41 | 41.9% | 0 | 0 | 0 | 7 |
| 55 | A → gpt-5.6-sol | 27 | 41.4% | 0 | 0 | 0 | 7 |
| 56 | A → gpt-5.6-terra | 32 | 41.2% | 0 | 0 | 0 | 7 |
| 57 | D → gemini-3.7-flash | 15 | 41.2% | 0 | 0 | 0 | 7 |
| 58 | A → gemini-3.7-flash | 27 | 37.9% | 0 | 0 | 0 | 7 |
| 59 | A → claude-fable-5 | 14 | 37.5% | 1 | 1 | 0 | 6 |
| 60 | C → gpt-5.6-sol | 25 | 37.0% | 0 | 0 | 0 | 6 |
| 61 | A → gpt-5.6-luna | 17 | 36.8% | 0 | 0 | 0 | 6 |
| 62 | D → glm-5.3-flash | 17 | 36.8% | 3 | 2 | 1 | 3 |
| 63 | D → gpt-5.6-luna | 23 | 36.0% | 0 | 0 | 0 | 3 |
| 64 | B → glm-5.3-flash | 39 | 34.1% | 2 | 0 | 2 | 1 |
| 65 | A → kimi-k3 | 34 | 33.3% | 0 | 0 | 0 | 1 |
| 66 | D → glm-5.3 | 20 | 31.8% | 1 | 0 | 1 | 0 |
| 67 | B → glm-5.3 | 25 | 29.6% | 0 | 0 | 0 | 0 |
| 68 | A → glm-5.3-flash | 27 | 27.6% | 0 | 0 | 0 | 0 |
| 69 | C → glm-5.3 | 28 | 26.7% | 0 | 0 | 0 | 0 |
| 70 | A → glm-5.3 | 26 | 25.0% | 0 | 0 | 0 | 0 |
| 71 | B → claude-fable-5 | 26 | 25.0% | 0 | 0 | 0 | 0 |
| 72 | C → claude-fable-5 | 15 | 23.5% | 0 | 0 | 0 | 0 |
| fallback | D fallback | — | — | 0 | 0 | 0 | 0 |

### Fold 3

| Rank | Rule | Fit support | Smoothed precision | Assigned | Correct | Incorrect | Remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C → gemini-3.1-pro-preview | 19 | 95.2% | 6 | 5 | 1 | 102 |
| 2 | A → gemini-3.1-pro-preview | 18 | 90.0% | 0 | 0 | 0 | 102 |
| 3 | B → gpt-6-astra | 17 | 89.5% | 5 | 5 | 0 | 97 |
| 4 | D → gpt-6-astra | 15 | 88.2% | 2 | 0 | 2 | 95 |
| 5 | A → gpt-6-astra | 19 | 85.7% | 1 | 1 | 0 | 94 |
| 6 | C → gpt-6-astra | 19 | 85.7% | 0 | 0 | 0 | 94 |
| 7 | B → gemini-3.1-pro-preview | 21 | 82.6% | 0 | 0 | 0 | 94 |
| 8 | D → claude-opus-4-8 | 14 | 81.2% | 5 | 4 | 1 | 89 |
| 9 | B → claude-opus-4-6 | 18 | 80.0% | 2 | 2 | 0 | 87 |
| 10 | C → qwen3.8-max | 22 | 79.2% | 7 | 3 | 4 | 80 |
| 11 | D → gemini-3.1-pro-preview | 28 | 76.7% | 1 | 0 | 1 | 79 |
| 12 | C → claude-opus-4-6 | 25 | 74.1% | 1 | 1 | 0 | 78 |
| 13 | B → claude-sonnet-5 | 17 | 73.7% | 4 | 3 | 1 | 74 |
| 14 | A → claude-opus-4-6 | 26 | 71.4% | 0 | 0 | 0 | 74 |
| 15 | D → claude-opus-5 | 21 | 69.6% | 5 | 5 | 0 | 69 |
| 16 | C → gemini-3.8-flash | 11 | 69.2% | 4 | 3 | 1 | 65 |
| 17 | A → claude-opus-5 | 20 | 68.2% | 0 | 0 | 0 | 65 |
| 18 | C → claude-opus-5 | 23 | 68.0% | 0 | 0 | 0 | 65 |
| 19 | D → claude-opus-4-6 | 30 | 65.6% | 3 | 3 | 0 | 62 |
| 20 | A → qwen3.8-max | 24 | 65.4% | 1 | 0 | 1 | 61 |
| 21 | B → claude-opus-5 | 23 | 64.0% | 0 | 0 | 0 | 61 |
| 22 | C → claude-sonnet-5 | 20 | 63.6% | 1 | 1 | 0 | 60 |
| 23 | A → claude-sonnet-5 | 21 | 60.9% | 2 | 1 | 1 | 58 |
| 24 | B → qwen3.8-max | 26 | 60.7% | 0 | 0 | 0 | 58 |
| 25 | D → gemini-3.8-flash | 18 | 60.0% | 3 | 1 | 2 | 55 |
| 26 | D → qwen3.8-max | 28 | 60.0% | 0 | 0 | 0 | 55 |
| 27 | D → claude-sonnet-5 | 33 | 60.0% | 4 | 1 | 3 | 51 |
| 28 | B → claude-opus-4-8 | 27 | 58.6% | 0 | 0 | 0 | 51 |
| 29 | C → claude-opus-4-8 | 22 | 58.3% | 1 | 0 | 1 | 50 |
| 30 | D → gpt-5.6-sol | 14 | 56.2% | 5 | 2 | 3 | 45 |
| 31 | A → gemini-3.8-flash | 23 | 56.0% | 1 | 0 | 1 | 44 |
| 32 | D → gemini-3.6-flash | 36 | 55.3% | 6 | 2 | 4 | 38 |
| 33 | A → claude-opus-4-8 | 20 | 54.5% | 0 | 0 | 0 | 38 |
| 34 | B → gemini-3.8-flash | 28 | 53.3% | 1 | 1 | 0 | 37 |
| 35 | B → gpt-5.5 | 15 | 52.9% | 2 | 1 | 1 | 35 |
| 36 | C → gpt-5.5 | 22 | 50.0% | 4 | 3 | 1 | 31 |
| 37 | A → claude-fable-5 | 12 | 50.0% | 5 | 2 | 3 | 26 |
| 38 | D → kimi-k3 | 12 | 50.0% | 2 | 2 | 0 | 24 |
| 39 | B → kimi-k3 | 27 | 48.3% | 1 | 1 | 0 | 23 |
| 40 | A → gemini-3.6-flash | 25 | 48.1% | 0 | 0 | 0 | 23 |
| 41 | A → gpt-5.6-sol | 21 | 47.8% | 4 | 1 | 3 | 19 |
| 42 | A → gpt-5.5 | 19 | 47.6% | 0 | 0 | 0 | 19 |
| 43 | B → gemini-3.6-flash | 34 | 47.2% | 0 | 0 | 0 | 19 |
| 44 | D → gemini-3.7-flash | 15 | 47.1% | 2 | 1 | 1 | 17 |
| 45 | C → gemini-3.6-flash | 20 | 45.5% | 0 | 0 | 0 | 17 |
| 46 | A → gpt-5.6-terra | 32 | 44.1% | 3 | 2 | 1 | 14 |
| 47 | C → glm-5.3-flash | 30 | 43.8% | 4 | 2 | 2 | 10 |
| 48 | D → gpt-5.6-terra | 28 | 43.3% | 1 | 0 | 1 | 9 |
| 49 | D → gpt-5.5 | 43 | 42.2% | 0 | 0 | 0 | 9 |
| 50 | B → gemini-3.7-flash | 15 | 41.2% | 0 | 0 | 0 | 9 |
| 51 | B → gpt-5.6-terra | 40 | 40.5% | 0 | 0 | 0 | 9 |
| 52 | D → claude-fable-5 | 26 | 39.3% | 4 | 2 | 2 | 5 |
| 53 | A → gpt-5.6-luna | 24 | 38.5% | 2 | 1 | 1 | 3 |
| 54 | D → glm-5.3-flash | 24 | 38.5% | 0 | 0 | 0 | 3 |
| 55 | C → gpt-5.6-terra | 35 | 37.8% | 0 | 0 | 0 | 3 |
| 56 | B → gpt-5.6-sol | 22 | 37.5% | 0 | 0 | 0 | 3 |
| 57 | C → gpt-5.6-luna | 25 | 37.0% | 0 | 0 | 0 | 3 |
| 58 | C → gpt-5.6-sol | 20 | 36.4% | 0 | 0 | 0 | 3 |
| 59 | C → gemini-3.7-flash | 45 | 36.2% | 0 | 0 | 0 | 3 |
| 60 | A → gemini-3.7-flash | 30 | 34.4% | 0 | 0 | 0 | 3 |
| 61 | C → kimi-k3 | 28 | 33.3% | 0 | 0 | 0 | 3 |
| 62 | D → gpt-5.6-luna | 28 | 33.3% | 0 | 0 | 0 | 3 |
| 63 | B → glm-5.3-flash | 41 | 32.6% | 1 | 0 | 1 | 2 |
| 64 | B → gpt-5.6-luna | 11 | 30.8% | 0 | 0 | 0 | 2 |
| 65 | C → claude-fable-5 | 21 | 30.4% | 0 | 0 | 0 | 2 |
| 66 | A → kimi-k3 | 40 | 28.6% | 0 | 0 | 0 | 2 |
| 67 | D → glm-5.3 | 19 | 28.6% | 2 | 1 | 1 | 0 |
| 68 | A → glm-5.3-flash | 34 | 27.8% | 0 | 0 | 0 | 0 |
| 69 | C → glm-5.3 | 25 | 25.9% | 0 | 0 | 0 | 0 |
| 70 | A → glm-5.3 | 24 | 23.1% | 0 | 0 | 0 | 0 |
| 71 | B → glm-5.3 | 21 | 21.7% | 0 | 0 | 0 | 0 |
| 72 | B → claude-fable-5 | 29 | 19.4% | 0 | 0 | 0 | 0 |
| fallback | D fallback | — | — | 0 | 0 | 0 | 0 |

### Fold 4

| Rank | Rule | Fit support | Smoothed precision | Assigned | Correct | Incorrect | Remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C → gemini-3.1-pro-preview | 20 | 90.9% | 5 | 5 | 0 | 103 |
| 2 | A → gpt-6-astra | 19 | 90.5% | 6 | 5 | 1 | 97 |
| 3 | C → gpt-6-astra | 19 | 90.5% | 0 | 0 | 0 | 97 |
| 4 | A → gemini-3.1-pro-preview | 18 | 90.0% | 1 | 0 | 1 | 96 |
| 5 | B → gpt-6-astra | 18 | 90.0% | 1 | 1 | 0 | 95 |
| 6 | B → claude-opus-4-6 | 13 | 86.7% | 7 | 5 | 2 | 88 |
| 7 | B → claude-opus-5 | 19 | 85.7% | 9 | 3 | 6 | 79 |
| 8 | B → gemini-3.1-pro-preview | 19 | 85.7% | 0 | 0 | 0 | 79 |
| 9 | D → claude-opus-4-8 | 15 | 82.4% | 4 | 3 | 1 | 75 |
| 10 | C → claude-opus-4-6 | 20 | 81.8% | 1 | 0 | 1 | 74 |
| 11 | C → claude-opus-5 | 20 | 81.8% | 2 | 0 | 2 | 72 |
| 12 | A → claude-opus-5 | 19 | 81.0% | 1 | 0 | 1 | 71 |
| 13 | D → gpt-6-astra | 18 | 80.0% | 0 | 0 | 0 | 71 |
| 14 | A → claude-opus-4-6 | 21 | 78.3% | 0 | 0 | 0 | 71 |
| 15 | D → gemini-3.1-pro-preview | 28 | 76.7% | 1 | 0 | 1 | 70 |
| 16 | D → claude-opus-4-6 | 27 | 75.9% | 1 | 0 | 1 | 69 |
| 17 | D → claude-opus-5 | 22 | 75.0% | 0 | 0 | 0 | 69 |
| 18 | B → claude-sonnet-5 | 17 | 68.4% | 4 | 4 | 0 | 65 |
| 19 | C → qwen3.8-max | 24 | 65.4% | 4 | 4 | 0 | 61 |
| 20 | C → gemini-3.8-flash | 12 | 64.3% | 3 | 3 | 0 | 58 |
| 21 | C → claude-sonnet-5 | 18 | 60.0% | 2 | 1 | 1 | 56 |
| 22 | A → qwen3.8-max | 27 | 58.6% | 0 | 0 | 0 | 56 |
| 23 | A → claude-opus-4-8 | 22 | 58.3% | 0 | 0 | 0 | 56 |
| 24 | D → gemini-3.8-flash | 22 | 58.3% | 1 | 0 | 1 | 55 |
| 25 | A → claude-sonnet-5 | 24 | 57.7% | 0 | 0 | 0 | 55 |
| 26 | A → gemini-3.8-flash | 24 | 57.7% | 1 | 0 | 1 | 54 |
| 27 | B → qwen3.8-max | 24 | 57.7% | 2 | 1 | 1 | 52 |
| 28 | D → claude-sonnet-5 | 35 | 56.8% | 3 | 1 | 2 | 49 |
| 29 | A → gpt-5.5 | 16 | 55.6% | 7 | 2 | 5 | 42 |
| 30 | D → gemini-3.6-flash | 34 | 55.6% | 8 | 3 | 5 | 34 |
| 31 | D → qwen3.8-max | 27 | 55.2% | 0 | 0 | 0 | 34 |
| 32 | C → gemini-3.6-flash | 18 | 55.0% | 0 | 0 | 0 | 34 |
| 33 | D → gpt-5.6-sol | 18 | 55.0% | 1 | 1 | 0 | 33 |
| 34 | C → gpt-5.5 | 20 | 54.5% | 1 | 1 | 0 | 32 |
| 35 | A → gpt-5.6-terra | 22 | 54.2% | 10 | 3 | 7 | 22 |
| 36 | D → kimi-k3 | 11 | 53.8% | 3 | 2 | 1 | 19 |
| 37 | A → claude-fable-5 | 13 | 53.3% | 2 | 1 | 1 | 17 |
| 38 | B → gpt-5.5 | 13 | 53.3% | 0 | 0 | 0 | 17 |
| 39 | B → claude-opus-4-8 | 30 | 53.1% | 0 | 0 | 0 | 17 |
| 40 | A → gemini-3.6-flash | 21 | 52.2% | 2 | 1 | 1 | 15 |
| 41 | B → kimi-k3 | 23 | 52.0% | 1 | 1 | 0 | 14 |
| 42 | B → gemini-3.8-flash | 25 | 51.9% | 1 | 0 | 1 | 13 |
| 43 | C → claude-opus-4-8 | 31 | 51.5% | 0 | 0 | 0 | 13 |
| 44 | B → gemini-3.6-flash | 35 | 51.4% | 0 | 0 | 0 | 13 |
| 45 | B → gemini-3.7-flash | 18 | 50.0% | 1 | 1 | 0 | 12 |
| 46 | D → gemini-3.7-flash | 12 | 50.0% | 0 | 0 | 0 | 12 |
| 47 | D → gpt-5.6-terra | 19 | 47.6% | 0 | 0 | 0 | 12 |
| 48 | A → gpt-5.6-sol | 32 | 47.1% | 0 | 0 | 0 | 12 |
| 49 | D → claude-fable-5 | 31 | 45.5% | 2 | 0 | 2 | 10 |
| 50 | B → gpt-5.6-terra | 36 | 44.7% | 0 | 0 | 0 | 10 |
| 51 | C → glm-5.3-flash | 28 | 43.3% | 5 | 2 | 3 | 5 |
| 52 | C → gpt-5.6-terra | 28 | 43.3% | 1 | 0 | 1 | 4 |
| 53 | A → gpt-5.6-luna | 26 | 42.9% | 1 | 0 | 1 | 3 |
| 54 | C → gpt-5.6-sol | 26 | 42.9% | 0 | 0 | 0 | 3 |
| 55 | C → gpt-5.6-luna | 27 | 41.4% | 0 | 0 | 0 | 3 |
| 56 | D → gpt-5.5 | 42 | 40.9% | 0 | 0 | 0 | 3 |
| 57 | C → gemini-3.7-flash | 45 | 40.4% | 0 | 0 | 0 | 3 |
| 58 | B → gpt-5.6-sol | 31 | 39.4% | 0 | 0 | 0 | 3 |
| 59 | C → kimi-k3 | 24 | 38.5% | 1 | 0 | 1 | 2 |
| 60 | D → gpt-5.6-luna | 31 | 36.4% | 0 | 0 | 0 | 2 |
| 61 | B → gpt-5.6-luna | 15 | 35.3% | 0 | 0 | 0 | 2 |
| 62 | A → gemini-3.7-flash | 33 | 34.3% | 0 | 0 | 0 | 2 |
| 63 | A → kimi-k3 | 35 | 32.4% | 0 | 0 | 0 | 2 |
| 64 | B → glm-5.3-flash | 35 | 32.4% | 1 | 1 | 0 | 1 |
| 65 | D → glm-5.3-flash | 20 | 31.8% | 0 | 0 | 0 | 1 |
| 66 | C → glm-5.3 | 30 | 31.2% | 1 | 0 | 1 | 0 |
| 67 | C → claude-fable-5 | 22 | 29.2% | 0 | 0 | 0 | 0 |
| 68 | D → glm-5.3 | 20 | 27.3% | 0 | 0 | 0 | 0 |
| 69 | B → claude-fable-5 | 35 | 27.0% | 0 | 0 | 0 | 0 |
| 70 | A → glm-5.3-flash | 33 | 25.7% | 0 | 0 | 0 | 0 |
| 71 | B → glm-5.3 | 26 | 25.0% | 0 | 0 | 0 | 0 |
| 72 | A → glm-5.3 | 27 | 24.1% | 0 | 0 | 0 | 0 |
| fallback | D fallback | — | — | 0 | 0 | 0 | 0 |

### Fold 5

| Rank | Rule | Fit support | Smoothed precision | Assigned | Correct | Incorrect | Remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B → gpt-6-astra | 16 | 94.4% | 6 | 5 | 1 | 102 |
| 2 | C → gemini-3.1-pro-preview | 21 | 91.3% | 4 | 4 | 0 | 98 |
| 3 | A → gpt-6-astra | 20 | 90.9% | 0 | 0 | 0 | 98 |
| 4 | C → gpt-6-astra | 20 | 90.9% | 1 | 0 | 1 | 97 |
| 5 | A → gemini-3.1-pro-preview | 20 | 86.4% | 1 | 1 | 0 | 96 |
| 6 | B → claude-sonnet-5 | 16 | 83.3% | 5 | 2 | 3 | 91 |
| 7 | B → gemini-3.1-pro-preview | 21 | 82.6% | 0 | 0 | 0 | 91 |
| 8 | C → gemini-3.8-flash (skipped: support < minimum) | 9 | 81.8% | 0 | 0 | 0 | 91 |
| 9 | B → claude-opus-4-6 | 18 | 80.0% | 2 | 2 | 0 | 89 |
| 10 | D → claude-opus-4-8 | 18 | 80.0% | 1 | 1 | 0 | 88 |
| 11 | D → gpt-6-astra | 17 | 78.9% | 0 | 0 | 0 | 88 |
| 12 | C → claude-opus-4-6 | 24 | 76.9% | 2 | 1 | 1 | 86 |
| 13 | A → claude-opus-4-6 | 23 | 76.0% | 1 | 1 | 0 | 85 |
| 14 | C → claude-opus-5 | 22 | 75.0% | 6 | 4 | 2 | 79 |
| 15 | D → gemini-3.1-pro-preview | 29 | 74.2% | 0 | 0 | 0 | 79 |
| 16 | D → claude-opus-5 | 21 | 73.9% | 0 | 0 | 0 | 79 |
| 17 | A → claude-opus-5 | 20 | 72.7% | 0 | 0 | 0 | 79 |
| 18 | D → claude-opus-4-6 | 30 | 71.9% | 1 | 1 | 0 | 78 |
| 19 | B → claude-opus-5 | 23 | 68.0% | 0 | 0 | 0 | 78 |
| 20 | C → qwen3.8-max | 23 | 68.0% | 6 | 5 | 1 | 72 |
| 21 | D → qwen3.8-max | 23 | 64.0% | 3 | 0 | 3 | 69 |
| 22 | C → claude-sonnet-5 | 22 | 62.5% | 1 | 0 | 1 | 68 |
| 23 | C → gemini-3.6-flash | 19 | 61.9% | 4 | 1 | 3 | 64 |
| 24 | D → claude-sonnet-5 | 32 | 61.8% | 7 | 4 | 3 | 57 |
| 25 | A → claude-sonnet-5 | 24 | 61.5% | 0 | 0 | 0 | 57 |
| 26 | C → claude-opus-4-8 | 24 | 61.5% | 2 | 0 | 2 | 55 |
| 27 | A → claude-opus-4-8 | 21 | 60.9% | 0 | 0 | 0 | 55 |
| 28 | A → gemini-3.8-flash | 21 | 60.9% | 7 | 3 | 4 | 48 |
| 29 | D → gemini-3.8-flash | 18 | 60.0% | 0 | 0 | 0 | 48 |
| 30 | B → qwen3.8-max | 25 | 59.3% | 0 | 0 | 0 | 48 |
| 31 | A → qwen3.8-max | 24 | 57.7% | 0 | 0 | 0 | 48 |
| 32 | B → claude-opus-4-8 | 31 | 57.6% | 0 | 0 | 0 | 48 |
| 33 | B → gemini-3.8-flash | 26 | 57.1% | 1 | 1 | 0 | 47 |
| 34 | B → kimi-k3 | 25 | 55.6% | 4 | 2 | 2 | 43 |
| 35 | C → gpt-5.5 | 25 | 55.6% | 3 | 1 | 2 | 40 |
| 36 | A → gemini-3.6-flash | 20 | 54.5% | 3 | 2 | 1 | 37 |
| 37 | D → kimi-k3 | 11 | 53.8% | 0 | 0 | 0 | 37 |
| 38 | B → gpt-5.5 | 13 | 53.3% | 1 | 1 | 0 | 36 |
| 39 | B → gemini-3.6-flash | 30 | 53.1% | 2 | 2 | 0 | 34 |
| 40 | A → gpt-5.5 | 21 | 52.2% | 0 | 0 | 0 | 34 |
| 41 | D → gemini-3.6-flash | 33 | 51.4% | 1 | 1 | 0 | 33 |
| 42 | D → gpt-5.6-sol | 16 | 50.0% | 4 | 3 | 1 | 29 |
| 43 | A → claude-fable-5 | 14 | 50.0% | 2 | 0 | 2 | 27 |
| 44 | D → claude-fable-5 | 29 | 48.4% | 3 | 1 | 2 | 24 |
| 45 | B → gemini-3.7-flash | 19 | 47.6% | 1 | 1 | 0 | 23 |
| 46 | C → glm-5.3-flash | 24 | 46.2% | 5 | 2 | 3 | 18 |
| 47 | A → gpt-5.6-sol | 25 | 44.4% | 4 | 1 | 3 | 14 |
| 48 | D → gpt-5.5 | 47 | 42.9% | 2 | 0 | 2 | 12 |
| 49 | C → gemini-3.7-flash | 45 | 42.6% | 0 | 0 | 0 | 12 |
| 50 | A → gpt-5.6-terra | 29 | 41.9% | 4 | 3 | 1 | 8 |
| 51 | D → gpt-5.6-terra | 27 | 41.4% | 0 | 0 | 0 | 8 |
| 52 | D → gemini-3.7-flash | 15 | 41.2% | 0 | 0 | 0 | 8 |
| 53 | B → gpt-5.6-terra | 37 | 41.0% | 1 | 0 | 1 | 7 |
| 54 | B → gpt-5.6-sol | 28 | 40.0% | 1 | 0 | 1 | 6 |
| 55 | A → gemini-3.7-flash | 34 | 38.9% | 0 | 0 | 0 | 6 |
| 56 | C → gpt-5.6-terra | 32 | 38.2% | 0 | 0 | 0 | 6 |
| 57 | A → gpt-5.6-luna | 22 | 37.5% | 3 | 2 | 1 | 3 |
| 58 | C → gpt-5.6-luna | 25 | 37.0% | 0 | 0 | 0 | 3 |
| 59 | C → gpt-5.6-sol | 20 | 36.4% | 0 | 0 | 0 | 3 |
| 60 | D → glm-5.3-flash | 20 | 36.4% | 0 | 0 | 0 | 3 |
| 61 | B → glm-5.3-flash | 39 | 34.1% | 2 | 0 | 2 | 1 |
| 62 | C → kimi-k3 | 25 | 33.3% | 0 | 0 | 0 | 1 |
| 63 | B → claude-fable-5 | 27 | 31.0% | 0 | 0 | 0 | 1 |
| 64 | C → claude-fable-5 | 24 | 30.8% | 0 | 0 | 0 | 1 |
| 65 | D → gpt-5.6-luna | 24 | 30.8% | 0 | 0 | 0 | 1 |
| 66 | C → glm-5.3 | 28 | 30.0% | 1 | 0 | 1 | 0 |
| 67 | D → glm-5.3 | 22 | 29.2% | 0 | 0 | 0 | 0 |
| 68 | A → kimi-k3 | 33 | 28.6% | 0 | 0 | 0 | 0 |
| 69 | B → glm-5.3 | 26 | 28.6% | 0 | 0 | 0 | 0 |
| 70 | B → gpt-5.6-luna | 12 | 28.6% | 0 | 0 | 0 | 0 |
| 71 | A → glm-5.3 | 27 | 27.6% | 0 | 0 | 0 | 0 |
| 72 | A → glm-5.3-flash | 34 | 25.0% | 0 | 0 | 0 | 0 |
| fallback | D fallback | — | — | 0 | 0 | 0 | 0 |

Artifacts: `oof_predictions.parquet`, `routing_decisions.parquet`, `rule_trace.parquet`, `rule_reliability.json`, and `metrics.json`.
