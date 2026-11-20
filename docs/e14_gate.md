# E14-B Focused Intervention Gate

Trust List: **trust-list-2-v037** · V0 rows: **2699** · five prompt-group outer folds.

E13-B remains the champion and is not modified. For each example, the gate evaluates switching from E13-B's family-max reliability label to its strongest differently labelled supported competitor. A CatBoost gate switches at a fixed probability threshold of 0.5.

E13-B baseline: **64.6536% accuracy**, **64.6465% balanced accuracy**; Kaggle public score **0.606** is reported separately.

## Result

| Method | Accuracy | Balanced accuracy | Δ balanced accuracy (pp) | Corrected | Introduced | Switched |
|---|---:|---:|---:|---:|---:|---:|
| E13-B | 64.6536% | 64.6465% | — | — | — | 0 |
| E14-B-GATE | 65.0241% | 65.0164% | +0.3699 pp | 52 | 42 | 118 |

The gate had a distinct supported competitor on 2621 rows; on 78 rows it kept E13-B unchanged.

## Fold results

| Fold | Rows | Gate eligible | No competitor | Accuracy | Balanced accuracy | Corrected | Introduced | Switches | Trees |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 540 | 525 | 15 | 63.3333% | 63.3333% | 13 | 13 | 32 | 1500 |
| 2 | 540 | 533 | 7 | 64.4444% | 64.4444% | 5 | 2 | 10 | 1500 |
| 3 | 540 | 520 | 20 | 66.6667% | 66.6667% | 14 | 12 | 32 | 1500 |
| 4 | 540 | 528 | 12 | 64.2593% | 64.2593% | 9 | 6 | 20 | 1500 |
| 5 | 539 | 515 | 24 | 66.4193% | 66.3857% | 11 | 9 | 24 | 1500 |

## Per-label precision and recall

| Label | Support | Precision | Recall |
|---|---:|---:|---:|
| claude-fable-5 | 149 | 40.0000% | 44.2953% |
| claude-opus-4-6 | 150 | 75.1412% | 88.6667% |
| claude-opus-4-8 | 150 | 71.7105% | 72.6667% |
| claude-opus-5 | 150 | 85.5072% | 78.6667% |
| claude-sonnet-5 | 150 | 68.9189% | 68.0000% |
| gemini-3.1-pro-preview | 150 | 87.7551% | 86.0000% |
| gemini-3.6-flash | 150 | 69.8864% | 82.0000% |
| gemini-3.7-flash | 150 | 59.1667% | 47.3333% |
| gemini-3.8-flash | 150 | 59.2593% | 64.0000% |
| glm-5.3 | 150 | 38.3333% | 30.6667% |
| glm-5.3-flash | 150 | 43.6364% | 48.0000% |
| gpt-5.5 | 150 | 61.5385% | 64.0000% |
| gpt-5.6-luna | 150 | 63.2911% | 66.6667% |
| gpt-5.6-sol | 150 | 57.4468% | 54.0000% |
| gpt-5.6-terra | 150 | 61.6883% | 63.3333% |
| gpt-6-astra | 150 | 83.7662% | 86.0000% |
| kimi-k3 | 150 | 63.0252% | 50.0000% |
| qwen3.8-max | 150 | 77.5510% | 76.0000% |

## Confusion matrix

```text
	claude-fable-5	claude-opus-4-6	claude-opus-4-8	claude-opus-5	claude-sonnet-5	gemini-3.1-pro-preview	gemini-3.6-flash	gemini-3.7-flash	gemini-3.8-flash	glm-5.3	glm-5.3-flash	gpt-5.5	gpt-5.6-luna	gpt-5.6-sol	gpt-5.6-terra	gpt-6-astra	kimi-k3	qwen3.8-max
claude-fable-5	66	9	12	5	7	0	0	0	2	18	26	0	0	1	0	0	3	0
claude-opus-4-6	4	133	1	0	4	0	0	0	3	3	1	0	0	0	0	0	1	0
claude-opus-4-8	13	5	109	2	11	0	0	0	0	4	2	0	0	0	0	0	3	1
claude-opus-5	7	2	2	118	5	0	0	0	0	4	9	0	0	0	0	0	3	0
claude-sonnet-5	12	13	11	0	102	0	1	0	1	2	3	1	0	0	0	0	4	0
gemini-3.1-pro-preview	0	1	0	0	0	129	12	5	2	1	0	0	0	0	0	0	0	0
gemini-3.6-flash	0	0	0	0	1	8	123	11	7	0	0	0	0	0	0	0	0	0
gemini-3.7-flash	0	1	0	0	0	6	24	71	48	0	0	0	0	0	0	0	0	0
gemini-3.8-flash	1	0	0	0	1	4	14	33	96	0	0	0	0	0	0	0	0	1
glm-5.3	32	8	4	5	6	0	0	0	0	46	44	0	1	0	0	0	4	0
glm-5.3-flash	20	1	7	4	7	0	0	0	0	30	72	0	1	0	0	0	6	2
gpt-5.5	0	0	1	1	0	0	0	0	0	3	0	96	5	9	16	0	10	9
gpt-5.6-luna	0	0	1	1	0	0	0	0	0	0	1	2	100	13	17	7	2	6
gpt-5.6-sol	0	0	0	0	0	0	0	0	1	0	0	8	25	81	20	8	1	6
gpt-5.6-terra	0	0	1	0	0	0	0	0	0	0	0	14	12	20	95	6	0	2
gpt-6-astra	0	1	0	0	0	0	0	0	0	0	0	2	9	6	1	129	1	1
kimi-k3	9	3	2	2	2	0	0	0	0	9	7	21	1	9	3	2	75	5
qwen3.8-max	1	0	1	0	2	0	2	0	2	0	0	12	4	2	2	2	6	114
```

Features include E13-B label scores and reliability margins, family-balanced probability margins, checkpoint probability margins, directional family-vote confusion, smoothed pairwise historical wins, and family overlap/support differences.

Artifacts: `artifacts/analysis/e14_gate/20261120T180106177007Z` (fold models, cross-fitted features, predictions, configuration and results).
