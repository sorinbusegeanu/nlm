# November 16 ensemble hypotheses

Cache-only equal-probability evaluation on all 2,699 V0 rows. No model training, inference, production ensemble changes, or Kaggle submissions were performed.

**Trust-list source:** `docs/trust-list-2-v025.csv` (repository_latest_pointer); SHA-256 `c93ae4cc0612ed36b7c126d6d50a5b50456ecfafd4e835369272076442e8cf9b`.
The named `2026-11-16T18-56_export.csv` was not present locally. Checkpoint IDs were resolved against the repository latest pointer, which pointed to `trust-list-2-v025`; all 12 requested IDs resolved uniquely there. The baseline reproduced the supplied score.

The Kaggle accuracy column below is an estimate using V0 balanced accuracy as a proxy; it is not a Kaggle score or a calibrated prediction of leaderboard accuracy. These V0 scores are development diagnostics because the checkpoint trust scores and checkpoint selection use V0.

## Results

| Experiment | Checkpoints | V0 accuracy | V0 balanced accuracy | Δ BA vs EQUAL9 | Log loss | Corrected | Introduced | Prediction changes vs EQUAL9 | Kaggle accuracy estimate (V0 BA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EQUAL9 | 9 | 65.3946% | 65.3898% | — | 1.132076 | — | — | — | 65.3898% |
| E1_replace_ultrabert | 9 | 64.6165% | 64.6135% | -0.776 pp | 1.136782 | 85 | 106 | 290 (10.74%) | 64.6135% |
| E2_add_diversity | 12 | 64.3201% | 64.3149% | -1.075 pp | 1.076320 | 87 | 116 | 299 (11.08%) | 64.3149% |
| E3_refresh_neural | 10 | 64.5795% | 64.5717% | -0.818 pp | 1.111992 | 131 | 153 | 402 (14.89%) | 64.5717% |

Equal weights were used in every row: `1/N` per checkpoint.

## Per-label changes

Recall and precision changes are percentage points versus EQUAL9.

| Model label | EQUAL9 recall / precision | E1 Δ recall / precision | E2 Δ recall / precision | E3 Δ recall / precision |
|---|---:|---:|---:|---:|
| claude-fable-5 | 52.3% / 40.6% | +4.0 / -0.6 pp | -2.0 / -2.0 pp | -8.7 / +0.5 pp |
| claude-opus-4-6 | 80.0% / 85.7% | +2.7 / -1.4 pp | +0.7 / -2.8 pp | +2.7 / -1.4 pp |
| claude-opus-4-8 | 76.7% / 75.7% | -8.0 / +2.4 pp | -4.0 / +3.9 pp | -2.7 / +1.4 pp |
| claude-opus-5 | 78.7% / 90.8% | -2.0 / -4.3 pp | -3.3 / -0.4 pp | -0.7 / -3.5 pp |
| claude-sonnet-5 | 73.3% / 68.8% | +0.7 / +2.4 pp | +0.7 / -1.5 pp | +1.3 / +1.2 pp |
| gemini-3.1-pro-preview | 78.0% / 95.9% | +4.0 / -5.5 pp | +2.0 / +0.1 pp | +3.3 / -2.1 pp |
| gemini-3.6-flash | 80.7% / 63.7% | -0.7 / +2.2 pp | +3.3 / +2.6 pp | +3.3 / +0.9 pp |
| gemini-3.7-flash | 22.0% / 66.0% | +0.7 / -1.8 pp | -12.0 / -6.0 pp | -8.7 / +3.0 pp |
| gemini-3.8-flash | 83.3% / 50.0% | -1.3 / +0.4 pp | +3.3 / -2.6 pp | +2.7 / -1.5 pp |
| glm-5.3 | 45.3% / 39.8% | -2.0 / -2.6 pp | -8.7 / -5.2 pp | -4.7 / -0.7 pp |
| glm-5.3-flash | 48.7% / 52.1% | +0.7 / -0.4 pp | -3.3 / -5.2 pp | +3.3 / -7.3 pp |
| gpt-5.5 | 58.0% / 65.9% | +3.3 / -0.2 pp | +5.3 / -3.4 pp | +1.3 / -3.7 pp |
| gpt-5.6-luna | 61.3% / 71.9% | -7.3 / +1.8 pp | +0.7 / -2.0 pp | -3.3 / +0.0 pp |
| gpt-5.6-sol | 54.7% / 64.1% | -6.7 / -6.0 pp | -0.7 / -3.2 pp | -4.7 / -3.6 pp |
| gpt-5.6-terra | 77.3% / 56.0% | -4.0 / -2.4 pp | -4.7 / +1.6 pp | -3.3 / +0.0 pp |
| gpt-6-astra | 72.0% / 90.0% | +0.0 / +0.0 pp | +1.3 / +1.7 pp | +4.0 / -3.0 pp |
| kimi-k3 | 53.3% / 67.8% | +3.3 / -0.3 pp | +2.0 / -2.4 pp | -1.3 / -2.8 pp |
| qwen3.8-max | 81.3% / 72.2% | -1.3 / +1.4 pp | +0.0 / +4.1 pp | +1.3 / +1.2 pp |

## Prediction similarity

Pairwise disagreement counts compare top-1 labels on identical V0 rows.

| Pair | Rows differing | Disagreement |
|---|---:|---:|
| EQUAL9 vs E1_replace_ultrabert | 290 | 10.74% |
| EQUAL9 vs E2_add_diversity | 299 | 11.08% |
| EQUAL9 vs E3_refresh_neural | 402 | 14.89% |
| E1_replace_ultrabert vs E2_add_diversity | 348 | 12.89% |
| E1_replace_ultrabert vs E3_refresh_neural | 345 | 12.78% |
| E2_add_diversity vs E3_refresh_neural | 226 | 8.37% |

## Recommendation

EQUAL9 reproduces the supplied baseline: 65.3898% balanced accuracy, 1765/2,699 correct.
Best V0 diagnostic among the three hypotheses: **E1_replace_ultrabert**, 64.6135% balanced accuracy (-0.776 pp vs baseline).
The alternatives are prediction-distinct: they change 10.7%, 11.1%, and 14.9% of rows versus EQUAL9. E1 vs E2 disagree on 12.9%, E1 vs E3 on 12.8%, and E2 vs E3 on 8.4%.
Recommendation: do not submit these three variants based on this evaluation; all score below EQUAL9 on V0 balanced accuracy (by 0.776, 1.075, and 0.818 pp). Their predictions differ substantially, but did not improve this metric. V0 is not an independent leaderboard estimate, so a Kaggle submission could still test distribution shift, but these results do not favor spending submissions on them.

## Reproducibility

- Machine-readable configuration, trust mappings and source cache hashes: `artifacts/analysis/ensemble_hypotheses_nov16/experiment_configs.json`.
- Metrics and per-label values: `artifacts/analysis/ensemble_hypotheses_nov16/evaluation_metrics.json`.
- Row-aligned ensemble predictions and probabilities: `artifacts/analysis/ensemble_hypotheses_nov16/v0_ensemble_predictions.parquet`.
- Checkpoint cache inventory: `artifacts/analysis/ensemble_hypotheses_nov16/checkpoint_cache_inventory.csv`.
- All predictions were checked for exact ordered V0 IDs, matching true labels, canonical 18-label ordering, finite normalized probabilities, and valid checkpoint paths.
