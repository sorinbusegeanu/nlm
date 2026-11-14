# Trust List 2 extended weighted ensembles: B6–B10

Cache-only evaluation on all 2,699 V0 responses. These are exploratory results because V0 was used for checkpoint selection and prior ensemble optimization. No inference, weight fitting, checkpoint reselection, Kaggle operation, or production change was performed.

## Results

The exact WEIGHTED5 weights were read from `docs/trust_list_2_w6_w7_evaluation.md`. Reconstructed W5 balanced accuracy is 60.3888%, and all 2,699 W5 row predictions exactly match the saved W6/W7 evaluation artifact. The previous W7-Combined reference is 61.9456%.

| Rank | Experiment | Base size | Variant | V0 accuracy | V0 balanced accuracy | Correct | Δ BA vs Bk | Δ BA vs W5 | Corrected vs Bk | Regressed vs Bk |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | B10-Baseline | 10 | Baseline | 63.9126% | 63.9073% | 1,725/2,699 | +0.000 pp | +3.519 pp | 0 | 0 |
| 2 | B10-Qwen | 10 | Qwen | 63.8014% | 63.7964% | 1,722/2,699 | -0.111 pp | +3.408 pp | 55 | 58 |
| 3 | B9-Qwen | 9 | Qwen | 63.6162% | 63.6112% | 1,717/2,699 | +0.333 pp | +3.222 pp | 67 | 58 |
| 4 | B9-Filtered | 9 | Filtered | 63.3197% | 63.3152% | 1,709/2,699 | +0.037 pp | +2.926 pp | 78 | 77 |
| 5 | B9-Baseline | 9 | Baseline | 63.2827% | 63.2782% | 1,708/2,699 | +0.000 pp | +2.889 pp | 0 | 0 |
| 6 | B9-Combined | 9 | Combined | 63.1345% | 63.1307% | 1,704/2,699 | -0.147 pp | +2.742 pp | 97 | 101 |
| 7 | B10-Filtered | 10 | Filtered | 63.0233% | 63.0194% | 1,701/2,699 | -0.888 pp | +2.631 pp | 53 | 77 |
| 8 | B7-Qwen | 7 | Qwen | 62.8751% | 62.8705% | 1,697/2,699 | +0.444 pp | +2.482 pp | 54 | 42 |
| 9 | B8-Filtered | 8 | Filtered | 62.8010% | 62.7969% | 1,695/2,699 | +0.037 pp | +2.408 pp | 67 | 66 |
| 10 | B8-Baseline | 8 | Baseline | 62.7640% | 62.7596% | 1,694/2,699 | +0.000 pp | +2.371 pp | 0 | 0 |
| 11 | B8-Combined | 8 | Combined | 62.6528% | 62.6493% | 1,691/2,699 | -0.110 pp | +2.261 pp | 88 | 91 |
| 12 | B8-Qwen | 8 | Qwen | 62.5787% | 62.5739% | 1,689/2,699 | -0.186 pp | +2.185 pp | 52 | 57 |
| 13 | B7-Filtered | 7 | Filtered | 62.5417% | 62.5369% | 1,688/2,699 | +0.111 pp | +2.148 pp | 64 | 61 |
| 14 | B6-Combined | 6 | Combined | 62.5046% | 62.5014% | 1,687/2,699 | +0.631 pp | +2.113 pp | 87 | 70 |
| 15 | B7-Combined | 7 | Combined | 62.4676% | 62.4638% | 1,686/2,699 | +0.038 pp | +2.075 pp | 89 | 88 |
| 16 | B10-Combined | 10 | Combined | 62.4676% | 62.4633% | 1,686/2,699 | -1.444 pp | +2.075 pp | 68 | 107 |
| 17 | B7-Baseline | 7 | Baseline | 62.4305% | 62.4261% | 1,685/2,699 | +0.000 pp | +2.037 pp | 0 | 0 |
| 18 | B6-Qwen | 6 | Qwen | 62.2823% | 62.2789% | 1,681/2,699 | +0.408 pp | +1.890 pp | 56 | 45 |
| 19 | B6-Filtered | 6 | Filtered | 62.2082% | 62.2046% | 1,679/2,699 | +0.334 pp | +1.816 pp | 60 | 51 |
| 20 | B6-Baseline | 6 | Baseline | 61.8748% | 61.8707% | 1,670/2,699 | +0.000 pp | +1.482 pp | 0 | 0 |

## Conclusions

- Best Bk probability mean by balanced accuracy: **B10** at 63.9073%.
- Best of the 20 combinations: **B10-Baseline** at 63.9073%, +1.962 percentage points versus previous W7-Combined.
- Beats previous W7-Combined (61.9456%): **yes**.
- Base Bk balanced accuracy across B6–B10: B6 61.8707%, B7 62.4261%, B8 62.7596%, B9 63.2782%, B10 63.9073%; monotonic: yes.
- Best variant at each size: B6 B6-Combined 62.5014%, B7 B7-Qwen 62.8705%, B8 B8-Filtered 62.7969%, B9 B9-Qwen 63.6112%, B10 B10-Baseline 63.9073%; monotonic: no.

## Per-label changes for top three

Deltas below are accuracy percentage points versus the corresponding Bk baseline, using the true label for each V0 row.

### B10-Baseline (rank 1)

| Label | Support | Bk accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|---:|
| claude-fable-5 | 149 | 49.7% | 49.7% | +0.0 pp |
| claude-opus-4-6 | 150 | 77.3% | 77.3% | +0.0 pp |
| claude-opus-4-8 | 150 | 70.0% | 70.0% | +0.0 pp |
| claude-opus-5 | 150 | 78.0% | 78.0% | +0.0 pp |
| claude-sonnet-5 | 150 | 72.0% | 72.0% | +0.0 pp |
| gemini-3.1-pro-preview | 150 | 78.0% | 78.0% | +0.0 pp |
| gemini-3.6-flash | 150 | 80.7% | 80.7% | +0.0 pp |
| gemini-3.7-flash | 150 | 23.3% | 23.3% | +0.0 pp |
| gemini-3.8-flash | 150 | 80.0% | 80.0% | +0.0 pp |
| glm-5.3 | 150 | 40.7% | 40.7% | +0.0 pp |
| glm-5.3-flash | 150 | 39.3% | 39.3% | +0.0 pp |
| gpt-5.5 | 150 | 57.3% | 57.3% | +0.0 pp |
| gpt-5.6-luna | 150 | 60.7% | 60.7% | +0.0 pp |
| gpt-5.6-sol | 150 | 56.0% | 56.0% | +0.0 pp |
| gpt-5.6-terra | 150 | 75.3% | 75.3% | +0.0 pp |
| gpt-6-astra | 150 | 75.3% | 75.3% | +0.0 pp |
| kimi-k3 | 150 | 55.3% | 55.3% | +0.0 pp |
| qwen3.8-max | 150 | 81.3% | 81.3% | +0.0 pp |

Gains: none. Regressions: none.

### B10-Qwen (rank 2)

| Label | Support | Bk accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|---:|
| claude-fable-5 | 149 | 49.7% | 50.3% | +0.7 pp |
| claude-opus-4-6 | 150 | 77.3% | 80.7% | +3.3 pp |
| claude-opus-4-8 | 150 | 70.0% | 72.0% | +2.0 pp |
| claude-opus-5 | 150 | 78.0% | 77.3% | -0.7 pp |
| claude-sonnet-5 | 150 | 72.0% | 72.0% | +0.0 pp |
| gemini-3.1-pro-preview | 150 | 78.0% | 76.7% | -1.3 pp |
| gemini-3.6-flash | 150 | 80.7% | 80.0% | -0.7 pp |
| gemini-3.7-flash | 150 | 23.3% | 18.0% | -5.3 pp |
| gemini-3.8-flash | 150 | 80.0% | 83.3% | +3.3 pp |
| glm-5.3 | 150 | 40.7% | 39.3% | -1.3 pp |
| glm-5.3-flash | 150 | 39.3% | 39.3% | +0.0 pp |
| gpt-5.5 | 150 | 57.3% | 60.0% | +2.7 pp |
| gpt-5.6-luna | 150 | 60.7% | 64.7% | +4.0 pp |
| gpt-5.6-sol | 150 | 56.0% | 51.3% | -4.7 pp |
| gpt-5.6-terra | 150 | 75.3% | 73.3% | -2.0 pp |
| gpt-6-astra | 150 | 75.3% | 74.7% | -0.7 pp |
| kimi-k3 | 150 | 55.3% | 54.0% | -1.3 pp |
| qwen3.8-max | 150 | 81.3% | 81.3% | +0.0 pp |

Gains: claude-fable-5, claude-opus-4-6, claude-opus-4-8, gemini-3.8-flash, gpt-5.5, gpt-5.6-luna. Regressions: claude-opus-5, gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, glm-5.3, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3.

### B9-Qwen (rank 3)

| Label | Support | Bk accuracy | Variant accuracy | Δ accuracy |
|---|---:|---:|---:|---:|
| claude-fable-5 | 149 | 51.0% | 50.3% | -0.7 pp |
| claude-opus-4-6 | 150 | 78.7% | 82.0% | +3.3 pp |
| claude-opus-4-8 | 150 | 64.0% | 69.3% | +5.3 pp |
| claude-opus-5 | 150 | 77.3% | 77.3% | +0.0 pp |
| claude-sonnet-5 | 150 | 73.3% | 72.0% | -1.3 pp |
| gemini-3.1-pro-preview | 150 | 79.3% | 78.0% | -1.3 pp |
| gemini-3.6-flash | 150 | 80.0% | 80.0% | +0.0 pp |
| gemini-3.7-flash | 150 | 22.0% | 14.7% | -7.3 pp |
| gemini-3.8-flash | 150 | 82.0% | 83.3% | +1.3 pp |
| glm-5.3 | 150 | 42.0% | 40.7% | -1.3 pp |
| glm-5.3-flash | 150 | 34.7% | 38.7% | +4.0 pp |
| gpt-5.5 | 150 | 61.3% | 62.0% | +0.7 pp |
| gpt-5.6-luna | 150 | 61.3% | 66.7% | +5.3 pp |
| gpt-5.6-sol | 150 | 49.3% | 50.7% | +1.3 pp |
| gpt-5.6-terra | 150 | 73.3% | 72.7% | -0.7 pp |
| gpt-6-astra | 150 | 78.0% | 76.0% | -2.0 pp |
| kimi-k3 | 150 | 50.7% | 51.3% | +0.7 pp |
| qwen3.8-max | 150 | 80.7% | 79.3% | -1.3 pp |

Gains: claude-opus-4-6, claude-opus-4-8, gemini-3.8-flash, glm-5.3-flash, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, kimi-k3. Regressions: claude-fable-5, claude-sonnet-5, gemini-3.1-pro-preview, gemini-3.7-flash, glm-5.3, gpt-5.6-terra, gpt-6-astra, qwen3.8-max.

## Fixed membership, weights, and inputs

- Greedy order from the v013 subset artifact: `N0071, N0056, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048`.
- Original WEIGHTED5 weights: `{"N0071":0.2811918866145629,"N0056":0.2324715095557317,"N0005":0.1833526736593965,"N0058":0.2345455729093069,"N0072":0.0684383572610019}`.
- Each Bk is `(5 × W5 + sum(P_i for i=6..k)) / k`; this retains relative W5 weights and assigns each added checkpoint weight `1/k`.
- Per size, variants use fixed coefficients: Baseline `Bk`; Qwen `0.85 Bk + 0.15 N0074`; Filtered `0.90 Bk + 0.10 N0076`; Combined `0.75 Bk + 0.15 N0074 + 0.10 N0076`.
- All cached probability matrices were checked for exactly 2,699 unique V0 IDs, canonical 18-class ordering, matching labels/prompt IDs, finite nonnegative values, and unit row sums.
- Cache paths, hashes, manifests, IDs, class order and formulas are recorded in `artifacts/analysis/trust2_extended_weighted/evaluation_metrics.json`.

## Artifacts

- `artifacts/analysis/trust2_extended_weighted/evaluation_metrics.json`.
- `artifacts/analysis/trust2_extended_weighted/v0_ensemble_predictions.parquet` (probabilities, predictions and correctness flags for the 20 runs plus W5/W7 references).
