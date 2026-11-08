# Checkpoint comparison

## A. Overall comparison

- Model: `ultrabert380`
- Previous: `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0`; dataset `V1 → V0`; checkpoint `ultrabert380_continued`.
- New: `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff`; dataset `V1-all → V0`; checkpoint `ultrabert380_continued`.
- Frozen V0 validation rows: 540; classes: 18; split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`; validation IDs SHA-256: `d3f39a2003dfdd9e99e5d662465066bd60368f8e43864b6ed10a733edbe74f9d`.
- Prediction source: previous `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0_validation.parquet`; new `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff_validation.parquet`.

| Metric | Previous | New | Change |
|---|---:|---:|---:|
| Balanced accuracy | 53.1% | 55.2% | +2.0 pp |
| Top-3 accuracy | 83.7% | 86.3% | +2.6 pp |
| MRR | 0.6970 | 0.7166 | +0.0196 |
| Log loss | 13.4997 | 2.5716 | -10.9280 |

## B. Per-model comparison

| Target model | Top-1 recall: previous → new | Top-3 accuracy: previous → new | Δ MRR (new − previous) |
|---|---:|---:|---:|
| **Claude Fable 5** | 23.3% → 53.3% | 73.3% → 73.3% | +0.1587 |
| Claude Opus 4 6 | 73.3% → 86.7% | 93.3% → 100.0% | +0.0861 |
| **Claude Opus 4.8** | 60.0% → 53.3% | 80.0% → 76.7% | -0.0580 |
| Claude Opus 5 | 70.0% → 66.7% | 86.7% → 76.7% | -0.0425 |
| Claude Sonnet 5 | 53.3% → 86.7% | 86.7% → 86.7% | +0.1849 |
| Gemini 3.1 Pro Preview | 80.0% → 90.0% | 93.3% → 96.7% | +0.0802 |
| Gemini 3.6 Flash | 43.3% → 76.7% | 93.3% → 100.0% | +0.1981 |
| Gemini 3.7 Flash | 73.3% → 26.7% | 96.7% → 93.3% | -0.2514 |
| Gemini 3.8 Flash | 36.7% → 46.7% | 73.3% → 90.0% | +0.0933 |
| **GLM 5.3** | 30.0% → 23.3% | 70.0% → 83.3% | -0.0033 |
| Glm 5.3 Flash | 50.0% → 30.0% | 83.3% → 96.7% | -0.0778 |
| Gpt 5.5 | 50.0% → 76.7% | 83.3% → 90.0% | +0.1675 |
| **GPT-5.6 Luna** | 40.0% → 36.7% | 83.3% → 83.3% | -0.0183 |
| **GPT-5.6 Sol** | 36.7% → 36.7% | 86.7% → 93.3% | +0.0333 |
| Gpt 5.6 Terra | 53.3% → 46.7% | 90.0% → 96.7% | -0.0213 |
| Gpt 6 Astra | 76.7% → 63.3% | 90.0% → 73.3% | -0.1066 |
| Kimi K3 | 36.7% → 26.7% | 56.7% → 60.0% | -0.0454 |
| Qwen3.8 Max | 70.0% → 66.7% | 86.7% → 83.3% | -0.0246 |

## C. Prediction movement

- Corrected top-1 predictions: **81**; regressed: **70**.
- True-label rank moved up: **142**; down: **123**; unchanged: **275**.
- Mean probability of the true label: 52.9% → 54.0% (+1.0 pp).

## D. Conclusions

- Overall top-1 balanced accuracy improved by +2.0 pp.
- Of the five previously weak classes, 1/5 improved top-1 recall (Claude Fable 5).
- Ranking improved independently of top-1 through both top-3 accuracy and MRR.
- Confidence quality improved.
- Prefer the new checkpoint on this frozen validation comparison.

These are development comparisons: the frozen V0 validation set has also been used for checkpoint selection.
