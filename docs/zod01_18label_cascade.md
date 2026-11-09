# ZOD01 Fixed 18-Label Cascade

Fixed-rule cascade evaluated on the frozen V0 validation set (540 responses, 30 prompt groups). Checkpoint order and label assignments were used exactly as specified; no model training, threshold tuning, or checkpoint reselection was performed.

Fallback resolved through the experiment registry: `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff`; parent checkpoint `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2`. Validation split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`.

## Rule results

| Order | Label | Checkpoint | Assigned | Correct | Incorrect | Cumulative coverage | Remaining |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | gemini-3.1-pro-preview | `N0070` | 14 | 14 | 0 | 2.6% | 526 |
| 2 | gpt-6-astra | `N0072` | 24 | 23 | 1 | 7.0% | 502 |
| 3 | claude-opus-4-6 | `N0058` | 28 | 24 | 4 | 12.2% | 474 |
| 4 | claude-opus-5 | `N0082` | 14 | 12 | 2 | 14.8% | 460 |
| 5 | claude-opus-4-8 | `D` | 18 | 15 | 3 | 18.1% | 442 |
| 6 | claude-sonnet-5 | `N0058` | 23 | 19 | 4 | 22.4% | 419 |
| 7 | gemini-3.8-flash | `C` | 15 | 11 | 4 | 25.2% | 404 |
| 8 | qwen3.8-max | `C` | 29 | 21 | 8 | 30.6% | 375 |
| 9 | gpt-5.5 | `N0065` | 19 | 14 | 5 | 34.1% | 356 |
| 10 | gpt-5.6-luna | `N0075` | 14 | 9 | 5 | 36.7% | 342 |
| 11 | gemini-3.6-flash | `N0083` | 19 | 13 | 6 | 40.2% | 323 |
| 12 | glm-5.3-flash | `N0031` | 9 | 6 | 3 | 41.9% | 314 |
| 13 | gpt-5.6-sol | `N0071` | 10 | 6 | 4 | 43.7% | 304 |
| 14 | gemini-3.7-flash | `N0082` | 15 | 10 | 5 | 46.5% | 289 |
| 15 | kimi-k3 | `D` | 11 | 6 | 5 | 48.5% | 278 |
| 16 | gpt-5.6-terra | `N0057` | 10 | 7 | 3 | 50.4% | 268 |
| 17 | claude-fable-5 | `A` | 13 | 8 | 5 | 52.8% | 255 |
| 18 | glm-5.3 | `N0018` | 9 | 4 | 5 | 54.4% | 246 |
| fallback | UltraBERT V1-all -> V0 | `D` | 246 | 109 | 137 | 100.0% | 0 |

## Overall comparison

| Method | Correct | Accuracy |
|---|---:|---:|
| 18-label cascade | 331/540 | 61.30% |
| UltraBERT V1-all → V0 fallback alone | 298/540 | 55.19% |
| Difference | +33 rows | +6.11% |

246 examples reached fallback after all 18 rules; fallback correctly classified 109 of those.

This is a development-set result: the saved checkpoint selection and specialist choices were informed by V0 validation performance.

Detailed per-row decisions and rule trace: `artifacts/analysis/zod01_18label_cascade`.
