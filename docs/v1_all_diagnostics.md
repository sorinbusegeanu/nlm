# V1-all Data Quality and Confusion Analysis

CPU-only diagnostics using saved datasets and saved UltraBERT probabilities. No models were trained or inference run.

## Five key findings

1. **V1.2_bm adds relatively few weak-label rows:** the three labels are 4.0% of BM (1,472 rows) versus 44.2% of V1.1 (32,241 rows). BM is not disproportionately concentrated on them.
2. **V1-all helps GLM Flash most, but hurts Fable precision:** A→B recall changes are Fable +3.3 pp, GLM 5.3 -3.3 pp, and GLM Flash +20.0 pp. Fable precision falls from 47.1% to 23.7% as its false positives rise from 9 to 29.
3. **V1-all→V0 changes the error balance again:** D raises Fable recall to 53.3% and cuts GLM 5.3 / Flash false positives to 18 / 17, but GLM Flash recall falls to 30.0%. On the three true weak classes combined, B→D corrects 12 and regresses 12 (net accuracy change +0.00 pp); overall accuracy rises from 52.2% to 55.2%.
4. **There is source concentration, but not a clear response-format mismatch:** V1.2_bm's largest sources are Fable prvsiyan/prvsiyans-extended-name-that-llm-dataset (96.9%), GLM 5.3 prvsiyan/prvsiyans-extended-name-that-llm-dataset (55.1%), and GLM Flash maniceet/name-that-llm-openrouter-openai-generations (62.8%). Weak-label mean chars differ only modestly from V0 (726–749); weak responses have somewhat higher punctuation density than V0.
5. **No direct duplicate or label contradiction was found:** normalized exact duplicate rows=0, cross-label duplicate groups=0, repeated long templates=0, V0 prompt overlap=0, direct provider conflicts=0. V1.2_bm has 9,413 rows without `origin_hint`; this is missing metadata, not evidence of false attribution.

## 1. Dataset composition

V1.1 contains 72,906 rows; V1.2_bm contains 37,061. Their union has 109,967 rows. The corresponding V1-all artifact ID digest is `ce3e047950324740d7c95f8a58b09c99259b3bf81297978b1f213c86ed39b3d4`.

| Label | V1.1 n (%) | V1.2_bm n (%) | V1-all n (%) |
|---|---:|---:|---:|
| claude-fable-5 | 9,585 (13.15%) | 225 (0.61%) | 9,810 (8.92%) |
| claude-opus-4-6 | 2,711 (3.72%) | 1,512 (4.08%) | 4,223 (3.84%) |
| claude-opus-4-8 | 2,711 (3.72%) | 1,875 (5.06%) | 4,586 (4.17%) |
| claude-opus-5 | 2,711 (3.72%) | 1,192 (3.22%) | 3,903 (3.55%) |
| claude-sonnet-5 | 2,711 (3.72%) | 2,051 (5.53%) | 4,762 (4.33%) |
| gemini-3.1-pro-preview | 2,711 (3.72%) | 1,240 (3.35%) | 3,951 (3.59%) |
| gemini-3.6-flash | 2,711 (3.72%) | 2,703 (7.29%) | 5,414 (4.92%) |
| gemini-3.7-flash | 2,711 (3.72%) | 1,324 (3.57%) | 4,035 (3.67%) |
| gemini-3.8-flash | 2,711 (3.72%) | 3,526 (9.51%) | 6,237 (5.67%) |
| glm-5.3 | 11,328 (15.54%) | 401 (1.08%) | 11,729 (10.67%) |
| glm-5.3-flash | 11,328 (15.54%) | 846 (2.28%) | 12,174 (11.07%) |
| gpt-5.5 | 2,711 (3.72%) | 2,814 (7.59%) | 5,525 (5.02%) |
| gpt-5.6-luna | 2,711 (3.72%) | 4,366 (11.78%) | 7,077 (6.44%) |
| gpt-5.6-sol | 2,711 (3.72%) | 3,823 (10.32%) | 6,534 (5.94%) |
| gpt-5.6-terra | 2,711 (3.72%) | 3,940 (10.63%) | 6,651 (6.05%) |
| gpt-6-astra | 2,711 (3.72%) | 2,298 (6.20%) | 5,009 (4.56%) |
| kimi-k3 | 2,711 (3.72%) | 1,874 (5.06%) | 4,585 (4.17%) |
| qwen3.8-max | 2,711 (3.72%) | 1,051 (2.84%) | 3,762 (3.42%) |

### Source distribution for the three focus labels

| Dataset | Label | Source dataset | Rows | Within-label share |
|---|---|---|---:|---:|
| v1.1 | claude-fable-5 | elliotthankinson/ntl-exact-gens-v3 | 1,702 | 17.8% |
| v1.1 | claude-fable-5 | tadeuhayashida/ntl-claude-fable-5-generations | 1,609 | 16.8% |
| v1.1 | claude-fable-5 | drjcfmd/ntl-generated-v2 | 1,295 | 13.5% |
| v1.1 | claude-fable-5 | arxitekton/ntl-claude-gen-v1 | 1,014 | 10.6% |
| v1.1 | claude-fable-5 | abdullah1afify/ntl-generations | 988 | 10.3% |
| v1.1 | claude-fable-5 | arxitekton/ntl-claude-gen-v2 | 934 | 9.7% |
| v1.1 | claude-fable-5 | michaelsemenoff/ntl-pool-v3 | 665 | 6.9% |
| v1.1 | claude-fable-5 | seyitkaangunes/ntl-extra-api-passages | 444 | 4.6% |
| v1.1 | claude-fable-5 | vbandaru/ntl-extra-passages-w05 | 353 | 3.7% |
| v1.1 | claude-fable-5 | lxdlxdlxd/ntl-synth-generations-v1 | 319 | 3.3% |
| v1.1 | claude-fable-5 | michaelsemenoff/ntl-own-extra-v1 | 178 | 1.9% |
| v1.1 | claude-fable-5 | bingli283/ntl-gen-pilot-v1 | 48 | 0.5% |
| v1.1 | claude-fable-5 | illiadolenko/ntl-generated-corpus-v1 | 19 | 0.2% |
| v1.1 | claude-fable-5 | earthmaker/ntl-own-prompt-generations | 17 | 0.2% |
| v1.1 | glm-5.3 | elliotthankinson/ntl-exact-gens-v3 | 2,202 | 19.4% |
| v1.1 | glm-5.3 | trnhip/ntl-api-gen | 2,018 | 17.8% |
| v1.1 | glm-5.3 | drjcfmd/ntl-generated-v2 | 1,241 | 11.0% |
| v1.1 | glm-5.3 | arxitekton/ntl-claude-gen-v1 | 1,011 | 8.9% |
| v1.1 | glm-5.3 | willrice/name-that-llm-openrouter-harvest | 966 | 8.5% |
| v1.1 | glm-5.3 | abdullah1afify/ntl-generations | 961 | 8.5% |
| v1.1 | glm-5.3 | kenandemirci/ntl-glm-gemini-generations | 774 | 6.8% |
| v1.1 | glm-5.3 | michaelsemenoff/ntl-pool-v3 | 662 | 5.8% |
| v1.1 | glm-5.3 | danielandresmiranda/ntl-openrouter-glm-kimi | 577 | 5.1% |
| v1.1 | glm-5.3 | lxdlxdlxd/ntl-synth-generations-v1 | 392 | 3.5% |
| v1.1 | glm-5.3 | michaelsemenoff/ntl-own-extra-v1 | 289 | 2.6% |
| v1.1 | glm-5.3 | hansamaqht/name-that-llm-glm-extra | 220 | 1.9% |
| v1.1 | glm-5.3 | earthmaker/ntl-own-prompt-generations | 15 | 0.1% |
| v1.1 | glm-5.3-flash | elliotthankinson/ntl-exact-gens-v3 | 2,047 | 18.1% |
| v1.1 | glm-5.3-flash | trnhip/ntl-api-gen | 1,992 | 17.6% |
| v1.1 | glm-5.3-flash | drjcfmd/ntl-generated-v2 | 1,237 | 10.9% |
| v1.1 | glm-5.3-flash | arxitekton/ntl-claude-gen-v1 | 1,010 | 8.9% |
| v1.1 | glm-5.3-flash | willrice/name-that-llm-openrouter-harvest | 988 | 8.7% |
| v1.1 | glm-5.3-flash | kenandemirci/ntl-glm-gemini-generations | 969 | 8.6% |
| v1.1 | glm-5.3-flash | abdullah1afify/ntl-generations | 951 | 8.4% |
| v1.1 | glm-5.3-flash | michaelsemenoff/ntl-pool-v3 | 665 | 5.9% |
| v1.1 | glm-5.3-flash | danielandresmiranda/ntl-openrouter-glm-kimi | 566 | 5.0% |
| v1.1 | glm-5.3-flash | lxdlxdlxd/ntl-synth-generations-v1 | 385 | 3.4% |
| v1.1 | glm-5.3-flash | michaelsemenoff/ntl-own-extra-v1 | 288 | 2.5% |
| v1.1 | glm-5.3-flash | hansamaqht/name-that-llm-glm-extra | 214 | 1.9% |
| v1.1 | glm-5.3-flash | earthmaker/ntl-own-prompt-generations | 16 | 0.1% |
| v1.2_bm | claude-fable-5 | prvsiyan/prvsiyans-extended-name-that-llm-dataset | 218 | 96.9% |
| v1.2_bm | claude-fable-5 | seyitkaangunes/ntl-extra-api-passages | 6 | 2.7% |
| v1.2_bm | claude-fable-5 | lxdlxdlxd/ntl-synth-generations-v1 | 1 | 0.4% |
| v1.2_bm | glm-5.3 | prvsiyan/prvsiyans-extended-name-that-llm-dataset | 221 | 55.1% |
| v1.2_bm | glm-5.3 | maniceet/name-that-llm-openrouter-openai-generations | 166 | 41.4% |
| v1.2_bm | glm-5.3 | willrice/name-that-llm-openrouter-harvest | 5 | 1.2% |
| v1.2_bm | glm-5.3 | arxitekton/ntl-claude-gen-v1 | 4 | 1.0% |
| v1.2_bm | glm-5.3 | michaelsemenoff/ntl-own-extra-v1 | 3 | 0.7% |
| v1.2_bm | glm-5.3 | lxdlxdlxd/ntl-synth-generations-v1 | 2 | 0.5% |
| v1.2_bm | glm-5.3-flash | maniceet/name-that-llm-openrouter-openai-generations | 531 | 62.8% |
| v1.2_bm | glm-5.3-flash | prvsiyan/prvsiyans-extended-name-that-llm-dataset | 223 | 26.4% |
| v1.2_bm | glm-5.3-flash | arxitekton/ntl-claude-gen-v1 | 43 | 5.1% |
| v1.2_bm | glm-5.3-flash | willrice/name-that-llm-openrouter-harvest | 25 | 3.0% |
| v1.2_bm | glm-5.3-flash | lxdlxdlxd/ntl-synth-generations-v1 | 15 | 1.8% |
| v1.2_bm | glm-5.3-flash | michaelsemenoff/ntl-own-extra-v1 | 9 | 1.1% |

### Response characteristics

Text length, word/punctuation counts and formatting indicators use all rows. Existing project stylometry features use a deterministic sample of up to 250 rows per dataset and class (all V0 validation rows are included). Means below are class-balanced within each cohort; the complete per-label distributions and row-pooled estimates are in `response_characteristics_by_class.csv` and `response_characteristics_summary.csv`.

| Measure | V0 weak | V1.1 weak | V1.2_bm weak | V0 other 15 | V1.1 other 15 | V1.2_bm other 15 |
|---|---:|---:|---:|---:|---:|---:|
| mean chars | 749.3 | 720.9 | 726.4 | 743.3 | 741.2 | 733.6 |
| mean words | 119.1 | 113.9 | 119.0 | 115.0 | 114.3 | 114.5 |
| punctuation / 1k chars | 27.9 | 29.6 | 30.3 | 25.1 | 27.0 | 27.0 |
| sentence marks / 1k chars | 7.3 | 7.5 | 7.5 | 7.0 | 7.6 | 7.5 |
| Markdown fraction | 0.00% | 0.03% | 0.04% | 0.00% | 0.05% | 0.04% |
| Code-fence fraction | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| URL fraction | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| stylometry: chars/word | 6.3 | 6.2 | 6.0 | 6.5 | 6.4 | 6.4 |
| stylometry: words/sentence | 23.6 | 24.0 | 24.5 | 23.3 | 22.5 | 22.8 |
| stylometry: uppercase fraction | 1.02% | 1.15% | 1.10% | 0.94% | 1.17% | 1.16% |
| stylometry: digit fraction | 0.13% | 0.21% | 0.27% | 0.04% | 0.18% | 0.21% |

## 2. Provenance, duplication, and split checks

| Dataset | Check | Rows | Percent |
|---|---|---:|---:|
| v1.1 | missing_source_dataset (all) | 0 | 0.00% |
| v1.1 | missing_source_file (all) | 1,953 | 2.68% |
| v1.1 | missing_source_license (all) | 0 | 0.00% |
| v1.1 | missing_origin_hint (all) | 58,871 | 80.75% |
| v1.1 | unavailable_provider_in_schema (all) | 72,906 | 100.00% |
| v1.1 | unavailable_model_version_in_schema (all) | 72,906 | 100.00% |
| v1.1 | unavailable_route_in_schema (all) | 72,906 | 100.00% |
| v1.1 | direct_model_version_label_conflict (all) | 0 | 0.00% |
| v1.1 | missing_source_dataset (weak_three) | 0 | 0.00% |
| v1.1 | missing_source_file (weak_three) | 434 | 1.35% |
| v1.1 | missing_source_license (weak_three) | 0 | 0.00% |
| v1.1 | missing_origin_hint (weak_three) | 25,673 | 79.63% |
| v1.1 | unavailable_provider_in_schema (weak_three) | 32,241 | 100.00% |
| v1.1 | unavailable_model_version_in_schema (weak_three) | 32,241 | 100.00% |
| v1.1 | unavailable_route_in_schema (weak_three) | 32,241 | 100.00% |
| v1.1 | direct_model_version_label_conflict (weak_three) | 0 | 0.00% |
| v1.2_bm | missing_source_dataset (all) | 0 | 0.00% |
| v1.2_bm | missing_source_file (all) | 0 | 0.00% |
| v1.2_bm | missing_source_license (all) | 0 | 0.00% |
| v1.2_bm | missing_origin_hint (all) | 9,413 | 25.40% |
| v1.2_bm | missing_provider (all) | 9,413 | 25.40% |
| v1.2_bm | missing_model_version (all) | 36,528 | 98.56% |
| v1.2_bm | missing_route (all) | 9,413 | 25.40% |
| v1.2_bm | direct_model_version_label_conflict (all) | 0 | 0.00% |
| v1.2_bm | missing_source_dataset (weak_three) | 0 | 0.00% |
| v1.2_bm | missing_source_file (weak_three) | 0 | 0.00% |
| v1.2_bm | missing_source_license (weak_three) | 0 | 0.00% |
| v1.2_bm | missing_origin_hint (weak_three) | 30 | 2.04% |
| v1.2_bm | missing_provider (weak_three) | 30 | 2.04% |
| v1.2_bm | missing_model_version (weak_three) | 1,472 | 100.00% |
| v1.2_bm | missing_route (weak_three) | 30 | 2.04% |
| v1.2_bm | direct_model_version_label_conflict (weak_three) | 0 | 0.00% |
| v1.1 | provider_label_conflict_observed (all) | 0 | 0.00% |
| v1.2_bm | provider_label_conflict_observed (all) | 0 | 0.00% |

The deterministic 3,000-row word TF-IDF nearest-neighbor sample had p90 cosine 0.423, max 0.692, and 0 rows with nearest cosine ≥0.90. This is a sampled screen, not proof of absence. Full normalized-text duplicate rows: 0; exact-text groups spanning labels: 0. Repeated normalized templates (≥3 rows, ≥100 chars): 0.

| Dataset | Training rows sharing V0 validation prompt IDs | Prompt groups |
|---|---:|---:|
| v1.1 | 0 | 0 |
| v1.2_bm | 0 | 0 |

Missing optional provenance fields are reported as missing metadata, not by themselves as proof that attribution is false. Provider/model-version conflicts count only when the stored fields directly contradict the canonical class mapping.

## 3. Checkpoint confusion

Full 18×18 matrices are saved as `confusion_A.csv`, `confusion_B.csv`, and `confusion_D.csv` (rows=true label, columns=predicted label). Class metrics for all 18 labels are in `per_label_metrics.csv`.

Complete per-label precision, recall, FP and FN are shown below and exported to `per_label_metrics.csv`.

| Checkpoint | Label | Precision | Recall | FP | FN |
|---|---|---:|---:|---:|---:|
| A | claude-fable-5 | 47.1% | 26.7% | 9 | 22 |
| A | claude-opus-4-6 | 75.9% | 73.3% | 7 | 8 |
| A | claude-opus-4-8 | 57.7% | 50.0% | 11 | 15 |
| A | claude-opus-5 | 76.0% | 63.3% | 6 | 11 |
| A | claude-sonnet-5 | 58.1% | 60.0% | 13 | 12 |
| A | gemini-3.1-pro-preview | 91.7% | 73.3% | 2 | 8 |
| A | gemini-3.6-flash | 50.0% | 46.7% | 14 | 16 |
| A | gemini-3.7-flash | 35.9% | 46.7% | 25 | 16 |
| A | gemini-3.8-flash | 58.6% | 56.7% | 12 | 13 |
| A | glm-5.3 | 24.2% | 26.7% | 25 | 22 |
| A | glm-5.3-flash | 25.0% | 33.3% | 30 | 20 |
| A | gpt-5.5 | 50.0% | 40.0% | 12 | 18 |
| A | gpt-5.6-luna | 39.3% | 36.7% | 17 | 19 |
| A | gpt-5.6-sol | 45.5% | 50.0% | 18 | 15 |
| A | gpt-5.6-terra | 44.4% | 53.3% | 20 | 14 |
| A | gpt-6-astra | 92.0% | 76.7% | 2 | 7 |
| A | kimi-k3 | 30.2% | 43.3% | 30 | 17 |
| A | qwen3.8-max | 63.3% | 63.3% | 11 | 11 |
| B | claude-fable-5 | 23.7% | 30.0% | 29 | 21 |
| B | claude-opus-4-6 | 85.0% | 56.7% | 3 | 13 |
| B | claude-opus-4-8 | 56.8% | 70.0% | 16 | 9 |
| B | claude-opus-5 | 71.4% | 66.7% | 8 | 10 |
| B | claude-sonnet-5 | 76.2% | 53.3% | 5 | 14 |
| B | gemini-3.1-pro-preview | 88.0% | 73.3% | 3 | 8 |
| B | gemini-3.6-flash | 51.2% | 70.0% | 20 | 9 |
| B | gemini-3.7-flash | 47.6% | 33.3% | 11 | 20 |
| B | gemini-3.8-flash | 55.9% | 63.3% | 15 | 11 |
| B | glm-5.3 | 24.1% | 23.3% | 22 | 23 |
| B | glm-5.3-flash | 32.7% | 53.3% | 33 | 14 |
| B | gpt-5.5 | 52.9% | 30.0% | 8 | 21 |
| B | gpt-5.6-luna | 33.3% | 16.7% | 10 | 25 |
| B | gpt-5.6-sol | 40.0% | 46.7% | 21 | 16 |
| B | gpt-5.6-terra | 41.7% | 66.7% | 28 | 10 |
| B | gpt-6-astra | 95.5% | 70.0% | 1 | 9 |
| B | kimi-k3 | 53.3% | 53.3% | 14 | 14 |
| B | qwen3.8-max | 63.3% | 63.3% | 11 | 11 |
| D | claude-fable-5 | 43.2% | 53.3% | 21 | 14 |
| D | claude-opus-4-6 | 72.2% | 86.7% | 10 | 4 |
| D | claude-opus-4-8 | 84.2% | 53.3% | 3 | 14 |
| D | claude-opus-5 | 76.9% | 66.7% | 6 | 10 |
| D | claude-sonnet-5 | 59.1% | 86.7% | 18 | 4 |
| D | gemini-3.1-pro-preview | 77.1% | 90.0% | 8 | 3 |
| D | gemini-3.6-flash | 53.5% | 76.7% | 20 | 7 |
| D | gemini-3.7-flash | 44.4% | 26.7% | 10 | 22 |
| D | gemini-3.8-flash | 58.3% | 46.7% | 10 | 16 |
| D | glm-5.3 | 28.0% | 23.3% | 18 | 23 |
| D | glm-5.3-flash | 34.6% | 30.0% | 17 | 21 |
| D | gpt-5.5 | 41.8% | 76.7% | 32 | 7 |
| D | gpt-5.6-luna | 34.4% | 36.7% | 21 | 19 |
| D | gpt-5.6-sol | 55.0% | 36.7% | 9 | 19 |
| D | gpt-5.6-terra | 43.8% | 46.7% | 18 | 16 |
| D | gpt-6-astra | 86.4% | 63.3% | 3 | 11 |
| D | kimi-k3 | 57.1% | 26.7% | 6 | 22 |
| D | qwen3.8-max | 62.5% | 66.7% | 12 | 10 |

### Weak-label confusion directions

| Checkpoint | Error direction | Weak label | Confused label | Rows |
|---|---|---|---|---:|
| A | weak_true_to_other_prediction | claude-fable-5 | glm-5.3-flash | 7 |
| A | other_true_to_weak_prediction | claude-fable-5 | glm-5.3 | 5 |
| A | weak_true_to_other_prediction | claude-fable-5 | glm-5.3 | 5 |
| A | weak_true_to_other_prediction | claude-fable-5 | claude-sonnet-5 | 3 |
| A | weak_true_to_other_prediction | claude-fable-5 | kimi-k3 | 2 |
| A | weak_true_to_other_prediction | glm-5.3 | glm-5.3-flash | 12 |
| A | other_true_to_weak_prediction | glm-5.3 | glm-5.3-flash | 11 |
| A | weak_true_to_other_prediction | glm-5.3 | claude-fable-5 | 5 |
| A | other_true_to_weak_prediction | glm-5.3 | claude-fable-5 | 5 |
| A | other_true_to_weak_prediction | glm-5.3 | claude-opus-5 | 4 |
| A | other_true_to_weak_prediction | glm-5.3-flash | glm-5.3 | 12 |
| A | weak_true_to_other_prediction | glm-5.3-flash | glm-5.3 | 11 |
| A | other_true_to_weak_prediction | glm-5.3-flash | claude-fable-5 | 7 |
| A | other_true_to_weak_prediction | glm-5.3-flash | claude-opus-5 | 4 |
| A | weak_true_to_other_prediction | glm-5.3-flash | kimi-k3 | 4 |
| B | weak_true_to_other_prediction | claude-fable-5 | glm-5.3 | 9 |
| B | other_true_to_weak_prediction | claude-fable-5 | claude-sonnet-5 | 8 |
| B | other_true_to_weak_prediction | claude-fable-5 | glm-5.3 | 7 |
| B | weak_true_to_other_prediction | claude-fable-5 | glm-5.3-flash | 6 |
| B | other_true_to_weak_prediction | claude-fable-5 | claude-opus-4-8 | 4 |
| B | weak_true_to_other_prediction | glm-5.3 | glm-5.3-flash | 13 |
| B | other_true_to_weak_prediction | glm-5.3 | claude-fable-5 | 9 |
| B | other_true_to_weak_prediction | glm-5.3 | glm-5.3-flash | 8 |
| B | weak_true_to_other_prediction | glm-5.3 | claude-fable-5 | 7 |
| B | weak_true_to_other_prediction | glm-5.3 | claude-opus-5 | 2 |
| B | other_true_to_weak_prediction | glm-5.3-flash | glm-5.3 | 13 |
| B | weak_true_to_other_prediction | glm-5.3-flash | glm-5.3 | 8 |
| B | other_true_to_weak_prediction | glm-5.3-flash | claude-fable-5 | 6 |
| B | other_true_to_weak_prediction | glm-5.3-flash | claude-opus-5 | 5 |
| B | weak_true_to_other_prediction | glm-5.3-flash | claude-fable-5 | 4 |
| D | other_true_to_weak_prediction | claude-fable-5 | glm-5.3 | 10 |
| D | other_true_to_weak_prediction | claude-fable-5 | glm-5.3-flash | 6 |
| D | weak_true_to_other_prediction | claude-fable-5 | claude-opus-4-6 | 4 |
| D | weak_true_to_other_prediction | claude-fable-5 | glm-5.3-flash | 3 |
| D | weak_true_to_other_prediction | claude-fable-5 | glm-5.3 | 3 |
| D | weak_true_to_other_prediction | glm-5.3 | claude-fable-5 | 10 |
| D | other_true_to_weak_prediction | glm-5.3 | glm-5.3-flash | 9 |
| D | weak_true_to_other_prediction | glm-5.3 | glm-5.3-flash | 6 |
| D | weak_true_to_other_prediction | glm-5.3 | claude-opus-5 | 4 |
| D | other_true_to_weak_prediction | glm-5.3 | claude-opus-5 | 4 |
| D | weak_true_to_other_prediction | glm-5.3-flash | glm-5.3 | 9 |
| D | weak_true_to_other_prediction | glm-5.3-flash | claude-fable-5 | 6 |
| D | other_true_to_weak_prediction | glm-5.3-flash | glm-5.3 | 6 |
| D | weak_true_to_other_prediction | glm-5.3-flash | claude-sonnet-5 | 4 |
| D | other_true_to_weak_prediction | glm-5.3-flash | claude-fable-5 | 3 |

### Prediction changes on identical V0 rows

| Transition | Scope | Changed | Corrected | Regressed | Accuracy change |
|---|---|---:|---:|---:|---:|
| A_to_B | all | 212 | 71 | 65 | +1.11 pp |
| A_to_B | weak_true | 56 | 21 | 15 | +6.67 pp |
| A_to_B | weak_involved | 87 | 22 | 20 | +1.55 pp |
| B_to_D | all | 200 | 78 | 62 | +2.96 pp |
| B_to_D | weak_true | 36 | 12 | 12 | +0.00 pp |
| B_to_D | weak_involved | 63 | 23 | 13 | +7.81 pp |

Selected changed examples involving a focus label:

- `A_to_B` regressed: `0ebadec0ddf4` true `glm-5.3`, `glm-5.3` → `claude-fable-5` — Network effects work because connections multiply faster than members: add one person to a network of two and you triple the possible links. Each newcomer enriches everyone already present, so value doesn't grow linearly—it compounds explosively, turning empty platforms into indispensable infrastructure that no one can…
- `A_to_B` regressed: `1649aa8dd59d` true `claude-sonnet-5`, `claude-sonnet-5` → `glm-5.3-flash` — The Collatz conjecture seduces with its childlike simplicity: take any positive integer, and if it's even, halve it; if it's odd, triple it and add one. Repeat forever, or so it seems, and watch the numbers tumble like hailstones through the sky—rising unpredictably, plummeting suddenly, yet always, in every case ever …
- `A_to_B` regressed: `168d8f48b33f` true `claude-fable-5`, `claude-fable-5` → `glm-5.3-flash` — In 1936, Alan Turing proved that no algorithm can ever exist that determines, for every possible program and input, whether that program will eventually halt or run forever. His argument is elegantly self-referential: suppose such a master "halting checker" existed. Turing showed we could then build a mischievous progr…
- `A_to_B` corrected: `1f317e09e595` true `glm-5.3`, `glm-5.3-flash` → `glm-5.3` — The torch sputters, and the darkness exhales. One step, two, and the flame's light wavers across wet stone, catching the swell of a wall that suddenly breathes with motion. There—an auroch, massive and black-maned, charges across the rock face, its horns flung wide, hooves that will never touch earth suspended mid-gall…
- `B_to_D` corrected: `030145156daf` true `claude-opus-5`, `glm-5.3` → `claude-opus-5` — In the solar system's violent infancy, Jupiter came rolling inward, a thousand Earths of hydrogen dragged sunward by the hissing gas of the young disk. Worlds half-formed stood in its path—rocky embryos gathering mass where Mercury and Venus would someday orbit—and it scattered them like pins, flinging some into the Su…
- `B_to_D` corrected: `05fd45356e2b` true `claude-fable-5`, `glm-5.3` → `claude-fable-5` — Look again at that pale point of light, and consider how small the stage on which we act out our grandeur. Every empire raised and razed, every border drawn in blood, every tyrant convinced of his own permanence—all of it unfolded on a mote of dust adrift in a shaft of sunlight. From this distance, our hatreds seem abs…
- `B_to_D` corrected: `0e7003f003f1` true `claude-sonnet-5`, `claude-fable-5` → `claude-sonnet-5` — Searle's Chinese Room argument exposes a gap between syntax and semantics that symbol manipulation alone cannot bridge. A person inside the room, following rules to match Chinese characters without understanding a word of Chinese, produces intelligent-seeming responses while remaining ignorant of meaning. This shows th…
- `B_to_D` corrected: `145abfe0654d` true `glm-5.3`, `glm-5.3-flash` → `glm-5.3` — In 1936, Alan Turing asked a deceptively simple question: given any program and its input, can we determine whether it will eventually stop or run forever? His answer was no. Suppose a master algorithm H existed that always decided this. Turing showed we could then build a mischievous program that uses H as a subroutin…

## 4. Explicit assessment

1. **Does V1.2_bm disproportionately contribute examples to the three weak labels?** No. They contribute 1,472 rows (4.0% of BM); compare V1.1's 32,241 (44.2%). See class counts above.
2. **Is there evidence of source/domain mismatch or data-quality problems?** There is evidence of source concentration and a modest style shift, not a demonstrated causal mismatch: 96.9% of BM Fable comes from one source, and the largest GLM sources differ by label. Weak-label response lengths are near V0, while punctuation is somewhat denser. No direct provider/model-version contradiction, exact duplicate conflict, or V0 prompt overlap was detected. Missing optional metadata remains an audit limitation: BM lacks `origin_hint`/provider/route on 25.4% overall and `model_version` on 98.6%; V1.1 has no provider/model-version fields in its schema.
3. **Are the weak labels overpredicted, underpredicted, or confused with specific related models?** In A, Fable is underpredicted (17 predictions for 30 examples); GLM 5.3 is slightly overpredicted (33), and Flash is overpredicted (40), with low precision. B overpredicts Fable (38) and Flash (49), while GLM 5.3 is slightly underpredicted (29). D still overpredicts Fable (37) and underpredicts both GLM labels (25 and 26). GLM 5.3 and Flash are strongly confused with each other in all checkpoints; the directed counts are tabulated above.
4. **Did adding V1-all data improve or worsen recognition?** A→B improves overall accuracy by 1.11 pp and accuracy on the three true weak labels by 6.67 pp (21 corrected, 15 regressed). It improves Flash recall by 20.0 pp and Fable recall by 3.3 pp, but worsens GLM 5.3 recall by 3.3 pp and Fable precision by 23.4 pp. B→D is a separate continuation: overall accuracy rises another 2.96 pp, while weak-label accuracy is unchanged (12 corrected, 12 regressed); it improves Fable and GLM 5.3 precision/recall tradeoffs but reduces Flash recall.
5. **Which three targeted changes should be tested next?** (a) Run a V1.1 vs V1.1+BM source-stratified ablation for the weak labels, keeping V0 prompts excluded and logging exact source mix. (b) Test pair-focused sampling or class weighting for GLM 5.3 vs GLM 5.3 Flash, because they exchange many errors in both directions. (c) Audit/hold out the dominant BM sources for Fable and GLM before adding rows, then compare precision and recall on the unchanged V0 rows; preserve the current checkpoint and recipe as controls.

## Reproducibility and artifact index

Run with `python -m src.v1_all_diagnostics`. All generated tables are under `artifacts/analysis/v1_all_diagnostics/`. Prediction files are aligned with the repository's strict V0 ID/prompt/label alignment utility. A/B/D resolve through `checkpoint_inventory.csv`.

The three checkpoints are A=V1, B=V1-all, and D=V1-all→V0. V1.2_bm is training-only. The official V0 validation set is used only as the evaluation reference.
