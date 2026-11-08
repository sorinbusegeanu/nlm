# ZOD01 Specialist Discovery — Existing NLM Checkpoints

## Scope and validation

Evaluated 89 distinct checkpoint runs (85 requested non-UltraBERT runs plus UltraBERT A–D) on the frozen V0 validation set: 540 rows, 30 prompt groups and 18 labels. Existing aligned probability caches were reused. Nonmatching cached validations were regenerated on frozen V0 from their saved checkpoints. No training was run.

All checkpoints have previously been compared or selected using validation data, and this specialist ranking selects rules on those same 540 labels. Treat precision and cascade scores as development diagnostics, not independent generalization estimates. Alternate splits of the V0 source that trained on frozen V0 validation rows remain inventoried but are excluded from ranked rules and cascade evaluation.

Rule rank uses Beta(1,1) smoothed precision `(TP+1)/(prediction_count+2)` and minimum support 10. The threshold counts below use raw observed precision with the same minimum support. Split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`; validation ID SHA-256: `d3f39a2003dfdd9e99e5d662465066bd60368f8e43864b6ed10a733edbe74f9d`.

## Checkpoint inventory

The checkpoint key is used in the global ranking; the exact experiment ID and checkpoint basename/path are in the inventory CSV. Accuracy is recomputed from aligned V0 probabilities. Epoch history and selected CatBoost iteration are recorded in the inventory CSV. The overlap column counts frozen V0 validation rows used in training.

| Key | Model | Training data | V0 accuracy | Selected epoch | Checkpoint present | V0 train overlap | Prediction source | Experiment ID |
|---|---|---|---:|---:|---|---:|---|---|
| A | UltraBERT-380M | v1 | 51.1% | — | yes | 0 | cached | `ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0` |
| B | UltraBERT-380M | v1_all | 52.2% | 2 | yes | 0 | cached | `ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2` |
| C | UltraBERT-380M | v1.5 | 53.1% | 5 | yes | 0 | cached | `ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0` |
| D | UltraBERT-380M | v0 | 55.2% | 2 | yes | 0 | cached | `ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff` |
| N0001 | CatBoost + Stylometry | catboost_a5_s2027 | 71.7% | — | yes | 342 | reused_generated_v0_cache | `catboost_stylometry_catboost_a5_s2027_data40a22ce8_split783f3a95_seed2027_cfgfa54eec9_code667e2d0` |
| N0002 | CatBoost + Stylometry | catboost_a5_s2028 | 54.4% | — | yes | 432 | reused_generated_v0_cache | `catboost_stylometry_catboost_a5_s2028_data40a22ce8_split3fbb4529_seed2028_cfgfa54eec9_code667e2d0` |
| N0003 | CatBoost + Stylometry | catboost_a5_s2029 | 59.6% | — | yes | 468 | reused_generated_v0_cache | `catboost_stylometry_catboost_a5_s2029_data40a22ce8_splitd83f8692_seed2029_cfgfa54eec9_code667e2d0` |
| N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | 30.6% | — | yes | 0 | reused_generated_v0_cache | `catboost_stylometry_catboost_v1_a5_s2027_datada64f007_split94737c59_seed2027_cfg263b702b_code667e2d0` |
| N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | 25.2% | — | yes | 0 | reused_generated_v0_cache | `catboost_stylometry_catboost_v1_a5_s2027_datada64f007_split94737c59_seed2027_cfgfa54eec9_code667e2d0` |
| N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | 31.7% | — | yes | 0 | reused_generated_v0_cache | `catboost_stylometry_catboost_v1_a5_s2028_datada64f007_split94737c59_seed2028_cfg263b702b_code667e2d0` |
| N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | 26.1% | — | yes | 0 | reused_generated_v0_cache | `catboost_stylometry_catboost_v1_a5_s2028_datada64f007_split94737c59_seed2028_cfgfa54eec9_code667e2d0` |
| N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | 31.7% | — | yes | 0 | reused_generated_v0_cache | `catboost_stylometry_catboost_v1_a5_s2029_datada64f007_split94737c59_seed2029_cfg263b702b_code667e2d0` |
| N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | 26.9% | — | yes | 0 | reused_generated_v0_cache | `catboost_stylometry_catboost_v1_a5_s2029_datada64f007_split94737c59_seed2029_cfgfa54eec9_code667e2d0` |
| N0010 | CatBoost + Stylometry | v0 | 29.4% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg1c3179e5_code667e2d0` |
| N0011 | CatBoost + Stylometry | v0 | 28.3% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg3519c09b_code667e2d0` |
| N0012 | CatBoost + Stylometry | v0 | 30.6% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg4baf60a7_code667e2d0` |
| N0013 | CatBoost + Stylometry | v0 | 30.6% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg52ca4114_code667e2d0` |
| N0014 | CatBoost + Stylometry | v0 | 27.6% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg60be5aa3_code667e2d0` |
| N0015 | CatBoost + Stylometry | v0 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg6a36110b_code667e2d0` |
| N0016 | CatBoost + Stylometry | v0 | 18.5% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg70ca5a1e_code667e2d0` |
| N0017 | CatBoost + Stylometry | v0 | 30.0% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg9973a364_code667e2d0` |
| N0018 | CatBoost + Stylometry | v0 | 25.9% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg9c8bace5_code667e2d0` |
| N0019 | CatBoost + Stylometry | v0 | 30.4% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfg9f0feebc_code667e2d0` |
| N0020 | CatBoost + Stylometry | v0 | 29.4% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgcbd2a6ae_code667e2d0` |
| N0021 | CatBoost + Stylometry | v0 | 30.6% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgcf0d09ab_code667e2d0` |
| N0022 | CatBoost + Stylometry | v0 | 29.8% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgd553b077_code667e2d0` |
| N0023 | CatBoost + Stylometry | v0 | 30.6% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgde52ae75_code667e2d0` |
| N0024 | CatBoost + Stylometry | v0 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfge31fa01b_code667e2d0` |
| N0025 | CatBoost + Stylometry | v0 | 28.3% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfge6177d6b_code667e2d0` |
| N0026 | CatBoost + Stylometry | v0 | 30.6% | — | yes | 0 | cached | `catboost_stylometry_v0_data40a22ce8_split147ce64e_seed2026_cfgfa54eec9_code667e2d0` |
| N0027 | CatBoost + Stylometry | v0 | 34.4% | — | yes | 0 | cached | `catboost_two_stage_20261108T0850Z_C_stage2` |
| N0028 | CatBoost + Stylometry | v1 | 33.0% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg11886585_code667e2d0` |
| N0029 | CatBoost + Stylometry | v1 | 28.1% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg1c3179e5_code667e2d0` |
| N0030 | CatBoost + Stylometry | v1 | 33.5% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg263b702b_code667e2d0` |
| N0031 | CatBoost + Stylometry | v1 | 30.6% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg3519c09b_code667e2d0` |
| N0032 | CatBoost + Stylometry | v1 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg52ca4114_code667e2d0` |
| N0033 | CatBoost + Stylometry | v1 | 25.6% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg60be5aa3_code667e2d0` |
| N0034 | CatBoost + Stylometry | v1 | 29.3% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg6a36110b_code667e2d0` |
| N0035 | CatBoost + Stylometry | v1 | 17.0% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg70ca5a1e_code667e2d0` |
| N0036 | CatBoost + Stylometry | v1 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg9973a364_code667e2d0` |
| N0037 | CatBoost + Stylometry | v1 | 27.4% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg9c8bace5_code667e2d0` |
| N0038 | CatBoost + Stylometry | v1 | 28.0% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfg9f0feebc_code667e2d0` |
| N0039 | CatBoost + Stylometry | v1 | 29.6% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgb69e0b3a_code667e2d0` |
| N0040 | CatBoost + Stylometry | v1 | 29.4% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgcbd2a6ae_code667e2d0` |
| N0041 | CatBoost + Stylometry | v1 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgcf0d09ab_code667e2d0` |
| N0042 | CatBoost + Stylometry | v1 | 30.2% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgd553b077_code667e2d0` |
| N0043 | CatBoost + Stylometry | v1 | 34.4% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgdb877fc5_code667e2d0` |
| N0044 | CatBoost + Stylometry | v1 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgde52ae75_code667e2d0` |
| N0045 | CatBoost + Stylometry | v1 | 26.9% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfge31fa01b_code667e2d0` |
| N0046 | CatBoost + Stylometry | v1 | 28.7% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfge6177d6b_code667e2d0` |
| N0047 | CatBoost + Stylometry | v1 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v1_datada64f007_split147ce64e_seed2026_cfgfa54eec9_code667e2d0` |
| N0048 | CatBoost + Stylometry | v1.1 | 27.6% | — | yes | 0 | cached | `catboost_stylometry_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg587e4f91_code667e2d0` |
| N0049 | CatBoost + Stylometry | v1.5 | 29.1% | — | yes | 0 | cached | `catboost_stylometry_v1.5_data490fe052_split215fd0d0_seed2026_cfg2f0c971b_code1b9bfc2` |
| N0050 | CatBoost + Stylometry | v1.5 | 34.4% | — | yes | 0 | cached | `catboost_stylometry_v1.5_data490fe052_split215fd0d0_seed2026_cfgdb877fc5_code1b9bfc2` |
| N0051 | EmbeddingGemma 2 | v0 | 22.2% | — | yes | 0 | cached | `embeddinggemma2_v0_continue10_from2304cd86_seed2026_cfg78bfcba7_code667e2d0` |
| N0052 | EmbeddingGemma 2 | v0 | 22.8% | — | yes | 0 | cached | `embeddinggemma2_v0_continue10_fromeab947d3_seed2026_cfg2ba833a7_code667e2d0` |
| N0053 | EmbeddingGemma 2 | v0 | 19.8% | 10 | yes | 0 | cached | `embeddinggemma2_v0_data40a22ce8_split147ce64e_seed2026_cfgdfb26ed6_code667e2d0` |
| N0054 | EmbeddingGemma 2 | v1 | 40.9% | 6 | yes | 0 | cached | `embeddinggemma2_v1_datada64f007_split147ce64e_seed2026_cfgdfb26ed6_code667e2d0` |
| N0055 | EmbeddingGemma 2 | v1.1 | 33.1% | 3 | yes | 0 | cached | `embeddinggemma2_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg576f1dd6_code667e2d0` |
| N0056 | EmbeddingGemma 2 | v1.1 | 33.9% | 13 | yes | 0 | cached | `embeddinggemma2_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbbaa9462_code667e2d0` |
| N0057 | EmbeddingGemma 2 | v1.5 | 41.9% | 8 | yes | 0 | cached | `embeddinggemma2_v1.5_data490fe052_split215fd0d0_seed2026_cfgd46e5fd0_code667e2d0` |
| N0058 | Gemma 4 E2B | v1.1 | 52.8% | 7 | yes | 0 | cached | `gemma4_e2b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgdeb08f01_code667e2d0` |
| N0059 | Granite 4.0 350M Base | v1.1 | 43.9% | 4 | yes | 0 | cached | `granite4_350m_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgd8cbfdb7_code667e2d0` |
| N0060 | Granite 4 H Micro | v0 | 31.5% | 6 | yes | 0 | cached | `granite4_h_micro_v0_data40a22ce8_split147ce64e_seed2026_cfg4420a27e_code667e2d0` |
| N0061 | Granite 4 H Micro | v1.1 | 51.5% | 3 | yes | 0 | cached | `granite4_h_micro_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf29156ed_code667e2d0` |
| N0062 | Llama 3.2 3B | v0 | 34.8% | 10 | yes | 0 | cached | `llama32_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg3d4f188e_code667e2d0` |
| N0063 | Llama 3.2 3B | v1.1 | 51.1% | 3 | yes | 0 | cached | `llama32_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgbe4d02d0_code667e2d0` |
| N0064 | Ministral 3 3B | v0 | 37.4% | 9 | yes | 0 | cached | `ministral3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg616b04ee_code667e2d0` |
| N0065 | Ministral 3 3B | v1.1 | 50.9% | 3 | yes | 0 | cached | `ministral3_3b_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgcc9e4f28_code667e2d0` |
| N0066 | Qwen3 0.6B | v0 | 36.1% | 3 | yes | 0 | cached | `qwen06_best_epoch03_v0_from3789_code8422c8c` |
| N0067 | Qwen3 0.6B | v0 | 30.2% | — | no (cached probabilities only) | 0 | cached | `qwen06_v0_continue4_fromc313b204_seed2026_cfgcc195d58_code8422c8c` |
| N0068 | Qwen3 0.6B | v0 | 32.2% | — | yes | 0 | cached | `qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg42943789_code8422c8c` |
| N0069 | Qwen3 0.6B | v0 | 34.4% | — | no (cached probabilities only) | 0 | cached | `qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg5cb8998d_code365dbc7` |
| N0070 | Qwen3 0.6B | v1.1 | 46.3% | 3 | yes | 0 | cached | `qwen06_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg4df85576_code667e2d0` |
| N0071 | Qwen3 0.6B | v1.1 | 47.0% | 3 | yes | 0 | cached | `qwen06_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfgf24b2061_code667e2d0` |
| N0072 | Qwen3 0.6B | v1.5 | 47.8% | 6 | yes | 0 | cached | `qwen06_v1.5_data490fe052_split215fd0d0_seed2026_cfg088db19e_code667e2d0` |
| N0073 | Qwen3 1.7B | v0 | 31.1% | — | yes | 0 | cached | `qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfg5d2d686c_code667e2d0` |
| N0074 | Qwen3 1.7B | v0 | 31.7% | 5 | yes | 0 | cached | `qwen17_v0_data40a22ce8_split147ce64e_seed2026_cfgf7e4738d_code667e2d0` |
| N0075 | Qwen3 1.7B | v1.1 | 46.3% | 3 | yes | 0 | cached | `qwen17_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg793e1ab7_code667e2d0` |
| N0076 | Qwen3 4B | v0 | 28.3% | 9 | yes | 0 | cached | `qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfg54d18b01_code667e2d0` |
| N0077 | Qwen3 4B | v0 | 26.9% | — | yes | 0 | cached | `qwen4_v0_data40a22ce8_split147ce64e_seed2026_cfgccd8ba38_code667e2d0` |
| N0078 | SmolLM3 3B | v0 | 30.9% | 9 | yes | 0 | cached | `smollm3_3b_v0_data40a22ce8_split147ce64e_seed2026_cfg58d8423d_code667e2d0` |
| N0079 | SmolLM3 3B | v1.5 | 50.7% | 5 | yes | 0 | cached | `smollm3_3b_v1.5_data490fe052_split215fd0d0_seed2026_cfg297336a7_code667e2d0` |
| N0080 | SVC | v0 | 27.2% | — | yes | 0 | cached | `svc_v0_data40a22ce8_split147ce64e_seed2026_cfg6c3be430_code365dbc7` |
| N0081 | SVC | v1 | 43.0% | — | yes | 0 | cached | `svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0` |
| N0082 | SVC | v1.1 | 38.9% | — | yes | 0 | cached | `svc_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg21d4b524_code1b9bfc2` |
| N0083 | SVC | v1.1 | 38.7% | — | yes | 0 | cached | `svc_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg50e77a40_code1b9bfc2` |
| N0084 | SVC | v1.1 | 38.3% | — | yes | 0 | cached | `svc_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg66dc1893_code3076bd5` |
| N0085 | SVC | v1.1 | 38.3% | — | yes | 0 | cached | `svc_v1.1_data72cbc9ee_splitb7cd72d7_seed2026_cfg702a030b_code1b9bfc2` |

Excluded from precision ranking and cascade because their fit partitions overlap frozen validation:
- `catboost_stylometry_catboost_a5_s2027_data40a22ce8_split783f3a95_seed2027_cfgfa54eec9_code667e2d0`: 342 frozen V0 validation rows in its training partition.
- `catboost_stylometry_catboost_a5_s2028_data40a22ce8_split3fbb4529_seed2028_cfgfa54eec9_code667e2d0`: 432 frozen V0 validation rows in its training partition.
- `catboost_stylometry_catboost_a5_s2029_data40a22ce8_splitd83f8692_seed2029_cfgfa54eec9_code667e2d0`: 468 frozen V0 validation rows in its training partition.

## Precision coverage

Counts require at least 10 top-1 predictions for a checkpoint-label pair. A label counts once if any eligible checkpoint meets the raw precision threshold.

| Candidate set | Labels with ≥80% precision | Labels with ≥90% precision |
|---|---:|---:|
| Additional checkpoints | 6/18 (claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5, gemini-3.1-pro-preview, gpt-6-astra) | 2/18 (gemini-3.1-pro-preview, gpt-6-astra) |
| Additional + UltraBERT A–D | 6/18 (claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5, gemini-3.1-pro-preview, gpt-6-astra) | 2/18 (gemini-3.1-pro-preview, gpt-6-astra) |

## Global ranked checkpoint–label rules

All 1485 supported rules are listed in descending smoothed precision. The complete table is included here; unsupported pairs remain in `ranked_checkpoint_label_rules.csv` with `eligible=false` and no rank.

| Rank | Key | Model | Dataset | Label | Precision | Smoothed precision | Recall | TP | FP | Predictions |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | N0070 | Qwen3 0.6B | v1.1 | gemini-3.1-pro-preview | 100.0% | 93.8% | 46.7% | 14 | 0 | 14 |
| 2 | C | UltraBERT-380M | v1.5 | gemini-3.1-pro-preview | 96.0% | 92.6% | 80.0% | 24 | 1 | 25 |
| 3 | N0072 | Qwen3 0.6B | v1.5 | gpt-6-astra | 95.8% | 92.3% | 76.7% | 23 | 1 | 24 |
| 4 | B | UltraBERT-380M | v1_all | gpt-6-astra | 95.5% | 91.7% | 70.0% | 21 | 1 | 22 |
| 5 | A | UltraBERT-380M | v1 | gpt-6-astra | 92.0% | 88.9% | 76.7% | 23 | 2 | 25 |
| 6 | C | UltraBERT-380M | v1.5 | gpt-6-astra | 92.0% | 88.9% | 76.7% | 23 | 2 | 25 |
| 7 | A | UltraBERT-380M | v1 | gemini-3.1-pro-preview | 91.7% | 88.5% | 73.3% | 22 | 2 | 24 |
| 8 | N0061 | Granite 4 H Micro | v1.1 | gemini-3.1-pro-preview | 91.3% | 88.0% | 70.0% | 21 | 2 | 23 |
| 9 | N0071 | Qwen3 0.6B | v1.1 | gemini-3.1-pro-preview | 90.0% | 86.4% | 60.0% | 18 | 2 | 20 |
| 10 | N0065 | Ministral 3 3B | v1.1 | gemini-3.1-pro-preview | 89.5% | 85.7% | 56.7% | 17 | 2 | 19 |
| 11 | B | UltraBERT-380M | v1_all | gemini-3.1-pro-preview | 88.0% | 85.2% | 73.3% | 22 | 3 | 25 |
| 12 | N0072 | Qwen3 0.6B | v1.5 | gemini-3.1-pro-preview | 87.5% | 84.6% | 70.0% | 21 | 3 | 24 |
| 13 | N0059 | Granite 4.0 350M Base | v1.1 | gemini-3.1-pro-preview | 88.2% | 84.2% | 50.0% | 15 | 2 | 17 |
| 14 | D | UltraBERT-380M | v0 | gpt-6-astra | 86.4% | 83.3% | 63.3% | 19 | 3 | 22 |
| 15 | N0058 | Gemma 4 E2B | v1.1 | claude-opus-4-6 | 85.7% | 83.3% | 80.0% | 24 | 4 | 28 |
| 16 | N0058 | Gemma 4 E2B | v1.1 | gemini-3.1-pro-preview | 85.2% | 82.8% | 76.7% | 23 | 4 | 27 |
| 17 | B | UltraBERT-380M | v1_all | claude-opus-4-6 | 85.0% | 81.8% | 56.7% | 17 | 3 | 20 |
| 18 | N0082 | SVC | v1.1 | claude-opus-5 | 85.7% | 81.2% | 40.0% | 12 | 2 | 14 |
| 19 | N0071 | Qwen3 0.6B | v1.1 | gpt-6-astra | 83.3% | 81.2% | 83.3% | 25 | 5 | 30 |
| 20 | D | UltraBERT-380M | v0 | claude-opus-4-8 | 84.2% | 81.0% | 53.3% | 16 | 3 | 19 |
| 21 | N0058 | Gemma 4 E2B | v1.1 | claude-opus-5 | 83.3% | 80.8% | 66.7% | 20 | 4 | 24 |
| 22 | N0058 | Gemma 4 E2B | v1.1 | claude-sonnet-5 | 82.6% | 80.0% | 63.3% | 19 | 4 | 23 |
| 23 | N0079 | SmolLM3 3B | v1.5 | gemini-3.1-pro-preview | 80.6% | 78.8% | 83.3% | 25 | 6 | 31 |
| 24 | N0058 | Gemma 4 E2B | v1.1 | claude-opus-4-8 | 81.2% | 77.8% | 43.3% | 13 | 3 | 16 |
| 25 | N0061 | Granite 4 H Micro | v1.1 | claude-opus-5 | 79.2% | 76.9% | 63.3% | 19 | 5 | 24 |
| 26 | C | UltraBERT-380M | v1.5 | claude-opus-4-6 | 78.6% | 76.7% | 73.3% | 22 | 6 | 28 |
| 27 | N0084 | SVC | v1.1 | claude-opus-5 | 80.0% | 76.5% | 40.0% | 12 | 3 | 15 |
| 28 | N0063 | Llama 3.2 3B | v1.1 | claude-opus-4-6 | 78.1% | 76.5% | 83.3% | 25 | 7 | 32 |
| 29 | N0082 | SVC | v1.1 | claude-opus-4-6 | 78.9% | 76.2% | 50.0% | 15 | 4 | 19 |
| 30 | N0058 | Gemma 4 E2B | v1.1 | gpt-6-astra | 77.8% | 75.9% | 70.0% | 21 | 6 | 27 |
| 31 | N0063 | Llama 3.2 3B | v1.1 | gpt-6-astra | 77.4% | 75.8% | 80.0% | 24 | 7 | 31 |
| 32 | N0070 | Qwen3 0.6B | v1.1 | gpt-6-astra | 77.4% | 75.8% | 80.0% | 24 | 7 | 31 |
| 33 | D | UltraBERT-380M | v0 | gemini-3.1-pro-preview | 77.1% | 75.7% | 90.0% | 27 | 8 | 35 |
| 34 | N0054 | EmbeddingGemma 2 | v1 | gemini-3.1-pro-preview | 77.3% | 75.0% | 56.7% | 17 | 5 | 22 |
| 35 | D | UltraBERT-380M | v0 | claude-opus-5 | 76.9% | 75.0% | 66.7% | 20 | 6 | 26 |
| 36 | N0063 | Llama 3.2 3B | v1.1 | gemini-3.1-pro-preview | 75.8% | 74.3% | 83.3% | 25 | 8 | 33 |
| 37 | A | UltraBERT-380M | v1 | claude-opus-4-6 | 75.9% | 74.2% | 73.3% | 22 | 7 | 29 |
| 38 | A | UltraBERT-380M | v1 | claude-opus-5 | 76.0% | 74.1% | 63.3% | 19 | 6 | 25 |
| 39 | B | UltraBERT-380M | v1_all | claude-sonnet-5 | 76.2% | 73.9% | 53.3% | 16 | 5 | 21 |
| 40 | N0072 | Qwen3 0.6B | v1.5 | claude-opus-5 | 76.5% | 73.7% | 43.3% | 13 | 4 | 17 |
| 41 | N0085 | SVC | v1.1 | claude-opus-5 | 76.5% | 73.7% | 43.3% | 13 | 4 | 17 |
| 42 | C | UltraBERT-380M | v1.5 | claude-opus-5 | 75.0% | 73.3% | 70.0% | 21 | 7 | 28 |
| 43 | N0075 | Qwen3 1.7B | v1.1 | gemini-3.1-pro-preview | 75.0% | 73.1% | 60.0% | 18 | 6 | 24 |
| 44 | N0068 | Qwen3 0.6B | v0 | gemini-3.1-pro-preview | 75.0% | 72.7% | 50.0% | 15 | 5 | 20 |
| 45 | N0063 | Llama 3.2 3B | v1.1 | claude-opus-4-8 | 75.0% | 72.2% | 40.0% | 12 | 4 | 16 |
| 46 | N0083 | SVC | v1.1 | claude-opus-5 | 75.0% | 72.2% | 40.0% | 12 | 4 | 16 |
| 47 | N0065 | Ministral 3 3B | v1.1 | gpt-6-astra | 73.5% | 72.2% | 83.3% | 25 | 9 | 34 |
| 48 | N0079 | SmolLM3 3B | v1.5 | gpt-6-astra | 73.1% | 71.4% | 63.3% | 19 | 7 | 26 |
| 49 | N0083 | SVC | v1.1 | gemini-3.1-pro-preview | 72.7% | 71.4% | 80.0% | 24 | 9 | 33 |
| 50 | N0084 | SVC | v1.1 | gemini-3.1-pro-preview | 72.7% | 71.4% | 80.0% | 24 | 9 | 33 |
| 51 | N0085 | SVC | v1.1 | gemini-3.1-pro-preview | 72.7% | 71.4% | 80.0% | 24 | 9 | 33 |
| 52 | D | UltraBERT-380M | v0 | claude-opus-4-6 | 72.2% | 71.1% | 86.7% | 26 | 10 | 36 |
| 53 | N0072 | Qwen3 0.6B | v1.5 | claude-opus-4-6 | 72.2% | 71.1% | 86.7% | 26 | 10 | 36 |
| 54 | C | UltraBERT-380M | v1.5 | qwen3.8-max | 72.4% | 71.0% | 70.0% | 21 | 8 | 29 |
| 55 | N0065 | Ministral 3 3B | v1.1 | claude-opus-5 | 72.4% | 71.0% | 70.0% | 21 | 8 | 29 |
| 56 | N0054 | EmbeddingGemma 2 | v1 | gpt-6-astra | 72.7% | 70.8% | 53.3% | 16 | 6 | 22 |
| 57 | N0084 | SVC | v1.1 | claude-opus-4-6 | 72.7% | 70.8% | 53.3% | 16 | 6 | 22 |
| 58 | C | UltraBERT-380M | v1.5 | gemini-3.8-flash | 73.3% | 70.6% | 36.7% | 11 | 4 | 15 |
| 59 | N0059 | Granite 4.0 350M Base | v1.1 | claude-opus-4-6 | 71.9% | 70.6% | 76.7% | 23 | 9 | 32 |
| 60 | N0061 | Granite 4 H Micro | v1.1 | claude-sonnet-5 | 72.0% | 70.4% | 60.0% | 18 | 7 | 25 |
| 61 | N0061 | Granite 4 H Micro | v1.1 | gpt-6-astra | 71.4% | 70.3% | 83.3% | 25 | 10 | 35 |
| 62 | B | UltraBERT-380M | v1_all | claude-opus-5 | 71.4% | 70.0% | 66.7% | 20 | 8 | 28 |
| 63 | N0065 | Ministral 3 3B | v1.1 | claude-opus-4-6 | 70.6% | 69.4% | 80.0% | 24 | 10 | 34 |
| 64 | N0082 | SVC | v1.1 | gemini-3.1-pro-preview | 70.6% | 69.4% | 80.0% | 24 | 10 | 34 |
| 65 | N0059 | Granite 4.0 350M Base | v1.1 | claude-opus-5 | 70.0% | 68.2% | 46.7% | 14 | 6 | 20 |
| 66 | N0065 | Ministral 3 3B | v1.1 | gpt-5.5 | 70.0% | 68.2% | 46.7% | 14 | 6 | 20 |
| 67 | N0085 | SVC | v1.1 | claude-opus-4-6 | 70.0% | 68.2% | 46.7% | 14 | 6 | 20 |
| 68 | N0054 | EmbeddingGemma 2 | v1 | claude-opus-4-6 | 69.2% | 67.9% | 60.0% | 18 | 8 | 26 |
| 69 | N0055 | EmbeddingGemma 2 | v1.1 | gemini-3.1-pro-preview | 69.2% | 67.9% | 60.0% | 18 | 8 | 26 |
| 70 | N0056 | EmbeddingGemma 2 | v1.1 | claude-opus-5 | 69.2% | 67.9% | 60.0% | 18 | 8 | 26 |
| 71 | N0075 | Qwen3 1.7B | v1.1 | claude-opus-4-6 | 68.8% | 67.6% | 73.3% | 22 | 10 | 32 |
| 72 | N0061 | Granite 4 H Micro | v1.1 | claude-opus-4-8 | 68.2% | 66.7% | 50.0% | 15 | 7 | 22 |
| 73 | N0071 | Qwen3 0.6B | v1.1 | claude-opus-5 | 68.2% | 66.7% | 50.0% | 15 | 7 | 22 |
| 74 | N0083 | SVC | v1.1 | claude-opus-4-6 | 68.2% | 66.7% | 50.0% | 15 | 7 | 22 |
| 75 | N0079 | SmolLM3 3B | v1.5 | qwen3.8-max | 68.0% | 66.7% | 56.7% | 17 | 8 | 25 |
| 76 | N0061 | Granite 4 H Micro | v1.1 | qwen3.8-max | 67.9% | 66.7% | 63.3% | 19 | 9 | 28 |
| 77 | N0063 | Llama 3.2 3B | v1.1 | claude-opus-5 | 67.7% | 66.7% | 70.0% | 21 | 10 | 31 |
| 78 | N0081 | SVC | v1 | gemini-3.1-pro-preview | 67.6% | 66.7% | 83.3% | 25 | 12 | 37 |
| 79 | N0070 | Qwen3 0.6B | v1.1 | claude-opus-5 | 66.7% | 65.6% | 66.7% | 20 | 10 | 30 |
| 80 | N0079 | SmolLM3 3B | v1.5 | claude-opus-5 | 65.8% | 65.0% | 83.3% | 25 | 13 | 38 |
| 81 | N0064 | Ministral 3 3B | v0 | claude-opus-4-8 | 66.7% | 64.7% | 33.3% | 10 | 5 | 15 |
| 82 | N0075 | Qwen3 1.7B | v1.1 | claude-opus-4-8 | 66.7% | 64.7% | 33.3% | 10 | 5 | 15 |
| 83 | N0075 | Qwen3 1.7B | v1.1 | claude-opus-5 | 65.6% | 64.7% | 70.0% | 21 | 11 | 32 |
| 84 | N0075 | Qwen3 1.7B | v1.1 | gpt-6-astra | 65.6% | 64.7% | 70.0% | 21 | 11 | 32 |
| 85 | N0073 | Qwen3 1.7B | v0 | claude-opus-5 | 65.5% | 64.5% | 63.3% | 19 | 10 | 29 |
| 86 | N0055 | EmbeddingGemma 2 | v1.1 | claude-sonnet-5 | 66.7% | 64.3% | 26.7% | 8 | 4 | 12 |
| 87 | N0059 | Granite 4.0 350M Base | v1.1 | gpt-6-astra | 64.7% | 63.9% | 73.3% | 22 | 12 | 34 |
| 88 | N0079 | SmolLM3 3B | v1.5 | claude-sonnet-5 | 64.7% | 63.9% | 73.3% | 22 | 12 | 34 |
| 89 | N0064 | Ministral 3 3B | v0 | gemini-3.1-pro-preview | 65.0% | 63.6% | 43.3% | 13 | 7 | 20 |
| 90 | N0056 | EmbeddingGemma 2 | v1.1 | claude-opus-4-6 | 64.3% | 63.3% | 60.0% | 18 | 10 | 28 |
| 91 | N0066 | Qwen3 0.6B | v0 | gemini-3.1-pro-preview | 64.3% | 63.3% | 60.0% | 18 | 10 | 28 |
| 92 | N0078 | SmolLM3 3B | v0 | gemini-3.1-pro-preview | 64.7% | 63.2% | 36.7% | 11 | 6 | 17 |
| 93 | N0052 | EmbeddingGemma 2 | v0 | gemini-3.1-pro-preview | 64.3% | 62.5% | 30.0% | 9 | 5 | 14 |
| 94 | N0075 | Qwen3 1.7B | v1.1 | gpt-5.6-luna | 64.3% | 62.5% | 30.0% | 9 | 5 | 14 |
| 95 | N0056 | EmbeddingGemma 2 | v1.1 | gpt-6-astra | 63.6% | 62.5% | 46.7% | 14 | 8 | 22 |
| 96 | N0061 | Granite 4 H Micro | v1.1 | gemini-3.8-flash | 63.6% | 62.5% | 46.7% | 14 | 8 | 22 |
| 97 | N0070 | Qwen3 0.6B | v1.1 | claude-sonnet-5 | 63.6% | 62.5% | 46.7% | 14 | 8 | 22 |
| 98 | A | UltraBERT-380M | v1 | qwen3.8-max | 63.3% | 62.5% | 63.3% | 19 | 11 | 30 |
| 99 | B | UltraBERT-380M | v1_all | qwen3.8-max | 63.3% | 62.5% | 63.3% | 19 | 11 | 30 |
| 100 | N0078 | SmolLM3 3B | v0 | claude-opus-5 | 63.2% | 61.9% | 40.0% | 12 | 7 | 19 |
| 101 | D | UltraBERT-380M | v0 | qwen3.8-max | 62.5% | 61.8% | 66.7% | 20 | 12 | 32 |
| 102 | N0057 | EmbeddingGemma 2 | v1.5 | claude-opus-4-6 | 62.5% | 61.8% | 66.7% | 20 | 12 | 32 |
| 103 | N0066 | Qwen3 0.6B | v0 | gpt-6-astra | 62.5% | 61.8% | 66.7% | 20 | 12 | 32 |
| 104 | N0059 | Granite 4.0 350M Base | v1.1 | claude-opus-4-8 | 63.6% | 61.5% | 23.3% | 7 | 4 | 11 |
| 105 | N0073 | Qwen3 1.7B | v0 | claude-opus-4-6 | 62.5% | 61.5% | 50.0% | 15 | 9 | 24 |
| 106 | N0058 | Gemma 4 E2B | v1.1 | gpt-5.5 | 62.1% | 61.3% | 60.0% | 18 | 11 | 29 |
| 107 | N0064 | Ministral 3 3B | v0 | claude-opus-5 | 62.5% | 61.1% | 33.3% | 10 | 6 | 16 |
| 108 | N0071 | Qwen3 0.6B | v1.1 | claude-opus-4-6 | 61.5% | 61.0% | 80.0% | 24 | 15 | 39 |
| 109 | N0057 | EmbeddingGemma 2 | v1.5 | gpt-6-astra | 61.9% | 60.9% | 43.3% | 13 | 8 | 21 |
| 110 | N0083 | SVC | v1.1 | gemini-3.6-flash | 61.9% | 60.9% | 43.3% | 13 | 8 | 21 |
| 111 | N0085 | SVC | v1.1 | gemini-3.6-flash | 61.9% | 60.9% | 43.3% | 13 | 8 | 21 |
| 112 | C | UltraBERT-380M | v1.5 | claude-sonnet-5 | 61.5% | 60.7% | 53.3% | 16 | 10 | 26 |
| 113 | N0073 | Qwen3 1.7B | v0 | gemini-3.1-pro-preview | 61.5% | 60.7% | 53.3% | 16 | 10 | 26 |
| 114 | N0056 | EmbeddingGemma 2 | v1.1 | gemini-3.1-pro-preview | 61.3% | 60.6% | 63.3% | 19 | 12 | 31 |
| 115 | N0063 | Llama 3.2 3B | v1.1 | claude-sonnet-5 | 61.3% | 60.6% | 63.3% | 19 | 12 | 31 |
| 116 | N0082 | SVC | v1.1 | gpt-6-astra | 61.1% | 60.5% | 73.3% | 22 | 14 | 36 |
| 117 | N0064 | Ministral 3 3B | v0 | gemini-3.8-flash | 61.1% | 60.0% | 36.7% | 11 | 7 | 18 |
| 118 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.1-pro-preview | 60.6% | 60.0% | 66.7% | 20 | 13 | 33 |
| 119 | N0084 | SVC | v1.1 | gpt-6-astra | 60.0% | 59.5% | 80.0% | 24 | 16 | 40 |
| 120 | N0071 | Qwen3 0.6B | v1.1 | claude-sonnet-5 | 60.0% | 59.1% | 40.0% | 12 | 8 | 20 |
| 121 | N0014 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 59.4% | 58.8% | 63.3% | 19 | 13 | 32 |
| 122 | N0031 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 59.4% | 58.8% | 63.3% | 19 | 13 | 32 |
| 123 | N0070 | Qwen3 0.6B | v1.1 | claude-opus-4-6 | 59.4% | 58.8% | 63.3% | 19 | 13 | 32 |
| 124 | D | UltraBERT-380M | v0 | claude-sonnet-5 | 59.1% | 58.7% | 86.7% | 26 | 18 | 44 |
| 125 | N0065 | Ministral 3 3B | v1.1 | gemini-3.8-flash | 59.3% | 58.6% | 53.3% | 16 | 11 | 27 |
| 126 | N0031 | CatBoost + Stylometry | v1 | glm-5.3-flash | 60.0% | 58.3% | 20.0% | 6 | 4 | 10 |
| 127 | N0071 | Qwen3 0.6B | v1.1 | gpt-5.6-sol | 60.0% | 58.3% | 20.0% | 6 | 4 | 10 |
| 128 | N0043 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 58.8% | 58.3% | 66.7% | 20 | 14 | 34 |
| 129 | N0050 | CatBoost + Stylometry | v1.5 | gemini-3.1-pro-preview | 58.8% | 58.3% | 66.7% | 20 | 14 | 34 |
| 130 | N0083 | SVC | v1.1 | gpt-6-astra | 58.5% | 58.1% | 80.0% | 24 | 17 | 41 |
| 131 | A | UltraBERT-380M | v1 | gemini-3.8-flash | 58.6% | 58.1% | 56.7% | 17 | 12 | 29 |
| 132 | N0082 | SVC | v1.1 | gemini-3.7-flash | 58.8% | 57.9% | 33.3% | 10 | 7 | 17 |
| 133 | D | UltraBERT-380M | v0 | gemini-3.8-flash | 58.3% | 57.7% | 46.7% | 14 | 10 | 24 |
| 134 | N0057 | EmbeddingGemma 2 | v1.5 | qwen3.8-max | 58.3% | 57.7% | 46.7% | 14 | 10 | 24 |
| 135 | A | UltraBERT-380M | v1 | claude-sonnet-5 | 58.1% | 57.6% | 60.0% | 18 | 13 | 31 |
| 136 | N0075 | Qwen3 1.7B | v1.1 | claude-sonnet-5 | 58.1% | 57.6% | 60.0% | 18 | 13 | 31 |
| 137 | A | UltraBERT-380M | v1 | claude-opus-4-8 | 57.7% | 57.1% | 50.0% | 15 | 11 | 26 |
| 138 | N0055 | EmbeddingGemma 2 | v1.1 | claude-opus-4-6 | 57.6% | 57.1% | 63.3% | 19 | 14 | 33 |
| 139 | N0085 | SVC | v1.1 | gpt-6-astra | 57.1% | 56.8% | 80.0% | 24 | 18 | 42 |
| 140 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.1-pro-preview | 57.1% | 56.8% | 66.7% | 20 | 15 | 35 |
| 141 | N0062 | Llama 3.2 3B | v0 | gemini-3.1-pro-preview | 57.1% | 56.8% | 66.7% | 20 | 15 | 35 |
| 142 | N0054 | EmbeddingGemma 2 | v1 | claude-sonnet-5 | 57.1% | 56.7% | 53.3% | 16 | 12 | 28 |
| 143 | N0069 | Qwen3 0.6B | v0 | claude-opus-4-6 | 57.1% | 56.7% | 53.3% | 16 | 12 | 28 |
| 144 | B | UltraBERT-380M | v1_all | claude-opus-4-8 | 56.8% | 56.4% | 70.0% | 21 | 16 | 37 |
| 145 | D | UltraBERT-380M | v0 | kimi-k3 | 57.1% | 56.2% | 26.7% | 8 | 6 | 14 |
| 146 | N0074 | Qwen3 1.7B | v0 | gemini-3.6-flash | 57.1% | 56.2% | 26.7% | 8 | 6 | 14 |
| 147 | N0032 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 56.7% | 56.2% | 56.7% | 17 | 13 | 30 |
| 148 | N0041 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 56.7% | 56.2% | 56.7% | 17 | 13 | 30 |
| 149 | N0044 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 56.7% | 56.2% | 56.7% | 17 | 13 | 30 |
| 150 | N0047 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 56.7% | 56.2% | 56.7% | 17 | 13 | 30 |
| 151 | N0049 | CatBoost + Stylometry | v1.5 | gemini-3.1-pro-preview | 56.7% | 56.2% | 56.7% | 17 | 13 | 30 |
| 152 | N0079 | SmolLM3 3B | v1.5 | claude-opus-4-8 | 56.5% | 56.0% | 43.3% | 13 | 10 | 23 |
| 153 | N0084 | SVC | v1.1 | claude-sonnet-5 | 56.5% | 56.0% | 43.3% | 13 | 10 | 23 |
| 154 | N0084 | SVC | v1.1 | gemini-3.6-flash | 56.5% | 56.0% | 43.3% | 13 | 10 | 23 |
| 155 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.1-pro-preview | 56.2% | 55.9% | 60.0% | 18 | 14 | 32 |
| 156 | B | UltraBERT-380M | v1_all | gemini-3.8-flash | 55.9% | 55.6% | 63.3% | 19 | 15 | 34 |
| 157 | N0071 | Qwen3 0.6B | v1.1 | gemini-3.8-flash | 55.9% | 55.6% | 63.3% | 19 | 15 | 34 |
| 158 | N0080 | SVC | v0 | gemini-3.1-pro-preview | 55.6% | 55.3% | 83.3% | 25 | 20 | 45 |
| 159 | N0069 | Qwen3 0.6B | v0 | gemini-3.1-pro-preview | 55.3% | 55.0% | 70.0% | 21 | 17 | 38 |
| 160 | N0079 | SmolLM3 3B | v1.5 | claude-opus-4-6 | 55.1% | 54.9% | 90.0% | 27 | 22 | 49 |
| 161 | N0058 | Gemma 4 E2B | v1.1 | qwen3.8-max | 55.2% | 54.8% | 53.3% | 16 | 13 | 29 |
| 162 | N0063 | Llama 3.2 3B | v1.1 | gemini-3.8-flash | 55.2% | 54.8% | 53.3% | 16 | 13 | 29 |
| 163 | N0078 | SmolLM3 3B | v0 | claude-opus-4-6 | 55.2% | 54.8% | 53.3% | 16 | 13 | 29 |
| 164 | D | UltraBERT-380M | v0 | gpt-5.6-sol | 55.0% | 54.5% | 36.7% | 11 | 9 | 20 |
| 165 | N0056 | EmbeddingGemma 2 | v1.1 | claude-sonnet-5 | 55.0% | 54.5% | 36.7% | 11 | 9 | 20 |
| 166 | N0067 | Qwen3 0.6B | v0 | gemini-3.1-pro-preview | 54.8% | 54.5% | 56.7% | 17 | 14 | 31 |
| 167 | N0061 | Granite 4 H Micro | v1.1 | claude-opus-4-6 | 54.8% | 54.5% | 76.7% | 23 | 19 | 42 |
| 168 | N0057 | EmbeddingGemma 2 | v1.5 | gemini-3.1-pro-preview | 54.5% | 54.3% | 80.0% | 24 | 20 | 44 |
| 169 | C | UltraBERT-380M | v1.5 | claude-opus-4-8 | 54.5% | 54.3% | 60.0% | 18 | 15 | 33 |
| 170 | N0063 | Llama 3.2 3B | v1.1 | gemini-3.6-flash | 54.5% | 54.3% | 60.0% | 18 | 15 | 33 |
| 171 | N0081 | SVC | v1 | gemini-3.8-flash | 54.5% | 54.3% | 60.0% | 18 | 15 | 33 |
| 172 | N0028 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 54.3% | 54.1% | 63.3% | 19 | 16 | 35 |
| 173 | C | UltraBERT-380M | v1.5 | gemini-3.6-flash | 54.2% | 53.8% | 43.3% | 13 | 11 | 24 |
| 174 | N0082 | SVC | v1.1 | claude-sonnet-5 | 54.2% | 53.8% | 43.3% | 13 | 11 | 24 |
| 175 | N0012 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 54.1% | 53.8% | 66.7% | 20 | 17 | 37 |
| 176 | N0013 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 54.1% | 53.8% | 66.7% | 20 | 17 | 37 |
| 177 | N0021 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 54.1% | 53.8% | 66.7% | 20 | 17 | 37 |
| 178 | N0023 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 54.1% | 53.8% | 66.7% | 20 | 17 | 37 |
| 179 | N0026 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 54.1% | 53.8% | 66.7% | 20 | 17 | 37 |
| 180 | N0027 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 54.1% | 53.8% | 66.7% | 20 | 17 | 37 |
| 181 | N0065 | Ministral 3 3B | v1.1 | qwen3.8-max | 53.8% | 53.7% | 70.0% | 21 | 18 | 39 |
| 182 | N0077 | Qwen3 4B | v0 | claude-opus-4-6 | 53.8% | 53.6% | 46.7% | 14 | 12 | 26 |
| 183 | C | UltraBERT-380M | v1.5 | gpt-5.5 | 53.6% | 53.3% | 50.0% | 15 | 13 | 28 |
| 184 | D | UltraBERT-380M | v0 | gemini-3.6-flash | 53.5% | 53.3% | 76.7% | 23 | 20 | 43 |
| 185 | N0081 | SVC | v1 | gpt-6-astra | 53.5% | 53.3% | 76.7% | 23 | 20 | 43 |
| 186 | B | UltraBERT-380M | v1_all | kimi-k3 | 53.3% | 53.1% | 53.3% | 16 | 14 | 30 |
| 187 | N0046 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 53.3% | 53.1% | 53.3% | 16 | 14 | 30 |
| 188 | N0054 | EmbeddingGemma 2 | v1 | claude-opus-5 | 53.3% | 53.1% | 53.3% | 16 | 14 | 30 |
| 189 | N0048 | CatBoost + Stylometry | v1.1 | gemini-3.1-pro-preview | 53.1% | 52.9% | 56.7% | 17 | 15 | 32 |
| 190 | N0083 | SVC | v1.1 | gemini-3.8-flash | 53.1% | 52.9% | 56.7% | 17 | 15 | 32 |
| 191 | N0085 | SVC | v1.1 | gemini-3.8-flash | 53.1% | 52.9% | 56.7% | 17 | 15 | 32 |
| 192 | N0030 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 52.9% | 52.8% | 60.0% | 18 | 16 | 34 |
| 193 | B | UltraBERT-380M | v1_all | gpt-5.5 | 52.9% | 52.6% | 30.0% | 9 | 8 | 17 |
| 194 | N0058 | Gemma 4 E2B | v1.1 | gpt-5.6-luna | 52.9% | 52.6% | 30.0% | 9 | 8 | 17 |
| 195 | N0042 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 52.6% | 52.5% | 66.7% | 20 | 18 | 38 |
| 196 | N0019 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 52.5% | 52.4% | 70.0% | 21 | 19 | 40 |
| 197 | N0083 | SVC | v1.1 | claude-sonnet-5 | 52.0% | 51.9% | 43.3% | 13 | 12 | 25 |
| 198 | N0085 | SVC | v1.1 | claude-sonnet-5 | 52.0% | 51.9% | 43.3% | 13 | 12 | 25 |
| 199 | N0070 | Qwen3 0.6B | v1.1 | gpt-5.6-sol | 51.9% | 51.7% | 46.7% | 14 | 13 | 27 |
| 200 | N0082 | SVC | v1.1 | gemini-3.6-flash | 51.9% | 51.7% | 46.7% | 14 | 13 | 27 |
| 201 | N0058 | Gemma 4 E2B | v1.1 | gemini-3.6-flash | 51.7% | 51.6% | 50.0% | 15 | 14 | 29 |
| 202 | N0079 | SmolLM3 3B | v1.5 | gemini-3.8-flash | 51.7% | 51.6% | 50.0% | 15 | 14 | 29 |
| 203 | N0081 | SVC | v1 | gemini-3.6-flash | 51.7% | 51.6% | 50.0% | 15 | 14 | 29 |
| 204 | N0076 | Qwen3 4B | v0 | claude-opus-4-6 | 51.6% | 51.5% | 53.3% | 16 | 15 | 31 |
| 205 | N0065 | Ministral 3 3B | v1.1 | claude-sonnet-5 | 51.5% | 51.4% | 56.7% | 17 | 16 | 33 |
| 206 | N0067 | Qwen3 0.6B | v0 | claude-opus-4-6 | 51.5% | 51.4% | 56.7% | 17 | 16 | 33 |
| 207 | N0084 | SVC | v1.1 | gemini-3.8-flash | 51.5% | 51.4% | 56.7% | 17 | 16 | 33 |
| 208 | N0081 | SVC | v1 | claude-opus-5 | 51.4% | 51.4% | 60.0% | 18 | 17 | 35 |
| 209 | N0015 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 51.4% | 51.3% | 63.3% | 19 | 18 | 37 |
| 210 | N0064 | Ministral 3 3B | v0 | claude-opus-4-6 | 51.4% | 51.3% | 63.3% | 19 | 18 | 37 |
| 211 | N0070 | Qwen3 0.6B | v1.1 | gemini-3.8-flash | 51.3% | 51.2% | 66.7% | 20 | 19 | 39 |
| 212 | B | UltraBERT-380M | v1_all | gemini-3.6-flash | 51.2% | 51.2% | 70.0% | 21 | 20 | 41 |
| 213 | N0020 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 51.2% | 51.2% | 70.0% | 21 | 20 | 41 |
| 214 | N0069 | Qwen3 0.6B | v0 | gpt-6-astra | 50.0% | 50.0% | 70.0% | 21 | 21 | 42 |
| 215 | N0081 | SVC | v1 | claude-opus-4-6 | 50.0% | 50.0% | 66.7% | 20 | 20 | 40 |
| 216 | N0039 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 50.0% | 50.0% | 60.0% | 18 | 18 | 36 |
| 217 | N0040 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 50.0% | 50.0% | 60.0% | 18 | 18 | 36 |
| 218 | N0057 | EmbeddingGemma 2 | v1.5 | claude-opus-5 | 50.0% | 50.0% | 60.0% | 18 | 18 | 36 |
| 219 | N0038 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 50.0% | 50.0% | 56.7% | 17 | 17 | 34 |
| 220 | N0070 | Qwen3 0.6B | v1.1 | qwen3.8-max | 50.0% | 50.0% | 56.7% | 17 | 17 | 34 |
| 221 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.1-pro-preview | 50.0% | 50.0% | 53.3% | 16 | 16 | 32 |
| 222 | N0057 | EmbeddingGemma 2 | v1.5 | gemini-3.8-flash | 50.0% | 50.0% | 53.3% | 16 | 16 | 32 |
| 223 | N0065 | Ministral 3 3B | v1.1 | gemini-3.7-flash | 50.0% | 50.0% | 50.0% | 15 | 15 | 30 |
| 224 | N0079 | SmolLM3 3B | v1.5 | glm-5.3-flash | 50.0% | 50.0% | 50.0% | 15 | 15 | 30 |
| 225 | A | UltraBERT-380M | v1 | gemini-3.6-flash | 50.0% | 50.0% | 46.7% | 14 | 14 | 28 |
| 226 | N0058 | Gemma 4 E2B | v1.1 | gemini-3.8-flash | 50.0% | 50.0% | 46.7% | 14 | 14 | 28 |
| 227 | N0061 | Granite 4 H Micro | v1.1 | kimi-k3 | 50.0% | 50.0% | 46.7% | 14 | 14 | 28 |
| 228 | N0072 | Qwen3 0.6B | v1.5 | claude-opus-4-8 | 50.0% | 50.0% | 46.7% | 14 | 14 | 28 |
| 229 | N0064 | Ministral 3 3B | v0 | claude-sonnet-5 | 50.0% | 50.0% | 43.3% | 13 | 13 | 26 |
| 230 | N0073 | Qwen3 1.7B | v0 | qwen3.8-max | 50.0% | 50.0% | 43.3% | 13 | 13 | 26 |
| 231 | N0074 | Qwen3 1.7B | v0 | claude-opus-4-6 | 50.0% | 50.0% | 43.3% | 13 | 13 | 26 |
| 232 | A | UltraBERT-380M | v1 | gpt-5.5 | 50.0% | 50.0% | 40.0% | 12 | 12 | 24 |
| 233 | N0063 | Llama 3.2 3B | v1.1 | gpt-5.5 | 50.0% | 50.0% | 36.7% | 11 | 11 | 22 |
| 234 | N0072 | Qwen3 0.6B | v1.5 | gemini-3.8-flash | 50.0% | 50.0% | 36.7% | 11 | 11 | 22 |
| 235 | N0057 | EmbeddingGemma 2 | v1.5 | gemini-3.7-flash | 50.0% | 50.0% | 33.3% | 10 | 10 | 20 |
| 236 | N0073 | Qwen3 1.7B | v0 | gpt-6-astra | 50.0% | 50.0% | 26.7% | 8 | 8 | 16 |
| 237 | N0057 | EmbeddingGemma 2 | v1.5 | gpt-5.6-terra | 50.0% | 50.0% | 23.3% | 7 | 7 | 14 |
| 238 | N0065 | Ministral 3 3B | v1.1 | claude-opus-4-8 | 50.0% | 50.0% | 16.7% | 5 | 5 | 10 |
| 239 | N0072 | Qwen3 0.6B | v1.5 | claude-sonnet-5 | 48.9% | 49.0% | 76.7% | 23 | 24 | 47 |
| 240 | N0061 | Granite 4 H Micro | v1.1 | gemini-3.6-flash | 48.8% | 48.9% | 70.0% | 21 | 22 | 43 |
| 241 | N0055 | EmbeddingGemma 2 | v1.1 | claude-opus-5 | 48.8% | 48.8% | 66.7% | 20 | 21 | 41 |
| 242 | N0022 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 48.6% | 48.7% | 60.0% | 18 | 19 | 37 |
| 243 | N0034 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 48.6% | 48.7% | 60.0% | 18 | 19 | 37 |
| 244 | N0068 | Qwen3 0.6B | v0 | claude-opus-4-6 | 48.5% | 48.6% | 53.3% | 16 | 17 | 33 |
| 245 | N0059 | Granite 4.0 350M Base | v1.1 | gpt-5.6-luna | 48.4% | 48.5% | 50.0% | 15 | 16 | 31 |
| 246 | N0062 | Llama 3.2 3B | v0 | gpt-6-astra | 48.4% | 48.5% | 50.0% | 15 | 16 | 31 |
| 247 | N0059 | Granite 4.0 350M Base | v1.1 | gemini-3.8-flash | 48.3% | 48.4% | 46.7% | 14 | 15 | 29 |
| 248 | N0070 | Qwen3 0.6B | v1.1 | gpt-5.5 | 48.3% | 48.4% | 46.7% | 14 | 15 | 29 |
| 249 | N0061 | Granite 4 H Micro | v1.1 | gpt-5.5 | 48.1% | 48.3% | 43.3% | 13 | 14 | 27 |
| 250 | N0071 | Qwen3 0.6B | v1.1 | gpt-5.6-luna | 48.1% | 48.3% | 43.3% | 13 | 14 | 27 |
| 251 | N0080 | SVC | v0 | claude-opus-4-6 | 48.0% | 48.1% | 40.0% | 12 | 13 | 25 |
| 252 | N0084 | SVC | v1.1 | gemini-3.7-flash | 48.0% | 48.1% | 40.0% | 12 | 13 | 25 |
| 253 | N0081 | SVC | v1 | gemini-3.7-flash | 47.8% | 48.0% | 36.7% | 11 | 12 | 23 |
| 254 | N0010 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 47.7% | 47.8% | 70.0% | 21 | 23 | 44 |
| 255 | B | UltraBERT-380M | v1_all | gemini-3.7-flash | 47.6% | 47.8% | 33.3% | 10 | 11 | 21 |
| 256 | N0055 | EmbeddingGemma 2 | v1.1 | claude-opus-4-8 | 47.6% | 47.8% | 33.3% | 10 | 11 | 21 |
| 257 | N0065 | Ministral 3 3B | v1.1 | kimi-k3 | 47.4% | 47.6% | 30.0% | 9 | 10 | 19 |
| 258 | N0066 | Qwen3 0.6B | v0 | claude-opus-4-6 | 47.4% | 47.5% | 60.0% | 18 | 20 | 38 |
| 259 | N0036 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 47.2% | 47.4% | 56.7% | 17 | 19 | 36 |
| 260 | A | UltraBERT-380M | v1 | claude-fable-5 | 47.1% | 47.4% | 26.7% | 8 | 9 | 17 |
| 261 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.1-pro-preview | 47.1% | 47.2% | 53.3% | 16 | 18 | 34 |
| 262 | N0037 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 47.1% | 47.2% | 53.3% | 16 | 18 | 34 |
| 263 | N0072 | Qwen3 0.6B | v1.5 | gpt-5.6-sol | 47.1% | 47.2% | 53.3% | 16 | 18 | 34 |
| 264 | N0060 | Granite 4 H Micro | v0 | gemini-3.7-flash | 46.7% | 47.1% | 23.3% | 7 | 8 | 15 |
| 265 | N0030 | CatBoost + Stylometry | v1 | gpt-6-astra | 46.7% | 46.9% | 46.7% | 14 | 16 | 30 |
| 266 | N0043 | CatBoost + Stylometry | v1 | gpt-6-astra | 46.7% | 46.9% | 46.7% | 14 | 16 | 30 |
| 267 | N0050 | CatBoost + Stylometry | v1.5 | gpt-6-astra | 46.7% | 46.9% | 46.7% | 14 | 16 | 30 |
| 268 | N0059 | Granite 4.0 350M Base | v1.1 | gpt-5.6-terra | 46.4% | 46.7% | 43.3% | 13 | 15 | 28 |
| 269 | N0068 | Qwen3 0.6B | v0 | claude-opus-5 | 46.4% | 46.7% | 43.3% | 13 | 15 | 28 |
| 270 | N0071 | Qwen3 0.6B | v1.1 | kimi-k3 | 46.2% | 46.7% | 20.0% | 6 | 7 | 13 |
| 271 | N0072 | Qwen3 0.6B | v1.5 | gpt-5.6-luna | 46.2% | 46.7% | 20.0% | 6 | 7 | 13 |
| 272 | N0075 | Qwen3 1.7B | v1.1 | kimi-k3 | 46.2% | 46.7% | 20.0% | 6 | 7 | 13 |
| 273 | N0011 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 46.3% | 46.5% | 63.3% | 19 | 22 | 41 |
| 274 | N0017 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 46.3% | 46.5% | 63.3% | 19 | 22 | 41 |
| 275 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-6-astra | 46.2% | 46.4% | 40.0% | 12 | 14 | 26 |
| 276 | N0083 | SVC | v1.1 | gemini-3.7-flash | 46.2% | 46.4% | 40.0% | 12 | 14 | 26 |
| 277 | N0085 | SVC | v1.1 | gemini-3.7-flash | 46.2% | 46.4% | 40.0% | 12 | 14 | 26 |
| 278 | N0082 | SVC | v1.1 | gemini-3.8-flash | 45.9% | 46.2% | 56.7% | 17 | 20 | 37 |
| 279 | N0054 | EmbeddingGemma 2 | v1 | gpt-5.5 | 45.8% | 46.2% | 36.7% | 11 | 13 | 24 |
| 280 | N0063 | Llama 3.2 3B | v1.1 | gemini-3.7-flash | 45.8% | 46.2% | 36.7% | 11 | 13 | 24 |
| 281 | N0075 | Qwen3 1.7B | v1.1 | gemini-3.6-flash | 45.8% | 46.2% | 36.7% | 11 | 13 | 24 |
| 282 | N0057 | EmbeddingGemma 2 | v1.5 | kimi-k3 | 45.5% | 46.2% | 16.7% | 5 | 6 | 11 |
| 283 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.1-pro-preview | 45.7% | 45.9% | 53.3% | 16 | 19 | 35 |
| 284 | N0058 | Gemma 4 E2B | v1.1 | gemini-3.7-flash | 45.7% | 45.9% | 53.3% | 16 | 19 | 35 |
| 285 | N0072 | Qwen3 0.6B | v1.5 | gemini-3.6-flash | 45.7% | 45.9% | 53.3% | 16 | 19 | 35 |
| 286 | N0024 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 45.7% | 45.8% | 70.0% | 21 | 25 | 46 |
| 287 | N0075 | Qwen3 1.7B | v1.1 | gpt-5.6-sol | 45.5% | 45.8% | 33.3% | 10 | 12 | 22 |
| 288 | A | UltraBERT-380M | v1 | gpt-5.6-sol | 45.5% | 45.7% | 50.0% | 15 | 18 | 33 |
| 289 | N0059 | Granite 4.0 350M Base | v1.1 | qwen3.8-max | 45.5% | 45.7% | 66.7% | 20 | 24 | 44 |
| 290 | N0065 | Ministral 3 3B | v1.1 | gemini-3.6-flash | 45.5% | 45.7% | 66.7% | 20 | 24 | 44 |
| 291 | N0059 | Granite 4.0 350M Base | v1.1 | claude-sonnet-5 | 45.3% | 45.5% | 80.0% | 24 | 29 | 53 |
| 292 | N0027 | CatBoost + Stylometry | v0 | gpt-6-astra | 45.2% | 45.5% | 63.3% | 19 | 23 | 42 |
| 293 | N0064 | Ministral 3 3B | v0 | gpt-6-astra | 45.2% | 45.5% | 63.3% | 19 | 23 | 42 |
| 294 | N0068 | Qwen3 0.6B | v0 | gpt-6-astra | 45.2% | 45.5% | 63.3% | 19 | 23 | 42 |
| 295 | N0022 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 45.2% | 45.5% | 46.7% | 14 | 17 | 31 |
| 296 | N0027 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 45.2% | 45.5% | 46.7% | 14 | 17 | 31 |
| 297 | N0028 | CatBoost + Stylometry | v1 | glm-5.3-flash | 45.0% | 45.5% | 30.0% | 9 | 11 | 20 |
| 298 | N0029 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 45.0% | 45.2% | 60.0% | 18 | 22 | 40 |
| 299 | N0045 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 45.0% | 45.2% | 60.0% | 18 | 22 | 40 |
| 300 | N0057 | EmbeddingGemma 2 | v1.5 | claude-sonnet-5 | 45.0% | 45.2% | 60.0% | 18 | 22 | 40 |
| 301 | N0071 | Qwen3 0.6B | v1.1 | qwen3.8-max | 45.0% | 45.2% | 60.0% | 18 | 22 | 40 |
| 302 | N0072 | Qwen3 0.6B | v1.5 | qwen3.8-max | 45.0% | 45.2% | 60.0% | 18 | 22 | 40 |
| 303 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-6-astra | 44.8% | 45.2% | 43.3% | 13 | 16 | 29 |
| 304 | N0062 | Llama 3.2 3B | v0 | gemini-3.7-flash | 44.8% | 45.2% | 43.3% | 13 | 16 | 29 |
| 305 | N0067 | Qwen3 0.6B | v0 | claude-opus-5 | 44.8% | 45.2% | 43.3% | 13 | 16 | 29 |
| 306 | D | UltraBERT-380M | v0 | gemini-3.7-flash | 44.4% | 45.0% | 26.7% | 8 | 10 | 18 |
| 307 | N0081 | SVC | v1 | gpt-5.6-sol | 44.4% | 45.0% | 26.7% | 8 | 10 | 18 |
| 308 | N0030 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 44.4% | 44.8% | 40.0% | 12 | 15 | 27 |
| 309 | A | UltraBERT-380M | v1 | gpt-5.6-terra | 44.4% | 44.7% | 53.3% | 16 | 20 | 36 |
| 310 | N0062 | Llama 3.2 3B | v0 | claude-opus-5 | 44.2% | 44.4% | 63.3% | 19 | 24 | 43 |
| 311 | N0077 | Qwen3 4B | v0 | claude-opus-5 | 44.1% | 44.4% | 50.0% | 15 | 19 | 34 |
| 312 | N0057 | EmbeddingGemma 2 | v1.5 | gemini-3.6-flash | 44.0% | 44.4% | 36.7% | 11 | 14 | 25 |
| 313 | N0064 | Ministral 3 3B | v0 | qwen3.8-max | 43.8% | 44.4% | 23.3% | 7 | 9 | 16 |
| 314 | N0065 | Ministral 3 3B | v1.1 | gpt-5.6-luna | 43.8% | 44.4% | 23.3% | 7 | 9 | 16 |
| 315 | N0068 | Qwen3 0.6B | v0 | gemini-3.7-flash | 43.8% | 44.4% | 23.3% | 7 | 9 | 16 |
| 316 | N0043 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 43.9% | 44.2% | 60.0% | 18 | 23 | 41 |
| 317 | N0050 | CatBoost + Stylometry | v1.5 | claude-opus-4-6 | 43.9% | 44.2% | 60.0% | 18 | 23 | 41 |
| 318 | N0060 | Granite 4 H Micro | v0 | claude-opus-4-6 | 43.9% | 44.2% | 60.0% | 18 | 23 | 41 |
| 319 | D | UltraBERT-380M | v0 | gpt-5.6-terra | 43.8% | 44.1% | 46.7% | 14 | 18 | 32 |
| 320 | N0051 | EmbeddingGemma 2 | v0 | claude-opus-4-6 | 43.5% | 44.0% | 33.3% | 10 | 13 | 23 |
| 321 | N0030 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 43.6% | 43.9% | 56.7% | 17 | 22 | 39 |
| 322 | N0058 | Gemma 4 E2B | v1.1 | kimi-k3 | 43.3% | 43.8% | 43.3% | 13 | 17 | 30 |
| 323 | N0062 | Llama 3.2 3B | v0 | gpt-5.6-terra | 43.3% | 43.8% | 43.3% | 13 | 17 | 30 |
| 324 | N0069 | Qwen3 0.6B | v0 | gemini-3.8-flash | 43.3% | 43.8% | 43.3% | 13 | 17 | 30 |
| 325 | N0038 | CatBoost + Stylometry | v1 | glm-5.3-flash | 42.9% | 43.8% | 20.0% | 6 | 8 | 14 |
| 326 | N0076 | Qwen3 4B | v0 | gemini-3.8-flash | 42.9% | 43.8% | 20.0% | 6 | 8 | 14 |
| 327 | D | UltraBERT-380M | v0 | claude-fable-5 | 43.2% | 43.6% | 53.3% | 16 | 21 | 37 |
| 328 | N0039 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 43.2% | 43.6% | 53.3% | 16 | 21 | 37 |
| 329 | N0033 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 42.9% | 43.3% | 40.0% | 12 | 16 | 28 |
| 330 | N0054 | EmbeddingGemma 2 | v1 | gemini-3.8-flash | 42.9% | 43.3% | 40.0% | 12 | 16 | 28 |
| 331 | C | UltraBERT-380M | v1.5 | glm-5.3-flash | 42.9% | 43.2% | 50.0% | 15 | 20 | 35 |
| 332 | N0075 | Qwen3 1.7B | v1.1 | gemini-3.8-flash | 42.9% | 43.2% | 50.0% | 15 | 20 | 35 |
| 333 | N0079 | SmolLM3 3B | v1.5 | gpt-5.5 | 42.9% | 43.2% | 50.0% | 15 | 20 | 35 |
| 334 | N0079 | SmolLM3 3B | v1.5 | gpt-5.6-terra | 42.9% | 43.2% | 50.0% | 15 | 20 | 35 |
| 335 | N0028 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 42.9% | 43.2% | 60.0% | 18 | 24 | 42 |
| 336 | N0071 | Qwen3 0.6B | v1.1 | gemini-3.6-flash | 42.6% | 42.9% | 66.7% | 20 | 27 | 47 |
| 337 | N0074 | Qwen3 1.7B | v0 | gemini-3.1-pro-preview | 42.6% | 42.9% | 66.7% | 20 | 27 | 47 |
| 338 | N0062 | Llama 3.2 3B | v0 | claude-opus-4-6 | 42.5% | 42.9% | 56.7% | 17 | 23 | 40 |
| 339 | N0056 | EmbeddingGemma 2 | v1.1 | gemini-3.8-flash | 42.4% | 42.9% | 46.7% | 14 | 19 | 33 |
| 340 | N0069 | Qwen3 0.6B | v0 | claude-opus-5 | 42.4% | 42.9% | 46.7% | 14 | 19 | 33 |
| 341 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-opus-4-8 | 42.3% | 42.9% | 36.7% | 11 | 15 | 26 |
| 342 | N0079 | SmolLM3 3B | v1.5 | gemini-3.6-flash | 42.1% | 42.9% | 26.7% | 8 | 11 | 19 |
| 343 | N0069 | Qwen3 0.6B | v0 | gpt-5.6-sol | 41.7% | 42.9% | 16.7% | 5 | 7 | 12 |
| 344 | N0030 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 42.2% | 42.6% | 63.3% | 19 | 26 | 45 |
| 345 | N0071 | Qwen3 0.6B | v1.1 | gpt-5.5 | 42.2% | 42.6% | 63.3% | 19 | 26 | 45 |
| 346 | N0060 | Granite 4 H Micro | v0 | gemini-3.1-pro-preview | 42.1% | 42.5% | 53.3% | 16 | 22 | 38 |
| 347 | N0028 | CatBoost + Stylometry | v1 | gpt-6-astra | 41.9% | 42.4% | 43.3% | 13 | 18 | 31 |
| 348 | N0053 | EmbeddingGemma 2 | v0 | gemini-3.1-pro-preview | 41.9% | 42.4% | 43.3% | 13 | 18 | 31 |
| 349 | N0072 | Qwen3 0.6B | v1.5 | gpt-5.5 | 41.9% | 42.4% | 43.3% | 13 | 18 | 31 |
| 350 | N0081 | SVC | v1 | claude-sonnet-5 | 42.0% | 42.3% | 70.0% | 21 | 29 | 50 |
| 351 | N0024 | CatBoost + Stylometry | v0 | qwen3.8-max | 41.7% | 42.3% | 33.3% | 10 | 14 | 24 |
| 352 | N0068 | Qwen3 0.6B | v0 | gemini-3.8-flash | 41.7% | 42.3% | 33.3% | 10 | 14 | 24 |
| 353 | N0018 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 41.9% | 42.2% | 60.0% | 18 | 25 | 43 |
| 354 | N0025 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 41.9% | 42.2% | 60.0% | 18 | 25 | 43 |
| 355 | N0074 | Qwen3 1.7B | v0 | gemini-3.7-flash | 41.9% | 42.2% | 60.0% | 18 | 25 | 43 |
| 356 | D | UltraBERT-380M | v0 | gpt-5.5 | 41.8% | 42.1% | 76.7% | 23 | 32 | 55 |
| 357 | N0051 | EmbeddingGemma 2 | v0 | gemini-3.1-pro-preview | 41.7% | 42.1% | 50.0% | 15 | 21 | 36 |
| 358 | N0054 | EmbeddingGemma 2 | v1 | qwen3.8-max | 41.7% | 42.1% | 50.0% | 15 | 21 | 36 |
| 359 | N0083 | SVC | v1.1 | qwen3.8-max | 41.7% | 42.1% | 50.0% | 15 | 21 | 36 |
| 360 | N0043 | CatBoost + Stylometry | v1 | glm-5.3-flash | 41.2% | 42.1% | 23.3% | 7 | 10 | 17 |
| 361 | N0050 | CatBoost + Stylometry | v1.5 | glm-5.3-flash | 41.2% | 42.1% | 23.3% | 7 | 10 | 17 |
| 362 | B | UltraBERT-380M | v1_all | gpt-5.6-terra | 41.7% | 42.0% | 66.7% | 20 | 28 | 48 |
| 363 | N0027 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 41.4% | 41.9% | 40.0% | 12 | 17 | 29 |
| 364 | N0061 | Granite 4 H Micro | v1.1 | gpt-5.6-luna | 41.4% | 41.9% | 40.0% | 12 | 17 | 29 |
| 365 | N0063 | Llama 3.2 3B | v1.1 | gpt-5.6-terra | 41.4% | 41.9% | 40.0% | 12 | 17 | 29 |
| 366 | N0078 | SmolLM3 3B | v0 | gemini-3.8-flash | 41.4% | 41.9% | 40.0% | 12 | 17 | 29 |
| 367 | N0061 | Granite 4 H Micro | v1.1 | gemini-3.7-flash | 41.2% | 41.7% | 46.7% | 14 | 20 | 34 |
| 368 | N0051 | EmbeddingGemma 2 | v0 | claude-opus-5 | 40.9% | 41.7% | 30.0% | 9 | 13 | 22 |
| 369 | N0060 | Granite 4 H Micro | v0 | kimi-k3 | 40.9% | 41.7% | 30.0% | 9 | 13 | 22 |
| 370 | N0083 | SVC | v1.1 | gpt-5.6-sol | 40.9% | 41.7% | 30.0% | 9 | 13 | 22 |
| 371 | N0085 | SVC | v1.1 | gpt-5.6-sol | 40.9% | 41.7% | 30.0% | 9 | 13 | 22 |
| 372 | N0040 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 40.0% | 41.7% | 13.3% | 4 | 6 | 10 |
| 373 | N0059 | Granite 4.0 350M Base | v1.1 | gemini-3.6-flash | 40.0% | 41.7% | 13.3% | 4 | 6 | 10 |
| 374 | N0070 | Qwen3 0.6B | v1.1 | gemini-3.6-flash | 41.2% | 41.5% | 70.0% | 21 | 30 | 51 |
| 375 | N0043 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 41.0% | 41.5% | 53.3% | 16 | 23 | 39 |
| 376 | N0050 | CatBoost + Stylometry | v1.5 | claude-sonnet-5 | 41.0% | 41.5% | 53.3% | 16 | 23 | 39 |
| 377 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-6-astra | 40.7% | 41.4% | 36.7% | 11 | 16 | 27 |
| 378 | N0054 | EmbeddingGemma 2 | v1 | gemini-3.6-flash | 40.7% | 41.4% | 36.7% | 11 | 16 | 27 |
| 379 | N0065 | Ministral 3 3B | v1.1 | gpt-5.6-sol | 40.7% | 41.4% | 36.7% | 11 | 16 | 27 |
| 380 | N0081 | SVC | v1 | glm-5.3-flash | 40.7% | 41.4% | 36.7% | 11 | 16 | 27 |
| 381 | N0052 | EmbeddingGemma 2 | v0 | claude-opus-4-6 | 40.9% | 41.3% | 60.0% | 18 | 26 | 44 |
| 382 | N0060 | Granite 4 H Micro | v0 | claude-sonnet-5 | 40.6% | 41.2% | 43.3% | 13 | 19 | 32 |
| 383 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.6-luna | 40.0% | 41.2% | 20.0% | 6 | 9 | 15 |
| 384 | N0066 | Qwen3 0.6B | v0 | claude-opus-4-8 | 40.0% | 41.2% | 20.0% | 6 | 9 | 15 |
| 385 | N0069 | Qwen3 0.6B | v0 | claude-sonnet-5 | 40.0% | 41.2% | 20.0% | 6 | 9 | 15 |
| 386 | N0077 | Qwen3 4B | v0 | gpt-6-astra | 40.0% | 41.2% | 20.0% | 6 | 9 | 15 |
| 387 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-opus-5 | 40.5% | 41.0% | 50.0% | 15 | 22 | 37 |
| 388 | N0028 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 40.5% | 41.0% | 50.0% | 15 | 22 | 37 |
| 389 | N0030 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 40.5% | 41.0% | 50.0% | 15 | 22 | 37 |
| 390 | N0075 | Qwen3 1.7B | v1.1 | gpt-5.5 | 40.5% | 41.0% | 50.0% | 15 | 22 | 37 |
| 391 | N0085 | SVC | v1.1 | qwen3.8-max | 40.5% | 41.0% | 50.0% | 15 | 22 | 37 |
| 392 | N0070 | Qwen3 0.6B | v1.1 | kimi-k3 | 40.0% | 40.9% | 26.7% | 8 | 12 | 20 |
| 393 | N0017 | CatBoost + Stylometry | v0 | claude-opus-5 | 40.4% | 40.7% | 70.0% | 21 | 31 | 52 |
| 394 | N0017 | CatBoost + Stylometry | v0 | gpt-6-astra | 40.4% | 40.7% | 70.0% | 21 | 31 | 52 |
| 395 | N0043 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 396 | N0050 | CatBoost + Stylometry | v1.5 | gemini-3.6-flash | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 397 | N0063 | Llama 3.2 3B | v1.1 | kimi-k3 | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 398 | N0078 | SmolLM3 3B | v0 | gpt-6-astra | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 399 | N0081 | SVC | v1 | gpt-5.5 | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 400 | N0083 | SVC | v1.1 | gpt-5.5 | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 401 | N0085 | SVC | v1.1 | gpt-5.5 | 40.0% | 40.7% | 33.3% | 10 | 15 | 25 |
| 402 | C | UltraBERT-380M | v1.5 | gpt-5.6-luna | 40.0% | 40.6% | 40.0% | 12 | 18 | 30 |
| 403 | N0042 | CatBoost + Stylometry | v1 | gpt-6-astra | 40.0% | 40.6% | 40.0% | 12 | 18 | 30 |
| 404 | N0081 | SVC | v1 | claude-opus-4-8 | 40.0% | 40.6% | 40.0% | 12 | 18 | 30 |
| 405 | B | UltraBERT-380M | v1_all | gpt-5.6-sol | 40.0% | 40.5% | 46.7% | 14 | 21 | 35 |
| 406 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.8-flash | 40.0% | 40.5% | 46.7% | 14 | 21 | 35 |
| 407 | C | UltraBERT-380M | v1.5 | gpt-5.6-terra | 40.0% | 40.5% | 53.3% | 16 | 24 | 40 |
| 408 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-opus-5 | 40.0% | 40.5% | 53.3% | 16 | 24 | 40 |
| 409 | N0043 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 40.0% | 40.5% | 53.3% | 16 | 24 | 40 |
| 410 | N0050 | CatBoost + Stylometry | v1.5 | gemini-3.8-flash | 40.0% | 40.5% | 53.3% | 16 | 24 | 40 |
| 411 | C | UltraBERT-380M | v1.5 | gemini-3.7-flash | 40.0% | 40.4% | 73.3% | 22 | 33 | 55 |
| 412 | N0065 | Ministral 3 3B | v1.1 | gpt-5.6-terra | 39.6% | 40.0% | 70.0% | 21 | 32 | 53 |
| 413 | N0063 | Llama 3.2 3B | v1.1 | qwen3.8-max | 39.6% | 40.0% | 63.3% | 19 | 29 | 48 |
| 414 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.8-flash | 39.5% | 40.0% | 50.0% | 15 | 23 | 38 |
| 415 | N0015 | CatBoost + Stylometry | v0 | gpt-6-astra | 39.5% | 40.0% | 50.0% | 15 | 23 | 38 |
| 416 | N0061 | Granite 4 H Micro | v1.1 | gpt-5.6-terra | 39.5% | 40.0% | 50.0% | 15 | 23 | 38 |
| 417 | N0076 | Qwen3 4B | v0 | claude-opus-5 | 39.5% | 40.0% | 50.0% | 15 | 23 | 38 |
| 418 | N0084 | SVC | v1.1 | qwen3.8-max | 39.5% | 40.0% | 50.0% | 15 | 23 | 38 |
| 419 | N0015 | CatBoost + Stylometry | v0 | qwen3.8-max | 39.4% | 40.0% | 43.3% | 13 | 20 | 33 |
| 420 | N0042 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 39.4% | 40.0% | 43.3% | 13 | 20 | 33 |
| 421 | A | UltraBERT-380M | v1 | gpt-5.6-luna | 39.3% | 40.0% | 36.7% | 11 | 17 | 28 |
| 422 | N0031 | CatBoost + Stylometry | v1 | gpt-6-astra | 39.3% | 40.0% | 36.7% | 11 | 17 | 28 |
| 423 | N0046 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 39.1% | 40.0% | 30.0% | 9 | 14 | 23 |
| 424 | N0014 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 38.9% | 40.0% | 23.3% | 7 | 11 | 18 |
| 425 | N0030 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 38.9% | 40.0% | 23.3% | 7 | 11 | 18 |
| 426 | N0057 | EmbeddingGemma 2 | v1.5 | glm-5.3-flash | 38.9% | 40.0% | 23.3% | 7 | 11 | 18 |
| 427 | N0054 | EmbeddingGemma 2 | v1 | gpt-5.6-luna | 38.5% | 40.0% | 16.7% | 5 | 8 | 13 |
| 428 | N0011 | CatBoost + Stylometry | v0 | claude-opus-5 | 39.1% | 39.6% | 60.0% | 18 | 28 | 46 |
| 429 | N0042 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 39.1% | 39.6% | 60.0% | 18 | 28 | 46 |
| 430 | N0077 | Qwen3 4B | v0 | gemini-3.1-pro-preview | 39.0% | 39.5% | 53.3% | 16 | 25 | 41 |
| 431 | N0067 | Qwen3 0.6B | v0 | gpt-6-astra | 38.9% | 39.5% | 46.7% | 14 | 22 | 36 |
| 432 | N0027 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 38.7% | 39.4% | 40.0% | 12 | 19 | 31 |
| 433 | N0046 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 38.7% | 39.4% | 40.0% | 12 | 19 | 31 |
| 434 | N0063 | Llama 3.2 3B | v1.1 | gpt-5.6-luna | 38.5% | 39.3% | 33.3% | 10 | 16 | 26 |
| 435 | N0067 | Qwen3 0.6B | v0 | qwen3.8-max | 38.5% | 39.3% | 33.3% | 10 | 16 | 26 |
| 436 | N0014 | CatBoost + Stylometry | v0 | claude-opus-5 | 38.8% | 39.2% | 63.3% | 19 | 30 | 49 |
| 437 | N0034 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 38.1% | 39.1% | 26.7% | 8 | 13 | 21 |
| 438 | N0064 | Ministral 3 3B | v0 | gpt-5.5 | 38.1% | 39.1% | 26.7% | 8 | 13 | 21 |
| 439 | N0084 | SVC | v1.1 | gpt-5.6-sol | 38.1% | 39.1% | 26.7% | 8 | 13 | 21 |
| 440 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-opus-5 | 38.5% | 39.0% | 50.0% | 15 | 24 | 39 |
| 441 | N0060 | Granite 4 H Micro | v0 | gpt-6-astra | 38.5% | 39.0% | 50.0% | 15 | 24 | 39 |
| 442 | N0076 | Qwen3 4B | v0 | gemini-3.1-pro-preview | 38.5% | 38.9% | 66.7% | 20 | 32 | 52 |
| 443 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.8-flash | 38.2% | 38.9% | 43.3% | 13 | 21 | 34 |
| 444 | N0012 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 38.2% | 38.9% | 43.3% | 13 | 21 | 34 |
| 445 | N0013 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 38.2% | 38.9% | 43.3% | 13 | 21 | 34 |
| 446 | N0021 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 38.2% | 38.9% | 43.3% | 13 | 21 | 34 |
| 447 | N0023 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 38.2% | 38.9% | 43.3% | 13 | 21 | 34 |
| 448 | N0026 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 38.2% | 38.9% | 43.3% | 13 | 21 | 34 |
| 449 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-opus-4-8 | 37.5% | 38.9% | 20.0% | 6 | 10 | 16 |
| 450 | N0010 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 37.5% | 38.9% | 20.0% | 6 | 10 | 16 |
| 451 | N0027 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 37.5% | 38.9% | 20.0% | 6 | 10 | 16 |
| 452 | N0067 | Qwen3 0.6B | v0 | gemini-3.7-flash | 37.5% | 38.9% | 20.0% | 6 | 10 | 16 |
| 453 | N0069 | Qwen3 0.6B | v0 | gpt-5.6-luna | 37.5% | 38.9% | 20.0% | 6 | 10 | 16 |
| 454 | N0070 | Qwen3 0.6B | v1.1 | gemini-3.7-flash | 37.5% | 38.9% | 20.0% | 6 | 10 | 16 |
| 455 | C | UltraBERT-380M | v1.5 | gpt-5.6-sol | 37.9% | 38.7% | 36.7% | 11 | 18 | 29 |
| 456 | N0034 | CatBoost + Stylometry | v1 | gpt-6-astra | 37.9% | 38.7% | 36.7% | 11 | 18 | 29 |
| 457 | N0053 | EmbeddingGemma 2 | v0 | qwen3.8-max | 37.9% | 38.7% | 36.7% | 11 | 18 | 29 |
| 458 | N0059 | Granite 4.0 350M Base | v1.1 | gpt-5.5 | 37.9% | 38.7% | 36.7% | 11 | 18 | 29 |
| 459 | N0079 | SmolLM3 3B | v1.5 | kimi-k3 | 37.9% | 38.7% | 36.7% | 11 | 18 | 29 |
| 460 | N0082 | SVC | v1.1 | gpt-5.5 | 37.9% | 38.7% | 36.7% | 11 | 18 | 29 |
| 461 | N0012 | CatBoost + Stylometry | v0 | gpt-6-astra | 38.1% | 38.6% | 53.3% | 16 | 26 | 42 |
| 462 | N0013 | CatBoost + Stylometry | v0 | gpt-6-astra | 38.1% | 38.6% | 53.3% | 16 | 26 | 42 |
| 463 | N0021 | CatBoost + Stylometry | v0 | gpt-6-astra | 38.1% | 38.6% | 53.3% | 16 | 26 | 42 |
| 464 | N0023 | CatBoost + Stylometry | v0 | gpt-6-astra | 38.1% | 38.6% | 53.3% | 16 | 26 | 42 |
| 465 | N0026 | CatBoost + Stylometry | v0 | gpt-6-astra | 38.1% | 38.6% | 53.3% | 16 | 26 | 42 |
| 466 | N0071 | Qwen3 0.6B | v1.1 | gpt-5.6-terra | 38.1% | 38.6% | 53.3% | 16 | 26 | 42 |
| 467 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-sonnet-5 | 37.8% | 38.5% | 46.7% | 14 | 23 | 37 |
| 468 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-sonnet-5 | 37.8% | 38.5% | 46.7% | 14 | 23 | 37 |
| 469 | N0030 | CatBoost + Stylometry | v1 | claude-opus-5 | 37.8% | 38.5% | 46.7% | 14 | 23 | 37 |
| 470 | N0062 | Llama 3.2 3B | v0 | gemini-3.8-flash | 37.8% | 38.5% | 46.7% | 14 | 23 | 37 |
| 471 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-opus-4-8 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 472 | N0017 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 473 | N0032 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 474 | N0033 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 475 | N0040 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 476 | N0041 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 477 | N0044 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 478 | N0047 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 479 | N0049 | CatBoost + Stylometry | v1.5 | claude-opus-4-8 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 480 | N0070 | Qwen3 0.6B | v1.1 | gpt-5.6-luna | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 481 | N0084 | SVC | v1.1 | gpt-5.5 | 37.5% | 38.5% | 30.0% | 9 | 15 | 24 |
| 482 | N0048 | CatBoost + Stylometry | v1.1 | glm-5.3-flash | 36.4% | 38.5% | 13.3% | 4 | 7 | 11 |
| 483 | N0032 | CatBoost + Stylometry | v1 | gpt-6-astra | 37.5% | 38.2% | 40.0% | 12 | 20 | 32 |
| 484 | N0041 | CatBoost + Stylometry | v1 | gpt-6-astra | 37.5% | 38.2% | 40.0% | 12 | 20 | 32 |
| 485 | N0044 | CatBoost + Stylometry | v1 | gpt-6-astra | 37.5% | 38.2% | 40.0% | 12 | 20 | 32 |
| 486 | N0047 | CatBoost + Stylometry | v1 | gpt-6-astra | 37.5% | 38.2% | 40.0% | 12 | 20 | 32 |
| 487 | N0049 | CatBoost + Stylometry | v1.5 | gpt-6-astra | 37.5% | 38.2% | 40.0% | 12 | 20 | 32 |
| 488 | N0072 | Qwen3 0.6B | v1.5 | gemini-3.7-flash | 37.5% | 38.1% | 50.0% | 15 | 25 | 40 |
| 489 | N0079 | SmolLM3 3B | v1.5 | gemini-3.7-flash | 37.5% | 38.1% | 50.0% | 15 | 25 | 40 |
| 490 | N0012 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 491 | N0013 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 492 | N0018 | CatBoost + Stylometry | v0 | glm-5.3 | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 493 | N0020 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 494 | N0021 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 495 | N0023 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 496 | N0026 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 497 | N0071 | Qwen3 0.6B | v1.1 | gemini-3.7-flash | 36.8% | 38.1% | 23.3% | 7 | 12 | 19 |
| 498 | N0038 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 37.0% | 37.9% | 33.3% | 10 | 17 | 27 |
| 499 | N0039 | CatBoost + Stylometry | v1 | gpt-6-astra | 37.0% | 37.9% | 33.3% | 10 | 17 | 27 |
| 500 | N0072 | Qwen3 0.6B | v1.5 | kimi-k3 | 37.0% | 37.9% | 33.3% | 10 | 17 | 27 |
| 501 | N0029 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 37.1% | 37.8% | 43.3% | 13 | 22 | 35 |
| 502 | N0063 | Llama 3.2 3B | v1.1 | glm-5.3-flash | 37.1% | 37.8% | 43.3% | 13 | 22 | 35 |
| 503 | N0036 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 37.2% | 37.8% | 53.3% | 16 | 27 | 43 |
| 504 | N0015 | CatBoost + Stylometry | v0 | claude-opus-5 | 37.3% | 37.7% | 63.3% | 19 | 32 | 51 |
| 505 | N0019 | CatBoost + Stylometry | v0 | claude-opus-5 | 37.0% | 37.5% | 66.7% | 20 | 34 | 54 |
| 506 | N0036 | CatBoost + Stylometry | v1 | claude-opus-5 | 37.0% | 37.5% | 56.7% | 17 | 29 | 46 |
| 507 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-sonnet-5 | 36.8% | 37.5% | 46.7% | 14 | 24 | 38 |
| 508 | N0075 | Qwen3 1.7B | v1.1 | gemini-3.7-flash | 36.8% | 37.5% | 46.7% | 14 | 24 | 38 |
| 509 | C | UltraBERT-380M | v1.5 | kimi-k3 | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 510 | N0015 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 511 | N0043 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 512 | N0043 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 513 | N0046 | CatBoost + Stylometry | v1 | gpt-6-astra | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 514 | N0048 | CatBoost + Stylometry | v1.1 | gpt-6-astra | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 515 | N0050 | CatBoost + Stylometry | v1.5 | claude-opus-4-8 | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 516 | N0050 | CatBoost + Stylometry | v1.5 | gemini-3.7-flash | 36.7% | 37.5% | 36.7% | 11 | 19 | 30 |
| 517 | N0025 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 36.4% | 37.5% | 26.7% | 8 | 14 | 22 |
| 518 | N0082 | SVC | v1.1 | gpt-5.6-sol | 36.4% | 37.5% | 26.7% | 8 | 14 | 22 |
| 519 | N0030 | CatBoost + Stylometry | v1 | glm-5.3-flash | 35.7% | 37.5% | 16.7% | 5 | 9 | 14 |
| 520 | N0024 | CatBoost + Stylometry | v0 | claude-opus-5 | 36.8% | 37.3% | 70.0% | 21 | 36 | 57 |
| 521 | N0028 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 36.6% | 37.2% | 50.0% | 15 | 26 | 41 |
| 522 | N0029 | CatBoost + Stylometry | v1 | claude-opus-5 | 36.6% | 37.2% | 50.0% | 15 | 26 | 41 |
| 523 | N0031 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 36.4% | 37.1% | 40.0% | 12 | 21 | 33 |
| 524 | N0058 | Gemma 4 E2B | v1.1 | gpt-5.6-terra | 36.5% | 37.0% | 63.3% | 19 | 33 | 52 |
| 525 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-opus-4-8 | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 526 | N0010 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 527 | N0011 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 528 | N0012 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 529 | N0013 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 530 | N0021 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 531 | N0023 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 532 | N0026 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 533 | N0042 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 534 | N0055 | EmbeddingGemma 2 | v1.1 | gpt-5.5 | 36.0% | 37.0% | 30.0% | 9 | 16 | 25 |
| 535 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-sonnet-5 | 36.4% | 37.0% | 53.3% | 16 | 28 | 44 |
| 536 | N0064 | Ministral 3 3B | v0 | gemini-3.7-flash | 36.4% | 37.0% | 53.3% | 16 | 28 | 44 |
| 537 | N0065 | Ministral 3 3B | v1.1 | claude-fable-5 | 36.4% | 37.0% | 53.3% | 16 | 28 | 44 |
| 538 | N0020 | CatBoost + Stylometry | v0 | claude-opus-5 | 36.4% | 36.8% | 66.7% | 20 | 35 | 55 |
| 539 | N0017 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 35.3% | 36.8% | 20.0% | 6 | 11 | 17 |
| 540 | N0024 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 35.3% | 36.8% | 20.0% | 6 | 11 | 17 |
| 541 | N0061 | Granite 4 H Micro | v1.1 | gpt-5.6-sol | 35.3% | 36.8% | 20.0% | 6 | 11 | 17 |
| 542 | N0066 | Qwen3 0.6B | v0 | kimi-k3 | 35.3% | 36.8% | 20.0% | 6 | 11 | 17 |
| 543 | N0079 | SmolLM3 3B | v1.5 | gpt-5.6-luna | 35.3% | 36.8% | 20.0% | 6 | 11 | 17 |
| 544 | N0020 | CatBoost + Stylometry | v0 | gpt-6-astra | 36.2% | 36.7% | 56.7% | 17 | 30 | 47 |
| 545 | N0024 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 35.7% | 36.7% | 33.3% | 10 | 18 | 28 |
| 546 | N0025 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 35.7% | 36.7% | 33.3% | 10 | 18 | 28 |
| 547 | N0034 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 35.7% | 36.7% | 33.3% | 10 | 18 | 28 |
| 548 | N0060 | Granite 4 H Micro | v0 | gemini-3.8-flash | 35.7% | 36.7% | 33.3% | 10 | 18 | 28 |
| 549 | A | UltraBERT-380M | v1 | gemini-3.7-flash | 35.9% | 36.6% | 46.7% | 14 | 25 | 39 |
| 550 | N0066 | Qwen3 0.6B | v0 | claude-opus-5 | 36.1% | 36.5% | 73.3% | 22 | 39 | 61 |
| 551 | N0012 | CatBoost + Stylometry | v0 | claude-opus-5 | 35.8% | 36.4% | 63.3% | 19 | 34 | 53 |
| 552 | N0013 | CatBoost + Stylometry | v0 | claude-opus-5 | 35.8% | 36.4% | 63.3% | 19 | 34 | 53 |
| 553 | N0021 | CatBoost + Stylometry | v0 | claude-opus-5 | 35.8% | 36.4% | 63.3% | 19 | 34 | 53 |
| 554 | N0023 | CatBoost + Stylometry | v0 | claude-opus-5 | 35.8% | 36.4% | 63.3% | 19 | 34 | 53 |
| 555 | N0026 | CatBoost + Stylometry | v0 | claude-opus-5 | 35.8% | 36.4% | 63.3% | 19 | 34 | 53 |
| 556 | N0031 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 35.7% | 36.4% | 50.0% | 15 | 27 | 42 |
| 557 | N0011 | CatBoost + Stylometry | v0 | qwen3.8-max | 35.5% | 36.4% | 36.7% | 11 | 20 | 31 |
| 558 | N0033 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 35.5% | 36.4% | 36.7% | 11 | 20 | 31 |
| 559 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.6-luna | 35.0% | 36.4% | 23.3% | 7 | 13 | 20 |
| 560 | N0024 | CatBoost + Stylometry | v0 | gpt-6-astra | 35.7% | 36.2% | 66.7% | 20 | 36 | 56 |
| 561 | N0075 | Qwen3 1.7B | v1.1 | qwen3.8-max | 35.7% | 36.2% | 66.7% | 20 | 36 | 56 |
| 562 | N0082 | SVC | v1.1 | qwen3.8-max | 35.6% | 36.2% | 53.3% | 16 | 29 | 45 |
| 563 | N0011 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 35.3% | 36.1% | 40.0% | 12 | 22 | 34 |
| 564 | N0039 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 35.3% | 36.1% | 40.0% | 12 | 22 | 34 |
| 565 | N0066 | Qwen3 0.6B | v0 | qwen3.8-max | 35.3% | 36.1% | 40.0% | 12 | 22 | 34 |
| 566 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-6-astra | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 567 | N0012 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 568 | N0013 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 569 | N0015 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 570 | N0021 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 571 | N0022 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 572 | N0023 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 573 | N0026 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 574 | N0053 | EmbeddingGemma 2 | v0 | claude-opus-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 575 | N0062 | Llama 3.2 3B | v0 | claude-opus-4-8 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 576 | N0068 | Qwen3 0.6B | v0 | claude-sonnet-5 | 34.8% | 36.0% | 26.7% | 8 | 15 | 23 |
| 577 | N0014 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 35.1% | 35.9% | 43.3% | 13 | 24 | 37 |
| 578 | N0027 | CatBoost + Stylometry | v0 | gpt-5.5 | 35.1% | 35.9% | 43.3% | 13 | 24 | 37 |
| 579 | N0034 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 35.1% | 35.9% | 43.3% | 13 | 24 | 37 |
| 580 | N0079 | SmolLM3 3B | v1.5 | gpt-5.6-sol | 35.1% | 35.9% | 43.3% | 13 | 24 | 37 |
| 581 | N0010 | CatBoost + Stylometry | v0 | gpt-6-astra | 35.3% | 35.8% | 60.0% | 18 | 33 | 51 |
| 582 | N0075 | Qwen3 1.7B | v1.1 | glm-5.3-flash | 35.3% | 35.8% | 60.0% | 18 | 33 | 51 |
| 583 | N0060 | Granite 4 H Micro | v0 | claude-opus-5 | 35.2% | 35.7% | 63.3% | 19 | 35 | 54 |
| 584 | N0027 | CatBoost + Stylometry | v0 | claude-opus-5 | 35.0% | 35.7% | 46.7% | 14 | 26 | 40 |
| 585 | N0040 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 35.0% | 35.7% | 46.7% | 14 | 26 | 40 |
| 586 | N0045 | CatBoost + Stylometry | v1 | claude-opus-5 | 35.0% | 35.7% | 46.7% | 14 | 26 | 40 |
| 587 | N0066 | Qwen3 0.6B | v0 | claude-sonnet-5 | 35.0% | 35.7% | 46.7% | 14 | 26 | 40 |
| 588 | N0070 | Qwen3 0.6B | v1.1 | gpt-5.6-terra | 35.0% | 35.7% | 46.7% | 14 | 26 | 40 |
| 589 | D | UltraBERT-380M | v0 | glm-5.3-flash | 34.6% | 35.7% | 30.0% | 9 | 17 | 26 |
| 590 | N0011 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.6% | 35.7% | 30.0% | 9 | 17 | 26 |
| 591 | N0019 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 34.6% | 35.7% | 30.0% | 9 | 17 | 26 |
| 592 | N0076 | Qwen3 4B | v0 | gpt-6-astra | 34.6% | 35.7% | 30.0% | 9 | 17 | 26 |
| 593 | N0031 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 33.3% | 35.7% | 13.3% | 4 | 8 | 12 |
| 594 | N0025 | CatBoost + Stylometry | v0 | gpt-6-astra | 35.1% | 35.6% | 66.7% | 20 | 37 | 57 |
| 595 | N0031 | CatBoost + Stylometry | v1 | claude-opus-5 | 34.9% | 35.6% | 50.0% | 15 | 28 | 43 |
| 596 | N0012 | CatBoost + Stylometry | v0 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 597 | N0013 | CatBoost + Stylometry | v0 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 598 | N0019 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 599 | N0021 | CatBoost + Stylometry | v0 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 600 | N0023 | CatBoost + Stylometry | v0 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 601 | N0026 | CatBoost + Stylometry | v0 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 602 | N0055 | EmbeddingGemma 2 | v1.1 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 603 | N0060 | Granite 4 H Micro | v0 | qwen3.8-max | 34.5% | 35.5% | 33.3% | 10 | 19 | 29 |
| 604 | N0011 | CatBoost + Stylometry | v0 | gpt-6-astra | 34.8% | 35.4% | 53.3% | 16 | 30 | 46 |
| 605 | N0058 | Gemma 4 E2B | v1.1 | claude-fable-5 | 34.8% | 35.4% | 53.3% | 16 | 30 | 46 |
| 606 | N0019 | CatBoost + Stylometry | v0 | gpt-6-astra | 34.7% | 35.3% | 56.7% | 17 | 32 | 49 |
| 607 | D | UltraBERT-380M | v0 | gpt-5.6-luna | 34.4% | 35.3% | 36.7% | 11 | 21 | 32 |
| 608 | N0010 | CatBoost + Stylometry | v0 | glm-5.3-flash | 34.4% | 35.3% | 36.7% | 11 | 21 | 32 |
| 609 | B | UltraBERT-380M | v1_all | gpt-5.6-luna | 33.3% | 35.3% | 16.7% | 5 | 10 | 15 |
| 610 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.6-sol | 33.3% | 35.3% | 16.7% | 5 | 10 | 15 |
| 611 | N0028 | CatBoost + Stylometry | v1 | glm-5.3 | 33.3% | 35.3% | 16.7% | 5 | 10 | 15 |
| 612 | N0037 | CatBoost + Stylometry | v1 | glm-5.3-flash | 33.3% | 35.3% | 16.7% | 5 | 10 | 15 |
| 613 | N0040 | CatBoost + Stylometry | v1 | glm-5.3-flash | 33.3% | 35.3% | 16.7% | 5 | 10 | 15 |
| 614 | N0046 | CatBoost + Stylometry | v1 | glm-5.3-flash | 33.3% | 35.3% | 16.7% | 5 | 10 | 15 |
| 615 | N0031 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 34.3% | 35.1% | 40.0% | 12 | 23 | 35 |
| 616 | N0077 | Qwen3 4B | v0 | claude-sonnet-5 | 34.3% | 35.1% | 40.0% | 12 | 23 | 35 |
| 617 | N0080 | SVC | v0 | gemini-3.6-flash | 34.2% | 35.0% | 43.3% | 13 | 25 | 38 |
| 618 | N0017 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 33.3% | 35.0% | 20.0% | 6 | 12 | 18 |
| 619 | N0019 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 33.3% | 35.0% | 20.0% | 6 | 12 | 18 |
| 620 | N0034 | CatBoost + Stylometry | v1 | glm-5.3-flash | 33.3% | 35.0% | 20.0% | 6 | 12 | 18 |
| 621 | N0043 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 33.3% | 35.0% | 20.0% | 6 | 12 | 18 |
| 622 | N0050 | CatBoost + Stylometry | v1.5 | gpt-5.6-luna | 33.3% | 35.0% | 20.0% | 6 | 12 | 18 |
| 623 | N0076 | Qwen3 4B | v0 | gpt-5.6-luna | 33.3% | 35.0% | 20.0% | 6 | 12 | 18 |
| 624 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-sonnet-5 | 34.1% | 34.9% | 46.7% | 14 | 27 | 41 |
| 625 | N0031 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 34.1% | 34.9% | 46.7% | 14 | 27 | 41 |
| 626 | N0045 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 34.1% | 34.9% | 46.7% | 14 | 27 | 41 |
| 627 | N0055 | EmbeddingGemma 2 | v1.1 | gemini-3.8-flash | 34.1% | 34.9% | 46.7% | 14 | 27 | 41 |
| 628 | N0078 | SmolLM3 3B | v0 | gemini-3.7-flash | 34.1% | 34.9% | 46.7% | 14 | 27 | 41 |
| 629 | N0014 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 630 | N0019 | CatBoost + Stylometry | v0 | glm-5.3 | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 631 | N0029 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 632 | N0048 | CatBoost + Stylometry | v1.1 | claude-opus-4-8 | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 633 | N0055 | EmbeddingGemma 2 | v1.1 | gpt-5.6-terra | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 634 | N0067 | Qwen3 0.6B | v0 | claude-sonnet-5 | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 635 | N0078 | SmolLM3 3B | v0 | claude-opus-4-8 | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 636 | N0078 | SmolLM3 3B | v0 | glm-5.3 | 33.3% | 34.8% | 23.3% | 7 | 14 | 21 |
| 637 | N0028 | CatBoost + Stylometry | v1 | claude-opus-5 | 34.0% | 34.7% | 53.3% | 16 | 31 | 47 |
| 638 | N0012 | CatBoost + Stylometry | v0 | glm-5.3 | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 639 | N0013 | CatBoost + Stylometry | v0 | glm-5.3 | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 640 | N0021 | CatBoost + Stylometry | v0 | glm-5.3 | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 641 | N0023 | CatBoost + Stylometry | v0 | glm-5.3 | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 642 | N0026 | CatBoost + Stylometry | v0 | glm-5.3 | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 643 | N0074 | Qwen3 1.7B | v0 | qwen3.8-max | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 644 | N0075 | Qwen3 1.7B | v1.1 | gpt-5.6-terra | 33.3% | 34.6% | 26.7% | 8 | 16 | 24 |
| 645 | N0029 | CatBoost + Stylometry | v1 | gpt-6-astra | 33.3% | 34.5% | 30.0% | 9 | 18 | 27 |
| 646 | N0040 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 33.3% | 34.5% | 30.0% | 9 | 18 | 27 |
| 647 | N0040 | CatBoost + Stylometry | v1 | gpt-6-astra | 33.3% | 34.5% | 30.0% | 9 | 18 | 27 |
| 648 | N0017 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 33.3% | 34.4% | 33.3% | 10 | 20 | 30 |
| 649 | N0028 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 33.3% | 34.4% | 33.3% | 10 | 20 | 30 |
| 650 | N0036 | CatBoost + Stylometry | v1 | gpt-6-astra | 33.3% | 34.4% | 33.3% | 10 | 20 | 30 |
| 651 | N0056 | EmbeddingGemma 2 | v1.1 | gpt-5.6-terra | 33.3% | 34.4% | 33.3% | 10 | 20 | 30 |
| 652 | N0061 | Granite 4 H Micro | v1.1 | claude-fable-5 | 33.3% | 34.4% | 33.3% | 10 | 20 | 30 |
| 653 | N0076 | Qwen3 4B | v0 | claude-sonnet-5 | 33.3% | 34.4% | 33.3% | 10 | 20 | 30 |
| 654 | N0020 | CatBoost + Stylometry | v0 | qwen3.8-max | 33.3% | 34.3% | 36.7% | 11 | 22 | 33 |
| 655 | N0022 | CatBoost + Stylometry | v0 | qwen3.8-max | 33.3% | 34.3% | 36.7% | 11 | 22 | 33 |
| 656 | N0027 | CatBoost + Stylometry | v0 | qwen3.8-max | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 657 | N0032 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 658 | N0041 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 659 | N0044 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 660 | N0047 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 661 | N0049 | CatBoost + Stylometry | v1.5 | gemini-3.8-flash | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 662 | N0054 | EmbeddingGemma 2 | v1 | glm-5.3 | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 663 | N0063 | Llama 3.2 3B | v1.1 | claude-fable-5 | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 664 | N0064 | Ministral 3 3B | v0 | kimi-k3 | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 665 | N0079 | SmolLM3 3B | v1.5 | claude-fable-5 | 33.3% | 34.2% | 40.0% | 12 | 24 | 36 |
| 666 | N0018 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 33.3% | 34.1% | 43.3% | 13 | 26 | 39 |
| 667 | N0048 | CatBoost + Stylometry | v1.1 | claude-sonnet-5 | 33.3% | 34.1% | 46.7% | 14 | 28 | 42 |
| 668 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-opus-4-6 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 669 | N0032 | CatBoost + Stylometry | v1 | claude-opus-5 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 670 | N0039 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 671 | N0041 | CatBoost + Stylometry | v1 | claude-opus-5 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 672 | N0044 | CatBoost + Stylometry | v1 | claude-opus-5 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 673 | N0047 | CatBoost + Stylometry | v1 | claude-opus-5 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 674 | N0049 | CatBoost + Stylometry | v1.5 | claude-opus-5 | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 675 | N0066 | Qwen3 0.6B | v0 | gpt-5.6-luna | 33.3% | 34.0% | 50.0% | 15 | 30 | 45 |
| 676 | N0043 | CatBoost + Stylometry | v1 | claude-opus-5 | 33.3% | 34.0% | 53.3% | 16 | 32 | 48 |
| 677 | N0050 | CatBoost + Stylometry | v1.5 | claude-opus-5 | 33.3% | 34.0% | 53.3% | 16 | 32 | 48 |
| 678 | N0016 | CatBoost + Stylometry | v0 | gemini-3.1-pro-preview | 33.3% | 34.0% | 56.7% | 17 | 34 | 51 |
| 679 | N0066 | Qwen3 0.6B | v0 | gemini-3.7-flash | 33.3% | 34.0% | 56.7% | 17 | 34 | 51 |
| 680 | N0033 | CatBoost + Stylometry | v1 | claude-opus-5 | 33.3% | 33.9% | 60.0% | 18 | 36 | 54 |
| 681 | N0073 | Qwen3 1.7B | v0 | gemini-3.7-flash | 33.3% | 33.8% | 70.0% | 21 | 42 | 63 |
| 682 | N0059 | Granite 4.0 350M Base | v1.1 | gemini-3.7-flash | 32.8% | 33.3% | 63.3% | 19 | 39 | 58 |
| 683 | N0081 | SVC | v1 | qwen3.8-max | 32.7% | 33.3% | 60.0% | 18 | 37 | 55 |
| 684 | B | UltraBERT-380M | v1_all | glm-5.3-flash | 32.7% | 33.3% | 53.3% | 16 | 33 | 49 |
| 685 | N0034 | CatBoost + Stylometry | v1 | claude-opus-5 | 32.7% | 33.3% | 53.3% | 16 | 33 | 49 |
| 686 | N0037 | CatBoost + Stylometry | v1 | qwen3.8-max | 32.7% | 33.3% | 53.3% | 16 | 33 | 49 |
| 687 | N0046 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 32.6% | 33.3% | 46.7% | 14 | 29 | 43 |
| 688 | N0053 | EmbeddingGemma 2 | v0 | claude-opus-4-6 | 32.6% | 33.3% | 46.7% | 14 | 29 | 43 |
| 689 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-sonnet-5 | 32.5% | 33.3% | 43.3% | 13 | 27 | 40 |
| 690 | N0054 | EmbeddingGemma 2 | v1 | gemini-3.7-flash | 32.5% | 33.3% | 43.3% | 13 | 27 | 40 |
| 691 | N0029 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 32.4% | 33.3% | 40.0% | 12 | 25 | 37 |
| 692 | N0014 | CatBoost + Stylometry | v0 | qwen3.8-max | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 693 | N0017 | CatBoost + Stylometry | v0 | qwen3.8-max | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 694 | N0032 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 695 | N0041 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 696 | N0042 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 697 | N0044 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 698 | N0047 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 699 | N0049 | CatBoost + Stylometry | v1.5 | claude-sonnet-5 | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 700 | N0070 | Qwen3 0.6B | v1.1 | glm-5.3-flash | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 701 | N0078 | SmolLM3 3B | v0 | qwen3.8-max | 32.4% | 33.3% | 36.7% | 11 | 23 | 34 |
| 702 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.7-flash | 32.3% | 33.3% | 33.3% | 10 | 21 | 31 |
| 703 | N0010 | CatBoost + Stylometry | v0 | qwen3.8-max | 32.3% | 33.3% | 33.3% | 10 | 21 | 31 |
| 704 | N0018 | CatBoost + Stylometry | v0 | qwen3.8-max | 32.3% | 33.3% | 33.3% | 10 | 21 | 31 |
| 705 | N0027 | CatBoost + Stylometry | v0 | glm-5.3-flash | 32.3% | 33.3% | 33.3% | 10 | 21 | 31 |
| 706 | N0072 | Qwen3 0.6B | v1.5 | glm-5.3-flash | 32.3% | 33.3% | 33.3% | 10 | 21 | 31 |
| 707 | N0076 | Qwen3 4B | v0 | kimi-k3 | 32.3% | 33.3% | 33.3% | 10 | 21 | 31 |
| 708 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.6-flash | 32.1% | 33.3% | 30.0% | 9 | 19 | 28 |
| 709 | N0029 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 32.1% | 33.3% | 30.0% | 9 | 19 | 28 |
| 710 | N0045 | CatBoost + Stylometry | v1 | gpt-6-astra | 32.1% | 33.3% | 30.0% | 9 | 19 | 28 |
| 711 | N0077 | Qwen3 4B | v0 | gemini-3.8-flash | 32.1% | 33.3% | 30.0% | 9 | 19 | 28 |
| 712 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.6-flash | 32.0% | 33.3% | 26.7% | 8 | 17 | 25 |
| 713 | N0027 | CatBoost + Stylometry | v0 | claude-fable-5 | 32.0% | 33.3% | 26.7% | 8 | 17 | 25 |
| 714 | N0046 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 32.0% | 33.3% | 26.7% | 8 | 17 | 25 |
| 715 | N0062 | Llama 3.2 3B | v0 | claude-sonnet-5 | 32.0% | 33.3% | 26.7% | 8 | 17 | 25 |
| 716 | N0068 | Qwen3 0.6B | v0 | qwen3.8-max | 32.0% | 33.3% | 26.7% | 8 | 17 | 25 |
| 717 | N0080 | SVC | v0 | qwen3.8-max | 32.0% | 33.3% | 26.7% | 8 | 17 | 25 |
| 718 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.6-terra | 31.8% | 33.3% | 23.3% | 7 | 15 | 22 |
| 719 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-6-astra | 31.8% | 33.3% | 23.3% | 7 | 15 | 22 |
| 720 | N0025 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 31.8% | 33.3% | 23.3% | 7 | 15 | 22 |
| 721 | N0036 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 31.8% | 33.3% | 23.3% | 7 | 15 | 22 |
| 722 | N0059 | Granite 4.0 350M Base | v1.1 | kimi-k3 | 31.8% | 33.3% | 23.3% | 7 | 15 | 22 |
| 723 | N0069 | Qwen3 0.6B | v0 | gemini-3.7-flash | 31.8% | 33.3% | 23.3% | 7 | 15 | 22 |
| 724 | N0010 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 725 | N0022 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 726 | N0028 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 727 | N0036 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 728 | N0042 | CatBoost + Stylometry | v1 | glm-5.3-flash | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 729 | N0062 | Llama 3.2 3B | v0 | qwen3.8-max | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 730 | N0068 | Qwen3 0.6B | v0 | claude-opus-4-8 | 31.6% | 33.3% | 20.0% | 6 | 13 | 19 |
| 731 | N0056 | EmbeddingGemma 2 | v1.1 | kimi-k3 | 31.2% | 33.3% | 16.7% | 5 | 11 | 16 |
| 732 | N0037 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 30.0% | 33.3% | 10.0% | 3 | 7 | 10 |
| 733 | N0038 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 30.0% | 33.3% | 10.0% | 3 | 7 | 10 |
| 734 | N0048 | CatBoost + Stylometry | v1.1 | gpt-5.6-luna | 30.0% | 33.3% | 10.0% | 3 | 7 | 10 |
| 735 | N0074 | Qwen3 1.7B | v0 | claude-fable-5 | 30.0% | 33.3% | 10.0% | 3 | 7 | 10 |
| 736 | N0081 | SVC | v1 | kimi-k3 | 30.0% | 33.3% | 10.0% | 3 | 7 | 10 |
| 737 | N0025 | CatBoost + Stylometry | v0 | claude-opus-5 | 32.2% | 32.8% | 63.3% | 19 | 40 | 59 |
| 738 | N0022 | CatBoost + Stylometry | v0 | gpt-6-astra | 32.1% | 32.7% | 56.7% | 17 | 36 | 53 |
| 739 | N0057 | EmbeddingGemma 2 | v1.5 | gpt-5.5 | 32.1% | 32.7% | 56.7% | 17 | 36 | 53 |
| 740 | N0029 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 31.8% | 32.6% | 46.7% | 14 | 30 | 44 |
| 741 | N0067 | Qwen3 0.6B | v0 | gpt-5.5 | 31.8% | 32.6% | 46.7% | 14 | 30 | 44 |
| 742 | N0037 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 31.7% | 32.6% | 43.3% | 13 | 28 | 41 |
| 743 | N0040 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 31.7% | 32.6% | 43.3% | 13 | 28 | 41 |
| 744 | N0042 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 31.4% | 32.4% | 36.7% | 11 | 24 | 35 |
| 745 | N0058 | Gemma 4 E2B | v1.1 | glm-5.3-flash | 31.4% | 32.4% | 36.7% | 11 | 24 | 35 |
| 746 | N0072 | Qwen3 0.6B | v1.5 | gpt-5.6-terra | 31.4% | 32.4% | 36.7% | 11 | 24 | 35 |
| 747 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.8-flash | 31.2% | 32.4% | 33.3% | 10 | 22 | 32 |
| 748 | N0027 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 31.2% | 32.4% | 33.3% | 10 | 22 | 32 |
| 749 | N0078 | SmolLM3 3B | v0 | gemini-3.6-flash | 31.2% | 32.4% | 33.3% | 10 | 22 | 32 |
| 750 | N0054 | EmbeddingGemma 2 | v1 | gpt-5.6-terra | 31.7% | 32.3% | 66.7% | 20 | 43 | 63 |
| 751 | N0010 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 31.0% | 32.3% | 30.0% | 9 | 20 | 29 |
| 752 | N0063 | Llama 3.2 3B | v1.1 | glm-5.3 | 31.0% | 32.3% | 30.0% | 9 | 20 | 29 |
| 753 | N0022 | CatBoost + Stylometry | v0 | claude-opus-5 | 31.6% | 32.2% | 60.0% | 18 | 39 | 57 |
| 754 | N0054 | EmbeddingGemma 2 | v1 | claude-opus-4-8 | 31.6% | 32.2% | 60.0% | 18 | 39 | 57 |
| 755 | N0074 | Qwen3 1.7B | v0 | claude-opus-5 | 31.7% | 32.1% | 86.7% | 26 | 56 | 82 |
| 756 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-opus-4-8 | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 757 | N0024 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 758 | N0030 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 759 | N0032 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 760 | N0041 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 761 | N0044 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 762 | N0047 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 763 | N0049 | CatBoost + Stylometry | v1.5 | gpt-5.6-terra | 30.8% | 32.1% | 26.7% | 8 | 18 | 26 |
| 764 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-opus-4-6 | 31.2% | 32.0% | 50.0% | 15 | 33 | 48 |
| 765 | N0010 | CatBoost + Stylometry | v0 | glm-5.3 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 766 | N0017 | CatBoost + Stylometry | v0 | glm-5.3 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 767 | N0020 | CatBoost + Stylometry | v0 | glm-5.3 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 768 | N0024 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 769 | N0025 | CatBoost + Stylometry | v0 | glm-5.3 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 770 | N0038 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 771 | N0051 | EmbeddingGemma 2 | v0 | qwen3.8-max | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 772 | N0054 | EmbeddingGemma 2 | v1 | kimi-k3 | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 773 | N0060 | Granite 4 H Micro | v0 | gemini-3.6-flash | 30.4% | 32.0% | 23.3% | 7 | 16 | 23 |
| 774 | N0010 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 30.0% | 31.8% | 20.0% | 6 | 14 | 20 |
| 775 | N0022 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 30.0% | 31.8% | 20.0% | 6 | 14 | 20 |
| 776 | N0067 | Qwen3 0.6B | v0 | gpt-5.6-luna | 30.0% | 31.8% | 20.0% | 6 | 14 | 20 |
| 777 | N0082 | SVC | v1.1 | gpt-5.6-terra | 30.0% | 31.8% | 20.0% | 6 | 14 | 20 |
| 778 | N0064 | Ministral 3 3B | v0 | gemini-3.6-flash | 30.8% | 31.7% | 40.0% | 12 | 27 | 39 |
| 779 | N0064 | Ministral 3 3B | v0 | gpt-5.6-sol | 30.8% | 31.7% | 40.0% | 12 | 27 | 39 |
| 780 | N0071 | Qwen3 0.6B | v1.1 | glm-5.3-flash | 30.8% | 31.7% | 40.0% | 12 | 27 | 39 |
| 781 | N0045 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 30.6% | 31.6% | 36.7% | 11 | 25 | 36 |
| 782 | N0056 | EmbeddingGemma 2 | v1.1 | gemini-3.7-flash | 30.6% | 31.6% | 36.7% | 11 | 25 | 36 |
| 783 | N0018 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 29.4% | 31.6% | 16.7% | 5 | 12 | 17 |
| 784 | N0022 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 29.4% | 31.6% | 16.7% | 5 | 12 | 17 |
| 785 | N0025 | CatBoost + Stylometry | v0 | glm-5.3-flash | 29.4% | 31.6% | 16.7% | 5 | 12 | 17 |
| 786 | N0036 | CatBoost + Stylometry | v1 | glm-5.3-flash | 29.4% | 31.6% | 16.7% | 5 | 12 | 17 |
| 787 | N0062 | Llama 3.2 3B | v0 | claude-fable-5 | 29.4% | 31.6% | 16.7% | 5 | 12 | 17 |
| 788 | N0080 | SVC | v0 | gpt-5.5 | 29.4% | 31.6% | 16.7% | 5 | 12 | 17 |
| 789 | N0018 | CatBoost + Stylometry | v0 | gpt-6-astra | 30.8% | 31.5% | 53.3% | 16 | 36 | 52 |
| 790 | N0068 | Qwen3 0.6B | v0 | gemini-3.6-flash | 30.8% | 31.5% | 53.3% | 16 | 36 | 52 |
| 791 | N0030 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 30.3% | 31.4% | 33.3% | 10 | 23 | 33 |
| 792 | N0052 | EmbeddingGemma 2 | v0 | gpt-6-astra | 30.3% | 31.4% | 33.3% | 10 | 23 | 33 |
| 793 | N0059 | Granite 4.0 350M Base | v1.1 | claude-fable-5 | 30.3% | 31.4% | 33.3% | 10 | 23 | 33 |
| 794 | N0067 | Qwen3 0.6B | v0 | gemini-3.8-flash | 30.3% | 31.4% | 33.3% | 10 | 23 | 33 |
| 795 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-opus-4-6 | 30.4% | 31.2% | 46.7% | 14 | 32 | 46 |
| 796 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-opus-5 | 30.4% | 31.2% | 46.7% | 14 | 32 | 46 |
| 797 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.6-flash | 30.0% | 31.2% | 30.0% | 9 | 21 | 30 |
| 798 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.5 | 30.0% | 31.2% | 30.0% | 9 | 21 | 30 |
| 799 | N0019 | CatBoost + Stylometry | v0 | qwen3.8-max | 30.0% | 31.2% | 30.0% | 9 | 21 | 30 |
| 800 | N0025 | CatBoost + Stylometry | v0 | qwen3.8-max | 30.0% | 31.2% | 30.0% | 9 | 21 | 30 |
| 801 | N0063 | Llama 3.2 3B | v1.1 | gpt-5.6-sol | 30.0% | 31.2% | 30.0% | 9 | 21 | 30 |
| 802 | N0076 | Qwen3 4B | v0 | gemini-3.6-flash | 30.0% | 31.2% | 30.0% | 9 | 21 | 30 |
| 803 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.6-luna | 28.6% | 31.2% | 13.3% | 4 | 10 | 14 |
| 804 | A | UltraBERT-380M | v1 | kimi-k3 | 30.2% | 31.1% | 43.3% | 13 | 30 | 43 |
| 805 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-opus-5 | 30.2% | 31.1% | 43.3% | 13 | 30 | 43 |
| 806 | N0075 | Qwen3 1.7B | v1.1 | claude-fable-5 | 30.2% | 31.1% | 43.3% | 13 | 30 | 43 |
| 807 | N0038 | CatBoost + Stylometry | v1 | gpt-6-astra | 29.6% | 31.0% | 26.7% | 8 | 19 | 27 |
| 808 | N0069 | Qwen3 0.6B | v0 | claude-opus-4-8 | 29.6% | 31.0% | 26.7% | 8 | 19 | 27 |
| 809 | N0038 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 30.0% | 31.0% | 40.0% | 12 | 28 | 40 |
| 810 | N0055 | EmbeddingGemma 2 | v1.1 | gemini-3.7-flash | 30.0% | 31.0% | 40.0% | 12 | 28 | 40 |
| 811 | N0057 | EmbeddingGemma 2 | v1.5 | gpt-5.6-luna | 30.0% | 31.0% | 40.0% | 12 | 28 | 40 |
| 812 | N0068 | Qwen3 0.6B | v0 | gpt-5.5 | 30.0% | 31.0% | 40.0% | 12 | 28 | 40 |
| 813 | N0069 | Qwen3 0.6B | v0 | qwen3.8-max | 30.0% | 31.0% | 40.0% | 12 | 28 | 40 |
| 814 | N0039 | CatBoost + Stylometry | v1 | claude-opus-5 | 30.2% | 30.9% | 53.3% | 16 | 37 | 53 |
| 815 | N0037 | CatBoost + Stylometry | v1 | gpt-6-astra | 30.0% | 30.8% | 50.0% | 15 | 35 | 50 |
| 816 | N0048 | CatBoost + Stylometry | v1.1 | gemini-3.8-flash | 29.7% | 30.8% | 36.7% | 11 | 26 | 37 |
| 817 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-6-astra | 29.2% | 30.8% | 23.3% | 7 | 17 | 24 |
| 818 | N0015 | CatBoost + Stylometry | v0 | glm-5.3 | 29.2% | 30.8% | 23.3% | 7 | 17 | 24 |
| 819 | N0033 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 29.2% | 30.8% | 23.3% | 7 | 17 | 24 |
| 820 | N0039 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 29.2% | 30.8% | 23.3% | 7 | 17 | 24 |
| 821 | N0048 | CatBoost + Stylometry | v1.1 | gpt-5.6-terra | 29.2% | 30.8% | 23.3% | 7 | 17 | 24 |
| 822 | N0077 | Qwen3 4B | v0 | gemini-3.6-flash | 29.2% | 30.8% | 23.3% | 7 | 17 | 24 |
| 823 | N0039 | CatBoost + Stylometry | v1 | glm-5.3 | 27.3% | 30.8% | 10.0% | 3 | 8 | 11 |
| 824 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-opus-5 | 29.8% | 30.6% | 46.7% | 14 | 33 | 47 |
| 825 | N0056 | EmbeddingGemma 2 | v1.1 | qwen3.8-max | 29.8% | 30.6% | 46.7% | 14 | 33 | 47 |
| 826 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.7-flash | 29.4% | 30.6% | 33.3% | 10 | 24 | 34 |
| 827 | N0052 | EmbeddingGemma 2 | v0 | qwen3.8-max | 29.4% | 30.6% | 33.3% | 10 | 24 | 34 |
| 828 | N0042 | CatBoost + Stylometry | v1 | claude-opus-5 | 29.8% | 30.5% | 56.7% | 17 | 40 | 57 |
| 829 | N0062 | Llama 3.2 3B | v0 | gpt-5.5 | 29.5% | 30.4% | 43.3% | 13 | 31 | 44 |
| 830 | N0012 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 831 | N0013 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 832 | N0014 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 833 | N0019 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 834 | N0019 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 835 | N0021 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 836 | N0023 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 837 | N0025 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 838 | N0026 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 839 | N0037 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 840 | N0051 | EmbeddingGemma 2 | v0 | claude-sonnet-5 | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 841 | N0073 | Qwen3 1.7B | v0 | gemini-3.6-flash | 28.6% | 30.4% | 20.0% | 6 | 15 | 21 |
| 842 | N0039 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 29.0% | 30.3% | 30.0% | 9 | 22 | 31 |
| 843 | N0029 | CatBoost + Stylometry | v1 | gpt-5.5 | 29.2% | 30.0% | 46.7% | 14 | 34 | 48 |
| 844 | N0040 | CatBoost + Stylometry | v1 | claude-opus-5 | 29.2% | 30.0% | 46.7% | 14 | 34 | 48 |
| 845 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.7-flash | 28.9% | 30.0% | 36.7% | 11 | 27 | 38 |
| 846 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.5 | 28.9% | 30.0% | 36.7% | 11 | 27 | 38 |
| 847 | N0015 | CatBoost + Stylometry | v0 | gpt-5.5 | 28.9% | 30.0% | 36.7% | 11 | 27 | 38 |
| 848 | N0019 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 28.9% | 30.0% | 36.7% | 11 | 27 | 38 |
| 849 | N0038 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 28.9% | 30.0% | 36.7% | 11 | 27 | 38 |
| 850 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.5 | 28.6% | 30.0% | 26.7% | 8 | 20 | 28 |
| 851 | N0028 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 28.6% | 30.0% | 26.7% | 8 | 20 | 28 |
| 852 | N0028 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 28.6% | 30.0% | 26.7% | 8 | 20 | 28 |
| 853 | N0030 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 28.6% | 30.0% | 26.7% | 8 | 20 | 28 |
| 854 | N0045 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 28.6% | 30.0% | 26.7% | 8 | 20 | 28 |
| 855 | N0011 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 27.8% | 30.0% | 16.7% | 5 | 13 | 18 |
| 856 | N0022 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 27.8% | 30.0% | 16.7% | 5 | 13 | 18 |
| 857 | N0069 | Qwen3 0.6B | v0 | gpt-5.6-terra | 27.8% | 30.0% | 16.7% | 5 | 13 | 18 |
| 858 | N0077 | Qwen3 4B | v0 | gemini-3.7-flash | 27.8% | 30.0% | 16.7% | 5 | 13 | 18 |
| 859 | N0046 | CatBoost + Stylometry | v1 | claude-opus-5 | 29.1% | 29.8% | 53.3% | 16 | 39 | 55 |
| 860 | N0058 | Gemma 4 E2B | v1.1 | glm-5.3 | 28.9% | 29.8% | 43.3% | 13 | 32 | 45 |
| 861 | N0074 | Qwen3 1.7B | v0 | glm-5.3-flash | 28.9% | 29.8% | 43.3% | 13 | 32 | 45 |
| 862 | N0045 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 28.6% | 29.7% | 33.3% | 10 | 25 | 35 |
| 863 | N0046 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 28.6% | 29.7% | 33.3% | 10 | 25 | 35 |
| 864 | D | UltraBERT-380M | v0 | glm-5.3 | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 865 | N0018 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 866 | N0028 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 867 | N0037 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 868 | N0039 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 869 | N0043 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 870 | N0050 | CatBoost + Stylometry | v1.5 | gpt-5.6-terra | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 871 | N0076 | Qwen3 4B | v0 | qwen3.8-max | 28.0% | 29.6% | 23.3% | 7 | 18 | 25 |
| 872 | N0034 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 28.6% | 29.5% | 40.0% | 12 | 30 | 42 |
| 873 | N0034 | CatBoost + Stylometry | v1 | qwen3.8-max | 28.6% | 29.4% | 46.7% | 14 | 35 | 49 |
| 874 | N0036 | CatBoost + Stylometry | v1 | qwen3.8-max | 28.6% | 29.4% | 46.7% | 14 | 35 | 49 |
| 875 | N0032 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 876 | N0034 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 877 | N0041 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 878 | N0044 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 879 | N0047 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 880 | N0049 | CatBoost + Stylometry | v1.5 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 881 | N0051 | EmbeddingGemma 2 | v0 | gemini-3.7-flash | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 882 | N0064 | Ministral 3 3B | v0 | gpt-5.6-luna | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 883 | N0068 | Qwen3 0.6B | v0 | kimi-k3 | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 884 | N0073 | Qwen3 1.7B | v0 | claude-opus-4-8 | 28.1% | 29.4% | 30.0% | 9 | 23 | 32 |
| 885 | N0027 | CatBoost + Stylometry | v0 | glm-5.3 | 26.7% | 29.4% | 13.3% | 4 | 11 | 15 |
| 886 | N0039 | CatBoost + Stylometry | v1 | glm-5.3-flash | 26.7% | 29.4% | 13.3% | 4 | 11 | 15 |
| 887 | N0070 | Qwen3 0.6B | v1.1 | claude-opus-4-8 | 26.7% | 29.4% | 13.3% | 4 | 11 | 15 |
| 888 | N0019 | CatBoost + Stylometry | v0 | gpt-5.5 | 28.2% | 29.3% | 36.7% | 11 | 28 | 39 |
| 889 | N0056 | EmbeddingGemma 2 | v1.1 | gpt-5.5 | 28.2% | 29.3% | 36.7% | 11 | 28 | 39 |
| 890 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.6-terra | 27.3% | 29.2% | 20.0% | 6 | 16 | 22 |
| 891 | N0020 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 27.3% | 29.2% | 20.0% | 6 | 16 | 22 |
| 892 | N0083 | SVC | v1.1 | gpt-5.6-terra | 27.3% | 29.2% | 20.0% | 6 | 16 | 22 |
| 893 | N0084 | SVC | v1.1 | gpt-5.6-terra | 27.3% | 29.2% | 20.0% | 6 | 16 | 22 |
| 894 | N0085 | SVC | v1.1 | gpt-5.6-terra | 27.3% | 29.2% | 20.0% | 6 | 16 | 22 |
| 895 | N0038 | CatBoost + Stylometry | v1 | claude-opus-5 | 28.3% | 29.1% | 50.0% | 15 | 38 | 53 |
| 896 | N0017 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 27.6% | 29.0% | 26.7% | 8 | 21 | 29 |
| 897 | N0027 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 27.6% | 29.0% | 26.7% | 8 | 21 | 29 |
| 898 | N0036 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 27.6% | 29.0% | 26.7% | 8 | 21 | 29 |
| 899 | N0074 | Qwen3 1.7B | v0 | gpt-5.6-luna | 27.6% | 29.0% | 26.7% | 8 | 21 | 29 |
| 900 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.5 | 27.8% | 28.9% | 33.3% | 10 | 26 | 36 |
| 901 | N0016 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 27.8% | 28.9% | 33.3% | 10 | 26 | 36 |
| 902 | N0020 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 27.8% | 28.9% | 33.3% | 10 | 26 | 36 |
| 903 | N0014 | CatBoost + Stylometry | v0 | gpt-5.5 | 27.9% | 28.9% | 40.0% | 12 | 31 | 43 |
| 904 | N0022 | CatBoost + Stylometry | v0 | gpt-5.5 | 27.9% | 28.9% | 40.0% | 12 | 31 | 43 |
| 905 | N0036 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 27.9% | 28.9% | 40.0% | 12 | 31 | 43 |
| 906 | N0045 | CatBoost + Stylometry | v1 | gpt-5.5 | 27.9% | 28.9% | 40.0% | 12 | 31 | 43 |
| 907 | N0078 | SmolLM3 3B | v0 | gpt-5.5 | 27.9% | 28.9% | 40.0% | 12 | 31 | 43 |
| 908 | N0014 | CatBoost + Stylometry | v0 | gpt-6-astra | 28.0% | 28.8% | 46.7% | 14 | 36 | 50 |
| 909 | N0070 | Qwen3 0.6B | v1.1 | claude-fable-5 | 28.0% | 28.8% | 46.7% | 14 | 36 | 50 |
| 910 | N0057 | EmbeddingGemma 2 | v1.5 | claude-opus-4-8 | 27.9% | 28.6% | 56.7% | 17 | 44 | 61 |
| 911 | N0010 | CatBoost + Stylometry | v0 | claude-opus-5 | 27.7% | 28.6% | 43.3% | 13 | 34 | 47 |
| 912 | N0014 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 27.5% | 28.6% | 36.7% | 11 | 29 | 40 |
| 913 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.7-flash | 27.3% | 28.6% | 30.0% | 9 | 24 | 33 |
| 914 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.8-flash | 27.3% | 28.6% | 30.0% | 9 | 24 | 33 |
| 915 | N0010 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 27.3% | 28.6% | 30.0% | 9 | 24 | 33 |
| 916 | N0037 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 27.3% | 28.6% | 30.0% | 9 | 24 | 33 |
| 917 | N0040 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 27.3% | 28.6% | 30.0% | 9 | 24 | 33 |
| 918 | C | UltraBERT-380M | v1.5 | claude-fable-5 | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 919 | N0020 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 920 | N0022 | CatBoost + Stylometry | v0 | glm-5.3 | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 921 | N0037 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 922 | N0045 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 923 | N0069 | Qwen3 0.6B | v0 | gemini-3.6-flash | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 924 | N0076 | Qwen3 4B | v0 | gpt-5.6-sol | 26.9% | 28.6% | 23.3% | 7 | 19 | 26 |
| 925 | N0067 | Qwen3 0.6B | v0 | claude-opus-4-8 | 26.3% | 28.6% | 16.7% | 5 | 14 | 19 |
| 926 | N0068 | Qwen3 0.6B | v0 | gpt-5.6-luna | 26.3% | 28.6% | 16.7% | 5 | 14 | 19 |
| 927 | N0024 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 25.0% | 28.6% | 10.0% | 3 | 9 | 12 |
| 928 | N0030 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 25.0% | 28.6% | 10.0% | 3 | 9 | 12 |
| 929 | N0034 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 25.0% | 28.6% | 10.0% | 3 | 9 | 12 |
| 930 | N0043 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 25.0% | 28.6% | 10.0% | 3 | 9 | 12 |
| 931 | N0050 | CatBoost + Stylometry | v1.5 | gpt-5.6-sol | 25.0% | 28.6% | 10.0% | 3 | 9 | 12 |
| 932 | N0055 | EmbeddingGemma 2 | v1.1 | gpt-6-astra | 27.9% | 28.4% | 80.0% | 24 | 62 | 86 |
| 933 | N0065 | Ministral 3 3B | v1.1 | glm-5.3 | 27.3% | 28.3% | 40.0% | 12 | 32 | 44 |
| 934 | N0040 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 27.0% | 28.2% | 33.3% | 10 | 27 | 37 |
| 935 | N0039 | CatBoost + Stylometry | v1 | qwen3.8-max | 27.4% | 28.1% | 56.7% | 17 | 45 | 62 |
| 936 | N0042 | CatBoost + Stylometry | v1 | qwen3.8-max | 27.4% | 28.1% | 56.7% | 17 | 45 | 62 |
| 937 | N0069 | Qwen3 0.6B | v0 | gpt-5.5 | 27.4% | 28.1% | 56.7% | 17 | 45 | 62 |
| 938 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gemini-3.6-flash | 26.7% | 28.1% | 26.7% | 8 | 22 | 30 |
| 939 | N0043 | CatBoost + Stylometry | v1 | gpt-5.5 | 26.7% | 28.1% | 26.7% | 8 | 22 | 30 |
| 940 | N0050 | CatBoost + Stylometry | v1.5 | gpt-5.5 | 26.7% | 28.1% | 26.7% | 8 | 22 | 30 |
| 941 | N0074 | Qwen3 1.7B | v0 | claude-opus-4-8 | 26.7% | 28.1% | 26.7% | 8 | 22 | 30 |
| 942 | N0048 | CatBoost + Stylometry | v1.1 | claude-opus-4-6 | 27.1% | 28.0% | 43.3% | 13 | 35 | 48 |
| 943 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-fable-5 | 26.1% | 28.0% | 20.0% | 6 | 17 | 23 |
| 944 | N0015 | CatBoost + Stylometry | v0 | glm-5.3-flash | 26.1% | 28.0% | 20.0% | 6 | 17 | 23 |
| 945 | N0045 | CatBoost + Stylometry | v1 | glm-5.3-flash | 26.1% | 28.0% | 20.0% | 6 | 17 | 23 |
| 946 | N0073 | Qwen3 1.7B | v0 | gpt-5.6-luna | 26.1% | 28.0% | 20.0% | 6 | 17 | 23 |
| 947 | N0080 | SVC | v0 | gpt-5.6-terra | 26.1% | 28.0% | 20.0% | 6 | 17 | 23 |
| 948 | N0081 | SVC | v1 | gpt-5.6-terra | 26.1% | 28.0% | 20.0% | 6 | 17 | 23 |
| 949 | N0032 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 26.8% | 27.9% | 36.7% | 11 | 30 | 41 |
| 950 | N0041 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 26.8% | 27.9% | 36.7% | 11 | 30 | 41 |
| 951 | N0044 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 26.8% | 27.9% | 36.7% | 11 | 30 | 41 |
| 952 | N0047 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 26.8% | 27.9% | 36.7% | 11 | 30 | 41 |
| 953 | N0049 | CatBoost + Stylometry | v1.5 | claude-opus-4-6 | 26.8% | 27.9% | 36.7% | 11 | 30 | 41 |
| 954 | N0077 | Qwen3 4B | v0 | qwen3.8-max | 26.8% | 27.9% | 36.7% | 11 | 30 | 41 |
| 955 | N0029 | CatBoost + Stylometry | v1 | qwen3.8-max | 26.9% | 27.8% | 46.7% | 14 | 38 | 52 |
| 956 | C | UltraBERT-380M | v1.5 | glm-5.3 | 26.5% | 27.8% | 30.0% | 9 | 25 | 34 |
| 957 | N0024 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 26.5% | 27.8% | 30.0% | 9 | 25 | 34 |
| 958 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | glm-5.3-flash | 25.0% | 27.8% | 13.3% | 4 | 12 | 16 |
| 959 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.6-luna | 25.0% | 27.8% | 13.3% | 4 | 12 | 16 |
| 960 | N0066 | Qwen3 0.6B | v0 | glm-5.3-flash | 25.0% | 27.8% | 13.3% | 4 | 12 | 16 |
| 961 | N0017 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.7% | 27.7% | 40.0% | 12 | 33 | 45 |
| 962 | N0020 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.7% | 27.7% | 40.0% | 12 | 33 | 45 |
| 963 | N0025 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.7% | 27.7% | 40.0% | 12 | 33 | 45 |
| 964 | N0033 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 26.7% | 27.7% | 40.0% | 12 | 33 | 45 |
| 965 | N0048 | CatBoost + Stylometry | v1.1 | claude-opus-5 | 26.7% | 27.7% | 40.0% | 12 | 33 | 45 |
| 966 | N0073 | Qwen3 1.7B | v0 | claude-sonnet-5 | 26.7% | 27.7% | 40.0% | 12 | 33 | 45 |
| 967 | N0045 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.9% | 27.6% | 23.3% | 7 | 20 | 27 |
| 968 | N0052 | EmbeddingGemma 2 | v0 | glm-5.3 | 25.9% | 27.6% | 23.3% | 7 | 20 | 27 |
| 969 | N0064 | Ministral 3 3B | v0 | gpt-5.6-terra | 25.9% | 27.6% | 23.3% | 7 | 20 | 27 |
| 970 | N0010 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.3% | 27.5% | 33.3% | 10 | 28 | 38 |
| 971 | N0020 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 26.3% | 27.5% | 33.3% | 10 | 28 | 38 |
| 972 | N0037 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 26.3% | 27.5% | 33.3% | 10 | 28 | 38 |
| 973 | N0061 | Granite 4 H Micro | v1.1 | glm-5.3 | 26.4% | 27.3% | 46.7% | 14 | 39 | 53 |
| 974 | N0011 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 975 | N0012 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 976 | N0013 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 977 | N0021 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 978 | N0023 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 979 | N0026 | CatBoost + Stylometry | v0 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 980 | N0034 | CatBoost + Stylometry | v1 | gpt-5.5 | 26.2% | 27.3% | 36.7% | 11 | 31 | 42 |
| 981 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.5 | 25.8% | 27.3% | 26.7% | 8 | 23 | 31 |
| 982 | N0027 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 25.8% | 27.3% | 26.7% | 8 | 23 | 31 |
| 983 | N0029 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.8% | 27.3% | 26.7% | 8 | 23 | 31 |
| 984 | N0062 | Llama 3.2 3B | v0 | gpt-5.6-sol | 25.8% | 27.3% | 26.7% | 8 | 23 | 31 |
| 985 | N0062 | Llama 3.2 3B | v0 | kimi-k3 | 25.8% | 27.3% | 26.7% | 8 | 23 | 31 |
| 986 | N0078 | SmolLM3 3B | v0 | claude-sonnet-5 | 25.8% | 27.3% | 26.7% | 8 | 23 | 31 |
| 987 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | glm-5.3-flash | 25.0% | 27.3% | 16.7% | 5 | 15 | 20 |
| 988 | N0019 | CatBoost + Stylometry | v0 | glm-5.3-flash | 25.0% | 27.3% | 16.7% | 5 | 15 | 20 |
| 989 | N0042 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 25.0% | 27.3% | 16.7% | 5 | 15 | 20 |
| 990 | N0052 | EmbeddingGemma 2 | v0 | claude-opus-4-8 | 25.0% | 27.3% | 16.7% | 5 | 15 | 20 |
| 991 | N0058 | Gemma 4 E2B | v1.1 | gpt-5.6-sol | 25.0% | 27.3% | 16.7% | 5 | 15 | 20 |
| 992 | N0061 | Granite 4 H Micro | v1.1 | glm-5.3-flash | 25.0% | 27.3% | 16.7% | 5 | 15 | 20 |
| 993 | N0040 | CatBoost + Stylometry | v1 | qwen3.8-max | 26.3% | 27.1% | 50.0% | 15 | 42 | 57 |
| 994 | N0066 | Qwen3 0.6B | v0 | claude-fable-5 | 26.1% | 27.1% | 40.0% | 12 | 34 | 46 |
| 995 | N0019 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 25.7% | 27.0% | 30.0% | 9 | 26 | 35 |
| 996 | N0027 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 25.7% | 27.0% | 30.0% | 9 | 26 | 35 |
| 997 | N0071 | Qwen3 0.6B | v1.1 | claude-fable-5 | 25.7% | 27.0% | 30.0% | 9 | 26 | 35 |
| 998 | N0038 | CatBoost + Stylometry | v1 | qwen3.8-max | 26.0% | 26.9% | 43.3% | 13 | 37 | 50 |
| 999 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-fable-5 | 25.0% | 26.9% | 20.0% | 6 | 18 | 24 |
| 1000 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.6-terra | 25.0% | 26.9% | 20.0% | 6 | 18 | 24 |
| 1001 | N0018 | CatBoost + Stylometry | v0 | gemini-3.8-flash | 25.0% | 26.9% | 20.0% | 6 | 18 | 24 |
| 1002 | N0051 | EmbeddingGemma 2 | v0 | gemini-3.6-flash | 25.0% | 26.9% | 20.0% | 6 | 18 | 24 |
| 1003 | N0033 | CatBoost + Stylometry | v1 | gpt-5.5 | 25.6% | 26.8% | 33.3% | 10 | 29 | 39 |
| 1004 | N0052 | EmbeddingGemma 2 | v0 | claude-opus-5 | 25.6% | 26.8% | 33.3% | 10 | 29 | 39 |
| 1005 | N0064 | Ministral 3 3B | v0 | glm-5.3 | 25.9% | 26.7% | 50.0% | 15 | 43 | 58 |
| 1006 | N0057 | EmbeddingGemma 2 | v1.5 | claude-fable-5 | 25.6% | 26.7% | 36.7% | 11 | 32 | 43 |
| 1007 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.6-flash | 25.0% | 26.7% | 23.3% | 7 | 21 | 28 |
| 1008 | N0031 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 25.0% | 26.7% | 23.3% | 7 | 21 | 28 |
| 1009 | N0060 | Granite 4 H Micro | v0 | claude-opus-4-8 | 25.0% | 26.7% | 23.3% | 7 | 21 | 28 |
| 1010 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | glm-5.3-flash | 23.1% | 26.7% | 10.0% | 3 | 10 | 13 |
| 1011 | N0033 | CatBoost + Stylometry | v1 | glm-5.3-flash | 23.1% | 26.7% | 10.0% | 3 | 10 | 13 |
| 1012 | N0031 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.0% | 26.5% | 26.7% | 8 | 24 | 32 |
| 1013 | N0031 | CatBoost + Stylometry | v1 | gpt-5.5 | 25.0% | 26.5% | 26.7% | 8 | 24 | 32 |
| 1014 | N0038 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 25.0% | 26.5% | 26.7% | 8 | 24 | 32 |
| 1015 | N0053 | EmbeddingGemma 2 | v0 | gpt-6-astra | 25.0% | 26.5% | 26.7% | 8 | 24 | 32 |
| 1016 | N0067 | Qwen3 0.6B | v0 | gemini-3.6-flash | 25.0% | 26.5% | 26.7% | 8 | 24 | 32 |
| 1017 | N0080 | SVC | v0 | gpt-5.6-luna | 25.0% | 26.5% | 26.7% | 8 | 24 | 32 |
| 1018 | N0035 | CatBoost + Stylometry | v1 | gemini-3.1-pro-preview | 25.7% | 26.4% | 60.0% | 18 | 52 | 70 |
| 1019 | N0011 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1020 | N0022 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1021 | N0022 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1022 | N0032 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1023 | N0041 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1024 | N0044 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1025 | N0047 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1026 | N0049 | CatBoost + Stylometry | v1.5 | gemini-3.6-flash | 25.0% | 26.3% | 30.0% | 9 | 27 | 36 |
| 1027 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-fable-5 | 23.5% | 26.3% | 13.3% | 4 | 13 | 17 |
| 1028 | N0017 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 23.5% | 26.3% | 13.3% | 4 | 13 | 17 |
| 1029 | N0029 | CatBoost + Stylometry | v1 | glm-5.3-flash | 23.5% | 26.3% | 13.3% | 4 | 13 | 17 |
| 1030 | N0066 | Qwen3 0.6B | v0 | gpt-5.5 | 25.6% | 26.2% | 66.7% | 20 | 58 | 78 |
| 1031 | A | UltraBERT-380M | v1 | glm-5.3-flash | 25.0% | 26.2% | 33.3% | 10 | 30 | 40 |
| 1032 | N0018 | CatBoost + Stylometry | v0 | claude-opus-5 | 25.0% | 26.2% | 33.3% | 10 | 30 | 40 |
| 1033 | N0038 | CatBoost + Stylometry | v1 | gpt-5.5 | 25.0% | 26.2% | 33.3% | 10 | 30 | 40 |
| 1034 | N0046 | CatBoost + Stylometry | v1 | gpt-5.5 | 25.0% | 26.2% | 33.3% | 10 | 30 | 40 |
| 1035 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-opus-4-8 | 23.8% | 26.1% | 16.7% | 5 | 16 | 21 |
| 1036 | N0011 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 23.8% | 26.1% | 16.7% | 5 | 16 | 21 |
| 1037 | N0018 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 23.8% | 26.1% | 16.7% | 5 | 16 | 21 |
| 1038 | N0031 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 23.8% | 26.1% | 16.7% | 5 | 16 | 21 |
| 1039 | N0053 | EmbeddingGemma 2 | v0 | gemini-3.7-flash | 23.8% | 26.1% | 16.7% | 5 | 16 | 21 |
| 1040 | N0076 | Qwen3 4B | v0 | gpt-5.6-terra | 23.8% | 26.1% | 16.7% | 5 | 16 | 21 |
| 1041 | N0036 | CatBoost + Stylometry | v1 | gpt-5.5 | 25.0% | 26.0% | 40.0% | 12 | 36 | 48 |
| 1042 | N0012 | CatBoost + Stylometry | v0 | glm-5.3-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1043 | N0012 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1044 | N0013 | CatBoost + Stylometry | v0 | glm-5.3-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1045 | N0013 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1046 | N0016 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1047 | N0017 | CatBoost + Stylometry | v0 | glm-5.3-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1048 | N0021 | CatBoost + Stylometry | v0 | glm-5.3-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1049 | N0021 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1050 | N0023 | CatBoost + Stylometry | v0 | glm-5.3-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1051 | N0023 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1052 | N0026 | CatBoost + Stylometry | v0 | glm-5.3-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1053 | N0026 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1054 | N0053 | EmbeddingGemma 2 | v0 | gemini-3.6-flash | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1055 | N0062 | Llama 3.2 3B | v0 | gpt-5.6-luna | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1056 | N0068 | Qwen3 0.6B | v0 | gpt-5.6-terra | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1057 | N0082 | SVC | v1.1 | gpt-5.6-luna | 24.0% | 25.9% | 20.0% | 6 | 19 | 25 |
| 1058 | B | UltraBERT-380M | v1_all | glm-5.3 | 24.1% | 25.8% | 23.3% | 7 | 22 | 29 |
| 1059 | N0022 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 24.1% | 25.8% | 23.3% | 7 | 22 | 29 |
| 1060 | N0028 | CatBoost + Stylometry | v1 | gpt-5.5 | 24.1% | 25.8% | 23.3% | 7 | 22 | 29 |
| 1061 | N0033 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 24.1% | 25.8% | 23.3% | 7 | 22 | 29 |
| 1062 | A | UltraBERT-380M | v1 | glm-5.3 | 24.2% | 25.7% | 26.7% | 8 | 25 | 33 |
| 1063 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.8-flash | 24.2% | 25.7% | 26.7% | 8 | 25 | 33 |
| 1064 | N0024 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 24.2% | 25.7% | 26.7% | 8 | 25 | 33 |
| 1065 | N0081 | SVC | v1 | gpt-5.6-luna | 24.2% | 25.7% | 26.7% | 8 | 25 | 33 |
| 1066 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.5 | 24.3% | 25.6% | 30.0% | 9 | 28 | 37 |
| 1067 | N0010 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 24.3% | 25.6% | 30.0% | 9 | 28 | 37 |
| 1068 | N0025 | CatBoost + Stylometry | v0 | claude-opus-4-6 | 24.3% | 25.6% | 30.0% | 9 | 28 | 37 |
| 1069 | N0051 | EmbeddingGemma 2 | v0 | gpt-6-astra | 24.3% | 25.6% | 30.0% | 9 | 28 | 37 |
| 1070 | N0052 | EmbeddingGemma 2 | v0 | gemini-3.7-flash | 24.3% | 25.6% | 30.0% | 9 | 28 | 37 |
| 1071 | N0078 | SmolLM3 3B | v0 | kimi-k3 | 24.3% | 25.6% | 30.0% | 9 | 28 | 37 |
| 1072 | N0012 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1073 | N0013 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1074 | N0021 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1075 | N0023 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1076 | N0026 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1077 | N0036 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1078 | N0048 | CatBoost + Stylometry | v1.1 | gpt-5.5 | 24.4% | 25.6% | 33.3% | 10 | 31 | 41 |
| 1079 | N0046 | CatBoost + Stylometry | v1 | qwen3.8-max | 24.5% | 25.5% | 43.3% | 13 | 40 | 53 |
| 1080 | N0074 | Qwen3 1.7B | v0 | kimi-k3 | 24.6% | 25.4% | 50.0% | 15 | 46 | 61 |
| 1081 | N0032 | CatBoost + Stylometry | v1 | qwen3.8-max | 24.1% | 25.0% | 43.3% | 13 | 41 | 54 |
| 1082 | N0041 | CatBoost + Stylometry | v1 | qwen3.8-max | 24.1% | 25.0% | 43.3% | 13 | 41 | 54 |
| 1083 | N0044 | CatBoost + Stylometry | v1 | qwen3.8-max | 24.1% | 25.0% | 43.3% | 13 | 41 | 54 |
| 1084 | N0047 | CatBoost + Stylometry | v1 | qwen3.8-max | 24.1% | 25.0% | 43.3% | 13 | 41 | 54 |
| 1085 | N0049 | CatBoost + Stylometry | v1.5 | qwen3.8-max | 24.1% | 25.0% | 43.3% | 13 | 41 | 54 |
| 1086 | N0077 | Qwen3 4B | v0 | kimi-k3 | 23.9% | 25.0% | 36.7% | 11 | 35 | 46 |
| 1087 | N0032 | CatBoost + Stylometry | v1 | gpt-5.5 | 23.8% | 25.0% | 33.3% | 10 | 32 | 42 |
| 1088 | N0041 | CatBoost + Stylometry | v1 | gpt-5.5 | 23.8% | 25.0% | 33.3% | 10 | 32 | 42 |
| 1089 | N0044 | CatBoost + Stylometry | v1 | gpt-5.5 | 23.8% | 25.0% | 33.3% | 10 | 32 | 42 |
| 1090 | N0047 | CatBoost + Stylometry | v1 | gpt-5.5 | 23.8% | 25.0% | 33.3% | 10 | 32 | 42 |
| 1091 | N0049 | CatBoost + Stylometry | v1.5 | gpt-5.5 | 23.8% | 25.0% | 33.3% | 10 | 32 | 42 |
| 1092 | B | UltraBERT-380M | v1_all | claude-fable-5 | 23.7% | 25.0% | 30.0% | 9 | 29 | 38 |
| 1093 | N0069 | Qwen3 0.6B | v0 | kimi-k3 | 23.7% | 25.0% | 30.0% | 9 | 29 | 38 |
| 1094 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.6-terra | 23.5% | 25.0% | 26.7% | 8 | 26 | 34 |
| 1095 | N0018 | CatBoost + Stylometry | v0 | gpt-5.5 | 23.5% | 25.0% | 26.7% | 8 | 26 | 34 |
| 1096 | N0035 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 23.3% | 25.0% | 23.3% | 7 | 23 | 30 |
| 1097 | N0036 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 23.3% | 25.0% | 23.3% | 7 | 23 | 30 |
| 1098 | N0080 | SVC | v0 | gemini-3.7-flash | 23.3% | 25.0% | 23.3% | 7 | 23 | 30 |
| 1099 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.6-terra | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1100 | N0014 | CatBoost + Stylometry | v0 | claude-fable-5 | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1101 | N0015 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1102 | N0016 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1103 | N0031 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1104 | N0038 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1105 | N0043 | CatBoost + Stylometry | v1 | claude-fable-5 | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1106 | N0050 | CatBoost + Stylometry | v1.5 | claude-fable-5 | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1107 | N0057 | EmbeddingGemma 2 | v1.5 | gpt-5.6-sol | 23.1% | 25.0% | 20.0% | 6 | 20 | 26 |
| 1108 | N0011 | CatBoost + Stylometry | v0 | glm-5.3 | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1109 | N0011 | CatBoost + Stylometry | v0 | glm-5.3-flash | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1110 | N0015 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1111 | N0015 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1112 | N0018 | CatBoost + Stylometry | v0 | gemini-3.7-flash | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1113 | N0020 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1114 | N0053 | EmbeddingGemma 2 | v0 | glm-5.3 | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1115 | N0060 | Granite 4 H Micro | v0 | claude-fable-5 | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1116 | N0078 | SmolLM3 3B | v0 | glm-5.3-flash | 22.7% | 25.0% | 16.7% | 5 | 17 | 22 |
| 1117 | N0020 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 22.2% | 25.0% | 13.3% | 4 | 14 | 18 |
| 1118 | N0059 | Granite 4.0 350M Base | v1.1 | gpt-5.6-sol | 22.2% | 25.0% | 13.3% | 4 | 14 | 18 |
| 1119 | N0065 | Ministral 3 3B | v1.1 | glm-5.3-flash | 22.2% | 25.0% | 13.3% | 4 | 14 | 18 |
| 1120 | N0066 | Qwen3 0.6B | v0 | gemini-3.6-flash | 22.2% | 25.0% | 13.3% | 4 | 14 | 18 |
| 1121 | N0069 | Qwen3 0.6B | v0 | glm-5.3-flash | 22.2% | 25.0% | 13.3% | 4 | 14 | 18 |
| 1122 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | glm-5.3-flash | 21.4% | 25.0% | 10.0% | 3 | 11 | 14 |
| 1123 | N0076 | Qwen3 4B | v0 | gemini-3.7-flash | 21.4% | 25.0% | 10.0% | 3 | 11 | 14 |
| 1124 | N0036 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 20.0% | 25.0% | 6.7% | 2 | 8 | 10 |
| 1125 | N0039 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 20.0% | 25.0% | 6.7% | 2 | 8 | 10 |
| 1126 | N0039 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 20.0% | 25.0% | 6.7% | 2 | 8 | 10 |
| 1127 | N0040 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 20.0% | 25.0% | 6.7% | 2 | 8 | 10 |
| 1128 | N0042 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 20.0% | 25.0% | 6.7% | 2 | 8 | 10 |
| 1129 | N0037 | CatBoost + Stylometry | v1 | claude-opus-5 | 23.6% | 24.6% | 43.3% | 13 | 42 | 55 |
| 1130 | N0080 | SVC | v0 | claude-opus-5 | 24.0% | 24.5% | 80.0% | 24 | 76 | 100 |
| 1131 | N0038 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 23.4% | 24.5% | 36.7% | 11 | 36 | 47 |
| 1132 | N0010 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 23.3% | 24.4% | 33.3% | 10 | 33 | 43 |
| 1133 | N0037 | CatBoost + Stylometry | v1 | gpt-5.5 | 23.3% | 24.4% | 33.3% | 10 | 33 | 43 |
| 1134 | N0033 | CatBoost + Stylometry | v1 | gpt-6-astra | 23.1% | 24.4% | 30.0% | 9 | 30 | 39 |
| 1135 | N0046 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 23.1% | 24.4% | 30.0% | 9 | 30 | 39 |
| 1136 | N0077 | Qwen3 4B | v0 | gpt-5.6-luna | 22.9% | 24.3% | 26.7% | 8 | 27 | 35 |
| 1137 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.6-flash | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1138 | N0015 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1139 | N0048 | CatBoost + Stylometry | v1.1 | gemini-3.6-flash | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1140 | N0048 | CatBoost + Stylometry | v1.1 | gemini-3.7-flash | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1141 | N0052 | EmbeddingGemma 2 | v0 | claude-fable-5 | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1142 | N0052 | EmbeddingGemma 2 | v0 | claude-sonnet-5 | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1143 | N0084 | SVC | v1.1 | gpt-5.6-luna | 22.6% | 24.2% | 23.3% | 7 | 24 | 31 |
| 1144 | N0018 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 22.2% | 24.1% | 20.0% | 6 | 21 | 27 |
| 1145 | N0024 | CatBoost + Stylometry | v0 | glm-5.3-flash | 22.2% | 24.1% | 20.0% | 6 | 21 | 27 |
| 1146 | N0030 | CatBoost + Stylometry | v1 | gpt-5.5 | 22.2% | 24.1% | 20.0% | 6 | 21 | 27 |
| 1147 | N0067 | Qwen3 0.6B | v0 | glm-5.3-flash | 22.2% | 24.1% | 20.0% | 6 | 21 | 27 |
| 1148 | N0083 | SVC | v1.1 | gpt-5.6-luna | 22.2% | 24.1% | 20.0% | 6 | 21 | 27 |
| 1149 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-fable-5 | 21.7% | 24.0% | 16.7% | 5 | 18 | 23 |
| 1150 | N0016 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 21.7% | 24.0% | 16.7% | 5 | 18 | 23 |
| 1151 | N0033 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 21.7% | 24.0% | 16.7% | 5 | 18 | 23 |
| 1152 | N0072 | Qwen3 0.6B | v1.5 | claude-fable-5 | 21.7% | 24.0% | 16.7% | 5 | 18 | 23 |
| 1153 | N0040 | CatBoost + Stylometry | v1 | gpt-5.5 | 22.7% | 23.9% | 33.3% | 10 | 34 | 44 |
| 1154 | N0069 | Qwen3 0.6B | v0 | glm-5.3 | 22.7% | 23.9% | 33.3% | 10 | 34 | 44 |
| 1155 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | qwen3.8-max | 23.0% | 23.8% | 46.7% | 14 | 47 | 61 |
| 1156 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | qwen3.8-max | 23.0% | 23.8% | 46.7% | 14 | 47 | 61 |
| 1157 | N0059 | Granite 4.0 350M Base | v1.1 | glm-5.3-flash | 23.0% | 23.8% | 46.7% | 14 | 47 | 61 |
| 1158 | N0042 | CatBoost + Stylometry | v1 | gpt-5.5 | 22.5% | 23.8% | 30.0% | 9 | 31 | 40 |
| 1159 | N0016 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1160 | N0020 | CatBoost + Stylometry | v0 | glm-5.3-flash | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1161 | N0032 | CatBoost + Stylometry | v1 | glm-5.3-flash | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1162 | N0041 | CatBoost + Stylometry | v1 | glm-5.3-flash | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1163 | N0044 | CatBoost + Stylometry | v1 | glm-5.3-flash | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1164 | N0047 | CatBoost + Stylometry | v1 | glm-5.3-flash | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1165 | N0049 | CatBoost + Stylometry | v1.5 | glm-5.3-flash | 21.1% | 23.8% | 13.3% | 4 | 15 | 19 |
| 1166 | N0034 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 22.2% | 23.7% | 26.7% | 8 | 28 | 36 |
| 1167 | N0046 | CatBoost + Stylometry | v1 | gemini-3.8-flash | 22.2% | 23.7% | 26.7% | 8 | 28 | 36 |
| 1168 | N0070 | Qwen3 0.6B | v1.1 | glm-5.3 | 21.9% | 23.5% | 23.3% | 7 | 25 | 32 |
| 1169 | N0077 | Qwen3 4B | v0 | gpt-5.6-terra | 21.9% | 23.5% | 23.3% | 7 | 25 | 32 |
| 1170 | N0080 | SVC | v0 | gemini-3.8-flash | 21.9% | 23.5% | 23.3% | 7 | 25 | 32 |
| 1171 | N0020 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 20.0% | 23.5% | 10.0% | 3 | 12 | 15 |
| 1172 | N0025 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 20.0% | 23.5% | 10.0% | 3 | 12 | 15 |
| 1173 | N0055 | EmbeddingGemma 2 | v1.1 | gpt-5.6-sol | 20.0% | 23.5% | 10.0% | 3 | 12 | 15 |
| 1174 | N0081 | SVC | v1 | glm-5.3 | 20.0% | 23.5% | 10.0% | 3 | 12 | 15 |
| 1175 | N0024 | CatBoost + Stylometry | v0 | gpt-5.5 | 22.2% | 23.4% | 33.3% | 10 | 35 | 45 |
| 1176 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.6-terra | 21.4% | 23.3% | 20.0% | 6 | 22 | 28 |
| 1177 | N0034 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 21.4% | 23.3% | 20.0% | 6 | 22 | 28 |
| 1178 | N0054 | EmbeddingGemma 2 | v1 | claude-fable-5 | 21.4% | 23.3% | 20.0% | 6 | 22 | 28 |
| 1179 | N0060 | Granite 4 H Micro | v0 | glm-5.3-flash | 21.4% | 23.3% | 20.0% | 6 | 22 | 28 |
| 1180 | N0082 | SVC | v1.1 | glm-5.3-flash | 22.6% | 23.3% | 63.3% | 19 | 65 | 84 |
| 1181 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | claude-opus-4-6 | 22.0% | 23.3% | 30.0% | 9 | 32 | 41 |
| 1182 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | qwen3.8-max | 22.4% | 23.2% | 50.0% | 15 | 52 | 67 |
| 1183 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gemini-3.7-flash | 21.6% | 23.1% | 26.7% | 8 | 29 | 37 |
| 1184 | N0019 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 21.6% | 23.1% | 26.7% | 8 | 29 | 37 |
| 1185 | N0067 | Qwen3 0.6B | v0 | kimi-k3 | 21.6% | 23.1% | 26.7% | 8 | 29 | 37 |
| 1186 | N0018 | CatBoost + Stylometry | v0 | glm-5.3-flash | 20.8% | 23.1% | 16.7% | 5 | 19 | 24 |
| 1187 | N0030 | CatBoost + Stylometry | v1 | claude-fable-5 | 20.8% | 23.1% | 16.7% | 5 | 19 | 24 |
| 1188 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | glm-5.3 | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1189 | N0028 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1190 | N0031 | CatBoost + Stylometry | v1 | glm-5.3 | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1191 | N0035 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1192 | N0037 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1193 | N0046 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1194 | N0048 | CatBoost + Stylometry | v1.1 | gpt-5.6-sol | 18.2% | 23.1% | 6.7% | 2 | 9 | 11 |
| 1195 | N0039 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 21.2% | 22.9% | 23.3% | 7 | 26 | 33 |
| 1196 | N0072 | Qwen3 0.6B | v1.5 | glm-5.3 | 21.2% | 22.9% | 23.3% | 7 | 26 | 33 |
| 1197 | N0015 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 21.4% | 22.7% | 30.0% | 9 | 33 | 42 |
| 1198 | N0052 | EmbeddingGemma 2 | v0 | gemini-3.8-flash | 21.4% | 22.7% | 30.0% | 9 | 33 | 42 |
| 1199 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | glm-5.3 | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1200 | N0014 | CatBoost + Stylometry | v0 | glm-5.3 | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1201 | N0018 | CatBoost + Stylometry | v0 | claude-fable-5 | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1202 | N0022 | CatBoost + Stylometry | v0 | kimi-k3 | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1203 | N0056 | EmbeddingGemma 2 | v1.1 | gemini-3.6-flash | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1204 | N0068 | Qwen3 0.6B | v0 | glm-5.3 | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1205 | N0077 | Qwen3 4B | v0 | claude-opus-4-8 | 20.0% | 22.7% | 13.3% | 4 | 16 | 20 |
| 1206 | N0040 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 20.7% | 22.6% | 20.0% | 6 | 23 | 29 |
| 1207 | N0024 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 21.1% | 22.5% | 26.7% | 8 | 30 | 38 |
| 1208 | N0073 | Qwen3 1.7B | v0 | kimi-k3 | 21.1% | 22.5% | 26.7% | 8 | 30 | 38 |
| 1209 | N0078 | SmolLM3 3B | v0 | claude-fable-5 | 21.1% | 22.5% | 26.7% | 8 | 30 | 38 |
| 1210 | N0060 | Granite 4 H Micro | v0 | gpt-5.5 | 21.3% | 22.4% | 33.3% | 10 | 37 | 47 |
| 1211 | N0045 | CatBoost + Stylometry | v1 | qwen3.8-max | 21.4% | 22.4% | 40.0% | 12 | 44 | 56 |
| 1212 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | qwen3.8-max | 21.5% | 22.4% | 46.7% | 14 | 51 | 65 |
| 1213 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | qwen3.8-max | 21.3% | 22.2% | 43.3% | 13 | 48 | 61 |
| 1214 | N0033 | CatBoost + Stylometry | v1 | qwen3.8-max | 21.3% | 22.2% | 43.3% | 13 | 48 | 61 |
| 1215 | N0071 | Qwen3 0.6B | v1.1 | glm-5.3 | 21.2% | 22.2% | 36.7% | 11 | 41 | 52 |
| 1216 | N0051 | EmbeddingGemma 2 | v0 | glm-5.3 | 20.9% | 22.2% | 30.0% | 9 | 34 | 43 |
| 1217 | N0034 | CatBoost + Stylometry | v1 | kimi-k3 | 20.6% | 22.2% | 23.3% | 7 | 27 | 34 |
| 1218 | N0037 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 20.6% | 22.2% | 23.3% | 7 | 27 | 34 |
| 1219 | N0060 | Granite 4 H Micro | v0 | gpt-5.6-sol | 20.6% | 22.2% | 23.3% | 7 | 27 | 34 |
| 1220 | N0019 | CatBoost + Stylometry | v0 | kimi-k3 | 20.0% | 22.2% | 16.7% | 5 | 20 | 25 |
| 1221 | N0028 | CatBoost + Stylometry | v1 | claude-fable-5 | 20.0% | 22.2% | 16.7% | 5 | 20 | 25 |
| 1222 | N0055 | EmbeddingGemma 2 | v1.1 | glm-5.3-flash | 20.0% | 22.2% | 16.7% | 5 | 20 | 25 |
| 1223 | N0015 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 18.8% | 22.2% | 10.0% | 3 | 13 | 16 |
| 1224 | N0020 | CatBoost + Stylometry | v0 | claude-fable-5 | 18.8% | 22.2% | 10.0% | 3 | 13 | 16 |
| 1225 | N0030 | CatBoost + Stylometry | v1 | glm-5.3 | 18.8% | 22.2% | 10.0% | 3 | 13 | 16 |
| 1226 | N0035 | CatBoost + Stylometry | v1 | gpt-5.6-terra | 18.8% | 22.2% | 10.0% | 3 | 13 | 16 |
| 1227 | N0043 | CatBoost + Stylometry | v1 | qwen3.8-max | 21.2% | 22.1% | 46.7% | 14 | 52 | 66 |
| 1228 | N0050 | CatBoost + Stylometry | v1.5 | qwen3.8-max | 21.2% | 22.1% | 46.7% | 14 | 52 | 66 |
| 1229 | N0035 | CatBoost + Stylometry | v1 | gemini-3.7-flash | 20.8% | 22.0% | 33.3% | 10 | 38 | 48 |
| 1230 | N0028 | CatBoost + Stylometry | v1 | qwen3.8-max | 21.0% | 21.9% | 43.3% | 13 | 49 | 62 |
| 1231 | N0031 | CatBoost + Stylometry | v1 | claude-fable-5 | 19.0% | 21.7% | 13.3% | 4 | 17 | 21 |
| 1232 | N0038 | CatBoost + Stylometry | v1 | kimi-k3 | 20.0% | 21.6% | 23.3% | 7 | 28 | 35 |
| 1233 | N0039 | CatBoost + Stylometry | v1 | gpt-5.5 | 20.0% | 21.6% | 23.3% | 7 | 28 | 35 |
| 1234 | N0054 | EmbeddingGemma 2 | v1 | gpt-5.6-sol | 20.0% | 21.6% | 23.3% | 7 | 28 | 35 |
| 1235 | N0048 | CatBoost + Stylometry | v1.1 | qwen3.8-max | 20.6% | 21.5% | 43.3% | 13 | 50 | 63 |
| 1236 | N0083 | SVC | v1.1 | glm-5.3-flash | 20.8% | 21.5% | 53.3% | 16 | 61 | 77 |
| 1237 | N0012 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 20.0% | 21.4% | 26.7% | 8 | 32 | 40 |
| 1238 | N0013 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 20.0% | 21.4% | 26.7% | 8 | 32 | 40 |
| 1239 | N0017 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 20.0% | 21.4% | 26.7% | 8 | 32 | 40 |
| 1240 | N0021 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 20.0% | 21.4% | 26.7% | 8 | 32 | 40 |
| 1241 | N0023 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 20.0% | 21.4% | 26.7% | 8 | 32 | 40 |
| 1242 | N0026 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 20.0% | 21.4% | 26.7% | 8 | 32 | 40 |
| 1243 | N0011 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 19.2% | 21.4% | 16.7% | 5 | 21 | 26 |
| 1244 | N0085 | SVC | v1.1 | gpt-5.6-luna | 19.2% | 21.4% | 16.7% | 5 | 21 | 26 |
| 1245 | N0036 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 16.7% | 21.4% | 6.7% | 2 | 10 | 12 |
| 1246 | N0067 | Qwen3 0.6B | v0 | gpt-5.6-sol | 20.0% | 21.3% | 30.0% | 9 | 36 | 45 |
| 1247 | N0037 | CatBoost + Stylometry | v1 | kimi-k3 | 19.4% | 21.2% | 20.0% | 6 | 25 | 31 |
| 1248 | N0052 | EmbeddingGemma 2 | v0 | gemini-3.6-flash | 19.4% | 21.2% | 20.0% | 6 | 25 | 31 |
| 1249 | N0073 | Qwen3 1.7B | v0 | gpt-5.6-terra | 19.4% | 21.2% | 20.0% | 6 | 25 | 31 |
| 1250 | N0016 | CatBoost + Stylometry | v0 | gpt-5.5 | 20.0% | 21.2% | 33.3% | 10 | 40 | 50 |
| 1251 | N0030 | CatBoost + Stylometry | v1 | qwen3.8-max | 20.3% | 21.1% | 46.7% | 14 | 55 | 69 |
| 1252 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gemini-3.7-flash | 19.4% | 21.1% | 23.3% | 7 | 29 | 36 |
| 1253 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.6-sol | 17.6% | 21.1% | 10.0% | 3 | 14 | 17 |
| 1254 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | gpt-5.6-sol | 17.6% | 21.1% | 10.0% | 3 | 14 | 17 |
| 1255 | N0016 | CatBoost + Stylometry | v0 | qwen3.8-max | 17.6% | 21.1% | 10.0% | 3 | 14 | 17 |
| 1256 | N0024 | CatBoost + Stylometry | v0 | claude-fable-5 | 17.6% | 21.1% | 10.0% | 3 | 14 | 17 |
| 1257 | N0035 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 17.6% | 21.1% | 10.0% | 3 | 14 | 17 |
| 1258 | N0076 | Qwen3 4B | v0 | gpt-5.5 | 20.0% | 20.9% | 43.3% | 13 | 52 | 65 |
| 1259 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-opus-4-6 | 19.6% | 20.8% | 30.0% | 9 | 37 | 46 |
| 1260 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-opus-4-6 | 19.6% | 20.8% | 30.0% | 9 | 37 | 46 |
| 1261 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | claude-fable-5 | 18.2% | 20.8% | 13.3% | 4 | 18 | 22 |
| 1262 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.6-sol | 18.2% | 20.8% | 13.3% | 4 | 18 | 22 |
| 1263 | N0020 | CatBoost + Stylometry | v0 | kimi-k3 | 18.2% | 20.8% | 13.3% | 4 | 18 | 22 |
| 1264 | N0027 | CatBoost + Stylometry | v0 | kimi-k3 | 18.2% | 20.8% | 13.3% | 4 | 18 | 22 |
| 1265 | N0076 | Qwen3 4B | v0 | claude-opus-4-8 | 18.2% | 20.8% | 13.3% | 4 | 18 | 22 |
| 1266 | N0085 | SVC | v1.1 | glm-5.3-flash | 20.0% | 20.8% | 50.0% | 15 | 60 | 75 |
| 1267 | N0084 | SVC | v1.1 | glm-5.3-flash | 20.0% | 20.7% | 53.3% | 16 | 64 | 80 |
| 1268 | N0035 | CatBoost + Stylometry | v1 | claude-opus-5 | 19.6% | 20.7% | 36.7% | 11 | 45 | 56 |
| 1269 | N0076 | Qwen3 4B | v0 | claude-fable-5 | 18.5% | 20.7% | 16.7% | 5 | 22 | 27 |
| 1270 | N0032 | CatBoost + Stylometry | v1 | kimi-k3 | 18.8% | 20.6% | 20.0% | 6 | 26 | 32 |
| 1271 | N0041 | CatBoost + Stylometry | v1 | kimi-k3 | 18.8% | 20.6% | 20.0% | 6 | 26 | 32 |
| 1272 | N0044 | CatBoost + Stylometry | v1 | kimi-k3 | 18.8% | 20.6% | 20.0% | 6 | 26 | 32 |
| 1273 | N0047 | CatBoost + Stylometry | v1 | kimi-k3 | 18.8% | 20.6% | 20.0% | 6 | 26 | 32 |
| 1274 | N0049 | CatBoost + Stylometry | v1.5 | kimi-k3 | 18.8% | 20.6% | 20.0% | 6 | 26 | 32 |
| 1275 | N0068 | Qwen3 0.6B | v0 | glm-5.3-flash | 18.8% | 20.6% | 20.0% | 6 | 26 | 32 |
| 1276 | N0025 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 18.9% | 20.5% | 23.3% | 7 | 30 | 37 |
| 1277 | N0062 | Llama 3.2 3B | v0 | glm-5.3-flash | 18.9% | 20.5% | 23.3% | 7 | 30 | 37 |
| 1278 | N0064 | Ministral 3 3B | v0 | glm-5.3-flash | 18.9% | 20.5% | 23.3% | 7 | 30 | 37 |
| 1279 | N0033 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 19.1% | 20.4% | 30.0% | 9 | 38 | 47 |
| 1280 | N0035 | CatBoost + Stylometry | v1 | claude-opus-4-6 | 19.1% | 20.4% | 30.0% | 9 | 38 | 47 |
| 1281 | N0031 | CatBoost + Stylometry | v1 | qwen3.8-max | 19.1% | 20.0% | 43.3% | 13 | 55 | 68 |
| 1282 | N0020 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 18.6% | 20.0% | 26.7% | 8 | 35 | 43 |
| 1283 | N0017 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 18.4% | 20.0% | 23.3% | 7 | 31 | 38 |
| 1284 | N0018 | CatBoost + Stylometry | v0 | gemini-3.6-flash | 18.2% | 20.0% | 20.0% | 6 | 27 | 33 |
| 1285 | N0036 | CatBoost + Stylometry | v1 | kimi-k3 | 18.2% | 20.0% | 20.0% | 6 | 27 | 33 |
| 1286 | N0046 | CatBoost + Stylometry | v1 | kimi-k3 | 18.2% | 20.0% | 20.0% | 6 | 27 | 33 |
| 1287 | N0073 | Qwen3 1.7B | v0 | gpt-5.6-sol | 18.2% | 20.0% | 20.0% | 6 | 27 | 33 |
| 1288 | N0014 | CatBoost + Stylometry | v0 | glm-5.3-flash | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1289 | N0032 | CatBoost + Stylometry | v1 | claude-fable-5 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1290 | N0033 | CatBoost + Stylometry | v1 | kimi-k3 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1291 | N0040 | CatBoost + Stylometry | v1 | kimi-k3 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1292 | N0041 | CatBoost + Stylometry | v1 | claude-fable-5 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1293 | N0044 | CatBoost + Stylometry | v1 | claude-fable-5 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1294 | N0047 | CatBoost + Stylometry | v1 | claude-fable-5 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1295 | N0049 | CatBoost + Stylometry | v1.5 | claude-fable-5 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1296 | N0073 | Qwen3 1.7B | v0 | glm-5.3-flash | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1297 | N0077 | Qwen3 4B | v0 | glm-5.3 | 17.9% | 20.0% | 16.7% | 5 | 23 | 28 |
| 1298 | N0024 | CatBoost + Stylometry | v0 | glm-5.3 | 17.4% | 20.0% | 13.3% | 4 | 19 | 23 |
| 1299 | N0025 | CatBoost + Stylometry | v0 | kimi-k3 | 17.4% | 20.0% | 13.3% | 4 | 19 | 23 |
| 1300 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.6-luna | 16.7% | 20.0% | 10.0% | 3 | 15 | 18 |
| 1301 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | glm-5.3-flash | 16.7% | 20.0% | 10.0% | 3 | 15 | 18 |
| 1302 | N0011 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 16.7% | 20.0% | 10.0% | 3 | 15 | 18 |
| 1303 | N0039 | CatBoost + Stylometry | v1 | kimi-k3 | 16.7% | 20.0% | 10.0% | 3 | 15 | 18 |
| 1304 | N0053 | EmbeddingGemma 2 | v0 | claude-sonnet-5 | 16.7% | 20.0% | 10.0% | 3 | 15 | 18 |
| 1305 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | glm-5.3 | 15.4% | 20.0% | 6.7% | 2 | 11 | 13 |
| 1306 | N0010 | CatBoost + Stylometry | v0 | claude-fable-5 | 15.4% | 20.0% | 6.7% | 2 | 11 | 13 |
| 1307 | N0016 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.4% | 20.0% | 6.7% | 2 | 11 | 13 |
| 1308 | N0017 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.4% | 20.0% | 6.7% | 2 | 11 | 13 |
| 1309 | N0060 | Granite 4 H Micro | v0 | gpt-5.6-luna | 15.4% | 20.0% | 6.7% | 2 | 11 | 13 |
| 1310 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | qwen3.8-max | 18.8% | 19.7% | 40.0% | 12 | 52 | 64 |
| 1311 | N0080 | SVC | v0 | gpt-6-astra | 18.9% | 19.6% | 60.0% | 18 | 77 | 95 |
| 1312 | N0045 | CatBoost + Stylometry | v1 | kimi-k3 | 17.6% | 19.4% | 20.0% | 6 | 28 | 34 |
| 1313 | N0073 | Qwen3 1.7B | v0 | claude-fable-5 | 17.6% | 19.4% | 20.0% | 6 | 28 | 34 |
| 1314 | N0029 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 17.2% | 19.4% | 16.7% | 5 | 24 | 29 |
| 1315 | N0048 | CatBoost + Stylometry | v1.1 | kimi-k3 | 17.2% | 19.4% | 16.7% | 5 | 24 | 29 |
| 1316 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | kimi-k3 | 16.7% | 19.2% | 13.3% | 4 | 20 | 24 |
| 1317 | N0017 | CatBoost + Stylometry | v0 | kimi-k3 | 16.7% | 19.2% | 13.3% | 4 | 20 | 24 |
| 1318 | N0036 | CatBoost + Stylometry | v1 | claude-fable-5 | 16.7% | 19.2% | 13.3% | 4 | 20 | 24 |
| 1319 | N0011 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 17.8% | 19.1% | 26.7% | 8 | 37 | 45 |
| 1320 | N0056 | EmbeddingGemma 2 | v1.1 | glm-5.3 | 17.8% | 19.1% | 26.7% | 8 | 37 | 45 |
| 1321 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.6-sol | 15.8% | 19.0% | 10.0% | 3 | 16 | 19 |
| 1322 | N0022 | CatBoost + Stylometry | v0 | glm-5.3-flash | 15.8% | 19.0% | 10.0% | 3 | 16 | 19 |
| 1323 | N0035 | CatBoost + Stylometry | v1 | qwen3.8-max | 15.8% | 19.0% | 10.0% | 3 | 16 | 19 |
| 1324 | N0077 | Qwen3 4B | v0 | claude-fable-5 | 15.8% | 19.0% | 10.0% | 3 | 16 | 19 |
| 1325 | N0062 | Llama 3.2 3B | v0 | glm-5.3 | 17.1% | 18.9% | 20.0% | 6 | 29 | 35 |
| 1326 | N0078 | SmolLM3 3B | v0 | gpt-5.6-luna | 17.6% | 18.9% | 30.0% | 9 | 42 | 51 |
| 1327 | N0045 | CatBoost + Stylometry | v1 | claude-opus-4-8 | 16.7% | 18.8% | 16.7% | 5 | 25 | 30 |
| 1328 | N0053 | EmbeddingGemma 2 | v0 | gemini-3.8-flash | 16.7% | 18.8% | 16.7% | 5 | 25 | 30 |
| 1329 | N0055 | EmbeddingGemma 2 | v1.1 | glm-5.3 | 16.7% | 18.8% | 16.7% | 5 | 25 | 30 |
| 1330 | N0076 | Qwen3 4B | v0 | glm-5.3 | 16.7% | 18.8% | 16.7% | 5 | 25 | 30 |
| 1331 | N0032 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1332 | N0033 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1333 | N0034 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1334 | N0041 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1335 | N0044 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1336 | N0047 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1337 | N0049 | CatBoost + Stylometry | v1.5 | gpt-5.6-sol | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1338 | N0081 | SVC | v1 | claude-fable-5 | 14.3% | 18.8% | 6.7% | 2 | 12 | 14 |
| 1339 | N0074 | Qwen3 1.7B | v0 | gpt-5.6-sol | 17.9% | 18.6% | 50.0% | 15 | 69 | 84 |
| 1340 | N0015 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 17.1% | 18.6% | 23.3% | 7 | 34 | 41 |
| 1341 | N0014 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 16.0% | 18.5% | 13.3% | 4 | 21 | 25 |
| 1342 | N0015 | CatBoost + Stylometry | v0 | kimi-k3 | 16.0% | 18.5% | 13.3% | 4 | 21 | 25 |
| 1343 | N0031 | CatBoost + Stylometry | v1 | kimi-k3 | 16.0% | 18.5% | 13.3% | 4 | 21 | 25 |
| 1344 | N0038 | CatBoost + Stylometry | v1 | claude-fable-5 | 16.0% | 18.5% | 13.3% | 4 | 21 | 25 |
| 1345 | N0042 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 16.0% | 18.5% | 13.3% | 4 | 21 | 25 |
| 1346 | N0056 | EmbeddingGemma 2 | v1.1 | glm-5.3-flash | 17.5% | 18.5% | 36.7% | 11 | 52 | 63 |
| 1347 | N0025 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 16.7% | 18.4% | 20.0% | 6 | 30 | 36 |
| 1348 | N0016 | CatBoost + Stylometry | v0 | claude-sonnet-5 | 16.1% | 18.2% | 16.7% | 5 | 26 | 31 |
| 1349 | N0051 | EmbeddingGemma 2 | v0 | gpt-5.6-terra | 16.1% | 18.2% | 16.7% | 5 | 26 | 31 |
| 1350 | N0053 | EmbeddingGemma 2 | v0 | gpt-5.6-terra | 16.1% | 18.2% | 16.7% | 5 | 26 | 31 |
| 1351 | N0069 | Qwen3 0.6B | v0 | claude-fable-5 | 16.1% | 18.2% | 16.7% | 5 | 26 | 31 |
| 1352 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | glm-5.3-flash | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1353 | N0012 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1354 | N0013 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1355 | N0021 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1356 | N0023 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1357 | N0026 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1358 | N0051 | EmbeddingGemma 2 | v0 | claude-fable-5 | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1359 | N0067 | Qwen3 0.6B | v0 | gpt-5.6-terra | 15.0% | 18.2% | 10.0% | 3 | 17 | 20 |
| 1360 | N0080 | SVC | v0 | kimi-k3 | 16.7% | 17.9% | 30.0% | 9 | 45 | 54 |
| 1361 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | claude-fable-5 | 15.4% | 17.9% | 13.3% | 4 | 22 | 26 |
| 1362 | N0018 | CatBoost + Stylometry | v0 | kimi-k3 | 15.4% | 17.9% | 13.3% | 4 | 22 | 26 |
| 1363 | N0051 | EmbeddingGemma 2 | v0 | gpt-5.5 | 16.3% | 17.6% | 26.7% | 8 | 41 | 49 |
| 1364 | N0019 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 13.3% | 17.6% | 6.7% | 2 | 13 | 15 |
| 1365 | N0025 | CatBoost + Stylometry | v0 | claude-fable-5 | 13.3% | 17.6% | 6.7% | 2 | 13 | 15 |
| 1366 | N0028 | CatBoost + Stylometry | v1 | kimi-k3 | 13.3% | 17.6% | 6.7% | 2 | 13 | 15 |
| 1367 | N0038 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 13.3% | 17.6% | 6.7% | 2 | 13 | 15 |
| 1368 | N0053 | EmbeddingGemma 2 | v0 | gpt-5.5 | 13.3% | 17.6% | 6.7% | 2 | 13 | 15 |
| 1369 | N0055 | EmbeddingGemma 2 | v1.1 | gemini-3.6-flash | 13.3% | 17.6% | 6.7% | 2 | 13 | 15 |
| 1370 | N0068 | Qwen3 0.6B | v0 | gpt-5.6-sol | 15.8% | 17.5% | 20.0% | 6 | 32 | 38 |
| 1371 | N0027 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 14.3% | 17.4% | 10.0% | 3 | 18 | 21 |
| 1372 | N0053 | EmbeddingGemma 2 | v0 | claude-opus-4-8 | 14.3% | 17.4% | 10.0% | 3 | 18 | 21 |
| 1373 | N0016 | CatBoost + Stylometry | v0 | gpt-5.6-terra | 14.8% | 17.2% | 13.3% | 4 | 23 | 27 |
| 1374 | N0067 | Qwen3 0.6B | v0 | glm-5.3 | 14.8% | 17.2% | 13.3% | 4 | 23 | 27 |
| 1375 | N0035 | CatBoost + Stylometry | v1 | claude-sonnet-5 | 16.1% | 17.2% | 33.3% | 10 | 52 | 62 |
| 1376 | N0016 | CatBoost + Stylometry | v0 | glm-5.3 | 15.2% | 17.1% | 16.7% | 5 | 28 | 33 |
| 1377 | N0051 | EmbeddingGemma 2 | v0 | claude-opus-4-8 | 15.2% | 17.1% | 16.7% | 5 | 28 | 33 |
| 1378 | N0077 | Qwen3 4B | v0 | gpt-5.5 | 15.8% | 16.9% | 30.0% | 9 | 48 | 57 |
| 1379 | N0068 | Qwen3 0.6B | v0 | claude-fable-5 | 15.4% | 16.7% | 26.7% | 8 | 44 | 52 |
| 1380 | N0060 | Granite 4 H Micro | v0 | glm-5.3 | 15.0% | 16.7% | 20.0% | 6 | 34 | 40 |
| 1381 | N0082 | SVC | v1.1 | claude-fable-5 | 15.0% | 16.7% | 20.0% | 6 | 34 | 40 |
| 1382 | N0039 | CatBoost + Stylometry | v1 | claude-fable-5 | 14.7% | 16.7% | 16.7% | 5 | 29 | 34 |
| 1383 | N0051 | EmbeddingGemma 2 | v0 | gemini-3.8-flash | 14.7% | 16.7% | 16.7% | 5 | 29 | 34 |
| 1384 | N0083 | SVC | v1.1 | claude-fable-5 | 14.7% | 16.7% | 16.7% | 5 | 29 | 34 |
| 1385 | N0085 | SVC | v1.1 | claude-fable-5 | 14.7% | 16.7% | 16.7% | 5 | 29 | 34 |
| 1386 | N0056 | EmbeddingGemma 2 | v1.1 | gpt-5.6-sol | 14.3% | 16.7% | 13.3% | 4 | 24 | 28 |
| 1387 | N0012 | CatBoost + Stylometry | v0 | kimi-k3 | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1388 | N0013 | CatBoost + Stylometry | v0 | kimi-k3 | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1389 | N0021 | CatBoost + Stylometry | v0 | kimi-k3 | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1390 | N0023 | CatBoost + Stylometry | v0 | kimi-k3 | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1391 | N0026 | CatBoost + Stylometry | v0 | kimi-k3 | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1392 | N0052 | EmbeddingGemma 2 | v0 | glm-5.3-flash | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1393 | N0052 | EmbeddingGemma 2 | v0 | kimi-k3 | 13.6% | 16.7% | 10.0% | 3 | 19 | 22 |
| 1394 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | glm-5.3 | 12.5% | 16.7% | 6.7% | 2 | 14 | 16 |
| 1395 | N0014 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 12.5% | 16.7% | 6.7% | 2 | 14 | 16 |
| 1396 | N0029 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 12.5% | 16.7% | 6.7% | 2 | 14 | 16 |
| 1397 | N0046 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 12.5% | 16.7% | 6.7% | 2 | 14 | 16 |
| 1398 | N0080 | SVC | v0 | glm-5.3 | 12.5% | 16.7% | 6.7% | 2 | 14 | 16 |
| 1399 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | glm-5.3 | 10.0% | 16.7% | 3.3% | 1 | 9 | 10 |
| 1400 | N0016 | CatBoost + Stylometry | v0 | glm-5.3-flash | 10.0% | 16.7% | 3.3% | 1 | 9 | 10 |
| 1401 | N0025 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 10.0% | 16.7% | 3.3% | 1 | 9 | 10 |
| 1402 | N0042 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 10.0% | 16.7% | 3.3% | 1 | 9 | 10 |
| 1403 | N0059 | Granite 4.0 350M Base | v1.1 | glm-5.3 | 10.0% | 16.7% | 3.3% | 1 | 9 | 10 |
| 1404 | N0016 | CatBoost + Stylometry | v0 | gpt-6-astra | 14.6% | 16.3% | 20.0% | 6 | 35 | 41 |
| 1405 | N0051 | EmbeddingGemma 2 | v0 | kimi-k3 | 14.6% | 16.3% | 20.0% | 6 | 35 | 41 |
| 1406 | N0048 | CatBoost + Stylometry | v1.1 | claude-fable-5 | 13.8% | 16.1% | 13.3% | 4 | 25 | 29 |
| 1407 | N0008 | CatBoost + Stylometry | catboost_v1_a5_s2029 | kimi-k3 | 13.0% | 16.0% | 10.0% | 3 | 20 | 23 |
| 1408 | N0073 | Qwen3 1.7B | v0 | gpt-5.5 | 14.5% | 15.8% | 26.7% | 8 | 47 | 55 |
| 1409 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | gpt-5.6-sol | 11.8% | 15.8% | 6.7% | 2 | 15 | 17 |
| 1410 | N0037 | CatBoost + Stylometry | v1 | claude-fable-5 | 11.8% | 15.8% | 6.7% | 2 | 15 | 17 |
| 1411 | N0052 | EmbeddingGemma 2 | v0 | gpt-5.6-sol | 11.8% | 15.8% | 6.7% | 2 | 15 | 17 |
| 1412 | N0064 | Ministral 3 3B | v0 | claude-fable-5 | 11.8% | 15.8% | 6.7% | 2 | 15 | 17 |
| 1413 | N0006 | CatBoost + Stylometry | catboost_v1_a5_s2028 | kimi-k3 | 12.5% | 15.4% | 10.0% | 3 | 21 | 24 |
| 1414 | N0024 | CatBoost + Stylometry | v0 | kimi-k3 | 12.5% | 15.4% | 10.0% | 3 | 21 | 24 |
| 1415 | N0030 | CatBoost + Stylometry | v1 | kimi-k3 | 12.5% | 15.4% | 10.0% | 3 | 21 | 24 |
| 1416 | N0043 | CatBoost + Stylometry | v1 | glm-5.3 | 9.1% | 15.4% | 3.3% | 1 | 10 | 11 |
| 1417 | N0045 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 9.1% | 15.4% | 3.3% | 1 | 10 | 11 |
| 1418 | N0050 | CatBoost + Stylometry | v1.5 | glm-5.3 | 9.1% | 15.4% | 3.3% | 1 | 10 | 11 |
| 1419 | N0073 | Qwen3 1.7B | v0 | glm-5.3 | 9.1% | 15.4% | 3.3% | 1 | 10 | 11 |
| 1420 | N0067 | Qwen3 0.6B | v0 | claude-fable-5 | 13.6% | 15.2% | 20.0% | 6 | 38 | 44 |
| 1421 | N0029 | CatBoost + Stylometry | v1 | kimi-k3 | 12.9% | 15.2% | 13.3% | 4 | 27 | 31 |
| 1422 | N0042 | CatBoost + Stylometry | v1 | claude-fable-5 | 12.9% | 15.2% | 13.3% | 4 | 27 | 31 |
| 1423 | N0051 | EmbeddingGemma 2 | v0 | gpt-5.6-luna | 12.9% | 15.2% | 13.3% | 4 | 27 | 31 |
| 1424 | N0084 | SVC | v1.1 | claude-fable-5 | 12.9% | 15.2% | 13.3% | 4 | 27 | 31 |
| 1425 | N0012 | CatBoost + Stylometry | v0 | claude-fable-5 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1426 | N0013 | CatBoost + Stylometry | v0 | claude-fable-5 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1427 | N0015 | CatBoost + Stylometry | v0 | claude-fable-5 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1428 | N0021 | CatBoost + Stylometry | v0 | claude-fable-5 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1429 | N0023 | CatBoost + Stylometry | v0 | claude-fable-5 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1430 | N0026 | CatBoost + Stylometry | v0 | claude-fable-5 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1431 | N0042 | CatBoost + Stylometry | v1 | kimi-k3 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1432 | N0043 | CatBoost + Stylometry | v1 | kimi-k3 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1433 | N0050 | CatBoost + Stylometry | v1.5 | kimi-k3 | 11.1% | 15.0% | 6.7% | 2 | 16 | 18 |
| 1434 | N0005 | CatBoost + Stylometry | catboost_v1_a5_s2027 | kimi-k3 | 12.0% | 14.8% | 10.0% | 3 | 22 | 25 |
| 1435 | N0040 | CatBoost + Stylometry | v1 | claude-fable-5 | 12.0% | 14.8% | 10.0% | 3 | 22 | 25 |
| 1436 | N0052 | EmbeddingGemma 2 | v0 | gpt-5.5 | 12.0% | 14.8% | 10.0% | 3 | 22 | 25 |
| 1437 | N0055 | EmbeddingGemma 2 | v1.1 | claude-fable-5 | 13.7% | 14.7% | 33.3% | 10 | 63 | 73 |
| 1438 | N0014 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 13.0% | 14.6% | 20.0% | 6 | 40 | 46 |
| 1439 | N0035 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 12.1% | 14.3% | 13.3% | 4 | 29 | 33 |
| 1440 | N0007 | CatBoost + Stylometry | catboost_v1_a5_s2028 | kimi-k3 | 11.5% | 14.3% | 10.0% | 3 | 23 | 26 |
| 1441 | N0029 | CatBoost + Stylometry | v1 | claude-fable-5 | 11.5% | 14.3% | 10.0% | 3 | 23 | 26 |
| 1442 | N0033 | CatBoost + Stylometry | v1 | claude-fable-5 | 11.5% | 14.3% | 10.0% | 3 | 23 | 26 |
| 1443 | N0051 | EmbeddingGemma 2 | v0 | glm-5.3-flash | 11.5% | 14.3% | 10.0% | 3 | 23 | 26 |
| 1444 | N0014 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 10.5% | 14.3% | 6.7% | 2 | 17 | 19 |
| 1445 | N0045 | CatBoost + Stylometry | v1 | claude-fable-5 | 10.5% | 14.3% | 6.7% | 2 | 17 | 19 |
| 1446 | N0046 | CatBoost + Stylometry | v1 | claude-fable-5 | 10.5% | 14.3% | 6.7% | 2 | 17 | 19 |
| 1447 | N0035 | CatBoost + Stylometry | v1 | gpt-5.5 | 12.5% | 13.8% | 23.3% | 7 | 49 | 56 |
| 1448 | N0004 | CatBoost + Stylometry | catboost_v1_a5_s2027 | kimi-k3 | 11.1% | 13.8% | 10.0% | 3 | 24 | 27 |
| 1449 | N0034 | CatBoost + Stylometry | v1 | claude-fable-5 | 11.1% | 13.8% | 10.0% | 3 | 24 | 27 |
| 1450 | N0016 | CatBoost + Stylometry | v0 | claude-opus-5 | 12.5% | 13.5% | 30.0% | 9 | 63 | 72 |
| 1451 | N0018 | CatBoost + Stylometry | v0 | claude-opus-4-8 | 11.6% | 13.3% | 16.7% | 5 | 38 | 43 |
| 1452 | N0016 | CatBoost + Stylometry | v0 | gpt-5.6-sol | 10.7% | 13.3% | 10.0% | 3 | 25 | 28 |
| 1453 | N0029 | CatBoost + Stylometry | v1 | gpt-5.6-sol | 7.7% | 13.3% | 3.3% | 1 | 12 | 13 |
| 1454 | N0083 | SVC | v1.1 | glm-5.3 | 12.2% | 13.2% | 30.0% | 9 | 65 | 74 |
| 1455 | N0010 | CatBoost + Stylometry | v0 | kimi-k3 | 9.5% | 13.0% | 6.7% | 2 | 19 | 21 |
| 1456 | N0053 | EmbeddingGemma 2 | v0 | claude-fable-5 | 10.3% | 12.9% | 10.0% | 3 | 26 | 29 |
| 1457 | N0085 | SVC | v1.1 | glm-5.3 | 11.8% | 12.8% | 30.0% | 9 | 67 | 76 |
| 1458 | N0078 | SmolLM3 3B | v0 | gpt-5.6-sol | 11.1% | 12.8% | 16.7% | 5 | 40 | 45 |
| 1459 | N0053 | EmbeddingGemma 2 | v0 | kimi-k3 | 11.3% | 12.5% | 23.3% | 7 | 55 | 62 |
| 1460 | N0053 | EmbeddingGemma 2 | v0 | glm-5.3-flash | 10.9% | 12.5% | 16.7% | 5 | 41 | 46 |
| 1461 | N0016 | CatBoost + Stylometry | v0 | kimi-k3 | 10.0% | 12.5% | 10.0% | 3 | 27 | 30 |
| 1462 | N0019 | CatBoost + Stylometry | v0 | claude-fable-5 | 9.1% | 12.5% | 6.7% | 2 | 20 | 22 |
| 1463 | N0009 | CatBoost + Stylometry | catboost_v1_a5_s2029 | gpt-5.6-luna | 7.1% | 12.5% | 3.3% | 1 | 13 | 14 |
| 1464 | N0035 | CatBoost + Stylometry | v1 | gemini-3.6-flash | 7.1% | 12.5% | 3.3% | 1 | 13 | 14 |
| 1465 | N0051 | EmbeddingGemma 2 | v0 | gpt-5.6-sol | 7.1% | 12.5% | 3.3% | 1 | 13 | 14 |
| 1466 | N0056 | EmbeddingGemma 2 | v1.1 | claude-fable-5 | 10.3% | 12.2% | 13.3% | 4 | 35 | 39 |
| 1467 | N0084 | SVC | v1.1 | glm-5.3 | 11.1% | 12.2% | 26.7% | 8 | 64 | 72 |
| 1468 | N0022 | CatBoost + Stylometry | v0 | claude-fable-5 | 8.7% | 12.0% | 6.7% | 2 | 21 | 23 |
| 1469 | N0014 | CatBoost + Stylometry | v0 | kimi-k3 | 6.7% | 11.8% | 3.3% | 1 | 14 | 15 |
| 1470 | N0011 | CatBoost + Stylometry | v0 | kimi-k3 | 8.3% | 11.5% | 6.7% | 2 | 22 | 24 |
| 1471 | N0082 | SVC | v1.1 | glm-5.3 | 10.2% | 11.5% | 20.0% | 6 | 53 | 59 |
| 1472 | N0077 | Qwen3 4B | v0 | glm-5.3-flash | 9.1% | 11.4% | 10.0% | 3 | 30 | 33 |
| 1473 | N0052 | EmbeddingGemma 2 | v0 | gpt-5.6-luna | 8.0% | 11.1% | 6.7% | 2 | 23 | 25 |
| 1474 | N0011 | CatBoost + Stylometry | v0 | claude-fable-5 | 5.9% | 10.5% | 3.3% | 1 | 16 | 17 |
| 1475 | N0045 | CatBoost + Stylometry | v1 | gpt-5.6-luna | 5.9% | 10.5% | 3.3% | 1 | 16 | 17 |
| 1476 | N0075 | Qwen3 1.7B | v1.1 | glm-5.3 | 5.9% | 10.5% | 3.3% | 1 | 16 | 17 |
| 1477 | N0010 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 5.6% | 10.0% | 3.3% | 1 | 17 | 18 |
| 1478 | N0017 | CatBoost + Stylometry | v0 | claude-fable-5 | 5.6% | 10.0% | 3.3% | 1 | 17 | 18 |
| 1479 | N0076 | Qwen3 4B | v0 | glm-5.3-flash | 7.5% | 9.5% | 10.0% | 3 | 37 | 40 |
| 1480 | N0053 | EmbeddingGemma 2 | v0 | gpt-5.6-luna | 6.5% | 9.1% | 6.7% | 2 | 29 | 31 |
| 1481 | N0053 | EmbeddingGemma 2 | v0 | gpt-5.6-sol | 6.5% | 9.1% | 6.7% | 2 | 29 | 31 |
| 1482 | N0052 | EmbeddingGemma 2 | v0 | gpt-5.6-terra | 6.5% | 8.3% | 10.0% | 3 | 43 | 46 |
| 1483 | N0024 | CatBoost + Stylometry | v0 | gpt-5.6-luna | 0.0% | 8.3% | 0.0% | 0 | 10 | 10 |
| 1484 | N0035 | CatBoost + Stylometry | v1 | gpt-6-astra | 6.1% | 7.8% | 10.0% | 3 | 46 | 49 |
| 1485 | N0074 | Qwen3 1.7B | v0 | glm-5.3 | 0.0% | 7.7% | 0.0% | 0 | 11 | 11 |

## Best precision checkpoint per label

Chosen from all evaluation-eligible checkpoints by smoothed precision, then raw precision.

| Label | Key | Model | Dataset | Precision (smoothed) | Recall | n |
|---|---|---|---|---:|---:|---:|
| claude-fable-5 | A | UltraBERT-380M | v1 | 47.1% (47.4%) | 26.7% | 17 |
| claude-opus-4-6 | N0058 | Gemma 4 E2B | v1.1 | 85.7% (83.3%) | 80.0% | 28 |
| claude-opus-4-8 | D | UltraBERT-380M | v0 | 84.2% (81.0%) | 53.3% | 19 |
| claude-opus-5 | N0082 | SVC | v1.1 | 85.7% (81.2%) | 40.0% | 14 |
| claude-sonnet-5 | N0058 | Gemma 4 E2B | v1.1 | 82.6% (80.0%) | 63.3% | 23 |
| gemini-3.1-pro-preview | N0070 | Qwen3 0.6B | v1.1 | 100.0% (93.8%) | 46.7% | 14 |
| gemini-3.6-flash | N0083 | SVC | v1.1 | 61.9% (60.9%) | 43.3% | 21 |
| gemini-3.7-flash | N0082 | SVC | v1.1 | 58.8% (57.9%) | 33.3% | 17 |
| gemini-3.8-flash | C | UltraBERT-380M | v1.5 | 73.3% (70.6%) | 36.7% | 15 |
| glm-5.3 | N0018 | CatBoost + Stylometry | v0 | 36.8% (38.1%) | 23.3% | 19 |
| glm-5.3-flash | N0031 | CatBoost + Stylometry | v1 | 60.0% (58.3%) | 20.0% | 10 |
| gpt-5.5 | N0065 | Ministral 3 3B | v1.1 | 70.0% (68.2%) | 46.7% | 20 |
| gpt-5.6-luna | N0075 | Qwen3 1.7B | v1.1 | 64.3% (62.5%) | 30.0% | 14 |
| gpt-5.6-sol | N0071 | Qwen3 0.6B | v1.1 | 60.0% (58.3%) | 20.0% | 10 |
| gpt-5.6-terra | N0057 | EmbeddingGemma 2 | v1.5 | 50.0% (50.0%) | 23.3% | 14 |
| gpt-6-astra | N0072 | Qwen3 0.6B | v1.5 | 95.8% (92.3%) | 76.7% | 24 |
| kimi-k3 | D | UltraBERT-380M | v0 | 57.1% (56.2%) | 26.7% | 14 |
| qwen3.8-max | C | UltraBERT-380M | v1.5 | 72.4% (71.0%) | 70.0% | 29 |

## Best additional checkpoint vs UltraBERT

The new-vs-UltraBERT comparison uses smoothed precision first, then raw precision. 'Outperforms' means the best additional checkpoint's smoothed precision exceeds the best supported UltraBERT A–D rule for that label.

| Label | Best additional checkpoint | Precision (smoothed) | Recall | n | Best UltraBERT | Precision (smoothed) | n | New improves? |
|---|---|---:|---:|---:|---|---:|---:|---|
| claude-fable-5 | N0065 Ministral 3 3B (v1.1) | 36.4% (37.0%) | 53.3% | 44 | A | 47.1% (47.4%) | 17 | no |
| claude-opus-4-6 | N0058 Gemma 4 E2B (v1.1) | 85.7% (83.3%) | 80.0% | 28 | B | 85.0% (81.8%) | 20 | yes |
| claude-opus-4-8 | N0058 Gemma 4 E2B (v1.1) | 81.2% (77.8%) | 43.3% | 16 | D | 84.2% (81.0%) | 19 | no |
| claude-opus-5 | N0082 SVC (v1.1) | 85.7% (81.2%) | 40.0% | 14 | D | 76.9% (75.0%) | 26 | yes |
| claude-sonnet-5 | N0058 Gemma 4 E2B (v1.1) | 82.6% (80.0%) | 63.3% | 23 | B | 76.2% (73.9%) | 21 | yes |
| gemini-3.1-pro-preview | N0070 Qwen3 0.6B (v1.1) | 100.0% (93.8%) | 46.7% | 14 | C | 96.0% (92.6%) | 25 | yes |
| gemini-3.6-flash | N0083 SVC (v1.1) | 61.9% (60.9%) | 43.3% | 21 | C | 54.2% (53.8%) | 24 | yes |
| gemini-3.7-flash | N0082 SVC (v1.1) | 58.8% (57.9%) | 33.3% | 17 | B | 47.6% (47.8%) | 21 | yes |
| gemini-3.8-flash | N0061 Granite 4 H Micro (v1.1) | 63.6% (62.5%) | 46.7% | 22 | C | 73.3% (70.6%) | 15 | no |
| glm-5.3 | N0018 CatBoost + Stylometry (v0) | 36.8% (38.1%) | 23.3% | 19 | D | 28.0% (29.6%) | 25 | yes |
| glm-5.3-flash | N0031 CatBoost + Stylometry (v1) | 60.0% (58.3%) | 20.0% | 10 | C | 42.9% (43.2%) | 35 | yes |
| gpt-5.5 | N0065 Ministral 3 3B (v1.1) | 70.0% (68.2%) | 46.7% | 20 | C | 53.6% (53.3%) | 28 | yes |
| gpt-5.6-luna | N0075 Qwen3 1.7B (v1.1) | 64.3% (62.5%) | 30.0% | 14 | C | 40.0% (40.6%) | 30 | yes |
| gpt-5.6-sol | N0071 Qwen3 0.6B (v1.1) | 60.0% (58.3%) | 20.0% | 10 | D | 55.0% (54.5%) | 20 | yes |
| gpt-5.6-terra | N0057 EmbeddingGemma 2 (v1.5) | 50.0% (50.0%) | 23.3% | 14 | A | 44.4% (44.7%) | 36 | yes |
| gpt-6-astra | N0072 Qwen3 0.6B (v1.5) | 95.8% (92.3%) | 76.7% | 24 | B | 95.5% (91.7%) | 22 | yes |
| kimi-k3 | N0061 Granite 4 H Micro (v1.1) | 50.0% (50.0%) | 46.7% | 28 | D | 57.1% (56.2%) | 14 | no |
| qwen3.8-max | N0079 SmolLM3 3B (v1.5) | 68.0% (66.7%) | 56.7% | 25 | C | 72.4% (71.0%) | 29 | no |

### Additional specialists exceeding UltraBERT precision

| Label | New specialist | Dataset | Precision (smoothed) | Recall | n | UltraBERT best | Precision (smoothed) |
|---|---|---|---:|---:|---:|---|---:|
| claude-opus-4-6 | N0058 Gemma 4 E2B | v1.1 | 85.7% (83.3%) | 80.0% | 28 | B | 85.0% (81.8%) |
| claude-opus-5 | N0082 SVC | v1.1 | 85.7% (81.2%) | 40.0% | 14 | D | 76.9% (75.0%) |
| claude-sonnet-5 | N0058 Gemma 4 E2B | v1.1 | 82.6% (80.0%) | 63.3% | 23 | B | 76.2% (73.9%) |
| gemini-3.1-pro-preview | N0070 Qwen3 0.6B | v1.1 | 100.0% (93.8%) | 46.7% | 14 | C | 96.0% (92.6%) |
| gemini-3.6-flash | N0083 SVC | v1.1 | 61.9% (60.9%) | 43.3% | 21 | C | 54.2% (53.8%) |
| gemini-3.7-flash | N0082 SVC | v1.1 | 58.8% (57.9%) | 33.3% | 17 | B | 47.6% (47.8%) |
| glm-5.3 | N0018 CatBoost + Stylometry | v0 | 36.8% (38.1%) | 23.3% | 19 | D | 28.0% (29.6%) |
| glm-5.3-flash | N0031 CatBoost + Stylometry | v1 | 60.0% (58.3%) | 20.0% | 10 | C | 42.9% (43.2%) |
| gpt-5.5 | N0065 Ministral 3 3B | v1.1 | 70.0% (68.2%) | 46.7% | 20 | C | 53.6% (53.3%) |
| gpt-5.6-luna | N0075 Qwen3 1.7B | v1.1 | 64.3% (62.5%) | 30.0% | 14 | C | 40.0% (40.6%) |
| gpt-5.6-sol | N0071 Qwen3 0.6B | v1.1 | 60.0% (58.3%) | 20.0% | 10 | D | 55.0% (54.5%) |
| gpt-5.6-terra | N0057 EmbeddingGemma 2 | v1.5 | 50.0% (50.0%) | 23.3% | 14 | A | 44.4% (44.7%) |
| gpt-6-astra | N0072 Qwen3 0.6B | v1.5 | 95.8% (92.3%) | 76.7% | 24 | B | 95.5% (91.7%) |

## Cross-fitted cascade check

As a direct sanity check, the original A–D cascade and the expanded candidate cascade were both fit and evaluated with identical five-fold prompt-group splits. Each fold fits rule order on four groups partitions and predicts the fifth.

| Candidate set | Correct | Accuracy | Delta vs A–D | Corrected vs A–D | Regressed vs A–D |
|---|---:|---:|---:|---:|---:|
| UltraBERT A–D | 290/540 | 53.7% | — | — | — |
| All evaluation-eligible checkpoints | 232/540 | 43.0% | -10.7% | 34 | 92 |
| Add new rules ≥70% smoothed precision, then A–D | 267/540 | 49.4% | -4.3% | 14 | 37 |
| Add new rules ≥75% smoothed precision, then A–D | 268/540 | 49.6% | -4.1% | 11 | 33 |
| Add new rules ≥80% smoothed precision, then A–D | 279/540 | 51.7% | -2.0% | 8 | 19 |
| Add new rules ≥85% smoothed precision, then A–D | 284/540 | 52.6% | -1.1% | 4 | 10 |
| Add new rules ≥90% smoothed precision, then A–D | 289/540 | 53.5% | -0.2% | 2 | 3 |

The expanded cascade includes only evaluation-eligible candidates. The cascade check is an added grouped development diagnostic. Exact per-row predictions and selected checkpoint keys are in `cascade_comparison_oof.parquet`; rule traces are in `zod01_ultrabert_rule_trace.csv` and `zod01_expanded_rule_trace.csv`.

## Artifacts and interpretation

Reusable V0 probability files are stored in `artifacts/analysis/zod01_specialist_discovery/predictions/`. `checkpoint_inventory.csv` maps compact keys to exact experiment IDs, checkpoint paths, training dataset, selected epoch, prediction source and V0 accuracy. `ranked_checkpoint_label_rules.csv` contains every checkpoint-label pair, including unsupported pairs. `metrics.json` contains per-label best rules, threshold coverage and grouped cascade metrics.

A high-precision rule may have low recall and only cover the subset of rows that its checkpoint predicts as that label. Precision estimates use 540 previously inspected/used validation labels; adding models can improve rule coverage, but these estimates can be optimistic and require confirmation on unseen examples.
