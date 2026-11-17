# Recall-based cascade experiment

Cache-only evaluation using Trust List 2 v025 and the existing full-V0 checkpoint probability predictions. No model inference, base-model training, production-ensemble changes, or Kaggle submission was performed.

Trust-list SHA-256: `c93ae4cc0612ed36b7c126d6d50a5b50456ecfafd4e835369272076442e8cf9b`. Cached checkpoint predictions: 78. V0 rows: 2699; prompt groups: 150. Existing five prompt-group folds and the minimum fitting-fold rule support of 10 were reused.

Cross-fitted results are primary. Although rule selection is fit-fold-only, checkpoints were previously selected using V0, so held-out folds do not remove checkpoint-selection exposure. Full-V0 diagnostics fit and score on the same rows and are optimistic descriptive results.

EQUAL9 baseline: 65.3946% accuracy, 65.3898% balanced accuracy (1765/2699 correct). C1 and C3 values below are taken from the existing aligned cascade prediction artifacts.

## Overall comparison

| Scope | Strategy | Rules | Accuracy | Balanced accuracy | Correct | Δ BA vs EQUAL9 | Corrected | Introduced | Disagreements | Fallback rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt_group_oof | EQUAL9 | 0.0 | 65.3946% | 65.3898% | 1765/2699 | +0.000 pp | 0 | 0 | 0 | 0 |
| prompt_group_oof | C1_precision | 0.0 | 64.8018% | 64.8006% | 1749/2699 | -0.589 pp | 103 | 119 | 310 | 0 |
| prompt_group_oof | C3_precision | 0.0 | 65.5057% | 65.5006% | 1768/2699 | +0.111 pp | 18 | 15 | 43 | 0 |
| prompt_group_oof | R1_highest_recall | 18.0 | 52.3898% | 52.3756% | 1414/2699 | -13.014 pp | 191 | 542 | 1074 | 115 |
| prompt_group_oof | R2_recall_x_prediction_count | 18.0 | 40.0889% | 40.0766% | 1082/2699 | -25.313 pp | 141 | 824 | 1412 | 66 |
| prompt_group_oof | R3_recall_x_true_label_count | 18.0 | 52.3898% | 52.3756% | 1414/2699 | -13.014 pp | 191 | 542 | 1074 | 115 |
| full_v0_diagnostic | EQUAL9 | 0.0 | 65.3946% | 65.3898% | 1765/2699 | +0.000 pp | 0 | 0 | 0 | 0 |
| full_v0_diagnostic | C1_precision | 0.0 | 66.1727% | 66.1718% | 1786/2699 | +0.782 pp | 118 | 97 | 288 | 0 |
| full_v0_diagnostic | C3_precision | 0.0 | 65.8021% | 65.7972% | 1776/2699 | +0.407 pp | 17 | 6 | 28 | 0 |
| full_v0_diagnostic | R1_highest_recall | 18.0 | 52.8714% | 52.8581% | 1427/2699 | -12.532 pp | 195 | 533 | 1070 | 92 |
| full_v0_diagnostic | R2_recall_x_prediction_count | 18.0 | 40.0148% | 40.0032% | 1080/2699 | -25.387 pp | 137 | 822 | 1412 | 56 |
| full_v0_diagnostic | R3_recall_x_true_label_count | 18.0 | 52.8714% | 52.8581% | 1427/2699 | -12.532 pp | 195 | 533 | 1070 | 92 |

## Rule selection and order

R1 selects each label's checkpoint by highest fit-fold recall. R2 selects by recall multiplied by that checkpoint's top-1 prediction count for the label. R3 selects by recall multiplied by the true-label support, which equals TP. Candidates with fewer than 10 matching fit-fold top-1 predictions are excluded. Ties are deterministic: selection ties use recall, precision, then lowest checkpoint sequence; global order ties use canonical label order then checkpoint sequence.

### Full-V0 diagnostic rules

**R1_highest_recall**
1. `gemini-3.8-flash` ← N0070 (ultrabert380); selection score 0.9267, recall 0.9267, predictions 353, TP 139.
2. `gemini-3.1-pro-preview` ← N0062 (ultrabert380); selection score 0.9200, recall 0.9200, predictions 221, TP 138.
3. `gemini-3.6-flash` ← N0081 (granite4_h_micro); selection score 0.9000, recall 0.9000, predictions 242, TP 135.
4. `claude-opus-4-6` ← N0062 (ultrabert380); selection score 0.8533, recall 0.8533, predictions 264, TP 128.
5. `gpt-6-astra` ← N0081 (granite4_h_micro); selection score 0.8533, recall 0.8533, predictions 152, TP 128.
6. `claude-opus-5` ← N0069 (ultrabert380); selection score 0.8467, recall 0.8467, predictions 197, TP 127.
7. `claude-opus-4-8` ← N0030 (embeddinggemma2); selection score 0.8133, recall 0.8133, predictions 330, TP 122.
8. `qwen3.8-max` ← N0063 (ultrabert380); selection score 0.7800, recall 0.7800, predictions 234, TP 117.
9. `gemini-3.7-flash` ← N0065 (ultrabert380); selection score 0.7733, recall 0.7733, predictions 340, TP 116.
10. `claude-sonnet-5` ← N0105 (ultrabert380); selection score 0.7533, recall 0.7533, predictions 181, TP 113.
11. `gpt-5.6-luna` ← N0106 (qwen06); selection score 0.7467, recall 0.7467, predictions 238, TP 112.
12. `gpt-5.5` ← N0105 (ultrabert380); selection score 0.7267, recall 0.7267, predictions 263, TP 109.
13. `gpt-5.6-terra` ← N0038 (ministral3_3b); selection score 0.6933, recall 0.6933, predictions 259, TP 104.
14. `glm-5.3-flash` ← N0061 (smollm3_3b); selection score 0.6267, recall 0.6267, predictions 317, TP 94.
15. `kimi-k3` ← N0067 (ultrabert380); selection score 0.6133, recall 0.6133, predictions 245, TP 92.
16. `gpt-5.6-sol` ← N0062 (ultrabert380); selection score 0.6000, recall 0.6000, predictions 256, TP 90.
17. `claude-fable-5` ← N0105 (ultrabert380); selection score 0.5369, recall 0.5369, predictions 278, TP 80.
18. `glm-5.3` ← N0075 (ultrabert380); selection score 0.4800, recall 0.4800, predictions 296, TP 72.

**R2_recall_x_prediction_count**
1. `gemini-3.8-flash` ← N0070 (ultrabert380); selection score 327.1133, recall 0.9267, predictions 353, TP 139.
2. `gpt-6-astra` ← N0031 (embeddinggemma2); selection score 288.2667, recall 0.7667, predictions 376, TP 115.
3. `claude-opus-4-8` ← N0030 (embeddinggemma2); selection score 268.4000, recall 0.8133, predictions 330, TP 122.
4. `gemini-3.7-flash` ← N0065 (ultrabert380); selection score 262.9333, recall 0.7733, predictions 340, TP 116.
5. `qwen3.8-max` ← N0009 (catboost_stylometry); selection score 230.1867, recall 0.6933, predictions 332, TP 104.
6. `claude-opus-4-6` ← N0062 (ultrabert380); selection score 225.2800, recall 0.8533, predictions 264, TP 128.
7. `gpt-5.6-sol` ← N0077 (ultrabert380); selection score 222.3467, recall 0.5867, predictions 379, TP 88.
8. `kimi-k3` ← N0063 (ultrabert380); selection score 222.1400, recall 0.5800, predictions 383, TP 87.
9. `gemini-3.6-flash` ← N0081 (granite4_h_micro); selection score 217.8000, recall 0.9000, predictions 242, TP 135.
10. `gpt-5.6-terra` ← N0030 (embeddinggemma2); selection score 214.2400, recall 0.6867, predictions 312, TP 103.
11. `gemini-3.1-pro-preview` ← N0062 (ultrabert380); selection score 203.3200, recall 0.9200, predictions 221, TP 138.
12. `glm-5.3-flash` ← N0061 (smollm3_3b); selection score 198.6533, recall 0.6267, predictions 317, TP 94.
13. `claude-sonnet-5` ← N0035 (granite4_350m); selection score 191.8000, recall 0.7000, predictions 274, TP 105.
14. `gpt-5.5` ← N0105 (ultrabert380); selection score 191.1133, recall 0.7267, predictions 263, TP 109.
15. `gpt-5.6-luna` ← N0057 (llama32_3b); selection score 182.0000, recall 0.7000, predictions 260, TP 105.
16. `claude-fable-5` ← N0031 (embeddinggemma2); selection score 171.8054, recall 0.4631, predictions 371, TP 69.
17. `claude-opus-5` ← N0069 (ultrabert380); selection score 166.7933, recall 0.8467, predictions 197, TP 127.
18. `glm-5.3` ← N0045 (svc); selection score 158.2933, recall 0.4267, predictions 371, TP 64.

**R3_recall_x_true_label_count**
1. `gemini-3.8-flash` ← N0070 (ultrabert380); selection score 139.0000, recall 0.9267, predictions 353, TP 139.
2. `gemini-3.1-pro-preview` ← N0062 (ultrabert380); selection score 138.0000, recall 0.9200, predictions 221, TP 138.
3. `gemini-3.6-flash` ← N0081 (granite4_h_micro); selection score 135.0000, recall 0.9000, predictions 242, TP 135.
4. `claude-opus-4-6` ← N0062 (ultrabert380); selection score 128.0000, recall 0.8533, predictions 264, TP 128.
5. `gpt-6-astra` ← N0081 (granite4_h_micro); selection score 128.0000, recall 0.8533, predictions 152, TP 128.
6. `claude-opus-5` ← N0069 (ultrabert380); selection score 127.0000, recall 0.8467, predictions 197, TP 127.
7. `claude-opus-4-8` ← N0030 (embeddinggemma2); selection score 122.0000, recall 0.8133, predictions 330, TP 122.
8. `qwen3.8-max` ← N0063 (ultrabert380); selection score 117.0000, recall 0.7800, predictions 234, TP 117.
9. `gemini-3.7-flash` ← N0065 (ultrabert380); selection score 116.0000, recall 0.7733, predictions 340, TP 116.
10. `claude-sonnet-5` ← N0105 (ultrabert380); selection score 113.0000, recall 0.7533, predictions 181, TP 113.
11. `gpt-5.6-luna` ← N0106 (qwen06); selection score 112.0000, recall 0.7467, predictions 238, TP 112.
12. `gpt-5.5` ← N0105 (ultrabert380); selection score 109.0000, recall 0.7267, predictions 263, TP 109.
13. `gpt-5.6-terra` ← N0038 (ministral3_3b); selection score 104.0000, recall 0.6933, predictions 259, TP 104.
14. `glm-5.3-flash` ← N0061 (smollm3_3b); selection score 94.0000, recall 0.6267, predictions 317, TP 94.
15. `kimi-k3` ← N0067 (ultrabert380); selection score 92.0000, recall 0.6133, predictions 245, TP 92.
16. `gpt-5.6-sol` ← N0062 (ultrabert380); selection score 90.0000, recall 0.6000, predictions 256, TP 90.
17. `claude-fable-5` ← N0105 (ultrabert380); selection score 80.0000, recall 0.5369, predictions 278, TP 80.
18. `glm-5.3` ← N0075 (ultrabert380); selection score 72.0000, recall 0.4800, predictions 296, TP 72.

### Exact fold-specific rule orders

Each fold's list was selected only on the other four folds and applied to the held-out fold:

**Fold 1**
- R1_highest_recall: N0070:gemini-3.8-flash → N0062:gemini-3.1-pro-preview → N0081:gemini-3.6-flash → N0081:gpt-6-astra → N0062:claude-opus-4-6 → N0069:claude-opus-5 → N0030:claude-opus-4-8 → N0063:qwen3.8-max → N0065:gemini-3.7-flash → N0105:claude-sonnet-5 → N0106:gpt-5.6-luna → N0105:gpt-5.5 → N0038:gpt-5.6-terra → N0061:glm-5.3-flash → N0062:gpt-5.6-sol → N0067:kimi-k3 → N0105:claude-fable-5 → N0058:glm-5.3
- R2_recall_x_prediction_count: N0070:gemini-3.8-flash → N0031:gpt-6-astra → N0030:claude-opus-4-8 → N0065:gemini-3.7-flash → N0077:gpt-5.6-sol → N0062:claude-opus-4-6 → N0005:qwen3.8-max → N0081:gemini-3.6-flash → N0030:gpt-5.6-terra → N0035:claude-sonnet-5 → N0063:kimi-k3 → N0062:gemini-3.1-pro-preview → N0061:glm-5.3-flash → N0057:gpt-5.6-luna → N0105:gpt-5.5 → N0021:claude-opus-5 → N0031:claude-fable-5 → N0045:glm-5.3
- R3_recall_x_true_label_count: N0070:gemini-3.8-flash → N0062:gemini-3.1-pro-preview → N0081:gemini-3.6-flash → N0081:gpt-6-astra → N0062:claude-opus-4-6 → N0069:claude-opus-5 → N0030:claude-opus-4-8 → N0063:qwen3.8-max → N0065:gemini-3.7-flash → N0105:claude-sonnet-5 → N0106:gpt-5.6-luna → N0105:gpt-5.5 → N0038:gpt-5.6-terra → N0061:glm-5.3-flash → N0062:gpt-5.6-sol → N0067:kimi-k3 → N0105:claude-fable-5 → N0058:glm-5.3

**Fold 2**
- R1_highest_recall: N0070:gemini-3.8-flash → N0062:gemini-3.1-pro-preview → N0081:gemini-3.6-flash → N0069:claude-opus-5 → N0062:claude-opus-4-6 → N0081:gpt-6-astra → N0030:claude-opus-4-8 → N0105:claude-sonnet-5 → N0065:gemini-3.7-flash → N0060:qwen3.8-max → N0081:gpt-5.5 → N0106:gpt-5.6-luna → N0038:gpt-5.6-terra → N0067:kimi-k3 → N0061:glm-5.3-flash → N0077:gpt-5.6-sol → N0058:claude-fable-5 → N0075:glm-5.3
- R2_recall_x_prediction_count: N0070:gemini-3.8-flash → N0031:gpt-6-astra → N0065:gemini-3.7-flash → N0030:claude-opus-4-8 → N0009:qwen3.8-max → N0063:kimi-k3 → N0062:claude-opus-4-6 → N0030:gpt-5.6-terra → N0077:gpt-5.6-sol → N0062:gemini-3.1-pro-preview → N0081:gemini-3.6-flash → N0061:glm-5.3-flash → N0035:claude-sonnet-5 → N0105:gpt-5.5 → N0077:gpt-5.6-luna → N0069:claude-opus-5 → N0031:claude-fable-5 → N0075:glm-5.3
- R3_recall_x_true_label_count: N0070:gemini-3.8-flash → N0062:gemini-3.1-pro-preview → N0081:gemini-3.6-flash → N0069:claude-opus-5 → N0062:claude-opus-4-6 → N0081:gpt-6-astra → N0030:claude-opus-4-8 → N0105:claude-sonnet-5 → N0065:gemini-3.7-flash → N0060:qwen3.8-max → N0081:gpt-5.5 → N0106:gpt-5.6-luna → N0038:gpt-5.6-terra → N0067:kimi-k3 → N0061:glm-5.3-flash → N0077:gpt-5.6-sol → N0058:claude-fable-5 → N0075:glm-5.3

**Fold 3**
- R1_highest_recall: N0062:gemini-3.1-pro-preview → N0070:gemini-3.8-flash → N0081:gemini-3.6-flash → N0081:gpt-6-astra → N0105:claude-opus-4-6 → N0069:claude-opus-5 → N0030:claude-opus-4-8 → N0063:qwen3.8-max → N0065:gemini-3.7-flash → N0105:claude-sonnet-5 → N0106:gpt-5.6-luna → N0105:gpt-5.5 → N0030:gpt-5.6-terra → N0061:glm-5.3-flash → N0067:kimi-k3 → N0062:gpt-5.6-sol → N0105:claude-fable-5 → N0057:glm-5.3
- R2_recall_x_prediction_count: N0070:gemini-3.8-flash → N0031:gpt-6-astra → N0065:gemini-3.7-flash → N0030:claude-opus-4-8 → N0003:qwen3.8-max → N0062:claude-opus-4-6 → N0030:gpt-5.6-terra → N0081:gemini-3.6-flash → N0063:kimi-k3 → N0077:gpt-5.6-sol → N0061:glm-5.3-flash → N0062:gemini-3.1-pro-preview → N0105:gpt-5.5 → N0035:claude-sonnet-5 → N0057:gpt-5.6-luna → N0045:glm-5.3 → N0069:claude-opus-5 → N0031:claude-fable-5
- R3_recall_x_true_label_count: N0062:gemini-3.1-pro-preview → N0070:gemini-3.8-flash → N0081:gemini-3.6-flash → N0081:gpt-6-astra → N0105:claude-opus-4-6 → N0069:claude-opus-5 → N0030:claude-opus-4-8 → N0063:qwen3.8-max → N0065:gemini-3.7-flash → N0105:claude-sonnet-5 → N0106:gpt-5.6-luna → N0105:gpt-5.5 → N0030:gpt-5.6-terra → N0061:glm-5.3-flash → N0067:kimi-k3 → N0062:gpt-5.6-sol → N0105:claude-fable-5 → N0057:glm-5.3

**Fold 4**
- R1_highest_recall: N0062:gemini-3.1-pro-preview → N0070:gemini-3.8-flash → N0081:gemini-3.6-flash → N0062:claude-opus-4-6 → N0081:gpt-6-astra → N0069:claude-opus-5 → N0030:claude-opus-4-8 → N0106:gpt-5.6-luna → N0081:qwen3.8-max → N0065:gemini-3.7-flash → N0105:claude-sonnet-5 → N0105:gpt-5.5 → N0030:gpt-5.6-terra → N0062:gpt-5.6-sol → N0067:kimi-k3 → N0061:glm-5.3-flash → N0058:claude-fable-5 → N0075:glm-5.3
- R2_recall_x_prediction_count: N0070:gemini-3.8-flash → N0031:gpt-6-astra → N0030:claude-opus-4-8 → N0065:gemini-3.7-flash → N0003:qwen3.8-max → N0062:claude-opus-4-6 → N0063:kimi-k3 → N0030:gpt-5.6-terra → N0081:gemini-3.6-flash → N0077:gpt-5.6-sol → N0062:gemini-3.1-pro-preview → N0035:claude-sonnet-5 → N0106:gpt-5.6-luna → N0061:glm-5.3-flash → N0105:gpt-5.5 → N0031:claude-fable-5 → N0045:glm-5.3 → N0069:claude-opus-5
- R3_recall_x_true_label_count: N0062:gemini-3.1-pro-preview → N0070:gemini-3.8-flash → N0081:gemini-3.6-flash → N0062:claude-opus-4-6 → N0081:gpt-6-astra → N0069:claude-opus-5 → N0030:claude-opus-4-8 → N0106:gpt-5.6-luna → N0081:qwen3.8-max → N0065:gemini-3.7-flash → N0105:claude-sonnet-5 → N0105:gpt-5.5 → N0030:gpt-5.6-terra → N0062:gpt-5.6-sol → N0067:kimi-k3 → N0061:glm-5.3-flash → N0058:claude-fable-5 → N0075:glm-5.3

**Fold 5**
- R1_highest_recall: N0062:gemini-3.1-pro-preview → N0070:gemini-3.8-flash → N0081:gemini-3.6-flash → N0069:claude-opus-5 → N0081:gpt-6-astra → N0062:claude-opus-4-6 → N0030:claude-opus-4-8 → N0063:qwen3.8-max → N0065:gemini-3.7-flash → N0106:gpt-5.6-luna → N0105:claude-sonnet-5 → N0105:gpt-5.5 → N0038:gpt-5.6-terra → N0061:glm-5.3-flash → N0062:gpt-5.6-sol → N0067:kimi-k3 → N0105:claude-fable-5 → N0075:glm-5.3
- R2_recall_x_prediction_count: N0070:gemini-3.8-flash → N0031:gpt-6-astra → N0030:claude-opus-4-8 → N0065:gemini-3.7-flash → N0077:gpt-5.6-sol → N0009:qwen3.8-max → N0063:kimi-k3 → N0081:gemini-3.6-flash → N0062:claude-opus-4-6 → N0061:glm-5.3-flash → N0062:gemini-3.1-pro-preview → N0079:gpt-5.6-terra → N0105:gpt-5.5 → N0057:gpt-5.6-luna → N0031:claude-fable-5 → N0035:claude-sonnet-5 → N0069:claude-opus-5 → N0045:glm-5.3
- R3_recall_x_true_label_count: N0062:gemini-3.1-pro-preview → N0070:gemini-3.8-flash → N0081:gemini-3.6-flash → N0069:claude-opus-5 → N0081:gpt-6-astra → N0062:claude-opus-4-6 → N0030:claude-opus-4-8 → N0063:qwen3.8-max → N0065:gemini-3.7-flash → N0106:gpt-5.6-luna → N0105:claude-sonnet-5 → N0105:gpt-5.5 → N0038:gpt-5.6-terra → N0061:glm-5.3-flash → N0062:gpt-5.6-sol → N0067:kimi-k3 → N0105:claude-fable-5 → N0075:glm-5.3

Fold JSON also preserves each selected rule's fitting-fold recall, prediction count, TP, score and model ID.

## Per-label precision and recall

Values below are prompt-group OOF precision / recall. C1 and C3 are included for direct comparison.

| Label | EQUAL9 | C1 | C3 | R1 | R2 | R3 |
|---|---:|---:|---:|---:|---:|---:|
| claude-fable-5 | 0.406 / 0.523 | 0.444 / 0.617 | 0.403 / 0.517 | 0.309 / 0.141 | 0.200 / 0.067 | 0.309 / 0.141 |
| claude-opus-4-6 | 0.857 / 0.800 | 0.879 / 0.773 | 0.857 / 0.800 | 0.537 / 0.820 | 0.627 / 0.627 | 0.537 / 0.820 |
| claude-opus-4-8 | 0.757 / 0.767 | 0.736 / 0.727 | 0.760 / 0.760 | 0.396 / 0.593 | 0.380 / 0.813 | 0.396 / 0.593 |
| claude-opus-5 | 0.908 / 0.787 | 0.854 / 0.780 | 0.855 / 0.787 | 0.635 / 0.833 | 0.812 / 0.460 | 0.635 / 0.833 |
| claude-sonnet-5 | 0.688 / 0.733 | 0.658 / 0.707 | 0.688 / 0.733 | 0.678 / 0.533 | 0.566 / 0.427 | 0.678 / 0.533 |
| gemini-3.1-pro-preview | 0.959 / 0.780 | 0.918 / 0.820 | 0.914 / 0.853 | 0.652 / 0.873 | 0.955 / 0.700 | 0.652 / 0.873 |
| gemini-3.6-flash | 0.637 / 0.807 | 0.669 / 0.700 | 0.684 / 0.793 | 0.659 / 0.387 | 0.577 / 0.273 | 0.659 / 0.387 |
| gemini-3.7-flash | 0.660 / 0.220 | 0.538 / 0.280 | 0.673 / 0.220 | 0.200 / 0.013 | 0.167 / 0.087 | 0.200 / 0.013 |
| gemini-3.8-flash | 0.500 / 0.833 | 0.496 / 0.800 | 0.500 / 0.833 | 0.400 / 0.867 | 0.393 / 0.927 | 0.400 / 0.867 |
| glm-5.3 | 0.398 / 0.453 | 0.463 / 0.460 | 0.405 / 0.453 | 0.389 / 0.140 | 0.302 / 0.087 | 0.389 / 0.140 |
| glm-5.3-flash | 0.521 / 0.487 | 0.583 / 0.587 | 0.514 / 0.473 | 0.358 / 0.393 | 0.273 / 0.200 | 0.358 / 0.393 |
| gpt-5.5 | 0.659 / 0.580 | 0.613 / 0.633 | 0.659 / 0.580 | 0.449 / 0.500 | 0.562 / 0.240 | 0.449 / 0.500 |
| gpt-5.6-luna | 0.719 / 0.613 | 0.691 / 0.627 | 0.710 / 0.587 | 0.517 / 0.693 | 0.467 / 0.093 | 0.517 / 0.693 |
| gpt-5.6-sol | 0.641 / 0.547 | 0.610 / 0.500 | 0.659 / 0.540 | 0.535 / 0.253 | 0.203 / 0.200 | 0.535 / 0.253 |
| gpt-5.6-terra | 0.560 / 0.773 | 0.541 / 0.700 | 0.562 / 0.760 | 0.500 / 0.380 | 0.344 / 0.287 | 0.500 / 0.380 |
| gpt-6-astra | 0.900 / 0.720 | 0.922 / 0.713 | 0.844 / 0.760 | 0.828 / 0.867 | 0.318 / 0.807 | 0.828 / 0.867 |
| kimi-k3 | 0.678 / 0.533 | 0.605 / 0.480 | 0.684 / 0.533 | 0.564 / 0.413 | 0.201 / 0.327 | 0.564 / 0.413 |
| qwen3.8-max | 0.722 / 0.813 | 0.713 / 0.760 | 0.725 / 0.807 | 0.548 / 0.727 | 0.401 / 0.593 | 0.548 / 0.727 |

## Distinctness and firing behavior

| Pair | OOF differing rows | OOF agreement | Full-V0 differing rows | Full-V0 agreement |
|---|---:|---:|---:|---:|
| R1_highest_recall vs R2_recall_x_prediction_count | 1193 | 55.7984% | 1194 | 55.7614% |
| R1_highest_recall vs R3_recall_x_true_label_count | 0 | 100.0000% | 0 | 100.0000% |
| R2_recall_x_prediction_count vs R3_recall_x_true_label_count | 1193 | 55.7984% | 1194 | 55.7614% |

Held-out cascade rule assignments (summed across the five folds):

| Strategy | Rule-assigned rows | Correct | Incorrect | Fallback rows |
|---|---:|---:|---:|---:|
| R1_highest_recall | 2584 | 1362 | 1222 | 115 |
| R2_recall_x_prediction_count | 2633 | 1043 | 1590 | 66 |
| R3_recall_x_true_label_count | 2584 | 1362 | 1222 | 115 |

## Interpretation

R3 is a control: because true-label support is constant across checkpoints for a given label, maximizing recall × support is exactly maximizing TP. R2 rewards the product of capture rate and firing volume; this can select a high-volume rule even when its recall is lower. A rule's fit-fold score is not its incremental precision after earlier rules have claimed examples; held-out firing counts show the cascade's actual first-match behavior.

## Reproducibility

- `experiment_config.json`: input hashes, roster, folds, support threshold, and exact tie-breaking.
- `selected_rules.json` and `folds/fold_*.json`: full-data diagnostics and each fit-fold rule sequence.
- `fold_assignments.csv`: exact saved five-fold ID/prompt mapping.
- `cascade_predictions.parquet`: aligned IDs, targets, EQUAL9/C1/C3 baselines, and OOF/full predictions.
- `metrics.csv`, `per_label_metrics.csv`, `fold_metrics.csv`, `oof_rule_contributions.csv`, and `full_v0_rule_contributions.csv`: metrics and rule firing details.
- `analysis_state.json`: completion state and input hashes.
