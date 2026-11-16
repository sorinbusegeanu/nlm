# ORACLE01 — Checkpoint oracle coverage

Trust List 2: `trust-list-2-v025` (`docs/trust-list-2-v025.csv`). Predictions: saved full-V0 probability caches under `artifacts/analysis/ensemble_precision_recall_overlap/`, aligned to V0 row IDs and canonical labels.

## Definition

For each V0 response, the oracle counts it as covered if **any checkpoint in the set has the true label as its top-1 prediction**. This is a theoretical coverage ceiling: it uses the true label to choose among checkpoint outputs and is not a deployable prediction rule.

- **EQUAL9:** `N0071, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048`.
- **All 78:** all checkpoints eligible in Trust List 2 v025: `evaluation_eligible`, `rule_eligible`, and `checkpoint_available` are true; validation-training overlap is false; validation set has 2,699 rows.

## Overall oracle accuracy

| Checkpoint set | Oracle-correct | Oracle accuracy | Missed by every checkpoint |
|---|---:|---:|---:|
| EQUAL9 | 2,471/2,699 | 91.55% | 228 (8.45%) |
| All 78 eligible | 2,663/2,699 | 98.67% | 36 (1.33%) |

## Per-label oracle recall

A row is covered for its true label if at least one checkpoint predicts that label at top-1.

| Label | V0 rows | EQUAL9 covered | EQUAL9 oracle recall | All 78 covered | All 78 oracle recall |
|---|---:|---:|---:|---:|---:|
| claude-fable-5 | 149 | 136/149 | 91.28% | 148/149 | 99.33% |
| claude-opus-4-6 | 150 | 143/150 | 95.33% | 150/150 | 100.00% |
| claude-opus-4-8 | 150 | 141/150 | 94.00% | 149/150 | 99.33% |
| claude-opus-5 | 150 | 135/150 | 90.00% | 148/150 | 98.67% |
| claude-sonnet-5 | 150 | 136/150 | 90.67% | 144/150 | 96.00% |
| gemini-3.1-pro-preview | 150 | 143/150 | 95.33% | 149/150 | 99.33% |
| gemini-3.6-flash | 150 | 147/150 | 98.00% | 150/150 | 100.00% |
| gemini-3.7-flash | 150 | 126/150 | 84.00% | 149/150 | 99.33% |
| gemini-3.8-flash | 150 | 146/150 | 97.33% | 149/150 | 99.33% |
| glm-5.3 | 150 | 134/150 | 89.33% | 149/150 | 99.33% |
| glm-5.3-flash | 150 | 138/150 | 92.00% | 148/150 | 98.67% |
| gpt-5.5 | 150 | 133/150 | 88.67% | 148/150 | 98.67% |
| gpt-5.6-luna | 150 | 141/150 | 94.00% | 150/150 | 100.00% |
| gpt-5.6-sol | 150 | 128/150 | 85.33% | 147/150 | 98.00% |
| gpt-5.6-terra | 150 | 143/150 | 95.33% | 148/150 | 98.67% |
| gpt-6-astra | 150 | 140/150 | 93.33% | 148/150 | 98.67% |
| kimi-k3 | 150 | 120/150 | 80.00% | 142/150 | 94.67% |
| qwen3.8-max | 150 | 141/150 | 94.00% | 147/150 | 98.00% |

## Summary

EQUAL9 oracle coverage is **2,471/2,699 (91.55%)**; **228 (8.45%)** responses are missed by all nine checkpoints.
Using all 78 eligible checkpoints raises oracle coverage to **2,663/2,699 (98.67%)**; **36 (1.33%)** remain missed by every checkpoint. The expanded set adds **192** oracle-covered responses.

This calculation measures available top-1 correctness across checkpoint predictions. It does not estimate the accuracy of a realizable ensemble, because the oracle has access to the true label when deciding which checkpoint to trust.
