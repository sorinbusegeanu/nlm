# EQUAL9 and C3 Kaggle comparison

## Configurations

EQUAL9 is the equal-probability mean of these nine Trust List 2 v025 checkpoints. Each member has weight `1/9`:

| Checkpoint | Model | Training data |
|---|---|---|
| N0071 | UltraBERT-380M | V2.14 |
| N0005 | CatBoost + Stylometry | V1 |
| N0058 | Ministral 3 3B | V1-all |
| N0072 | SVC | V2.14 |
| N0030 | EmbeddingGemma 2 | V1 |
| N0061 | SmolLM3 3B | V1-all |
| N0044 | SVC | V1.1 |
| N0009 | CatBoost + Stylometry | V1 |
| N0048 | UltraBERT-380M | V1-all |

The C3 submission uses EQUAL9 when none of these ordered rules matches. A rule matches when its checkpoint's top-1 prediction equals the rule label; the first match wins.

| Priority | Checkpoint | Required top-1 label | Full-V0 matches | Full-V0 precision |
|---:|---|---|---:|---:|
| 1 | N0071 | `gpt-6-astra` | 100 | 93.0% |
| 2 | N0105 | `gemini-3.1-pro-preview` | 130 | 92.31% |
| 3 | N0048 | `claude-opus-4-6` | 88 | 90.91% |

Rule selection and order match the full-V0 diagnostic configuration in [the cascade comparison](ensemble_cascade_comparison.md). Eligibility required at least 10 matching rows and precision ≥90%. Since these rules were selected on full V0, their diagnostic score is optimistic; the prompt-group cross-fit score is also reported below.

## V0 validation

| Variant | Accuracy | Balanced accuracy | Evaluation scope |
|---|---:|---:|---|
| EQUAL9 | 65.3946% | 65.3898% | Fixed ensemble on full V0; its fixed predictions are the same across folds |
| EQUAL9 + C3 | 65.5057% | 65.5006% | Five-fold prompt-group cross-fit for rule selection |
| EQUAL9 + C3 | 65.8021% | 65.7972% | Full-V0 rule-selection diagnostic |

These are V0 development metrics, not Kaggle leaderboard scores. The cross-fit does not remove the checkpoints' prior V0 selection exposure.

## Kaggle submissions

| Variant | Kernel and version | Submission ID | Public score | Difference vs EQUAL9 | Status |
|---|---|---:|---:|---:|---|
| EQUAL9 | `zodrak/equal9-v025-baseline`, v4 | 58444305 | Pending | Pending | Pending |
| EQUAL9 + C3 | `zodrak/equal9-v025-plus-c3-cascade`, v2 | Pending | Pending | Pending | Not submitted yet |

Both variants were generated from the competition test rows in the same order: `00353fb1c541`, `005f5a9a6139`, `0062a53d5487`. The schema is exactly `id,model`, with three unique IDs and valid competition labels. The two variants happen to predict the same label for each of these three test rows.

## Checkpoint resolution and artifacts

Checkpoint IDs and experiment identities were resolved against `docs/trust-list-2-v025.csv` (SHA-256 `c93ae4cc0612ed36b7c126d6d50a5b50456ecfafd4e835369272076442e8cf9b`). The N0105 inference files were packaged from its selected checkpoint and the packaged manifest records the source run and file hashes.

Kaggle outputs are saved under `artifacts/kaggle/equal9-v025-kernel/output-v4/` and `artifacts/kaggle/equal9-c3-v025-kernel/`.
