# Trust List 2.2

Version: `trust-list-2.2-v0004`. This list contains checkpoint-label scores only for checkpoints whose recorded training dataset is a V2.x dataset.

Source: [trust-list-2-v024](../docs/trust-list-2-v024.csv) (77 checkpoints, 1,386 rows).
Validation: 2,699 full V0 rows; SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.

The source Trust List 2 files remain unchanged. The dashboard reads the Trust List 2.2 latest pointer and does not display Trust List 1 or the general Trust List 2.

## Included training datasets

| Dataset | Checkpoints | Checkpoint-label rows | Rule-eligible rows |
|---|---:|---:|---:|
| `v2.0` | 4 | 72 | 72 |
| `v2.0.chunks.200TokBalanced` | 1 | 18 | 17 |
| `v2.0.chunks.300TokBalanced` | 1 | 18 | 18 |
| `v2.0.chunks800` | 1 | 18 | 18 |
| `v2.1` | 1 | 18 | 17 |
| `v2.14` | 8 | 144 | 142 |
| `v2.14-prediction-filtered` | 1 | 18 | 17 |
| `v2.14-prediction-filtered-80pct` | 1 | 18 | 18 |
| `v2.14.balanced` | 1 | 18 | 18 |
| `v2.15_extrato_14` | 1 | 18 | 17 |
| `v2.16` | 1 | 18 | 18 |

## Totals

- Included checkpoints: 21
- Checkpoint-label rows: 378
- Rule-eligible rows: 372
- Excluded checkpoints: 56
- Precision, recall, and support values are copied from Trust List 2; the existing eligibility threshold and sort order are reapplied.
