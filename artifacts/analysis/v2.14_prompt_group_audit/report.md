# V2.14 prompt-group and ID audit

Dataset: `data/processed/dataset_v2.14.parquet`
SHA256: `4b9f3d1efdc28a720526f2c10a3f74df0d6ac5c7c0787a0ac59f19c4bb2e2fff`
Group keys: V14 uses non-empty `prompt_canon`; B2M additions use source `prompt_id` separately (230,664 keyed rows; 38,055 observed groups).
Normalization for response duplicate checks: Unicode NFKC, case-fold, collapsed whitespace; punctuation preserved.

## Group metrics

| Metric | Result | Definition |
|---|---:|---|
| V2.14 response rows | 231,336 | All rows in the dataset artifact. |
| Rows with usable prompt key | 230,664 | V14 rows use prompt_canon; B2M additions use their source prompt_id in a separate namespace. |
| Observed keyed prompt groups | 38,055 | V14 canonical groups plus separately keyed B2M prompt IDs; cross-pool matches cannot be merged. |
| Groups containing at least two model labels | 26,383 | Distinct labels within each observed keyed prompt group. |
| Groups with multiple rows for at least one model | 4,107 | Any model occurs more than once in the observed keyed prompt group. |
| Groups with an identical normalized response repeated for the same model | 0 | NFKC + case-fold + collapsed-whitespace response equality within a group and model. |
| Excess identical-response rows | 0 | Rows beyond the first for each repeated group/model/normalized-response tuple. |
| Mean responses per observed keyed group | 6.06 | Rows per canonical V14 or source-keyed B2M group. |
| Median responses per observed keyed group | 2.00 | Rows per canonical V14 or source-keyed B2M group. |
| Maximum group size | 98 | Rows for one observed keyed prompt group. |

## Group-size distribution

Counts are by rows per observed keyed prompt group. V14 and B2M key spaces remain separate; possible V14-to-B2M prompt matches cannot be merged. The full exact-size distribution is in `group_size_distribution_exact.csv`.

| Responses in group | Groups | Response rows | Share of groups |
|---:|---:|---:|---:|
| 1 | 11,462 | 11,462 | 30.1% |
| 2 | 8,030 | 16,060 | 21.1% |
| 3 | 3,853 | 11,559 | 10.1% |
| 4 | 2,649 | 10,596 | 7.0% |
| 5 | 2,041 | 10,205 | 5.4% |
| 6-8 | 1,393 | 10,057 | 3.7% |
| 9-15 | 5,262 | 59,012 | 13.8% |
| 16-30 | 2,193 | 40,438 | 5.8% |
| 31-50 | 501 | 20,814 | 1.3% |
| 51-100 | 671 | 40,461 | 1.8% |

## Prompt-ID consistency

| Check | Count | Interpretation |
|---|---:|---|
| Unique prompt_id values in V2.14 | 53,164 | prompt_id values; source-local and file-level fallbacks are included. |
| prompt_id values mapped to at least one prompt_canon | 50,505 | Only rows with a non-empty prompt_canon are used. |
| prompt_id values mapping to multiple canonical prompts | 752 | Of 50505 mapped IDs; maximum distinct prompt_canon values for one ID is 2914. |
| V14 prompt_id values reused across source_dataset values | 0 | Among rows with prompt_canon; none are shared across source datasets. |
| prompt_canon groups present in multiple source datasets | 5,342 | Of 36068 V14 canonical prompt groups; all 5342 have multiple prompt_id values. |
| B2M prompt_id values reused across source_dataset values | 267 | Of 1987 B2M IDs; prompt text/canonical keys are absent, so reused IDs cannot be verified against content. |
| V14 rows lacking a usable prompt key | 672 | These rows lack prompt_canon and are not B2M rows with source-provided prompt IDs. |

## Group metrics by key basis

| Key basis | Rows | Groups | Multi-label groups | Groups with repeated same-model rows |
|---|---:|---:|---:|---:|
| V14 canonical prompt_canon | 223,168 | 36,068 | 25,339 | 3,861 |
| B2M source prompt_id | 7,496 | 1,987 | 1,044 | 246 |
| V14 rows without usable prompt key | 672 | 0 | 0 | 0 |

**Finding:** V14 `prompt_id` is not a globally consistent prompt key: all 5,342 `prompt_canon` groups that occur in multiple source datasets have multiple `prompt_id` values, and 752 V14 IDs map to more than one canonical prompt (up to 2,914). Use `prompt_canon` for cross-source grouping where present. B2M has 1,987 source prompt IDs, 267 reused across source datasets; its 7,496 rows have no `prompt_canon` or prompt text, so those IDs cannot be validated against V14 or independently checked against prompt content.

There are **672 additional V14 rows** without a usable prompt key. Their prompt identity cannot be inferred from the file-level fallback ID. See `rows_without_prompt_canon_by_source.csv`.

## Output tables

- `summary_metrics.csv` — requested group metrics.
- `group_metrics_by_key_basis.csv` — V14 canonical and B2M source-keyed strata.
- `group_size_distribution.csv` — grouped size ranges.
- `group_size_distribution_exact.csv` — exact group-size frequencies.
- `prompt_id_consistency.csv` — prompt ID mapping checks.
- `rows_without_prompt_canon_by_source.csv` — rows excluded from canonical-group metrics.
