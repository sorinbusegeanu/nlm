# Open Pool v16 Prompt-Group Audit for E21

## Executive summary

**Open Pool v16 is suitable for E21 classification and group-ranking training**, provided ranking uses canonical prompt groups first and a source/file-namespaced fallback only when `prompt_canon` is absent. The main parquet has 231,036 valid, labeled response rows; all 18 classes are represented. All rows have a usable identifier under this rule; the source-ID fallback carries a panel-label caveat described below. Of those, 219,294 responses (94.92%) belong to 26,347 groups with at least two distinct labels. That yields 170,233 ranked items and 812,835 pairwise label comparisons per epoch when each group contributes one response per label.

**Recommendation:** use minimum 2 distinct labels. Retain both two-label and three-label groups; two-label groups supply 8,052 direct comparisons over 16,311 responses, while minimum 3 would remove those useful examples and save only those 8,052 comparisons. Rotate the distinct same-label responses across epochs. Use `prompt_canon` as the primary key; never use `prompt_group` by itself. No embedding-based reconstruction is indicated for this dataset.

## Dataset and key audit

Input is the original `data/raw/open_pool/v16/ntl_open_pool.parquet` (231,036 rows; SHA-256 `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`). The separately supplied `prompts.parquet` contains canonical prompt text for 17,671 prompt keys (SHA-256 `e4646c25025955489f5773bca001fba2bd8768b451cc7080e4035953f80aa165`). The incremental dataset was not used.

| Check | Result |
|---|---:|
| Classification responses with nonempty text and canonical label | 231,036/231,036 (100.00%) |
| Canonical labels represented | 18/18 |
| Rows with `prompt_canon` | 225,110 (97.44%) |
| Rows using fallback source prompt identifier | 5,926 (2.56%) |
| Rows with no usable identifier | 0 |
| Distinct `prompt_canon` groups | 36,068 |
| Canonical prompt text available / unavailable | 17,671 / 18,397 groups; 40,775 rows lack prompt text |
| Canonical keys with conflicting prompt text | 0 |
| `panel_labels` metadata vs observed distinct labels | 36,068/36,068 canonical groups match |
| Canonical groups with inconsistent `panel_labels` across rows | 0 |
| Fallback multi-label groups where `panel_labels` disagrees with observed label count | 1,008 groups / 5,123 rows |
| Source prompt IDs shared across source datasets | 0 |
| Source prompt IDs reused across files within a dataset | 235; fallback key includes `source_file` |
| `prompt_group` IDs mapping to multiple `prompt_canon` keys | 752 (maximum 2,914 canonical keys for one source ID) |
| Canonical prompt groups spanning multiple source datasets | 5,979; canonical key deliberately merges these source-local IDs |

Of 36,068 canonical group keys, 18,397 (51.00%) have no corresponding text row in `prompts.parquet`; 40,775 response rows (18.11% of rows with `prompt_canon`) lack inspectable canonical prompt text. These keys remain usable as identifiers and showed no conflicting text among the available prompt records. The limitation is manual semantic inspection, not key ambiguity.

`prompt_group` is not safe as a primary key: 752 source IDs map to multiple canonical prompts (the largest maps to 2,914). For rows with `prompt_canon`, grouping by that canonical key avoids this conflict. For the 5,926 rows with no `prompt_canon`, use `(source_dataset, source_file, prompt_group)`. There are no cross-source raw ID collisions. The 1,811 fallback source IDs are disjoint from IDs on rows with `prompt_canon`; none are reused across source files or source datasets. Source-file namespacing is retained as the conservative key. The fallback keys have 1,811 groups, of which 1,008 are multi-label. Those 1,008 groups (5,123 rows) have `panel_labels=1` while observed membership has 2–12 labels; this field is inconsistent for fallback groups. Derive ranking cardinality from observed distinct `model` labels. See `source_prompt_id_conflicts.csv`, `source_namespaced_prompt_id_mapping.csv`, and `missing_canon_source_fallback_by_source.csv`.

## Classification population

All **231,036** rows are valid classification examples, with no missing/empty text and no invalid labels. Response IDs are unique. All 18 labels appear; their counts and ranking coverage are in `classification_label_distribution.csv` and `eligible_responses_by_label.csv`. The `lab` family metadata agrees with the canonical label for all rows, and there are 0 contradictions for served-model names that can be mapped exactly to a canonical label.

## Group inventory and ranking eligibility

Using normalized `prompt_canon` first and the source-namespaced fallback for rows lacking it yields **37,879 groups** across all 231,036 responses. Group size by responses has median 2, mean 6.10, p90 15, p95 18, p99 58, and maximum 98. The maximum is 98 response rows due to repeat labels; the maximum distinct-label count is 18, and no group has more than 18 labels.

### Groups by distinct label count

| Distinct labels | Groups | Response rows |
|---:|---:|---:|
| 1 | 11,532 | 11,742 |
| 2 | 8,052 | 16,311 |
| 3 | 3,483 | 10,585 |
| 4 | 2,649 | 10,728 |
| 5 | 2,188 | 11,220 |
| 6 | 594 | 4,167 |
| 7 | 400 | 3,149 |
| 8 | 756 | 6,619 |
| 9 | 1,520 | 14,005 |
| 10 | 535 | 5,463 |
| 11 | 804 | 9,000 |
| 12 | 897 | 11,384 |
| 13 | 595 | 8,245 |
| 14 | 458 | 7,286 |
| 15 | 483 | 8,312 |
| 16 | 877 | 17,586 |
| 17 | 1,338 | 43,234 |
| 18 | 718 | 32,000 |

### Group size by number of response rows

| Responses per group | Groups | Response rows | Share of groups |
|---:|---:|---:|---:|
| 1 | 11,329 | 11,329 | 29.91% |
| 2 | 8,078 | 16,156 | 21.33% |
| 3 | 3,561 | 10,683 | 9.40% |
| 4–5 | 4,841 | 21,533 | 12.78% |
| 6–9 | 3,056 | 24,677 | 8.07% |
| 10–18 | 5,181 | 70,601 | 13.68% |
| 19–30 | 661 | 14,764 | 1.75% |
| 31–50 | 500 | 20,779 | 1.32% |
| 51–100 | 672 | 40,514 | 1.77% |

### Ranking eligibility thresholds

Excluded counts are relative to all 37,879 reliable groups / 231,036 responses. Pairwise comparisons count one comparison per pair of distinct labels in one ranking set per epoch; duplicate same-label outputs rotate across epochs and do not multiply same-epoch comparisons.

| Minimum distinct labels | Eligible groups | Eligible response rows | Excluded groups | Excluded response rows | Rows retained | Ranking items / epoch | Pairwise comparisons / epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 26,347 | 219,294 | 11,532 | 11,742 | 94.92% | 170,233 | 812,835 |
| 3 | 18,295 | 202,983 | 19,584 | 28,053 | 87.86% | 154,129 | 804,783 |
| 4 | 14,812 | 192,398 | 23,067 | 38,638 | 83.28% | 143,680 | 794,334 |
| 5 | 12,163 | 181,670 | 25,716 | 49,366 | 78.63% | 133,084 | 778,440 |

At the recommended minimum of two labels, 26,347 groups provide 219,294 response rows (94.92% of the full pool), 170,233 selected response-label items per epoch, and 812,835 pairwise comparisons per epoch. Every one of the 18 labels appears in eligible groups. Per-label eligible row counts and coverage are in `eligible_responses_by_label.csv`.

| Label | Classification rows | Ranking-eligible rows | Eligible / class rows |
|---|---:|---:|---:|
| `claude-fable-5` | 12,773 | 11,546 | 90.39% |
| `claude-opus-4-6` | 11,168 | 11,117 | 99.54% |
| `claude-opus-4-8` | 12,158 | 12,112 | 99.62% |
| `claude-opus-5` | 9,880 | 9,547 | 96.63% |
| `claude-sonnet-5` | 13,248 | 13,066 | 98.63% |
| `gemini-3.1-pro-preview` | 10,513 | 9,029 | 85.88% |
| `gemini-3.6-flash` | 18,447 | 15,588 | 84.50% |
| `gemini-3.7-flash` | 3,541 | 3,150 | 88.96% |
| `gemini-3.8-flash` | 23,542 | 20,239 | 85.97% |
| `glm-5.3` | 12,450 | 12,258 | 98.46% |
| `glm-5.3-flash` | 12,745 | 12,660 | 99.33% |
| `gpt-5.5` | 14,315 | 14,214 | 99.29% |
| `gpt-5.6-luna` | 18,894 | 18,044 | 95.50% |
| `gpt-5.6-sol` | 14,720 | 14,485 | 98.40% |
| `gpt-5.6-terra` | 15,719 | 15,601 | 99.25% |
| `gpt-6-astra` | 11,841 | 11,799 | 99.65% |
| `kimi-k3` | 10,516 | 10,293 | 97.88% |
| `qwen3.8-max` | 4,566 | 4,546 | 99.56% |

## Duplicate and consistency checks

- Exact duplicate normalized response clusters across the main parquet: **0** (0 excess rows). Conflicting-label duplicate text clusters: **0**.
- Groups with repeated rows for at least one model: **3,975**; these contain **21,347** repeated group/label pairs and **70,618** rows.
- Every repeated group/label pair has distinct normalized response text (100.00%); exact duplicate outputs within a group and model: **0**.
- Canonical `panel_labels` equals the observed distinct-label count in all 36,068 canonical groups; there are no prompt-text conflicts in the supplied canonical prompt table.
- Fallback metadata caveat: `panel_labels=1` in 1,008 multi-label source-ID groups (5,123 rows), though observed membership has 2–12 labels; do not use this field to set rank cardinality.
- Groups with more than 18 distinct labels: **0**. Group sizes above 18 are repeated responses for existing labels, not extra classes.
- Potential grouping risk: the 752 ambiguous `prompt_group` IDs must not be used instead of `prompt_canon`. No `prompt_canon` group conflicts were found.

For ranking, sample one response per label per group at each step and rotate through the additional distinct responses for that label across epochs. Do not treat repeated same-label outputs as separate label candidates in the same ranking set. This preserves 18-way label uniqueness while allowing every eligible response to contribute over time.

## V0 overlap

Exact response ID overlap: 0; NFKC/casefold/whitespace-normalized response-text overlap: 0; exact overlap between Open Pool prompt identifiers and V0 `prompt_id` values: 0. V0 contains no original prompt text, so semantic prompt overlap cannot be tested without a reconstruction step.

## Comparison with the V2.14 audit

The prior audit reported 231,336 total rows, 230,664 keyed rows, 38,055 groups (36,068 V14 canonical groups plus 1,987 B2M IDs), 26,383 multi-label groups, and 4,107 groups with repeated model rows. It also found 267 B2M `prompt_id` values reused across source datasets. Recomputing with source-specific B2M namespaces produces 2,256 B2M keys and the strict comparison below. This avoids merging reused source IDs; 672 V14 rows without a key remain outside ranking groups.

| Metric | Open Pool v16 | V2.14 (source-namespaced re-audit) |
|---|---:|---:|
| Classification rows | 231,036 | 231,336 |
| Rows with reliable group key | 231,036 | 230,664 |
| Reliable groups | 37,879 | 38,324 |
| Groups with at least 2 labels | 26,347 | 26,466 |
| Ranking-eligible responses | 219,294 (94.92%) | 218,589 (94.49%) |
| Pairwise comparisons per epoch | 812,835 | 804,628 |
| Groups with repeated same-model rows | 3,975 | 3,894 |
| Exact repeated same-label response excess | 0 | 0 (per prior audit) |
| Prompt-key rows unavailable | 0 | 672 |

V16 therefore adds +705 ranking-eligible response rows (+0.43 percentage points of pool coverage) and +8,207 pairwise comparisons per epoch compared with the strict V2.14 re-audit. V2.14's original published grouping totals differ because the B2M IDs were not source-namespaced in the headline group count; both values are retained in the CSVs for traceability.

## E21 recommendations and answers

1. **Suitable? Yes.** All 231,036 examples are valid for multiclass classification. Reliable prompt IDs support 219,294 ranking-eligible responses and all 18 labels, with no exact duplicate response texts or canonical prompt metadata conflicts.
2. **Retain groups with 2 or 3 distinct labels? Yes.** Keep both. A two-label group is a valid direct contrast. Moving from minimum 2 to minimum 3 drops 8,052 groups and 16,311 ranking-eligible response rows while removing only 8,052 pairwise comparisons per epoch (about 0.99% of the minimum-2 comparisons).
3. **Duplicate model labels?** Select one response per label for each ranking step; rotate the distinct outputs across epochs. Exact duplicate responses are absent, so duplicate rows are useful alternative examples rather than redundant copies.
4. **Ranking coverage?** 94.92% of the dataset (219,294/231,036 responses) can participate under the recommended two-label minimum.
5. **Need embedding-based reconstruction? No.** All rows have a canonical or source-namespaced prompt identifier, canonical group metadata aligns with observed label counts, and no response-ID/text overlap with V0 exists. Reconstructing prompts would add cost and uncertainty; reserve it for a later audit of fallback-only sources if their provenance is challenged.

### Critical issue requiring care

Do not construct groups with raw `prompt_group` alone. It maps to multiple canonical prompt IDs for 752 source identifiers, including identifiers that behave as source-wide collection IDs. Use `prompt_canon` whenever present. Namespace the missing-canon fallback by `source_dataset`, `source_file`, and `prompt_group`. The remaining limitations are that 18,397 canonical groups have no prompt text for semantic spot-checking, and the fallback `panel_labels` value is inconsistent for 1,008 multi-label groups. The source IDs remain usable when namespaced; derive label cardinality from the response rows.

## Supporting files

- `summary.json` — aggregate exact metrics and definitions.
- `classification_label_distribution.csv`, `eligible_responses_by_label.csv` — label populations.
- `groups_by_distinct_label_count.csv`, `groups_by_label_bands.csv` — groups and response rows by 1–18 labels.
- `group_size_distribution_exact.csv` — full exact response-count group-size distribution.
- `ranking_minimum_label_thresholds.csv` — exclusions, item counts and comparisons for minimum 2/3/4/5 labels.
- `group_eligibility_by_key_basis.csv` — canonical versus fallback contributions.
- `source_prompt_id_mapping.csv`, `source_namespaced_prompt_id_mapping.csv`, `source_prompt_id_conflicts.csv`, `fallback_prompt_id_namespace_audit.json` — source ID reuse and ambiguity.
- `canonical_prompt_metadata.csv`, `canonical_group_metadata_consistency.csv`, `prompt_metadata_group_conflicts.csv` — prompt text and panel metadata checks.
- `repeated_model_labels_by_group.csv`, `duplicate_normalized_responses.csv`, `problematic_group_examples.csv` — duplicate and review evidence.
- `v2.14_comparison.csv`, `v2.14_strict_metrics.json` — prior-audit comparison and source-namespaced recalculation.
- `analyze.py`, `write_comparison.py`, `write_report.py` — read-only reproducible audit tools.
