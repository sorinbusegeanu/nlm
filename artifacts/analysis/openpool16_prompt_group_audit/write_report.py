#!/usr/bin/env python3
"""Render the Open Pool v16 E21 grouping eligibility report from audit outputs."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/analysis/openpool16_prompt_group_audit'
m=json.loads((OUT/'summary.json').read_text())
v14=json.loads((OUT/'v2.14_strict_metrics.json').read_text())

def pc(x,n): return f'{x:,} ({100*x/n:.2f}%)' if n else str(x)
def p(x): return f'{x:.2f}%'
def csv(name): return pd.read_csv(OUT/name)
# Group response-count distribution, rolling exact sizes into readable bins.
sz=csv('group_size_distribution_exact.csv')
bins=[('1',1,1),('2',2,2),('3',3,3),('4–5',4,5),('6–9',6,9),('10–18',10,18),('19–30',19,30),('31–50',31,50),('51–100',51,100)]
size_lines=[]
for name,lo,hi in bins:
    q=sz[sz.rows.between(lo,hi)]
    size_lines.append(f"| {name} | {int(q.groups.sum()):,} | {int(q.responses.sum()):,} | {100*q.groups.sum()/m['reliable_prompt_groups']:.2f}% |")
# V2.14 historical comparison metrics as originally reported, plus strict re-key values.
comp=csv('v2.14_comparison.csv')
def val(ds,metric):
    r=comp[(comp.dataset==ds)&(comp.metric==metric)]
    return r.value.iloc[0] if len(r) else None
v16=m
# Write source-specific fallback summary for provenance review.
raw=pd.read_parquet(ROOT/'data/raw/open_pool/v16/ntl_open_pool.parquet')
raw['prompt_canon']=raw.prompt_canon.fillna('').astype(str).str.strip()
raw['source_dataset']=raw.source_dataset.fillna('').astype(str).str.strip()
raw['source_file']=raw.source_file.fillna('').astype(str).str.strip()
raw['prompt_group']=raw.prompt_group.fillna('').astype(str).str.strip()
fallback=raw[raw.prompt_canon.eq('')].groupby(['source_dataset','source_file','prompt_group']).agg(
    responses=('id','size'),distinct_labels=('model','nunique'),response_ids=('id',lambda s:'|'.join(s.astype(str).head(8)))).reset_index()
fb=fallback.groupby('source_dataset').agg(groups=('prompt_group','size'),responses=('responses','sum'),
    multi_label_groups=('distinct_labels',lambda s:int(s.ge(2).sum())),
    eligible_responses=('responses',lambda s:int(s[fallback.loc[s.index,'distinct_labels'].ge(2)].sum()))).reset_index()
fb.to_csv(OUT/'missing_canon_source_fallback_by_source.csv',index=False)
# Prompt ID conflict file is written separately by write_comparison.py.
lines=[
'# Open Pool v16 Prompt-Group Audit for E21', '',
'## Executive summary','',
'**Open Pool v16 is suitable for E21 classification and group-ranking training**, provided ranking uses canonical prompt groups first and a source/file-namespaced fallback only when `prompt_canon` is absent. The main parquet has 231,036 valid, labeled response rows; all 18 classes are represented. All rows have a usable identifier under this rule; the source-ID fallback carries a panel-label caveat described below. Of those, 219,294 responses (94.92%) belong to 26,347 groups with at least two distinct labels. That yields 170,233 ranked items and 812,835 pairwise label comparisons per epoch when each group contributes one response per label.','',
'**Recommendation:** use minimum 2 distinct labels. Retain both two-label and three-label groups; two-label groups supply 8,052 direct comparisons over 16,311 responses, while minimum 3 would remove those useful examples and save only those 8,052 comparisons. Rotate the distinct same-label responses across epochs. Use `prompt_canon` as the primary key; never use `prompt_group` by itself. No embedding-based reconstruction is indicated for this dataset.','',
'## Dataset and key audit','',
'Input is the original `data/raw/open_pool/v16/ntl_open_pool.parquet` (231,036 rows; SHA-256 `'+m['dataset_sha256']+'`). The separately supplied `prompts.parquet` contains canonical prompt text for 17,671 prompt keys (SHA-256 `'+m['prompt_metadata_sha256']+'`). The incremental dataset was not used.','',
'| Check | Result |', '|---|---:|',
 f"| Classification responses with nonempty text and canonical label | {m['valid_classification_responses']:,}/{m['dataset_rows']:,} (100.00%) |",
 f"| Canonical labels represented | {m['classification_label_coverage']}/18 |",
 f"| Rows with `prompt_canon` | {m['rows_with_prompt_canon']:,} ({m['prompt_canon_missing_pct'] and 100-m['prompt_canon_missing_pct']:.2f}%) |",
 f"| Rows using fallback source prompt identifier | {m['rows_missing_prompt_canon']:,} ({m['prompt_canon_missing_pct']:.2f}%) |",
 f"| Rows with no usable identifier | {m['rows_missing_any_usable_id']:,} |",
 f"| Distinct `prompt_canon` groups | {m['canonical_prompt_keys_joined']:,} |",
 f"| Canonical prompt text available / unavailable | {m['canonical_prompt_keys_with_text']:,} / {m['canonical_prompt_keys_without_text']:,} groups; {m['canonical_prompt_rows_without_prompt_text']:,} rows lack prompt text |",
 f"| Canonical keys with conflicting prompt text | {m['prompt_canon_keys_with_conflicting_canonical_text']:,} |",
 f"| `panel_labels` metadata vs observed distinct labels | {m['canonical_groups_where_panel_labels_equal_observed_distinct_labels']:,}/{m['canonical_prompt_keys_joined']:,} canonical groups match |",
 f"| Canonical groups with inconsistent `panel_labels` across rows | {m['canonical_groups_with_inconsistent_panel_labels']:,} |",
 f"| Fallback multi-label groups where `panel_labels` disagrees with observed label count | {m['fallback_groups_where_panel_labels_differ_from_observed_distinct_labels']:,} groups / {m['fallback_group_rows_with_panel_label_mismatch']:,} rows |",
 f"| Source prompt IDs shared across source datasets | {m['source_prompt_ids_shared_across_datasets']:,} |",
 f"| Source prompt IDs reused across files within a dataset | {m['source_dataset_prompt_ids_reused_across_files']:,}; fallback key includes `source_file` |",
 f"| `prompt_group` IDs mapping to multiple `prompt_canon` keys | {m['source_prompt_ids_mapping_multiple_prompt_canon']:,} (maximum {int(csv('source_prompt_id_mapping.csv').canon_keys.max()):,} canonical keys for one source ID) |",
 f"| Canonical prompt groups spanning multiple source datasets | {m['canonical_groups_spanning_sources']:,}; canonical key deliberately merges these source-local IDs |",
 '',
'Of 36,068 canonical group keys, 18,397 (51.00%) have no corresponding text row in `prompts.parquet`; 40,775 response rows (18.11% of rows with `prompt_canon`) lack inspectable canonical prompt text. These keys remain usable as identifiers and showed no conflicting text among the available prompt records. The limitation is manual semantic inspection, not key ambiguity.','',
'`prompt_group` is not safe as a primary key: 752 source IDs map to multiple canonical prompts (the largest maps to 2,914). For rows with `prompt_canon`, grouping by that canonical key avoids this conflict. For the 5,926 rows with no `prompt_canon`, use `(source_dataset, source_file, prompt_group)`. There are no cross-source raw ID collisions. The 1,811 fallback source IDs are disjoint from IDs on rows with `prompt_canon`; none are reused across source files or source datasets. Source-file namespacing is retained as the conservative key. The fallback keys have 1,811 groups, of which 1,008 are multi-label. Those 1,008 groups (5,123 rows) have `panel_labels=1` while observed membership has 2–12 labels; this field is inconsistent for fallback groups. Derive ranking cardinality from observed distinct `model` labels. See `source_prompt_id_conflicts.csv`, `source_namespaced_prompt_id_mapping.csv`, and `missing_canon_source_fallback_by_source.csv`.','',
'## Classification population','',
f"All **{m['valid_classification_responses']:,}** rows are valid classification examples, with no missing/empty text and no invalid labels. Response IDs are unique. All 18 labels appear; their counts and ranking coverage are in `classification_label_distribution.csv` and `eligible_responses_by_label.csv`. The `lab` family metadata agrees with the canonical label for all rows, and there are {m['label_metadata']['known_canonical_served_model_mismatches']:,} contradictions for served-model names that can be mapped exactly to a canonical label.",'',
'## Group inventory and ranking eligibility','',
f"Using normalized `prompt_canon` first and the source-namespaced fallback for rows lacking it yields **{m['reliable_prompt_groups']:,} groups** across all {m['reliable_grouped_responses']:,} responses. Group size by responses has median {m['group_sizes_summary']['median']:.0f}, mean {m['group_sizes_summary']['mean']:.2f}, p90 {m['group_sizes_summary']['p90']:.0f}, p95 {m['group_sizes_summary']['p95']:.0f}, p99 {m['group_sizes_summary']['p99']:.0f}, and maximum {m['group_sizes_summary']['max']}. The maximum is 98 response rows due to repeat labels; the maximum distinct-label count is {m['max_distinct_labels']}, and no group has more than 18 labels.",'',
'### Groups by distinct label count','',
'| Distinct labels | Groups | Response rows |', '|---:|---:|---:|']
for r in csv('groups_by_distinct_label_count.csv').itertuples(index=False):
    lines.append(f'| {r.distinct_labels} | {r.groups:,} | {r.responses:,} |')
lines += ['', '### Group size by number of response rows', '',
'| Responses per group | Groups | Response rows | Share of groups |', '|---:|---:|---:|---:|',*size_lines,
'', '### Ranking eligibility thresholds', '',
'Excluded counts are relative to all 37,879 reliable groups / 231,036 responses. Pairwise comparisons count one comparison per pair of distinct labels in one ranking set per epoch; duplicate same-label outputs rotate across epochs and do not multiply same-epoch comparisons.', '',
'| Minimum distinct labels | Eligible groups | Eligible response rows | Excluded groups | Excluded response rows | Rows retained | Ranking items / epoch | Pairwise comparisons / epoch |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in csv('ranking_minimum_label_thresholds.csv').itertuples(index=False):
    lines.append(f'| {r.minimum_distinct_labels} | {r.eligible_groups:,} | {r.eligible_responses:,} | {r.excluded_groups:,} | {r.excluded_responses:,} | {r.eligible_response_pct_of_all_v16:.2f}% | {r.ranking_items_per_epoch:,} | {r.pairwise_label_comparisons_per_epoch:,} |')
lines += ['',
'At the recommended minimum of two labels, 26,347 groups provide 219,294 response rows (94.92% of the full pool), 170,233 selected response-label items per epoch, and 812,835 pairwise comparisons per epoch. Every one of the 18 labels appears in eligible groups. Per-label eligible row counts and coverage are in `eligible_responses_by_label.csv`.','',
'| Label | Classification rows | Ranking-eligible rows | Eligible / class rows |', '|---|---:|---:|---:|']
for r in csv('eligible_responses_by_label.csv').itertuples(index=False):
    lines.append(f'| `{r.label}` | {r.classification_responses:,} | {r.ranking_responses:,} | {r.ranking_response_pct_of_classification:.2f}% |')
lines += ['', '## Duplicate and consistency checks','',
 f"- Exact duplicate normalized response clusters across the main parquet: **{m['duplicate_summary']['normalized_text_duplicate_clusters']:,}** ({m['duplicate_summary']['duplicate_excess_rows']:,} excess rows). Conflicting-label duplicate text clusters: **{m['duplicate_summary']['conflicting_label_duplicate_clusters']:,}**.",
 f"- Groups with repeated rows for at least one model: **{m['groups_with_repeated_same_model']:,}**; these contain **{m['repeated_label_summary']['group_label_pairs_with_repeated_rows']:,}** repeated group/label pairs and **{m['responses_in_repeated_label_groups']:,}** rows.",
 f"- Every repeated group/label pair has distinct normalized response text ({m['repeated_label_summary']['repeated_label_outputs_distinct_pct']:.2f}%); exact duplicate outputs within a group and model: **{m['repeated_label_exact_duplicate_excess_rows']:,}**.",
 '- Canonical `panel_labels` equals the observed distinct-label count in all 36,068 canonical groups; there are no prompt-text conflicts in the supplied canonical prompt table.',
 f"- Fallback metadata caveat: `panel_labels=1` in {m['fallback_groups_where_panel_labels_differ_from_observed_distinct_labels']:,} multi-label source-ID groups ({m['fallback_group_rows_with_panel_label_mismatch']:,} rows), though observed membership has 2–12 labels; do not use this field to set rank cardinality.",
 '- Groups with more than 18 distinct labels: **0**. Group sizes above 18 are repeated responses for existing labels, not extra classes.',
 '- Potential grouping risk: the 752 ambiguous `prompt_group` IDs must not be used instead of `prompt_canon`. No `prompt_canon` group conflicts were found.',
 '',
'For ranking, sample one response per label per group at each step and rotate through the additional distinct responses for that label across epochs. Do not treat repeated same-label outputs as separate label candidates in the same ranking set. This preserves 18-way label uniqueness while allowing every eligible response to contribute over time.','',
'## V0 overlap','',
f"Exact response ID overlap: {m['v0_overlap']['response_id_overlap_rows']:,}; NFKC/casefold/whitespace-normalized response-text overlap: {m['v0_overlap']['normalized_response_text_overlap_rows']:,}; exact overlap between Open Pool prompt identifiers and V0 `prompt_id` values: {m['v0_overlap']['openpool_rows_with_prompt_canon_matching_v0_prompt_id']+m['v0_overlap']['openpool_rows_with_prompt_group_matching_v0_prompt_id']:,}. V0 contains no original prompt text, so semantic prompt overlap cannot be tested without a reconstruction step.",'',
'## Comparison with the V2.14 audit','',
'The prior audit reported 231,336 total rows, 230,664 keyed rows, 38,055 groups (36,068 V14 canonical groups plus 1,987 B2M IDs), 26,383 multi-label groups, and 4,107 groups with repeated model rows. It also found 267 B2M `prompt_id` values reused across source datasets. Recomputing with source-specific B2M namespaces produces 2,256 B2M keys and the strict comparison below. This avoids merging reused source IDs; 672 V14 rows without a key remain outside ranking groups.','',
'| Metric | Open Pool v16 | V2.14 (source-namespaced re-audit) |', '|---|---:|---:|',
 f"| Classification rows | {m['valid_classification_responses']:,} | {int(v14['classification_responses']):,} |",
 f"| Rows with reliable group key | {m['reliable_grouped_responses']:,} | {int(v14['rows_with_reliable_prompt_key']):,} |",
 f"| Reliable groups | {m['reliable_prompt_groups']:,} | {int(v14['groups']):,} |",
 f"| Groups with at least 2 labels | {m['groups_with_2plus_labels']:,} | {int(v14['multi_label_groups']):,} |",
 f"| Ranking-eligible responses | {m['responses_in_groups_with_2plus_labels']:,} ({m['ranking_responses_pct_all']:.2f}%) | {int(v14['multi_label_responses']):,} ({v14['multi_label_response_pct_of_all']:.2f}%) |",
 f"| Pairwise comparisons per epoch | {m['pairwise_label_comparisons_per_epoch_min2']:,} | {int(v14['pairwise_comparisons_per_epoch']):,} |",
 f"| Groups with repeated same-model rows | {m['groups_with_repeated_same_model']:,} | {int(v14['groups_with_repeated_same_model_responses']):,} |",
 f"| Exact repeated same-label response excess | {m['repeated_label_exact_duplicate_excess_rows']:,} | 0 (per prior audit) |",
 f"| Prompt-key rows unavailable | {m['rows_missing_any_usable_id']:,} | {int(v14['unkeyed_rows']):,} |",
 '',
 f"V16 therefore adds {int(m['responses_in_groups_with_2plus_labels']-v14['multi_label_responses']):+,} ranking-eligible response rows (+{m['ranking_responses_pct_all']-v14['multi_label_response_pct_of_all']:.2f} percentage points of pool coverage) and {int(m['pairwise_label_comparisons_per_epoch_min2']-v14['pairwise_comparisons_per_epoch']):+,} pairwise comparisons per epoch compared with the strict V2.14 re-audit. V2.14's original published grouping totals differ because the B2M IDs were not source-namespaced in the headline group count; both values are retained in the CSVs for traceability.",'',
'## E21 recommendations and answers','',
'1. **Suitable? Yes.** All 231,036 examples are valid for multiclass classification. Reliable prompt IDs support 219,294 ranking-eligible responses and all 18 labels, with no exact duplicate response texts or canonical prompt metadata conflicts.',
'2. **Retain groups with 2 or 3 distinct labels? Yes.** Keep both. A two-label group is a valid direct contrast. Moving from minimum 2 to minimum 3 drops 8,052 groups and 16,311 ranking-eligible response rows while removing only 8,052 pairwise comparisons per epoch (about 0.99% of the minimum-2 comparisons).',
'3. **Duplicate model labels?** Select one response per label for each ranking step; rotate the distinct outputs across epochs. Exact duplicate responses are absent, so duplicate rows are useful alternative examples rather than redundant copies.',
 f"4. **Ranking coverage?** {m['ranking_responses_pct_all']:.2f}% of the dataset ({m['responses_in_groups_with_2plus_labels']:,}/{m['dataset_rows']:,} responses) can participate under the recommended two-label minimum.",
'5. **Need embedding-based reconstruction? No.** All rows have a canonical or source-namespaced prompt identifier, canonical group metadata aligns with observed label counts, and no response-ID/text overlap with V0 exists. Reconstructing prompts would add cost and uncertainty; reserve it for a later audit of fallback-only sources if their provenance is challenged.','',
'### Critical issue requiring care','',
'Do not construct groups with raw `prompt_group` alone. It maps to multiple canonical prompt IDs for 752 source identifiers, including identifiers that behave as source-wide collection IDs. Use `prompt_canon` whenever present. Namespace the missing-canon fallback by `source_dataset`, `source_file`, and `prompt_group`. The remaining limitations are that 18,397 canonical groups have no prompt text for semantic spot-checking, and the fallback `panel_labels` value is inconsistent for 1,008 multi-label groups. The source IDs remain usable when namespaced; derive label cardinality from the response rows.','',
'## Supporting files','',
'- `summary.json` — aggregate exact metrics and definitions.',
'- `classification_label_distribution.csv`, `eligible_responses_by_label.csv` — label populations.',
'- `groups_by_distinct_label_count.csv`, `groups_by_label_bands.csv` — groups and response rows by 1–18 labels.',
'- `group_size_distribution_exact.csv` — full exact response-count group-size distribution.',
'- `ranking_minimum_label_thresholds.csv` — exclusions, item counts and comparisons for minimum 2/3/4/5 labels.',
'- `group_eligibility_by_key_basis.csv` — canonical versus fallback contributions.',
'- `source_prompt_id_mapping.csv`, `source_namespaced_prompt_id_mapping.csv`, `source_prompt_id_conflicts.csv`, `fallback_prompt_id_namespace_audit.json` — source ID reuse and ambiguity.',
'- `canonical_prompt_metadata.csv`, `canonical_group_metadata_consistency.csv`, `prompt_metadata_group_conflicts.csv` — prompt text and panel metadata checks.',
'- `repeated_model_labels_by_group.csv`, `duplicate_normalized_responses.csv`, `problematic_group_examples.csv` — duplicate and review evidence.',
'- `v2.14_comparison.csv`, `v2.14_strict_metrics.json` — prior-audit comparison and source-namespaced recalculation.',
'- `analyze.py`, `write_comparison.py`, `write_report.py` — read-only reproducible audit tools.'
]
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(OUT/'report.md')
