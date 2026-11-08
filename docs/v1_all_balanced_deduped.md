# V1-all Balanced and Deduped Dataset

Input: `data/processed/dataset_v1_all.parquet`  
Output: `data/processed/dataset_v1-all_balanced_deduped.parquet`  
Method: `v1_all_quality_minhash_adaptive_diversity`; seed 2026; word-trigram MinHash candidates verified with character 5-gram Jaccard at threshold 0.90.

## Dataset size and reduction

| Metric | V1-all | New dataset | Change |
|---|---:|---:|---:|
| Total rows | 109,967 | 80,324 | -29,643 (73.0% retained) |
| Unique prompts | 24,098 | 20,411 | -3,687 |
| Unique sources | 55 | 55 | +0 |
| Near-duplicate clusters | N/A | 0 | Verified pairs: 0 |

## Per-class summary

| Model | Original | Retained | Retained % | Unique prompts | Sources |
|---|---:|---:|---:|---:|---:|
| glm-5.3-flash | 12,174 | 12,065 | 99.1% | 8,643 | 15 |
| glm-5.3 | 11,729 | 11,425 | 97.4% | 8,495 | 15 |
| claude-fable-5 | 9,810 | 9,199 | 93.8% | 7,292 | 15 |
| gpt-5.6-luna | 7,077 | 4,133 | 58.4% | 3,500 | 41 |
| gpt-5.6-sol | 6,534 | 4,038 | 61.8% | 3,403 | 41 |
| gpt-5.6-terra | 6,651 | 4,032 | 60.6% | 3,408 | 43 |
| kimi-k3 | 4,585 | 3,757 | 81.9% | 3,614 | 10 |
| gpt-5.5 | 5,525 | 3,746 | 67.8% | 3,151 | 39 |
| qwen3.8-max | 3,762 | 3,669 | 97.5% | 3,345 | 10 |
| gemini-3.7-flash | 4,035 | 3,389 | 84.0% | 3,347 | 6 |
| gemini-3.8-flash | 6,237 | 3,119 | 50.0% | 2,747 | 44 |
| gpt-6-astra | 5,009 | 2,832 | 56.5% | 2,500 | 38 |
| claude-sonnet-5 | 4,762 | 2,802 | 58.8% | 2,741 | 30 |
| claude-opus-5 | 3,903 | 2,561 | 65.6% | 2,529 | 26 |
| gemini-3.6-flash | 5,414 | 2,555 | 47.2% | 2,271 | 38 |
| claude-opus-4-8 | 4,586 | 2,524 | 55.0% | 2,486 | 29 |
| claude-opus-4-6 | 4,223 | 2,429 | 57.5% | 2,392 | 27 |
| gemini-3.1-pro-preview | 3,951 | 2,049 | 51.9% | 1,983 | 34 |

## Filtering summary

| Removal reason | Rows |
|---|---:|
| adaptive_source_redundancy_cap | 29,414 |
| adaptive_prompt_redundancy_cap | 229 |
| missing_or_structurally_empty_required_fields | 0 |
| duplicate_row_id | 0 |
| malformed_placeholder_or_extraction_artifact | 0 |
| explicit_truncation_marker | 0 |
| prompt_echo_or_extraction | 0 |
| provenance_label_conflict | 0 |
| exact_response_label_conflict | 0 |
| near_duplicate_response | 0 |
| repeated_template_boilerplate | 0 |
| exact_duplicate_response | 0 |

Rows removed: **29,643**. Suspicious exact cross-label conflicts: **0 rows**; near-duplicate cross-label clusters flagged for review: **0**.

| Length percentile (characters) | V1-all | New dataset |
|---|---:|---:|
| Min | 189 | 193 |
| P10 | 325 | 330 |
| Median | 645 | 641 |
| P90 | 1,302 | 1,282 |
| Max | 2,123 | 2,123 |

| Source concentration | V1-all | New dataset |
|---|---:|---:|
| Mean largest-source share across classes | 21.1% | 11.7% |
| Mean top-three-source share across classes | 48.3% | 31.3% |

## Assessment

Selection was label-preserving and used no model predictions or validation scores. Exact duplicate text was checked across labels; word-trigram MinHash candidates were verified by character n-gram Jaccard. The reported zero near-duplicate clusters means no candidate pair passed verification; MinHash candidate generation is approximate and does not prove that no near-duplicates exist. Long repeated templates and statistical prompt outliers were reduced within each label. The adaptive source cap is max(Q3, min(Tukey upper fence, 3 times the class median source count)). There are no equal class quotas. Source-cap selection round-robins response-length, formatting and lexical-richness bins while preferring complete provenance.

All 18 labels remain. Validation isolation: 540 frozen V0 rows, split hash `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`, ID overlap 0, prompt overlap 0. The V0 validation data and the V1-all source artifact were read-only.

Removal audit: `data/processed/v1-all_balanced_deduped_removals.csv`. Profile: `data/processed/dataset_profile_v1-all_balanced_deduped.json`.
