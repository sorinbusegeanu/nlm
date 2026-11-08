# V1-all Clean and Compact Datasets

Input: `data/processed/dataset_v1_all.parquet`. Clean removes only explicit defects; compact starts from clean and removes verified same-label near duplicates. Neither pass uses source caps, class quotas, classifier outputs, or validation scores.

## Dataset overview

| Metric | V1-all | Clean | Compact |
|---|---:|---:|---:|
| Total rows | 109,967 | 109,967 | 109,967 |
| Unique prompts | 24,098 | 24,098 | 24,098 |
| Unique sources | 55 | 55 | 55 |
| Exact/normalized duplicate responses removed in Clean | — | 0 | — |
| Verified near-duplicate pairs | — | — | 0 |
| Model-label conflict rows removed in Clean | — | 0 | — |

## Per-model row counts

| Model | V1-all | Clean | Compact | Clean retained | Compact retained from Clean |
|---|---:|---:|---:|---:|---:|
| claude-fable-5 | 9,810 | 9,810 | 9,810 | 100.0% | 100.0% |
| claude-opus-4-6 | 4,223 | 4,223 | 4,223 | 100.0% | 100.0% |
| claude-opus-4-8 | 4,586 | 4,586 | 4,586 | 100.0% | 100.0% |
| claude-opus-5 | 3,903 | 3,903 | 3,903 | 100.0% | 100.0% |
| claude-sonnet-5 | 4,762 | 4,762 | 4,762 | 100.0% | 100.0% |
| gemini-3.1-pro-preview | 3,951 | 3,951 | 3,951 | 100.0% | 100.0% |
| gemini-3.6-flash | 5,414 | 5,414 | 5,414 | 100.0% | 100.0% |
| gemini-3.7-flash | 4,035 | 4,035 | 4,035 | 100.0% | 100.0% |
| gemini-3.8-flash | 6,237 | 6,237 | 6,237 | 100.0% | 100.0% |
| glm-5.3 | 11,729 | 11,729 | 11,729 | 100.0% | 100.0% |
| glm-5.3-flash | 12,174 | 12,174 | 12,174 | 100.0% | 100.0% |
| gpt-5.5 | 5,525 | 5,525 | 5,525 | 100.0% | 100.0% |
| gpt-5.6-luna | 7,077 | 7,077 | 7,077 | 100.0% | 100.0% |
| gpt-5.6-sol | 6,534 | 6,534 | 6,534 | 100.0% | 100.0% |
| gpt-5.6-terra | 6,651 | 6,651 | 6,651 | 100.0% | 100.0% |
| gpt-6-astra | 5,009 | 5,009 | 5,009 | 100.0% | 100.0% |
| kimi-k3 | 4,585 | 4,585 | 4,585 | 100.0% | 100.0% |
| qwen3.8-max | 3,762 | 3,762 | 3,762 | 100.0% | 100.0% |

## Removal audit

### Clean

| Reason | Rows removed |
|---|---:|
| `conflicting_labels_for_identical_response` | 0 |
| `contradictory_model_version` | 0 |
| `duplicate_row_id` | 0 |
| `explicit_truncation_or_failure_marker` | 0 |
| `missing_or_empty_required_field` | 0 |
| `normalized_duplicate_response` | 0 |
| `response_is_copied_prompt` | 0 |
| `serialized_envelope_or_extraction_artifact` | 0 |

### Clean examples

No rows were removed in this pass.

### Compact examples

No rows were removed in this pass.

## Redundancy diagnostics

Near-duplicate detection unions character 5-gram and word-trigram MinHash candidates (candidate threshold 0.80), then verifies exact character 5-gram Jaccard ≥ 0.92 and word 3-gram Jaccard ≥ 0.85. Verified pairs: **0**; connected clusters removed: **0**. Cluster sizes: `[]`.
Of 6,967 same-label candidate pairs, character 5-gram Jaccard P50/P90/max = 0.05824111822947001, 0.3371884652365922, 0.8495821727019499; word 3-gram Jaccard P50/P90/max = 0.0, 0.12246938775510206, 0.7575757575757576. Verified-pair medians = None (character) and None (word); pairs sharing a prompt ID: 0.
Repeated normalized templates occurring at least three times: 0 templates / 0 rows. Prompt groups with multiple same-label rows: 10,733 groups / 28,611 rows. These diagnostic counts did not independently cause removals.
TF-IDF diagnostic: word unigram/bigram TF-IDF cosine nearest-neighbor on deterministic sample; sample 3,000; nearest-neighbor cosine similarity P50/P90/max = 0.1763935089111328, 0.4228316843509674, 0.6924915313720703. Diagnostic only; not used to select rows.

## Response characteristics

| Statistic | V1-all | Clean | Compact |
|---|---:|---:|---:|
| Character length median (P10–P90) | 645 (325–1302) | 645 (325–1302) | 645 (325–1302) |
| Word count median (P10–P90) | 101 (49–206) | 101 (49–206) | 101 (49–206) |
| Mean type-token ratio | 79.33% | 79.33% | 79.33% |
| Markdown/structured formatting fraction | 0.04% | 0.04% | 0.04% |
| Code block fraction | 0.00% | 0.00% | 0.00% |
| URL fraction | 0.00% | 0.00% | 0.00% |
| Median paragraphs per response | 1.0 | 1.0 | 1.0 |
| Opening templates repeated ≥10 times | 201 | 201 | 201 |
| Ending templates repeated ≥10 times | 4 | 4 | 4 |

## Integrity and validation

All 18 labels remain. Clean rows originate from V1-all; compact is a subset of clean. Both profiles reuse the frozen V0 validation set (540 rows; split hash `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`), with zero validation ID or prompt-group overlap. The data loader adds the same V0 validation rows at training time; these Parquet artifacts contain training rows only.

Current V1-all removals by rule: `missing_or_empty_required_field` 0, `duplicate_row_id` 0, `serialized_envelope_or_extraction_artifact` 0, `explicit_truncation_or_failure_marker` 0, `conflicting_labels_for_identical_response` 0, `normalized_duplicate_response` 0, `contradictory_model_version` 0, `response_is_copied_prompt` 0.
Removed rows with a retained representative: Clean N/A (no rows removed); Compact N/A (no rows removed). Similarity is evidence of redundancy, not proof that either response is defective or that training accuracy will improve.

### Ambiguous findings retained

The source has no explicit completion-status or finish-reason field. Therefore only explicit text markers and direct metadata contradictions were treated as defects; plausible but unmarked truncation, unusual writing, and uncertain provenance were retained. `model_version` conflicts were checked only when the value exactly matched another canonical label. No attempt was made to verify that the claimed model generated a response.

MinHash is an approximate candidate index. Zero verified pairs means no candidate pair passed both configured thresholds; it is not proof that no near-duplicates exist. TF-IDF diagnostics cover only the deterministic 3,000-row sample and did not affect selection.

Profiles: `dataset_profile_v1-all_clean.json`, `dataset_profile_v1-all_compact.json`. Audits: `data/processed/v1-all_clean_removals.csv`, `data/processed/v1-all_compact_removals.csv`.
