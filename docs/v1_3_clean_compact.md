# V1.3 Clean and Compact Datasets

Input: `data/processed/dataset_v1.3.parquet`. Clean removes only explicit defects; compact starts from clean and removes verified same-label near duplicates. Neither pass uses source caps, class quotas, classifier outputs, or validation scores.

## Dataset overview

| Metric | V1.3 | Clean | Compact |
|---|---:|---:|---:|
| Total rows | 101,359 | 92,096 | 92,093 |
| Unique prompts | 43,300 | 42,879 | 42,879 |
| Unique sources | 3 | 3 | 3 |
| Exact/normalized duplicate responses removed in Clean | — | 18 | — |
| Verified near-duplicate pairs | — | — | 3 |
| Model-label conflict rows removed in Clean | — | 0 | — |

## Per-model row counts

| Model | V1.3 | Clean | Compact | Clean retained | Compact retained from Clean |
|---|---:|---:|---:|---:|---:|
| glm-5.3-flash | 86,006 | 85,068 | 85,065 | 98.9% | 100.0% |
| gpt-5.6-luna | 886 | 886 | 886 | 100.0% | 100.0% |
| gpt-5.6-sol | 14,467 | 6,142 | 6,142 | 42.5% | 100.0% |

## Removal audit

### Clean

| Reason | Rows removed |
|---|---:|
| `missing_or_empty_required_field` | 8,313 |
| `source_explicit_truncation_metadata` | 912 |
| `normalized_duplicate_response` | 18 |
| `serialized_envelope_or_extraction_artifact` | 14 |
| `explicit_truncation_or_failure_marker` | 6 |
| `conflicting_labels_for_identical_response` | 0 |
| `contradictory_model_version` | 0 |
| `duplicate_row_id` | 0 |
| `response_is_copied_prompt` | 0 |
| `source_explicit_cap_hit` | 0 |

### Clean examples

**`explicit_truncation_or_failure_marker` — 6 rows.** Example rows (up to five):
- `v13_095e09e21de7eddc1aa28687c4846b25`: response ends with an explicit truncation/failure marker.
- `v13_844a658d1f07d33cee7db00444f8b8da`: response ends with an explicit truncation/failure marker.
- `v13_b2bfd53afa7386765ca7f79c4a3b0550`: response ends with an explicit truncation/failure marker.
- `v13_b9df229a85af8f4c2f4e3175b4afa5c2`: response ends with an explicit truncation/failure marker.
- `v13_df58594b7b7b6b3dba199e44f071bbf6`: response ends with an explicit truncation/failure marker.
**`missing_or_empty_required_field` — 8,313 rows.** Example rows (up to five):
- `v13_001146a74c8e499f71c01fc799970801`: one or more required fields are null/empty.
- `v13_0019f2212d7b19f0690f2ca513bc9455`: one or more required fields are null/empty.
- `v13_00245ba93e9b7da339d076e5531a4cd3`: one or more required fields are null/empty.
- `v13_0038f2e8fa8af673b13269b1f5f41063`: one or more required fields are null/empty.
- `v13_003ebc64f2d8375901baa4ef80d6b560`: one or more required fields are null/empty.
**`normalized_duplicate_response` — 18 rows.** Example rows (up to five):
- `v13_0402def81fe7051fb4359263379cb565`: normalized text equals retained representative v13_d876f09a39386c9ddd0c35fd2e04561d.
- `v13_19a24d6fe616d50ee782a80efc34d3d9`: normalized text equals retained representative v13_d876f09a39386c9ddd0c35fd2e04561d.
- `v13_1a0d4375d36b687454054ba8d59f0496`: normalized text equals retained representative v13_65706274f4a5a2bc5fd477ad42690172.
- `v13_219adfa6942d03d6bf8507def13ee71d`: normalized text equals retained representative v13_77b92f33d1815d78d94b0273e5904220.
- `v13_2751f1aed1793a54b224a08a9cd39d68`: normalized text equals retained representative v13_d37d7ecba84741ba90791b3b9f2510ae.
**`serialized_envelope_or_extraction_artifact` — 14 rows.** Example rows (up to five):
- `v13_105aee02f837c03103369665fc17bc2a`: matched explicit serialized role/object/tool-log marker.
- `v13_193fbce84922f6e42e9a1999150432f7`: matched explicit serialized role/object/tool-log marker.
- `v13_1fae0336aeb47a01b0d9c48508a3ee26`: matched explicit serialized role/object/tool-log marker.
- `v13_34721eca7ebc485679930fa3bdac3b26`: matched explicit serialized role/object/tool-log marker.
- `v13_367439be508baf7bb6ed43d3a4838cb7`: matched explicit serialized role/object/tool-log marker.
**`source_explicit_truncation_metadata` — 912 rows.** Example rows (up to five):
- `v13_0091fed636382eeb8b156a31a7d41f6e`: source metadata explicitly marks a truncated or length-capped response.
- `v13_01665f7e3d2907305f20bfff86c88255`: source metadata explicitly marks a truncated or length-capped response.
- `v13_016a68cc84708bef15a40939686d6f36`: source metadata explicitly marks a truncated or length-capped response.
- `v13_0173879fffdef0f0696f458f36149573`: source metadata explicitly marks a truncated or length-capped response.
- `v13_01a953c2965d1c19efbcdfa78d16d251`: source metadata explicitly marks a truncated or length-capped response.

### Compact examples

**`verified_same_label_near_duplicate` — 3 rows.** Example rows (up to five):
- `v13_8b2c91ebce6754f98a485b26f6fa0ec8`; representative `v13_5f0ec57ce0f499cfe90f684dbab023ce`: char5_jaccard=0.9646; word3_jaccard=0.9254.
- `v13_b9a4648ca5c978c088d95beb63b365c0`; representative `v13_f8b4586f0cc904c7b41ef0c58e75523e`: char5_jaccard=0.9577; word3_jaccard=0.9077.
- `v13_eb396eaa4d4e24f4b02be1b5fd1bede4`; representative `v13_51ae6324e1241a72938ec763771cc2ad`: char5_jaccard=0.9511; word3_jaccard=0.8545.

## Redundancy diagnostics

Near-duplicate detection unions character 5-gram and word-trigram MinHash candidates (candidate thresholds: character 0.92, word 0.85), then verifies exact character 5-gram Jaccard ≥ 0.92 and word 3-gram Jaccard ≥ 0.85. Verified pairs: **3**; connected clusters removed: **3**. Cluster sizes: `[2, 2, 2]`.
Of 82 same-label candidate pairs, character 5-gram Jaccard P50/P90/max = 0.894749820868578, 0.9602169628748469, 0.9721448467966574; word 3-gram Jaccard P50/P90/max = 0.603988603988604, 0.8233193277310924, 0.9253731343283582. Verified-pair medians = 0.9576547231270358 (character) and 0.9076923076923077 (word); pairs sharing a prompt ID: 3.
Repeated normalized templates occurring at least three times: 2 templates / 10 rows. Prompt groups with multiple same-label rows: 39,598 groups / 88,815 rows. These diagnostic counts did not independently cause removals.
TF-IDF diagnostic: word unigram/bigram TF-IDF cosine nearest-neighbor on deterministic sample; sample 3,000; nearest-neighbor cosine similarity P50/P90/max = 0.27235153317451477, 0.46031653881073, 0.9808803200721741. Diagnostic only; not used to select rows.

## Response characteristics

| Statistic | V1.3 | Clean | Compact |
|---|---:|---:|---:|
| Character length median (P10–P90) | 5897 (268–18036) | 6190 (1098–18136) | 6190 (1099–18136) |
| Word count median (P10–P90) | 810 (40–3053) | 858 (174–3068) | 858 (174–3068) |
| Mean type-token ratio | 34.67% | 38.04% | 38.04% |
| Markdown/structured formatting fraction | 72.63% | 78.93% | 78.93% |
| Code block fraction | 34.13% | 37.06% | 37.06% |
| URL fraction | 6.21% | 6.43% | 6.43% |
| Median paragraphs per response | 24.0 | 26.0 | 26.0 |
| Opening templates repeated ≥10 times | 800 | 793 | 793 |
| Ending templates repeated ≥10 times | 104 | 101 | 101 |

## Integrity and validation

All 3 labels in the source remain. Compact is a subset of clean. Both profiles are checked against the frozen V0 validation set (split hash `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`), with zero validation ID or prompt-group overlap. These Parquet artifacts contain training rows only.

V1.3 removals by rule: `missing_or_empty_required_field` 8,313, `duplicate_row_id` 0, `serialized_envelope_or_extraction_artifact` 14, `explicit_truncation_or_failure_marker` 6, `source_explicit_truncation_metadata` 912, `source_explicit_cap_hit` 0, `conflicting_labels_for_identical_response` 0, `normalized_duplicate_response` 18, `contradictory_model_version` 0, `response_is_copied_prompt` 0.
Removed rows with a retained representative: Clean 0.0%; Compact 100.0%. Similarity is evidence of redundancy, not proof that either response is defective or that training accuracy will improve.

### Ambiguous findings retained

Only explicit text markers and available source metadata were treated as defects; plausible but unmarked truncation, unusual writing, and uncertain provenance were retained. `model_version` conflicts were checked only when the value exactly matched another canonical label. No attempt was made to verify that the claimed model generated a response.
MinHash is an approximate candidate index. Zero verified pairs means no candidate pair passed both configured thresholds; it is not proof that no near-duplicates exist. TF-IDF diagnostics cover only the deterministic 3,000-row sample and did not affect selection.

Profiles: `dataset_profile_v1.3_clean.json`, `dataset_profile_v1.3_compact.json`. Audits: `data/processed/v1.3_clean_removals.csv`, `data/processed/v1.3_compact_removals.csv`.
