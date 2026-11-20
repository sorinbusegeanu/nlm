# Open Pool v16 incremental dataset

Downloaded and curated the latest `crtproduction/name-that-llm-open-pool` snapshot against the immutable V2.16 parent. V2.16 was not modified; no training or Kaggle submission was run.

## Source snapshot

- Kaggle dataset version: **18** (current version when downloaded).
- Publisher release: **Open Pool v16, 2026-11-18**; Kaggle last updated: `2026-11-18T14:49:47.833Z`.
- Main parquet SHA-256: `7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492`.
- Metadata SHA-256: `8a191e405a11b0b1c562f09c7c37f4e87503e2355b88615a219fb934a4e743e7`.
- V2.16 parent SHA-256: `3bcd2d4695ddd5accaf30dbff2df01f99dfc5110b57f78145adeeaed63e6a5d8`.
- V0 SHA-256: `45f41931bfffeb54c286ec6a814b723d6fcb000910171b7be5dbd9205b7b32b5`.
- Raw files remain under [`data/raw/open_pool/v16`](../data/raw/open_pool/v16). Per-file SHA-256 values are in [`dataset_profile_v2.16_openpool16_increment.json`](../data/processed/dataset_profile_v2.16_openpool16_increment.json).
- The archive main file carries the v16 source pool; the separate `gemini37_unverified.parquet` sidecar was excluded in full.

## Row accounting

| Metric | Rows |
|---|---:|
| Downloaded Open Pool v16 main parquet | 231,036 |
| Already present in V2.16 by response ID or normalized text | 223,840 |
| New before quality filtering (after within-file exact deduplication) | 7,196 |
| Removed by quality criteria | 6 |
| Final accepted additions | 7,190 |
| Within-download duplicate rows (also reported separately) | 0 |
| Within-download conflicting-label rows (also reported separately) | 0 |
| Separate unverified Gemini 3.7 sidecar rows excluded | 10,838 |

Within-file duplicate/conflict rows are accounted for before the ‘new before quality filtering’ count; row accounting reconciles to the downloaded main parquet.

## Quality criteria and exclusions

The v16 main parquet already applies the publisher’s documented source-quality review, including API dump/refusal/off-genre/incomplete-text and known-label checks. Local filtering additionally requires valid text, canonical label, source/file/license/prompt provenance, a known CC license, at most 2,000 characters, no exact V2.16/V0 ID or normalized-response overlap, no known V0 prompt-group overlap, no explicit serialization/refusal/truncation marker, and no direct served-model/canonical-label contradiction. Existing Open Pool rules restrict `gemini-3.7-flash` to the two approved Vertex sources in `src/data_factory.py`; all other such rows were excluded. Publisher flags for routing, near-train-prompt, and train-prompt regeneration are preserved and checked.

| Exclusion reason | Rows |
|---|---:|
| `response_over_2000_characters` | 6 |

The source v15/v16 append logs also report their own pre-publication quality exclusions. These have already been applied upstream and are not added again to the local 231,036-row download accounting:

| Publisher append-log filter | V15 rows | V16 rows | Combined |
|---|---:|---:|---:|
| Accepted-source input rows | 3,104 | 6,937 | 10,041 |
| Already in previous pool/train | 0 | 1,490 | 1,490 |
| Unknown model label | 0 | 0 | 0 |
| Same text with conflicting labels | 0 | 0 | 0 |
| Duplicate within new source rows | 0 | 0 | 0 |
| Served-model label mismatch removed | 0 | 0 | 0 |
| Outside 36–275 word range | 11 | 76 | 87 |
| Newline/Markdown formatting | 527 | 0 | 527 |
| Raw API response | 0 | 0 | 0 |
| Refusal | 0 | 0 | 0 |
| Chat preamble/CJK | 7 | 170 | 177 |
| Cut off mid-sentence | 2 | 180 | 182 |
| Disallowed em-dash typography | 124 | 121 | 245 |
| Too similar to train passage | 81 | 50 | 131 |
| Near train prompt | 1 | 5 | 6 |
| Added to main pool | 2,351 | 4,845 | 7,196 |


### Accepted and excluded rows by label

| Label | Accepted | Excluded from snapshot |
|---|---:|---:|
| `claude-fable-5` | 0 | 12,773 |
| `claude-opus-4-6` | 302 | 10,866 |
| `claude-opus-4-8` | 282 | 11,876 |
| `claude-opus-5` | 463 | 9,417 |
| `claude-sonnet-5` | 532 | 12,716 |
| `gemini-3.1-pro-preview` | 335 | 10,178 |
| `gemini-3.6-flash` | 373 | 18,074 |
| `gemini-3.7-flash` | 0 | 3,541 |
| `gemini-3.8-flash` | 350 | 23,192 |
| `glm-5.3` | 559 | 11,891 |
| `glm-5.3-flash` | 531 | 12,214 |
| `gpt-5.5` | 769 | 13,546 |
| `gpt-5.6-luna` | 1,153 | 17,741 |
| `gpt-5.6-sol` | 129 | 14,591 |
| `gpt-5.6-terra` | 419 | 15,300 |
| `gpt-6-astra` | 208 | 11,633 |
| `kimi-k3` | 785 | 9,731 |
| `qwen3.8-max` | 0 | 4,566 |

### Accepted and excluded rows by source

| Source dataset | Accepted | Excluded from snapshot | Quality exclusions |
|---|---:|---:|---:|
| `abdullah1afify/ntl-generations` | 0 | 14,314 | 0 |
| `abhinavsaipodugu/ntl-proxy-gpt-gemini-passages` | 0 | 3,914 | 0 |
| `adurey/ntl-proxy-gen-20261031` | 0 | 4,742 | 0 |
| `aitahmednasser/ntl-kaggle-ai-gen` | 593 | 0 | 0 |
| `alessandrolair/ntl-kaggle-benchmarks-generated-passages` | 0 | 3,701 | 0 |
| `alperen5252525/ntl-public-passage-pool-v1` | 0 | 594 | 0 |
| `alperen5252525/ntl-vertex-gemini-proxy-gpt-generations` | 0 | 3,573 | 0 |
| `aniruduh9/ntl-hard-supplement` | 0 | 495 | 0 |
| `arxitekton/ntl-claude-gen-v1` | 0 | 12,463 | 0 |
| `arxitekton/ntl-claude-gen-v2` | 0 | 3,313 | 0 |
| `asiralabi/ntl-src` | 0 | 715 | 0 |
| `aswathvanan/ntl-proxy-passages` | 0 | 4,254 | 0 |
| `bingli283/ntl-gen-pilot-v1` | 0 | 1,477 | 0 |
| `carlosoponce/ntl-own-proxy-passages` | 0 | 973 | 0 |
| `chosunghwan/ntl-gemini-new-prompts` | 0 | 1,440 | 0 |
| `danielandresmiranda/ntl-openrouter-glm-kimi` | 0 | 1,482 | 0 |
| `darragh1walsh/ntl-kaggle-ai-gen-run-1` | 0 | 7,589 | 0 |
| `davidpiln/ntl-vertex-gemini-gen` | 0 | 4,187 | 0 |
| `dedruck/ntl-proxy-gen` | 0 | 2,560 | 0 |
| `dhananjayk2609/ntl-panel-gap-passages-kaggle-benchmarks` | 1,937 | 5 | 5 |
| `dominicjamil/ntl-pools` | 0 | 10 | 0 |
| `dophuchung/ntl-proxy-generations` | 0 | 963 | 0 |
| `drjcfmd/ntl-generated-v2` | 0 | 20,400 | 0 |
| `earthmaker/ntl-own-prompt-generations` | 0 | 698 | 0 |
| `elliotthankinson/ntl-exact-gens-v3` | 0 | 13,359 | 0 |
| `elliotthankinson/ntl-tells-v4` | 4,392 | 1 | 1 |
| `haileycheng/name-that-llm-same-prompt-passages` | 0 | 899 | 0 |
| `hansamaqht/name-that-llm-glm-extra` | 0 | 434 | 0 |
| `harshitkmr/ntl-google-gen-v1` | 0 | 17,798 | 0 |
| `illiadolenko/ntl-generated-corpus-v1` | 0 | 882 | 0 |
| `jadkutbi/ntl-own-proxy-generations` | 0 | 874 | 0 |
| `kaplinskiy/ntl-gpt-proxy-passages` | 0 | 887 | 0 |
| `kenandemirci/ntl-glm-gemini-generations` | 0 | 2,279 | 0 |
| `keshavatearth/ntl-gpt-new-prompts-v1` | 0 | 496 | 0 |
| `kilupy/ntl-18-model-corpus` | 0 | 3,359 | 0 |
| `kilupy/ntl-claude-family-passages` | 0 | 14,148 | 0 |
| `lucasboesen/ntl-free-proxy-generations` | 0 | 3,674 | 0 |
| `lxdlxdlxd/ntl-synth-generations-v1` | 0 | 4,348 | 0 |
| `mathleteutd/ntl-12-model-proxy-passages` | 0 | 2,063 | 0 |
| `michaelsemenoff/ntl-own-extra-v1` | 0 | 2,532 | 0 |
| `michaelsemenoff/ntl-pool-v3` | 0 | 2,800 | 0 |
| `mrralphie/ntl-pool-b2m` | 0 | 30 | 0 |
| `nimithashreea/ntl-proxy-candidate-generations-20261110` | 218 | 0 | 0 |
| `nuridudak/ntl-130k-clean-passages` | 0 | 714 | 0 |
| `pardheev/ntl-proxy-extra-passages` | 0 | 2,819 | 0 |
| `parkjinsu1/ntl-kb-proxy-panels` | 0 | 2,908 | 0 |
| `philipvallen/ntl-pools` | 0 | 1,628 | 0 |
| `prvsiyan/ntl-own-prompt-generations-v1` | 0 | 703 | 0 |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | 0 | 266 | 0 |
| `rainycloudss/ntl-free-proxy-fresh-prompts` | 0 | 513 | 0 |
| `seyitkaangunes/ntl-extra-api-passages` | 0 | 2,314 | 0 |
| `simonrba/ntl-pool-extra` | 0 | 737 | 0 |
| `swaroopmeher/ntl-public-data-sub1020` | 0 | 7 | 0 |
| `tadeuhayashida/ntl-claude-fable-5-generations` | 0 | 1,609 | 0 |
| `tamerlanomralinov/ntl-matched-bench-gen-r1` | 50 | 0 | 0 |
| `toanplam/ntl-fresh-generation-pilot-nov08` | 0 | 103 | 0 |
| `trnhip/ntl-api-gen` | 0 | 16,818 | 0 |
| `trnhip/ntl-proxy-gen` | 0 | 1,456 | 0 |
| `vbandaru/ntl-extra-passages-w05` | 0 | 4,359 | 0 |
| `vidhikishorwaghela/ntl-own-proxy-generations` | 0 | 2,018 | 0 |
| `willrice/name-that-llm-openrouter-harvest` | 0 | 12,083 | 0 |
| `yutoab/ntl-free-proxy-thin-labels` | 0 | 2,119 | 0 |
| `zbigniewkirchner/ntl-extra-gpt-passages` | 0 | 1,949 | 0 |
| `zeminglii/ntl-own-exact-route-generations` | 0 | 2,000 | 0 |
| `zitaoxu2002/ntl-proxy-passages-pre-ruling` | 0 | 1,026 | 0 |
| `ziyuzhou1207/ntl-zz-proxy-generations` | 0 | 972 | 0 |

### Exclusion reasons

| Reason | Rows |
|---|---:|
| `already_present_in_v2.16` | 223,840 |
| `response_over_2000_characters` | 6 |

Every excluded response is recorded with its label, source, original row ID and reason in the exclusion ledger [`v2.16_openpool16_increment_exclusions.csv`](../data/processed/v2.16_openpool16_increment_exclusions.csv).

## V15/V16 changelog and priority labels

The v16 changelog reports **4,845** new rows over v15: 4,393 from `elliotthankinson/ntl-tells-v4` and 452 new panel-gap passages from `dhananjayk2609/ntl-panel-gap-passages-kaggle-benchmarks`. Because V2.16 has the earlier pool but not all Open Pool v15 appends, its unseen rows total 7,196: 2,351 from the v15 append log plus 4,845 from v16. Six rows exceed the 2,000-character cap, leaving 7,190 accepted additions. The source table above includes rows from both append cohorts. The exact cohort assignment of the six length exclusions is not present in the published row schema, so it is not inferred. V16 reports 785 `kimi-k3`, 559 `glm-5.3`, 531 `glm-5.3-flash`, plus GPT 5.5/5.6 coverage. The Tells hard labels use paid/local routes and a discriminator gate; route, served-model, reasoning, match-weight, source and flags are retained.

| Label | Accepted additions in this V2.16 increment | Accepted from Tells v4 |
|---|---:|---:|
| `kimi-k3` | 785 | 785 |
| `glm-5.3` | 559 | 559 |
| `glm-5.3-flash` | 531 | 531 |
| `gpt-5.5` | 769 | 678 |
| `gpt-5.6-luna` | 1,153 | 1,023 |
| `gpt-5.6-sol` | 129 | 41 |
| `gpt-5.6-terra` | 419 | 337 |
| `gpt-6-astra` | 208 | 104 |

## Output and verification

- Increment: [`data/processed/dataset_v2.16_openpool16_increment.parquet`](../data/processed/dataset_v2.16_openpool16_increment.parquet) (**7,190 rows**).
- Output Parquet SHA-256: `a704c1da7bbf46a8422756ba357f461a74e2200fed7eae49ed0cca2f73984f7a`.
- Canonical content SHA-256: `b794a2e1d70f813cdb1640367f55d880aadd7b65048023e97833545ab1a96783`.
- Exclusion ledger SHA-256: `9a78493464bd075e9f85e62e9cc4a39cc061ae66e433090a33d054b59149f59f`.
- All structural, class, ID, normalized-response, V0 prompt-group, source-license and length checks passed.
- Existing Open Pool language detection (`langdetect`, seed 2026): `{'en': 7190}`; non-English/undetermined rows: **0**.
- Visible response scan counts: `{'thinking_tags': 0, 'explicit_chain_of_thought_markers': 0, 'reasoning_self_talk': 0, 'assistant_preamble': 0, 'serialized_chat_or_api': 0, 'fenced_code': 0, 'code_like_standalone_line': 0, 'stack_trace': 0, 'markdown_heading_or_list': 0}`. The checks cover thinking tags/markers, self-referential reasoning text, assistant preambles, serialized chat/API output, fenced or standalone code, stack traces, and Markdown headings/lists.
- `gen_reasoning` is retained source metadata describing the generation setting, not a chain-of-thought text field; accepted rows have setting counts recorded in the profile.
- Normalization: Unicode NFKC, case-folding, collapsed whitespace; punctuation preserved.
- Text line breaks are collapsed to single spaces, preserving original punctuation.
- `prompt_id` uses `prompt_canon` where supplied and falls back to source `prompt_group` for v16 rows whose source has no canonical prompt hash; original `prompt_group` remains intact.
- The official V0 parquet does not contain prompt text. Therefore prompt-group ID isolation is verified, but semantic prompt-text overlap cannot be independently recomputed from V0.
- Rebuild with `python scripts/build_openpool_v16_increment.py`. A second build to a temporary path reproduces the same canonical content hash.
