# External candidate dataset quality assessment

Read-only analysis of the primary exports identified in `data/raw/external_candidates/README.md`. No raw data was changed.

## 1. Summary

| Dataset | Rows | Valid prompt + response | Attributed model | Empty response | Exact duplicate extra rows | Normalized duplicate extra rows | Unique normalized responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | 86,006 | 86,006 | 86,006 | 0 | 8 | 8 | 85,998 |
| GPT-5.6 Sol/Luna | 15,353 | 7,040 | 15,353 | 8,313 | 10 | 10 | 7,030 |

GLM model attribution is inherited from the dataset repository (not per-row). GPT attribution is parsed from row-level `teacher_model` (Sol) or `model` (Luna); row JSON parse failures: **0**. All five primary Parquet files were read successfully. Raw exact overlap compares literal strings; normalized overlap uses the normalization specified below.

## 2. Per-model statistics

| Model | Rows | Valid prompt/response | Empty prompts | Empty responses | Unique prompts | Prompts with >1 response | Same-model duplicate groups | Cross-model identical text groups | Char P10 / median / P90 / max | Word P10 / median / P90 / max | Source token metadata P10 / median / P90 / max | Language metadata |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| glm-5.3-flash | 86,006 | 86,006 | 0 | 0 | 40,988 | 38,604 | 6 | 0 | 2146 / 6412 / 20567 / 841540 | 353 / 941 / 3407 / 38353 | 541 / 1600 / 6144 / 49114 | `{"en": 84297, "zh": 1709}` |
| gpt-5.6-luna | 886 | 886 | 0 | 0 | 886 | 0 | 0 | 0 | 1220 / 2068 / 4104 / 10390 | 198 / 338 / 628 / 1443 | N/A | `{"NaN": 886}` |
| gpt-5.6-sol | 14,467 | 6,154 | 0 | 8,313 | 1,426 | 1,424 | 10 | 0 | 0 / 0 / 749 / 21682 | 0 / 0 / 95 / 3196 | N/A | `{"code-review": 3551, "python": 2920, "go": 1757, "ts": 1576, "java": 766, "csharp": 607, "powershell": 605, "rust": 520, "multi": 498, "bash": 349, "c": 242, "py": 226, "ruby": 197, "cpp": 179, "seed-authoring": 177, "zsh": 165, "js": 70, "asm": 39, "typescript": 22, "hcl": 1}` |

### Source and provenance

- GLM primary file row counts: `{"glm53_harvest.parquet": 46611, "glm53_harvest_r2.parquet": 18858, "glm53_harvest_r3.parquet": 15337, "glm53_harvest_r4.parquet": 5200}`. Repository card license: `mit`. Explicit register counts are measured from row `register`; GLM row label is dataset-level. Round 3/4 provenance joins by `id`; matched fields nonmissing counts: `{"temp": 25016, "round": 25016, "source": 25016, "mode": 25016, "turn": 2710, "conv_id": 2710, "has_system": 25016, "tokens": 25016, "finish": 13873, "cap_hit": 25016, "legacy_cap": 25016, "truncated": 25016, "think_tag_in_off": 25016, "meta_json": 25016}`.
- Auxiliary GLM provenance had 9,480 repeated IDs; the first matching row was joined deterministically.
- GLM register distribution: `{"code": 17985, "prose": 15923, "structured": 12161, "reasoning": 10632, "chat": 10141, "agentic": 5640, "multiturn": 4846, "music": 4400, "zh": 1709, "agentic_coding": 810, "agentic_tooling": 565, "long_context": 543, "code_edit": 476, "deep_reasoning": 96, "operator": 79}`; provenance round: `{"NaN": 60990, "3.0": 19816, "4.0": 5200}`; source: `{"NaN": 60990, "r3-bank-seed43": 19816, "open-perfectblend-glm5.2-regen": 2710, "github-code-clean": 1019, "swe-bench-verified": 810, "synthetic": 565, "hendrycks-math": 96}`; finish indicators: `{"NaN": 72133, "stop": 12974, "length": 899}`; `truncated`: `{"NaN": 60990, "false": 24103, "true": 913}`; cap-hit: `{"NaN": 60990, "false": 24208, "true": 808}`.
- GLM thinking flag: `{"false": 63990, "true": 22016}`; turns: `{"1": 81552, "3": 1970, "5": 942, "7": 387, "9": 328, "11": 283, "13": 222, "15": 168, "17": 103, "19": 51}`. The separate provenance file covers R3/R4 IDs only; absent fields are unavailable, not negative assertions.
- GPT repository card license: `cc-by-4.0`. Row-level model counts: `{"gpt-5.6-sol": 14467, "gpt-5.6-luna": 886}`; source dataset counts: `{"greghavens/gpt-5.6-sol-coding-and-debugging-traces": 14467, "empero-ai/gpt-5.6-luna-sft-900x": 886}`; row `seen_count`: `{"1": 15353}`. Its row JSON has messages, programming-language tags, split/task/domain/category, reasoning effort, assistant step counts, and trace metadata. The card declares English; row `lang` is a programming-language/task tag, not natural-language identification: `{"gpt-5.6-luna": {"NaN": 886}, "gpt-5.6-sol": {"code-review": 3551, "python": 2920, "go": 1757, "ts": 1576, "java": 766, "csharp": 607, "powershell": 605, "rust": 520, "multi": 498, "bash": 349, "c": 242, "py": 226, "ruby": 197, "cpp": 179, "seed-authoring": 177, "zsh": 165, "js": 70, "asm": 39, "typescript": 22, "hcl": 1}}`.
- Model identifier fields are Sol `teacher_model` and Luna `model`; `openai/` is removed only for a normalized report label, while the raw identifier is retained in JSON metrics. GPT repository card license: `cc-by-4.0`. Row-level model counts: `{"gpt-5.6-sol": 14467, "gpt-5.6-luna": 886}`; source dataset counts: `{"greghavens/gpt-5.6-sol-coding-and-debugging-traces": 14467, "empero-ai/gpt-5.6-luna-sft-900x": 886}`; row `seen_count`: `{"1": 15353}`. The card declares English; row `lang` is a programming-language/task tag, not natural-language identification: `{"gpt-5.6-luna": {"NaN": 886}, "gpt-5.6-sol": {"code-review": 3551, "python": 2920, "go": 1757, "ts": 1576, "java": 766, "csharp": 607, "powershell": 605, "rust": 520, "multi": 498, "bash": 349, "c": 242, "py": 226, "ruby": 197, "cpp": 179, "seed-authoring": 177, "zsh": 165, "js": 70, "asm": 39, "typescript": 22, "hcl": 1}}`.
- GPT source/model identifier disagreement against source naming: 0 rows; all row labels are present. This checks recorded fields only.
- Model attribution completeness measures field presence/parseability only. It does not verify that the named model generated the text.

### Primary-file schemas and missingness

Missingness below is measured from raw primary Parquet columns (null values); derived empty prompt/response counts are shown in the per-model table.

| Primary file | Rows | Column: type (missing %) |
|---|---:|---|
| glm53_harvest.parquet | 46,611 | `id`: string (0.00%); `register`: string (0.00%); `lang`: string (0.00%); `thinking`: bool (0.00%); `turns`: int32 (0.00%); `temp`: float (0.00%); `prompt`: string (0.00%); `response`: string (0.00%); `tokens`: int32 (0.00%); `messages_json`: string (100.00%) |
| glm53_harvest_r2.parquet | 18,858 | `id`: string (0.00%); `register`: string (0.00%); `lang`: string (0.00%); `thinking`: bool (0.00%); `turns`: int32 (0.00%); `temp`: float (0.00%); `prompt`: string (0.00%); `response`: string (0.00%); `tokens`: int32 (0.00%); `messages_json`: string (99.52%) |
| glm53_harvest_r3.parquet | 15,337 | `id`: string (0.00%); `register`: string (0.00%); `lang`: string (0.00%); `thinking`: bool (0.00%); `turns`: int32 (0.00%); `temp`: float (0.00%); `prompt`: string (0.00%); `response`: string (0.00%); `tokens`: int32 (0.00%); `messages_json`: string (86.66%) |
| glm53_harvest_r4.parquet | 5,200 | `id`: string (0.00%); `register`: string (0.00%); `lang`: string (0.00%); `thinking`: bool (0.00%); `turns`: int32 (0.00%); `temp`: float (0.00%); `prompt`: string (0.00%); `response`: string (0.00%); `tokens`: int32 (0.00%); `messages_json`: string (41.38%) |
| train.parquet | 15,353 | `row_hash`: string (0.00%); `first_source_dataset`: string (0.00%); `first_source_config`: string (0.00%); `first_source_split`: string (0.00%); `first_source_row_index`: int64 (0.00%); `seen_count`: int64 (0.00%); `row_json`: string (0.00%) |

### Attribution/model counts

- GLM-5.3-Flash: {"glm-5.3-flash": 86006}; attribution is dataset-level. GPT-5.6 rows: {"gpt-5.6-sol": 14467, "gpt-5.6-luna": 886}; target names are extracted from row JSON `teacher_model` (Sol) or `model` (Luna).

## 3. Redundancy, diversity and response characteristics

- Near duplicates: **77 verified pairs** in **26 connected clusters**, using 64-permutation word-trigram MinHash LSH candidates, verified by exact character 5-gram Jaccard; candidate length ratio >= 0.75 at Jaccard **0.90**. Indexed unique responses: 202,992. Pair scope counts: `{"within:glm53_flash": 5, "within:gpt56_sol_luna": 72}`. Cross-dataset verified pairs: 0.
- LSH candidate retrieval is approximate; reported pairs are verified, but zero reported pairs would not prove zero near duplicates.
- `words` counts Unicode word-like sequences. GLM's source-provided `tokens` field is retained separately; `approx_tokens` is only a word-count proxy, not tokenizer output. GPT has no row-level token-count field in the primary schema. Formatting flags are deterministic regex heuristics. GPT Sol's 8,313 rows with empty final assistant text all have a structured `tool_calls` payload; these are empty text outputs, not JSON parse failures. Length quantiles include them as zero-length text.

| Dataset | Markdown | Code fences | URLs | Lists | Structured markers |
|---|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | 76,351 | 34,302 | 6,125 | 54,165 | 20,644 |
| GPT-5.6 Sol/Luna | 4,883 | 295 | 187 | 1,116 | 2,902 |

Vocabulary counts below are distinct normalized whitespace tokens across each model's extracted text; mean word length uses Unicode word-like sequences.

| Model | Distinct word forms | Mean word length (characters) |
|---|---:|---:|
| glm-5.3-flash | 5,213,674 | 4.48 |
| gpt-5.6-luna | 38,431 | 4.78 |
| gpt-5.6-sol | 96,620 | 2.36 |
- GLM `thinking` is explicit metadata (`{"false": 63990, "true": 22016}`); `turns` records the source turn count. GPT metadata: `reasoning_effort` `{"xhigh": 14467, "NaN": 886}`, `continuation` `{"false": 14182, "null": 886, "true": 285}`, and `original_n_messages` among nonmissing rows has P10/median/P90/max `{"p10": 15.0, "median": 22.0, "p90": 45.0, "max": 126.0}`. Rows are trace chunks, so a single-turn/multiturn classification is not a uniform conversation-level measure. GPT final assistant text contains `<think>` tags in 886 rows (heuristic); reasoning effort is explicit metadata.
- The script records repeated normalized 120-character prefixes and suffixes. Top repeats are shown below; common prefixes can reflect shared source/task templates rather than copied full responses.

| Dataset / model | Top repeated response prefix (count) | Top repeated response suffix (count) |
|---|---|---|
| GLM / glm-5.3-flash | `[["<think>the user asks me to find the flaw in the argument: \"all fast code is short; this code is short, so it is fast.\" t", 194], ["<think>the user wants me to plan a task: check tomorrow's weather in three cities and message the team which one suits a", 71], ["<think>the user wants me to find the flaw in the argument: \"all fast code is short; this code is short, so it is fast.\" ", 65]]` | `[["<fatal-error>lp failure</fatal-error>", 4], [" three cases, design consumers to be **idempotent** — with at-least-once delivery, duplicates are a *when*, not an *if*.", 2], ["j = 0; j < colsb; j++) { for (let k = 0; k < colsa; k++) { result[i][j] += a[i][k] * b[k][j]; } } } return result; } ```", 2]]` |
| GPT Sol/Luna / gpt-5.6-luna | `[["<think> this is a two-sided one-sample t test because the population standard deviation is unknown and the sample standa", 2], ["<think> the rewrite should preserve the two essential facts: the hybrid model begins immediately, and employees are expe", 1], ["<think> the notes provide four shipments and all required fields. “still moving” indicates in_transit, while “held up at", 1]]` | `[["rrangement. send any questions to your manager or hr so we can support you through the transition. thank you, management", 1], ["\",\"origin_city\":\"austin\",\"destination_city\":\"dallas\",\"weight_kg\":0.5,\"status\":\"delivered\",\"delivery_date\":\"2025-03-06\"}]", 1], ["ry\":\"the enterprise customer was charged twice on the monthly invoice and requests a refund for the duplicate charge.\"}]", 1]]` |
| GPT Sol/Luna / gpt-5.6-sol | `[["{\"verdict\":\"needs_human\",\"scope_creep\":{\"found\":false,\"evidence\":[]},\"missed_requirements\":{\"found\":false,\"evidence\":[]}", 540], ["{ \"verdict\": \"needs_human\", \"scope_creep\": { \"found\": false, \"evidence\": [] }, \"missed_requirements\": { \"found\": false, ", 69], ["{\"verdict\":\"accept\",\"scope_creep\":{\"found\":false,\"evidence\":[\"`git status --short --untracked-files=all` reported only `", 30]]` | `[["\"evidence\":[]},\"non_working_code\":{\"found\":false,\"evidence\":[]},\"requirements\":[],\"summary\":\"verification in progress.\"}", 32], ["idence\":[]},\"non_working_code\":{\"found\":false,\"evidence\":[]},\"requirements\":[],\"summary\":\"verification is in progress.\"}", 7], ["tation paths, and the required race-enabled test command. i’ll return only the strict json verdict after verification.\"}", 4]]` |

## 4. Comparison with V1-all

Common response normalization: Unicode NFKC, casefold, and collapse all whitespace runs; punctuation and word order are preserved. Prompt overlap uses the same normalization against V1-all `prompt_canon` (or `prompt_id` fallback).

| Candidate source | Rows | Raw exact response overlap rows | Normalized overlap rows | Unique normalized responses | Unique not in V1-all | Prompt overlap rows | Near-pair counts (all datasets) |
|---|---:|---:|---:|---:|---:|---:|---|
| glm53_flash | 86,006 | 0 | 0 | 85,998 | 85,998 | 0 | see near-duplicate scope above |
| gpt56_sol_luna | 15,353 | 0 | 0 | 7,030 | 7,030 | 0 | see near-duplicate scope above |

### Overlap by attributed model

| Candidate model | Rows | Normalized response overlap rows | % of rows | Unique response strings not in V1-all |
|---|---:|---:|---:|---:|
| glm-5.3-flash | 86,006 | 0 | 0.00% | 85,998 |
| gpt-5.6-luna | 886 | 0 | 0.00% | 886 |
| gpt-5.6-sol | 14,467 | 0 | 0.00% | 6,144 |

### Length and formatting distribution versus V1-all

Formatting values are row counts and percentages; markers are heuristic regex detections.

| Dataset | Char median (P10–P90) | Word median (P10–P90) | Markdown % | Code % | URL % | List % | Structured % |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1-all | 645 (325–1302) | 102 (50–208) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| GLM-5.3-Flash | 6412 (2146–20567) | 941 (353–3407) | 88.8% | 39.9% | 7.1% | 63.0% | 24.0% |
| GPT-5.6 Sol/Luna | 0 (0–1276) | 0 (0–191) | 31.8% | 1.9% | 1.2% | 7.3% | 18.9% |

Differences are descriptive and do not establish source quality or model authorship.

## Conclusions

1. The primary files contain **86,006** GLM rows and **15,353** GPT Sol/Luna rows; the latter split is `{"gpt-5.6-sol": 14467, "gpt-5.6-luna": 886}`.
2. Dataset-level attribution applies to GLM; GPT row-level teacher-model attribution parsed for 15,353/15,353 rows.
3. Normalized response overlap with V1-all is 0 rows across the two sources; distinct responses outside V1-all are reported per source above.
4. The current estimate found 77 MinHash candidates verified at or above character 5-gram Jaccard 0.90; approximate candidate retrieval limits recall.
5. Provenance and metadata are uneven: GLM has structured generation fields, with separate finish/truncation metadata only for matched R3/R4 rows; GPT supplies row-level source/model/trace fields. These fields describe recorded metadata, not independent verification.

## Method and reproducibility

Run: `/home/zodrak/cuda13.x/bin/python scripts/analyze_external_candidates.py`. Fixed seed: 2026; MinHash permutations: 64; near-duplicate verification threshold: 0.90. The script reads the four GLM round files, one GPT train file, GLM provenance metadata and V1-all. Register/thinking/config exports are alternative or overlapping views and are not added as rows.
GPT prompt extraction uses the final user message; response text uses the content of the final assistant message. An empty assistant content remains an empty response string even when a structured tool-call payload exists. GLM prompt and response columns are read directly.

Missing or non-applicable measures are reported as N/A in source schemas or left unavailable when metadata is absent. Language counts reflect the source `lang` fields, not language identification performed by this analysis.
