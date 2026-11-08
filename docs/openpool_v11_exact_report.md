# Open Pool V11 — Raw Dataset Summary

**Scope:** Direct read-only analysis of `data/raw/open_pool/ntl_open_pool.parquet`, before NLM filtering, balancing, or sampling. Counts and rates below are computed from that file; no dataset or project artifacts were changed.

## Overview

- **190,588 rows and 190,588 unique response texts** (exact and NFKC/case/whitespace normalized); IDs are also unique.
- **18 models**, **6 providers** (`anthropic`, `google`, `moonshot`, `openai`, `qwen`, `zai`), **55 source datasets**, **73 nonmissing source files**, **44,041 `prompt_group` values**.
- `prompt_canon` has **31,355 distinct nonmissing values**; **1,925 rows** lack it.
- No missing model labels; no exact or normalized duplicate responses; no repeated normalized response text with conflicting model labels.
- All rows have `source_dataset` and `source_license`; **5,708 (3.00%)** lack `source_file`. **157,495 (82.63%)** lack `origin_hint`.
- Fields include model/provider and source identifiers, license, generation-origin hint, prompt and panel metadata, source-reuse count, similarity scores, style score, four quality/routing flags, and `match_weight`.

## Distribution by model

“Unique prompts” here means distinct `prompt_group` values. “Largest source” is the largest `source_dataset` share within that model. Duplicate rows count repeated normalized response text within the model.

| Provider | Model | Rows | % of dataset | Unique prompts | Source datasets | Largest source (%) | Duplicate rows | Quality flags |
|---|---|---:|---:|---:|---:|---:|---:|---|
| google | gemini-3.8-flash | 21,550 | 11.31% | 16,739 | 45 | 34.2% | 0 | KB proxy 5,945 |
| google | gemini-3.6-flash | 16,942 | 8.89% | 14,935 | 40 | 43.0% | 0 | KB proxy 4,687 |
| openai | gpt-5.6-luna | 15,869 | 8.33% | 13,172 | 42 | 11.3% | 0 | KB proxy 8,670 |
| openai | gpt-5.6-terra | 13,597 | 7.13% | 11,135 | 43 | 13.7% | 0 | KB proxy 6,858 |
| openai | gpt-5.6-sol | 12,861 | 6.75% | 10,632 | 42 | 15.5% | 0 | KB proxy 6,324 |
| openai | gpt-5.5 | 11,755 | 6.17% | 9,970 | 40 | 16.2% | 0 | KB proxy 5,676 |
| zai | glm-5.3-flash | 11,657 | 6.12% | 9,303 | 14 | 17.8% | 0 | KB proxy 2,390 |
| zai | glm-5.3 | 11,347 | 5.95% | 8,918 | 14 | 19.4% | 0 | KB proxy 2,488 |
| openai | gpt-6-astra | 9,947 | 5.22% | 8,485 | 39 | 19.3% | 0 | KB proxy 4,230 |
| anthropic | claude-fable-5 | 9,594 | 5.03% | 7,780 | 15 | 17.7% | 0 | KB proxy 3,838 |
| moonshot | kimi-k3 | 9,426 | 4.95% | 5,128 | 11 | 28.8% | 0 | KB proxy 3,940 |
| google | gemini-3.1-pro-preview | 9,364 | 4.91% | 7,630 | 33 | 33.6% | 0 | KB proxy 3,483 |
| anthropic | claude-sonnet-5 | 8,536 | 4.48% | 7,206 | 30 | 15.1% | 0 | KB proxy 4,404 |
| anthropic | claude-opus-4-8 | 7,507 | 3.94% | 6,316 | 29 | 17.4% | 0 | KB proxy 3,618 |
| anthropic | claude-opus-4-6 | 6,799 | 3.57% | 5,626 | 27 | 18.0% | 0 | KB proxy 3,466 |
| anthropic | claude-opus-5 | 6,288 | 3.30% | 5,279 | 25 | 21.0% | 0 | KB proxy 3,254 |
| qwen | qwen3.8-max | 4,271 | 2.24% | 3,961 | 10 | 28.3% | 0 | KB proxy 1,271 |
| google | gemini-3.7-flash | 3,278 | 1.72% | 1,274 | 4 | 61.2% | 0 | KB proxy 704 |
| **Total** | — | **190,588** | **100.00%** | **44,041** | **55** | **10.7%** | **0** | **KB proxy 75,246** |

The other three Boolean flags—Gemini 3.7 routing risk, near-train prompt, and train-prompt regeneration—are false on every row in this main parquet.

## Provenance by model

Each source entry is `source_dataset: rows (percent of model)`. Percentages may not sum to 100 because only the top five sources are shown.

| Model | Five largest source datasets |
|---|---|
| gemini-3.8-flash | harshitkmr/ntl-google-gen-v1: 7,371 (34.2%); davidpiln/ntl-vertex-gemini-gen: 2,180 (10.1%); trnhip/ntl-api-gen: 1,572 (7.3%); willrice/name-that-llm-openrouter-harvest: 1,207 (5.6%); drjcfmd/ntl-generated-v2: 859 (4.0%) |
| gemini-3.6-flash | harshitkmr/ntl-google-gen-v1: 7,285 (43.0%); trnhip/ntl-api-gen: 1,812 (10.7%); lucasboesen/ntl-free-proxy-generations: 962 (5.7%); drjcfmd/ntl-generated-v2: 799 (4.7%); willrice/name-that-llm-openrouter-harvest: 783 (4.6%) |
| gpt-5.6-luna | abdullah1afify/ntl-generations: 1,800 (11.3%); trnhip/ntl-api-gen: 1,670 (10.5%); elliotthankinson/ntl-exact-gens-v3: 1,535 (9.7%); drjcfmd/ntl-generated-v2: 1,285 (8.1%); pardheev/ntl-proxy-extra-passages: 1,211 (7.6%) |
| gpt-5.6-terra | trnhip/ntl-api-gen: 1,861 (13.7%); elliotthankinson/ntl-exact-gens-v3: 1,425 (10.5%); drjcfmd/ntl-generated-v2: 1,316 (9.7%); abdullah1afify/ntl-generations: 1,062 (7.8%); arxitekton/ntl-claude-gen-v1: 1,013 (7.5%) |
| gpt-5.6-sol | trnhip/ntl-api-gen: 1,988 (15.5%); elliotthankinson/ntl-exact-gens-v3: 1,345 (10.5%); drjcfmd/ntl-generated-v2: 1,283 (10.0%); arxitekton/ntl-claude-gen-v1: 1,014 (7.9%); abdullah1afify/ntl-generations: 1,013 (7.9%) |
| gpt-5.5 | trnhip/ntl-api-gen: 1,907 (16.2%); drjcfmd/ntl-generated-v2: 1,308 (11.1%); abdullah1afify/ntl-generations: 1,004 (8.5%); willrice/name-that-llm-openrouter-harvest: 987 (8.4%); arxitekton/ntl-claude-gen-v1: 699 (5.9%) |
| glm-5.3-flash | elliotthankinson/ntl-exact-gens-v3: 2,077 (17.8%); trnhip/ntl-api-gen: 2,068 (17.7%); drjcfmd/ntl-generated-v2: 1,265 (10.9%); arxitekton/ntl-claude-gen-v1: 1,050 (9.0%); willrice/name-that-llm-openrouter-harvest: 1,010 (8.7%) |
| glm-5.3 | elliotthankinson/ntl-exact-gens-v3: 2,202 (19.4%); trnhip/ntl-api-gen: 2,018 (17.8%); drjcfmd/ntl-generated-v2: 1,241 (10.9%); arxitekton/ntl-claude-gen-v1: 1,011 (8.9%); willrice/name-that-llm-openrouter-harvest: 966 (8.5%) |
| gpt-6-astra | trnhip/ntl-api-gen: 1,922 (19.3%); drjcfmd/ntl-generated-v2: 1,285 (12.9%); willrice/name-that-llm-openrouter-harvest: 967 (9.7%); arxitekton/ntl-claude-gen-v1: 668 (6.7%); lucasboesen/ntl-free-proxy-generations: 627 (6.3%) |
| claude-fable-5 | elliotthankinson/ntl-exact-gens-v3: 1,702 (17.7%); tadeuhayashida/ntl-claude-fable-5-generations: 1,609 (16.8%); drjcfmd/ntl-generated-v2: 1,295 (13.5%); arxitekton/ntl-claude-gen-v1: 1,014 (10.6%); abdullah1afify/ntl-generations: 988 (10.3%) |
| kimi-k3 | elliotthankinson/ntl-exact-gens-v3: 2,717 (28.8%); willrice/name-that-llm-openrouter-harvest: 2,547 (27.0%); drjcfmd/ntl-generated-v2: 1,247 (13.2%); arxitekton/ntl-claude-gen-v2: 956 (10.1%); arxitekton/ntl-claude-gen-v1: 757 (8.0%) |
| gemini-3.1-pro-preview | harshitkmr/ntl-google-gen-v1: 3,142 (33.6%); drjcfmd/ntl-generated-v2: 869 (9.3%); lucasboesen/ntl-free-proxy-generations: 853 (9.1%); chosunghwan/ntl-gemini-new-prompts: 527 (5.6%); abhinavsaipodugu/ntl-proxy-gpt-gemini-passages: 487 (5.2%) |
| claude-sonnet-5 | drjcfmd/ntl-generated-v2: 1,289 (15.1%); arxitekton/ntl-claude-gen-v1: 973 (11.4%); abdullah1afify/ntl-generations: 971 (11.4%); yutoab/ntl-free-proxy-thin-labels: 591 (6.9%); seyitkaangunes/ntl-extra-api-passages: 465 (5.4%) |
| claude-opus-4-8 | drjcfmd/ntl-generated-v2: 1,305 (17.4%); abdullah1afify/ntl-generations: 989 (13.2%); arxitekton/ntl-claude-gen-v1: 980 (13.1%); yutoab/ntl-free-proxy-thin-labels: 522 (7.0%); seyitkaangunes/ntl-extra-api-passages: 471 (6.3%) |
| claude-opus-4-6 | drjcfmd/ntl-generated-v2: 1,223 (18.0%); abdullah1afify/ntl-generations: 950 (14.0%); arxitekton/ntl-claude-gen-v1: 685 (10.1%); yutoab/ntl-free-proxy-thin-labels: 556 (8.2%); seyitkaangunes/ntl-extra-api-passages: 464 (6.8%) |
| claude-opus-5 | drjcfmd/ntl-generated-v2: 1,322 (21.0%); abdullah1afify/ntl-generations: 988 (15.7%); seyitkaangunes/ntl-extra-api-passages: 470 (7.5%); arxitekton/ntl-claude-gen-v1: 362 (5.8%); yutoab/ntl-free-proxy-thin-labels: 353 (5.6%) |
| qwen3.8-max | drjcfmd/ntl-generated-v2: 1,209 (28.3%); arxitekton/ntl-claude-gen-v2: 956 (22.4%); willrice/name-that-llm-openrouter-harvest: 601 (14.1%); arxitekton/ntl-claude-gen-v1: 436 (10.2%); kenandemirci/ntl-glm-gemini-generations: 382 (8.9%) |
| gemini-3.7-flash | davidpiln/ntl-vertex-gemini-gen: 2,007 (61.2%); alperen5252525/ntl-vertex-gemini-proxy-gpt-generations: 704 (21.5%); lxdlxdlxd/ntl-synth-generations-v1: 301 (9.2%); prvsiyan/prvsiyans-extended-name-that-llm-dataset: 266 (8.1%) |

## Source files, licences, and provenance

The raw table has three licence values: **CC BY 4.0: 158,393 (83.12%)**, **CC-BY-SA-4.0: 28,154 (14.77%)**, and **CC0-1.0: 4,041 (2.12%)**. No licence values are missing.

For each model below, licence percentages are listed as `CC BY / CC-BY-SA / CC0`. “Files” counts distinct nonmissing `source_file`; “lead file” is its most frequent file. `origin_hint` shows missing rows.

| Model | Files; lead file (share) | Licence mix (%) | Missing origin_hint / source_file |
|---|---|---:|---:|
| gemini-3.8-flash | 57; google_gen_v1.csv (34.2%) | 90.1 / 8.5 / 1.3 | 19,708 / 397 |
| gemini-3.6-flash | 45; google_gen_v1.csv (43.0%) | 90.2 / 8.4 / 1.3 | 15,504 / 806 |
| gpt-5.6-luna | 52; passages.csv (29.8%) | 84.0 / 12.8 / 3.2 | 13,382 / 141 |
| gpt-5.6-terra | 51; passages.csv (33.4%) | 82.8 / 15.2 / 2.0 | 11,176 / 131 |
| gpt-5.6-sol | 50; passages.csv (33.7%) | 81.9 / 15.9 / 2.2 | 10,898 / 130 |
| gpt-5.5 | 42; passages.csv (32.5%) | 77.9 / 19.2 / 2.9 | 9,783 / 397 |
| glm-5.3-flash | 16; passages.csv (26.9%) | 87.2 / 10.9 / 1.8 | 9,398 / 214 |
| glm-5.3 | 16; passages.csv (26.8%) | 87.0 / 11.1 / 1.9 | 9,136 / 220 |
| gpt-6-astra | 41; passages.csv (35.4%) | 76.2 / 20.7 / 3.1 | 8,067 / 396 |
| claude-fable-5 | 17; passages.csv (20.5%) | 86.2 / 13.6 / 0.2 | 7,399 / 0 |
| kimi-k3 | 16; neuralloot_gen_v3_join (21.5%) | 86.7 / 13.3 / 0 | 7,667 / 0 |
| gemini-3.1-pro-preview | 35; google_gen_v1.csv (33.6%) | 82.0 / 17.0 / 1.0 | 7,931 / 854 |
| claude-sonnet-5 | 33; passages.csv (25.7%) | 77.3 / 18.5 / 4.2 | 6,294 / 591 |
| claude-opus-4-8 | 31; passages.csv (22.2%) | 74.3 / 21.2 / 4.5 | 5,523 / 522 |
| claude-opus-4-6 | 31; passages.csv (20.5%) | 71.3 / 24.4 / 4.2 | 5,069 / 556 |
| claude-opus-5 | 29; passages.csv (16.0%) | 68.6 / 26.7 / 4.7 | 4,539 / 353 |
| qwen3.8-max | 11; passages.csv (33.0%) | 71.6 / 28.4 / 0 | 2,743 / 0 |
| gemini-3.7-flash | 5; passages.csv (58.1%) | 100 / 0 / 0 | 3,278 / 0 |

## Response lengths and quality indicators

Lengths are P10 / median / P90. Word counts are measured from response text with a consistent regex; character counts include whitespace. The parquet’s `n_words` field is close but not identical.

| Model | Characters | Words |
|---|---:|---:|
| gemini-3.8-flash | 325 / 623 / 1,312 | 48 / 91 / 192 |
| gemini-3.6-flash | 325 / 613 / 1,255 | 48 / 90 / 188 |
| gpt-5.6-luna | 336 / 668 / 1,324 | 51 / 104 / 208 |
| gpt-5.6-terra | 327 / 643 / 1,304 | 50 / 102 / 207 |
| gpt-5.6-sol | 319 / 664 / 1,337 | 48 / 104 / 211 |
| gpt-5.5 | 312 / 633 / 1,251 | 49 / 103 / 204 |
| glm-5.3-flash | 328 / 662 / 1,296 | 51 / 106 / 208 |
| glm-5.3 | 313 / 622 / 1,298 | 49 / 98 / 208 |
| gpt-6-astra | 325 / 674 / 1,368 | 48 / 103 / 211 |
| claude-fable-5 | 324 / 650 / 1,292 | 49 / 102 / 210 |
| kimi-k3 | 321 / 610 / 1,234 | 49 / 96 / 198 |
| gemini-3.1-pro-preview | 323 / 616 / 1,252 | 48 / 93 / 191 |
| claude-sonnet-5 | 355 / 734 / 1,393 | 53 / 116 / 223 |
| claude-opus-4-8 | 339 / 687 / 1,372 | 52 / 109 / 221 |
| claude-opus-4-6 | 354 / 703 / 1,337 | 53 / 110 / 216 |
| claude-opus-5 | 318 / 669 / 1,319 | 51 / 110 / 218 |
| qwen3.8-max | 319 / 632 / 1,275 | 48 / 99 / 203 |
| gemini-3.7-flash | 333 / 641 / 1,365 | 49 / 93 / 197 |

Global `n_sources` is P10 1, median 3, P90 11, max 19. `max_cos_train` is median 0.090 / P90 0.143 / max 0.300; `max_sim_train_prompt` median 0.084 / P90 0.154 / max 0.399. `style_score` median 0.211 and P90 0.542. `match_weight` median 0.821 / P90 1.934 / P99 4.315 / max 50. These are stored metadata measures; their provenance and methods are not independently established by this parquet.

No `language` field exists, so language distribution is **N/A**. The three false near-train/routing/regeneration flags do not prove absence of all semantic duplication or routing problems; they only report the values stored in this file.

## Conclusions

- **Strongest coverage/diversity:** Gemini 3.8 Flash has the most rows (21,550), 45 source datasets, and a 34.2% leading-source share. The OpenAI models have broad source coverage (39–43 datasets each) and relatively dispersed top-source contributions.
- **Limited coverage / provenance concern:** Gemini 3.7 Flash is smallest (3,278 rows), has only four source datasets, and 61.2% comes from one. Qwen3.8-max is next smallest (4,271 rows, 10 source datasets). `origin_hint` is absent for most rows across the corpus; this limits row-level generation traceability. The Gemini 3.7 routing-risk flag is false throughout this main parquet, which by itself does not verify its generation route.
- **Imbalance:** largest class has 6.57 times the rows of the smallest. Gemini 3.6 and 3.8 each draw over 40% from the same source dataset; this is a potential source shortcut.
- **Training implication:** raw examples are diverse in source count, but model class and source are not independent by construction. Random row splits could let source-specific collection or formatting cues carry over. For generalization to official Kaggle examples, compare per-model results and use prompt- and source-aware checks; the dataset’s own metadata cannot establish the quality of every label or representativeness of the Kaggle test distribution.
