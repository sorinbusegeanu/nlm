# Open Pool V14 comparison (download folder `open_pool_v12`)

## Release identity and method

The requested Kaggle dataset is currently **V14**, distributed as Kaggle dataset version **16**; Kaggle metadata dates that version to **2026-11-09**. The local folder name `open_pool_v12` is retained as requested, but the downloaded release is not V12. Kaggle’s listing describes V14 as 223,840 passages and documents 11,501 additions plus removal of 284 rows in the V14 update. The download was made 2026-11-10. Source and per-file sizes/SHA-256 hashes are recorded in [`data/raw/open_pool_v12/README.md`](../data/raw/open_pool_v12/README.md). [Kaggle dataset page](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) · [Kaggle dataset metadata](https://www.kaggle.com/api/v1/datasets/view/crtproduction/name-that-llm-open-pool)

Comparison uses the main V14, V11 and B2M parquet files, with separate accounting for the Gemini 3.7 sidecars. A response key is Unicode NFKC-normalized, case-folded, and whitespace-collapsed. Punctuation and wording are otherwise unchanged. These are exact normalized-text matches, not semantic or near-duplicate matches. The main files each contain one row per normalized response, with no empty response or within-file normalized duplicate; no conflicting labels were found within any main file.

## Dataset sizes and per-model changes

| Model | V11 rows | B2M rows | V14 rows | Same response V14/V11 | Added vs V11 | Removed from V11 | New vs V11 + B2M (+ V11 sidecar) |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude-fable-5 | 9,594 | 3,396 | 12,773 | 9,575 | 3,198 | 19 | 3,198 |
| claude-opus-4-6 | 6,799 | 2,379 | 10,866 | 6,799 | 4,067 | 0 | 4,067 |
| claude-opus-4-8 | 7,507 | 2,819 | 11,875 | 7,464 | 4,411 | 43 | 4,411 |
| claude-opus-5 | 6,288 | 1,927 | 9,417 | 6,288 | 3,129 | 0 | 3,129 |
| claude-sonnet-5 | 8,536 | 2,908 | 12,714 | 8,536 | 4,178 | 0 | 4,178 |
| gemini-3.1-pro-preview | 9,364 | 1,665 | 10,178 | 9,364 | 814 | 0 | 814 |
| gemini-3.6-flash | 16,942 | 3,195 | 18,074 | 16,821 | 1,253 | 121 | 1,250 |
| gemini-3.7-flash | 3,278 | 1,958 | 3,541 | 3,278 | 263 | 0 | 263 |
| gemini-3.8-flash | 21,550 | 4,026 | 23,190 | 21,550 | 1,640 | 0 | 1,640 |
| glm-5.3 | 11,347 | 2,978 | 11,891 | 11,347 | 544 | 0 | 544 |
| glm-5.3-flash | 11,657 | 3,407 | 12,214 | 11,657 | 557 | 0 | 557 |
| gpt-5.5 | 11,755 | 3,593 | 13,546 | 11,755 | 1,791 | 0 | 1,788 |
| gpt-5.6-luna | 15,869 | 5,066 | 17,740 | 15,768 | 1,972 | 101 | 1,970 |
| gpt-5.6-sol | 12,861 | 4,598 | 14,591 | 12,861 | 1,730 | 0 | 1,727 |
| gpt-5.6-terra | 13,597 | 4,702 | 15,300 | 13,597 | 1,703 | 0 | 1,703 |
| gpt-6-astra | 9,947 | 3,082 | 11,633 | 9,947 | 1,686 | 0 | 1,681 |
| kimi-k3 | 9,426 | 2,637 | 9,731 | 9,426 | 305 | 0 | 305 |
| qwen3.8-max | 4,271 | 2,517 | 4,566 | 4,271 | 295 | 0 | 295 |
| **Total** | **190,588** | **56,853** | **223,840** | **190,304** | **33,536** | **284** | **33,520** |

V14 has **18 models**, V11 has 18, and B2M has 18. By normalized response text, V14 versus V11 consists of 190,304 unchanged, 33,536 new, and 284 no longer present in the V14 main file. The 284 absent rows are identified in the V14 removal log as `served_model_not_the_label`; the dataset card attributes them to one source whose own model field disagreed with the assigned label. They comprise 121 `gemini-3.6-flash`, 101 `gpt-5.6-luna`, 43 `claude-opus-4-8`, and 19 `claude-fable-5`.

Against B2M alone, V14 shares **49,357** normalized responses, has **174,483** responses absent from B2M, and lacks **7,496** B2M responses. The 33,536 V14 additions relative to V11 include 16 responses already present in B2M, leaving **33,520** unique to V14 after deduplication against both existing main datasets and the V11 Gemini 3.7 sidecar. The last table column gives that per-model count. Across V14 and either prior main dataset, no shared response text had a different model label.

Row counts by model vary substantially: V14 ranges from **3,541 Gemini 3.7 Flash** to **23,190 Gemini 3.8 Flash**. The B2M set has 56,853 rows and 4,897 prompt IDs (per its local README); V11 has 190,588 rows; V14 has 223,840.

## Sources and provenance

| Main file | Rows | Distinct source datasets | Source licence values |
|---|---:|---:|---:|
| V11 | 190,588 | 55 | CC BY 4.0: 158,393; CC BY-SA 4.0: 28,154; CC0-1.0: 4,041 |
| B2M | 56,853 | 21 | CC BY-SA 4.0 (dataset README; per-source licences vary) |
| V14 | 223,840 | 61 | CC BY 4.0: 166,253; CC BY-SA 4.0: 31,062; CC BY 4.0 stated in source README/Kaggle licence unset: 17,507; CC0-1.0: 9,018 |

V14 adds six source datasets relative to the V11 inventory. By rows newly present versus V11, the largest contributions are `kilupy/ntl-claude-family-passages` (14,148), `darragh1walsh/ntl-kaggle-ai-gen-run-1` (5,158), `adurey/ntl-proxy-gen-20261031` (4,152), `kilupy/ntl-18-model-corpus` (3,359), `parkjinsu1/ntl-kb-proxy-panels` (2,908), and `zeminglii/ntl-own-exact-route-generations` (2,000). The V14 archive adds `source_protocol.csv` and generation fields (`gen_reasoning*`, `gen_route`, `gen_served_model`, `gen_date`, and `flag_kb_proxy_after_ruling`) that are absent from V11’s main schema. The V14 card says generation reasoning is known for 54% of rows; in the parquet, route, served-model, and date fields are blank on 152,105, 141,123, and 192,772 rows respectively. V11 has source, file and licence metadata, but lacks those generation-route fields. B2M has `source` and a row-index-aligned provenance sidecar with route/provider/model-version fields.

## Gemini 3.7 Flash attribution and routing risk

| Location | Rows | Change from corresponding V11 file | Evidence and flags |
|---|---:|---:|---|
| V11 main | 3,278 | — | Main pool; routing-risk flag false on all 3,278 |
| V14 main | 3,541 | +263 | Risk flag false on all 3,541; the 263 additional rows are from `zeminglii/ntl-own-exact-route-generations` and record route `openrouter` and served model `google/gemini-3.7-flash` |
| V11 unverified sidecar | 10,593 | — | Separate from main file; 8,297 risk flags true, 2,296 false |
| V14 unverified sidecar | 10,838 | +245 | Contains all 10,593 V11 sidecar responses plus 245 from `kilupy/ntl-18-model-corpus`; 8,297 risk flags true, 2,541 false |
| B2M | 1,958 | — | Provenance sidecar has 158 rows with source `prvsiyan`; the other 1,800 are from sources whose local README reports no row-level served model version |

The V14 card says the main file retains Gemini 3.7 rows only from documented Vertex AI routes because some other APIs served Gemini 3.8 under that label after 2026-10-08. That is the dataset author’s attribution policy, not independent proof of every generation. In the V14 main parquet, 1,534/3,541 rows have non-empty row-level route and served-model fields; 2,007 rows from `davidpiln/ntl-vertex-gemini-gen` have those two fields blank, so their attribution rests on source-level documentation rather than those row fields. The 263 newly added main rows have explicit `google/gemini-3.7-flash` served-model metadata. The risk flag is false for all main-file Gemini 3.7 rows; that should be read as the dataset’s screening result, not as a guarantee.

The V14 unverified sidecar preserves every V11 sidecar response and adds 245 rows; those 245 are not present in V14 main, V11 main, B2M, or the prior V11 sidecar. All 245 have `flag_gemini37_routing_risk=false`, despite being placed in the unverified sidecar, so sidecar membership is the more conservative attribution signal. The V14 sidecar has 889 normalized response overlaps with B2M; the new 245 sidecar rows have zero B2M overlap. V14 main Gemini 3.7 has 1,067 overlaps with B2M. V14 main and its sidecar have no response-text overlap with each other. The two B2M Gemini 3.7 responses not found in either V14 main or sidecar remain B2M-only under this exact-text comparison.

## Findings and limitations

- The downloaded release is V14/Kaggle version 16, not release V12; the requested local directory name is only a path label.
- V14 is 33,252 rows larger than V11 overall, after 33,536 additions and 284 removals.
- Deduplicating V14 against both existing main datasets leaves **33,520** new response texts, distributed across all 18 labels.
- Provenance is more detailed in V14, but route/model/date fields remain blank on many rows; source-level evidence and row-level serving evidence should not be conflated.
- For Gemini 3.7, V14 adds 263 main-file rows with explicit served-model IDs and 245 separate sidecar rows. The sidecar intentionally marks an attribution category outside the main-file acceptance policy; its false routing flags do not independently verify those claims.
- This is exact normalized-text overlap only. It does not measure semantic similarity, confirm that a model generated a passage, or validate the truth of any source’s attribution claim.
