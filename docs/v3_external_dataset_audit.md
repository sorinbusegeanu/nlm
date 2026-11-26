# New external datasets compared with V3.0

**Audit date:** 2026-11-25. Exact response overlap uses V3.0 IDs and the data-factory response normalization (Unicode NFKC, casefold, whitespace collapse), hashed with SHA-256. No training, inference, ranking refresh or factory-state update was performed.

## Dataset summary

| Dataset | Local version | Latest version | Total responses | Already in V3.0 | New to V3.0 | Included in quality ranking | Included in improvement factory |
|---|---|---:|---:|---:|---:|---|---|
| `maniceet/name-that-llm-openrouter-openai-generations` | Cached extract, version unrecorded: 3,003 rows | Public Kaggle card: 7,753 rows (10 labels); latest raw file could not be downloaded (HTTP 403) | 7,753 reported | 0 of 3,003 cached rows | 3,003 confirmed cached; up to 4,750 additional rows unavailable for audit | No (0 source-label pairs) | No |
| `crtproduction/name-that-llm-open-pool` | v16, 231,036 rows at `data/raw/open_pool/v16/` (also an unversioned 190,588-row snapshot at `data/raw/open_pool/`) | v17, 238,750 rows, downloaded to `data/raw/external_candidates/v3_external_audit_2026-11-25/crtproduction_v17/` | 238,750 | 159,832 | 78,918 | Aggregate is not a contributor; new v17 source-label pairs: 0 | New v17 contributors: No |

### How to read the Open Pool counts

All 159,832 V3.0 responses match the v17 Open Pool by response ID and normalized response hash. The 78,918 rows called “new to V3.0” include **71,204 rows already present in local v16 but deliberately excluded** from V3.0 construction, plus **7,714 actual v17 additions** since v16. None of the 7,714 additions overlap V3.0. The additional unversioned 190,588-row local snapshot shares 190,304 IDs/texts with v17; 284 of its rows are absent from v17. The current V3 quality-ranking and improvement-factory results predate those four newly added source contributors.

### Maniceet local cache by label

All 3,003 cached responses have unique normalized text, zero normalized-text overlap with V3.0, and zero normalized-text overlap with the downloaded v17 Open Pool. The cached file has no original response-ID field, so response-ID overlap cannot be tested for it. These are confirmed new cached rows, but they are not a complete snapshot of the public 7,753-row release. The public Kaggle card reports the following current label counts: `gemini-3.7-flash 22, glm-5.3 1,839, glm-5.3-flash 2,828, gpt-5.5 370, gpt-5.6-luna 1,058, gpt-5.6-sol 759, gpt-5.6-terra 759, gpt-6-astra 50, kimi-k3 63, qwen3.8-max 5`.

| Label | Cached rows | Already in V3.0 | Duplicate in CRT v17 | Confirmed new cached rows |
|---|---:|---:|---:|---:|
| glm-5.3 | 166 | 0 | 0 | 166 |
| glm-5.3-flash | 531 | 0 | 0 | 531 |
| gpt-5.6-luna | 835 | 0 | 0 | 835 |
| gpt-5.6-sol | 735 | 0 | 0 | 735 |
| gpt-5.6-terra | 736 | 0 | 0 | 736 |

## Actual v17 additions by contributor and label

These 7,714 are the genuinely new records added between local v16 and Kaggle v17; every row is absent from V3.0. The full 78,918-row V3.0-missing breakdown (including V16 rows omitted during V3 construction) is in [`crtproduction_v17_new_to_v3_by_contributor_label.csv`](../artifacts/analysis/v3_external_dataset_audit/crtproduction_v17_new_to_v3_by_contributor_label.csv).

| Contributor | Label | New v17 rows | Included in existing quality ranking | Included in improvement factory |
|---|---|---:|---|---|
| ankiteshtiwari/ntl-blocks-v1 | claude-sonnet-5 | 788 | No | No |
| ankiteshtiwari/ntl-blocks-v1 | gemini-3.6-flash | 239 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | claude-opus-4-6 | 28 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | claude-opus-4-8 | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | claude-opus-5 | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | claude-sonnet-5 | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gemini-3.1-pro-preview | 22 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gemini-3.6-flash | 23 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gemini-3.8-flash | 22 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gpt-5.5 | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gpt-5.6-luna | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gpt-5.6-sol | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gpt-5.6-terra | 29 | No | No |
| beroli/ntl-benchmark-c1-generated-panels | gpt-6-astra | 30 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | claude-opus-4-6 | 27 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | claude-opus-4-8 | 37 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | claude-opus-5 | 24 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | claude-sonnet-5 | 45 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gemini-3.1-pro-preview | 37 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gemini-3.6-flash | 35 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gemini-3.8-flash | 37 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gpt-5.5 | 17 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gpt-5.6-luna | 49 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gpt-5.6-sol | 33 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gpt-5.6-terra | 43 | No | No |
| beroli/ntl-benchmark-c2-generated-panels | gpt-6-astra | 48 | No | No |
| tadeuhayashida/ntl-gen003-claude-panel | claude-opus-4-6 | 1,488 | No | No |
| tadeuhayashida/ntl-gen003-claude-panel | claude-opus-4-8 | 1,491 | No | No |
| tadeuhayashida/ntl-gen003-claude-panel | claude-opus-5 | 1,464 | No | No |
| tadeuhayashida/ntl-gen003-claude-panel | claude-sonnet-5 | 1,484 | No | No |

## Quality process status

The current quality ranking contains 490 contributor-label entries from 53 contributors; the contributor-level improvement factory ranks 53 sources. None of the v17 increment’s four contributors or 30 source-label subsets are represented. The Maniceet contributor and its cached label subsets are also absent. For existing Open Pool source contributors, the detailed CSV marks whether each missing source-label pair already has a V3 ranking entry and whether the source appears in the contributor-level factory; the new v17 additions themselves have no V3 rows and are not in either process.

## Recommendation

1. **Maniceet:** add the full current 7,753-row dataset as its own contributor after access is restored; its local 3,003-row cache is already demonstrably disjoint from V3.0 and Open Pool v17. The public card describes per-row model, provider/route and prompt provenance, one-paragraph responses, 10 labels, and CC BY 4.0. Before ranking, complete the missing 4,750-row audit and run the normal V0/V3 response and prompt isolation gates.

2. **V17 additions:** add the four new source datasets—not the aggregate Open Pool as a single contributor—to the next contributor-label quality-ranking input: `tadeuhayashida/ntl-gen003-claude-panel` (5,927), `ankiteshtiwari/ntl-blocks-v1` (1,027), `beroli/ntl-benchmark-c2-generated-panels` (432), and `beroli/ntl-benchmark-c1-generated-panels` (328). This preserves the factory’s source attribution and lets the existing per-label ranking compare them fairly. They currently have no V3 quality results, so no quality claim is made here.

## Files and provenance

- Latest Open Pool v17 Parquet: `data/raw/external_candidates/v3_external_audit_2026-11-25/crtproduction_v17/ntl_open_pool.parquet` (238,750 rows).
- Kaggle metadata snapshot: `data/raw/external_candidates/v3_external_audit_2026-11-25/crtproduction_v17/kaggle_dataset_metadata_api.json`.
- Previous local Open Pool v16: `data/raw/open_pool/v16/ntl_open_pool.parquet` (231,036 rows).
- Additional older Open Pool snapshot: `data/raw/open_pool/ntl_open_pool.parquet` (190,588 rows; no version metadata alongside it).
- Maniceet cached extract: `data/raw/external_candidates/v2.15_extrato_14/maniceet_cached/maniceet_cached.parquet` (3,003 rows) and adjacent provenance Parquet.
- Detailed contributor-label counts and process-membership flags: `artifacts/analysis/v3_external_dataset_audit/`.
- V3.0 source profile SHA-256: `25c8b40824c0b809e2e6479960dd71960c9f4c695353144ba779738178c66c92`.
- Open Pool v17 Parquet SHA-256: `fe6ece20e6f9f2fe5fb3c47f2aaa9749394c1056bc688a8eab865e0dc2bb56df`.
- Kaggle pages: [Maniceet dataset](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations), [Open Pool v17](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool).
