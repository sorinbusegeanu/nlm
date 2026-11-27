# GLM-5.3-Flash English Quality Filter

Applied the reusable [NLM external dataset quality criteria](dataset_quality_criteria.md) to the four files in the [Hugging Face harvest](https://huggingface.co/datasets/Zek-Takai/glm53-flash-harvest). This filter is source-independent; it is not the V1.3-specific 1,900-character recipe. Raw downloads and V1.3 artifacts were left unchanged.

## Results

| Measure | Rows |
|---|---:|
| Raw input | 86,006 |
| Retained | 7,083 |
| Removed | 78,923 |
| Reduction | 91.76% |

The retained set contains only `glm-5.3-flash`. It has **7,083 unique normalized responses**, each explicitly tagged English and 50–2,000 characters long. The exact filter removed all records outside those bounds, explicit source truncation/cap records, explicit serialization/failure artifacts, and normalized duplicates when present. The audit uses first-match reasons, so the reason counts below are mutually exclusive and do not count overlapping defects more than once.

| First-match exclusion reason | Rows |
|---|---:|
| Not explicitly English (`lang` was `zh`) | 1,709 |
| Over 2,000 characters | 77,192 |
| Under 50 characters | 18 |
| Normalized duplicate response | 4 |
| **Total removed** | **78,923** |

No retained row has source metadata marking it truncated or cap-hit. The retained Parquet preserves all source columns and adds canonical NLM `id`, `prompt_id`, `text`, `model`, `source_dataset`, `source_file`, `source_row_id`, `source_original_id`, `language`, and `prompt_canon` fields.

## Isolation and limitations

- No canonical ID, source ID, normalized response, or prompt-ID overlap was found against the complete 2,699-row V0 artifact.
- V0 has no canonical prompt text in the local artifact, so prompt-text overlap with V0 could not be measured.
- No normalized response or prompt text overlaps V3.1, based on comparison of the full raw harvest against V3.1; this retained set is a subset of that harvest.
- Near-duplicate similarity is not an automatic exclusion in this general filter. Use the separate documented compaction step when near-duplicate removal is desired.
- Language eligibility uses the source's explicit `lang` field; the filter does not guess language from response content.
- Source model attribution is dataset-level. It is preserved as provenance, not independently verified.

## Artifacts

- Filtered data: [`glm53_flash_quality_en.parquet`](../data/processed/external_candidates/glm53_flash_quality_en.parquet)
- Row-level audit: [`glm53_flash_quality_en_removals.csv`](../data/processed/external_candidates/glm53_flash_quality_en_removals.csv)
- Profile: [`glm53_flash_quality_en_profile.json`](../data/processed/external_candidates/glm53_flash_quality_en_profile.json)
- Configuration: [`quality_glm53_flash.yaml`](../configs/data/quality_glm53_flash.yaml)
- Reproduce with: `python -m src.dataset_quality --config configs/data/quality_glm53_flash.yaml`

SHA-256 of filtered Parquet: `90e178347e6e79f8fb7b3663050429f27457ca118d2dba15adf6a91ad0d6cf0a`.
