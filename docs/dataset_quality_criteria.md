# Reusable NLM External Dataset Quality Criteria

This policy is source-independent. A source adapter maps its fields into the canonical NLM row format (`id`, `prompt_id`, `text`, `model`, `source_dataset`, `source_row_id`) and provides an auditable language field. The policy is applied only when requested; it does not change existing datasets automatically.

## Eligibility rules

Each row is evaluated in order and receives one first-match exclusion reason:

1. Required response identity, prompt identity, text, model label, or source provenance is missing; canonical response IDs must be unique.
2. The model label is outside the configured canonical NLM label set.
3. The source language field does not explicitly identify English (`en`, `eng`, or `english` by default). Unknown language is excluded in English-only runs.
4. Response length is outside the configured character bounds. The reusable defaults are **50–2,000 characters inclusive**; source-specific alternatives must be explicit in the run config.
5. Available source metadata marks a response truncated, capped, or length-finished.
6. The response contains an unmistakable serialized message/tool-log artifact or an explicit truncation/failure marker.
7. A normalized response duplicates another row. If identical normalized text has conflicting labels, all members are excluded; otherwise the lexically smallest canonical ID is retained.
8. ID, normalized response, known prompt ID, or available canonical prompt text overlaps the supplied validation dataset.

Normalization uses `src.data_factory._normal_text` (Unicode NFKC, case folding, and whitespace collapse). All input fields are preserved on retained rows. Near-duplicate similarity is reported separately and is not an automatic exclusion in this exact filter; approximate candidate retrieval can miss pairs and similarity thresholds require an explicit compaction decision.

The filter does not reject legitimate code, Markdown, reasoning-style text, or multi-turn-derived responses merely because they have those characteristics. It does not infer English from response content when the source language is unknown; source adapters must provide an auditable language field or a separately validated detector.

## Run interface

Use `python -m src.dataset_quality --config <yaml>`. The source adapter accepts Parquet, CSV, JSON, and JSONL/NDJSON. Source paths, field mappings, allowed labels, bounds, validation artifact, outputs, and provenance joins are supplied by the YAML. The command writes a Parquet file, row-level exclusion audit, and a profile with input hashes and row counts. It refuses to overwrite existing outputs.

The GLM-5.3-Flash application is configured in `configs/data/quality_glm53_flash.yaml`.

## Applied GLM-5.3-Flash result

The current run produced `data/processed/external_candidates/glm53_flash_quality_en.parquet`, with its profile and row-level exclusion audit beside it. The detailed counts, hashes, validation checks, and source limitations are in [`glm53_flash_quality_en.md`](glm53_flash_quality_en.md). This is a one-label supplemental corpus, not a stand-alone 18-class training dataset.
