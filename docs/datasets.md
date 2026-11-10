# Dataset Catalog

This page is generated from [`configs/datasets.yaml`](../configs/datasets.yaml). Add or update dataset entries in that file, then run `python -m src.dataset_catalog --render-docs`.

Canonical descriptions and artifact locations for processed datasets in the NLM workspace. Add new maintained datasets here; the dashboard and generated docs/datasets.md read from this catalog.

## Baseline and historical split

| Name | Artifact | Rows | Description | Validation | Profile | Related reports |
|---|---|---:|---|---|---|---|
| **V0 — Kaggle baseline** | [`dataset_v0.parquet`](../data/processed/dataset_v0.parquet) | 2,699 | Original labeled Name That LLM Kaggle training data with 18 model classes. | Historical frozen prompt-group split: 2,159 train and 540 validation rows. | [`dataset_profile.json`](../data/processed/dataset_profile.json) | — |

## Training datasets

| Name | Artifact | Rows | Description | Validation | Profile | Related reports |
|---|---|---:|---|---|---|---|
| **V1 — Balanced Open Pool** | [`dataset_v1.parquet`](../data/processed/dataset_v1.parquet) | 48,798 | Quality-filtered Open Pool responses, reduced to 2,711 items per class with a fixed-seed source-proportional sample. Quota-excluded rows are not necessarily low quality. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1.json`](../data/processed/dataset_profile_v1.json) | — |
| **V1.1 — Error-focused Open Pool** | [`dataset_v1.1.parquet`](../data/processed/dataset_v1.1.parquet) | 72,906 | V1 plus shorter quality-filtered Open Pool responses for `claude-fable-5`, `glm-5.3`, and `glm-5.3-flash`. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1.1.json`](../data/processed/dataset_profile_v1.1.json) | — |
| **V1.2-BM — B2M responses absent from V1.1** | [`dataset_v1.2_bm.parquet`](../data/processed/dataset_v1.2_bm.parquet) | 37,061 | B2M response texts not present in V1.1, retaining source, prompt, route, and provider provenance. No additional quality filtering or balancing was applied. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1.2_bm.json`](../data/processed/dataset_profile_v1.2_bm.json) | — |
| **V1-all — V1.1 plus V1.2-BM** | [`dataset_v1_all.parquet`](../data/processed/dataset_v1_all.parquet) | 109,967 | Union of V1.1 and V1.2-BM after exact response deduplication; natural class counts are retained without additional balancing. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1_all.json`](../data/processed/dataset_profile_v1_all.json) | — |
| **V1.5 — V0 plus V1** | [`dataset_v1.5.parquet`](../data/processed/dataset_v1.5.parquet) | 51,497 | Combines V1 Open Pool rows with the V0 training partition. | 50,957 train and the unchanged 540 V0 validation rows, with explicit split assignments. | [`dataset_profile_v1.5.json`](../data/processed/dataset_profile_v1.5.json) | — |
| **V1.3 — External GLM and GPT sources** | [`dataset_v1.3.parquet`](../data/processed/dataset_v1.3.parquet) | 21,201 | Normalized collection from external GLM-5.3-Flash and GPT-5.6 Sol/Luna corpora, before defect cleaning. | External training rows; only three of the 18 model labels are represented. | [`dataset_profile_v1.3.json`](../data/processed/dataset_profile_v1.3.json) | [v1.3.md](../docs/v1.3.md) |
| **V1.3 Clean** | [`dataset_v1.3_clean.parquet`](../data/processed/dataset_v1.3_clean.parquet) | 12,869 | V1.3 after explicit defect cleaning. | External training rows; only three of the 18 model labels are represented. | [`dataset_profile_v1.3_clean.json`](../data/processed/dataset_profile_v1.3_clean.json) | — |
| **V1.3 Compact** | [`dataset_v1.3_compact.parquet`](../data/processed/dataset_v1.3_compact.parquet) | 12,866 | V1.3 Clean with verified same-label near-duplicate rows removed. | External training rows; only three of the 18 model labels are represented. | [`dataset_profile_v1.3_compact.json`](../data/processed/dataset_profile_v1.3_compact.json) | — |
| **V1-all Clean** | [`dataset_v1-all_clean.parquet`](../data/processed/dataset_v1-all_clean.parquet) | 109,967 | V1-all screened for demonstrable defects. The audit found no defect rows, so all source rows remain. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1-all_clean.json`](../data/processed/dataset_profile_v1-all_clean.json) | [v1_all_clean_compact.md](../docs/v1_all_clean_compact.md) |
| **V1-all Compact** | [`dataset_v1-all_compact.parquet`](../data/processed/dataset_v1-all_compact.parquet) | 109,967 | V1-all Clean screened for measured redundancy. No qualifying near-duplicate rows were found, so all source rows remain. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1-all_compact.json`](../data/processed/dataset_profile_v1-all_compact.json) | [v1_all_clean_compact.md](../docs/v1_all_clean_compact.md) |
| **V1-all Balanced and Deduped** | [`dataset_v1-all_balanced_deduped.parquet`](../data/processed/dataset_v1-all_balanced_deduped.parquet) | 80,324 | V1-all curated with verified near-duplicate detection and adaptive source/prompt diversity selection; it does not enforce equal class quotas. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v1-all_balanced_deduped.json`](../data/processed/dataset_profile_v1-all_balanced_deduped.json) | [v1_all_balanced_deduped.md](../docs/v1_all_balanced_deduped.md) |
| **V2 — Full eligible Open Pool plus V0** | [`dataset_v2.parquet`](../data/processed/dataset_v2.parquet) | 190,794 | Combines the eligible Open Pool rows with the V0 training partition while preserving the Open Pool class distribution. | Historical split contains 190,254 train and the unchanged 540 V0 validation rows. | [`dataset_profile_v2.json`](../data/processed/dataset_profile_v2.json) | — |
| **V2.0 — Balanced external corpus** | [`dataset_v2.0.parquet`](../data/processed/dataset_v2.0.parquet) | 51,606 | Balanced external-only corpus assembled from Open Pool, B2M, and available V1.3 sources; sampling targets 3,000 examples per class, reduced to 2,867 by the limiting class. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v2.0.json`](../data/processed/dataset_profile_v2.0.json) | [v2.0.md](../docs/v2.0.md) |
| **V2.1 — Balanced external corpus with unverified Gemini 3.7 source** | [`dataset_v2.1.parquet`](../data/processed/dataset_v2.1.parquet) | 54,000 | 3,000 examples per class from Open Pool, B2M, V1.3, and unflagged rows in the unverified Gemini 3.7 sidecar; unverified attribution is retained as a caveat. | External training rows; use the V0 validation dataset in training experiments. | [`dataset_profile_v2.1.json`](../data/processed/dataset_profile_v2.1.json) | [v2.1.md](../docs/v2.1.md) |
| **V2.14 — Open Pool V14 plus B2M additions** | [`dataset_v2.14.parquet`](../data/processed/dataset_v2.14.parquet) | 231,336 | All 223,840 V14 main-file rows plus 7,496 B2M response texts absent from V14 after NFKC, case-fold, and whitespace normalization. No quality filtering or balancing was added; 891 added B2M Gemini 3.7 rows retain their quarantine flag. The V14 unverified sidecar is excluded. | External training rows; all 2,699 V0 rows are used for validation by the full-V0 training workflow. | [`dataset_profile_v2.14.json`](../data/processed/dataset_profile_v2.14.json) | [v2.14.md](../docs/v2.14.md) · [openpool_v14_comparison.md](../docs/openpool_v14_comparison.md) |
| **V2.14 Balanced — quality-filtered V2.14** | [`dataset_v2.14.balanced.parquet`](../data/processed/dataset_v2.14.balanced.parquet) | 63,702 | V2.14 filtered by response length, V0 overlap, normalized duplicates/conflicts, language, routing, truncation, multi-turn, serialized output, code, and explicit reasoning traces; balanced at the smallest eligible class count of 3,539. | External training rows; all 2,699 V0 rows are used for validation by the full-V0 training workflow. | [`dataset_profile_v2.14.balanced.json`](../data/processed/dataset_profile_v2.14.balanced.json) | [v2.14.balanced.md](../docs/v2.14.balanced.md) |

## Staging datasets

| Name | Artifact | Rows | Description | Validation | Profile | Related reports |
|---|---|---:|---|---|---|---|
| **V2 Open Pool — eligible source rows** | [`dataset_v2_openpool.parquet`](../data/processed/dataset_v2_openpool.parquet) | 188,095 | English Open Pool responses retained by the hard quality filters, before assembling the V2 mix. | Staging training source; no V0 rows are included. | [`dataset_profile_v2_openpool.json`](../data/processed/dataset_profile_v2_openpool.json) | — |

## Notes

- The V0 baseline has 2,699 rows; its historical frozen split is 2,159 train and 540 validation rows. External-only training can use full V0 validation where the training workflow is configured to do so.
- Open Pool-derived datasets in this catalog use the locally downloaded Open Pool V11 source unless a description names a later release. Some older artifact profiles use the `openpool_v11_*` curation naming.
- `data/processed/test_v0.parquet` is a small test fixture, not a maintained dataset or the Kaggle test set, and is intentionally not cataloged.
- There is no V1.6 dataset artifact or build recipe registered at this time.

## Resources

- [Data factories](../configs/data/): Dataset factory configuration files.
- [V1 curation audit](../data/processed/openpool_v1_removals.csv): Row-level V1 quality exclusions and quota sampling decisions.
- [V1.1 curation audit](../data/processed/openpool_v1.1_removals.csv): Row-level V1.1 quality exclusions and selection decisions.
- [V2 Open Pool curation audit](../data/processed/openpool_v2_openpool_removals.csv): Row-level hard-quality exclusions for the V2 Open Pool staging dataset.
