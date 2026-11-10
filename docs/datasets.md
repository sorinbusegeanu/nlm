# Dataset Catalog

This file is the index of the named datasets currently present in the NLM
workspace. Use these names in experiment descriptions and reports.

## Versioned training datasets

| Name | Artifact | Rows | Description | Validation |
|---|---|---:|---|---|
| **V0 — Kaggle baseline** | [`dataset_v0.parquet`](../data/processed/dataset_v0.parquet) | 2,699 | Original labeled Name That LLM Kaggle training data, with 18 model classes. This is the baseline dataset. | Frozen prompt-grouped split in [`splits_v0.parquet`](../data/processed/splits_v0.parquet): 2,159 train and 540 validation rows. |
| **V1 — Balanced Open Pool** | [`dataset_v1.parquet`](../data/processed/dataset_v1.parquet) | 48,798 | Quality-filtered Open Pool responses, reduced to 2,711 items per class to match the smallest eligible class. Remaining items are selected with a fixed-seed, source-proportional sample. Quota-excluded rows are not automatically low quality. | Training rows only. Evaluation uses the same frozen 540-row V0 validation set. |
| **V1.1 — Error-focused Open Pool** | [`dataset_v1.1.parquet`](../data/processed/dataset_v1.1.parquet) | 72,906 | Keeps all V1 rows and adds shorter, quality-filtered Open Pool responses for the repeatedly difficult classes `claude-fable-5`, `glm-5.3`, and `glm-5.3-flash`. Final counts: 9,585, 11,328, and 11,328 respectively; other classes remain at 2,711 each. | Training rows only. Evaluation uses the frozen 540-row V0 validation set. |
| **V1.2-BM — B2M responses absent from V1.1** | [`dataset_v1.2_bm.parquet`](../data/processed/dataset_v1.2_bm.parquet) | 37,061 | Rows from Kaggle `mrralphie/ntl-pool-b2m` whose exact response text is not present in V1.1. Original B2M source, prompt, model-route and provider provenance are retained. This is an imbalanced source-difference extract; no additional NLM quality filtering or balancing was applied. | Training rows only. Evaluation uses the frozen 540-row V0 validation set. |
| **V1-all — V1.1 + V1.2-BM** | [`dataset_v1_all.parquet`](../data/processed/dataset_v1_all.parquet) | 109,967 | Union of all V1.1 and V1.2-BM training rows. Exact response text was deduplicated between the components; none overlapped, so every row is retained. Class counts follow the combined source distributions; no balancing or additional quality filtering was applied. | Training rows only. Evaluation uses the frozen 540-row V0 validation set. |
| **V1.5 — V0 + V1** | [`dataset_v1.5.parquet`](../data/processed/dataset_v1.5.parquet) | 51,497 | Combines V1’s 48,798 Open Pool rows with V0’s 2,159 training rows. Keeps a single explicit train/validation split. | 50,957 train and the unchanged 540 V0 validation rows, recorded in the parquet and [`splits_v1.5.parquet`](../data/processed/splits_v1.5.parquet). |
| **V2.14 — Open Pool V14 + B2M additions** | [`dataset_v2.14.parquet`](../data/processed/dataset_v2.14.parquet) | 231,336 | All 223,840 rows from the V14 main file plus 7,496 B2M response texts absent from V14 after NFKC, case-fold and whitespace normalization. Natural class counts and original source provenance are retained; no quality filtering or balancing was added. The V14 unverified sidecar is excluded. The 891 added B2M Gemini 3.7 rows retain their `quarantine` attribution flag. | External training rows only; the training factory uses full V0 validation (2,699 rows). Profile: [`dataset_profile_v2.14.json`](../data/processed/dataset_profile_v2.14.json). |
| **V2 — Full eligible Open Pool + V0** | [`dataset_v2.parquet`](../data/processed/dataset_v2.parquet) | 190,794 | Combines all 188,095 Open Pool responses that pass the hard quality filters with V0’s 2,159 training rows. It preserves the Open Pool class imbalance; “full” means all eligible rows after hard exclusions, with no per-class quota downsampling. | 190,254 train and the unchanged 540 V0 validation rows, recorded in the parquet and [`splits_v2.parquet`](../data/processed/splits_v2.parquet). |

## Open Pool staging dataset

| Name | Artifact | Rows | Description |
|---|---|---:|---|
| **V2 Open Pool — eligible source rows** | [`dataset_v2_openpool.parquet`](../data/processed/dataset_v2_openpool.parquet) | 188,095 | Quality-filtered English Open Pool responses before adding V0 training rows. Preserves the natural class distribution. This is the source component used to assemble V2, not the final V2 training dataset. |

## Validation and provenance notes

- V0’s 540 validation examples are held fixed across V1, V1.1, V1.5, and V2. V1 and V1.1 contain only their Open Pool training rows; their validation examples are referenced from V0 rather than copied into those parquet files.
- V1 and V1.1 use hard quality filters before their selection rules. V1 applies a per-class quota; V1.1 deliberately adds more examples for three difficult classes. V2 Open Pool keeps all rows that pass the hard filters and has no class quota.
- The current profiles identify the raw source as [`ntl_open_pool.parquet`](../data/raw/open_pool/ntl_open_pool.parquet) and label the curation methods `openpool_v11_*`. The recent shorthand “OpenPool V10” conflicts with that recorded provenance; this catalog follows the artifact profiles and records the source as Open Pool V11.
- There is currently no V1.6 dataset artifact or recipe in the workspace.
- [`test_v0.parquet`](../data/processed/test_v0.parquet) is a local 3-row test fixture, not the full Kaggle test set and not training data.

## Supporting files

- Dataset profiles: [`dataset_profile.json`](../data/processed/dataset_profile.json), [`dataset_profile_v1.json`](../data/processed/dataset_profile_v1.json), [`dataset_profile_v1.1.json`](../data/processed/dataset_profile_v1.1.json), [`dataset_profile_v1.2_bm.json`](../data/processed/dataset_profile_v1.2_bm.json), [`dataset_profile_v1_all.json`](../data/processed/dataset_profile_v1_all.json), [`dataset_profile_v1.5.json`](../data/processed/dataset_profile_v1.5.json), [`dataset_profile_v2.json`](../data/processed/dataset_profile_v2.json), [`dataset_profile_v2.14.json`](../data/processed/dataset_profile_v2.14.json), and [`dataset_profile_v2_openpool.json`](../data/processed/dataset_profile_v2_openpool.json).
- Curation audits: [`openpool_v1_removals.csv`](../data/processed/openpool_v1_removals.csv), [`openpool_v1.1_removals.csv`](../data/processed/openpool_v1.1_removals.csv), and [`openpool_v2_openpool_removals.csv`](../data/processed/openpool_v2_openpool_removals.csv).
- Factory definitions: [`openpool_v1.yaml`](../configs/data/openpool_v1.yaml), [`openpool_v1_1.yaml`](../configs/data/openpool_v1_1.yaml), [`openpool_v2.yaml`](../configs/data/openpool_v2.yaml), [`mixed_v1_5.yaml`](../configs/data/mixed_v1_5.yaml), and [`mixed_v2.yaml`](../configs/data/mixed_v2.yaml).
