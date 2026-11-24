# V2.GR.OR16 Sequential Training Queue

Queue: `e32_gr_or16_models_20261124T022231Z`  
Status: **completed**  
Dependency: E32 status **Scored**  
Dataset: `v2.GR.OR16` (231,036 rows; SHA-256 `9373bbd85f03de1f384cb8ae3b6f9b2b27d576629b8fcf0ab344d966fa266a0a`)  
Validation: full V0 (2,699 rows)  
Checkpoint selection: validation balanced accuracy.

Jobs run sequentially after E32's scored artifacts are verified and the GPU/training process list is idle.
SVC and CatBoost use their existing single-fit procedures. MPNet uses its existing five-epoch full fine-tune recipe.
EmbeddingGemma 2's registered recipe is LoRA with 10 configured epochs (the existing config is epoch-based); it is preserved as-is.
Granite 4.0 350M uses its existing LoRA recipe with epochs changed to the requested five.

| Order | Model | Epochs/configuration | Status | Duration (s) | V0 balanced accuracy | Selected checkpoint |
|---:|---|---|---|---:|---:|---|
| 1 | svc | None | completed | 658.7 | 46.9321% | `artifacts/runs/svc_v2.GR.OR16_valv0_data5cf9c6ef_split421e6ba8_seed2026_cfg66dc1893_code996b0b9/checkpoints/svc_foldvalidation.joblib` |
| 2 | catboost_stylometry | None | completed | 1.1 | 17.1153% | `artifacts/runs/catboost_stylometry_v2.GR.OR16_valv0_data5cf9c6ef_split421e6ba8_seed2026_cfga1d56f56_code996b0b9/checkpoints/catboost_stylometry_foldvalidation` |
| 3 | mpnet_base | 5 | completed | 3537.3 | 48.6791% | `artifacts/runs/mpnet_base_v2.GR.OR16_valv0_data5cf9c6ef_split421e6ba8_seed2026_cfg4d404e9a_code996b0b9/checkpoints/mpnet_base_foldvalidation` |
| 4 | embeddinggemma2 | 10 | completed | 17022.3 | 47.1959% | `artifacts/runs/embeddinggemma2_v2.GR.OR16_valv0_data5cf9c6ef_split421e6ba8_seed2026_cfgcf7064ea_code996b0b9/checkpoints/embeddinggemma2_foldvalidation` |
| 5 | granite4_350m | 5 | completed | 6765.3 | 51.9364% | `artifacts/runs/granite4_350m_v2.GR.OR16_valv0_data5cf9c6ef_split421e6ba8_seed2026_cfgf6c5aa37_codef5d96d5/checkpoints/granite4_350m_foldvalidation` |

Queue state: `artifacts/training_queues/e32_gr_or16_models_20261124T022231Z/queue.json`
