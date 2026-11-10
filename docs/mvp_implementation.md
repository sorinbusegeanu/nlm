# MVP Implementation Runbook

## Operating protocol

The V0 workflow uses one frozen, prompt-grouped 80/20 split. The Data Factory searches 1,000 deterministic grouped candidates and selects the split with the closest validation row fraction and class distribution to the complete training CSV. The selection uses IDs, prompt groups, and labels only. It verifies that every label appears in both partitions and that no prompt group crosses the boundary.

For the current 2,699-row dataset, the split is 2,159 train rows and 540 validation rows across 120 and 30 prompt groups respectively. Each default model fits once on those 2,159 rows. All model and blend scores use the same 540 validation rows. Since the validation rows are reused for model comparisons, scores are development evidence and can become optimistic; there is no separate holdout score in this protocol.

## Clean restart

From the repository root, run:

```bash
python -m src.pipeline restart
```

This command clears generated contents under `data/processed/` and `artifacts/`, deletes `experiments/registry.sqlite` plus SQLite WAL/SHM files, then runs the Data, Feature, Model, and Ensemble factories in order. It retains `data/raw/` and the Hugging Face model cache. The registry is rebuilt from sequence 0001. By default `svc`, `catboost_stylometry`, `ultrabert380`, and `qwen06` are trained; use `--models svc catboost_stylometry` to choose a subset, and `--seed 2027` to create a different deterministic split.

## Individual factories

```bash
python -m src.pipeline data prepare --config configs/data/official_v0.yaml
python -m src.pipeline data curate-openpool --config configs/data/openpool_v1.yaml
python -m src.pipeline features build --dataset v0
python -m src.pipeline models train --model svc --dataset v0
python -m src.pipeline models train --model catboost_stylometry --dataset v0
python -m src.pipeline models train --model ultrabert380 --dataset v0
python -m src.pipeline models train --model qwen06 --dataset v0
python -m src.pipeline models train --model smollm3_3b --dataset v0
python -m src.pipeline models train --model granite4_h_micro --dataset v0
python -m src.pipeline ensemble evaluate --dataset v0
```

All model configs must point to the intended model and an immutable Hugging Face revision. The Qwen model remains configurable through `configs/models/qwen06.yaml`; LoRA target module names must match the chosen architecture. The ensemble factory predicts the train partition with each fitted checkpoint, chooses a regularized weight vector using only those train labels/predictions, freezes it, and reports its validation score alongside the equal-weight score. Those train predictions are in-sample, so the train score is descriptive; only the validation rows compare transfer of the weights.

LoRA model configs can enable a GPU batch-size probe before training. It exercises candidate per-device batches with a forward/backward pass up to `max_batch_size`, preserving a configurable memory reserve. The largest successful size is used for training and validation, with the attempts and selected size recorded in the experiment manifest. If CUDA is unavailable, the configured `batch_size` is used. LoRA configs may set `lr_scheduler_type` to `linear`, `cosine`, `constant`, or `epoch_decay`, plus `warmup_ratio` for the standard schedulers. The Qwen starting configs use LR `5e-5`, linear decay, and 2% warmup; set `learning_rate_decay` with `epoch_decay` for the previous multiplicative per-epoch schedule. Validation and best-checkpoint selection continue once per epoch.

### Open Pool response dataset

`data curate-openpool` reads only the retained `ntl_open_pool.parquet`. It keeps English responses with source, file, license, and canonical prompt-group metadata, drops exact normalized overlaps with official `train.csv`, and creates 2,711 responses per class. Hard defects (including missing required provenance, invalid Gemini 3.7 routing, duplicate/conflicting responses, official-train overlap, and non-English or undetermined text) are recorded by reason. Remaining eligible rows are selected with a fixed-seed, source-proportional sample; quota sampling is not a claim that the discarded rows are defective. The unverified Gemini 3.7 sidecar is never included. This is a training-only corpus: it has no Open Pool train/validation split file. When `--dataset v1` is selected, all Open Pool rows form the training dataset and validation uses all rows in V0; no V0 training rows are added. Prompt groups are checked for train/validation isolation. Prompt text is not a model input; `prompt_id` is retained as metadata.

The generated files are `data/processed/dataset_v1.parquet` and `dataset_profile_v1.json`. Passing `--dataset v1` trains on every Open Pool row; `--validation-dataset v0` validates on all 2,699 official V0 examples. The V0 data is not included in V1 training. Build features and train with the explicit dataset pair, for example:

```bash
python -m src.pipeline features build --dataset v1 --validation-dataset v0
python -m src.pipeline models train --model svc --dataset v1 --validation-dataset v0
python -m src.pipeline models train --model qwen06 --dataset v1 --validation-dataset v0
```

The dataset profile is stored separately from V0 and records source hashes, filter counts, class/source/language distributions, and the V0 reference used during curation. `openpool_v1_removals.csv` records the source row, ID, label, source and removal reason for every excluded row, including quota sampling. Curation stops if any class has fewer than 2,711 eligible English responses.

### Error-focused imbalanced Open Pool dataset (V1.1)

V1.1 preserves every row and label count from V1, then adds cleaned Open Pool responses for the three labels with the most recurring UltraBERT validation errors: `claude-fable-5`, `glm-5.3`, and `glm-5.3-flash`. The choice is based on per-class errors over the same 540 frozen V0 validation rows in the strongest available V0, V1, and V1.5 UltraBERT runs. Across those runs, GLM-5.3 had 66 errors and GLM-5.3-Flash 64; Claude Fable had 62 overall and more errors than the tied GPT-5.6-Sol class in the two newer runs.

The cap is the number of eligible GLM-5.3 records (11,328 total, including its 2,711 V1 rows). The current cleaned pool has fewer Claude Fable rows, so the final counts are 9,585 Claude Fable, 11,328 GLM-5.3, 11,328 GLM-5.3-Flash, and 2,711 for each other label, for 72,906 total rows. V1 rows are retained exactly; additions are chosen by shortest response (characters, then words, then ID) after the existing hard quality filters. V1.1 continues to use the frozen V0 validation partition and does not add validation rows.

Create it with:

```bash
python -m src.pipeline data curate-openpool --config configs/data/openpool_v1_1.yaml
```

The data factory writes `dataset_v1.1.parquet`, `dataset_profile_v1.1.json`, and `openpool_v1.1_removals.csv` under `data/processed/`. The profile includes the per-class counts, baseline hash, added rows, removals, and frozen validation split hash.

### Complete Open Pool V11 plus original V0 training data (V2)

V2 uses the same hard quality filters as V1 but does not downsample to a fixed per-class quota. The source artifact `dataset_v2_openpool.parquet` retains every eligible English response from `ntl_open_pool.parquet`, preserving class imbalance and source frequencies. It removes rows with missing text/label/provenance or canonical prompt, invalid Gemini 3.7 source routing, exact normalized overlap with official training text, duplicate/conflicting normalized response text, or non-English/undetermined language. The unverified Gemini 3.7 sidecar remains excluded. Its audit (`openpool_v2_openpool_removals.csv`) contains hard-quality exclusions only; there is no quota-sampling removal.

The final `dataset_v2.parquet` adds all 2,159 original V0 training rows to the 188,095 eligible Open Pool responses and preserves the same 540 V0 validation rows. Thus V2 contains 190,254 training rows and 540 validation rows. The V0 validation examples remain held out to prevent evaluation leakage. Build it with:

```bash
python -m src.pipeline data curate-openpool --config configs/data/openpool_v2.yaml
python -m src.pipeline data assemble --config configs/data/mixed_v2.yaml
```

The mixed factory writes `dataset_v2.parquet`, `splits_v2.parquet`, and `dataset_profile_v2.json`. `training_data_version` distinguishes original V0 rows from V2 Open Pool rows; these are metadata, not model features. V1 and V1.5 artifacts remain unchanged.

### Consolidated V1.5 training dataset

To train on both V0 training rows and V1 Open Pool rows in one explicit dataset, run:

```bash
python -m src.pipeline data assemble --config configs/data/mixed_v1_5.yaml
```

The factory writes `dataset_v1.5.parquet`, `splits_v1.5.parquet`, and `dataset_profile_v1.5.json`. It combines the 2,159 V0 training rows with all 48,798 curated V1 rows (50,957 training rows total) and preserves the existing 540 V0 validation rows and their prompt-group assignments. It makes no new validation split. Every V1 row is assigned to train, and the factory refuses prompt leakage, overlapping IDs, changed source data, or attempts to overwrite a different V1.5 artifact. Unlike `--dataset v1`, `--dataset v1.5` includes the V0 training partition.

The combined parquet is one dataset with an explicit `split` column; the assignments file is its integrity sidecar. The `training_data_version` column records whether each row came from V0 or V1 and is metadata, not a model feature. Use the resulting dataset with the existing factories, for example:

```bash
python -m src.pipeline features build --dataset v1.5
python -m src.pipeline models train --model ultrabert380 --dataset v1.5
```

## Artifacts and dashboard

- `data/processed/dataset_v0.parquet`: canonical sorted train data.
- `data/processed/splits_v0.parquet`: ID, prompt group, label, and frozen `train` or `validation` assignment.
- `data/processed/dataset_profile.json`: source hashes, class counts, partition counts, seed, candidate count, and split hash.
- `data/processed/dataset_v1.5.parquet` and `splits_v1.5.parquet`: consolidated V0+V1 training rows with frozen V0 validation assignments.
- `artifacts/features/`: train-fitted vectorizer cache and feature manifest.
- `artifacts/runs/`: model checkpoints.
- `artifacts/predictions/`: per-model validation predictions/manifests and blend comparisons.
- `experiments/registry.sqlite`: experiment order, descriptions, configs, metrics, runtime, and artifact pointers.

Run the dashboard with `streamlit run dashboard/app.py`. It reports the source split and experiment sequence; model validation metrics should not be interpreted as an independent final estimate after repeated comparisons.

## Kaggle inference from a selected checkpoint

Training and continuation runs retain their checkpoints and manifests. Compare those runs using their recorded validation metrics, select the desired checkpoint, and use that exact artifact for inference; there is no separate all-rows retraining stage. Pass the selected training or continuation manifest explicitly to `export --checkpoint` and `package-kaggle --manifest`. The export path accepts the manifest's `checkpoint` or `checkpoint_path` field, while the package path preserves support for saved checkpoint formats and prediction bundles.
