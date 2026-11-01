# MVP Implementation Runbook

## Operating protocol

The V0 workflow uses one frozen, prompt-grouped 80/20 split. The Data Factory searches 1,000 deterministic grouped candidates and selects the split with the closest validation row fraction and class distribution to the complete training CSV. The selection uses IDs, prompt groups, and labels only. It verifies that every label appears in both partitions and that no prompt group crosses the boundary.

For the current 2,699-row dataset, the split is 2,159 train rows and 540 validation rows across 120 and 30 prompt groups respectively. Each of the three models fits once on those 2,159 rows. All model and blend scores use the same 540 validation rows. Since the validation rows are reused for model comparisons, scores are development evidence and can become optimistic; there is no separate holdout score in this protocol.

## Clean restart

From the repository root, run:

```bash
python -m src.pipeline restart
```

This command clears generated contents under `data/processed/` and `artifacts/`, deletes `experiments/registry.sqlite` plus SQLite WAL/SHM files, then runs the Data, Feature, Model, and Ensemble factories in order. It retains `data/raw/` and the Hugging Face model cache. The registry is rebuilt from sequence 0001. By default all configured models are trained; use `--models svc ultrabert380` to choose a subset, and `--seed 2027` to create a different deterministic split.

## Individual factories

```bash
python -m src.pipeline data prepare --config configs/data/official_v0.yaml
python -m src.pipeline features build --dataset v0
python -m src.pipeline models train --model svc --dataset v0
python -m src.pipeline models train --model ultrabert380 --dataset v0
python -m src.pipeline models train --model qwen06 --dataset v0
python -m src.pipeline ensemble evaluate --dataset v0
```

All model configs must point to the intended model and an immutable Hugging Face revision. The Qwen model remains configurable through `configs/models/qwen06.yaml`; LoRA target module names must match the chosen architecture. The ensemble factory predicts the train partition with each fitted checkpoint, chooses a regularized weight vector using only those train labels/predictions, freezes it, and reports its validation score alongside the equal-weight score. Those train predictions are in-sample, so the train score is descriptive; only the validation rows compare transfer of the weights.

## Artifacts and dashboard

- `data/processed/dataset_v0.parquet`: canonical sorted train data.
- `data/processed/splits_v0.parquet`: ID, prompt group, label, and frozen `train` or `validation` assignment.
- `data/processed/dataset_profile.json`: source hashes, class counts, partition counts, seed, candidate count, and split hash.
- `artifacts/features/`: train-fitted vectorizer cache and feature manifest.
- `artifacts/runs/`: model checkpoints.
- `artifacts/predictions/`: per-model validation predictions/manifests and blend comparisons.
- `experiments/registry.sqlite`: experiment order, descriptions, configs, metrics, runtime, and artifact pointers.

Run the dashboard with `streamlit run dashboard/app.py`. It reports the source split and experiment sequence; model validation metrics should not be interpreted as an independent final estimate after repeated comparisons.

## Final inference model

After selecting a recipe, `python -m src.pipeline models fit-final --model <name> --dataset v0` trains that recipe on all labelled rows and writes an inference checkpoint/manifest. This full-data fit is separate from validation scoring and does not create another validation result.
