# Name That LLM

Reproducible experiments for identifying the model that generated a passage. The MVP specification is in [`docs/mvp_design.md`](docs/mvp_design.md) and [`docs/mvp_implementation.md`](docs/mvp_implementation.md).

## Local setup

Use Python 3.10 or newer. Install dependencies on the computer that will run the experiments:

```bash
uv sync --locked --extra dev --extra dashboard
source .venv/bin/activate
```

For transformer training and inference, install a PyTorch build suited to the local hardware using the official PyTorch selector. Then install the remaining locked neural dependencies with `uv sync --locked --extra neural --extra dev --extra dashboard --no-install-package torch`. The lock includes a generic torch package for resolution; skip it when installing so the hardware-specific build remains selected.

Before neural training, edit the selected model config and set an immutable Hugging Face checkpoint revision. `ultrabert150` requires an UltraBERT model ID and revision; the pipeline intentionally fails with an actionable error while either is missing. Qwen defaults to `Qwen/Qwen3-0.6B-Base`, but also requires an immutable revision.

## V0 workflow

Prepare the canonical dataset, a frozen prompt-grouped 20% holdout, five development folds, and the data profile:

```bash
python -m src.pipeline data prepare --config configs/data/official_v0.yaml
```

Build and cache fold-fitted features, then train each model on the same folds:

```bash
python -m src.pipeline features build --dataset v0
python -m src.pipeline models train --model svc --dataset v0
python -m src.pipeline models train --model ultrabert150 --dataset v0
python -m src.pipeline models train --model qwen06 --dataset v0
```

Feature ablations use `--features-config configs/features/formatting.yaml` or `configs/features/stylometry.yaml`; every recipe receives a separate experiment ID. The SVC defaults to character 3–5 gram and word 1–2 gram TF-IDF. All vectorizers and scalers are fit inside each training fold.

Evaluate aligned model outputs and ensembles on development OOF predictions:

```bash
python -m src.pipeline ensemble evaluate --dataset v0
```

After selecting a frozen recipe, run the untouched holdout once, then fit the selected model on all labelled rows for inference:

```bash
python -m src.pipeline models evaluate-holdout --model svc --dataset v0
python -m src.pipeline models fit-final --model svc --dataset v0
```

The holdout command writes a dataset-wide guard and refuses subsequent holdout evaluation runs. Do not use holdout results to tune models.

Export generic offline predictions using the observed `test.csv` schema (`id,text`) and `id,model` output:

```bash
python -m src.pipeline export --model svc --checkpoint artifacts/checkpoints/svc_v0_final.json --test data/raw/test.csv --output submission.parquet
```

For neural exports, pass the final manifest JSON as `--checkpoint` and choose the matching model. Neural inference loads local model files only. Competition-specific schema, internet-off Kaggle packaging, and runtime limits must be checked against competition rules before submission.

## Results and tests

Run tests with `python -m unittest discover -s tests -v`. Neural adapter tests create tiny local models and skip when the optional neural dependencies are absent. Run the dashboard with `streamlit run dashboard/app.py`.

Raw competition inputs are in `data/raw/`. Processed Parquet data, feature caches, model checkpoints, predictions, manifests, and the SQLite registry are generated under `data/processed/`, `artifacts/`, and `experiments/`; generated outputs are ignored by Git. Each experiment manifest records input and split hashes, model revision, configuration, code revision, fold IDs, calibrated probabilities, raw scores, runtime, and artifact sizes.
