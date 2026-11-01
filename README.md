# Name That LLM

Reproducible experiments for identifying the model that generated a passage. The MVP specification is in [`docs/mvp_design.md`](docs/mvp_design.md) and [`docs/mvp_implementation.md`](docs/mvp_implementation.md).

## Local setup

Use Python 3.10 or newer. Install dependencies on the computer that will run the experiments:

```bash
uv sync --locked --extra dev --extra dashboard
source .venv/bin/activate
```

For transformer training and inference, install a PyTorch build suited to the local hardware using the official PyTorch selector. Then install the remaining locked neural dependencies with `uv sync --locked --extra neural --extra dev --extra dashboard --no-install-package torch`. The lock includes a generic torch package for resolution; skip it when installing so the hardware-specific build remains selected.

Before neural training, set an immutable Hugging Face checkpoint revision in the selected model config. `ultrabert380` uses `dmis-lab/UltraBERT-380M` with pinned remote model code. The Qwen LoRA trainer defaults to [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B); choose another compatible causal LM by changing `checkpoint` and `revision` in `configs/models/qwen06.yaml`. Adjust `lora_target_modules` when the selected architecture uses different attention projection names. The neural extra pins Transformers to 5.16–5.19 for compatibility with the CUDA environment's vLLM; UltraBERT loads its pinned custom code through `trust_remote_code=True`.

## V0 workflow

Prepare the canonical dataset, a frozen prompt-grouped 20% holdout, five development folds, and the data profile:

```bash
python -m src.pipeline data prepare --config configs/data/official_v0.yaml
```

Build and cache fold-fitted features, then train each model on the same folds:

```bash
python -m src.pipeline features build --dataset v0
python -m src.pipeline models train --model svc --dataset v0
python -m src.pipeline models train --model ultrabert380 --dataset v0
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

Build a portable model/code bundle for the included generic Kaggle notebook:

```bash
python -m src.pipeline package-kaggle --manifest artifacts/checkpoints/svc_v0_final.json --output artifacts/kaggle/model_package.zip
```

Attach `model_package.zip` and the competition input data as Kaggle input datasets. Open [`notebooks/kaggle_inference.ipynb`](notebooks/kaggle_inference.ipynb), update its input paths, and disable Internet in Kaggle notebook settings. The notebook writes `submission.parquet` using the observed `id,model` schema. Verify the actual competition schema, filenames, memory, and runtime limits before submitting.

## Results and tests

Run tests with `python -m unittest discover -s tests -v`. Neural adapter tests create tiny local models and skip when the optional neural dependencies are absent. Run the dashboard with `streamlit run dashboard/app.py`.

Raw competition inputs are in `data/raw/`. Processed Parquet data, feature caches, model checkpoints, predictions, manifests, and the SQLite registry are generated under `data/processed/`, `artifacts/`, and `experiments/`; generated outputs are ignored by Git. Each experiment manifest records input and split hashes, model revision, configuration, code revision, fold IDs, calibrated probabilities, raw scores, runtime, and artifact sizes.
