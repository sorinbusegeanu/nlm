# Kaggle packaging and submission preflight

All future NLM Kaggle notebook releases should go through `scripts/kaggle_preflight.py` before a kernel is pushed and before any competition submission is sent.

## What the preflight checks

### Before upload

- Kaggle kernel metadata fields, code file and versioned model-source references.
- Notebook JSON, cell syntax and likely undefined global names across cells.
- Strict JSON in package manifests and metadata.
- With `--remote`, availability of every attached dataset/model source, base model `config.json` and tokenizer files.
- Manifest-referenced checkpoint asset filenames and sizes against the remote dataset inventory.

The push command inserts an idempotent first notebook cell. On Kaggle it runs before inference and checks mounted dataset/model paths, expected checkpoint files and sizes, required Python distributions, configured exact package versions, and whether each base model config/tokenizer (and marked processor) loads with the installed Transformers runtime. It writes `kaggle_preflight.json` into `/kaggle/working` and stops the run with the full list of failures. It also routes notebook `json.dump(s)` calls through a JSON-safe converter for NumPy arrays/scalars and paths, avoiding a late serialization error.

### Before submission

The submit command fetches output from the requested kernel version, requires a successful `kaggle_preflight.json` runtime report, caller-supplied expected test IDs and canonical labels, validates every output's exact `id,model` schema, uniqueness, nulls, label vocabulary and row order, confirms all output variants share the same ordered IDs, then submits the files from that exact kernel version.

## Commands

Check a package locally and verify its attached Kaggle inventories:

```bash
python -m scripts.kaggle_preflight check \
  --kernel-dir artifacts/kaggle/e30-kernel \
  --remote
```

Push through the preflight gate. The default saves the notebook version without executing it; add `--run` only when the run is intended:

```bash
python -m scripts.kaggle_preflight push \
  --kernel-dir artifacts/kaggle/e30-kernel \
  --strict-version scikit-learn=1.9.1 \
  --processor-source owner/model/Transformers/instance/1 \
  --run
```

Supply strict package versions when serialized artifacts require them. Mark only model sources that actually need `AutoProcessor`. The runtime preflight always checks `AutoConfig` and `AutoTokenizer` for attached model sources.

After the exact kernel version finishes, fetch, validate and submit one or more outputs:

```bash
python -m scripts.kaggle_preflight submit \
  --kernel owner/kernel-slug \
  --version 12 \
  --competition name-that-llm \
  --output-dir artifacts/kaggle/submission-v12 \
  --file submission.parquet \
  --file submission_e19_f_a.parquet \
  --file submission_e19_f_c.parquet \
  --expected-ids data/raw/test.csv \
  --labels-json artifacts/analysis/e30/e30_b_manifest.json \
  --message 'E30 v12 {file}'
```

The Kaggle API token must already be configured. After submission, refresh the competition submission list and report the submission ID, status and public score separately. A blank score is not a score.

## Failure cases this is intended to catch

The archived NLM runs failed on local/Kaggle version drift (`LogisticRegression.multi_class`; scikit-learn 1.9.1 artifacts loaded in Kaggle 1.6.1), missing Qwen processor metadata, an unattached Llama base model, incompatible optional TorchAO, an unregistered `smollm3_text_base` model type, NumPy values passed directly to JSON serialization, and E30's undefined `RUNTIME` notebook variable. These failures happened before or during setup despite reusing known checkpoint files. The gate checks package structure and the actual Kaggle mounts/runtime rather than treating checkpoint identity as sufficient evidence of readiness.

`scripts/prepare_e30_kaggle.py` now runs local package validation and emits the runtime gate whenever E30 is packaged. The currently queued E30 kernel version was pushed before this change; it does not contain the new gate.
