# Kaggle packaging and submission preflight

All future NLM Kaggle notebook releases should go through `scripts/kaggle_preflight.py` before a kernel is pushed and before any competition submission is sent.

## What the preflight checks

### Before upload

- Kaggle kernel metadata fields, code file and versioned model-source references.
- Notebook JSON, cell syntax and likely undefined global names across cells.
- Strict JSON in package manifests and metadata.
- With `--remote`, availability of every attached dataset/model source, base model `config.json` and tokenizer files.
- Manifest-referenced checkpoint asset filenames and sizes against the remote dataset inventory.
- Manifested scikit-learn `.joblib` assets are loaded locally where present; known version-drift repairs are recorded and probability APIs are smoke-tested. The Kaggle startup gate repeats this against the actual mounted artifacts before inference.
- Check checkpoint-relative paths against the inference worker's bundle root. Paths for packaged additions must resolve under `checkpoints/<checkpoint-id>/` unless they already include that prefix.

The push command inserts an idempotent first notebook cell. On Kaggle it runs before inference and checks mounted dataset/model paths, expected checkpoint files and sizes, required Python distributions, configured exact package versions, and whether each base model config/tokenizer (and marked processor) loads with the installed Transformers runtime. It writes `kaggle_preflight.json` into `/kaggle/working` and stops the run with the full list of failures. It also routes notebook `json.dump(s)` calls through a JSON-safe converter for NumPy arrays/scalars and paths, avoiding a late serialization error.

For manifested `.joblib` assets, runtime preflight loads each artifact before model inference, captures scikit-learn version warnings, applies the known `LogisticRegression.multi_class="auto"` compatibility repair when missing, and smoke-tests `predict_proba`. Subsequent `joblib.load` calls reuse the repaired objects. This prevents the recurring scikit-learn 1.9-to-1.6 failure from first appearing after GPU inference.

### Before submission

For Name That LLM, the competition submission file must be named exactly `submission.parquet`. A variant saved only as `submission_e28_b.parquet` can pass inference checks but Kaggle may reject the competition submission with HTTP 400. Put the selected variant under the canonical filename in the completed kernel version before submitting.

The submit command fetches output from the exact completed kernel version, requires a successful `kaggle_preflight.json` runtime report, caller-supplied expected test IDs and canonical labels, validates the exact `id,model` schema, uniqueness, nulls, label vocabulary and row order, and requires the canonical filename before submitting. For multiple variants, select one per kernel version and make that variant `submission.parquet`.

“Submit” is complete only after the Kaggle competition submit API accepts the prediction. A saved or completed kernel, a passing preflight, or a local parquet file is not a competition submission. Refresh `kaggle competitions submissions` afterward and record the submission ID, status, and public score separately.

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
  --expected-ids data/raw/test.csv \
  --labels-json artifacts/analysis/e30/e30_b_manifest.json \
  --message 'E30 v12 {file}'
```

The Kaggle API token must already be configured. After submission, refresh the competition submission list and report the submission ID, status and public score separately. A blank score is not a score.

## Failure cases this is intended to catch

The checklist now covers these observed failures:

- scikit-learn 1.9.1 artifacts loaded in Kaggle 1.6.1: missing `LogisticRegression.multi_class`, caught and repaired before inference.
- Missing or mismatched checkpoint bundle paths: added checkpoints such as N0030 had `N0030` in the manifest while worker assets were under `checkpoints/N0030/`.
- Noncanonical output filenames: variant files like `submission_e28_b.parquet` were valid parquet outputs but were rejected by the code-submission API; use `submission.parquet`.
- `src` package shadowing that imported the wrong `inference.py` and hid required API arguments.
- Grouper assets stored under `deployment/` while code assumed the dataset root.
- JSON wrapper rewriting `json.dumps` inside embedded worker source strings; rewrite executable tokens only.
- Missing Qwen processor metadata, unattached Llama base model, incompatible optional TorchAO, unregistered `smollm3_text_base`, NumPy values passed directly to JSON serialization, and E30's undefined `RUNTIME` notebook variable.
- Kaggle's maximum of two concurrent GPU sessions, which can reject additional kernel launches.
- Kernel-title/URL-slug mismatch when creating the E31 kernel.

These failures show why a completed kernel alone is insufficient. Preflight must inspect the actual package and runtime contract; submission completion additionally requires a successful competition submission and a refreshed submission status.

`scripts/prepare_e30_kaggle.py` now runs local package validation and emits the runtime gate whenever E30 is packaged. The currently queued E30 kernel version was pushed before this change; it does not contain the new gate.
