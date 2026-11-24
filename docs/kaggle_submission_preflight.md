# Kaggle packaging and submission preflight

All future NLM Kaggle notebook releases should go through `scripts/kaggle_preflight.py` before a kernel is pushed and before any competition submission is sent.

## What the preflight checks

### Before upload

- Kaggle kernel metadata fields, code file and versioned model-source references.
- Notebook JSON, cell syntax and likely undefined global names across cells.
- Strict JSON in package manifests and metadata.
- With `--remote`, availability of every attached dataset/model source, base model `config.json` and tokenizer files. Model files may be nested below a single root folder; preflight must identify a unique config/tokenizer pair and Kaggle runtime preflight must load that exact directory through Transformers.
- Manifest-referenced checkpoint asset filenames and sizes against the remote dataset inventory.
- Explicitly declared scikit-learn compatibility `.joblib` assets are loaded locally where present; known version-drift repairs are recorded and probability APIs are smoke-tested. Other checkpoint `.joblib` files still receive manifest presence/size checks and are loaded by the inference runtime at the appropriate point.
- Check checkpoint-relative paths against the inference worker's bundle root. Paths for packaged additions must resolve under `checkpoints/<checkpoint-id>/` unless they already include that prefix.

The push command inserts an idempotent first notebook cell. On Kaggle it runs before inference and checks mounted dataset/model paths, expected checkpoint files and sizes, required Python distributions, configured exact package versions, and whether each base model config/tokenizer (and marked processor) loads with the installed Transformers runtime. It writes `kaggle_preflight.json` into `/kaggle/working` and stops the run with the full list of failures. It also routes notebook `json.dump(s)` calls through a JSON-safe converter for NumPy arrays/scalars and paths, avoiding a late serialization error.

For assets listed in `sklearn_compatibility_assets`, runtime preflight loads each artifact before model inference, captures scikit-learn version warnings, applies the known `LogisticRegression.multi_class="auto"` compatibility repair when missing, and smoke-tests `predict_proba`. Subsequent `joblib.load` calls reuse repaired objects. Broad checkpoint manifests may list joblib checkpoints that need a project runtime installed later; these are not deserialized in the early gate.

### Before submission

For Name That LLM, the competition submission file must be named exactly `submission.parquet`. A variant saved only as `submission_e28_b.parquet` can pass inference checks but Kaggle may reject the competition submission with HTTP 400. Put the selected variant under the canonical filename in the completed kernel version before submitting.

The submit command fetches output from the exact completed kernel version, requires a successful `kaggle_preflight.json` runtime report, caller-supplied expected test IDs and canonical labels, validates the exact `id,model` schema, uniqueness, nulls, label vocabulary and row order, and requires the canonical filename before submitting. For multiple variants from one completed kernel, validate all variant files against the same ordered test IDs and labels, then copy each selected variant into its own temporary directory as `submission.parquet` before calling Kaggle. Never submit an alternate basename such as `submission_e19_f_a.parquet` directly.

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
- Broad manifest `.joblib` audit attempted to deserialize the E13-B N0082 SVC checkpoint before its packaged project `src` runtime was installed (`ModuleNotFoundError: No module named 'src'`). Only explicit `sklearn_compatibility_assets` are now loaded at startup (the external grouper); all other manifest assets still receive existence/size checks and are loaded by inference after runtime setup.
- A prepare-time compatibility allowlist was lost when the push command regenerated the startup gate. The allowlist is now persisted in kernel metadata and tested across a second preflight embedding, so push preserves the same narrowed audit.
- Missing or mismatched checkpoint bundle paths: added checkpoints such as N0030 had `N0030` in the manifest while worker assets were under `checkpoints/N0030/`.
- Noncanonical output filenames: variant files like `submission_e28_b.parquet` were valid parquet outputs but were rejected by the code-submission API; use `submission.parquet`.
- `src` package shadowing that imported the wrong `inference.py` and hid required API arguments.
- Grouper assets stored under `deployment/` while code assumed the dataset root.
- JSON wrapper rewriting `json.dumps` inside embedded worker source strings; rewrite executable tokens only.
- Missing Qwen processor metadata, unattached Llama base model, incompatible optional TorchAO, unregistered `smollm3_text_base`, NumPy values passed directly to JSON serialization, and E30's undefined `RUNTIME` notebook variable.
- Kaggle's maximum of two concurrent GPU sessions, which can reject additional kernel launches.
- E30 v6 launch was rejected because E28 v10/v11 occupied Kaggle's two-session allowance. Treat this as capacity rejection (not a submitted version); check active/queued kernel statuses and retry only after a slot frees.
- Repeated status checks across kernel versions triggered Kaggle HTTP 429 rate limiting. Use one status refresh per interval and wait after a 429; do not poll each kernel/version in parallel.
- Kernel-title/URL-slug mismatch when creating the E31 kernel.
- Nested Kaggle model payloads such as `Qwen3.5-4B/config.json` and a direct-root tokenizer directory such as EmbeddingGemma can defeat root-only file checks. Find and verify a unique config/tokenizer pair, then use that resolved path for both AutoConfig/AutoTokenizer and AutoProcessor smoke tests.
- Multi-variant submission from one notebook output must fetch every named variant, validate identical exact test-ID order, and stage each independently under the canonical basename before its competition submit API call.

These failures show why a completed kernel alone is insufficient. Preflight must inspect the actual package and runtime contract; submission completion additionally requires a successful competition submission and a refreshed submission status.

`scripts/prepare_e30_kaggle.py` now runs local package validation and emits the runtime gate whenever E30 is packaged. The currently queued E30 kernel version was pushed before this change; it does not contain the new gate.
