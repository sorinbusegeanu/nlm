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
- Read the saved per-checkpoint/model batch calibration before packaging. Use a measured maximum stable batch when the model revision, tokenizer, input length and inference settings match. Do not substitute a small ad hoc probe or arbitrary lower default. For any auxiliary model without a matching calibration, calibrate that exact model on the longest-tokenized V0 inputs using the production prompt, input cap and output-token cap; save the attempted/successful batch ladder, peak VRAM and throughput. Record whether a value is a tested batch or a measured maximum, and have the Kaggle runtime read the saved calibration while keeping OOM recovery enabled.

The push command inserts an idempotent first notebook cell. On Kaggle it runs before inference and checks mounted dataset/model paths, expected checkpoint files and sizes, required Python distributions, configured exact package versions, and whether each base model config/tokenizer (and marked processor) loads with the installed Transformers runtime. It writes `kaggle_preflight.json` into `/kaggle/working` and stops the run with the full list of failures. It also routes notebook `json.dump(s)` calls through a JSON-safe converter for NumPy arrays/scalars and paths, avoiding a late serialization error.

For assets listed in `sklearn_compatibility_assets`, runtime preflight loads each artifact before model inference, captures scikit-learn version warnings, applies the known `LogisticRegression.multi_class="auto"` compatibility repair when missing, and smoke-tests `predict_proba`. Subsequent `joblib.load` calls reuse repaired objects. Broad checkpoint manifests may list joblib checkpoints that need a project runtime installed later; these are not deserialized in the early gate.

### Before submission

- Check the competition's current submission mode before generating or uploading files. Some code competitions accept only notebook-originated submissions. For Name That LLM, a direct `kaggle competitions submit -f submission.parquet` attempt returned HTTP 400 `FAILED_PRECONDITION: Submission not allowed: This competition only accepts Submissions from Notebooks`. Do not retry direct file uploads in this mode; submit a validated output from a successful completed Kaggle notebook version. If the user forbids running a notebook and no successful notebook output exists, report that the requested submission is externally blocked instead of claiming success.

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
- Batch sizes must come from the matching calibration artifact (such as the full-V0 checkpoint calibration and the E31 per-checkpoint benchmark). The auxiliary E33 Qwen3.5-4B calibration illustrates the required fallback when model identity differs: the related E20-D Qwen3.5-2B batch-64 result is not interchangeable with the 4B model.
- Validate every runtime module imported by generated notebook code against the mounted source dataset/package, including dynamically inserted experiment modules. E33-F kernel v3 passed generic package preflight and completed both checkpoint workers, then failed at scoring with `ImportError: cannot import name 'e33' from 'src'`: the attached E31 runtime provided `src`, but the E33 deployment dataset omitted `src/e33.py`. Before upload, verify the actual import path and required module file for every `from src import ...`/`import src...` in generated cells, and package an explicit runtime overlay or the missing module. Do not count a completed inference phase as a completed prediction run; require all expected output files before submission.
- Smoke-test the exact text-only generation-input helper against each mounted generative model, not just `AutoProcessor.from_pretrained()`. E33-F kernel v4 loaded Qwen3.5-4B successfully but failed because the Kaggle processor had no chat template. Reuse the working E20-D `TextOnlyQwenProcessor` adapter around `AutoTokenizer`, and require a successful `_generation_inputs()` shape check before launch.
- When retrying a failed Kaggle submission without rerunning a kernel, first confirm that the failed run left complete checkpoint probability caches and that every final prediction variant can be regenerated locally. Reuse the checkpoint-bearing dataset (`zodrak/e31-checkpoints-v001` for E33-F) and verify all 40 roster IDs, cache shapes and test-row order. A failed run that stopped before grouping/scoring produced no submission parquet, so it cannot be submitted as-is.
- A local prediction parquet is not necessarily submit-able to a notebook-only competition. Validate the submission mode and source requirement before uploading; the E33-F local recovery files were refused by the API because this competition requires notebook-originated outputs. The only safe path is a successful notebook version with validated outputs, unless Kaggle changes the competition setting.
- For Qwen3.5 text-only reconstruction, test the tokenizer-level chat template even when the packaged model's `AutoProcessor` loads successfully. A passing processor/tokenizer file preflight alone does not prove `_generation_inputs()` can construct chat inputs.

These failures show why a completed kernel alone is insufficient. Preflight must inspect the actual package and runtime contract; submission completion additionally requires a successful competition submission and a refreshed submission status.

`scripts/prepare_e30_kaggle.py` now runs local package validation and emits the runtime gate whenever E30 is packaged. The currently queued E30 kernel version was pushed before this change; it does not contain the new gate.
