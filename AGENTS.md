# Repository workflow instructions

## Experiment catalogue

- Register every new ensemble or evaluation experiment in `configs/ensemble_catalog_v2.json` as part of the initial implementation, before running training or evaluation.
- Use a planned/running status initially, then update the same catalogue entry with its results, status, and artifact/documentation paths when available.
- The dashboard reads this JSON at runtime; keep the catalogue entry complete enough to appear there without a separate user request.

## Kaggle submissions

- “Submit” means submit a prediction to the Kaggle competition from a completed kernel output. Running inference, saving a kernel version, or passing preflight is not a submission; continue through the competition submit API and refresh submission status.
- Before launching inference, verify the exact output filename required by the competition. For Name That LLM code submissions, package the selected prediction as exactly `submission.parquet`; variant filenames alone can pass inference preflight but be rejected by Kaggle.
- Use `scripts/kaggle_preflight.py` before pushing and before submitting. Validate the completed version’s runtime preflight, exact test IDs and order, `id,model` schema, canonical labels, non-null predictions, and the presence of `submission.parquet` in that version’s outputs.
- Before packaging inference, read the repository’s precomputed batch calibration for every deployed model (for example `artifacts/analysis/e20d_full_v0_batch_calibration.json` and the per-checkpoint records in `artifacts/analysis/e31/inference_benchmark.json`). Reuse the measured maximum stable batch when the exact model/checkpoint, tokenizer, input cap and generation settings match. Do not replace a matching calibration with a small ad hoc batch probe or a hardcoded conservative default.
- For an auxiliary model with no matching calibration, calibrate that exact model revision using longest-tokenized V0 inputs and the production prompt, input cap and output-token cap; record OOM attempts, successful maximum batch, VRAM peak and throughput in a reproducible artifact. Configure the Kaggle package to read that saved value and retain OOM fallback for T4 differences. A tested batch is not necessarily the maximum stable batch; preserve that distinction.
- Resolve serialized scikit-learn compatibility before launch: inspect manifested `.joblib` assets, record version warnings, apply known compatibility fixes, and smoke-test `predict_proba` in preflight.
- Restrict early `.joblib` compatibility deserialization to explicit runtime artifacts. Broad checkpoint catalog manifests can list a deployed model (such as N0082 SVC) whose required project `src` runtime is installed later; preflight must still check its file and size, but must not load it before that runtime exists. Declare actual preflight targets with `sklearn_compatibility_assets`.
- Persist the `.joblib` compatibility allowlist in `kernel-metadata.json`; the `push` command rebuilds the startup gate, so a prepare-only allowlist otherwise gets discarded and the broad-manifest failure returns.
- Verify checkpoint paths against the actual worker bundle root. For packaged additions, paths must resolve under `checkpoints/<checkpoint-id>/` unless the manifest already supplies that prefix.
- Include inference/import checks for `src` package shadowing, nested deployment asset locations, JSON serialization of NumPy values, and Kaggle’s current GPU-session limit.
- If Kaggle rejects a run with `Maximum batch GPU session count of 2 reached`, do not repeat the push or report a kernel version as launched. Refresh the statuses of all active/queued GPU kernels, wait for a slot, then push once and capture the resulting version number.
- Do not stop after a kernel run or report the work as submitted until the Kaggle competition submission is accepted and its current status has been refreshed. If Kaggle rejects a submission, diagnose and fix the cause, then submit the corrected output; do not substitute “kernel completed” for “prediction submitted.”
- Avoid rapid Kaggle status polling. The status API rate-limits repeated requests (HTTP 429); poll at sensible intervals, and after a 429 wait before retrying rather than issuing parallel status requests.
- When the user explicitly asks to submit all requested jobs, submit each requested experiment after validation even if its visible sample predictions match another run; do not silently skip a requested job based on a duplicate sample output.
