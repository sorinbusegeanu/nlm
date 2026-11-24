# Repository workflow instructions

## Experiment catalogue

- Register every new ensemble or evaluation experiment in `configs/ensemble_catalog_v2.json` as part of the initial implementation, before running training or evaluation.
- Use a planned/running status initially, then update the same catalogue entry with its results, status, and artifact/documentation paths when available.
- The dashboard reads this JSON at runtime; keep the catalogue entry complete enough to appear there without a separate user request.

## Kaggle submissions

- “Submit” means submit a prediction to the Kaggle competition from a completed kernel output. Running inference, saving a kernel version, or passing preflight is not a submission; continue through the competition submit API and refresh submission status.
- Before launching inference, verify the exact output filename required by the competition. For Name That LLM code submissions, package the selected prediction as exactly `submission.parquet`; variant filenames alone can pass inference preflight but be rejected by Kaggle.
- Use `scripts/kaggle_preflight.py` before pushing and before submitting. Validate the completed version’s runtime preflight, exact test IDs and order, `id,model` schema, canonical labels, non-null predictions, and the presence of `submission.parquet` in that version’s outputs.
- Resolve serialized scikit-learn compatibility before launch: inspect manifested `.joblib` assets, record version warnings, apply known compatibility fixes, and smoke-test `predict_proba` in preflight.
- Verify checkpoint paths against the actual worker bundle root. For packaged additions, paths must resolve under `checkpoints/<checkpoint-id>/` unless the manifest already supplies that prefix.
- Include inference/import checks for `src` package shadowing, nested deployment asset locations, JSON serialization of NumPy values, and Kaggle’s current GPU-session limit.
- Do not stop after a kernel run or report the work as submitted until the Kaggle competition submission is accepted and its current status has been refreshed. If Kaggle rejects a submission, diagnose and fix the cause, then submit the corrected output; do not substitute “kernel completed” for “prediction submitted.”
- When the user explicitly asks to submit all requested jobs, submit each requested experiment after validation even if its visible sample predictions match another run; do not silently skip a requested job based on a duplicate sample output.
