# V3 contributor-label training factory

The factory evaluates the fixed ranking queue in ascending rank order. Each
candidate episode fits SVC, CatBoost + Stylometry, and UltraBERT-380M from
scratch on V3.0-BASELINE, previously accepted contributor-label rows, and the
current contributor-label subset. Rows are deduplicated by response ID. Prompt
IDs and prompt groups do not participate in training selection, router fitting,
fold construction, scoring, or acceptance.

V0 and V0-EXT have equal importance for checkpoint selection and acceptance.
Five deterministic, row-level folds are stratified by dataset and label. The
reliability router uses only out-of-fold predictions and fold-local reliability
statistics; its outer-fold scores are pooled as the arithmetic mean of V0 and
V0-EXT balanced accuracy. A candidate is accepted only on a strict improvement
over the last accepted mean score, before rounding.

V0-EXT remains validation-only and is never used for training. Its use for
checkpoint selection is a scoped exception for this explicitly requested
factory, recorded in `configs/datasets.yaml` and the experiment catalogue.

## Model configuration

| Model | Device | Configuration |
|---|---|---|
| SVC | CPU | Existing `configs/models/svc.yaml`, fresh fit, seed 2026 |
| CatBoost + Stylometry | CPU | Existing feature recipe, 800 iterations, depth 7, learning rate 0.05, seed 2026, early stopping 100; validation rows receive equal dataset-label weights |
| UltraBERT-380M | GPU | Pinned original base checkpoint, 20 epochs maximum, patience 2, learning rate 8e-6, linear schedule, automatic physical batch selection, existing gradient accumulation setting, validation and inference batch size 600, seed 2026 |

The UltraBERT effective batch remains selected physical batch multiplied by
configured gradient accumulation. CUDA OOM recovery is provided by the existing
encoder training wrapper. The SVC and CatBoost CPU workers run concurrently with
the UltraBERT GPU worker in separate processes and isolated output directories.
The parent waits for all workers to finish before recording an episode failure;
it does not terminate a worker process.

## Outputs and resume behavior

Persistent state and result CSVs are stored under `artifacts/analysis/v3/`.
State updates are atomic. Completed episode decisions are committed once, and
the fixed queue rank resumes after interruption. Failed fits do not count as
rejections and are retried at the same rank. One temporary episode directory,
`artifacts/analysis/v3/tmp/`, contains generated model and feature artifacts;
it is cleared before a retry/next episode and after successful completion.

Launch with:

```bash
python scripts/run_v3_contributor_label_training_factory.py
```

No factory training was started as part of implementation. Focused tests use
mock episode runners and do not allocate a GPU.
