# Full V0 Validation Scope

For an external-only dataset, the training factory uses every row in that
dataset for gradient fitting and all 2,699 official V0 rows for validation:

```bash
python -m src.pipeline models train \
  --model ultrabert380 \
  --dataset v1
```

This is the default for profiles marked `training_only`. The optional
`--validation-scope full_v0` makes the choice explicit. The historical
`frozen_v0_540` behavior remains for non-external-only datasets and older runs;
it cannot be requested for an external-only dataset.

The factory rejects a training dataset that overlaps V0 by row ID, normalized
response text, prompt ID, or normalized canonical prompt. The full V0 set is
passed to the model trainer for per-epoch validation, early stopping, and
checkpoint selection by **balanced accuracy**. Calibration remains fitted on
training data only. For CatBoost, validation predictions are scored after each
tree iteration and the tree count with the highest balanced accuracy is kept.
Models without epochs (SVC and frozen-backbone readouts) are fitted once and
scored on V0.

The generated scope artifacts are kept separate from the original dataset and
its legacy split:

- `data/processed/dataset_profile_<dataset>_full_v0.json`
- `data/processed/splits_<dataset>_full_v0.parquet`
- `artifacts/predictions/<experiment>_full_v0.parquet`

The profile and manifest contain row counts and hashes for the complete
training and validation sets, plus overlap counts. The manifest records the
per-epoch metrics and selected checkpoint metric. Mixed datasets that contain
V0 training rows cannot be used in this mode; use their external-only source
dataset, such as `v2_openpool` instead of mixed `v2`. Full-V0 runs are not
automatically added to the existing 540-row prediction roster or checkpoint
catalog.
