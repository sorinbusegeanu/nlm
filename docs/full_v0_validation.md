# Full V0 Validation Scope

For an external-only dataset, the training factory uses every row in that
dataset for gradient fitting and all 2,699 official V0 rows for validation:

```bash
python -m src.pipeline models train \
  --model ultrabert380 \
  --dataset v1
```

For V1, this means all 48,798 rows are training rows; there is no 80/20
external split. The same rule applies to V1.1, V2.0, and other supported
external-only datasets. This is the default for those datasets. The optional
`--validation-scope full_v0` makes the choice explicit. The historical
`frozen_v0_540` behavior remains for V0-only training and prior experiment
records; it cannot be requested for an external-only dataset. Mixed datasets
that contain V0 training rows are rejected for new external-data training
runs, because those rows would overlap the full V0 validation set.

The factory rejects a training dataset that overlaps V0 by row ID, normalized
response text, prompt ID, or normalized canonical prompt when both datasets
provide it. The full V0 set is
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
training and validation sets, plus overlap counts. Each epoch evaluation also
writes its full V0 logits and uncalibrated probabilities to a Parquet file under
`artifacts/predictions/<experiment>_epoch_predictions/`; the manifest lists each
file, hash, epoch, class order, and checkpoint path when an epoch checkpoint is
available. The selected checkpoint prediction remains in the main prediction
file. These saved validation predictions can be reused for later per-label
precision and recall calculations without running inference again. The manifest
records the per-epoch metrics and selected checkpoint metric. Mixed datasets that contain
V0 training rows cannot be used in this mode; use their external-only source
dataset, such as `v2_openpool` instead of mixed `v2`. Full-V0 runs are not
automatically added to the existing 540-row prediction roster or checkpoint
catalog.
