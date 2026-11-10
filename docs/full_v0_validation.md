# Training and validation datasets

Training commands take two explicit dataset names:

```bash
python -m src.pipeline models train --model ultrabert380 --dataset v1 --validation-dataset v0
```

The trainer uses every row in `dataset_v1.parquet` for training and every row in
`dataset_v0.parquet` for validation. The V0 artifact contains 2,699 examples.
The default validation dataset is `v0`; `validation_scope` is no longer used.

Before fitting, the pipeline checks that training and validation have the same
label set and do not overlap by row ID, normalized response, prompt group, or
normalized prompt content when available. If a dataset includes V0 rows, it
cannot be trained against V0 validation; choose a disjoint validation dataset.
The pipeline records dataset names, row counts, and content hashes in each
experiment manifest. Neural checkpoints select and early-stop on validation
balanced accuracy. Temperature calibration uses training rows only.

Feature generation takes the same pair of dataset names:

```bash
python -m src.pipeline features build --dataset v1 --validation-dataset v0
```

Historical split assignments and checkpoints remain available for reading and
comparison. They do not change the rows used by new training runs.
