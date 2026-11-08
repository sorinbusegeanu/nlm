# Prediction factory

The prediction factory evaluates ensembles from saved validation prediction files. It does not retrain base models or fit a validation-trained meta-model.

## Active model roster

`artifacts/predictions/active_model_roster.json` holds one active checkpoint per canonical model. The initial roster is built from the best validation balanced-accuracy checkpoint for every trained model whose predictions align with the frozen V0 validation rows. After a model finishes training:

1. Read its validation manifest and compare it with the current champion for that canonical model.
2. Replace the champion only when the new checkpoint has a higher validation balanced accuracy. A model with no champion is added.
3. Evaluate the active roster against the frozen validation predictions.
4. If the roster changed since the last recorded ensemble, append the next version (`v2`, `v3`, ...) to `docs/ensembles.md` before the sequential training queue starts its next model.

A completed run that does not improve or add a champion still triggers an evaluation, but does not create a duplicate ensemble version. The registry retains each version's exact candidate roster, selected members, checkpoint filenames, validation balanced accuracy, and submitted Kaggle score. A new version's submitted score starts at `0.0` and is updated separately after Kaggle submission.

Run a refresh manually with:

```bash
python -m src.pipeline prediction refresh --dataset v0 --data-dir data/processed --artifact-dir artifacts
```

Evaluate the current active roster without changing the registry with:

```bash
python -m src.pipeline prediction evaluate --dataset v0 --artifact-dir artifacts
```

## Evaluation and reliability weights

The factory exhaustively searches every non-empty equal-probability blend of active checkpoints and also evaluates a deterministic per-response reliability selector. Reliability is fitted only from predictions on the V0 training partition. For each model and predicted label it estimates smoothed precision:

```text
trust = (true_positive + 1) / (predicted_count + 2)
```

The selector requires at least 10 training predictions for a label. If no model has enough support for a response, it falls back to the active checkpoint with the highest V0 training balanced accuracy. Validation labels are used to compare the training-fitted selector with equal weighting and to report validation results; they are not used to fit the reliability table.

The primary registry metric is validation balanced accuracy, matching the competition metric. Ordinary accuracy and log loss are also recorded. Among the equal blend and reliability selector, the higher validation balanced accuracy is recorded, with ties preferring fewer members and then stable strategy ordering. Since the candidate method is selected using the same validation set on which it is reported, this score is exploratory and can be optimistic.

## Current baseline

The original `v1` search used 18 model/version candidates and selected a six-checkpoint equal-weight blend. Its V0 validation balanced accuracy was 0.600, and its submitted Kaggle score was 0.546. See [`ensembles.md`](ensembles.md) for the checkpoint list and version history.

## Prediction bundles

```bash
python -m src.pipeline prediction package \
  --manifest artifacts/predictions/prediction_v0_<run-hash>_manifest.json \
  --output artifacts/kaggle/prediction_bundle.zip

python -m src.pipeline prediction export \
  --bundle artifacts/kaggle/prediction_bundle.zip \
  --test data/raw/test.csv \
  --output submission.parquet
```

The package contains only the selected method's required checkpoints, inference code, class ordering, and training-fitted reliability table when applicable. It writes `id,model` predictions in input row order. The Kaggle notebook expects the bundle ZIP as a private Kaggle dataset.
