# Prediction factory

The prediction factory evaluates ensembles from saved validation prediction files. It does not retrain base models or fit a validation-trained meta-model.

## Active model roster

`artifacts/predictions/active_model_roster.json` holds one active checkpoint per canonical model. The initial roster is built from the best validation balanced-accuracy checkpoint for every trained model whose predictions align with the frozen V0 validation rows. After a model finishes training:

1. Read its validation manifest and compare it with the current champion for that canonical model.
2. Replace the champion only when the new checkpoint has a higher validation balanced accuracy. A model with no champion is added.
3. Evaluate the active roster against the frozen validation predictions.
4. Record the next ensemble version when either the roster or optimizer protocol changes, preserving all prior entries.

A completed run that does not improve or add a model checkpoint still triggers evaluation, but does not create a duplicate version under the same optimizer protocol. The registry retains each version's candidate roster, selected members, selected weights, grouped out-of-fold metrics, checkpoint filenames, and submitted Kaggle score.

Run a refresh manually with:

```bash
python -m src.pipeline prediction refresh --dataset v0 --data-dir data/processed --artifact-dir artifacts
```

Evaluate the current active roster without changing the registry with:

```bash
python -m src.pipeline prediction evaluate --dataset v0 --artifact-dir artifacts
```

## Forward selection and weights

The factory loads saved validation probabilities and does not load model checkpoints during optimization. It starts from the candidate with the best individual accuracy, then greedily adds the candidate that improves accuracy most. Equal accuracy is resolved by lower multiclass log loss, then fewer models and stable candidate ID ordering. Selection stops when neither accuracy nor log loss improves.

After selection, it compares an equal-weight blend and a nonnegative weighted blend. Weight fitting reuses `fit_weights()`: multiclass log loss is minimized with weights summing to one and L2 regularization toward equal weights. The previous champion is evaluated as a fixed comparison.

Selection and weight fitting are evaluated using five prompt-group folds over saved validation probabilities. Each fold holds out complete prompt groups from both selection and weight fitting. Ordinary accuracy is the primary metric; log loss and ensemble size break ties. The selected final weights are then fitted on all available validation rows for inference. Full-data scores are recorded as exploratory; the grouped out-of-fold scores are the development estimate. The saved checkpoints and active roster may themselves have been selected using this validation set, so these folds do not make the entire comparison independent. Only an untouched competition test score is an independent final estimate.

The existing per-response reliability implementation remains available to read historical manifests and prediction bundles, but new optimization runs do not generate in-sample training predictions or select that strategy.

## Current baseline

The current recorded champion is a nine-checkpoint equal-weight blend with full V0 validation accuracy 0.600. The earlier submitted `v1` ensemble scored 0.546 on Kaggle; the current champion has not been submitted. The updated optimizer records grouped out-of-fold comparisons separately. See [`ensembles.md`](ensembles.md) for checkpoint lists and version history.

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

The package contains only the selected method's required checkpoints, inference code, canonical class ordering, calibration parameters, and saved member weights when applicable. Inference supports equal-weight, optimized-weight, and historical reliability bundles. It accepts the repository's `id,text` test input and writes `id,model` predictions in input row order. The Kaggle notebook expects the bundle ZIP as a private Kaggle dataset.
