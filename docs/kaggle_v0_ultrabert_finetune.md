# Kaggle-only UltraBERT V0 fine-tune

## Run

- Experiment: `kaggle_only_ultrabert380_v0_fromb5a07dd9_split9803515f_cfg77cc6af7`
- Source checkpoint: `ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgb9e735ff_code81bce4a` (full V2.14 run, selected epoch 4)
- Training: 2,159 rows from the frozen V0 training partition; validation: 540 prompt-grouped V0 holdout rows.
- Initialization: source weights only, with a fresh optimizer and linear schedule.
- Parameters: LR 2e-6, up to 10 epochs, no warmup, AdamW, weight decay 0, patience 2, max length 581.
- Batch probe: requested 8, selected microbatch 152, gradient accumulation 1; peak probe VRAM 15.15 GiB.
- Selected checkpoint: epoch 4 at `/home/zodrak/nlm/artifacts/kaggle_only/runs/kaggle_only_ultrabert380_v0_fromb5a07dd9_split9803515f_cfg77cc6af7/checkpoints/ultrabert380_v0_finetuned`.

## Epoch metrics

| Epoch | Learning rate | Train loss | Raw validation loss | Accuracy | Balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.813e-06 | 1.8875 | 1.5901 | 57.9630% | 57.9630% |
| 2 | 1.613e-06 | 0.9048 | 1.4415 | 57.5926% | 57.5926% |
| 3 | 1.413e-06 | 0.4713 | 1.4181 | 58.3333% | 58.3333% |
| 4 | 1.213e-06 | 0.2195 | 1.4516 | 58.7037% | 58.7037% |
| 5 | 1.013e-06 | 0.1000 | 1.5120 | 57.9630% | 57.9630% |
| 6 | 8.133e-07 | 0.0417 | 1.5752 | 58.3333% | 58.3333% |

## Comparison on the same 540 rows

| Checkpoint | Correct | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|---:|
| Full V2.14 source, saved full-V0 predictions filtered to holdout | 308 / 540 | 57.0370% | 57.0370% | 1.9443 |
| V0 fine-tune, selected epoch 4 | 317 / 540 | 58.7037% | 58.7037% | 1.4515 |
| Change | +9 | +1.6667% | +1.6667% | -0.4927 |

The selected fine-tune corrected 47 source errors and regressed 38 source-correct examples, for a net gain of 9 correct rows.

## Interpretation and Kaggle

This is a development-set diagnostic, not an independent generalization estimate: the source checkpoint was selected using all 2,699 V0 examples, including these 540 holdout rows. The fine-tune itself trained on only the other 2,159 V0 examples and selected epochs on these 540 rows.

- Kaggle model package: `artifacts/kaggle/ultrabert380-v2-14-v0-finetuned-model`
- Kaggle notebook package: `artifacts/kaggle/ultrabert-v2-14-v0-ft-name-that-llm-kernel`
- Kaggle model instance: `zodrak/ultrabert380-v2/PyTorch/ultrabert380-v2-14-v0-finetuned/1` (instance ID 781940).
- Kaggle notebook: `zodrak/ultrabert-v0-fine-tune-name-that-llm`, version 2 completed; output validated as 3 rows in test order with exactly `id,model`.
- Competition submission: `58276356`, currently `PENDING`; public score is not available yet.
- Notebook version 1 failed to locate the mounted model; version 2 completed successfully. Version 3 is queued as a cycle-safe path-search revision and is left running in the background.
- This Kaggle-only experiment is intentionally excluded from Trust List 2 and the active checkpoint registry.
