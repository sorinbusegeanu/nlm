# E42 — Qwen3.5-4B-Base QLoRA layer-removal comparison

This training block contains four independent runs, queued after the E41 Qwen3.5-0.8B layer-removal queue succeeds. The pre-existing E42 E13-B/E38/E39 Kaggle router retains its catalog identity and is unchanged; this training experiment is registered as `E42-QWEN35-4B-QLORA`.

## Model and recipe

Base model: `Qwen/Qwen3.5-4B-Base`, pinned revision `1001bb4d826a52d1f399e183466143f4da7b741b`. It is the pretrained-only checkpoint; the post-trained `Qwen/Qwen3.5-4B` cached for E19 is a different model and is not used here. The base weights are being cached locally for offline queue execution.

All variants train independently from the pretrained base with QLoRA4 and the N0137 recipe: V2.14 training, all 2,699 V0 rows for validation, 10 epoch maximum, patience 2, best validation balanced-accuracy checkpoint, requested physical batch 4 with automatic calibration, gradient accumulation 1, LR `5e-5`, AdamW, linear scheduler, 2% warmup, weight decay 0, LoRA rank 16/alpha 32/dropout 0.05, SDPA, chat template, and max length 562. Each run independently probes its training and inference batch size under the 95% VRAM safety limit. Inference keeps adaptive OOM batch reduction.

## Variants

The Qwen3.5-4B backbone has 32 hybrid layers. Each pruned variant removes 11 layers and retains the other 21, in original order.

| Variant | Removed original indices | Retained count |
|---|---|---:|
| E42 baseline | none | 32 |
| E42-A | 21–31 (final third) | 21 |
| E42-B | 10–20 (middle third) | 21 |
| E42-C | distributed: 2, 5, 8, 11, 14, 16, 19, 22, 25, 28, 31 | 21 |

The shared pruning helper updates hybrid layer-type metadata and attention/cache indices. Each selected checkpoint stores its exact original indices and modified architecture configuration.

## Queue order and status

1. Full 32-layer QLoRA baseline.
2. E42-A.
3. E42-B.
4. E42-C.

The queue worker waits for E41's queue to report success and for GPUs to be idle. It stops on the first failed job. Queue state and frozen config snapshots are under `artifacts/training_queues/e42_qwen35_4b_qlora/`.

## Results

| Variant | V0 balanced accuracy | Δ vs baseline (pp) | Training time | Safe train batch | Safe inference batch | V0 inference time | Peak training/inference VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| E42 baseline | pending | — | pending | pending | pending | pending | pending |
| E42-A | pending | pending | pending | pending | pending | pending | pending |
| E42-B | pending | pending | pending | pending | pending | pending | pending |
| E42-C | pending | pending | pending | pending | pending | pending | pending |

V0 is used for checkpoint selection and final comparison, so these are development metrics rather than independent validation results. After all four jobs succeed, the queue runs full-V0 inference and fills this table with measured values.
