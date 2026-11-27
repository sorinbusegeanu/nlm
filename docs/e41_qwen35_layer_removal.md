# E41 — Qwen3.5-0.8B layer removal

This is the queued E41 LoRA comparison for Qwen3.5-0.8B-Base. The historical E41 equal-probability E38/E39 blend remains documented in [docs/e41.md](e41.md); this experiment has a distinct catalogue ID because E41 is already in use.

## Fixed recipe

All three variants start independently from the pinned pretrained revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`; none resumes N0137's adapter. They reuse N0137's LoRA training recipe: V2.14 training data, full V0 validation (2,699 rows), 10-epoch maximum, patience 2, best validation balanced-accuracy checkpoint, batch size 4 with automatic probing, gradient accumulation 1, LR `5e-5`, AdamW, linear schedule, 2% warmup, weight decay 0, LoRA rank 16/alpha 32/dropout 0.05, SDPA, chat template and maximum sequence length 562. Inference batch size is independently calibrated up to 1,024 using the same V0 rows, with OOM retry retained for later inference.

| Variant | Removed original layer indices | Retained original layer indices | Count |
|---|---|---|---:|
| E41-A | 16–23 (final third) | 0–15 | 16 |
| E41-B | 8–15 (middle third) | 0–7, 16–23 | 16 |
| E41-C | 2, 5, 8, 11, 14, 17, 20, 23 (distributed) | all remaining indices in original order | 16 |

The pruning helper updates Qwen3.5 hybrid `layer_types`, block metadata, hidden-layer count and retained attention/cache indices. The exact retained indices and modified architecture config are saved in each checkpoint.

## Queue

The sequential E41 queue waits until the V3.1 contributor factory reports `Complete`, its persisted rank state has no active contributor, and GPUs are idle. It then runs E41-A, E41-B and E41-C in order. A failed job stops the queue. Training has not started as part of setup; the worker is waiting on V3.1.

Queue state: `artifacts/training_queues/e41_qwen35_layer_removal/queue.json`.

## Results

| Model | V0 balanced accuracy | Δ vs N0137 (pp) | Training duration | Effective train batch | Effective inference batch | Inference duration | Peak train/inference VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| N0137 baseline | 54.7122% | — | 63013.68 s | 4 | 8 | pending benchmark | 13.67 GB train / pending inference |
| E41-A | pending | pending | pending | pending | pending | pending | pending |
| E41-B | pending | pending | pending | pending | pending | pending | pending |
| E41-C | pending | pending | pending | pending | pending | pending | pending |

After all three jobs succeed, the queue runs the full-V0 evaluator and fills this table with measured results. These V0 results are development results because V0 is used for checkpoint selection.

## Commands

The queue worker is `scripts/run_e41_qwen35_layer_removal_queue.py`; post-training measurement is `scripts/evaluate_e41_qwen35_layer_removal.py`.
