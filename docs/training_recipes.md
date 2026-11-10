# Training recipes and confirmation policy

## Scope and evidence

These are conservative starting values for this repository's 18-class author
classifier, not claimed optimal values. Most upstream recommendations are for
generative supervised fine-tuning (SFT), while this project uses a classifier
head over pooled hidden states. Treat upstream learning rates as a reference
range, then compare epochs using the frozen validation split.

Transformers training uses `adamw_torch`, with its standard Adam betas and
epsilon, the YAML's explicit weight decay, and the scheduler shown below. The
values are explicit in each model YAML, so experiment manifests record the
effective recipe. Batch size can be GPU-probed where `auto_batch_size` is enabled.

| Model/config | Training method | LR | Scheduler / warmup | Notes |
| --- | --- | ---: | --- | --- |
| UltraBERT-380M (`ultrabert380*.yaml`) | Full encoder fine-tune | `8e-6` | Linear / `0` | Repository baseline from prior validation comparisons. Keep the classifier-head setup and low LR; this is empirical project evidence. |
| Microsoft Phi-3.5 Mini Instruct | LoRA plus trainable classification head | `5e-5` | Linear / `0.02` | Pinned Microsoft checkpoint; LoRA targets `qkv_proj` and `o_proj`; locally cached before offline training. |
| Qwen3 0.6B, 1.7B, 4B | LoRA plus trainable classification head | `5e-5` | Linear / `0.02` | Below Qwen's published `3e-4` SFT script because the target task and head differ. Rank 16, alpha 32. |
| EmbeddingGemma 2 | LoRA plus classifier head | `5e-5` | Linear / `0.02` | Google's LoRA tutorial uses `5e-5`; its Keras recipe uses AdamW and weight decay `0.01`, but is not this Transformers classifier recipe. |
| Gemma 4 E4B | LoRA plus classifier head | `5e-5` | Linear / `0.02` | Gemma family reference only; monitor validation closely due to model size. |
| SmolLM3 3B | LoRA plus classifier head | `5e-5` | Linear / `0.02` | Hugging Face course gives `5e-5` as the conservative SFT point (`1e-4` balanced). |
| Llama 3.2 3B | LoRA plus classifier head | `1e-4` | Linear / `0.02` | Meta's cookbook example lists `1e-4`, weight decay `0`, and per-epoch gamma `0.85`; our factory uses linear decay to align scheduling across runs. |
| Granite 4 H Micro | LoRA plus classifier head | `2e-4` | Linear / `0.02` | IBM's LoRA recipe default; this is a high starting point, so validation checkpoints matter. |
| Ministral 3 3B | LoRA plus classifier head | `1e-4` | Linear / `0.02` | Family-level LoRA guidance, not a task-specific classification result. |
| EmbeddingGemma 2, Gemma 4 E4B, Ministral 3 3B, if set to `training_mode: frozen` | Frozen backbone plus LinearSVC readout | No neural optimizer | N/A | The downloaded representations are fixed; only the sklearn readout is fitted. Current configs select LoRA, so they use the LoRA recipe above. |
| SVC | TF-IDF plus LinearSVC, sigmoid calibration | `C=0.5` | N/A | `class_weight=balanced`; no neural optimizer or epoch schedule. |
| CatBoost + Stylometry | Gradient-boosted trees | `0.05` | N/A | 800 iterations, depth 7, L2 leaf regularization 5; validation early stopping selects iterations. |

The downloaded `Qwen3-VL-4B-Thinking-FP8` and `kaitchup/Qwen3-VL-2B-Instruct-NVFP4`
cache entries do not have model configs in `MODEL_CONFIGS` and are not runnable
through the current training factory. The supported config list is the source
of truth for trainability, not cache presence alone.

## Early stopping

Every epoch-based neural model config uses `early_stopping_patience: 3`. After
each validation epoch, the trainer compares ordinary validation accuracy with
the best seen so far; ties count as no improvement. It stops after three
consecutive non-improving validation epochs and retains the best-accuracy
checkpoint. SVC is a single fit with no epoch loop. CatBoost keeps its separate
`early_stopping_rounds` setting, measured in boosting iterations. Training and
continuation runs use their validation split for checkpoint selection; preserve
the selected checkpoint and its manifest for inference.

## Why the settings differ from upstream recipes

The official Qwen LoRA script uses LR `3e-4`, weight decay `0.1`, beta2 `0.95`,
warmup `0.01`, and cosine scheduling for generative SFT. The current classifier
keeps a lower `5e-5` LR and Transformers' AdamW defaults because its objective,
classification head, label data, and evaluation differ. This is a conservative
transfer, not evidence that `5e-5` is optimal.

Google's Gemma Keras tutorial uses LR `5e-5`, AdamW, and weight decay `0.01`.
Our Transformers configs retain the explicit framework default weight decay of
zero until a controlled comparison shows that adding decay helps this task.
Meta's Llama Cookbook lists LR `1e-4`, weight decay `0`, and multiplicative
per-epoch gamma `0.85`; we use linear scheduling to keep the schedule consistent
with the other classifier runs. IBM lists Granite LoRA LR `2e-4` as a default.
Hugging Face's SmolLM3 course describes `5e-5` as a conservative SFT value.

## Before a training starts

`python -m src.pipeline models train` prints the model's configured training
settings and asks whether to override any. Answer `y` to edit displayed fields
interactively, or press Enter to keep each field. Press `n` or Enter at the
initial question to use the configured defaults. A non-interactive invocation
must explicitly pass `--accept-config-defaults`; without it, the factory exits
before training. This is an acknowledgement of the displayed config, not a
hyperparameter search.

Example:

```bash
/home/zodrak/cuda13.x/bin/python -m src.pipeline models train \
  --model qwen06 --dataset v1 --accept-config-defaults
```

To make a repeatable override, edit or copy the model YAML and pass it with
`--config`; interactive overrides are included in the run's saved config and
experiment hash.

## Primary references

- [Google Gemma LoRA tutorial](https://ai.google.dev/gemma/docs/core/lora_tuning): LR `5e-5`, AdamW, weight decay `0.01`.
- [Qwen official LoRA single-GPU script](https://github.com/QwenLM/Qwen/blob/main/finetune/finetune_lora_single_gpu.sh): LR `3e-4`, weight decay `0.1`, beta2 `0.95`, warmup `0.01`, cosine schedule; generative SFT context.
- [Qwen fine-tuning guidance](https://github.com/QwenLM/Qwen/tree/main/recipes/finetune): LoRA/QLoRA support and memory guidance.
- [Hugging Face SmolLM3 SFT guide](https://huggingface.co/learn/smol-course/en/unit1/3): recommends starting at `5e-5`, with `1e-4` as a balanced SFT setting.
- [Meta Llama Cookbook fine-tuning config](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/README.md): LR `1e-4`, weight decay `0`, per-epoch gamma `0.85`.
- [IBM Granite LoRA recipe](https://github.com/ibm-granite/granite.build/blob/main/configurations/assets/environments/bash/steps/lora-finetune/README.md): LR default `2e-4`, LoRA rank 16, alpha 32.
- [NVIDIA Megatron Bridge Ministral 3 example](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/mistral/ministral3): model-family LoRA/DoRA example; exact values are recipe- and task-dependent.
