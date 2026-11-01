# Name That LLM — V0.5 Implementation Plan

**Version:** 0.5  
**Date:** 2026-11-01  
**Companion:** `v0_5_design.md`  
**Implementation base:** The latest **local** completed MVP (GitHub `main` is stale).  
**Target hardware:** RTX 5070 Ti 16 GB; existing Kaggle inference environment.

## 0. Rule for V0.5

**Only add Qwen3-1.7B-Base and Qwen3-4B-Base as new trained members/candidates in the current ensemble.** Use the existing data factory, feature factory, neural trainer, model-prediction schema, calibration, evaluator, ensemble factory, registry, dashboard and Kaggle exporter. Make the minimum code changes required by model size, checkpoint loading and inference packaging.

Do not implement the broader V1 shortlist or external-data phase as part of this change. Do not alter official data, recompute split assignments or tune the 20% holdout.

## M0 — Inspect local MVP and capture frozen baseline

**Actions**

- Read local model configuration and CLI registration, Qwen3-0.6B training/inference code, exact OOF manifest format, ensemble input selection, dashboard and tests.
- Capture hashes for existing official dataset, prompt-group splits, label order and baseline OOF IDs.
- Identify **one completed baseline experiment each** for `svc`, `ultrabert150`, `qwen06`; write a single V0.5 benchmark manifest with those exact experiment IDs.
- Compute/retrieve E0: existing `svc + ultrabert150 + qwen06` OOF blend and current balanced accuracy. Preserve prior model artifacts.

**Output:** `artifacts/v0_5/baseline_manifest.json` (or same role in existing registry), frozen E0 metric.

**Done when:** the baseline data and model-selection hashes are explicit and the original results remain unchanged.

## M1 — Register two new Qwen models

**Actions**

- Add model IDs `qwen17` and `qwen4` to the **current** registry/CLI, using the existing Qwen adapter.
- Add `configs/models/qwen17.yaml` and `configs/models/qwen4.yaml` (or equivalent local config scheme):

```yaml
# qwen17.yaml — initial recipe
model: qwen17
checkpoint: Qwen/Qwen3-1.7B-Base
revision: <PINNED_40_CHARACTER_HF_COMMIT_SHA>
epochs: 3
learning_rate: 0.0001
max_length: 384
batch_size: 2
gradient_accumulation_steps: 4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_checkpointing: false
quantization: none
```

```yaml
# qwen4.yaml — initial recipe
model: qwen4
checkpoint: Qwen/Qwen3-4B-Base
revision: <PINNED_40_CHARACTER_HF_COMMIT_SHA>
epochs: 3
learning_rate: 0.0001
max_length: 384
batch_size: 1
gradient_accumulation_steps: 8
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_checkpointing: true
quantization: none  # change to 4bit only if BF16 LoRA cannot fit
```

- Parameterize the existing Qwen fold trainer instead of copying it; make checkpoint directory/model ID distinct for each variant.
- Check local `transformers`, `peft`, `torch`, CUDA and optional `bitsandbytes` compatibility. Record installed versions in manifests.

**Done when:** both IDs resolve to the correct **Base** checkpoint, have pinned SHA, and run through the existing CLI/config loader.

## M2 — Memory and correctness smoke tests

For each new size:

1. Load pinned tokenizer and backbone, run one labelled minibatch forward/backward pass and check LoRA/classification-head gradients.
2. Verify masked pooling and label mapping are identical to baseline; use 18 finite logits and normalized probabilities.
3. Confirm save, unload, reload and prediction equivalence on a small held-back minibatch.
4. Measure peak VRAM and step time at the intended 384-token length.
5. If 4B exceeds available memory, keep the same task/model/labels and switch **only** the memory recipe to 4-bit QLoRA plus gradient checkpointing, then rerun the smoke test. Record this deviation in the model config and manifest.

**Done when:** both models fit on an actual training batch and can be reloaded for inference on the intended runtime. A small smoke run is not reported as full cross-validation.

## M3 — Run full grouped OOF for qwen17 and qwen4

**Actions**

- Train independently on the **same frozen five development folds** as the MVP, with the existing per-fold seed schedule.
- Store for each model: `id`, `prompt_id`, `fold`, `y_true`, all 18 `score::<label>` and `prob::<label>` columns, fold checkpoints, recipes and timings.
- Confirm every development row has exactly one prediction; label order, ID set, fold hash and group isolation match the captured baseline.
- Log standalone balanced accuracy, per-class recall, confusion, macro-F1, log loss, time, peak VRAM, inference rate and artifact bytes.
- Reuse existing OOF outputs of baseline models; only train them again if baseline OOF is missing.

**Done when:** two complete and aligned immutable OOF manifests exist. If training cannot complete, record a precise hardware or software blocker and preserve partial artifacts as **screening only**.

## M4 — Evaluate only the requested ensemble additions

Use an **explicit roster of the selected exact manifests** to avoid mixing unrelated runs:

```text
E0 = svc + ultrabert150 + qwen06  (frozen baseline)
E1 = qwen17                        (standalone)
E2 = E0 + qwen17
E3 = qwen4                         (standalone)
E4 = E0 + qwen4
E5 = E0 + qwen17 + qwen4
E6 = equal vs existing cross-fitted regularized weights on the best candidate roster
```

For E1/E3 report also the standalone delta relative to `qwen06` and optionally the `qwen17 + qwen4` pair. For E2/E4/E5 report **delta balanced accuracy relative to E0**, paired prompt-group bootstrap intervals, unique corrected cases and per-class changes. The existing cross-fitted blend method suffices; preserve its current objective/regularization and label that objective accurately.

**Selection:** record the best validated model subset, weights/method, per-class implications and additional inference cost. Include both new models only if both earn a place through measured complementary benefit.

**Done when:** the existing dashboard/report clearly shows E0–E6 side by side with full-OOF provenance and hashes.

## M5 — Final inference and submission compatibility

- If the chosen ensemble includes qwen17/qwen4, train selected new members on all official labelled rows **after** experiment selection. Keep the one-shot official holdout policy: evaluate the frozen ensemble once only if that holdout is genuinely untouched.
- Extend the current model loader/packager to load new Qwen sizes and classification heads using the existing local artifacts; keep model-specific tokenizers and pooling settings.
- Perform sequential model inference and aggregate aligned 18-class probabilities using the chosen blend; load/unload to respect 16 GB local GPU VRAM and Kaggle GPU constraints.
- Reuse the existing internet-disabled Kaggle notebook and ensure valid `submission.parquet` with exact `id`, `model` columns, input IDs, allowed labels and full-run memory/runtime checks.

**Done when:** the selected ensemble is reproducibly packaged and runs through the existing Kaggle submission path.

## Required code changes (map to actual local files)

| Location | Minimal change |
|---|---|
| Existing model-config directory | Add `qwen17` and `qwen4` YAML recipes |
| Current model registry/CLI | Register both names with the current Qwen trainer |
| Existing Qwen training function | Expose model-specific batch/LoRA/gradient-checkpointing/optional 4-bit fields; unique output paths |
| Existing inference/packaging | Support both model IDs, heads, tokenizer and sequential ensemble loading |
| Ensemble orchestration | Explicit input-manifest selection and E0–E6 comparisons using existing scoring |
| Current dashboard/registry | Display new model rows and marginal blend deltas |
| Existing tests | Add small adapter, alignment and blend regression cases |

## Minimal test suite

- **Frozen benchmark:** raw-data hash, split hash, stable 18-label order and existing baseline OOF files unchanged.
- **Adapter:** classifier/LoRA gradients, mask-aware pooling, finite `[N, 18]` output, probability row sums and save/reload parity.
- **Fold isolation:** no prompt group shared between train and validation; complete OOF predictions matching baseline ID set.
- **Ensemble:** E0 result reproduces exactly; E2/E4/E5 are invariant to shuffled input order; model identity/weights recorded.
- **Calibration:** per-fold calibration uses training-side data only under the existing MVP method.
- **Offline export:** sequential selected-model inference produces one permitted label per test `id` and fits Kaggle memory/time constraints.

## Execution commands (illustrative; adapt to the current local MVP CLI)

```bash
python -m src.pipeline models train --model qwen17 --dataset v0
python -m src.pipeline models train --model qwen4 --dataset v0
python -m src.pipeline ensemble evaluate --dataset v0
streamlit run dashboard/app.py
```

Before calling the ensemble command, ensure that it selects the **three frozen baseline manifests plus the two new manifests**, rather than collecting arbitrary experiments with matching dataset tags.

## Definition of done

- Two new Qwen3 sizes trained and evaluated on unchanged official grouped folds, with full same-row OOF probabilities.
- Reproducible comparisons E0–E6 and retained baseline metric.
- Dashboard identifies standalone Qwen scaling and each model's actual ensemble gain.
- Selected subset and blend method include only members supported by measured OOF contribution.
- Kaggle export continues to work for the selected final ensemble.
- Data expansion, extra encoders, other providers, and redesigned blending algorithms remain outside V0.5.

## References

- https://huggingface.co/Qwen/Qwen3-1.7B-Base
- https://huggingface.co/Qwen/Qwen3-4B-Base
- https://www.kaggle.com/competitions/name-that-llm
