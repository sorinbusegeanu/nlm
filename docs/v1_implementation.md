# Name That LLM — V1 Implementation Plan

**Version:** 1.0 | **Date:** 2026-11-01  
**Companion:** `v1_design.md`  
**Scope:** Fixed-official-data model and ensemble discovery; external data begins in V2.  
**Environment:** Latest local NLM MVP (GitHub `main` is behind). Local workstation RTX 5070 Ti 16 GB / 64 GB RAM; optional Kaggle GPUs for heavier jobs.

## 0. Execution principles

1. **Inspect the current local MVP code first.** Map existing models, CLI, cached predictions, registry and test suite to V1 interfaces. Reuse working functionality; this plan gives contracts and targeted changes rather than presuming the older GitHub tree is current.
2. **Freeze the official dataset/folds.** Record actual CSV hash, 18-label order, train/dev/holdout IDs, prompt-group fold hash and baseline metrics. Treat the previous baseline as immutable.
3. **Change one experimental dimension per comparison:** backbone model/size first, ensemble blending second. Data stays fixed throughout V1.
4. Keep experimental checks small but decisive; save complete OOF results for accepted candidates. Every run has an immutable ID and a transparent compute record.
5. Use training jobs independently; shared artifacts are Parquet OOF files, checkpoints and JSON manifests rather than a single runtime with incompatible dependencies.

## M0 — Reconcile local MVP and freeze benchmark (0.5–1 developer-day)

**Implement / verify**

- Inspect local model factory, current configs, CLI, tests, data manifests and results; produce an implementation compatibility map.
- Read V0 dataset/split/profile, compute or verify hashes, and save `configs/experiments/v1_frozen_baseline.yaml` plus a `v1_baseline_manifest.json` pointing to existing immutable artifacts.
- Record SVC, UltraBERT-150M, Qwen3-0.6B and any currently completed ensemble OOF scores; run missing V0 baseline components using current local code.
- Check `dataset_id`, folds, 18-class mapping, probability/calibration semantics, and whether the holdout has already been accessed. Preserve existing split files.
- Fix reproducibility blockers only where they prevent a trustworthy comparison; use separate commits/config versions.

**Acceptance:** one frozen V0 benchmark table and complete split/label metadata; 20% holdout flagged untouched or already consumed truthfully.

## M1 — Generalize model registration and experiment controls (1–2 days)

**Add / adapt**

- A generic `ClassifierAdapter` registry with `smoke_test`, `fit_fold`, `predict_scores`, `package_offline` operations; register existing SVC, encoder and Qwen adapters.
- Model recipes in `configs/models/v1/`, with explicit model ID, repo, SHA revision, model family, head/pooling, LoRA targets/rank/alpha, learning rate, epochs, max length, microbatch, accumulation, quantization, seed, tokenizer flags and runtime environment.
- A lightweight two-stage runner: `screen` (one fold, isolated) and `oof` (all five fixed grouped folds). Never publish incomplete OOF as a full score.
- Env-specific runner support: isolated UltraBERT-compatible Transformers 5.x, and tested compatible versions for other backbones. Log environment/torch/CUDA versions.
- An inference-ready manifest and deterministic 18-column outputs, reusing existing MVP conventions.

**Illustrative CLI extensions** (adapt to actual local CLI naming):

```bash
python -m src.pipeline models screen --model qwen4 --dataset v0
python -m src.pipeline models train --model qwen4 --dataset v0
python -m src.pipeline ensemble evaluate --dataset v0
```

**Acceptance:** existing MVP runs still work; all new recipes can be discovered and share one prediction/manifest format; fake small-model integration tests pass without downloading large weights.

## M2 — Baseline encoder scaling (1–2 days + GPU time)

**Run:** `svc`, `ultrabert150`, `ultrabert380`.

- Resolve `dmis-lab/UltraBERT-150M` and `dmis-lab/UltraBERT-380M` at pinned commits.
- Verify `AutoModelForSequenceClassification` loading with supported Transformers (official UltraBERT docs: native `>=5.20`), head shape and checkpoint save/reload.
- Start with reference settings for both sizes: 384 or 512 tokens (audit truncation), learning rate **1e-5 for 150M**, **8e-6 for 380M**, 2–3 epochs; actual settings are initial hypotheses, not claims of optimality.
- Smoke train and then grouped 5-fold OOF; record score, per-class delta, unique correct vs SVC, cost.
- Keep **DeBERTa-v3-large optional**. Run it only if UltraBERT-380M accuracy/complementarity is unconvincing or an encoder challenger is justified. Treat it as a head-to-head challenger, not an automatic ensemble member.

**Acceptance:** validated encoder scaling comparison; only the more useful encoder(s) advance to final ensemble tests.

## M3 — Qwen size scaling (2–4 days + GPU time)

**Run:** `qwen06`, `qwen4`, `qwen8`; optionally `qwen17` only if useful as a cost/quality bridge.

- Start from pinned `Qwen3-4B-Base` and `Qwen3-8B-Base` weights; preserve the existing Qwen3-0.6B readout as baseline.
- Reuse mask-aware pooling, head, stable label order and training loop; make LoRA/QLoRA settings configurable. Run smoke tests for gradients, padding, mixed precision, quantized save/reload and inference equivalence.
- First recipe: 3 epochs, LoRA rank 16, alpha 32, 384–768 tokens, microbatch 1–2 for 8B, gradient accumulation as needed; tune only when training proves unstable or strongly underfits/overfits. These are starting points, not optimized hyperparameters.
- Quantization: 4-bit QLoRA for 8B on the local 16 GB card, with gradient checkpointing if needed; compare a representative inference batch after reload.
- Run one fixed fold as `screen`; complete five folds for promising candidates. Record Qwen scaling curve: model size → OOF balanced accuracy → unique errors fixed → training hours → test-set inference budget.

**Acceptance:** qwen4 and qwen8 have reproducible full OOF predictions or explicit documented hardware block; demonstrated marginal gain determines selection, not parameter count.

## M4 — Other model-provider families (2–4 days + GPU time)

**Run:** `gemma3_4b` and `mistral7`.

- Gemma 3: verify terms/access to `google/gemma-3-4b-pt`, its processor/tokenizer, text-only hidden-state path and classifier head. Implement its adapter using the supported text backbone or multimodal wrapper; save all processor/config dependencies. Test text-only gradient flow, 18 logits and offline reload before full training.
- Mistral: use `mistralai/Mistral-7B-v0.3`, base checkpoint, with its tokenizer and a classification readout/QLoRA. Verify pad/end-token behavior, save/load and hardware constraints.
- For both, start from the same experimental policy as Qwen (fixed data/folds, auditable recipe, 5-fold OOF) and record differences in max token coverage and training budget.
- Compare against the **current best ensemble**, not only standalone accuracy. A weaker standalone model can be kept if it reproducibly corrects complementary examples.

**Acceptance:** both families have documented performance and marginal contribution or a precise runtime/access blocker; no model is included solely for provider diversity.

## M5 — Ensemble discovery and selection (1–2 days)

**Implement three ensemble methods:**

1. Equal mean of calibrated probabilities.
2. Regularized nonnegative global weights (sum 1), evaluated through grouped meta cross-fitting.
3. Regularized class-wise logit biases + global model weights, trained on meta-training OOF and tested on meta-held-out rows.

**Selection algorithm:**

- Evaluate every completed singleton and chosen initial pairs.
- Starting from the best reliable single model, add the candidate with greatest **cross-fitted delta in balanced accuracy**; compare to current ensemble and run paired prompt-group bootstrap.
- Stop when additional model costs outweigh robust gain; aim for a compact 2–4 model set. Record leave-one-out deltas for final shortlist.
- Summarize per-label recall and errors corrected; keep probabilities and raw scores in separately named fields.

**Important:** Fit blend parameters only on OOF rows outside the meta-evaluation partition, and keep holdout isolated. The final blend weights can be fit on all development OOF **after** selecting the ensemble and method.

**Acceptance:** a ranked table with standalone/ensemble metrics, delta CI, corrected examples and cost; selected ensemble definition is reproducible from manifests.

## M6 — Dashboard additions (0.5–1 day)

Keep the existing Streamlit dashboard and add:

- **Model leaderboard:** architecture, backbone, parameter scale, complete OOF balanced accuracy, runtime, VRAM, artifact size.
- **Qwen scaling:** 0.6B, 4B, 8B performance and marginal contribution.
- **Encoder comparison:** UltraBERT-150M vs 380M; DeBERTa optional head-to-head.
- **Cross-family complementarity:** pairwise error overlaps, unique corrections, confusion by candidate label, marginal gain vs cost.
- **Ensemble experiments:** equal / weighted / class-adjusted cross-fitted results, selected final roster and holdout availability status.

**Acceptance:** every number comes from an immutable recorded run; screening and full OOF results are visibly separated.

## M7 — Final holdout, offline notebook and submission (1–2 days + Kaggle validation)

- Freeze the model roster, recipe choices and ensemble decision rule based on development OOF.
- If truly untouched, evaluate the frozen final ensemble on the 20% prompt-group holdout **once**. The holdout evaluation must reproduce **all members** (trained on 80% dev) plus the fixed ensemble weights; evaluating only a single model is not equivalent.
- Fit selected models with all official labelled data; carry forward model-specific preprocessing and finalized ensemble parameters.
- Package checkpoint weights and model adapters for internet-off use. Verify actual Kaggle `test.csv` interface and write `/kaggle/working/submission.parquet` with exactly `id`/`model` and valid label values.
- Perform a clean Kaggle notebook run; model full ~4,000-row inference time with a safety margin below 9 hours. Check memory for loading multiple selected models sequentially.
- Submit and record public leaderboard score and timestamp. Compare with the **observed** tenth-place score on that date; do not infer a top-10 cutoff from older published solutions.

**Acceptance:** portable submission executes offline and creates a schema-valid result; recorded local and public metrics are linked to the exact candidate manifests.

## 8. Experiment matrix — keep the number of decision branches controlled

| ID | Baseline / candidate | Changed variable | Main question |
|---|---|---|---|
| V1-E00 | MVP V0 SVC, UltraBERT150, Qwen06 | None | What is the immutable starting point? |
| V1-E01 | UltraBERT150 → UltraBERT380 | Encoder capacity | Does the larger encoder help? |
| V1-E02 | Qwen06 → Qwen4 | Qwen capacity | Is 4B worth its training/inference cost? |
| V1-E03 | Qwen4 → Qwen8 | Qwen capacity | Does 8B add accuracy or complementary corrections? |
| V1-E04 | Gemma3 4B | Backbone family | Does Gemma improve the best current ensemble? |
| V1-E05 | Mistral7 | Backbone family | Does Mistral improve the best current ensemble? |
| V1-E06 | Equal / weighted / class-adjusted | Decision rule | Which leakage-aware ensemble method improves balanced accuracy? |
| V1-E07 | DeBERTa-v3-large (optional) | Alternative encoder | Is DeBERTa a better or complementary encoder than UltraBERT380? |

`Qwen3-1.7B` (optional bridge) and `Qwen3-14B` (post-V1) remain registered candidate ideas, not required work. Do not multiply hyperparameter search branches until a strong architecture/ensemble shortlist exists.

## 9. Target file changes (align paths with actual local MVP)

```text
configs/experiments/v1_frozen_baseline.yaml
configs/models/v1/{ultrabert380,qwen4,qwen8,gemma3_4b,mistral7}.yaml
configs/models/v1/deberta_v3_large.yaml      # optional
src/model_factory/adapters/{encoder,qwen,gemma,mistral}.py  # or equivalent existing structure
src/model_factory/registry.py                 # adapter registration
src/evaluation/ensemble_search.py             # or extend existing ensemble factory
src/pipeline.py                               # screen/oof orchestration
experiments/registry.sqlite                   # existing registry, new manifests
artifacts/predictions/*_oof.parquet            # aligned 18-class outputs
dashboard/app.py                              # additive V1 views
tests/test_v1_*.py
notebooks/kaggle_inference.ipynb               # existing packager adapted to multiple models
```

## 10. Tests to implement as part of the named milestones

- Fold hashes and original Kaggle IDs unchanged; no `prompt_id` shared between each fit/validation or fit/holdout set.
- Prediction permutation test: shuffled row order yields same aligned ensemble results.
- Class-order test: exact 18 labels and `[N,18]` finite matrices with normalized probabilities.
- Adapter smoke: trainable head receives gradients, padded tokens excluded from pooling, checkpoint reload reproduces logits within precision tolerance.
- Meta validation test: class offsets/weights are fitted only on meta-training rows; grouped meta partitions are distinct.
- Artifact compatibility: existing baseline reading/registry/export remains functional.
- Kaggle offline: all dependencies attached; correct filename, column names, IDs, and inferencing time/memory.

## 11. Effort and compute assumptions

| Milestone | Engineering effort, excluding training wall time |
|---|---:|
| M0 frozen benchmark | 0.5–1 day |
| M1 generic adapters and runner | 1–2 days |
| M2 encoders | 1–2 days |
| M3 Qwen size variants | 2–4 days |
| M4 Gemma and Mistral | 2–4 days |
| M5 ensemble selection | 1–2 days |
| M6 dashboard | 0.5–1 day |
| M7 submission | 1–2 days |
| **Total** | **9–18 developer-days** |

Training time is unestimated until the **one-fold smoke benchmarks** provide actual throughput and VRAM. A 5-fold 8B QLoRA run may take materially longer than engineering work. Use measured token/s × examples × epochs × folds for each model's forecast; do not present a fabricated GPU-hour estimate.

## 12. Definition of done / V2 handoff

1. Frozen official V0 corpus and split metadata remain the **stable cross-model benchmark**.
2. All required V1 backbone candidates have full OOF metrics or documented access/compute blockers.
3. Dashboard ranks standalone and marginal ensemble gains with per-class recall and costs.
4. Final member selection, ensemble method and one-shot holdout are fully recorded.
5. Kaggle notebook works offline under competition limits with a valid `submission.parquet`.
6. Final report states observed LB score, observed top-10 benchmark, uncertainty and next hypothesis.
7. **V2 begins only after this**: add external data versions, retrain the selected 2–4 model recipes on identical folds and quantify data effects separately from architecture effects.

## 13. Primary reference links

- Competition requirements: https://www.kaggle.com/competitions/name-that-llm/overview/description
- UltraBERT HF release and API notes: https://huggingface.co/blog/dmis-lab/ultrabert
- Qwen3 Base models: https://huggingface.co/collections/Qwen/qwen3
- Gemma pretrained 4B: https://huggingface.co/google/gemma-3-4b-pt
- Mistral 7B base: https://huggingface.co/mistralai/Mistral-7B-v0.3
- Optional DeBERTa: https://huggingface.co/microsoft/deberta-v3-large
