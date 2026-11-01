# Name That LLM — V0.5 Design: Qwen3 1.7B + 4B Ensemble Extension

**Version:** 0.5  
**Status:** Implementation-ready design  
**Date:** 2026-11-01  
**Builds on:** The **latest local** implemented MVP, `mvp_design.md`, and `mvp_implementation.md`  
**Scope:** Add two Qwen3 classifier sizes to the existing frozen-data model comparison and ensemble. The GitHub `main` branch is older than the local implementation and is not the implementation baseline.

## 1. Objective

Determine whether **Qwen3-1.7B-Base** and **Qwen3-4B-Base**, fine-tuned as 18-class passage-origin classifiers, improve the existing ensemble's **grouped out-of-fold (OOF) balanced accuracy**. Preserve the stable official-Kaggle-data benchmark so that gains can be attributed to model size and complementarity.

**V0.5 changes only:** two model recipes (`qwen17`, `qwen4`), the smallest necessary size/configuration support in the existing Qwen trainer and offline inference path, their aligned OOF predictions, and ensemble comparisons that include them.

**Preserve:** official V0 data, original texts and labels, canonical 18-class order, existing prompt-group splits, baseline models, existing metrics, dashboard, registry, validation protocol, and Kaggle submission contract. External data and other architectures remain for a later version.

## 2. Frozen benchmark

- Official Kaggle labelled dataset V0: **2,699 passages, 18 classes, roughly 150 prompt groups**; use the actual locally recorded dataset hash rather than regenerating data artifacts.
- Use the **existing** 20% untouched prompt-group holdout and five grouped development folds, if that is how the completed local MVP was initialized. Preserve the exact stored assignment and split hash. If the local MVP's split differs, preserve its existing split and record the difference.
- Reuse completed baseline OOF artifacts for `svc`, `ultrabert150`, and `qwen06`; record their immutable experiment IDs and prediction hashes.
- New model train/validation row IDs must exactly match baseline OOF row IDs, with each validation prompt group excluded from that model's fold training.
- Keep the holdout untouched for a final, preselected ensemble confirmation. If previously used, mark its status accurately.

## 3. Model roster

| Model ID | Hugging Face checkpoint | Role in V0.5 | Action |
|---|---|---|---|
| `svc` | Existing TF-IDF + LinearSVC | Existing baseline member | Reuse OOF |
| `ultrabert150` | Existing UltraBERT-150M encoder | Existing baseline member | Reuse OOF |
| `qwen06` | `Qwen/Qwen3-0.6B-Base` | Existing Qwen baseline member | Reuse OOF |
| **`qwen17`** | **`Qwen/Qwen3-1.7B-Base`** | New Qwen candidate | Train and evaluate |
| **`qwen4`** | **`Qwen/Qwen3-4B-Base`** | New Qwen candidate | Train and evaluate |

Both new models are **backbone classifiers**, producing one of the same 18 Kaggle author labels—not new output labels. Use official **Base** checkpoints, with exact immutable commit SHA pinned in each recipe. Do not substitute instruction-tuned models under these experiment IDs.

## 4. Model training and resource approach

Reuse the local MVP's **Qwen classification architecture**: 18-class supervised head, attention-mask-aware hidden-state pooling, LoRA, class mapping, and logits/probability emission. Train independent models for every existing grouped development fold.

**Initial recipes** (controlled starting points, subject to one-fold memory smoke tests):

| Setting | `qwen17` | `qwen4` |
|---|---|---|
| Backbone | 1.7B Base | 4B Base |
| Tokens | Same baseline truncation policy, initially 384 | Same baseline truncation policy, initially 384 |
| Epochs | 3 | 3 |
| Learning rate | 1e-4 | 1e-4 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Initial microbatch / accumulation | 2 / 4 | 1 / 8 |
| Effective batch per device | 8 | 8 |
| Precision | BF16 if supported | BF16 if supported |
| Memory strategy | LoRA, optional gradient checkpointing | LoRA + gradient checkpointing; **4-bit QLoRA fallback** if 16 GB VRAM is insufficient |
| Hardware | RTX 5070 Ti 16 GB | RTX 5070 Ti 16 GB |

Use the **same seed schedule**, label order, pooling definition, and maximum-length policy as the completed Qwen baseline to the extent the implementation permits. Log any capacity-driven deviations. Qwen3 requires a suitable Transformers version (model card indicates at least `4.51.0`), with compatible PEFT/quantization dependencies verified against the local environment.

On the 16 GB GPU, train one model at a time. Package the selected inference models so they can be loaded and unloaded **sequentially**, accumulating probabilities rather than holding all models in VRAM concurrently.

## 5. Minimal software changes

1. Register `qwen17` and `qwen4` in the **existing** model configuration/CLI mapping.
2. Generalize the existing Qwen training call to accept model ID, checkpoint, revision, batch/accumulation, precision, gradient checkpointing, and optional 4-bit quantization. Reuse code instead of cloning the Qwen trainer.
3. Give each checkpoint, log, prediction file, and manifest a unique model-specific experiment ID. Preserve existing MVP field names and schemas.
4. Extend inference/packaging only as needed for the two new Qwen checkpoints and the chosen ensemble. Preserve the existing Kaggle notebook/export pathway.
5. Make ensemble evaluation consume an **explicit list of baseline and candidate OOF manifests** (or equivalent exact run selection) so unrelated historical experiments cannot accidentally enter the blend.
6. Preserve all existing metrics and dashboard components; extend the model comparison and ensemble displays using those artifacts.

## 6. Experiments: only additions to the existing blend

Let **E0** denote the already measured MVP blend of `svc + ultrabert150 + qwen06`, evaluated from its fixed OOF files.

| Experiment | Model set | Question |
|---|---|---|
| `V05-E0` | SVC + UltraBERT150 + Qwen06 | Frozen existing ensemble score |
| `V05-E1` | Qwen17 alone | Is 1.7B stronger than Qwen06? |
| `V05-E2` | E0 + Qwen17 | What is the marginal ensemble contribution? |
| `V05-E3` | Qwen4 alone | Is 4B stronger than Qwen06 / 1.7B? |
| `V05-E4` | E0 + Qwen4 | What is the marginal ensemble contribution? |
| `V05-E5` | E0 + Qwen17 + Qwen4 | Does combining both additions help? |
| `V05-E6` | Best member subset; equal vs existing cross-fitted regularized weights | Can the existing blend logic improve the selected roster? |

Also report the three individual Qwen sizes side by side and, where useful, the pair `Qwen17 + Qwen4`. Reuse existing ensemble mechanics: equal probability average and the MVP's cross-fitted weight fitting; no new stacking system, class-offset optimizer, prompt-assignment correction, or extra ensemble method.

**Selection rule:** retain new model(s) when they produce a credible, repeatable improvement in **grouped OOF balanced accuracy** against E0, considering group-bootstrap uncertainty, class-recall trade-offs, complementarity and inference cost. Report minor gains as uncertain rather than assigning an arbitrary acceptance threshold.

## 7. Measurements and comparisons

For every new model and blend, save:

- 18-class grouped OOF balanced accuracy (**primary**), macro-F1, per-class recall, 18×18 confusion matrix.
- Absolute and relative delta vs Qwen06 and **absolute ensemble delta vs E0**.
- Unique correct predictions, pairwise error overlap/disagreement and prompt-group paired bootstrap interval for the ensemble delta.
- Log loss/calibration when available; use the same probability calibration discipline as the local MVP.
- Training time per fold, peak GPU VRAM, inference rows/second, artifact size, actual quantization/precision and checkpoint SHA.

A one-fold smoke test is labeled **screening**. An accepted score must be based on the **complete same-row grouped OOF** set.

## 8. Finalization and Kaggle execution

- Select the V0.5 model subset and blend rule based only on grouped development OOF.
- Confirm the frozen selected ensemble on the untouched holdout **once**, if still available. Every participating member must have holdout predictions generated by fitting on development only.
- Retrain **selected** new Qwen models on all official labelled rows using frozen recipes; reuse or appropriately retrain existing selected baseline models per the MVP final-training policy.
- Package all selected checkpoints and their tokenizer/head/config dependencies. Validate the existing internet-off Kaggle inference path and `submission.parquet` (`id`, `model`) using the competition's exact interface.
- Measure sequential load/inference runtime and memory on the chosen Kaggle environment; verify its actual allowed limits before submitting.

## 9. Definition of done

1. The original official data, split assignments, canonical class order and baseline OOF artifacts are unchanged.
2. `qwen17` and `qwen4` each produce valid, complete **five-fold grouped OOF** 18-class predictions, or a precisely documented resource limitation.
3. Baseline E0 and all `V05-E1` through `V05-E6` comparisons are reproducible from explicitly selected, aligned model artifacts.
4. Dashboard/report shows Qwen06 → Qwen17 → Qwen4 standalone performance and each new model's **marginal gain to E0**, class-level impact and cost.
5. Selected final blend has saved model roster, weights, hashes, inference plan and one-shot holdout result when applicable.
6. The chosen blend can run through the existing offline Kaggle packaging/submission pathway.
7. Data and new-provider experiments are cleanly deferred to V1/V2.

## 10. Checkpoint references

- Qwen3-1.7B-Base: https://huggingface.co/Qwen/Qwen3-1.7B-Base
- Qwen3-4B-Base: https://huggingface.co/Qwen/Qwen3-4B-Base
- Existing baseline Qwen3-0.6B-Base: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Competition: https://www.kaggle.com/competitions/name-that-llm

Resolve and pin the exact immutable source commit hash during implementation. The checkpoint names above identify model families, not pinned revisions.
