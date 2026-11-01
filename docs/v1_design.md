# Name That LLM — V1 Model & Ensemble Discovery Design

**Status:** Proposed | **Version:** 1.0 | **Date:** 2026-11-01  
**Competition:** [Kaggle — Name That LLM](https://www.kaggle.com/competitions/name-that-llm/overview/description)  
**Relationship to MVP:** Extension of `docs/mvp_design.md` and the **latest local MVP implementation**. The GitHub `main` branch is known to lag behind the local code; reconcile interfaces against local code before changing anything.

## 1. Objective and fixed experimental boundary

**Objective:** Identify which pretrained classifier architectures, sizes, and model families improve **18-class balanced accuracy**, then assemble the smallest reliably stronger ensemble. V1 is the **model-discovery phase**, with the official Kaggle labelled dataset fixed. Target top-10 competitiveness, measured against the actual leaderboard; no assumed top-10 cutoff or promised score.

**Frozen for all V1 comparisons:**

- Official labelled Kaggle corpus **V0 only**: 2,699 examples, 18 classes, ~150 prompt groups (verify the exact raw-file hash and profile).
- Canonical original `text`, original `id`, `prompt_id`, `model`, label order, fixed prompt-group folds and split-assignment hash.
- MVP's **20% untouched prompt-group holdout** plus **5 development prompt-group folds**, provided the local implementation already used this protocol. Preserve the existing splits; do not regenerate them to improve scores.
- Main evaluation metric, class mapping, row alignment, held-out prompt isolation, artifact schema and one-off holdout release policy.
- Comparable baseline *data* and *folds* for all models. Model-specific tokenizers, memory configurations and fine-tuning recipes are versioned separately.

**Separate phase (V2):** externally collected or generated text, synthetic data, new prompt distributions, model outputs purchased or sourced from APIs, dataset balancing or large corpus mixing. This keeps model and ensemble effects identifiable before data changes.

**Competition constraints (verify again before submission):** 18 valid labels, balanced accuracy, unseen prompts, approximately 4,000 hidden test passages, notebook-produced `submission.parquet` with `id` and `model`, internet disabled for scoring, and 9-hour CPU/GPU notebook limits. Final deadline: 2027-01-13 at 23:59 UTC.

## 2. Architecture — extend the MVP, keep its contracts

```text
Frozen official corpus V0 + frozen prompt-group assignments
                        |
                        v
            [Existing Data/Feature Factories]
                        |
      +-----------------+---------------------+
      |                 |                     |
 [SVC / stylometry] [Bidirectional encoders] [Causal LM backbones]
      |          UltraBERT 150/380           Qwen 0.6/4/8
      |          DeBERTa optional             Gemma 3 4B / Mistral 7B
      +-----------------+---------------------+
                        |
              [Model Adapter Contract]
              logits + probability scores
              per fold, IDs, 18 labels
                        |
                        v
                [OOF Model Registry]
                        |
            [Ensemble Experiment Factory]
       single / pairs / forward addition / calibrated blend
                        |
                        v
         [Evaluation + Dashboard + Experiment Reports]
                        |
               [One-shot holdout]
                        |
             [Final training + Kaggle packager]
```

The existing CLI, SQLite registry, Parquet OOF predictions, Streamlit dashboard and Kaggle offline exporter remain the foundation. Introduce generic model-adapter registration rather than separate hard-coded branches for every new model.

## 3. Model roster and priority

| ID | Model / proposed checkpoint | Architecture | V1 role | Priority |
|---|---|---|---|---|
| `svc` | Existing multi-view char/word/punctuation/shape TF-IDF + LinearSVC | Sparse discriminative | Statistical/style baseline | Required |
| `ultrabert150` | `dmis-lab/UltraBERT-150M` | Bidirectional encoder | Existing small-encoder baseline | Required |
| `qwen06` | `Qwen/Qwen3-0.6B-Base` | Causal transformer + classification head/LoRA | Existing small-causal baseline | Required |
| `ultrabert380` | `dmis-lab/UltraBERT-380M` | Bidirectional encoder | Encoder scaling study | Required |
| `qwen4` | `Qwen/Qwen3-4B-Base` | Causal transformer + LoRA | Causal scaling study | Required |
| `qwen8` | `Qwen/Qwen3-8B-Base` | Causal transformer + QLoRA | Larger Qwen; incremental contribution | Required |
| `gemma3_4b` | `google/gemma-3-4b-pt` | Gemma 3 pretrained multimodal backbone, text-only input | Independent model family | Required, subject to access/runtime smoke test |
| `mistral7` | `mistralai/Mistral-7B-v0.3` | Causal transformer + QLoRA | Independent model family | Required, subject to runtime smoke test |
| `deberta_v3_large` | `microsoft/deberta-v3-large` | Bidirectional encoder | Optional **challenger to UltraBERT-380M**, not assumed ensemble member | Optional |

`Qwen3-1.7B-Base` may be added as an **optional bridge** if 4B training is unstable or expensive. `Qwen3-14B-Base` is a **post-V1 capacity experiment** only if 8B demonstrates a credible gain and compute permits. The shortlist is a candidate pool, not a pre-selected final ensemble.

**Model identity ≠ competition label:** Qwen, Gemma, Mistral etc. are **classifier backbones** trained to predict the 18 Kaggle author labels. A classifier need not itself be one of the 18 candidate text generators.

### Encoder vs causal backbone

- **Encoders:** attach an 18-class head and fine-tune for discrimination. Test UltraBERT-150M vs 380M first. DeBERTa is only justified if it improves official grouped OOF or ensemble complementarity against the selected encoder.
- **Causal LMs:** classify passage embeddings/hidden states with an 18-class head. Reuse the MVP's mask-aware pooling and LoRA strategy as the initial recipe; implement a model-specific adapter when architecture differs. Record exactly which hidden state, pooling rule and parameters are trained.
- **Gemma 3 4B:** its official `-pt` checkpoint is multimodal. Use **text-only passage inputs**, a supported text-backbone or multimodal-wrapper path, and a classifier head. Smoke-test tokenizer/processor, hidden-state access, PEFT targeting, save/load and **offline** inference before full CV. Gemma terms/access must be accepted where required.
- **Mistral 7B:** start from the official base (not instruction-tuned) checkpoint and pin exact revision. Test quantized training and inference parity.

### Controlled training recipes

- Freeze one **initial reference recipe per model family** (initial LR, LoRA rank, epochs, pooling, batch size, max length and seed); document resource-driven deviations before comparisons.
- Run a shared input-length audit; choose a defensible max length (initially 384–768 tokens, considering tokenizer differences and 40–250 word passage lengths) and log per-model truncation rate. Preserve the entire passage where practical.
- Prefer full fine-tuning for encoders that fit; use LoRA/QLoRA for the 4B–8B causal backbones on the local RTX 5070 Ti 16 GB. Quantization, gradient checkpointing and batch/accumulation are explicit versioned config fields.
- Use a **single 18-class softmax head** for primary comparison. Any alternative pooling/head design is a separately named ablation after the initial architecture comparison.
- The local 16 GB GPU is for sequential experiments; Kaggle GPU resources can be used for larger trials when available. Offline submission must satisfy the actual selected Kaggle runtime/hardware limits.
- UltraBERT needs a supported Transformers **5.x** environment (official documentation currently notes native support at `transformers>=5.20`); the MVP repo's `transformers<5` dependency is incompatible. Use versioned per-adapter environments if necessary, without destabilizing existing SVC/Qwen training.

## 4. Experiment protocol: fair comparisons with small official data

### 4.1 Immutable split and OOF requirements

1. A row's OOF prediction must come from a checkpoint that **never trained on that row's prompt group**.
2. All model OOF outputs must use identical development `id` sets, canonical 18-label mapping, fold assignment hash and original text.
3. Fit preprocessing, feature vocabulary, scalers, calibration and neural weights on training rows inside the corresponding outer fold. Calibration derived from train predictions requires train-only internal validation/cross-fitting for reliable probability quality.
4. The 20% holdout is evaluated **once, after model and ensemble selection**. Do not re-open it for model discovery. If the local MVP already consumed it, label it as development evidence rather than presenting it as untouched confirmation.
5. Training on the final combined official labelled data is reserved for packaging after decisions are frozen.

### 4.2 Two-stage experimental execution

- **Smoke / triage:** checkpoint access, one minibatch, save/load, inference shape, memory; optionally one fixed development fold for early model rejection. A one-fold result is labelled `screening` and is **not** used for definitive ensemble comparisons.
- **Publication:** full 5-fold grouped OOF for candidates passing smoke/triage. Publish one manifest for the complete run, including fold-level scores and training times. Prefer a second seed for finalists when differences are small.

### 4.3 Primary and supporting metrics

| Metric | Operational use |
|---|---|
| **OOF balanced accuracy** | Primary classification selection metric (18-class macro recall) |
| **Δ ensemble balanced accuracy** | Value of adding a model to the existing selected set |
| Per-class recall, 18×18 confusion | Family-level and sibling-model failure modes |
| Unique correct / error overlap / oracle ceiling | Complementarity diagnostics |
| Prompt-level paired bootstrap delta CI | Difference uncertainty across unseen prompt groups |
| Fold stability, optional seed variance | Robustness against small development sample |
| Log loss / Brier / calibration | Suitability for probability blending |
| Truncation rate, GPU hours, peak VRAM, inference seconds | Operational trade-offs |

Do not interpret a high theoretical oracle ceiling as proof that a blend can realize it. Avoid selecting a model solely because of unstable +0.001 improvements. Proposed thresholds **for triage, not guarantees**: repeatable >+0.01 absolute balanced accuracy merits investigation; >+0.02 with robust fold/seed behavior is particularly attractive.

## 5. Ensemble discovery

All candidate models produce **row-aligned 18-class probabilities**. Keep their raw logits/margins as separate artifacts.

Run three decision-rule families:

1. **Equal blend:** calibrated probabilities averaged across selected members.
2. **Regularized global weighted blend:** nonnegative weights summing to 1; optimize on training OOF rows of an outer meta-fold, evaluate on held-out meta-fold rows. Primary selection uses the resulting **cross-fitted balanced accuracy**; log-loss can stabilize the optimizer, but must be reported separately.
3. **Weighted blend + regularized per-class logit offsets:** class-specific offsets learned only on meta-training folds (with shrinkage); evaluate strictly on held-out meta-fold rows to improve macro recall under class imbalance.

**Member search:** evaluate each singleton; all requested initial pairs; forward-add the model with the largest reproducible incremental balanced-accuracy improvement; optionally perform leave-one-out ablations of finalists. With nine candidates, exhaustive subsets are unnecessary. Maintain a compact **2–4 model** inference target unless a fifth member contributes clear lift at acceptable cost.

**Meta-leakage:** ordinary OOF stacking can reuse predictions from base models trained on other meta-fold labels. For definitive selection, use nested/cross-fitted meta-evaluation and preserve audit trails, then confirm the frozen ensemble once on the untouched prompt holdout. Distinguish cross-fitted estimates from a final weight fit on all development OOF.

**Optional later research:** provider-family specialist heads, test-set balancing / assignment methods, feature-level distillation. These enter only after confusion and complementarity evidence identify a specific problem.

## 6. Artifact schema and dashboard

Minimal adapter interface:

```python
class ClassifierAdapter:
    def smoke_test(self, config, sample_rows) -> dict: ...
    def fit_fold(self, train_rows, val_rows, config, output_dir) -> dict: ...
    def predict_scores(self, checkpoint, rows, classes) -> dict: ...
    def package_offline(self, checkpoint, output_dir) -> dict: ...
```

Each returned fold prediction contains `id`, `prompt_id`, `fold`, `y_true`, `score::<label>` and `prob::<label>` for all 18 labels. Model manifest includes `model_id`, HF repo and immutable SHA, tokenizer/processor revision, architecture/adapter version, training-data and split hashes, exact config, seed, code commit, checkpoint path, train/infer times, VRAM, tokenizer truncation rate and calibrated metric summary. Preserve existing MVP schema when it already offers these fields; add compatible extensions.

Dashboard additions: model comparison table; individual-versus-ensemble bars; marginal gain vs cost; per-class change heatmap; Qwen model-size curve; encoder-vs-causal comparison; model errors corrected by peers; holdout gate and artifact audit summary. Every plotted metric is tagged `screening`, `development OOF`, `holdout`, or `public leaderboard`.

## 7. Kaggle delivery

- Final model(s) trained on **all official labelled rows** under frozen recipes; selected ensemble weights fitted from valid development OOF predictions.
- Package every tokenizer, configuration, quantized/merged or adapter checkpoint, classifier head and runtime dependency needed for inference.
- Notebook reads official `test.csv`, writes exactly one row per input `id` to `/kaggle/working/submission.parquet` with exact allowed model labels. Execute with Internet disabled.
- Validate on a clean Kaggle runtime, with peak memory and measured inference time for the full ~4,000-row hidden workload; keep margin under the 9-hour limit. A local or interactive 1-row Kaggle test does not establish full hidden-test runtime.
- Maintain local CV as the primary selection signal; public leaderboard is a secondary distribution-shift check.

## 8. Success / definition of done

- V0 raw data and prompt-group folds remain unchanged (hashes verified).
- Required candidate models have completed **smoke and/or a recorded reason for resource/access block**; models selected for comparison have full 5-fold OOF outputs.
- Both UltraBERT sizes, both larger Qwens, and the Gemma/Mistral cross-family candidates have been explicitly assessed with common score/cost criteria.
- Every selected ensemble member has a documented marginal contribution and inference cost.
- Equal, regularized weighted, and class-adjusted blends have leakage-aware comparative results.
- Selected ensemble is confirmed on the one-shot holdout, when genuinely untouched, and packaged for reproducible internet-off Kaggle execution.
- A baseline public LB score, a final submission score and the contemporaneous tenth-place threshold are recorded **as observed numbers**, never assumptions.
- V2 expansion can import new training data while retaining the frozen V0 model-discovery benchmark.

## 9. Sources and pinned checkpoints to resolve

- Competition: https://www.kaggle.com/competitions/name-that-llm/overview/description
- UltraBERT technical requirements: https://huggingface.co/blog/dmis-lab/ultrabert
- UltraBERT checkpoints: https://huggingface.co/dmis-lab/UltraBERT-150M ; https://huggingface.co/dmis-lab/UltraBERT-380M
- Qwen checkpoints: https://huggingface.co/Qwen/Qwen3-0.6B-Base ; https://huggingface.co/Qwen/Qwen3-4B-Base ; https://huggingface.co/Qwen/Qwen3-8B-Base
- Gemma 3 pretrained: https://huggingface.co/google/gemma-3-4b-pt
- Mistral base: https://huggingface.co/mistralai/Mistral-7B-v0.3
- Optional DeBERTa challenger: https://huggingface.co/microsoft/deberta-v3-large

At implementation time resolve and **pin exact commit SHAs** for all checkpoints. Linked repo names are discovery targets, not immutable versions.
