# Name That LLM — MVP Design

**Version:** 0.1  
**Scope:** Official competition training data only (2,699 labelled passages, 18 classes, approximately 150 prompt groups).  
**Purpose:** Establish a reproducible baseline for model comparison, ensemble complementarity and future dataset experiments.

## 1. Goals and boundaries

The MVP must:

1. Train and validate three different classifiers on one shared, prompt-grouped 80/20 split: char/word TF-IDF + LinearSVC, UltraBERT-380M and a configurable causal language model, defaulting to Qwen3-0.6B, with LoRA.
2. Produce aligned predictions for all validation rows and compare equal-weight blends with weights fitted on train rows.
3. Quantify each classifier's standalone accuracy, marginal ensemble contribution, errors and compute cost.
4. Record all dataset, code, split, model and prediction versions.
5. Show experiments and ablations on a minimal local dashboard.
6. Export a standalone, internet-off Kaggle inference notebook and `submission.parquet` when submission requirements are verified.

**Initial exclusions:** External data acquisition, autonomous research agents, complex Gaussian heads, prompt reconstruction, Sinkhorn balancing, hard typography rules and elaborate stacking. They may be tested later as explicitly named experiments, with evidence.

The MVP is an **evaluation and training system**, not a promise of a particular public leaderboard score.

## 2. Architecture

```text
Official train.csv
      |
      v
[Data Factory]  --> dataset V0 + frozen prompt-grouped 80/20 split
      |
      v
[Feature Factory] --> train-fitted TF-IDF + stylometry, cached
      |
      v
[Model Factory] --> fit on 80% train / predict shared 20% validation
      |                     | checkpoint + aligned validation probabilities
      v                     v
[Ensemble Factory] --> equal-weight + train-fitted weighted blends
      |
      v
[Evaluation + Experiment Registry] --> SQLite + Parquet + JSON
      |
      v
[Streamlit Dashboard] --> metrics, errors, ablations, costs
```

A single Python CLI/controller calls factories. Every completed step saves immutable artifacts under an experiment ID and can be reused when inputs/configuration have not changed.

## 3. Data Factory

### Steps

1. Read official data and confirm required fields `id`, `prompt_id`, `text`, `model`.
2. Validate unique IDs, nonempty texts, recognised labels, duplicate rows, prompt-group size, language/length distribution; report exceptions.
3. Freeze sorted class order and original text; normalize whitespace only in explicitly versioned derived views.
4. Assign one deterministic 80/20 train/validation split by prompt group. Search a fixed number of candidate `GroupShuffleSplit` assignments and select the split closest to 20% of rows and the full dataset label proportions. Candidate selection uses only labels and IDs, never predictions. Verify every class appears in both partitions and no `prompt_id` crosses partitions. Save assignments and a hash once.
5. Save canonical `dataset_v0.parquet`, `splits_v0.parquet`, `dataset_profile.json`, dataset hash, and per-partition class counts.

**Important:** Fit all statistics, TF-IDF vocabularies, scalers and calibrators using only the 80% train partition. Use the 20% validation partition only for scoring and model comparison; do not train or early-stop on it.

### Future V1 extension

Accept a human-supplied dataset URL or file. Preserve exact source, license, version, generation-model provenance, text, prompt ID if available and parsing decisions. Clean, deduplicate against all known examples, assess source/topic/format bias, and construct a reproducible candidate corpus. Keep official grouped validation examples outside the external training pool when assessing V1.

## 4. Feature Factory

| Feature | Default | Purpose |
|---|---|---|
| Character TF-IDF | 3–5 grams | Writing/typography patterns |
| Word TF-IDF | 1–2 grams | Lexical patterns |
| Handcrafted stylometry | Sentence/word lengths, punctuation, case, digits, function-word rates | Numeric style features for CatBoost |
| Optional punctuation/word view | 1–3 grams | Surface structure |
| Optional shape/function-word view | Explicit transforms | Topic-reduced stylistic patterns |
| Handcrafted stylometry | Sentence/word lengths, quote/dash counts, function-word rates | Diagnostics, later ablations |

MVP SVC begins with char + word features; adding other views is a named ablation. Cache the vectorizer fitted on the train partition and the resulting train/validation sparse matrices. Feature definitions and versions are logged.

## 5. Model Factory

All models expose logically equivalent operations:

```python
fit(train_rows, validation_rows, config) -> Checkpoint
predict_scores(checkpoint, rows) -> ndarray  # [n_rows, 18]
```

All predictions must use the **same stable class order** and row IDs. Store raw logits/decision scores and calibrated probabilities separately.

### Initial models

| Model | Training | Starting settings | Hardware |
|---|---|---|---|
| Character + word TF-IDF + LinearSVC | CPU supervised classification | C=0.5, balanced class weights; calibrate using training-only data | CPU |
| CatBoost + stylometry | CPU multiclass classification over handcrafted row-local style statistics | 800 trees, depth 7, balanced class weights; train-only temperature calibration | CPU |
| UltraBERT-380M (`dmis-lab/UltraBERT-380M`) | Fine-tune 18-class encoder head | 384 tokens, 3 epochs, LR 8e-6, batch 8, gradient accumulation 2, BF16/FP16 as supported | RTX 5070 Ti 16 GB |
| Configurable causal LM (default `Qwen/Qwen3-0.6B`) | LoRA + 18-class readout | 384 tokens, 3 epochs, LoRA rank 16 / alpha 32 / LR 1e-4, batch 4–8; target modules configured per architecture | RTX 5070 Ti 16 GB |

Settings are *initial hypotheses*, not optimized values. UltraBERT-380M is the selected encoder; its model card shows an 8e-6 classification learning rate for the 380M variant. This environment keeps Transformers below 5.20 for vLLM compatibility, so load UltraBERT's custom implementation through `trust_remote_code=True` and a pinned Hugging Face revision. For Qwen, select a deterministic pooling/readout method (e.g. masked mean of specified hidden layers) and log it; early/middle-layer truncation is a **later ablation**, not assumed superior.

Train each model once on the 80% train partition, save probabilities for the shared 20% validation rows, and save the checkpoint. The validation score is used for model selection and is not an unbiased final estimate. Retrain the selected recipe on all labelled data only when preparing an inference model.

## 6. Ensemble Factory

**MVP:** Compare the equal-weight average with one train-fitted weighted average over calibrated 18-class probabilities. Fit weights using only the 80% train rows, on predictions from the fitted models, and regularize weights toward equal shares. Freeze those weights before scoring on the 20% validation rows. Report the weighted and equal-weight validation metrics side by side; do not use validation labels to tune weights. Because training predictions are in-sample, the train fit score is descriptive; the validation comparison shows whether those weights transfer.

For each added model compute:
- Standalone balanced accuracy.
- Ensemble score before vs after the addition, using the same evaluation protocol.
- Pairwise error correlation/disagreement, unique correct examples and oracle upper bound.
- Per-class changes and compute/inference cost.

Log both the best single model and best validated ensemble. Model complexity is justified by *measured incremental gain*.

## 7. Evaluation protocol

### Split and leakage discipline

- Use one frozen 80% train / 20% validation split grouped by `prompt_id`; every model and ensemble uses identical validation IDs.
- Candidate split selection optimizes validation row fraction and per-class label proportions using IDs/groups/labels only. Record seed, candidate count, class counts, and split hash.
- A prompt group appears in exactly one partition. Train-only features, model fitting, and calibration must never use validation rows.
- Validation scores support development decisions but are optimistic after repeated model comparisons. Do not describe them as an untouched holdout or an unbiased final estimate.
- Report prompt-level paired bootstrap intervals for score differences as descriptive uncertainty, not as a cure for repeated validation selection.
- Kaggle public LB is an external secondary check, not the primary optimization target.

### Metrics

| Category | Metric |
|---|---|
| Primary | Balanced accuracy = mean recall over 18 classes |
| Classification | Per-class recall, macro-F1, 18×18 confusion matrix |
| Probability quality | Log loss, calibration/Brier diagnostics (where calibrated) |
| Complementarity | Error overlap, disagreement, unique correct predictions, oracle accuracy, marginal ensemble gain |
| Efficiency | Training wall time, inference throughput, peak GPU VRAM, artifact size |
| Reproducibility | Split seed and hash, dataset hash, code revision, model revision, config hash |

Report model and blend performance on the shared validation split; state that repeated comparison can overfit this validation score.

## 8. MVP ablations

Ablations use the **same frozen V0 80/20 split**:

1. **Statistical features:** char+word SVC; add punctuation/shape; add handcrafted stylometry (as a separate classifier or supported sparse features).
2. **Neural representations:** UltraBERT-380M baseline; Qwen3-0.6B baseline vs a larger small Qwen (e.g. 1.7B) only after baseline works.
3. **Ensemble complementarity:** SVC, UltraBERT, Qwen individually; all pairs; all three; equal-weight blends. Learned weights are deferred.

Predeclare one variable per experiment and save both positive and negative results. The 380M and 1.7B runs are **optional post-MVP ablations**.

## 9. Dashboard

Use **Streamlit + Plotly** reading SQLite and stored predictions. No separate frontend service.

- **Overview:** dataset version, best validation scores, experiments, model roster, training time.
- **Experiments:** sortable comparison table; baseline vs candidate, score delta, confidence interval, runtime, configuration link.
- **Errors:** class recall, 18×18 confusion heatmap, model-pair disagreement, unique corrections.
- **Ablations:** before/after tables for features, model families and ensemble membership.
- **Data:** class, prompt and length distributions; future source/topic distributions.

Every displayed metric includes its source split and experiment ID. Label the split as validation and show its row count. Dashboard examples must never be confused with measured results.

## 10. Storage and reproducibility

```text
configs/{data,features,models,experiments}/
data/raw/  data/processed/
src/{data_factory,feature_factory,model_factory,ensemble_factory,evaluation}/
src/pipeline.py  src/registry.py
artifacts/{features,checkpoints,predictions}/
experiments/registry.sqlite
dashboard/app.py
tests/
```

Use Parquet for row tables and validation predictions, joblib/safetensors for model artifacts where appropriate, YAML for configurations, SQLite for the experiment registry and JSON for metric summaries. Key artifacts by input/config/code hashes.

## 11. Libraries

Python, `pandas`, `numpy`, `pyarrow`, `scipy`, `scikit-learn`, `torch`, `transformers`, `peft`, `accelerate`, `safetensors`, `datasets` (if useful), `streamlit`, `plotly`, `pyyaml`, `pytest` and SQLite (stdlib). Use a locked environment and a tested GPU-compatible PyTorch build.

## 12. MVP acceptance criteria

1. Official data is validated; a fixed prompt-grouped, class-balanced 80/20 split is preserved.
2. SVC, one small encoder and one small causal transformer train on the same 80% and validate on identical 20% rows.
3. Aligned validation scores and probabilities are generated for every validation example, with missing-value checks.
4. Standalone and combined validation balanced accuracy, error overlap, class recall and costs are reproducible.
5. Dashboard correctly reads real experiment artifacts.
6. A Kaggle submission can be packaged and run offline under competition resource constraints.
7. Repeating a run with the same data, code, seed and configuration produces comparable outputs.
8. One restart command clears only generated processed data, artifacts and registry state, then reruns every factory while preserving raw data and cached model downloads.

**Next phase:** Keep selected model recipes fixed, introduce an external corpus V1 through the Data Factory, retrain under the same frozen 80/20 split, compare data effects, and only then revisit model selection.
