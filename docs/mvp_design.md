# Name That LLM — MVP Design

**Version:** 0.1  
**Scope:** Official competition training data only (2,699 labelled passages, 18 classes, approximately 150 prompt groups).  
**Purpose:** Establish a reproducible baseline for model comparison, ensemble complementarity and future dataset experiments.

## 1. Goals and boundaries

The MVP must:

1. Train and evaluate three different classifiers on identical prompt-grouped folds: four-view TF-IDF + LinearSVC, UltraBERT-150M and Qwen3-0.6B-Base with LoRA.
2. Produce aligned out-of-fold (OOF) scores for all 18 model labels, and a simple ensemble.
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
[Data Factory]  --> dataset V0 + frozen prompt-group CV folds
      |
      v
[Feature Factory] --> fold-fitted TF-IDF + stylometry, cached
      |
      v
[Model Factory] --> SVC / UltraBERT-150M / Qwen3-0.6B LoRA
      |                     | checkpoints + aligned OOF probabilities
      v                     v
[Ensemble Factory] --> equal / regularized weighted probability blends
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
4. Assign **five fixed prompt-grouped folds**, using `StratifiedGroupKFold` when suitable; verify no `prompt_id` occurs in both sides. Save fold assignments once.
5. Save canonical `dataset_v0.parquet`, `folds_v0.parquet`, `dataset_profile.json`, dataset hash.

**Important:** Fit all statistics, TF-IDF vocabularies, scalers and calibrators using only the training portion of each fold. Neural validation examples are never used for training or early stopping beyond the predefined validation logic.

### Future V1 extension

Accept a human-supplied dataset URL or file. Preserve exact source, license, version, generation-model provenance, text, prompt ID if available and parsing decisions. Clean, deduplicate against all known examples, assess source/topic/format bias, and construct a reproducible candidate corpus. Keep official grouped validation examples outside the external training pool when assessing V1.

## 4. Feature Factory

| Feature | Default | Purpose |
|---|---|---|
| Character TF-IDF | 3–5 grams | Writing/typography patterns |
| Word TF-IDF | 1–2 grams | Lexical patterns |
| Optional punctuation/word view | 1–3 grams | Surface structure |
| Optional shape/function-word view | Explicit transforms | Topic-reduced stylistic patterns |
| Handcrafted stylometry | Sentence/word lengths, quote/dash counts, function-word rates | Diagnostics, later ablations |

MVP SVC begins with char + word features; adding other views is a named ablation. Cache fitted fold-specific vectorizers and resulting sparse matrices. Feature definitions and versions are logged.

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
| Four-view TF-IDF + LinearSVC | CPU supervised classification | C=0.5, balanced class weights; calibrate using training-only data | CPU |
| UltraBERT-150M | Fine-tune 18-class encoder head | 384 tokens, 3 epochs, LR 2e-5, batch 16, BF16/FP16 as supported | RTX 5070 Ti 16 GB |
| Qwen3-0.6B-Base | LoRA + 18-class readout | 384 tokens, 3 epochs, LoRA rank 16 / alpha 32 / LR 1e-4, batch 4–8 | RTX 5070 Ti 16 GB |

Settings are *initial hypotheses*, not optimized values. Pin exact model repos/revisions and compatible software versions after a smoke test. Confirm UltraBERT's actual checkpoint/API availability before implementing its trainer; use an equivalently small supported encoder if unavailable, documenting the substitution. For Qwen, select a deterministic pooling/readout method (e.g. masked mean of specified hidden layers) and log it; early/middle-layer truncation is a **later ablation**, not assumed superior.

Train each fold independently, save OOF probabilities on its validation rows, and save inference-ready checkpoints. Train full-data models only after selecting a final experiment.

## 6. Ensemble Factory

**MVP:** Equal-weight average of calibrated 18-class probabilities.

Then compare with a small, explicitly regularized global-weight blend; learned weights must be evaluated on prediction rows not used to choose those weights (nested/cross-fitted meta-evaluation). Do not report stacking training accuracy as validation accuracy.

For each added model compute:
- Standalone balanced accuracy.
- Ensemble score before vs after the addition, using the same evaluation protocol.
- Pairwise error correlation/disagreement, unique correct examples and oracle upper bound.
- Per-class changes and compute/inference cost.

Log both the best single model and best validated ensemble. Model complexity is justified by *measured incremental gain*.

## 7. Evaluation protocol

### Split and leakage discipline

- **Development:** Five fixed `prompt_id`-grouped folds over official data; all candidate models use identical fold IDs.
- **Optional final check:** A separate ~20% prompt-group holdout can be carved out *before* CV when a truly untouched confirmation set is desired. If used, the five folds run on the remaining ~80% only. Choose one protocol at project start and freeze it; the MVP default is **80% development / 20% untouched holdout**, with 5-fold grouped CV inside development.
- OOF predictions are generated only by a model trained without that row's prompt group.
- For learned ensemble weights or meta-classifiers use nested/cross-fitted evaluation over OOF predictions; avoid evaluating and fitting on the same rows.
- Report prompt-level paired bootstrap intervals for score differences where informative.
- Kaggle public LB is an external secondary check, not the primary optimization target.

### Metrics

| Category | Metric |
|---|---|
| Primary | Balanced accuracy = mean recall over 18 classes |
| Classification | Per-class recall, macro-F1, 18×18 confusion matrix |
| Probability quality | Log loss, calibration/Brier diagnostics (where calibrated) |
| Complementarity | Error overlap, disagreement, unique correct predictions, oracle accuracy, marginal ensemble gain |
| Efficiency | Training wall time, inference throughput, peak GPU VRAM, artifact size |
| Reproducibility | Seed, folds, dataset hash, code revision, model revision, config hash |

Report performance separately on grouped CV and the untouched holdout; do not combine them into one score.

## 8. MVP ablations

Ablations use **fixed V0 data and folds**:

1. **Statistical features:** char+word SVC; add punctuation/shape; add handcrafted stylometry (as a separate classifier or supported sparse features).
2. **Neural representations:** UltraBERT-150M vs UltraBERT-380M if resources permit; Qwen3-0.6B baseline vs a larger small Qwen (e.g. 1.7B) only after baseline works.
3. **Ensemble complementarity:** SVC, UltraBERT, Qwen individually; all pairs; all three; equal vs cross-fitted learned weights.

Predeclare one variable per experiment and save both positive and negative results. The 380M and 1.7B runs are **optional post-MVP ablations**.

## 9. Dashboard

Use **Streamlit + Plotly** reading SQLite and stored predictions. No separate frontend service.

- **Overview:** dataset version, best CV/holdout scores, experiments, model roster, training time.
- **Experiments:** sortable comparison table; baseline vs candidate, score delta, confidence interval, runtime, configuration link.
- **Errors:** class recall, 18×18 confusion heatmap, model-pair disagreement, unique corrections.
- **Ablations:** before/after tables for features, model families and ensemble membership.
- **Data:** class, prompt and length distributions; future source/topic distributions.

Every displayed metric includes its source split and experiment ID. Dashboard examples must never be confused with measured results.

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

Use Parquet for row tables and OOF predictions, joblib/safetensors for model artifacts where appropriate, YAML for configurations, SQLite for the experiment registry and JSON for metric summaries. Key artifacts by input/config/code hashes.

## 11. Libraries

Python, `pandas`, `numpy`, `pyarrow`, `scipy`, `scikit-learn`, `torch`, `transformers`, `peft`, `accelerate`, `safetensors`, `datasets` (if useful), `streamlit`, `plotly`, `pyyaml`, `pytest` and SQLite (stdlib). Use a locked environment and a tested GPU-compatible PyTorch build.

## 12. MVP acceptance criteria

1. Official data is validated; model order and fixed grouped folds are preserved.
2. SVC, one small encoder and one small causal transformer train using identical validation rows.
3. Aligned OOF scores and probabilities are generated for every development example, with missing-value checks.
4. Standalone and combined balanced accuracy, error overlap, class recall and costs are reproducible.
5. Dashboard correctly reads real experiment artifacts.
6. A Kaggle submission can be packaged and run offline under competition resource constraints.
7. Repeating a run with the same data, code, seed and configuration produces comparable outputs.

**Next phase:** Keep selected model recipes fixed, introduce an external corpus V1 through the Data Factory, retrain under the same folds, compare data effects, and only then revisit model selection.
