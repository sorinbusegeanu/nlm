# Name That LLM — MVP Implementation Plan

**Companion:** `mvp_design.md`  
**Estimated development:** 6–11 developer-days using a coding assistant, excluding unpredictable GPU queue/wall-clock time.  
**Target hardware:** Local RTX 5070 Ti 16 GB; Kaggle T4×2 for optional submission/inference testing.

## 0. Foundation (half day)

Create the repository layout described in `mvp_design.md`. Pin dependencies after checking the local Python/PyTorch/CUDA environment; test minimal loading of the intended UltraBERT and Qwen checkpoints before committing to a training implementation. Store source URLs/revisions in configuration.

Suggested dependencies:

```text
numpy pandas pyarrow scipy scikit-learn
pyyaml joblib pytest
pytorch transformers peft accelerate safetensors
streamlit plotly
```

Use the package name `torch` in actual install commands; choose an appropriate CUDA wheel from the official PyTorch selector. Lock the resulting tested environment rather than relying on the unpinned list above.

CLI contract:

```bash
python -m src.pipeline data prepare --config configs/data/official_v0.yaml
python -m src.pipeline features build --dataset v0
python -m src.pipeline models train --model svc --dataset v0
python -m src.pipeline models train --model ultrabert150 --dataset v0
python -m src.pipeline models train --model qwen06 --dataset v0
python -m src.pipeline ensemble evaluate --dataset v0
streamlit run dashboard/app.py
```

These are **target commands to implement**, not existing executable commands.

## 1. Milestone M1 — Data, folds, SVC baseline (1–2 days)

### Build

- Import official `train.csv`, require `id`, `prompt_id`, `text`, `model`.
- Validate 18 classes, unique IDs, missing text and class distribution.
- Generate a deterministic group-aware **20% official holdout** by `prompt_id`, with 5-fold `StratifiedGroupKFold` on the 80% development portion. Persist the exact row/fold assignments; check group isolation and reasonable class coverage.
- Implement fold-specific char TF-IDF (3–5), word TF-IDF (1–2), `LinearSVC(C=0.5, class_weight='balanced')`.
- Convert decision scores to probabilities through a train-only calibration procedure; keep validation fold disjoint from calibrator fit. Record calibrated and uncalibrated metrics separately.
- Save development OOF predictions and model artifacts.

### Test

- No prompt ID overlap across fit/validation or fit/holdout.
- Every development row has exactly one OOF prediction, with 18 columns in canonical order.
- Fitted vectorizers/calibrators never use validation or holdout rows.
- Identical seed/config reproducibly yields matching folds and approximately matching scores.

**Deliverables:** `dataset_v0.parquet`, `folds_v0.parquet`, `quality_report.json`, SVC checkpoints, OOF Parquet, baseline metric JSON.

## 2. Milestone M2 — Feature and evaluation services (0.5–1 day)

### Build

- Feature Factory interfaces for sparse text views and optional numeric stylometry.
- Versioned feature schemas and fold-fitted vectorizer caching.
- Central evaluator: balanced accuracy, class recall, macro-F1, confusion matrix, log loss (for calibrated probabilities).
- Paired comparison utility for two aligned experiments, with prompt-group bootstrap confidence intervals.
- Experiment record with dataset hash, fold ID, model config, seed, timestamps, run duration and code revision.

### Test

- Feature cache invalidates on feature configuration, data or split change.
- ID/class joins reject mismatches.
- Metrics match direct scikit-learn calculations on synthetic examples.

**Deliverables:** Feature caches, evaluator module, SQLite schema and regression tests.

## 3. Milestone M3 — UltraBERT-150M trainer (1–2 days)

### Build

- Resolve exact supported pretrained checkpoint and Transformers integration; pin the revision.
- Implement tokenizer, 18-class head, deterministic seed and data collator.
- Start at max 384 tokens, 3 epochs, learning rate 2e-5, batch 16, mixed precision where supported. Add gradient accumulation when required.
- Train one fold as a smoke test; then all five grouped development folds.
- Write checkpoints, OOF logits/probabilities, training curves, peak VRAM and latency.

### Test

- Small-batch forward/backward pass and save/load equivalence.
- Training loss changes as expected and no NaNs occur.
- Predictions have shape `[N, 18]` with finite values and normalized probabilities.
- Validation row IDs align exactly with folds.

**Deliverables:** Encoder training adapter, 5-fold OOF probabilities and checkpoint manifests.

## 4. Milestone M4 — Qwen3-0.6B LoRA trainer (1–2 days)

### Build

- Pin `Qwen3-0.6B-Base` checkpoint/revision and tokenizer.
- Implement causal-backbone 18-class readout with a **specified, saved pooling definition** (for example, masked mean of selected hidden layers) and consistent end-token handling.
- Add LoRA with starting rank 16, alpha 32, learning rate 1e-4 and 3 epochs; start with max 384 tokens and batch 4–8, adjusting for 16 GB VRAM.
- Train one fold and inspect learning curves; execute five folds once stable.
- Log trainable parameter count, memory, checkpoint size and inference throughput.

### Test

- LoRA parameters and readout receive gradients; intended base parameters remain frozen.
- Pooling respects attention masks and is stable with padding.
- Reloaded model reproduces logits within precision tolerance.
- No duplicated validation groups, malformed logits or missing OOF rows.

**Deliverables:** Qwen LoRA trainer, readout checkpoint, OOF probabilities, runtime and VRAM reports.

## 5. Milestone M5 — Ensemble, complementarity and ablations (1 day)

### Build

- Align all OOF files by row ID and canonical class mapping.
- Compute single-model metrics plus equal-weight probability blends for SVC + UltraBERT, SVC + Qwen, UltraBERT + Qwen and all three.
- Calculate pairwise disagreement, error co-occurrence, unique correct counts, theoretical oracle accuracy, per-class recall changes.
- Implement a regularized weight-fit experiment with **nested/cross-fitted meta-evaluation**; retain equal-weight baseline as control.
- Generate the predefined feature and ensemble ablation tables.

### Test

- Shuffling OOF input rows leaves joined predictions/metrics unchanged.
- Duplicated/missing IDs fail loudly.
- Weight fitting never scores on its fitting rows.
- Summed probability weights and output class probabilities are valid.

**Deliverables:** Ensemble reports, error/ablation Parquet or JSON, selected ensemble manifest.

## 6. Milestone M6 — Dashboard and reporting (1–2 days)

### Build

Create one Streamlit app with:

1. **Overview:** dataset V0, number of experiments, best grouped-CV score, held-out confirmation (when released), compute usage.
2. **Experiment comparison:** model/config table, delta vs fixed baseline, fold consistency, cost.
3. **Error analysis:** 18×18 confusion heatmap, class recall and model disagreement table.
4. **Ablations:** effects of features, model additions and blend methods.
5. **Data profile:** class counts, prompt groups, passage lengths and formatting.

All figures read actual artifacts and clearly label `development OOF`, `holdout` or `Kaggle LB`. Provide export of experiment reports as CSV/JSON.

### Test

- Dashboard loads with an empty registry and with one run.
- Scores are the evaluator outputs, not recomputed with different joins or split logic.
- Every row links to experiment metadata.

**Deliverable:** `dashboard/app.py` plus sample screenshots from genuine runs.

## 7. Milestone M7 — Final training and Kaggle submission (1–2 days)

### Build

- Choose the best validated single model or compact ensemble using development OOF; perform a **one-time** 20% holdout confirmation before release selection.
- Retrain selected base models on all officially available labelled data for final inference, using frozen recipes. Keep holdout-evaluation checkpoints/reports distinct from final full-data checkpoints.
- Create a Kaggle notebook that loads only attached model/input artifacts, with internet disabled.
- Verify the exact required submission schema/filenames against the competition rules/sample submission; generate `submission.parquet` accordingly.
- Measure end-to-end runtime and maximum memory on the intended Kaggle GPU/CPU configuration.
- Submit and record the public leaderboard score as secondary evidence.

### Test

- Inference works with internet disabled and clean environment.
- Sample submission rows/IDs exactly match test IDs; classes use competition label order.
- No missing, infinite or invalid predictions.
- Runtime fits submission limits with a safety margin.

**Deliverables:** Export notebook, immutable selected-model manifests, final artifacts and Kaggle score record.

## 8. Initial experiment matrix

Run the smallest useful set of comparisons on fixed **dataset V0**:

| ID | Question | Change | Main measurement |
|---|---|---|---|
| E01 | Statistical baseline | Char + word SVC | Grouped OOF balanced accuracy |
| E02 | Do formatting cues help? | Add punctuation/shape view | Δ balanced accuracy vs E01 |
| E03 | Does a small encoder help? | UltraBERT-150M | Standalone score and unique corrections |
| E04 | Does small causal LM help? | Qwen3-0.6B LoRA | Standalone score and unique corrections |
| E05 | Does ensemble diversity help? | Each pair and all three | Δ ensemble balanced accuracy |
| E06 | Do learned weights help? | Cross-fitted regularized blend | Δ vs equal-weight blend |

Optional after MVP: UltraBERT-380M, Qwen3-1.7B, additional stylometry and a comparison with published public checkpoints. A new model should enter the ensemble only after evaluating its marginal contribution.

## 9. Data-version experiments after V0

Once V0 is stable:

1. Human provides an external dataset URL/file.
2. Data Factory validates provenance and label mapping, deduplicates, profiles source/topic/format bias and produces **candidate V1**.
3. Keep validation folds, model architectures, seeds and training recipes fixed when comparing V0 to V1.
4. Train inexpensive SVC comparison first as a **diagnostic**, then a selected neural model to test predictive usefulness.
5. Evaluate per-class gains, ensemble gain, runtime and uncertainty; log the decision and artifacts.
6. Revisit model diversity when the corpus changes materially.

Do not automatically reject good-quality data solely because a cheap proxy classifier fails to improve; that proxy does not establish neural-model utility.

## 10. Definition of done

- All four factories operate end-to-end from official CSV to aligned OOF predictions.
- Fixed prompt-grouped folds and leakage checks pass.
- SVC, UltraBERT-150M and Qwen3-0.6B each produce reproducible 18-class probabilities.
- Dashboard shows real measurements, model complementarity, ablations and runtime.
- Selected ensemble uses leakage-safe evaluation of learned weights, if enabled.
- Offline Kaggle notebook runs and produces a valid competition submission artifact.
- One documented baseline V0 exists for future external-data comparisons.
