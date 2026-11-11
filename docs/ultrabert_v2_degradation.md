# UltraBERT V2.0/V2.1 degradation investigation

## Executive summary

The V1/V2 gap is reproduced on the same 2,699 V0 rows. The supplied per-label values are precision, not recall. All three source datasets are exactly class-balanced, so simple class imbalance is ruled out.

The strongest shared configuration lead is sequence length: V1 recorded 384 tokens, while V2.0 and V2.1 automatically selected 1,595 from the longest combined train/validation response. V2's memory probe then reduced its training microbatch from 8 to 4 (with gradient accumulation 2). V2 validation loss and accuracy peak at epoch 2 and deteriorate afterward while train loss continues falling, consistent with overfitting or unstable optimization; this does not by itself isolate the cause.

V2.1 includes 2,135 unverified Gemini 3.7 rows (71.2% of that class), but V2.0 without those rows degrades similarly. That addition is a specific V2.1 risk, not an explanation for the common V2 drop. The old V1 manifest also points to a 51,497-row data snapshot whose hash is not present in the current V1 or V1.5 profiles, limiting exact data-construction attribution.

## Common V0 comparison

Evaluation uses the same V0 IDs and labels for all cached checkpoints. V2 epoch selection used the full V0 set; its metrics are therefore selection-contaminated and should not be read as an unbiased final estimate.

| version   |   rows |   accuracy |   balanced_accuracy |   macro_f1 |
|:----------|-------:|-----------:|--------------------:|-----------:|
| V1        |   2699 |     0.8807 |              0.8807 |     0.8813 |
| V2.0      |   2699 |     0.4939 |              0.4938 |     0.4796 |
| V2.1      |   2699 |     0.4861 |              0.4860 |     0.4802 |

| Label | V1 precision | V2.0 precision | V2.1 precision | V1 recall | V2.0 recall | V2.1 recall |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-astra | 98.6% | 76.8% | 77.9% | 95.3% | 70.7% | 70.7% |
| gemini-3.1-pro-preview | 98.6% | 62.4% | 93.1% | 94.0% | 92.0% | 72.0% |
| claude-opus-4-6 | 94.7% | 48.5% | 74.3% | 94.7% | 85.3% | 67.3% |
| qwen3.8-max | 92.7% | 62.4% | 50.0% | 93.3% | 65.3% | 78.0% |

Full per-label precision/recall/support/predicted-count metrics: [`v0_per_label_metrics.csv`](../artifacts/analysis/ultrabert_v2_degradation/v0_per_label_metrics.csv). Confusion matrices and error-pair summaries are in the same analysis directory.

## Data construction and label evidence

Class counts are exact and uniform:

- **V1:** 48,798 rows, 18 labels, 2,711–2,711 rows/class; profile SHA256 `35cd4dd79c77e13807c49eee5197f4f69c377e49cfdc2ed7c2c260e763341a0f`.
- **V2.0:** 51,606 rows, 18 labels, 2,867–2,867 rows/class; profile SHA256 `20d81275a9cfc414b4ec37ba54cdaf17d0bee2d68bdb8d24c0f7d53ebf01232c`.
- **V2.1:** 54,000 rows, 18 labels, 3,000–3,000 rows/class; profile SHA256 `93cdf13bf1a809c6ad42803268880ee02d3bcd9209cba4f3b42b3577e182c79c`.

Evidence field coverage (non-missing rows; source-specific breakdown is in the CSV):

| field                    | V1     | V2.0   | V2.1   |
|:-------------------------|:-------|:-------|:-------|
| match_weight             | 100.0% | 85.3%  | 81.9%  |
| model_version            | n/a    | 14.7%  | 14.1%  |
| panel_labels             | 100.0% | 85.3%  | 81.9%  |
| prompt_evidence          | 100.0% | 85.3%  | 81.9%  |
| provider                 | n/a    | 14.7%  | 14.1%  |
| route                    | n/a    | 12.4%  | 11.8%  |
| source_model_attribution | n/a    | 2.3%   | 6.3%   |


Normalized text overlap is measured using NFKC, casefolding, and collapsed whitespace. V2.0 and V2.1 share 49,604 unique normalized responses. V2 training rows have 0 / 0 normalized-response overlaps with V0 and 0 / 0 prompt-ID overlaps.
V2.0/V2.1 have 0 / 0 normalized-text overlaps and 0 / 0 prompt-ID overlaps with the V1.5 540-row holdout.

V2.1's unverified Gemini 3.7 cohort is explicitly marked unverified in the source profile; no attribution is inferred from missing metadata alone. A conservative provider-family check flagged 0 explicit candidate contradictions for source-level review; this is a review queue, not proof of mislabeling. Source composition, per-label source concentration, attribution-field coverage, duplicate groups, and cross-label conflicts are in `source_composition_and_evidence.csv`, `source_diversity_by_label.csv`, `evidence_field_coverage.csv`, `duplicate_audit.csv`, and `provider_conflict_candidates.csv`.

## Sequence length and training dynamics

| dataset      | source_type   |   rows |   p50_tokens |   p90_tokens |   p95_tokens |   p99_tokens |   max_tokens |   rows_gt_384_tokens |   rows_gt_512_tokens |   rows_gt_1024_tokens |   rows_gt_1500_tokens |
|:-------------|:--------------|-------:|-------------:|-------------:|-------------:|-------------:|-------------:|---------------------:|---------------------:|----------------------:|----------------------:|
| V1           | all           |  48798 |        129.0 |        260.0 |        287.0 |        324.0 |          494 |                   22 |                    0 |                     0 |                     0 |
| V2.0         | all           |  51606 |        133.0 |        268.0 |        296.0 |        414.0 |         1595 |                  628 |                  199 |                     3 |                     1 |
| V2.1         | all           |  54000 |        132.0 |        267.0 |        296.0 |        415.0 |         1595 |                  661 |                  211 |                     3 |                     1 |
| V1.5_holdout | all           |    540 |        129.5 |        253.0 |        273.2 |        308.0 |          329 |                    0 |                    0 |                     0 |                     0 |
| V0           | all           |   2699 |        130.0 |        259.0 |        281.0 |        312.0 |          372 |                    0 |                    0 |                     0 |                     0 |

| version   |   epoch |   step |   train_loss |   eval_loss |   eval_accuracy |   eval_balanced_accuracy |
|:----------|--------:|-------:|-------------:|------------:|----------------:|-------------------------:|
| V1        |       1 |   3185 |       3.2038 |    nan      |        nan      |                 nan      |
| V1        |       2 |   6370 |       1.8261 |    nan      |        nan      |                 nan      |
| V1        |       3 |   9555 |       0.6258 |    nan      |        nan      |                 nan      |
| V2.0      |       1 |   6451 |       3.1878 |      1.4555 |          0.4561 |                   0.4562 |
| V2.0      |       2 |  12902 |       2.0114 |      1.3422 |          0.4939 |                   0.4938 |
| V2.0      |       3 |  19353 |       0.8988 |      2.0958 |          0.4639 |                   0.4638 |
| V2.0      |       4 |  25804 |       0.4322 |      4.0181 |          0.4909 |                   0.4909 |
| V2.1      |       1 |   6750 |       3.1214 |      1.4017 |          0.4557 |                   0.4556 |
| V2.1      |       2 |  13500 |       1.8843 |      1.4276 |          0.4861 |                   0.4860 |
| V2.1      |       3 |  20250 |       0.8386 |      2.1241 |          0.4820 |                   0.4820 |
| V2.1      |       4 |  27000 |       0.4564 |      4.6923 |          0.4676 |                   0.4675 |

Token-length counts use the saved UltraBERT tokenizer and do not truncate. Only 628/51,606 V2.0 rows (1.22%) and 661/54,000 V2.1 rows (1.22%) exceed V1's 384-token limit; 3 examples in each exceed 1,024. Nearly all rows above 384 tokens come from the V1.3 source family. The 1,595 limit is set by a single long-tail response, so its direct truncation effect applies to a small minority, though those long rows can still change memory use and batch probing.
The V1 run logged 3,185 optimizer steps/epoch and ended at step 9,555. V2.0 used 6,451 steps/epoch and selected epoch 2 at step 12,902; V2.1 used 6,750 steps/epoch and selected epoch 2 at step 13,500. With the configured gradient accumulation of 2, the probed V2 microbatch 4 implies an optimizer batch of 8, versus V1's configured 8×2=16. This is a substantial training-dynamics confound alongside sequence length.

## Independent V2 transfer check

The saved V2 checkpoints were run once on the V1.5 prompt-group holdout, which has zero normalized-text and prompt-ID overlap with V2 training. V1 selected its own checkpoint on this holdout, so these results are reported for V2 transfer only, not as an unbiased head-to-head comparison.

| version   |   rows |   accuracy |   balanced_accuracy |   macro_f1 |   elapsed_seconds |
|:----------|-------:|-----------:|--------------------:|-----------:|------------------:|
| V2.0      |    540 |     0.4870 |              0.4870 |     0.4722 |            5.9982 |
| V2.1      |    540 |     0.4833 |              0.4833 |     0.4798 |            5.4894 |

Per-label holdout results and predictions are in `v1_5_holdout_per_label_metrics.csv` and `v1_5_holdout_predictions.parquet`.

## Limits and next causal test

This audit establishes reproducible associations, not a causal winner among data source, sequence length, and optimizer dynamics. V2.0 and V2.1 share most normalized response text, and both use the same newer code path; their similar V0 scores narrow the shared cause but do not prove it. The V1 model's original training data snapshot cannot be reconstructed from the current profiles.

If this evidence does not settle the cause, the next experiment should train matched checkpoints with a sealed prompt-group evaluation set and vary one factor at a time: 384 versus automatic sequence length; V2.0 versus V2.1 source composition; and V2.1 with versus without the unverified Gemini 3.7 rows. Do not use the sealed evaluation groups for epoch selection.

## Reproducibility

Analysis script: [`analyze_ultrabert_v2_degradation.py`](../scripts/analyze_ultrabert_v2_degradation.py). Output directory: `/home/zodrak/nlm/artifacts/analysis/ultrabert_v2_degradation`. Input manifests, dataset profile hashes, inference runtimes, batch sizes, and overlap checks are saved in `audit_summary.json`.
