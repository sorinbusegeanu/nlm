# Trusted Checkpoint Ensemble — UltraBERT

## Method and validation boundary

The analysis uses 540 frozen V0 rows, 18 labels and 30 prompt groups. Five-fold GroupKFold keeps every prompt group wholly in one fold. Specialist assignments, precision smoothing and minimum-support decisions are fitted on the other four folds only; routing is evaluated on the held-out fold. The four checkpoints and their roster were previously selected using V0, so these are development diagnostics, not independent generalization estimates.

Reliability rule: minimum predicted-label support **10**. Precision is smoothed with Beta(α=1, β=1): `(TP+α)/(predicted_count+α+β)`. For each label the specialist is selected by highest fitting-fold recall, then raw precision, then F1; exact ties use candidate order `A, B, C, D`. For routing, eligible proposals rank by assigned-specialist flag, smoothed precision, specialist recall, candidate order and canonical label order. If none meet support, route to D.

Split SHA-256: `147ce64eb0f54e5536807603701d11f373834e72f627262fb4c9436dd380bcab`. Validation ID SHA-256: `d3f39a2003dfdd9e99e5d662465066bd60368f8e43864b6ed10a733edbe74f9d`.

## A. Specialist coverage matrix

The highest-recall checkpoint and best-precision checkpoint columns are descriptive measurements over all 540 saved predictions (the checkpoints were already selected using V0). Fold assignments show the independently re-fit experts used by the OOF router; their OOF specialist-only metrics are reported in section C.

| Label | Highest recall checkpoint | Recall | Precision | Best precision checkpoint | Precision | Recall | Fold assignment counts |
|---|---|---:|---:|---|---:|---:|---|
| claude-fable-5 | D | 53.3% | 43.2% | A | 47.1% | 26.7% | D:5 |
| claude-opus-4-6 | D | 86.7% **≥80%** | 72.2% | B | 85.0% | 56.7% | D:5 |
| claude-opus-4-8 | B | 70.0% | 56.8% | D | 84.2% | 53.3% | B:5 |
| claude-opus-5 | C | 70.0% | 75.0% | D | 76.9% | 66.7% | B:1, C:4 |
| claude-sonnet-5 | D | 86.7% **≥80%** | 59.1% | B | 76.2% | 53.3% | D:5 |
| gemini-3.1-pro-preview | D | 90.0% **≥90%** | 77.1% | C | 96.0% | 80.0% | D:5 |
| gemini-3.6-flash | D | 76.7% | 53.5% | C | 54.2% | 43.3% | D:5 |
| gemini-3.7-flash | C | 73.3% | 40.0% | B | 47.6% | 33.3% | C:5 |
| gemini-3.8-flash | B | 63.3% | 55.9% | C | 73.3% | 36.7% | A:1, B:4 |
| glm-5.3 | C | 30.0% | 26.5% | D | 28.0% | 23.3% | A:1, B:1, C:3 |
| glm-5.3-flash | B | 53.3% | 32.7% | C | 42.9% | 50.0% | B:3, C:2 |
| gpt-5.5 | D | 76.7% | 41.8% | C | 53.6% | 50.0% | D:5 |
| gpt-5.6-luna | C | 40.0% | 40.0% | C | 40.0% | 40.0% | A:3, C:2 |
| gpt-5.6-sol | A | 50.0% | 45.5% | D | 55.0% | 36.7% | A:3, B:2 |
| gpt-5.6-terra | B | 66.7% | 41.7% | A | 44.4% | 53.3% | B:5 |
| gpt-6-astra | A | 76.7% | 92.0% | B | 95.5% | 70.0% | A:5 |
| kimi-k3 | B | 53.3% | 53.3% | D | 57.1% | 26.7% | B:5 |
| qwen3.8-max | C | 70.0% | 72.4% | C | 72.4% | 70.0% | C:5 |

## B. Trusted ensemble per-label performance

TP counts are out of 30 true examples per label. The specialist column shows the most frequently fitted expert, its fold count, and which checkpoint actually supplied this label's routed predictions.

| Label | D recall (TP/30) | Best individual specialist recall (TP/30) | Trusted recall (TP/30) | Trusted precision | Selected expert / routed rows |
|---|---:|---:|---:|---:|---|
| claude-fable-5 | 53.3% (16/30) | 53.3% (16/30) | 43.3% (13/30) | 38.2% | D (5/5); A:1, D:33 |
| claude-opus-4-6 | 86.7% (26/30) | 86.7% (26/30) | 86.7% (26/30) | 72.2% | D (5/5); D:36 |
| claude-opus-4-8 | 53.3% (16/30) | 70.0% (21/30) | 50.0% (15/30) | 75.0% | B (5/5); B:18, D:2 |
| claude-opus-5 | 66.7% (20/30) | 66.7% (20/30) | 66.7% (20/30) | 64.5% | C (4/5); B:9, C:21, D:1 |
| claude-sonnet-5 | 86.7% (26/30) | 86.7% (26/30) | 86.7% (26/30) | 59.1% | D (5/5); D:44 |
| gemini-3.1-pro-preview | 90.0% (27/30) | 90.0% (27/30) | 90.0% (27/30) | 77.1% | D (5/5); D:35 |
| gemini-3.6-flash | 76.7% (23/30) | 76.7% (23/30) | 70.0% (21/30) | 53.8% | D (5/5); D:39 |
| gemini-3.7-flash | 26.7% (8/30) | 73.3% (22/30) | 23.3% (7/30) | 46.7% | C (5/5); C:15 |
| gemini-3.8-flash | 46.7% (14/30) | 53.3% (16/30) | 53.3% (16/30) | 51.6% | B (4/5); A:5, B:25, C:1 |
| glm-5.3 | 23.3% (7/30) | 13.3% (4/30) | 6.7% (2/30) | 16.7% | C (3/5); A:3, B:2, C:5, D:2 |
| glm-5.3-flash | 30.0% (9/30) | 43.3% (13/30) | 46.7% (14/30) | 35.9% | B (3/5); B:22, C:16, D:1 |
| gpt-5.5 | 76.7% (23/30) | 76.7% (23/30) | 50.0% (15/30) | 55.6% | D (5/5); B:1, D:26 |
| gpt-5.6-luna | 36.7% (11/30) | 30.0% (9/30) | 26.7% (8/30) | 33.3% | A (3/5); A:10, C:13, D:1 |
| gpt-5.6-sol | 36.7% (11/30) | 40.0% (12/30) | 36.7% (11/30) | 40.7% | A (3/5); A:18, B:7, D:2 |
| gpt-5.6-terra | 46.7% (14/30) | 66.7% (20/30) | 43.3% (13/30) | 38.2% | B (5/5); A:1, B:32, C:1 |
| gpt-6-astra | 63.3% (19/30) | 76.7% (23/30) | 76.7% (23/30) | 88.5% | A (5/5); A:25, D:1 |
| kimi-k3 | 26.7% (8/30) | 53.3% (16/30) | 53.3% (16/30) | 50.0% | B (5/5); B:29, C:1, D:2 |
| qwen3.8-max | 66.7% (20/30) | 70.0% (21/30) | 70.0% (21/30) | 61.8% | C (5/5); B:5, C:28, D:1 |

## C. High-accuracy label coverage

Best individual specialists are selected within each fitting fold and measured on held-out rows. The additional any-individual-checkpoint column is a pooled descriptive diagnostic over all V0 predictions and is optimistic because it selects the best checkpoint per label after seeing these same labels. Precision requires at least the configured minimum predicted-label support. D and trusted-router values are ordinary multiclass one-vs-rest metrics.

| Metric | D | Best individual specialists (OOF) | Any individual checkpoint (pooled) | Trusted ensemble (OOF) |
|---|---:|---:|---:|---:|
| Labels with recall ≥80% | 3 | 3 | 3 | 3 |
| Labels with recall ≥90% | 1 | 1 | 1 | 1 |
| Labels with precision ≥80% | 2 | 1 | 4 | 1 |
| Labels with precision ≥90% | 0 | 1 | 2 | 0 |
| Labels with both precision and recall ≥80% | 0 | 0 | 1 | 0 |
| Labels with both precision and recall ≥90% | 0 | 0 | 0 | 0 |

## D. Routing analysis

### Checkpoint usage

| Checkpoint | Selected predictions | Correct | Incorrect |
|---|---:|---:|---:|
| A | 63 | 38 | 25 |
| B | 150 | 70 | 80 |
| C | 101 | 56 | 45 |
| D | 226 | 130 | 96 |

Rows with conflicting proposals (different top-1 labels): 312 / 540; mean distinct proposed labels: 1.76. Fallback to D for insufficient support: 0 rows.

Compared with D, trusted routing corrected 37 rows and regressed 41; identical top-1 label on 428 rows.

Per-checkpoint, per-label selected correct/incorrect counts:

**A**
| Label | Routed rows | Correct | Incorrect |
|---|---:|---:|---:|
| claude-fable-5 | 1 | 0 | 1 |
| gemini-3.8-flash | 5 | 3 | 2 |
| glm-5.3 | 3 | 0 | 3 |
| gpt-5.6-luna | 10 | 3 | 7 |
| gpt-5.6-sol | 18 | 8 | 10 |
| gpt-5.6-terra | 1 | 1 | 0 |
| gpt-6-astra | 25 | 23 | 2 |

**B**
| Label | Routed rows | Correct | Incorrect |
|---|---:|---:|---:|
| claude-opus-4-8 | 18 | 15 | 3 |
| claude-opus-5 | 9 | 3 | 6 |
| gemini-3.8-flash | 25 | 13 | 12 |
| glm-5.3 | 2 | 0 | 2 |
| glm-5.3-flash | 22 | 7 | 15 |
| gpt-5.5 | 1 | 1 | 0 |
| gpt-5.6-sol | 7 | 2 | 5 |
| gpt-5.6-terra | 32 | 12 | 20 |
| kimi-k3 | 29 | 16 | 13 |
| qwen3.8-max | 5 | 1 | 4 |

**C**
| Label | Routed rows | Correct | Incorrect |
|---|---:|---:|---:|
| claude-opus-5 | 21 | 17 | 4 |
| gemini-3.7-flash | 15 | 7 | 8 |
| gemini-3.8-flash | 1 | 0 | 1 |
| glm-5.3 | 5 | 1 | 4 |
| glm-5.3-flash | 16 | 6 | 10 |
| gpt-5.6-luna | 13 | 5 | 8 |
| gpt-5.6-terra | 1 | 0 | 1 |
| kimi-k3 | 1 | 0 | 1 |
| qwen3.8-max | 28 | 20 | 8 |

**D**
| Label | Routed rows | Correct | Incorrect |
|---|---:|---:|---:|
| claude-fable-5 | 33 | 13 | 20 |
| claude-opus-4-6 | 36 | 26 | 10 |
| claude-opus-4-8 | 2 | 0 | 2 |
| claude-opus-5 | 1 | 0 | 1 |
| claude-sonnet-5 | 44 | 26 | 18 |
| gemini-3.1-pro-preview | 35 | 27 | 8 |
| gemini-3.6-flash | 39 | 21 | 18 |
| glm-5.3 | 2 | 1 | 1 |
| glm-5.3-flash | 1 | 1 | 0 |
| gpt-5.5 | 26 | 14 | 12 |
| gpt-5.6-luna | 1 | 0 | 1 |
| gpt-5.6-sol | 2 | 1 | 1 |
| gpt-6-astra | 1 | 0 | 1 |
| kimi-k3 | 2 | 0 | 2 |
| qwen3.8-max | 1 | 0 | 1 |

Specialist assignments were re-fit in each fold. Fold assignment counts appear in the coverage matrix; full fold-by-fold confusion counts are in `specialist_assignments.csv` and `specialist_reliability.json`.

## E. Overall performance

| Method | Balanced accuracy | Total correct | Macro F1 | Top-3 | MRR | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 55.2% | 298/540 | 54.2% | 86.3% | 0.7166 | 2.5716 | 0.7288 |
| B+D equal probability | 56.7% | 306/540 | 56.3% | 88.0% | 0.7265 | 1.6213 | 0.6220 |
| Trusted checkpoint ensemble | 54.4% | 294/540 | 53.0% | N/A | N/A | N/A | N/A |
| Four-checkpoint top-1 oracle (theoretical) | 76.5% | 413/540 | N/A | N/A | N/A | N/A | N/A |

## F. Conclusions

1. **Labels at ≥80% recall:** 3 of 18 after routing.
2. **Labels at ≥90% recall:** 1 of 18 after routing.
3. **Reliable specialists:** The coverage matrix shows best observed per-label checkpoints; fold assignment counts show which specialists the OOF fitting folds selected.
4. **Recall/precision trade-off:** trusted routing yields 0 labels with both precision and recall ≥80%, and 0 with both ≥90%. The routed predictions' false positives are shown per label above.
5. **Overall balanced accuracy:** 54.4% versus D 55.2% and B+D 56.7%.
6. **Main limit:** 312 rows had competing labels; see corrected/regressed counts and class-specific false positives to assess whether conflicts or weak expert coverage dominate.
7. **Other architectures:** Yes. The router consumes candidate IDs, ordered classes and aligned probability matrices; model-specific code is confined to loading the saved candidates.

The four-checkpoint oracle reaches 76.5% by using the true label to choose a correct checkpoint. It is an upper bound, not a prediction method. All results use V0, which had previously informed checkpoint selection.

Artifacts: `oof_predictions.parquet`, `routing_decisions.parquet`, `specialist_assignments.csv`, `specialist_reliability.json`, and `metrics.json`.
