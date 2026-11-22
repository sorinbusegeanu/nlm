# E20-D Kaggle Failure Analysis

## Finding

The packaged E20-D scoring path reproduces its saved V0 OOF candidate scores and Hungarian labels exactly within floating-point tolerance. The deployment gap is therefore not explained by a class-order, checkpoint-probability, E13-router, CatBoost feature-order, or assignment-code mismatch. Group distribution is a confirmed contributor: full-batch deployment-style groups cost 1.96 pp versus the original OOF setup, oracle group features add 2.33 pp over reconstructed features on held-out rows, and the corrected incomplete-group simulation reaches 70.79% BA (7.41 pp below full-group OOF). Those V0 effects still do not explain the full 22.2 pp gap to the 0.558 Kaggle score; hidden-test distribution shift remains the leading unverified cause.

The submitted E20-D kernel version 11 is recorded by Kaggle as submission **58710430**, public score **0.558**. E13-B submission 58609286 scored 0.606. The packaged local output contains only the three-row smoke-test input; the scored hidden evaluation predictions are not present locally. Kaggle describes the competition as approximately 4,000 unlabeled passages ([competition description](https://www.kaggle.com/competitions/name-that-llm/overview/description)), so the three-row local output cannot explain the leaderboard result.

## 1. Deployment parity

| Stage | Result |
|---|---:|
| Checkpoint class order | Exact match to canonical 18-label order |
| Packaged V0 probabilities vs local cache | max absolute difference `0` |
| Packaged router reliability vs refit path | max absolute difference `0` |
| E13 candidate-label scores | max absolute difference `2.48e-08` |
| CatBoost input feature values (same OOF rows/groups) | max absolute difference `3.35e-08` |
| CatBoost candidate scores vs saved E20-D OOF | max absolute difference `2.98e-08` |
| Hungarian predictions vs saved OOF | `0` changed rows across five folds |
| CatBoost feature names/order | Exact schema; package selects the saved model's feature order |

First divergence is group construction when moving from OOF held-fold batches to deployment's single full batch: saved all-V0 package grouping has **258 groups**, whereas OOF predictions cluster each held fold independently. The package-style group assignment exactly matches the saved all-V0 diagnostic grouping (ARI 1.0). This changes group-derived features, but the row-held-out model replay remains well above the Kaggle score.

Per-fold parity details are in `artifacts/analysis/e20d_kaggle_failure_analysis/deployment_parity.csv`.

## 2. Five-fold aggregation

| Evaluation | Accuracy | Balanced accuracy | Interpretation |
|---|---:|---:|---|
| Saved E20-D OOF | 78.2142% | 78.2058% | Reference; one held-out fold model and OOF groups |
| Per-row fold model, package-style full-batch groups | 76.2505% | 76.2419% | Leakage-controlled model choice; measures group-feature/decoder shift |
| Per-row fold model, row-normalized scores + Hungarian | 76.1023% | 76.0932% | Leakage-controlled; per-row scores normalized before assignment |
| All five models averaged; independent labels | 84.5869% | 84.5819% | Optimistic/in-sample on V0; each row is in four models' training folds |
| All five raw scores averaged + Hungarian | 86.1801% | 86.1747% | Optimistic/in-sample; not a valid Kaggle generalization estimate |
| All five per-model normalized scores + Hungarian | 87.1804% | 87.1752% | Optimistic/in-sample; normalization changes 80 V0 predictions |

All-five-on-V0 is not an honest analogue of Kaggle: the five CatBoost fold models were each trained on 80% of V0, so every V0 row is in four models' fitting data. Its high score must not be used to claim the deployment aggregation generalizes. The valid fold-held-out replay with deployment-style grouping is the second row; it falls from 78.21% to 76.24% BA, a 1.96 pp drop, still far short of 55.8%.

Moving from saved OOF to all-five raw/Hungarian changes 420 rows: 280 corrected and 65 regressed. This is contaminated by fold-training overlap.

Per-model row normalization changes 80 rows relative to raw-score averaging (47 corrected, 20 regressed; `+1.000` pp BA). This is also in-sample on V0.

On the leakage-controlled per-row held-fold scores, row normalization changes 113 assignments (36 corrected, 40 regressed) and changes BA from 76.2419% to 76.0932%. It does not improve this held-out replay.

### Per-label precision and recall

| Label | E13-B P/R | E20-D OOF P/R | Fold-held package groups P/R | Checkpoint-only OOF P/R |
|---|---:|---:|---:|---:|
| claude-fable-5 | 0.384/0.456 | 0.557/0.557 | 0.552/0.530 | 0.584/0.584 |
| claude-opus-4-6 | 0.763/0.880 | 0.933/0.927 | 0.880/0.927 | 0.898/0.940 |
| claude-opus-4-8 | 0.719/0.733 | 0.853/0.853 | 0.855/0.827 | 0.903/0.873 |
| claude-opus-5 | 0.855/0.787 | 0.912/0.893 | 0.879/0.873 | 0.864/0.847 |
| claude-sonnet-5 | 0.689/0.680 | 0.853/0.853 | 0.797/0.840 | 0.853/0.853 |
| gemini-3.1-pro-preview | 0.889/0.853 | 0.940/0.940 | 0.927/0.927 | 0.945/0.920 |
| gemini-3.6-flash | 0.694/0.833 | 0.876/0.893 | 0.860/0.900 | 0.858/0.887 |
| gemini-3.7-flash | 0.626/0.447 | 0.757/0.727 | 0.737/0.673 | 0.694/0.667 |
| gemini-3.8-flash | 0.586/0.680 | 0.758/0.773 | 0.713/0.760 | 0.737/0.747 |
| glm-5.3 | 0.376/0.273 | 0.530/0.527 | 0.548/0.493 | 0.556/0.567 |
| glm-5.3-flash | 0.426/0.460 | 0.588/0.580 | 0.546/0.513 | 0.623/0.607 |
| gpt-5.5 | 0.614/0.627 | 0.786/0.807 | 0.768/0.773 | 0.781/0.787 |
| gpt-5.6-luna | 0.620/0.653 | 0.800/0.800 | 0.800/0.800 | 0.823/0.807 |
| gpt-5.6-sol | 0.566/0.540 | 0.753/0.733 | 0.752/0.727 | 0.758/0.753 |
| gpt-5.6-terra | 0.606/0.647 | 0.748/0.753 | 0.717/0.727 | 0.758/0.773 |
| gpt-6-astra | 0.834/0.840 | 0.946/0.933 | 0.932/0.920 | 0.906/0.900 |
| kimi-k3 | 0.615/0.500 | 0.654/0.667 | 0.588/0.647 | 0.637/0.667 |
| qwen3.8-max | 0.762/0.747 | 0.832/0.860 | 0.839/0.867 | 0.886/0.880 |

### Per-label change in precision / recall vs E13-B

| Label | E20-D saved OOF ΔP/ΔR | Package groups, held-fold ΔP/ΔR | Checkpoint-only OOF ΔP/ΔR |
|---|---:|---:|---:|
| claude-fable-5 | +0.173/+0.101 | +0.168/+0.074 | +0.200/+0.128 |
| claude-opus-4-6 | +0.170/+0.047 | +0.117/+0.047 | +0.135/+0.060 |
| claude-opus-4-8 | +0.134/+0.120 | +0.136/+0.093 | +0.184/+0.140 |
| claude-opus-5 | +0.056/+0.107 | +0.024/+0.087 | +0.009/+0.060 |
| claude-sonnet-5 | +0.164/+0.173 | +0.108/+0.160 | +0.164/+0.173 |
| gemini-3.1-pro-preview | +0.051/+0.087 | +0.038/+0.073 | +0.056/+0.067 |
| gemini-3.6-flash | +0.181/+0.060 | +0.165/+0.067 | +0.164/+0.053 |
| gemini-3.7-flash | +0.131/+0.280 | +0.111/+0.227 | +0.068/+0.220 |
| gemini-3.8-flash | +0.172/+0.093 | +0.126/+0.080 | +0.151/+0.067 |
| glm-5.3 | +0.154/+0.253 | +0.172/+0.220 | +0.179/+0.293 |
| glm-5.3-flash | +0.162/+0.120 | +0.120/+0.053 | +0.197/+0.147 |
| gpt-5.5 | +0.171/+0.180 | +0.154/+0.147 | +0.167/+0.160 |
| gpt-5.6-luna | +0.180/+0.147 | +0.180/+0.147 | +0.203/+0.153 |
| gpt-5.6-sol | +0.187/+0.193 | +0.185/+0.187 | +0.192/+0.213 |
| gpt-5.6-terra | +0.142/+0.107 | +0.111/+0.080 | +0.152/+0.127 |
| gpt-6-astra | +0.112/+0.093 | +0.098/+0.080 | +0.072/+0.060 |
| kimi-k3 | +0.039/+0.167 | -0.027/+0.147 | +0.022/+0.167 |
| qwen3.8-max | +0.070/+0.113 | +0.077/+0.120 | +0.124/+0.133 |

The full per-label precision/recall/F1 and confusion matrices for every aggregation variant are retained in `metrics.json`; row-aligned labels are in `v0_predictions.parquet`.

## 3. Group-feature distribution shift

The same fold classifier scores were recomputed with true prompt groups and reconstructed groups. Feature shifts are ranked by standardized mean difference; CatBoost importance shows which shifts are used by the scorer.

| Feature | Mean absolute SMD | Mean absolute row delta | Mean CatBoost importance |
|---|---:|---:|---:|
| `group::group_edge_max` | 18.959 | 0.9933 | 0.000 |
| `group::group_edge_mean` | 17.512 | 0.9811 | 0.000 |
| `group::group_edge_min` | 9.113 | 0.9250 | 0.000 |
| `group::group_size` | 1.021 | 3.6169 | 0.000 |
| `group::group_other_supported_label_count` | 0.896 | 2.5595 | 0.729 |
| `group::group_confidence` | 0.545 | 0.0137 | 0.000 |
| `group::group_text_sim_min` | 0.521 | 0.0226 | 0.495 |
| `group::group_candidate_second_other` | 0.440 | 0.1223 | 3.697 |
| `group::group_candidate_best_other` | 0.398 | 0.1353 | 15.141 |
| `group::group_candidate_rank` | 0.353 | 1.8108 | 1.159 |
| `group::group_candidate_margin_other` | 0.271 | 0.1353 | 1.955 |
| `group::assignment_advantage` | 0.271 | 0.1353 | 1.463 |

With held-out fold models and the same reconstructed-group Hungarian decoder, replacing reconstructed group features with true-group features changes 153 rows (83 corrected, 20 regressed) and moves BA from 76.2419% to 78.5759% (+2.334 pp). This is measurable, but not enough to account for the Kaggle gap.

The largest distribution shifts are group-edge confidence and group size. Some highly shifted edge features have zero or near-zero CatBoost importance; the more influential group candidate competition features have moderate shifts. Detailed per-fold values are in `group_feature_shift_by_fold.csv`.

## 4. Rebuilt incomplete-group evaluation

Unlike the previous E20-D incomplete-group result, this evaluation reconstructs the pair graph on each exact sampled subset, recomputes group-dependent features, scores all five saved classifiers, and then runs independent or Hungarian decoding. It also reports a leakage-controlled variant that uses only the fold model that held out each row.

| Method | Mean BA | SD | Mean accuracy |
|---|---:|---:|---:|
| E20-D-fold-heldout-Hungarian | 70.7928% | 1.6402% | 70.8757% |
| E20-D-fold-heldout-independent | 70.6118% | 1.6020% | 70.6928% |
| E20-D-rescored-Hungarian | 79.8491% | 1.7379% | 79.8945% |
| E20-D-rescored-independent | 79.3911% | 1.3340% | 79.4501% |
| E20-checkpoint-only-Hungarian | 85.2195% | 1.9470% | 85.2431% |
| E20-checkpoint-only-fold-heldout-Hungarian | 72.9157% | 1.8790% | 72.9427% |
| E20-checkpoint-only-fold-heldout-independent | 67.8793% | 1.2371% | 67.9708% |
| E20-checkpoint-only-independent | 79.9679% | 1.1399% | 80.0060% |
| legacy-full-group-score-Hungarian | 77.6010% | 1.1227% | 77.6703% |

### Leakage-controlled results by reconstructed group size

These are the fold-held-out models, averaged over the existing five seeds per scenario. Sizes not represented in a sampled scenario are shown as —.

| Size | B rows | B independent BA | B Hungarian BA | C rows | C independent BA | C Hungarian BA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 66.6% | 66.6% | 56 | 50.8% | 50.8% |
| 2 | 308 | 62.7% | 62.9% | 242 | 65.4% | 65.7% |
| 3 | 381 | 62.9% | 62.4% | 345 | 66.7% | 67.4% |
| 4 | 352 | 69.0% | 68.6% | 440 | 65.9% | 65.7% |
| 5 | 450 | 66.7% | 66.3% | 395 | 71.6% | 70.5% |
| 6 | 576 | 69.7% | 70.5% | 408 | 64.4% | 66.1% |
| 7 | 539 | 70.2% | 69.8% | 364 | 66.8% | 66.3% |
| 8 | 640 | 71.8% | 72.2% | 392 | 71.4% | 70.8% |
| 9 | 666 | 73.6% | 74.7% | 459 | 69.1% | 69.3% |
| 10 | 550 | 74.0% | 73.4% | 530 | 72.7% | 72.2% |
| 11 | 0 | — | — | 396 | 73.5% | 73.9% |
| 12 | 0 | — | — | 720 | 75.7% | 74.7% |
| 13 | 0 | — | — | 481 | 73.4% | 72.7% |
| 14 | 0 | — | — | 504 | 70.2% | 70.8% |
| 15 | 0 | — | — | 435 | 75.8% | 78.2% |
| 16 | 0 | — | — | 448 | 78.3% | 80.0% |
| 17 | 0 | — | — | 459 | 75.7% | 76.1% |
| 18 | 0 | — | — | 414 | 81.2% | 81.6% |

The all-five methods are optimistic because sampled V0 rows were seen by four fold models. Use the `*-fold-heldout-*` rows for validation interpretation. Metrics by reconstructed group size (including singletons through size 18 where present) are saved in `incomplete_group_by_size.csv`; row-level sample predictions are in `incomplete_group_predictions_rescored.parquet`.

## 5. Checkpoint-only ablation

| Checkpoint-only scoring | Accuracy | Balanced accuracy |
|---|---:|---:|
| OOF independent | 68.2104% | 68.2028% |
| OOF + reconstructed-group Hungarian | 78.1030% | 78.0957% |
| All-five independent (in-sample) | 80.1037% | 80.0994% |
| All-five + Hungarian (in-sample) | 91.8859% | 91.8819% |

On the in-sample all-five diagnostic, checkpoint-only is +5.707 pp over E20-D and changes 235 labels (178 corrected, 24 regressed). This dramatic gain is a leakage artifact, not evidence that the checkpoint-only model is deployably stronger. OOF checkpoint-only scores are the valid comparator.

## 6. Root-cause assessment

### Confirmed

- No packaged inference parity defect was found on identical V0 rows and OOF groups: all five folds reproduce saved E20-D scores to <3.0e-8 maximum absolute error and produce zero changed Hungarian predictions.
- Deployment grouping is a different batch regime from OOF: 258 package-style all-V0 clusters versus fold-local OOF partitions. Under held-out fold models this costs about 1.96 pp BA, not the 22.2 pp Kaggle gap.
- The earlier incomplete-group evaluation was invalid for the requested question because it reused full-group CatBoost scores. The rebuilt sample evaluation shows the proper group-reconstruction path and is saved separately.
- All-five fold-model averaging on V0 and checkpoint-only all-five scores are contaminated: each V0 example appears in four models' training folds. They cannot explain the hidden-test result.

### Suspected, not proven

- Hidden-test domain/prompt/style shift is the leading explanation left by the parity checks. Kaggle scoring uses a hidden evaluation set; the scored-run predictions/labels are not available in the local v11 output.
- E20-C trains its CatBoost scorer with oracle prompt-group features, while E20-D serves reconstructed groups. This is a real train/serve feature mismatch, but leakage-controlled V0 impact is modest in the current replay; it may be larger on hidden groups.
- Test-time group completeness and reconstruction quality may differ from V0. V0 subset simulations do not establish the hidden test's group-size distribution.
- A checkpoint-specific inference or class-order defect on hidden test rows is not ruled out by V0 parity alone; the archived v11 output has only three smoke-test rows and does not include the hidden scored-run probability cache.

## Recommended deployment corrections (ranked)

1. **Obtain scored-run outputs before changing the model:** preserve per-checkpoint probability matrices, reconstructed groups/pair scores, per-fold 18-label CatBoost scores, and final assignments from the actual hidden evaluation run; verify canonical class order and finite normalized checkpoint probabilities there.
2. **Validate the exact serving aggregation with an untouched proxy set:** evaluate all five fold models on rows unseen by every fold model (or train a single final scorer on all V0 and evaluate on a truly external labeled set). Do not use the contaminated all-five-on-V0 score.
3. **Reduce train/serve group-feature mismatch:** compare E20-B (reconstructed-group training and inference) with E20-D on a genuinely unseen labeled set; if mismatch persists, deploy E20-B or remove the most shifted group features based on held-out evidence.
4. **Measure hidden-test grouping diagnostics:** expose group-size histogram, singleton share, pair confidence, and cluster purity proxy in the run manifest. Do not force Hungarian uniqueness on weak-confidence/singleton groups without validation.
5. **Check hidden inference class mapping end-to-end:** emit per-checkpoint class map, row-ID hash, probability row sums and score matrix checksum in the scored-run artifacts.

## Artifacts

Analysis code: `src/e20d_kaggle_failure_analysis.py`; regression tests: `tests/test_e20d_kaggle_failure_analysis.py`.
Machine-readable results: `artifacts/analysis/e20d_kaggle_failure_analysis/metrics.json`, `deployment_scores_v0.npz`, `deployment_parity.csv`, `group_feature_shift_summary.csv`, `per_label_changes.csv`, `incomplete_group_metrics.csv`, `incomplete_group_by_size.csv`, and `incomplete_group_predictions_rescored.parquet`.

No neural inference, retraining, queue changes, or Kaggle submission was performed.
