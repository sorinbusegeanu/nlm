# Checkpoint label catalog and trust list

[`checkpoint_label_catalog.csv`](checkpoint_label_catalog.csv) is the durable checkpoint metrics catalog. It contains one row per checkpoint and target label, with frozen V0 validation precision, recall and prediction support.

[`checkpoint_label_trust_list.csv`](checkpoint_label_trust_list.csv) is the dashboard's and ZOD03's shared ranked reference list. Every checkpoint-label catalog row appears once and receives a numeric `SeqID`. The adjacent [`checkpoint_label_trust_list.json`](checkpoint_label_trust_list.json) records its version, UTC generation time, source-catalog hash, support threshold, and ranking rule. The dashboard reads this generated list and metadata; it does not compute rankings.

## Retained-checkpoint scope

The catalog keeps one selected checkpoint per experiment, not every epoch checkpoint. On refresh it resolves an explicit `best_checkpoint_path`/`selected_checkpoint_path` from the experiment prediction manifest, and sets `checkpoint_available` from the resulting local path. Missing weight files remain as historical metric rows marked unavailable; they are not runnable checkpoint candidates. Catalog experiments are retained only when at least one label rule appears in the first 500 rows of the current ranked catalog, Trust List 2, or Trust List 2.2. Membership is matched by `experiment_id`.

The latest Trust List 2 and 2.2 snapshots now use the same stable `sequence_id` as this catalog for shared experiments. Trust List 2 v027 repaired collisions that had assigned the same N-number to different checkpoints; Trust List 2.2 v0007 inherits those IDs. Older immutable snapshots retain their historical IDs, so use `experiment_id` when referring to a checkpoint from an older list version.

## IDs and rows

- `N####` is a stable checkpoint sequence ID, such as `N0072`. All 18 target-label rows for one checkpoint share that ID.
- Retained specialist-discovery IDs keep their original sequence numbers; removed IDs are not reused. The four older UltraBERT aliases A–D map to `N0086`–`N0089`; their old alias remains recorded in the catalog. New checkpoints receive the next unused ID and IDs are never reassigned.
- `Model label` in the dashboard is the individual predicted NLM label. `Checkpoint model` identifies the architecture or classifier that produced the predictions.
- Label accuracy is precision: true positives divided by predictions for that label. Recall is true positives divided by actual validation examples for that label. `Predicted rows` gives the precision denominator.

## Sources and updates

- The 86 retained specialist-discovery checkpoints and their per-label values are copied from the existing ZOD01 specialist-discovery inventory and ranked-rules snapshot.
- New full-V0 checkpoints are added from their saved probability predictions after alignment with all 540 frozen V0 catalog IDs, prompt IDs and labels. Runs that saved all 2,699 V0 predictions are reduced to those exact historical 540 IDs for this catalog; their rows note that V0 was used for checkpoint selection. No model is loaded for catalog updates.
- The training pipeline refreshes this catalog after prediction-factory evaluation. Any registered model with saved V0 predictions (540 or 2,699 rows) can be appended; partial family-specific validations and V0-trained lineages are excluded. To refresh manually, run `python -m src.checkpoint_catalog`.
- Catalog refresh regenerates the trust list when the catalog changes. Versions increase only when the source catalog changes; rerunning without catalog changes preserves the current version and generation time.
- `SeqID` is the deterministic position in the complete ranked checkpoint-label list. The first rows satisfying zero validation-overlap and minimum support 10 are `rule_eligible=true`; lower-support or otherwise ineligible rows remain in the reference list after eligible rows with their own stable `SeqID`.
- Ranking uses raw label accuracy (precision) descending, then recall descending, checkpoint sequence ID ascending, and target label ascending. Smoothed precision remains available for analysis but does not determine SeqID. ZOD03 applies the complete ordered list while skipping ineligible rows, lower-support repeated-label rules, and candidates after two applied rules for a label.
- `checkpoint_date` is rendered as `MM.DD` in Europe/Zurich time. The full UTC timestamp and validation split hash remain in the CSV.
- Rows with V0 training overlap stay visible and carry a validation note. Their scores are development-set diagnostics and should not be interpreted as independent estimates.
- The dashboard reads the generated trust-list CSV and JSON metadata. It does not run inference, calculate checkpoint metrics, or evaluate ensembles while rendering the page.
