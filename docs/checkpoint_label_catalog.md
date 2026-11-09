# Checkpoint label catalog

[`checkpoint_label_catalog.csv`](checkpoint_label_catalog.csv) is the durable source for the dashboard's **Prediction factory** page. It contains one row per checkpoint and target label, with frozen V0 validation precision, recall and prediction support.

## IDs and rows

- `N####` is a stable checkpoint sequence ID, such as `N0072`. All 18 target-label rows for one checkpoint share that ID.
- Existing IDs `N0001`–`N0085` are kept from the specialist-discovery snapshot. The four older UltraBERT aliases A–D map to `N0086`–`N0089`; their old alias remains recorded in the catalog. New checkpoints receive the next unused ID and IDs are never reassigned.
- `Model label` in the dashboard is the individual predicted NLM label. `Checkpoint model` identifies the architecture or classifier that produced the predictions.
- Label accuracy is precision: true positives divided by predictions for that label. Recall is true positives divided by actual validation examples for that label. `Predicted rows` gives the precision denominator.

## Sources and updates

- The first 89 checkpoints and their per-label values are copied from the existing ZOD01 specialist-discovery inventory and ranked-rules snapshot.
- New V1-all checkpoints are added from their saved probability predictions after alignment with all 540 frozen V0 validation IDs, prompt IDs and labels. No model is loaded for catalog updates.
- The training pipeline refreshes this catalog after prediction-factory evaluation. To refresh manually or import other completed V1-all runs, run `python -m src.checkpoint_catalog`.
- `checkpoint_date` is rendered as `MM.DD` in Europe/Zurich time. The full UTC timestamp and validation split hash remain in the CSV.
- Rows with V0 training overlap stay visible and carry a validation note. Their scores are development-set diagnostics and should not be interpreted as independent estimates.
- The dashboard reads this CSV only. It does not run inference, calculate checkpoint metrics, or evaluate ensembles while rendering the page.
