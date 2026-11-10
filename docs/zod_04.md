# ZOD_04 — UltraBERT Multi-Layer Readout

ZOD_04 extracts masked-mean representations from each of the 24 transformer layers of the existing best UltraBERT-380M V0 fine-tuned checkpoint. The backbone remains frozen. A `StandardScaler` plus fixed scikit-learn multinomial Logistic Regression classifier is trained for each layer and for the best 2, 4, and 8 CV-ranked layers, plus all 24 layers.

Layer ranking uses five-fold `StratifiedGroupKFold` on the V0 training partition, grouped by `prompt_id`. Final classifiers fit the complete V0 training partition and are evaluated on the frozen V0 validation partition. The existing UltraBERT classification head is scored on those same validation IDs as the baseline.

## Execute

From the repository root, run:

```bash
python -m src.pipeline analysis zod04
```

This uses the recorded V1-all → V0 best checkpoint by default. To override the batch size, number of prompt-group folds, or checkpoint, use `--batch-size`, `--n-splits`, or `--experiment-id`. Use `--refresh-features` to rebuild the frozen representation cache; otherwise a cache is reused only when its checkpoint, V0 rows, max length, and pooling metadata match.

Generated files go under `artifacts/analysis/zod_04/`: the row-aligned frozen feature cache, feature manifest, grouped out-of-fold predictions, layer ranking, saved classifier bundles, validation metrics, and ID-aligned validation probabilities/predictions. The executed run writes the full results report here, including layer and multi-layer balanced accuracy, original-head comparison, and per-label recall.

**Status:** implementation prepared; feature extraction, classifier fitting, and evaluation have not been run.
