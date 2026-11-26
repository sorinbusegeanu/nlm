# UltraBERT-380M training enhancements

The existing `ultrabert380` model now has two independent, opt-in training
features. With both `enabled: false`, training and inference continue to use
the existing `AutoModelForSequenceClassification` path and existing checkpoint
format.

## Variants

| Variant | Configuration | Supervised contrastive | Multi-layer aggregation |
|---|---|---:|---:|
| A — baseline | `configs/models/ultrabert380.yaml` | Off | Off |
| B — contrastive | `configs/models/ultrabert380_contrastive.yaml` | On | Off |
| C — multi-layer | `configs/models/ultrabert380_multilayer.yaml` | Off | On |
| D — combined | `configs/models/ultrabert380_contrastive_multilayer.yaml` | On | On |

The new configurations preserve the baseline model, dataset, optimizer,
scheduler, batch sizing and early-stopping settings. They differ only in the
enhancement toggles and experiment name. They are ready for future runs; no
training queue entries were changed as part of this implementation.

## Supervised contrastive learning

When enabled, the model uses its pooled response representation to produce a
128-dimensional projection, then L2-normalizes it. The training objective is
`cross_entropy + weight * supervised_contrastive_loss`. The loss uses
temperature-scaled cosine similarity in float32, removes self-comparisons and
skips anchors without another response of the same label in their batch.
Validation remains ordinary, unweighted cross-entropy and the inference output
remains the existing 18 classification logits.

The label-aware sampler presents every training response once per epoch. It
tries to place same-label responses from different prompts together and favors
same-prompt responses of different labels as hard negatives. Prompt IDs are
used only for arranging training batches; they are not input features. Without
prompt IDs, distinct row IDs are used, so positive pairs still use label only.
Epoch-based deterministic seeds make the batches reproducible and vary pairings
between epochs. A batch with no positive pairs contributes zero contrastive
loss while still training the classification head.

Configuration defaults:

```yaml
supervised_contrastive:
  enabled: false
  weight: 0.1
  temperature: 0.07
  projection_dim: 128
```

## Multi-layer feature aggregation

When enabled, the model dynamically finds the transformer's block stack and
selects evenly spaced block indices, including the final block. The default is
four layers. Forward hooks capture only those selected block outputs. Each is
mean-pooled using the attention mask, then the pooled vectors are combined by
a trainable softmax-weighted sum initialized to equal weights. The result feeds
the same 18-class linear classification head; variant D also feeds the
contrastive projection head.

Selected outputs remain connected to autograd so all selected layers and the
aggregation logits train jointly. Activation/gradient checkpointing is disabled
for this variant because reentrant checkpointing runs hooked blocks without
gradients during the capture pass. Automatic batch-size probing and OOM
recovery remain enabled; the effective checkpointing choice is recorded in
`batch_size_probe.json`. This can use more activation memory than the baseline.

```yaml
multi_layer_aggregation:
  enabled: false
  num_layers: 4
  pooling: mean
  aggregation: learned_weighted_sum
```

The implementation rejects unsupported pooling or aggregation modes and a
layer count larger than the backbone stack. The selected block indices,
backbone layer count and module path are written to manifests.

## Checkpoints and continuation

Enhanced checkpoints store the backbone in the normal Hugging Face format and
the classifier head, optional contrastive projection head and optional
aggregation logits in sidecar files. `checkpoint_manifest.json` records the
effective enhancement configuration, canonical class ordering and selected
layer indices. The inference factory detects this manifest and reconstructs
the enhanced readout; the contrastive projection is not needed for inference.

Continuation loads those same sidecars and requires the enhancement settings
and selected layer indices to match the source manifest. A baseline checkpoint
can initialize a new enhanced run: its backbone and compatible classification
head are loaded, while the new projection head and aggregation weights start
fresh. An enhanced checkpoint cannot silently continue through the baseline
path. Existing class-weighted CE support remains available; its weights are
applied in canonical class order on the active device. Validation CE remains
unweighted.

The current sampler deliberately cannot be combined with weighted sampling or
the E21 group-ranking sampler because all three define batch composition. Run
the enhancements independently from those sampling modes. Classification-only
training is unchanged when both enhancement flags are disabled.

## Tests

Use the CUDA-enabled project Python environment for the CPU unit suite:

```bash
/home/zodrak/cuda13.x/bin/python -m pytest -q \
  tests/test_ultrabert_training_enhancements.py \
  tests/test_group_aware_training.py
```

GPU tests are prepared separately and were not run while the GPU was occupied:

```bash
/home/zodrak/cuda13.x/bin/python -m pytest -q \
  tests/test_ultrabert_training_enhancements_gpu.py
```

## Future training commands

These commands are examples for a later, explicitly scheduled run. They were
not executed during implementation.

```bash
python -m src.pipeline models train --model ultrabert380 --dataset v2.14 \
  --config configs/models/ultrabert380_contrastive.yaml --accept-config-defaults
python -m src.pipeline models train --model ultrabert380 --dataset v2.14 \
  --config configs/models/ultrabert380_multilayer.yaml --accept-config-defaults
python -m src.pipeline models train --model ultrabert380 --dataset v2.14 \
  --config configs/models/ultrabert380_contrastive_multilayer.yaml --accept-config-defaults
```

Replace `v2.14` with the intended dataset and use the pipeline's normal
validation-dataset selection. Variant A uses the existing baseline command and
configuration.
