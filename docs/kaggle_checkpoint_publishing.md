# Publishing NLM checkpoints to Kaggle

## Kaggle version behavior

A canary against `zodrak/zod006-models` showed that `kaggle datasets version` replaces the latest version's file set with the files uploaded for that version; it does not inherit omitted files. Version 3 contained only the three canary files, so the previous checkpoints disappeared from the latest input even though version 2 remained in history. The local publisher now refuses to publish a delta as a new version of the shared dataset.

Therefore, a single growing Kaggle dataset cannot meet the requirement to add checkpoint files without uploading old files again. Use **immutable checkpoint shard datasets**, one per coherent release or batch (for example, the LC3/LC4 v035 bundle). Each shard contains only that release's new assets and its catalog/manifest. Keep using existing datasets such as `zodrak/zod006-models` as earlier shards; attach the needed shards to the notebook. Batch new checkpoints into meaningful releases rather than making one dataset per job. Kaggle's `datasets version` command creates a version from updated files, while `datasets create` can publish a separate immutable shard. [Dataset version docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md#kaggle-datasets-version)

## Asset and catalog layout

Store each unique file once under a SHA-256-derived name (`asset__<sha256>`). A shard-local `checkpoint_catalog.json` maps experiment IDs to model metadata, training data, class order, base-model revision and each checkpoint-relative path. If a checkpoint's exact file hash already exists in an older mounted shard, reference its original dataset and filename instead of copying it. The LC3/LC4 packager writes these references into the run manifest so the notebook resolves files across its mounted inputs.

Include only inference assets: selected checkpoint weights, adapter/classifier files, tokenizers/configuration, necessary custom code and pinned base assets not otherwise available as compatible Kaggle model sources. Exclude optimizer state, epoch snapshots, logs, prediction caches and training datasets. Keep stable notebooks per inference workflow; update kernel metadata only to attach a new shard or when inference code changes. Record kernel version, every mounted shard, checkpoint ID and hashes in the run output.

## Existing inventory and LC3/LC4 preparation

`zodrak/zod006-models` was the closest existing shared bundle: LC3/LC4 already used it. Before the canary it was version 2 with 37 files (1,695,910,896 bytes); the local package had 390 files (53,762,989,129 bytes). The LC3/LC4 manifest referenced 69 checkpoints and 320 unique files (36,488,066,411 bytes). A whole-folder upload would have republished unrelated assets and checkpoint copies. The canary-only version 3 was recovered as version 4 with the complete prior version 2 files and refreshed catalog.

Prepare LC3/LC4 as the immutable shard `zodrak/lc3-lc4-checkpoints-v035`. The shard builder reuses files from ZOD006 and existing checkpoint datasets only after matching the experiment ID, relative filename, SHA-256 and Kaggle's remote size. The v035 audit reused 91 assets (1.63 GB) from ZOD006, 27 assets (4.69 GB) from `zodrak/three-strategy-checkpoints`, and 18 assets (2.53 GB) from `zodrak/ensemble-hypotheses-nov15-extra-checkpoints`. The `trust2-v013-top5-models` candidate files were already represented in ZOD006, so that dataset does not need to be mounted. All 320 candidate-file references were checked against their local checkpoint sources. The remaining 36.91 GiB is staged in eight directory archives, each below 5 GiB, along with the pinned MiniCPM5 and Granite bases, runtime modules and manifests. The LC3/LC4 notebook resolves assets across its mounted inputs. Upload the notebook with `kaggle kernels push --no-run` to save a version without executing inference.

The v035 shard was published privately as dataset version 1 with 169 files totaling 39,627,076,411 bytes after Kaggle expanded the eight directory archives. Its inventory contains all 163 shard-local assets and six direct metadata/manifest files. The kernel `zodrak/lc3-and-lc4-trust-list-2-v035` is at version 2, saved with `--no-run`; Kaggle's worker logs contain notebook conversion only and no cell execution.

The generic `scripts/publish_kaggle_checkpoint_store.py` remains a dry-run delta planner and rejects `--publish`; it must not create a replacement version of `zod006-models`. Use `scripts/prepare_lc3_lc4_checkpoint_shard.py` for the LC3/LC4 release. Its `--publish` option creates a new dataset only and refuses to reuse an existing shard slug.

## Validation and rollout

The publisher tests cover content hashing, reuse, duplicate files, catalog lookup, checkpoint-ID conflicts and remote size conflicts. For a new shard, review the dry-run byte count and staged file list, create the dataset once, verify its file inventory and status, then push the LC3/LC4 notebook metadata referencing that shard. Keep prior shards and dataset versions intact; do not rely on dataset versions to append files.
