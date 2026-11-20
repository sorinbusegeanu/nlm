# ORACLE04 Kaggle submissions

## Failure diagnosis and repair (2026-11-17)

The first ORACLE04 notebook versions were built from the three rows in the local
public `test.csv` and exported those cached predictions. Kaggle reruns notebook
submissions against a separate hidden test input, whose IDs and row count differ;
the fixed-ID equality check therefore stopped those reruns before producing a
valid hidden-test submission. The kernels themselves completed on the local
three-row input, which had obscured the problem.

The replacement package runs all 78 Trust List 2 v025 checkpoints against the
runtime `test.csv`, aligns probabilities to the canonical 18 labels, applies
the selected ORACLE04 strategy, and writes `id,model` in runtime input order.
The asset bundle is one tar archive so it stays under Kaggle's 50-root-file
limit; the runner extracts one checkpoint at a time. QLoRA inference now uses
BF16 when supported and FP16 on T4 hardware, and the runner applies the
existing SmolLM3 and TorchAO compatibility fixes used by the NLM Kaggle
notebooks.

The new code and archive passed local syntax and archive-coverage checks. The
first upload attempt did not create a Kaggle dataset. The waiting service then
failed because its systemd environment could not resolve the `kaggle` command.
The archive also needed a nested member directory: Kaggle expands dataset
archives, so 470 flat archive members would exceed its 50 top-level file limit.

Both issues are fixed. The archive now expands under `oracle04_assets/`, the
deployment runner calls the Kaggle CLI by absolute path, and the runner itself
performs dataset creation before pushing kernels and submitting them. A new
41.8 GiB upload began on 2026-11-18 at 05:09 UTC. Kernel publication and all
three code submissions are pending upload and dataset verification.

## Result

ORACLE04 used the restored Trust List 2 v025 roster of 78 checkpoints. Test probabilities for all 78, including N0074, were already cached for the three competition test rows. The saved ORACLE04 deployment models were applied to those same aligned probabilities; no checkpoint was retrained or re-inferred for these submissions.

| Strategy | V0 prompt-group OOF balanced accuracy | Kaggle kernel | Version | Submission ID | Status | Public score |
|---|---:|---|---:|---:|---|---|
| `combined_logistic_core` | 71.169% | [kernel](https://www.kaggle.com/code/zodrak/oracle04-combined-logistic-core) | 1 | 58479914 | Hidden rerun failed | — |
| `global_logistic_core` | 71.168% | [kernel](https://www.kaggle.com/code/zodrak/oracle04-global-logistic-core) | 1 | — | Kernel complete; not submitted | — |
| `global_catboost_core` | 70.799% | [kernel](https://www.kaggle.com/code/zodrak/oracle04-global-catboost-core) | 1 | 58479937 | Hidden rerun failed | — |

Submissions **58479914** and **58479937** were recorded as `COMPLETE` but received no score: Kaggle's hidden rerun failed with an unhandled notebook error because the artifacts were tied to the three local public-test IDs. The global-logistic kernel completed on those same three rows but was not submitted because the daily submission quota was exhausted. It produced the same top-1 predictions as combined logistic on the visible rows.

## Test rows and validation

The local competition test input has three rows. All three kernels check that `submission.parquet` contains exactly `id, model`, one prediction per `test.csv` ID in input order, and only valid competition labels. Local package validation passed before push. Kaggle kernel output download was denied for the private kernels, but the competition submission service accepted and scored the two submitted kernel outputs.

| Test ID | Combined logistic | Global logistic | Global CatBoost |
|---|---|---|---|
| `00353fb1c541` | `claude-opus-4-8` | `claude-opus-4-8` | `glm-5.3` |
| `005f5a9a6139` | `gpt-5.6-sol` | `gpt-5.6-sol` | `gpt-5.5` |
| `0062a53d5487` | `gemini-3.1-pro-preview` | `gemini-3.1-pro-preview` | `gemini-3.1-pro-preview` |

## Caveat

The V0 values are cross-fitted development scores, not Kaggle score predictions. V0 influenced checkpoint and roster selection and the Oracle04 thresholds, so Kaggle generalization remains unknown until a public score appears. Exact local submissions, cached probabilities, kernel notebooks, and hashes are recorded in `artifacts/kaggle/oracle04-submissions-dataset/oracle04_submission_manifest.json`.
