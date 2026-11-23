# E31 — E13-B Expansion from 17 to 40 Checkpoints

## Scope and baseline reconciliation

E31 uses the current corrected E30-B manifest as the fixed baseline. The manifest's actual roster contains **17 checkpoints**, despite the earlier description of E30 as a 20-checkpoint deployment. E31 preserves those 17 and selects 23 additional cached v037 checkpoints to create the requested fixed 40-member deployment. E30 files and artifacts are not changed by E31.

The candidate pool is the 70 available Trust List 2 v037 checkpoints outside the E30 roster. The 17 shared baseline probability-cache hashes are checked against the E30 manifest before selection. All V0 rows, canonical labels and response IDs follow the E13-B conventions.

## Selection and evaluation protocol

For each of the five saved E13-B prompt-group outer folds, greedy selection begins with the 17-member baseline. Each candidate addition is evaluated by refitting the E13-B CatBoost router through four prompt-group-disjoint inner folds and comparing inner OOF predictions. Balanced accuracy is primary; paired net correct predictions, introduced errors and checkpoint ID resolve ties. The process evaluates all 23 additions, even when an addition has a negative marginal score.

The fixed 40-member roster is formed by fold consensus: selection frequency, mean selected rank, marginal balanced-accuracy gain, and checkpoint ID. Every candidate evaluation records BA, corrections, introduced errors and net correct predictions. The outer held-out predictions are generated separately for each fold-local roster size from 17 through 40; these cross-fitted results determine the best roster size from 20 through 40.

The final fixed-roster replay compares E30-B's saved 17-member deployment against E31's 40-member router on identical V0 rows and 50 `v0_group` scenarios. It also applies the same frozen E19-F-A and E19-F-C groupers followed by Hungarian assignment. Grouping quality and group-size distributions are measured against prompt IDs for diagnosis only; prompt IDs and labels do not enter grouping inference.

V0 has historical checkpoint-selection exposure. Outer-fold scores are prompt-group held-out estimates within that development set. The fixed-roster router and its reliability statistics are fit using V0 training examples, so its full-V0 replay is an in-sample deployment diagnostic, not an independent test result.

## Results

**Experiment is in progress.** Results, per-label changes, fold stability, scenario performance, runtime and recommendation will be added after the cross-fitted selection and deployment benchmarks finish.

## Reproducibility artifacts

- Selection, candidate ranking, outer-fold predictions and input cache hashes: `artifacts/analysis/e31/selection_results.json`
- Restartable nested candidate outputs: `artifacts/analysis/e31/selection/candidate_oof_cache/`
- Fixed-roster router and manifest: `artifacts/analysis/e31/e31_router.cbm`, `artifacts/analysis/e31/e31_manifest.json`
- Fixed-roster and 50-scenario predictions/metrics: `artifacts/analysis/e31/`
- Local Kaggle packaging script (no submission): `scripts/prepare_e31_kaggle.py`
