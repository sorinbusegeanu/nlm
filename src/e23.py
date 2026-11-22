"""E23: 20-checkpoint caps for E22 learned candidate-label ensembles.

Checkpoint rosters are selected inside each training fold using greedy
complementary top-1 coverage. All model fitting and inference use CPU and
complete cached V0 probabilities; no base neural inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import e20, e22
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e23"
MAX_CHECKPOINTS = 20
FEATURE_SCHEMA_VERSION = 1


def select_roster(ctx, train_rows, candidates, *, mandatory=(), max_checkpoints=MAX_CHECKPOINTS):
    """Greedily maximize training-only balanced correct-label coverage.

    Mandatory members are reserved first (N0135/N0136 in B/C), then each
    addition maximizes new correctly-covered examples, with family novelty,
    standalone balanced accuracy, and checkpoint ID as deterministic tie-breaks.
    """
    train_rows = np.asarray(train_rows, dtype=np.int64)
    candidates = list(dict.fromkeys(map(str, candidates)))
    mandatory = list(dict.fromkeys(map(str, mandatory)))
    if max_checkpoints < len(mandatory):
        raise ValueError("Roster cap is smaller than its mandatory checkpoint count")
    unknown = set(candidates + mandatory) - set(ctx["checkpoint_ids"])
    if unknown or not set(mandatory).issubset(candidates):
        raise ValueError(f"Invalid E23 roster candidates: {sorted(unknown or set(mandatory)-set(candidates))}")
    y = ctx["y"][train_rows]
    cp_index = {sid: i for i, sid in enumerate(ctx["checkpoint_ids"])}
    predictions = {sid: ctx["matrices"][cp_index[sid], train_rows].argmax(axis=1)
                   for sid in candidates}
    supports = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    standalone = {sid: float(balanced_accuracy_score(y, predictions[sid])) for sid in candidates}
    selected = []
    covered = np.zeros(len(train_rows), dtype=bool)
    steps = []

    def add(sid, reason):
        nonlocal covered
        correct = predictions[sid] == y
        newly = correct & ~covered
        gain = float(np.sum(1.0 / supports[y[newly]]) / len(LABELS)) if newly.any() else 0.0
        selected.append(sid)
        covered |= correct
        steps.append({"checkpoint_id": sid, "reason": reason,
            "standalone_balanced_accuracy": standalone[sid],
            "new_correct_rows": int(newly.sum()), "weighted_coverage_gain": gain,
            "family": ctx["families"][cp_index[sid]]})

    if mandatory:
        for sid in mandatory:
            add(sid, "mandatory comparison member")
    else:
        first = min(candidates, key=lambda sid: (-standalone[sid], int(sid[1:])))
        add(first, "highest standalone balanced accuracy")
    while len(selected) < min(max_checkpoints, len(candidates)):
        remaining = [sid for sid in candidates if sid not in selected]
        selected_families = {ctx["families"][cp_index[sid]] for sid in selected}

        def rank(sid):
            correct = (predictions[sid] == y) & ~covered
            gain = float(np.sum(1.0 / supports[y[correct]]) / len(LABELS)) if correct.any() else 0.0
            family_new = int(ctx["families"][cp_index[sid]] not in selected_families)
            return (gain, family_new, standalone[sid], -int(sid[1:]))

        chosen = max(remaining, key=rank)
        add(chosen, "greedy complementary correct-label coverage")
    if len(selected) > max_checkpoints or len(selected) != len(set(selected)):
        raise RuntimeError("E23 roster violates the checkpoint cap or uniqueness")
    return selected, {"train_rows": int(len(train_rows)), "max_checkpoints": max_checkpoints,
        "mandatory": mandatory, "selection_steps": steps,
        "weighted_training_coverage": float(np.sum(covered / supports[y]) / len(LABELS))}


def _specialist_details(ctx, train_rows, pool):
    selected, details = e22._specialists(ctx, train_rows, pool)
    serial = {}
    for label, cps in selected.items():
        label_details = details[LABELS[label]]
        serial[LABELS[label]] = [{"checkpoint_id": ctx["checkpoint_ids"][cp],
            "precision": float(label_details[index]["precision"]),
            "recall": float(label_details[index]["recall"])}
            for index, cp in enumerate(cps)]
    return selected, serial


def _run_experiment(ctx, name, candidates, *, mandatory=(), specialist=False, output=OUT):
    output = Path(output)
    saved_metrics = output / f"metrics_{name}.json"
    saved_predictions = output / f"predictions_{name}.npz"
    if saved_metrics.is_file() and saved_predictions.is_file():
        saved = json.loads(saved_metrics.read_text())
        if saved.get("feature_schema_version") == FEATURE_SCHEMA_VERSION:
            return saved
    y = ctx["y"]
    scores = np.full((len(y), len(LABELS)), np.nan, dtype=np.float32)
    independent = np.full(len(y), -1, dtype=np.int64)
    pred_d = np.full(len(y), -1, dtype=np.int64)
    pred_f = np.full(len(y), -1, dtype=np.int64)
    pred_oracle = np.full(len(y), -1, dtype=np.int64)
    folds_result = []; importance = []; specialists_by_fold = {}
    outer_rosters = {}; started = time.monotonic()
    for outer in ctx["folds"]:
        fold, fit_rows, held = outer["fold"], outer["fit_rows"], outer["held_rows"]
        roster_ids, roster_info = select_roster(ctx, fit_rows, candidates, mandatory=mandatory)
        pool = [ctx["checkpoint_ids"].index(sid) for sid in roster_ids]
        outer_rosters[str(fold)] = {"checkpoint_ids": roster_ids, **roster_info}
        specialist_detail = None
        if specialist:
            selected_specialists, specialist_detail = _specialist_details(ctx, fit_rows, pool)
            specialists_by_fold[str(fold)] = specialist_detail

        def builder_for(inner_fit_rows):
            inner_ids, _ = select_roster(ctx, inner_fit_rows, candidates, mandatory=mandatory)
            inner_pool = [ctx["checkpoint_ids"].index(sid) for sid in inner_ids]
            inner_specialists = None
            if specialist:
                inner_specialists, _ = _specialist_details(ctx, inner_fit_rows, inner_pool)
            return lambda target_rows: e22.build_features(ctx, target_rows, inner_pool,
                                                            specialists=inner_specialists)

        outer_builder = lambda target_rows: e22.build_features(ctx, target_rows, pool,
            specialists=(selected_specialists if specialist else None))
        xfit = outer_builder(fit_rows)
        tuning, tuning_records = e22._tune(ctx, fit_rows, builder_for, ctx["group_d"], fold)
        model = e20._cat_train(xfit, e22._target(y[fit_rows]), tuning["depth"],
                               tuning["learning_rate"])
        xheld = outer_builder(held)
        mat = e20._predict_candidate(model, xheld)
        scores[held] = mat
        independent[held] = mat.argmax(axis=1)
        pred_d[held] = e20.hungarian(mat, ctx["group_d"][held], ctx["row_ids"][held])
        pred_f[held] = e20.hungarian(mat, ctx["group_f"][held], ctx["row_ids"][held])
        pred_oracle[held] = e20.hungarian(mat, ctx["groups"][held], ctx["row_ids"][held])
        model.save_model(str(output / f"{name}_fold{fold}.cbm"))
        fi = pd.DataFrame({"feature": model.feature_names_, "importance": model.get_feature_importance()})
        importance.extend({"fold": fold, "feature": row.feature,
                           "importance": float(row.importance)} for row in fi.itertuples())
        fold_result = {"fold": fold, "fit_rows": len(fit_rows), "held_rows": len(held),
            "selected_hyperparameters": {"depth": tuning["depth"], "learning_rate": tuning["learning_rate"]},
            "tuning": tuning_records, "checkpoint_roster": roster_ids,
            "roster_selection": roster_info,
            "independent": e20._metric(y[held], independent[held]),
            "E19-D_hungarian": e20._metric(y[held], pred_d[held]),
            "E19-F-A_hungarian": e20._metric(y[held], pred_f[held]),
            "oracle_hungarian": e20._metric(y[held], pred_oracle[held]),
            "specialists": specialist_detail}
        folds_result.append(fold_result)
        print(f"{name} fold {fold}: roster={len(roster_ids)} D={balanced_accuracy_score(y[held],pred_d[held]):.4%}, F={balanced_accuracy_score(y[held],pred_f[held]):.4%}, oracle={balanced_accuracy_score(y[held],pred_oracle[held]):.4%}", flush=True)
    if len(candidates) < MAX_CHECKPOINTS or any(len(v["checkpoint_ids"]) > MAX_CHECKPOINTS for v in outer_rosters.values()):
        raise RuntimeError(f"{name} violated the 20-checkpoint roster cap")
    if not np.isfinite(scores).all() or (independent < 0).any():
        raise RuntimeError(f"{name} failed to cover every V0 row")
    metrics = {"independent": e20._metric(y, independent), "E19-D": e20._metric(y, pred_d),
        "E19-F-A": e20._metric(y, pred_f), "oracle": e20._metric(y, pred_oracle)}
    comparison = {}
    base_path = output / "predictions_E23-A.npz"
    if name != "E23-A" and base_path.is_file():
        baseline = np.load(base_path)["pred_E19D"]
        comparison = e20._comparison(y, baseline, pred_d)
        comparison["per_label"] = {label: {
            "corrected": int(((baseline != y) & (pred_d == y) & (y == i)).sum()),
            "introduced": int(((baseline == y) & (pred_d != y) & (y == i)).sum())}
            for i, label in enumerate(LABELS)}
    result = {"experiment": name, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "checkpoint_count": MAX_CHECKPOINTS, "candidate_checkpoint_count": len(candidates),
        "mandatory_checkpoints": list(mandatory), "checkpoint_rosters_by_fold": outer_rosters,
        "metrics": metrics, "folds": folds_result, "specialist_selections": specialists_by_fold,
        "comparison_vs_E23-A_E19-D": comparison, "feature_importance": importance,
        "incomplete_group_simulations": e22._evaluate_incomplete(ctx, scores),
        "runtime_seconds": time.monotonic() - started}
    (output / f"metrics_{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(importance).to_csv(output / f"feature_importance_{name}.csv", index=False)
    np.savez_compressed(output / f"predictions_{name}.npz", row_ids=ctx["row_ids"], y=y,
        scores=scores, pred_independent=independent, pred_E19D=pred_d,
        pred_E19F_A=pred_f, pred_oracle=pred_oracle)
    return result


def _write_docs(root, config, results, result_doc):
    lines = ["# E23 — 20-checkpoint learned ensembles", "",
        "E23 reruns E22-A/B/C with a hard cap of 20 checkpoints per outer-fold roster. Rosters are selected using only each outer training partition; inner tuning reselects from its own fitting partition. Greedy selection starts with the highest standalone balanced accuracy (except mandatory N0135/N0136 in B/C), then adds checkpoints by new correctly-covered rows with class-balanced weighting, family diversity and standalone balanced accuracy as tie-breakers.", "",
        "All 2,699 V0 rows use identical E13-B prompt-group outer folds. E20's checkpoint-only candidate feature builder, CatBoost candidate scoring and Hungarian decoding are reused. E19-D and E19-F-A groups are OOF; true V0 groups are oracle-only. V0-selected checkpoint metrics make these development results, not an unbiased deployment estimate.", "",
        "| Experiment | Independent BA | E19-D BA | E19-F-A BA | Oracle BA | E19-D fold mean ± SD | Distinct checkpoint IDs | Roster per fold | Corrected vs A | Introduced vs A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        mets = result["metrics"]
        fold_ba = [f["E19-D_hungarian"]["balanced_accuracy"] for f in result["folds"]]
        comp = result.get("comparison_vs_E23-A_E19-D", {})
        roster_sizes = [len(x["checkpoint_ids"]) for x in result["checkpoint_rosters_by_fold"].values()]
        distinct_ids = {sid for fold in result["checkpoint_rosters_by_fold"].values() for sid in fold["checkpoint_ids"]}
        lines.append(f"| {result['experiment']} | {mets['independent']['balanced_accuracy']:.4%} | {mets['E19-D']['balanced_accuracy']:.4%} | {mets['E19-F-A']['balanced_accuracy']:.4%} | {mets['oracle']['balanced_accuracy']:.4%} | {np.mean(fold_ba):.4%} ± {np.std(fold_ba,ddof=1):.4%} | {len(distinct_ids)} | {min(roster_sizes)}–{max(roster_sizes)} | {comp.get('corrected','—')} | {comp.get('introduced','—')} |")
    lines += ["", "## Fold results and rosters", ""]
    for result in results:
        lines.append(f"### {result['experiment']}")
        lines.append("")
        for fold in result["folds"]:
            lines.append(f"- Fold {fold['fold']}: BA independent {fold['independent']['balanced_accuracy']:.4%}; E19-D {fold['E19-D_hungarian']['balanced_accuracy']:.4%}; E19-F-A {fold['E19-F-A_hungarian']['balanced_accuracy']:.4%}; oracle {fold['oracle_hungarian']['balanced_accuracy']:.4%}.")
            lines.append(f"  Roster: {', '.join(fold['checkpoint_roster'])}")
        lines.append("")
    lines += ["## E21 contribution", "",
        "E23-B and E23-C reserve N0135 and N0136 and use the remaining 18 slots for fold-local complementary coverage. Their corrections, introduced errors, per-label metrics, feature importance and specialist selections are in the corresponding metrics JSON files.", "",
        "## Reproducibility", "",
        "The configuration records Trust List and cache hashes, registry resolution, row IDs, fold definitions, roster-selection policy and source-code hash. Full OOF predictions and score matrices, fold models, metrics and feature importance are under `artifacts/analysis/e23/`.", ""]
    (root / "docs/e23.md").write_text("\n".join(lines))


def _register(root, results, *, reset=False):
    path = root / "configs/ensemble_catalog_v2.json"
    data = json.loads(path.read_text())
    config = json.loads((root / "artifacts/analysis/e23/configuration.json").read_text())
    trust_version = config["trust_list"]["version"]
    by_id = {x["id"]: x for x in data.get("experiments", [])}
    names = {"E23-A": "20-checkpoint stacker · control",
        "E23-B": "20-checkpoint stacker · plus E21 checkpoints",
        "E23-C": "20-checkpoint per-label specialist stacker"}
    for eid, name in names.items():
        entry = by_id.get(eid, {})
        entry.update({"id": eid, "name": name, "category": "Learned checkpoint ensemble",
            "description": "Fold-local roster capped at 20; E20 candidate-label CatBoost with E19-D Hungarian decoding.",
            "trust_list_version": trust_version, "checkpoints_per_fold": MAX_CHECKPOINTS,
            "primary_metric": "Balanced accuracy", "status": "Running",
            "result_source": f"artifacts/analysis/e23/metrics_{eid}.json", "documentation": "docs/e23.md"})
        if reset:
            for key in ("accuracy", "balanced_accuracy", "primary_score", "checkpoints_used", "checkpoint_count"):
                entry.pop(key, None)
        by_id[eid] = entry
    for result in results:
        eid = result["experiment"]; metric = result["metrics"]["E19-D"]
        distinct_ids = {sid for fold in result["checkpoint_rosters_by_fold"].values() for sid in fold["checkpoint_ids"]}
        by_id[eid] = {"id": eid, "name": names[eid], "category": "Learned checkpoint ensemble",
            "description": "Fold-local roster capped at 20; E20 candidate-label CatBoost with E19-D Hungarian decoding.",
            "accuracy": metric["accuracy"], "balanced_accuracy": metric["balanced_accuracy"],
            "primary_metric": "Balanced accuracy", "primary_score": metric["balanced_accuracy"],
            "trust_list_version": trust_version, "status": "Scored", "result_source": f"artifacts/analysis/e23/metrics_{eid}.json",
            "documentation": "docs/e23.md", "checkpoint_count": len(distinct_ids),
            "checkpoints_used": len(distinct_ids), "checkpoints_per_fold": MAX_CHECKPOINTS,
            "checkpoint_count_note": f"{len(distinct_ids)} distinct IDs across five fold rosters; 20 per fold."}
    data["experiments"] = list(by_id.values())
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run(root=ROOT, output=OUT):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ctx = e22.load_context(root)
    ids = list(ctx["checkpoint_ids"])
    pool_a = [sid for sid in ids if sid not in {"N0135", "N0136"}]
    pool_b = ids
    mandatory = [sid for sid in ("N0135", "N0136") if sid in ids]
    if not set(mandatory).issubset(pool_b) or len(mandatory) != 2:
        raise ValueError("E23 requires cached N0135 and N0136 probabilities")
    fold_definitions = [{"fold": f["fold"], "fit_ids": ctx["row_ids"][f["fit_rows"]].tolist(),
        "held_ids": ctx["row_ids"][f["held_rows"]].tolist()} for f in ctx["folds"]]
    (output / "fold_definitions.json").write_text(json.dumps(fold_definitions, indent=2) + "\n")
    v0_path = root / "data/processed/dataset_v0.parquet"
    config = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "max_checkpoints": MAX_CHECKPOINTS,
        "selection": "training-only greedy complementary correct-label coverage; inverse class support weighting; family novelty and standalone BA tie-breaks",
        "forced_E21_members_for_B_C": mandatory, "seed": e22.SEED, "threads": e22.THREADS,
        "depths": e22.DEPTHS, "learning_rates": e22.LEARNING_RATES, "iterations": e22.ITERATIONS,
        "labels": list(LABELS), "trust_list": ctx["snapshot"], "trust_list_sha256": ctx["trust_sha256"],
        "registry_resolved_experiments": ctx["registry_count"], "eligible_cached_checkpoint_count": len(ids),
        "excluded_checkpoints": ctx["excluded"], "checkpoint_cache_sha256": ctx["cache_hashes"],
        "v0_dataset_sha256": e22._sha(v0_path),
        "E19-D_group_file_sha256": e22._sha(root / "artifacts/analysis/e19d/oof_group_predictions.parquet"),
        "E19-F-A_group_file_sha256": e22._sha(root / "artifacts/analysis/e19f/predicted_group_assignments.parquet"),
        "source_code_sha256": e22._sha(Path(__file__)), "fold_definitions": str(output / "fold_definitions.json"),
        "V0_selection_limitation": "Trust List and N0135/N0136 checkpoint selection used V0; all E23 V0 scores are development evidence."}
    (output / "configuration.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    _register(root, [], reset=True)
    results = []
    with threadpool_limits(limits=e22.THREADS):
        results.append(_run_experiment(ctx, "E23-A", pool_a, output=output)); _register(root, results)
        results.append(_run_experiment(ctx, "E23-B", pool_b, mandatory=mandatory, output=output)); _register(root, results)
        results.append(_run_experiment(ctx, "E23-C", pool_b, mandatory=mandatory, specialist=True, output=output)); _register(root, results)
    doc = {"trust_list": ctx["snapshot"], "max_checkpoints": MAX_CHECKPOINTS,
        "experiments": [{"id": r["experiment"], "roster_size_by_fold": {
            k: len(v["checkpoint_ids"]) for k, v in r["checkpoint_rosters_by_fold"].items()},
            "independent": r["metrics"]["independent"], "E19-D": r["metrics"]["E19-D"],
            "E19-F-A": r["metrics"]["E19-F-A"], "oracle": r["metrics"]["oracle"],
            "corrected_vs_E23-A": r["comparison_vs_E23-A_E19-D"].get("corrected"),
            "introduced_vs_E23-A": r["comparison_vs_E23-A_E19-D"].get("introduced")}
            for r in results],
        "rosters": {r["experiment"]: r["checkpoint_rosters_by_fold"] for r in results}}
    (output / "results.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    _write_docs(root, config, results, doc)
    _register(root, results)
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(output=args.output), indent=2))


if __name__ == "__main__":
    main()
