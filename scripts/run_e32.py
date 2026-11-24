"""Select 20 E32 additions and evaluate a frozen 60-checkpoint router."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_e13_b_kaggle import _deployment
from scripts.run_e30_corrected_selection import metric, route
from scripts.run_e31 import _fold_definitions
from src import e13_reliability_router as e13
from src.e20 import hungarian
from src.e31_selection import E31GreedySelector
from src.e32_selection import E32GreedySelector
from src.ensemble_catalog import _load_inputs, cascades
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e32"
E31_MANIFEST = ROOT / "artifacts/analysis/e31/e31_manifest.json"
E31_SELECTION = ROOT / "artifacts/analysis/e31/selection_results.json"
E31_FULL = ROOT / "artifacts/analysis/e31/fixed_roster_full_v0_predictions.parquet"
E31_SCORES = ROOT / "artifacts/analysis/e31/roster_scenario_evaluation/predictions_and_scores.npz"
E31_SCENARIOS = ROOT / "artifacts/analysis/e31/roster_scenario_evaluation/scenario_predictions.parquet"
E13_RESULTS = ROOT / "artifacts/analysis/e13/20261120T093318893046Z/results.json"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _truth(v0):
    y = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if len(y) != 2699 or (y < 0).any():
        raise ValueError("E32 requires all 2,699 full-V0 rows and canonical labels")
    return y


def _e13_iterations():
    doc = json.loads(E13_RESULTS.read_text())
    e13a = next(x for x in doc["results"] if x["ensemble"] == "E13-A")
    return max(1, int(np.median([f["best_iterations"] for f in e13a["folds"]])))


def _fit_deployment(v0, roster, ids, members, probabilities, truth):
    router, fit = _deployment(v0, ids, members, probabilities,
                              [ids.index(sid) for sid in roster])
    path = OUT / "e32_router.cbm"
    router.save_model(str(path))
    stats = e13.reliability_statistics(
        np.stack([probabilities[sid] for sid in roster]).argmax(2), truth)
    stats_json = {k: v.tolist() for k, v in stats.items()}
    pred, score = route(v0, roster, members, probabilities, path, stats_json)
    return path, fit, stats_json, pred, score


def _compare_scenarios(v0, truth, e32_pred, e32_scores, e31_pred, e31_scores):
    scenarios = pd.read_parquet(ROOT / "data/processed/dataset_v0_group.parquet")
    saved = pd.read_parquet(E31_SCENARIOS)
    if scenarios.scenario_id.nunique() != 50 or len(scenarios) != 25_000:
        raise ValueError("Expected the fixed 50-scenario, 25,000-row v0_group benchmark")
    records = {name: [] for name in ("Independent", "E19-F-A", "E19-F-C", "E19-D")}
    row_index = {str(rid): i for i, rid in enumerate(v0.id.astype(str))}
    preds_out = []
    for scenario_id, frame in scenarios.groupby("scenario_id", sort=True):
        frame = frame.reset_index(drop=True)
        rowix = np.asarray([row_index[str(x)] for x in frame.original_id], dtype=np.int64)
        sid = frame.scenario_row_id.astype(str).to_numpy()
        y = truth[rowix]
        method_preds = {"Independent": e32_pred[rowix]}
        base_preds = {"Independent": e31_pred[rowix]}
        for method in ("E19-F-A", "E19-F-C", "E19-D"):
            old = saved[(saved.checkpoint_count == 40) &
                        (saved.configuration == method) &
                        (saved.scenario_id.astype(str) == str(scenario_id))]
            if (len(old) != len(frame) or
                    old.scenario_row_id.astype(str).tolist() != sid.tolist()):
                raise ValueError(f"E31 saved {method} groups do not align to {scenario_id}")
            groups = old.predicted_group.astype(str).to_numpy()
            if method == "Independent":
                raise AssertionError("Unexpected independent grouping path")
            method_preds[method] = hungarian(e32_scores[rowix], groups, sid)
            base_preds[method] = hungarian(e31_scores[rowix], groups, sid)
            # The frozen 40 baseline must reproduce the earlier E31 table input.
            saved_predictions = old.prediction.astype(str).tolist()
            predicted_labels = np.asarray(LABELS, dtype=object)[base_preds[method]].tolist()
            if saved_predictions != predicted_labels:
                raise AssertionError(f"E31 40 prediction parity failed for {method}/{scenario_id}")
        base_preds["Independent"] = e31_pred[rowix]
        for method in records:
            p40, p60 = base_preds[method], method_preds[method]
            records[method].append({"scenario_id": str(scenario_id),
                "balanced_accuracy_40": metric(y, p40)["balanced_accuracy"],
                "balanced_accuracy_60": metric(y, p60)["balanced_accuracy"],
                "accuracy_40": metric(y, p40)["accuracy"],
                "accuracy_60": metric(y, p60)["accuracy"]})
            preds_out.append(pd.DataFrame({"scenario_id": scenario_id,
                "scenario_row_id": sid, "original_id": frame.original_id.astype(str),
                "checkpoint_count": 60, "configuration": method,
                "truth": np.asarray(LABELS, dtype=object)[y],
                "prediction": np.asarray(LABELS, dtype=object)[p60],
                "predicted_group": sid if method == "Independent" else
                    saved[(saved.checkpoint_count == 40) &
                          (saved.configuration == method) &
                          (saved.scenario_id.astype(str) == str(scenario_id))]
                        .predicted_group.astype(str).to_numpy()}))

    summary = {}
    for method, rows in records.items():
        a = np.asarray([x["balanced_accuracy_40"] for x in rows])
        b = np.asarray([x["balanced_accuracy_60"] for x in rows])
        summary[method] = {"mean_balanced_accuracy_40": float(a.mean()),
            "mean_balanced_accuracy_60": float(b.mean()),
            "delta_balanced_accuracy": float(b.mean() - a.mean()),
            "scenario_count": len(rows), "scenario_results": rows}
    return summary, pd.concat(preds_out, ignore_index=True)


def run_selection():
    OUT.mkdir(parents=True, exist_ok=True)
    if not E31_MANIFEST.is_file() or not E31_SELECTION.is_file():
        raise FileNotFoundError("Final E31 40-checkpoint roster/artifacts are required")
    e31_manifest = json.loads(E31_MANIFEST.read_text())
    e31_selection = json.loads(E31_SELECTION.read_text())
    roster40 = list(map(str, e31_manifest["roster"]))
    if len(roster40) != 40 or roster40 != e31_selection["final_roster"]:
        raise ValueError("E31 manifest and selection result do not agree on the fixed roster")
    v0 = cascades._v0(ROOT)
    truth = _truth(v0)
    _, members, probabilities, excluded, snapshot = _load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    if excluded or not set(roster40) <= set(probabilities):
        raise ValueError(f"E32 checkpoint cache is incomplete: {excluded}")
    ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    if set(ids) != set(members):
        raise ValueError("Checkpoint metadata and probability cache roster differ")
    iterations = _e13_iterations()
    selector = E32GreedySelector(v0, members, probabilities, roster40,
        OUT / "selection", outer_folds=5, inner_folds=4, steps=20,
        iterations=iterations)
    fold_definitions, _ = _fold_definitions(v0)
    selection = selector.run(fold_definitions)
    roster60 = list(map(str, selection["final_roster"]))
    if len(roster60) != 60 or len(set(roster60)) != 60 or roster60[:40] != roster40:
        raise AssertionError("E32 selection did not produce a fixed 40+20 roster")
    router_path, router_fit, stats, pred60, scores60 = _fit_deployment(
        v0, roster60, ids, members, probabilities, truth)

    e31_ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    e31_roster = roster40
    pred40, scores40 = route(v0, e31_roster, members, probabilities,
        ROOT / "artifacts/analysis/e31/e31_router.cbm", e31_manifest["reliability_statistics"])
    saved_full = pd.read_parquet(E31_FULL)
    saved_pred40 = pd.Categorical(saved_full["E31-40"].astype(str), categories=LABELS).codes
    if saved_full.id.astype(str).tolist() != v0.id.astype(str).tolist() or not np.array_equal(pred40, saved_pred40):
        raise AssertionError("E31 40-member deployment router replay failed")
    # Reuse E31's 40 score cache as an exact score-level parity check.
    e31_score_cache = np.load(E31_SCORES, allow_pickle=True)
    if (e31_score_cache["row_ids"].astype(str).tolist() != v0.id.astype(str).tolist()
            or not np.allclose(scores40, e31_score_cache["score_40"], rtol=0, atol=1e-10)):
        raise AssertionError("E31 40-member assignment scores differ from the frozen cache")

    groups = pd.read_parquet(E31_SCENARIOS)
    scenario_summary, scenario_predictions = _compare_scenarios(
        v0, truth, pred60, scores60, pred40, scores40)
    scenario_predictions.to_parquet(OUT / "v0_group_predictions.parquet", index=False)
    np.savez_compressed(OUT / "v0_scores.npz", row_ids=v0.id.astype(str).to_numpy(),
        classes=np.asarray(LABELS), scores_40=scores40, scores_60=scores60,
        predictions_40=pred40, predictions_60=pred60)

    router_sha = _sha(router_path)
    batch_parameters = {}
    benchmark_path = OUT / "inference_benchmark.json"
    if benchmark_path.is_file():
        bench = json.loads(benchmark_path.read_text())
        batch_parameters = {x["checkpoint_id"]: {
            "maximum_stable_batch_size": x.get("maximum_stable_batch_size"),
            "peak_allocated_mib": x.get("peak_allocated_mib"),
            "peak_reserved_mib": x.get("peak_reserved_mib")}
            for x in bench.get("checkpoints", [])}
    manifest = {"experiment": "E32", "baseline_experiment": "E31",
        "baseline_checkpoint_count": 40, "added_checkpoint_count": 20,
        "checkpoint_count": 60, "roster": roster60,
        "added_checkpoint_ids": roster60[40:], "classes": list(LABELS),
        "families": [_model_family(members[sid]["model"]) for sid in roster60],
        "checkpoint_members": {sid: {key: members[sid].get(key) for key in
            ("experiment_id", "model", "training_dataset", "checkpoint_path",
             "model_revision", "config", "calibration_temperature", "cache_sha256")}
            for sid in roster60},
        "router": "e32_router.cbm", "router_sha256": router_sha,
        "router_fit": router_fit, "reliability_statistics": stats,
        "checkpoint_cache_sha256": {sid: members[sid]["cache_sha256"] for sid in roster60},
        "inference_batch_parameters": batch_parameters,
        "selection_results": "selection/selection_results.json",
        "selection_rule": selection["selection_rule"],
        "v0_group_method_comparison": scenario_summary,
        "source_hashes": {"e31_manifest": _sha(E31_MANIFEST),
            "e31_selection_results": _sha(E31_SELECTION),
            "e31_scenario_predictions": _sha(E31_SCENARIOS)}}
    (OUT / "e32_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "v0_group_results.json").write_text(json.dumps({
        "scenario_count": 50, "rows_per_scenario": 500,
        "summary": scenario_summary,
        "note": "Same scenarios and saved E31 reconstructed groups; predictions and Hungarian assignments were recomputed from the 40- and 60-roster score matrices."}, indent=2) + "\n")
    return manifest, selection, scenario_summary


if __name__ == "__main__":
    manifest, selection, summary = run_selection()
    print(json.dumps({"roster": manifest["roster"],
        "additions": manifest["added_checkpoint_ids"],
        "best_oof_roster_size_40_to_60": selection["best_oof_roster_size_40_to_60"],
        "scenario_summary": {k: {name: value for name, value in row.items()
            if name != "scenario_results"} for k, row in summary.items()}}, indent=2))
