"""Run E31's 17-to-40 E13-B checkpoint expansion and V0 diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_fscore_support

from scripts.prepare_e13_b_kaggle import _deployment
from scripts.run_e30_corrected_selection import _paired_comparison, metric, route
from src import e13_reliability_router as e13
from src.e20 import hungarian
from src.e31_selection import E31GreedySelector
from src.e19b_evaluate import grouping_metrics
from src.ensemble_catalog import _load_inputs, cascades
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e31"
E30_MANIFEST = ROOT / "artifacts/analysis/e30/e30_b_manifest.json"
E30_ROUTER = ROOT / "artifacts/analysis/e30/e30_b_router.cbm"
E30_FULL_PREDICTIONS = ROOT / "artifacts/analysis/e30/fixed_roster_full_v0_predictions.parquet"
E13_RESULTS = ROOT / "artifacts/analysis/e13/20261120T093318893046Z/results.json"
V0_GROUP = ROOT / "data/processed/dataset_v0_group.parquet"
GROUPER_C = ROOT / "artifacts/analysis/e28/e19_f_c_external_grouper.joblib"
EMBEDDINGS = ROOT / "artifacts/analysis/e19f/response_embeddings.npy"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _truth(v0):
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any():
        raise ValueError("V0 contains labels outside canonical order")
    return truth


def _fold_definitions(v0):
    result_doc = json.loads(E13_RESULTS.read_text())
    item = next(x for x in result_doc["results"] if x["ensemble"] == "E13-B")
    row_index = {str(rid): i for i, rid in enumerate(v0.id.astype(str))}
    definitions = []
    held_all = []
    for fold in item["folds"]:
        fit = np.asarray([row_index[str(rid)] for rid in fold["fit_ids"]], dtype=np.int64)
        held = np.asarray([row_index[str(rid)] for rid in fold["held_ids"]], dtype=np.int64)
        if set(v0.prompt_id.iloc[fit].astype(str)) & set(v0.prompt_id.iloc[held].astype(str)):
            raise AssertionError(f"E31 prompt leakage in outer fold {fold['fold']}")
        definitions.append((fit, held))
        held_all.extend(held.tolist())
    if sorted(held_all) != list(range(len(v0))):
        raise AssertionError("E31 outer held folds do not cover V0 exactly once")
    return definitions, item["folds"]


def _per_label(truth, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i])} for i in range(len(LABELS))}


def _fixed_scenarios(v0, truth, base_pred, base_scores, new_pred, new_scores,
                     row_ids, probabilities, members, baseline_roster, new_roster,
                     e30_manifest, e31_stats):
    import joblib
    from src.e30_deploy import predict_f_groups

    scenarios = pd.read_parquet(V0_GROUP)
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    grouper_c = joblib.load(GROUPER_C)
    grouper_a = {"threshold": 0.85, "neighbors": 40,
                 "vectorizers": grouper_c["vectorizers"]}
    row_index = {rid: i for i, rid in enumerate(row_ids)}
    methods = ("independent", "E19-F-A", "E19-F-C")
    result = {method: {} for method in methods}
    predictions = []
    group_cache_hashes = {}
    pooled_rows = {method: {"truth": [], "baseline": [], "e31": []}
                   for method in methods}

    for scenario_id, frame in scenarios.groupby("scenario_id", sort=True):
        scenario_number = int(str(scenario_id).replace("v0s", ""))
        rowix = np.asarray([row_index[str(x)] for x in frame.original_id], dtype=np.int64)
        scenario_ids = frame.scenario_row_id.astype(str).to_numpy()
        base = base_pred[rowix]
        new = new_pred[rowix]
        per_method_groups = {"independent": None}
        for method, bundle in (("E19-F-A", grouper_a), ("E19-F-C", grouper_c)):
            group_path = ROOT / "artifacts/analysis/e30" / f"groups_{method}_{scenario_id}.npy"
            if group_path.is_file():
                groups = np.load(group_path, allow_pickle=True).astype(str)
                if len(groups) != len(frame):
                    raise ValueError(f"Stale E30 grouping cache: {group_path}")
                group_cache_hashes[f"{method}/{scenario_id}"] = _sha256(group_path)
            else:
                groups = predict_f_groups(method, scenario_ids,
                    frame.text.astype(str).to_numpy(), np.asarray(embeddings)[rowix], bundle).astype(str)
                np.save(OUT / f"groups_{method}_{scenario_id}.npy", groups, allow_pickle=True)
            if any(np.count_nonzero(groups == group) > len(LABELS)
                   for group in set(groups.tolist())):
                raise ValueError(f"Reconstructed {method} group exceeds 18 rows in {scenario_id}")
            per_method_groups[method] = groups

        for method in methods:
            groups = per_method_groups[method]
            if groups is None:
                pred_base, pred_new = base.copy(), new.copy()
            else:
                pred_base = hungarian(base_scores[rowix], groups, scenario_ids)
                pred_new = hungarian(new_scores[rowix], groups, scenario_ids)
            y = truth[rowix]
            comp = _paired_comparison(y, pred_base, pred_new)
            group_metrics = (None if groups is None else
                grouping_metrics(frame.original_prompt_id.astype(str).to_numpy(), groups))
            result[method][str(scenario_number)] = {
                "rows": len(y), "baseline": metric(y, pred_base),
                "e31": metric(y, pred_new), "comparison": comp,
                "grouping_metrics": group_metrics}
            pooled_rows[method]["truth"].extend(y.tolist())
            pooled_rows[method]["baseline"].extend(np.asarray(pred_base).tolist())
            pooled_rows[method]["e31"].extend(np.asarray(pred_new).tolist())
            predictions.append(pd.DataFrame({"scenario_id": scenario_id,
                "scenario_row_id": scenario_ids, "original_id": frame.original_id.astype(str),
                "truth": np.asarray(LABELS, dtype=object)[y],
                "E30-B": np.asarray(LABELS, dtype=object)[pred_base],
                "E31-40": np.asarray(LABELS, dtype=object)[pred_new],
                "grouping_method": method,
                "predicted_group": (groups if groups is not None else scenario_ids)}))

    summary = {}
    for method, scenarios_by_id in result.items():
        base_ba = np.asarray([x["baseline"]["balanced_accuracy"]
                              for x in scenarios_by_id.values()])
        new_ba = np.asarray([x["e31"]["balanced_accuracy"]
                             for x in scenarios_by_id.values()])
        summary[method] = {"baseline_mean_balanced_accuracy": float(base_ba.mean()),
            "e31_mean_balanced_accuracy": float(new_ba.mean()),
            "mean_delta_balanced_accuracy": float((new_ba - base_ba).mean()),
            "std_balanced_accuracy": float(new_ba.std(ddof=1)),
            "minimum_balanced_accuracy": float(new_ba.min()),
            "p10_balanced_accuracy": float(np.percentile(new_ba, 10)),
            "maximum_balanced_accuracy": float(new_ba.max()),
            "scenarios_improved": int((new_ba > base_ba).sum()),
            "scenarios_regressed": int((new_ba < base_ba).sum()),
            "scenarios_tied": int((new_ba == base_ba).sum()),
            "comparison_totals": {key: int(sum(x["comparison"][key]
                for x in scenarios_by_id.values()))
                for key in ("corrected", "introduced", "changed", "net_correct")}}
        group_records = [x["grouping_metrics"] for x in scenarios_by_id.values()
                         if x.get("grouping_metrics") is not None]
        if group_records:
            summary[method]["grouping"] = {
                "mean_pairwise_precision": float(np.mean([x["pairwise_precision"] for x in group_records])),
                "mean_pairwise_recall": float(np.mean([x["pairwise_recall"] for x in group_records])),
                "mean_pairwise_f1": float(np.mean([x["pairwise_f1"] for x in group_records])),
                "mean_reconstructed_group_count": float(np.mean([x["predicted_group_count"] for x in group_records])),
                "mean_singleton_responses": float(np.mean([x["singleton_responses"] for x in group_records])),
                "mean_incorrectly_merged_groups": float(np.mean([x["incorrectly_merged_groups"] for x in group_records])),
                "mean_split_original_groups": float(np.mean([x["split_original_groups"] for x in group_records])),
                "mean_predicted_group_size_distribution": {
                    size: float(np.mean([x["predicted_size_distribution"].get(size, 0)
                                         for x in group_records]))
                    for size in sorted({size for x in group_records
                                        for size in x["predicted_size_distribution"]})}}

    pooled_label_metrics = {}
    for method in methods:
        y_all = np.asarray(pooled_rows[method]["truth"], dtype=np.int64)
        base_all = np.asarray(pooled_rows[method]["baseline"], dtype=np.int64)
        new_all = np.asarray(pooled_rows[method]["e31"], dtype=np.int64)
        base_label = _per_label(y_all, base_all)
        new_label = _per_label(y_all, new_all)
        pooled_label_metrics[method] = {label: {
            "E30_recall": base_label[label]["recall"],
            "E31_recall": new_label[label]["recall"],
            "recall_delta": new_label[label]["recall"] - base_label[label]["recall"]}
            for label in LABELS}

    pd.concat(predictions, ignore_index=True).to_parquet(
        OUT / "v0_group_scenario_predictions.parquet", index=False)
    scenario_source_hash = hashlib.sha256(pd.util.hash_pandas_object(
        scenarios[["scenario_id", "scenario_row_id", "original_id", "original_prompt_id"]],
        index=False).values.tobytes()).hexdigest()
    return {"scenario_count": len(result["independent"]), "scenarios": result,
            "summary": summary, "pooled_per_label_recall": pooled_label_metrics,
            "scenario_source_sha256": scenario_source_hash,
            "reused_e30_group_cache_sha256": group_cache_hashes,
            "note": "The 50 v0_group scenarios are correlated reuses of V0 rows and are deployment diagnostics, not independent validation."}


def _fit_and_evaluate_fixed_roster(v0, members, probabilities, snapshot, selection):
    import joblib
    from src.e30_deploy import predict_f_groups

    OUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(E30_MANIFEST.read_text())
    roster = list(selection["final_roster"])
    if len(roster) != 40 or len(set(roster)) != 40:
        raise ValueError("E31 deployment roster must contain exactly 40 unique checkpoints")
    if set(roster) - set(probabilities):
        raise ValueError("E31 final roster has missing v037 probabilities")
    truth, row_ids = _truth(v0), v0.id.astype(str).to_numpy()
    ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    roster_indices = [ids.index(sid) for sid in roster]
    router, router_fit = _deployment(v0, ids, members, probabilities, roster_indices)
    router_path = OUT / "e31_router.cbm"
    router.save_model(str(router_path))
    families = [_model_family(members[sid]["model"]) for sid in roster]
    stats = e13.reliability_statistics(
        np.stack([probabilities[sid] for sid in roster]).argmax(2), truth)
    stats_json = {key: value.tolist() for key, value in stats.items()}
    e31_pred, e31_scores = route(v0, roster, members, probabilities, router_path, stats_json)
    baseline_roster = list(baseline["roster"])
    e30_pred, e30_scores = route(v0, baseline_roster, members, probabilities,
                                  E30_ROUTER, baseline["reliability_statistics"])

    saved = pd.read_parquet(E30_FULL_PREDICTIONS)
    if saved.id.astype(str).tolist() != row_ids.tolist():
        raise AssertionError("E30 fixed V0 predictions are not row aligned")
    saved_e30 = pd.Categorical(saved["E30-B"].astype(str), categories=LABELS).codes
    if not np.array_equal(e30_pred, saved_e30):
        raise AssertionError("E31 baseline reproduction differs from frozen E30-B predictions")

    base_metric, new_metric = metric(truth, e30_pred), metric(truth, e31_pred)
    full = {"baseline_E30_B_17": {**base_metric, "per_label": _per_label(truth, e30_pred)},
            "E31_40": {**new_metric, "per_label": _per_label(truth, e31_pred)},
            "comparison": _paired_comparison(truth, e30_pred, e31_pred),
            "v0_usage_note": "Fixed-roster CatBoost routers and reliability features were fit from V0 out-of-fold records/full-V0 reliability statistics; this full-V0 replay is an in-sample development diagnostic."}
    grouper_c = joblib.load(GROUPER_C)
    grouper_a = {"threshold": 0.85, "neighbors": 40,
                 "vectorizers": grouper_c["vectorizers"]}
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    row_ids = v0.id.astype(str).to_numpy()
    texts = v0.text.astype(str).to_numpy()
    true_groups = v0.prompt_id.astype(str).to_numpy()
    full_grouped = {}
    grouped_frames = []
    for method, bundle in (("E19-F-A", grouper_a), ("E19-F-C", grouper_c)):
        before = time.perf_counter()
        groups = predict_f_groups(method, row_ids, texts, embeddings, bundle).astype(str)
        elapsed = time.perf_counter() - before
        if any(np.count_nonzero(groups == group) > len(LABELS) for group in set(groups.tolist())):
            raise ValueError(f"Full-V0 {method} group exceeds 18 rows")
        base_grouped = hungarian(e30_scores, groups, row_ids)
        new_grouped = hungarian(e31_scores, groups, row_ids)
        gm = grouping_metrics(true_groups, groups)
        full_grouped[method] = {"baseline": metric(truth, base_grouped),
            "e31": metric(truth, new_grouped),
            "comparison": _paired_comparison(truth, base_grouped, new_grouped),
            "grouping_metrics": gm, "grouping_runtime_seconds": float(elapsed)}
        grouped_frames.append(pd.DataFrame({"id": row_ids, "prompt_id": true_groups,
            "predicted_group": groups, "truth": np.asarray(LABELS, dtype=object)[truth],
            "E30-B": np.asarray(LABELS, dtype=object)[base_grouped],
            "E31-40": np.asarray(LABELS, dtype=object)[new_grouped],
            "grouping_method": method}))
    pd.concat(grouped_frames, ignore_index=True).to_parquet(
        OUT / "fixed_roster_full_v0_grouped_predictions.parquet", index=False)
    full["reconstructed_group_hungarian"] = full_grouped
    pd.DataFrame({"id": row_ids, "prompt_id": v0.prompt_id.astype(str),
        "y_true": np.asarray(LABELS, dtype=object)[truth],
        "E30-B": np.asarray(LABELS, dtype=object)[e30_pred],
        "E31-40": np.asarray(LABELS, dtype=object)[e31_pred]}).to_parquet(
            OUT / "fixed_roster_full_v0_predictions.parquet", index=False)

    group_scenarios = _fixed_scenarios(v0, truth, e30_pred, e30_scores,
        e31_pred, e31_scores, row_ids, probabilities, members, baseline_roster,
        roster, baseline, stats_json)

    manifest = {"experiment": "E31", "trust_list_version": snapshot["version"],
        "baseline_experiment": "E30-B", "baseline_roster": baseline_roster,
        "roster": roster, "classes": list(LABELS), "families": families,
        "router": "e31_router.cbm", "router_sha256": _sha256(router_path),
        "router_fit": router_fit,
        "reliability_statistics": stats_json,
        "baseline_router_sha256": _sha256(E30_ROUTER),
        "probability_cache_sha256": {sid: members[sid]["cache_sha256"] for sid in roster},
        "selection_consensus": selection["candidate_ranking"],
        "selection": {"consensus_additions": selection["consensus_additions"],
            "candidate_ranking": selection["candidate_ranking"],
            "best_oof_roster_size_20_to_40": selection["best_oof_roster_size_20_to_40"],
            "best_oof_balanced_accuracy_20_to_40": selection["best_oof_balanced_accuracy_20_to_40"]},
        "baseline": {"roster": baseline_roster},
        "base_models": baseline.get("base_models", {}),
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in (
            ROOT / "src/e31_selection.py", ROOT / "src/e30_selection.py",
            ROOT / "src/e13_reliability_router.py", ROOT / "src/e13_b_inference.py",
            ROOT / "src/e30_deploy.py", ROOT / "scripts/run_e31.py")},
        "grouping_inputs_sha256": {str(path.relative_to(ROOT)): _sha256(path)
            for path in (GROUPER_C, EMBEDDINGS, V0_GROUP)},
        "selection_rule": selection["selection_rule"],
        "v0_labels_used_for": "Fold-local candidate/router fitting, final router fit, reliability statistics and reported evaluation; V0 is development data, not an untouched test set."}
    (OUT / "e31_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "fixed_roster_results.json").write_text(json.dumps({
        "trust_list_version": snapshot["version"], "checkpoint_count": len(roster),
        "baseline_checkpoint_count": len(baseline_roster), "full_v0": full,
        "v0_group_scenarios": group_scenarios}, indent=2) + "\n")
    return {"full_v0": full, "scenario_summary": group_scenarios["summary"],
            "manifest": manifest}


def run(force_selection=False):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = cascades._v0(ROOT)
    _, members, probabilities, excluded, snapshot = _load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    if len(probabilities) != 87 or excluded:
        raise ValueError(f"Expected 87 cached v037 probability sets, got {len(probabilities)}; excluded={excluded}")
    e30_manifest = json.loads(E30_MANIFEST.read_text())
    baseline_roster = list(map(str, e30_manifest["roster"]))
    if len(baseline_roster) != 17 or not set(baseline_roster) <= set(probabilities):
        raise ValueError("Current E30-B 17-checkpoint roster is unavailable in v037 probability caches")
    for sid in baseline_roster:
        if members[sid]["cache_sha256"] != e30_manifest["probability_cache_sha256"][sid]:
            raise AssertionError(f"E30 baseline cache changed for {sid}")
    definitions, fold_records = _fold_definitions(v0)
    e13_results = json.loads(E13_RESULTS.read_text())
    e13a = next(x for x in e13_results["results"] if x["ensemble"] == "E13-A")
    iterations = max(1, int(np.median([f["best_iterations"] for f in e13a["folds"]])))
    selection_path = OUT / "selection_results.json"
    selection = None
    if selection_path.is_file() and not force_selection:
        existing = json.loads(selection_path.read_text())
        if existing.get("baseline_roster") == baseline_roster and len(existing.get("final_roster", [])) == 40:
            selection = existing
    if selection is None:
        selector = E31GreedySelector(v0, members, probabilities, baseline_roster,
            OUT / "selection", outer_folds=5, inner_folds=4, steps=23,
            iterations=iterations)
        selection = selector.run(definitions)
    # Store fold sizes and exact row coverage alongside the consensus roster.
    selection["outer_fold_definitions"] = [{"fold": i + 1,
        "fit_rows": len(fit), "held_rows": len(held),
        "fit_prompt_groups": len(set(v0.prompt_id.iloc[fit].astype(str))),
        "held_prompt_groups": len(set(v0.prompt_id.iloc[held].astype(str)))}
        for i, (fit, held) in enumerate(definitions)]
    selection["v0_row_ids_sha256"] = hashlib.sha256(
        "\n".join(v0.id.astype(str)).encode()).hexdigest()
    selection["trust_list_version"] = snapshot["version"]
    selection["candidate_pool_size"] = len(probabilities) - len(baseline_roster)
    selection["probability_cache_sha256"] = {
        sid: members[sid]["cache_sha256"] for sid in sorted(
            probabilities, key=lambda checkpoint_id: int(checkpoint_id[1:]))}
    selection["selection_configuration"] = {
        "baseline_checkpoint_count": len(baseline_roster), "target_checkpoint_count": 40,
        "additions": 23, "candidate_pool_count": len(probabilities) - len(baseline_roster),
        "outer_folds": 5, "inner_folds": 4, "fold_grouping": "prompt_id GroupKFold",
        "objective": "maximize inner out-of-fold balanced accuracy; tie-break net correct, fewer introduced errors, checkpoint ID",
        "router_iterations": iterations, "router_method": "E13-B family_max CatBoost"}
    selection["selection_runtime_seconds"] = time.perf_counter() - started
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    deployment = _fit_and_evaluate_fixed_roster(v0, members, probabilities,
                                                 snapshot, selection)
    output = {"experiment": "E31", "trust_list_version": snapshot["version"],
        "baseline_roster_size": len(baseline_roster), "target_roster_size": 40,
        "selection": {key: selection[key] for key in (
            "consensus_additions", "best_oof_roster_size_20_to_40",
            "best_oof_balanced_accuracy_20_to_40")},
        "full_v0": deployment["full_v0"],
        "scenario_summary": deployment["scenario_summary"],
        "total_runtime_seconds": time.perf_counter() - started}
    (OUT / "results_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-selection", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(force_selection=args.force_selection), indent=2), flush=True)
