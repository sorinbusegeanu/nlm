"""Run leakage-controlled E30 greedy additions and fixed-roster V0 diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from scripts.prepare_e13_b_kaggle import _deployment
from src import e13_reliability_router as e13
from src.e13_b_inference import family_max_scores, route_family_max
from src.e19f import _transform_lexical
from src.e20 import hungarian
from src.e30_deploy import predict_f_groups
from src.e30_selection import E30GreedySelector
from src.ensemble_catalog import _load_inputs, cascades
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e30"
V0_GROUP = ROOT / "data/processed/dataset_v0_group.parquet"
V037_MANIFEST = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
V037_SCORES = ROOT / "artifacts/analysis/e28_fixed_roster/fixed_roster_scores.npz"
V037_ROUTER = ROOT / "artifacts/analysis/e28_fixed_roster/e13_b_router.cbm"
E13_RESULTS = ROOT / "artifacts/analysis/e13/20261120T093318893046Z/results.json"
GROUPER_C = ROOT / "artifacts/analysis/e28/e19_f_c_external_grouper.joblib"
EMBEDDINGS = ROOT / "artifacts/analysis/e19f/response_embeddings.npy"


def metric(y, pred):
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {"rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "per_label": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(support[i])} for i in range(len(LABELS))}}


def route(v0, roster, members, probabilities, router_path, reliability_statistics):
    p = np.stack([probabilities[sid] for sid in roster]).astype(np.float64)
    families = [_model_family(members[sid]["model"]) for sid in roster]
    tensor = p
    feature_members = [{"candidate_id": sid, "family": fam}
                       for sid, fam in zip(roster, families)]
    features = __import__("src.e13_b_inference", fromlist=["make_features"]).make_features(
        tensor, feature_members, LABELS, reliability_statistics)
    model = CatBoostClassifier(); model.load_model(str(router_path))
    rel = model.predict_proba(features)[:, 1].reshape(len(v0), len(roster)).T
    family_probs = e13.family_balanced_probabilities(tensor, families)
    pred = route_family_max(tensor.argmax(2), rel, family_probs, families)
    _, assign = family_max_scores(tensor.argmax(2), rel, family_probs, families)
    return pred, assign


def _paired_comparison(y, base, new):
    return {"corrected": int(((base != y) & (new == y)).sum()),
            "introduced": int(((base == y) & (new != y)).sum()),
            "changed": int((base != new).sum()),
            "net_correct": int((new == y).sum() - (base == y).sum())}


def evaluate_scenarios(v0, members, probs, roster_a, roster_b, router_b, stats_b):
    import joblib
    scenarios = pd.read_parquet(V0_GROUP)
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    grouper_c = joblib.load(GROUPER_C)
    grouper_a = {"threshold": 0.85, "neighbors": 40,
                 "vectorizers": grouper_c["vectorizers"]}
    ids = v0.id.astype(str).to_numpy()
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    row_lookup = {rid: i for i, rid in enumerate(ids)}
    full_roster_stats_a = json.loads(V037_MANIFEST.read_text())["deployment"]["reliability_statistics"]
    # Cached group predictions depend only on texts and the scenario, not the roster.
    results = {"metadata": {"scenario_count": int(scenarios.scenario_id.nunique()),
        "scenario_rows": int(len(scenarios)), "v0_reuse_note": "Each scenario reuses rows from V0; these are correlated in-sample deployment diagnostics."},
        "scenarios": {}, "summary": {}}
    for method, grouper in (("independent", None), ("E19-F-A", grouper_a), ("E19-F-C", grouper_c)):
        for scenario_id, frame in scenarios.groupby("scenario_id", sort=True):
            rowix = np.asarray([row_lookup[x] for x in frame.original_id.astype(str)], dtype=np.int64)
            scenario_ids = frame.scenario_row_id.astype(str).to_numpy()
            scenario_probs = {sid: probs[sid][rowix] for sid in set(roster_a) | set(roster_b)}
            a_pred, a_scores = route(v0.iloc[rowix], roster_a, members, scenario_probs, V037_ROUTER,
                                     full_roster_stats_a)
            b_pred, b_scores = route(v0.iloc[rowix], roster_b, members, scenario_probs, router_b, stats_b)
            if method != "independent":
                cache = OUT / f"groups_{method}_{scenario_id}.npy"
                if cache.is_file():
                    groups = np.load(cache, allow_pickle=True).astype(str)
                    if len(groups) != len(frame):
                        raise ValueError(f"stale E30 group cache {cache}")
                else:
                    groups = predict_f_groups(method, scenario_ids,
                        frame.text.astype(str).to_numpy(), np.asarray(embeddings)[rowix], grouper).astype(str)
                    np.save(cache, groups, allow_pickle=True)
                a_pred = hungarian(a_scores, groups, scenario_ids)
                b_pred = hungarian(b_scores, groups, scenario_ids)
            y = truth[rowix]
            results["scenarios"].setdefault(method, {})[str(scenario_id)] = {
                "row_ids": frame.original_id.astype(str).tolist(),
                "truth": y.tolist(), "E30-A": {**metric(y, a_pred), "predictions": a_pred.tolist()},
                "E30-B": {**metric(y, b_pred), "predictions": b_pred.tolist(),
                          "comparison_vs_E30-A": _paired_comparison(y, a_pred, b_pred)}}
            if method == "independent":
                del a_scores, b_scores
        for variant in ("E30-A", "E30-B"):
            blocks = [results["scenarios"][method][str(sid)][variant]
                      for sid in sorted(scenarios.scenario_id.unique())]
            vals = np.asarray([block["balanced_accuracy"] for block in blocks], dtype=float)
            y = np.concatenate([np.asarray(results["scenarios"][method][str(sid)]["truth"])
                                for sid in sorted(scenarios.scenario_id.unique())])
            pred = np.concatenate([np.asarray(block["predictions"]) for block in blocks])
            pooled = metric(y, pred)
            results["summary"].setdefault(method, {})[variant] = {
                "mean_balanced_accuracy": float(vals.mean()), "std_balanced_accuracy": float(vals.std(ddof=1)),
                "minimum_balanced_accuracy": float(vals.min()), "maximum_balanced_accuracy": float(vals.max()),
                "p10_balanced_accuracy": float(np.quantile(vals, 0.10)),
                "pooled_metrics": pooled}
        a = results["summary"][method]["E30-A"]
        b = results["summary"][method]["E30-B"]
        b["scenarios_improved_vs_E30-A"] = int(sum(
            results["scenarios"][method][str(sid)]["E30-B"]["balanced_accuracy"] >
            results["scenarios"][method][str(sid)]["E30-A"]["balanced_accuracy"]
            for sid in sorted(scenarios.scenario_id.unique())))
        b["scenarios_regressed_vs_E30-A"] = int(sum(
            results["scenarios"][method][str(sid)]["E30-B"]["balanced_accuracy"] <
            results["scenarios"][method][str(sid)]["E30-A"]["balanced_accuracy"]
            for sid in sorted(scenarios.scenario_id.unique())))
        b["comparison_totals"] = {key: sum(results["scenarios"][method][str(sid)]["E30-B"][
            "comparison_vs_E30-A"][key] for sid in sorted(scenarios.scenario_id.unique()))
            for key in ("corrected", "introduced", "changed", "net_correct")}
        b["per_label_recall_delta"] = {label: b["pooled_metrics"]["per_label"][label]["recall"] -
            a["pooled_metrics"]["per_label"][label]["recall"] for label in LABELS}
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = cascades._v0(ROOT)
    _, members, probabilities, excluded, snapshot = _load_inputs(
        ROOT, v0, version="trust-list-2-v039")
    if len(probabilities) != 89 or excluded:
        raise ValueError(f"Expected 89 available v039 checkpoints, got {len(probabilities)}; exclusions={excluded}")
    v037 = json.loads(V037_MANIFEST.read_text())
    baseline_roster = list(v037["deployment"]["checkpoint_ids"])
    if len(baseline_roster) != 15 or not set(baseline_roster) <= set(probabilities):
        raise ValueError("E13-B v037 baseline roster is incomplete")
    fixed = np.load(V037_SCORES, allow_pickle=True)
    base_pred, base_scores = route(v0, baseline_roster, members, probabilities,
        V037_ROUTER, v037["deployment"]["reliability_statistics"])
    if (fixed["row_ids"].astype(str).tolist() != v0.id.astype(str).tolist() or
            not np.array_equal(base_pred, fixed["base_predictions"]) or
            not np.allclose(base_scores, fixed["assignment_scores"], rtol=0, atol=1e-10)):
        raise AssertionError("Exact frozen E13-B v037 baseline reproduction failed")

    e13_results = json.loads(E13_RESULTS.read_text())
    e13a = next(x for x in e13_results["results"] if x["ensemble"] == "E13-A")
    iterations = max(1, int(np.median([f["best_iterations"] for f in e13a["folds"]])))
    fold_record = next(x for x in e13_results["results"] if x["ensemble"] == "E13-B")["folds"]
    row_index = {rid: i for i, rid in enumerate(v0.id.astype(str))}
    definitions = []
    for item in fold_record:
        fit = np.asarray([row_index[rid] for rid in item["fit_ids"]], dtype=np.int64)
        held = np.asarray([row_index[rid] for rid in item["held_ids"]], dtype=np.int64)
        if set(v0.prompt_id.iloc[fit]) & set(v0.prompt_id.iloc[held]):
            raise AssertionError("Saved E13 outer fold has prompt leakage")
        definitions.append((fit, held))
    selector = E30GreedySelector(v0, members, probabilities, baseline_roster,
        OUT / "selection", outer_folds=5, inner_folds=4, steps=5, iterations=iterations)
    selected = selector.run(definitions)

    roster = selected["final_roster"]
    if not 15 <= len(roster) <= 20:
        raise ValueError(f"Unexpected final E30 roster size: {len(roster)}")
    router, router_fit = _deployment(v0, roster, members, probabilities, list(range(len(roster))))
    router_path = OUT / "e30_b_router.cbm"; router.save_model(str(router_path))
    stats = e13.reliability_statistics(
        np.stack([probabilities[sid] for sid in roster]).argmax(2),
        pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64))
    stats_json = {k: v.tolist() for k, v in stats.items()}
    final_pred, final_scores = route(v0, roster, members, probabilities, router_path, stats_json)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    full_result = {"E30-A-frozen-v037": metric(truth, base_pred),
                   "E30-B-final-roster-in-sample": metric(truth, final_pred),
                   "comparison": _paired_comparison(truth, base_pred, final_pred)}
    full_rows = pd.DataFrame({"id": v0.id.astype(str), "prompt_id": v0.prompt_id.astype(str),
        "y_true": np.asarray(LABELS, dtype=object)[truth],
        "E30-A": np.asarray(LABELS, dtype=object)[base_pred],
        "E30-B": np.asarray(LABELS, dtype=object)[final_pred]})
    full_rows.to_parquet(OUT / "fixed_roster_full_v0_predictions.parquet", index=False)

    scenarios = evaluate_scenarios(v0, members, probabilities, baseline_roster,
                                   roster, router_path, stats_json)
    (OUT / "v0_group_evaluation.json").write_text(json.dumps(scenarios, indent=2) + "\n")
    deployment = {"experiment": "E30-B", "roster": roster, "classes": list(LABELS),
        "families": [_model_family(members[sid]["model"]) for sid in roster],
        "router": "e30_b_router.cbm", "router_sha256": hashlib.sha256(router_path.read_bytes()).hexdigest(),
        "reliability_statistics": stats_json, "router_fit": router_fit,
        "trust_list_version": snapshot["version"], "baseline": {
            "ensemble": "E13-B", "version": "v037", "roster": baseline_roster,
            "router_sha256": hashlib.sha256(V037_ROUTER.read_bytes()).hexdigest()},
        "selection": selected["candidate_ranking"],
        "selection_rule": selected["selection_rule"],
        "inference_batch_parameters": {},
        "probability_cache_sha256": {sid: members[sid]["cache_sha256"] for sid in roster},
        "checkpoint_members": {sid: {k: members[sid][k] for k in
            ("experiment_id", "model", "training_dataset", "checkpoint_path", "cache_sha256")}
            for sid in roster}}
    (OUT / "e30_b_manifest.json").write_text(json.dumps(deployment, indent=2) + "\n")
    (OUT / "final_v0_metrics.json").write_text(json.dumps(full_result, indent=2) + "\n")
    print(json.dumps({"roster": roster, "additions": roster[15:],
        "outer_oof_metrics_by_step": selected["outer_oof_metrics_by_step"],
        "full_v0": full_result,
        "scenario_summary": scenarios["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
