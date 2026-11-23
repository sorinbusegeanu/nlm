"""E30: fixed 15 vs 20 checkpoint E13-B family-max routers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from . import e13_reliability_router as e13
from . import e19f, e20, e27
from .e30_deploy import hungarian, predict_f_groups
from .e13_b_inference import family_max_scores, make_features, route_family_max as shared_route
from .ensemble_catalog import _load_inputs, cascades
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e30"
E28_MANIFEST = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
E28_SCORES = ROOT / "artifacts/analysis/e28_fixed_roster/fixed_roster_scores.npz"
E28_ROUTER = ROOT / "artifacts/analysis/e28_fixed_roster/e13_b_router.cbm"
SCENARIOS = ROOT / "data/processed/dataset_v0_group.parquet"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_five(v0, members, probabilities, base_ids):
    """Choose five distinct new families by complementary correct coverage.

    V0 is the only labelled pool available for these saved caches, so this is
    explicitly development-set roster selection, not an unbiased estimate.
    """
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes
    base = np.stack([probabilities[sid] for sid in base_ids]).argmax(2)
    base_correct = (base == truth).any(0)
    current_families = {_model_family(members[sid]["model"]) for sid in base_ids}
    candidates = []
    for sid in sorted(probabilities, key=lambda x: int(x[1:])):
        if sid in base_ids:
            continue
        family = _model_family(members[sid]["model"])
        if family in current_families:
            continue
        hard = probabilities[sid].argmax(1)
        hit = hard == truth
        candidates.append({"id": sid, "family": family, "hit": hit,
            "accuracy": float(hit.mean()),
            "unique_correct": int((hit & ~base_correct).sum()),
            "correct_when_base_wrong": int((hit & ~base_correct).sum()),
            "agreement_with_base_roster": float(np.mean(hard[None, :] == base))})
    # Greedy coverage, with standalone accuracy and lower mean error agreement
    # as tie breakers; the family restriction makes every addition diverse.
    selected, covered = [], base_correct.copy()
    available = candidates.copy()
    while len(selected) < 5:
        if not available:
            raise ValueError("Eligible cached pool cannot supply five new model families")
        chosen = max(available, key=lambda c: (
            int((c["hit"] & ~covered).sum()), c["accuracy"],
            -c["agreement_with_base_roster"], c["id"]))
        chosen["added_correct_coverage"] = int((chosen["hit"] & ~covered).sum())
        selected.append(chosen)
        covered |= chosen["hit"]
        # Diversity is enforced across the five additions as well as against
        # the inherited roster: only one new checkpoint per added family.
        available = [candidate for candidate in available
                     if candidate["family"] != chosen["family"]]
    return selected


def _probability_scores(ids, members, probabilities, truth, *, router_path, stats, families):
    tensor = np.stack([probabilities[sid] for sid in ids]).astype(np.float64)
    feature_members = [{"candidate_id": sid, "family": family}
                       for sid, family in zip(ids, families)]
    features = make_features(tensor, feature_members, LABELS, stats)
    router = CatBoostClassifier(); router.load_model(str(router_path))
    reliability = router.predict_proba(features)[:, 1].reshape(len(truth), len(ids)).T
    family_probs = e13.family_balanced_probabilities(tensor, families)
    raw_scores, assignment = family_max_scores(tensor.argmax(2), reliability, family_probs, families)
    pred = shared_route(tensor.argmax(2), reliability, family_probs, families)
    return {"predictions": pred, "assignment_scores": assignment,
            "reliability": reliability, "family_probabilities": family_probs}


def _metrics(y, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_label": {LABELS[i]: {"precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(LABELS))}}


def run_evaluation():
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = cascades._v0(ROOT)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    _, members, probs, _, snapshot = _load_inputs(ROOT, v0, version="trust-list-2-v037")
    e28 = json.loads(E28_MANIFEST.read_text())
    roster15 = list(e28["deployment"]["checkpoint_ids"])
    if len(roster15) != 15 or any(sid not in probs for sid in roster15):
        raise ValueError("E28 fixed roster is incomplete in the probability cache")
    additions = select_five(v0, members, probs, roster15)
    roster20 = roster15 + [item["id"] for item in additions]
    families15 = [_model_family(members[sid]["model"]) for sid in roster15]
    families20 = [_model_family(members[sid]["model"]) for sid in roster20]

    # E30-A reuses E28's exact router and frozen deployment statistics.
    stats15 = e28["deployment"]["reliability_statistics"]
    a = _probability_scores(roster15, members, probs, truth,
        router_path=E28_ROUTER, stats=stats15, families=families15)
    fixed15 = np.load(E28_SCORES, allow_pickle=True)
    if (fixed15["row_ids"].astype(str).tolist() != v0.id.astype(str).tolist() or
            not np.array_equal(a["predictions"], fixed15["base_predictions"]) or
            not np.allclose(a["assignment_scores"], fixed15["assignment_scores"],
                            rtol=0, atol=1e-10)):
        raise AssertionError("E30-A no longer reproduces the exact E28 fixed 15-checkpoint route")
    # E30-B follows the E13-B deployment procedure on the fixed 20 roster.
    router20, dep20 = __import__("scripts.prepare_e13_b_kaggle", fromlist=["_deployment"])._deployment(
        v0, roster20, members, probs, list(range(20)))
    router20_path = OUT / "e30_b_router.cbm"; router20.save_model(str(router20_path))
    tensor20 = np.stack([probs[sid] for sid in roster20]).astype(np.float64)
    stats20 = e13.reliability_statistics(tensor20.argmax(2), truth)
    dep20["reliability_statistics"] = {k: v.tolist() for k, v in stats20.items()}
    b = _probability_scores(roster20, members, probs, truth,
        router_path=router20_path, stats=dep20["reliability_statistics"], families=families20)

    ids = v0.id.astype(str).to_numpy()
    score_file = np.load(E28_SCORES, allow_pickle=True)
    if score_file["row_ids"].astype(str).tolist() != ids.tolist():
        raise ValueError("E28 fixed scores are not aligned with current V0")
    embeddings = np.load(ROOT / "artifacts/analysis/e19f/response_embeddings.npy", mmap_mode="r")
    fc = joblib.load(ROOT / "artifacts/analysis/e28/e19_f_c_external_grouper.joblib")
    fa = {"threshold": .85, "neighbors": 40, "vectorizers": fc["vectorizers"]}
    scenarios = pd.read_parquet(SCENARIOS)
    results = {"metadata": {"experiment": "E30", "trust_list_version": snapshot["version"],
        "scenario_count": int(scenarios.scenario_id.nunique()), "scenario_rows": int(len(scenarios)),
        "v0_exposure_note": "All scenarios reuse V0 responses. Roster selection and router fitting use V0 labels; these results are in-sample robustness diagnostics, not independent validation.",
        "roster_A": roster15, "roster_B": roster20, "families_A": families15,
        "families_B": families20, "additions": [{k:v for k,v in x.items() if k != "hit"} for x in additions],
        "router_A_sha256": sha256(E28_ROUTER), "router_B_sha256": sha256(router20_path)},
        "scenarios": {}, "summary": {}}
    id_to_row = {rid:i for i,rid in enumerate(ids)}
    matrices = {"E30-A": score_file["assignment_scores"], "E30-B": b["assignment_scores"]}
    np.savez_compressed(OUT / "v0_score_matrices.npz", row_ids=ids, labels=np.asarray(LABELS),
        e30_a=matrices["E30-A"], e30_b=matrices["E30-B"])
    for method, bundle in (("E19-F-A", fa), ("E19-F-C", fc)):
        for sid, frame in scenarios.groupby("scenario_id", sort=True):
            rowix = np.asarray([id_to_row[x] for x in frame.original_id.astype(str)])
            group_ids = frame.scenario_row_id.astype(str).to_numpy()
            group_cache = OUT / f"groups_{method}_{sid}.npy"
            if group_cache.is_file():
                groups = np.load(group_cache, allow_pickle=True).astype(str)
                if len(groups) != len(rowix):
                    raise ValueError(f"cached {method} groups have stale row count for {sid}")
            else:
                groups = predict_f_groups(method, group_ids, frame.text.astype(str).to_numpy(),
                                          np.asarray(embeddings)[rowix], bundle).astype(str)
                np.save(group_cache, groups, allow_pickle=True)
            for variant, score in matrices.items():
                pred = e20.hungarian(score[rowix], groups, group_ids)
                true = truth[rowix]
                results["scenarios"].setdefault(method, {}).setdefault(str(sid), {})[variant] = {
                    "rows": int(len(rowix)), **_metrics(true, pred),
                    "predictions": pred.tolist(), "truth": true.tolist(),
                    "ids": frame.original_id.astype(str).tolist(),
                    "corrected_vs_other": None}
            scenario_result = results["scenarios"][method][str(sid)]
            pa = np.asarray(scenario_result["E30-A"]["predictions"], dtype=np.int64)
            pb = np.asarray(scenario_result["E30-B"]["predictions"], dtype=np.int64)
            yy = np.asarray(scenario_result["E30-A"]["truth"], dtype=np.int64)
            scenario_result["E30-B"]["comparison_vs_E30-A"] = {
                "corrected": int(((pa != yy) & (pb == yy)).sum()),
                "introduced": int(((pa == yy) & (pb != yy)).sum()),
                "changed": int((pa != pb).sum()),
                "net_correct": int((pb == yy).sum() - (pa == yy).sum())}
        for variant in matrices:
            vals = [results["scenarios"][method][s][variant]["balanced_accuracy"]
                    for s in results["scenarios"][method]]
            results["summary"].setdefault(method, {})[variant] = {
                "mean_balanced_accuracy": float(np.mean(vals)),
                "std_balanced_accuracy": float(np.std(vals, ddof=1)),
                "minimum_balanced_accuracy": float(np.min(vals)),
                "maximum_balanced_accuracy": float(np.max(vals)),
                "p10_balanced_accuracy": float(np.quantile(vals, .1)),
                "scenarios_improved_vs_E30-A": int(sum(
                    results["scenarios"][method][s]["E30-B"]["balanced_accuracy"] >
                    results["scenarios"][method][s]["E30-A"]["balanced_accuracy"]
                    for s in results["scenarios"][method])) if variant == "E30-B" else None,
                "scenarios_regressed_vs_E30-A": int(sum(
                    results["scenarios"][method][s]["E30-B"]["balanced_accuracy"] <
                    results["scenarios"][method][s]["E30-A"]["balanced_accuracy"]
                    for s in results["scenarios"][method])) if variant == "E30-B" else None}
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    deployment = {"experiment": "E30-B", "roster": roster20, "classes": list(LABELS),
        "families": families20, "router": "e30_b_router.cbm", "router_sha256": sha256(router20_path),
        "reliability_statistics": dep20["reliability_statistics"], "router_fit": dep20,
        "trust_list_version": snapshot["version"], "groupers": {
            "E19-F-A": {"threshold": .85, "source": "E28 frozen external grouper/vectorizers"},
            "E19-F-C": {"threshold": float(fc["threshold"]), "source": "E28 frozen external grouper"}},
        "inference_batch_parameters": {}, "selection": results["metadata"]["additions"],
        "probability_cache_sha256": {sid: members[sid]["cache_sha256"] for sid in roster20}}
    (OUT / "e30_b_manifest.json").write_text(json.dumps(deployment, indent=2) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(run_evaluation()["summary"], indent=2))
