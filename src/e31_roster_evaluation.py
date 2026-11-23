"""Comparable E31 v0_group evaluation for fixed checkpoint rosters.

The module evaluates deployment routers on the same 50 scenario batches and
rebuilds each frozen E19 grouping method independently inside each batch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.prepare_e13_b_kaggle import _deployment
from scripts.run_e30_corrected_selection import metric
from src import e13_reliability_router as e13
from src import e28
from src.e20 import hungarian
from src.ensemble_catalog import _load_inputs, cascades
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e31/roster_scenario_evaluation"
V0_GROUP = ROOT / "data/processed/dataset_v0_group.parquet"
E28_MANIFEST = ROOT / "artifacts/analysis/e28_fixed_roster/e13_b_manifest.json"
E28_ROUTER = ROOT / "artifacts/analysis/e28_fixed_roster/e13_b_router.cbm"
E30_MANIFEST = ROOT / "artifacts/analysis/e30/e30_b_manifest.json"
E30_ROUTER = ROOT / "artifacts/analysis/e30/e30_b_router.cbm"
E30_FULL = ROOT / "artifacts/analysis/e30/fixed_roster_full_v0_predictions.parquet"
E31_MANIFEST = ROOT / "artifacts/analysis/e31/e31_manifest.json"
E31_ROUTER = ROOT / "artifacts/analysis/e31/e31_router.cbm"
E31_FULL = ROOT / "artifacts/analysis/e31/fixed_roster_full_v0_predictions.parquet"
ROSTER20_SOURCE = ROOT / "artifacts/analysis/e30/evaluation.json"
METHODS = ("Independent", "E19-F-A", "E19-F-C", "E19-D")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rosters(probabilities):
    d15 = json.loads((ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json").read_text())
    d17 = json.loads(E30_MANIFEST.read_text())
    d20 = json.loads(ROSTER20_SOURCE.read_text())["metadata"]
    result = {
        15: list(map(str, d15["deployment"]["checkpoint_ids"])),
        17: list(map(str, d17["roster"])),
        20: list(map(str, d20["roster_B"])),
    }
    selection_path = ROOT / "artifacts/analysis/e31/selection_results.json"
    if E31_MANIFEST.is_file():
        result[40] = list(map(str, json.loads(E31_MANIFEST.read_text())["roster"]))
    elif selection_path.is_file():
        result[40] = list(map(str, json.loads(selection_path.read_text())["final_roster"]))
    else:
        result[40] = None
    for count, roster in result.items():
        if roster is None:
            continue
        if len(roster) != count or len(set(roster)) != count:
            raise ValueError(f"Roster {count} is not fixed at {count} unique checkpoints")
        missing = set(roster) - set(probabilities)
        if missing:
            raise FileNotFoundError(f"Roster {count} lacks cached probabilities: {sorted(missing)}")
    return result


def _stats(roster, probabilities, truth):
    tensor = np.stack([probabilities[sid] for sid in roster]).astype(np.float64)
    return {k: v.tolist() for k, v in e13.reliability_statistics(
        tensor.argmax(axis=2), truth).items()}


def _router_for(count, roster, v0, ids, members, probabilities, truth, stats):
    if count == 15:
        frozen = json.loads((ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json").read_text())
        saved = frozen["deployment"]["reliability_statistics"]
        _check_stats(stats, saved, count)
        return E28_ROUTER, saved, "frozen E13-B v037 deployment router"
    if count == 17:
        saved = json.loads(E30_MANIFEST.read_text())
        _check_stats(stats, saved["reliability_statistics"], count)
        return E30_ROUTER, saved["reliability_statistics"], "frozen corrected E30-B deployment router"
    if count == 40:
        if not E31_ROUTER.is_file():
            return None, stats, "waiting for E31 40-checkpoint deployment router"
        if E31_MANIFEST.is_file():
            saved = json.loads(E31_MANIFEST.read_text())
            if saved["roster"] != roster or saved["router_sha256"] != _sha(E31_ROUTER):
                raise AssertionError("E31 40-checkpoint router/roster manifest mismatch")
            _check_stats(stats, saved["reliability_statistics"], count)
            saved_stats = saved["reliability_statistics"]
            source = "frozen E31 deployment router"
        else:
            selected = json.loads((ROOT / "artifacts/analysis/e31/selection_results.json").read_text())
            if selected["final_roster"] != roster:
                raise AssertionError("E31 selection result and 40-checkpoint roster mismatch")
            # The E31 run saved its trained router and complete-V0 prediction
            # replay before failing in later analysis/report work. Rebuild the
            # same fixed full-V0 reliability statistics from the saved roster.
            saved_stats = stats
            source = "trained E31 deployment router; verified against saved full-V0 replay"
        return E31_ROUTER, saved_stats, source
    # The historical 20-member candidate has no saved deployment router. Fit
    # its deployment router with the same 4-fold OOF procedure as E28/E30/E31.
    path = OUT / "router_20.cbm"
    manifest_path = OUT / "router_20.json"
    if path.is_file() and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if saved["roster"] == roster and saved["router_sha256"] == _sha(path):
            return path, saved["reliability_statistics"], "saved fixed 20-checkpoint deployment router"
    router, fit = _deployment(v0, ids, members, probabilities,
                              [ids.index(sid) for sid in roster])
    router.save_model(str(path))
    manifest_path.write_text(json.dumps({"checkpoint_count": count, "roster": roster,
        "router_sha256": _sha(path), "router_fit": fit,
        "reliability_statistics": stats,
        "note": "V0 diagnostic deployment router; training uses prompt-group OOF records."},
        indent=2) + "\n")
    return path, stats, "new fixed 20-checkpoint deployment router"


def _check_stats(expected, saved, count):
    for key, values in expected.items():
        if key not in saved or not np.array_equal(np.asarray(values), np.asarray(saved[key])):
            raise AssertionError(f"Roster {count} frozen reliability statistic differs: {key}")


def _groups_for_scenario(method, frame, rowix, embeddings, f_a, f_c, d_model,
                         d_recon, d_semantic):
    ids = frame.scenario_row_id.astype(str).to_numpy()
    texts = frame.text.astype(str).to_numpy()
    if method in ("E19-F-A", "E19-F-C"):
        bundle = f_a if method == "E19-F-A" else f_c
        return e28._predict_f_groups(method, ids, texts,
            np.asarray(embeddings)[rowix], bundle).astype(str)
    scenario_recon = d_recon.iloc[rowix].reset_index(drop=True).copy()
    # Give the cached per-response reconstruction features scenario-local IDs;
    # the underlying text/reconstruction and semantic embeddings stay aligned.
    scenario_recon["id"] = ids
    return e28._predict_d_groups(ids, texts, scenario_recon,
        np.asarray(d_semantic)[rowix], d_model).astype(str)


def evaluate(*, allow_partial=False):
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = cascades._v0(ROOT)
    _, members, probabilities, excluded, snapshot = _load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    if excluded:
        raise ValueError(f"Cached checkpoint probability set has exclusions: {excluded}")
    ids = v0.id.astype(str).to_numpy()
    id_to_row = {rid: i for i, rid in enumerate(ids)}
    checkpoint_ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any() or len(v0) != 2699:
        raise ValueError("Expected 2,699 full V0 rows in canonical label order")
    rosters = _rosters(probabilities)
    scenarios = pd.read_parquet(V0_GROUP)
    if scenarios.scenario_id.nunique() != 50:
        raise ValueError("v0_group must contain exactly 50 scenarios")
    scenario_frames = [(str(sid), frame.reset_index(drop=True))
                       for sid, frame in scenarios.groupby("scenario_id", sort=True)]
    for sid, frame in scenario_frames:
        if frame.original_id.astype(str).duplicated().any():
            raise ValueError(f"Scenario {sid} repeats an original V0 response")
        if len(frame) != 500:
            raise ValueError(f"Scenario {sid} has {len(frame)} rows; expected 500")
        if set(frame.original_id.astype(str)) - set(id_to_row):
            raise ValueError(f"Scenario {sid} contains an ID outside V0")

    import joblib
    f_c = joblib.load(e28.OUT / "e19_f_c_external_grouper.joblib")
    f_a = {"threshold": 0.85, "neighbors": 40,
           "max_cluster_size": 18, "vectorizers": f_c["vectorizers"]}
    d_model = joblib.load(e28.OUT / "e19_d_external_grouper.joblib")
    embeddings = np.load(ROOT / "artifacts/analysis/e19f/response_embeddings.npy", mmap_mode="r")
    d_recon = pd.read_parquet(ROOT / "artifacts/analysis/e19d/reconstructed_prompts.parquet")
    d_semantic = np.load(ROOT / "artifacts/analysis/e19d/semantic_embeddings.npy", mmap_mode="r")
    if (len(embeddings) != len(v0) or d_recon.id.astype(str).tolist() != ids.tolist()
            or len(d_semantic) != len(v0)):
        raise ValueError("Frozen E19 grouping inputs do not align to full V0")

    router_info = {}
    full_predictions = {}
    score_matrices = {}
    pending = []
    for count, roster in sorted(rosters.items()):
        if roster is None:
            pending.append(count)
            continue
        stats = _stats(roster, probabilities, truth)
        router_path, router_stats, router_source = _router_for(
            count, roster, v0, checkpoint_ids, members, probabilities, truth, stats)
        if router_path is None:
            pending.append(count)
            continue
        # Shared E13 features and family-max routing are used for both the
        # independent decision and the Hungarian assignment score matrix.
        from src.e13_b_inference import make_features, family_max_scores, route_family_max
        tensor = np.stack([probabilities[sid] for sid in roster]).astype(np.float64)
        families = [_model_family(members[sid]["model"]) for sid in roster]
        feature_members = [{"candidate_id": sid, "family": fam}
                           for sid, fam in zip(roster, families)]
        features = make_features(tensor, feature_members, LABELS, router_stats)
        from catboost import CatBoostClassifier
        router = CatBoostClassifier(); router.load_model(str(router_path))
        reliability = router.predict_proba(features)[:, 1].reshape(len(v0), len(roster)).T
        family_probs = e13.family_balanced_probabilities(tensor, families)
        _, assignment_scores = family_max_scores(tensor.argmax(2), reliability,
                                                 family_probs, families)
        pred = route_family_max(tensor.argmax(2), reliability, family_probs, families)
        if count == 15:
            frozen = np.load(ROOT / "artifacts/analysis/e28_fixed_roster/fixed_roster_scores.npz",
                             allow_pickle=True)
            if (frozen["row_ids"].astype(str).tolist() != ids.tolist()
                    or not np.array_equal(pred, frozen["base_predictions"])
                    or not np.allclose(assignment_scores, frozen["assignment_scores"],
                                       rtol=0, atol=1e-10)):
                raise AssertionError("15-checkpoint E13-B deployment parity failed")
        elif count == 17 and E30_FULL.is_file():
            saved = pd.read_parquet(E30_FULL)
            saved_pred = pd.Categorical(saved["E30-B"].astype(str), categories=LABELS).codes
            if saved.id.astype(str).tolist() != ids.tolist() or not np.array_equal(pred, saved_pred):
                raise AssertionError("17-checkpoint E30-B deployment parity failed")
        elif count == 40 and E31_FULL.is_file():
            saved = pd.read_parquet(E31_FULL)
            saved_pred = pd.Categorical(saved["E31-40"].astype(str), categories=LABELS).codes
            if saved.id.astype(str).tolist() != ids.tolist() or not np.array_equal(pred, saved_pred):
                raise AssertionError("40-checkpoint E31 deployment parity failed")
        score_matrices[count] = assignment_scores
        full_predictions[count] = pred
        router_info[str(count)] = {"roster": roster, "router": str(router_path.relative_to(ROOT)),
            "router_sha256": _sha(router_path), "router_source": router_source,
            "reliability_statistics_source": "full V0 checkpoint predictions and labels",
            "checkpoint_prediction_cache_sha256": {
                sid: members[sid]["cache_sha256"] for sid in roster}}

    if pending and not allow_partial:
        raise RuntimeError(f"Deployment router/roster not yet available for: {pending}")
    results = {str(k): {} for k in score_matrices}
    prediction_blocks = []
    for scenario_id, frame in scenario_frames:
        rowix = np.asarray([id_to_row[x] for x in frame.original_id.astype(str)], dtype=np.int64)
        y = truth[rowix]
        groups_by_method = {}
        for method in METHODS[1:]:
            groups_by_method[method] = _groups_for_scenario(method, frame, rowix,
                embeddings, f_a, f_c, d_model, d_recon, d_semantic)
            groups = groups_by_method[method]
            if len(groups) != len(frame) or pd.Series(groups).value_counts().max() > len(LABELS):
                raise ValueError(f"{method} produced invalid groups for {scenario_id}")
        for count, score in score_matrices.items():
            independent = full_predictions[count][rowix]
            roster_res = results[str(count)].setdefault(scenario_id, {})
            variants = {"Independent": independent}
            for method in METHODS[1:]:
                variants[method] = hungarian(score[rowix], groups_by_method[method],
                                               frame.scenario_row_id.astype(str).to_numpy())
            for method, pred in variants.items():
                roster_res[method] = metric(y, pred)
                prediction_blocks.append(pd.DataFrame({"scenario_id": scenario_id,
                    "scenario_row_id": frame.scenario_row_id.astype(str),
                    "original_id": frame.original_id.astype(str),
                    "checkpoint_count": count, "configuration": method,
                    "truth": np.asarray(LABELS, dtype=object)[y],
                    "prediction": np.asarray(LABELS, dtype=object)[pred],
                    "predicted_group": (frame.scenario_row_id.astype(str).to_numpy()
                        if method == "Independent" else groups_by_method[method])}))

    summary = {}
    for count, by_scenario in results.items():
        summary[count] = {}
        for method in METHODS:
            vals = np.asarray([by_scenario[sid][method]["balanced_accuracy"]
                               for sid, _ in scenario_frames])
            summary[count][method] = {"mean_balanced_accuracy": float(vals.mean()),
                "scenario_count": int(len(vals)), "scenario_balanced_accuracy": vals.tolist()}

    np.savez_compressed(OUT / "predictions_and_scores.npz",
        **{f"score_{n}": v for n, v in score_matrices.items()},
        **{f"independent_{n}": v for n, v in full_predictions.items()}, row_ids=ids,
        classes=np.asarray(LABELS))
    pd.concat(prediction_blocks, ignore_index=True).to_parquet(OUT / "scenario_predictions.parquet", index=False)
    result = {"experiment": "E31 fixed-roster scenario evaluation", "checkpoint_counts": sorted(map(int, score_matrices)),
        "methods": list(METHODS), "scenario_ids": [sid for sid, _ in scenario_frames],
        "scenario_count": len(scenario_frames), "rows_per_scenario": 500,
        "summary": summary, "routers": router_info, "pending_checkpoint_counts": pending,
        "source_sha256": {"v0_group": _sha(V0_GROUP)},
        "note": "Mean of balanced accuracy across the identical 50 v0_group scenarios. Groups are rebuilt separately per scenario with frozen E19-F-A, E19-F-C and E19-D implementations. Scenarios reuse V0 rows and are not independent validation."}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

