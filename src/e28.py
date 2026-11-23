"""E28: frozen E13-B v037 scores with external prompt grouping.

All grouping scorers are fitted on the external V2.14 artifacts produced by
E27. V0 labels are read only by the final scoring/reporting functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import e13_reliability_router as e13
from . import e19b_evaluate, e19b_group, e19f, e20, e27
from .ensemble_catalog import _load_inputs, cascades
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e28"
E13_RUN = ROOT / "artifacts/analysis/e13/20261120T093318893046Z"
E13_PREDICTIONS = E13_RUN / "predictions.parquet"
E13_RESULTS = E13_RUN / "results.json"
E13_MODELS = E13_RUN
GROUPERS = ("E19-F-C", "E19-F-A", "E19-D")
SCENARIO_SEED = 2026


def family_max_label_scores(hard, reliability, families):
    """Return [rows, labels] family-max reliability scores, -inf if unsupported."""
    hard = np.asarray(hard, dtype=np.int64)
    reliability = np.asarray(reliability, dtype=np.float64)
    families = np.asarray(families, dtype=str)
    if hard.ndim != 2 or hard.shape != reliability.shape or len(families) != hard.shape[0]:
        raise ValueError("E28 hard predictions, reliability and family roster do not align")
    if not np.isfinite(reliability).all() or ((hard < 0) | (hard >= len(LABELS))).any():
        raise ValueError("E28 routing evidence contains invalid values")
    rows = hard.shape[1]
    result = np.full((rows, len(LABELS)), -np.inf, dtype=np.float64)
    for row in range(rows):
        for label in range(len(LABELS)):
            voters = np.flatnonzero(hard[:, row] == label)
            if not len(voters):
                continue
            family_max = {}
            for cp in voters:
                family_max[families[cp]] = max(
                    family_max.get(families[cp], -np.inf), reliability[cp, row])
            result[row, label] = np.mean(list(family_max.values()))
    return result


def finite_assignment_scores(label_scores, family_probabilities):
    """Replace unsupported labels with finite, lower-ranked probability ties."""
    scores = np.asarray(label_scores, dtype=np.float64).copy()
    probs = np.asarray(family_probabilities, dtype=np.float64)
    if scores.shape != probs.shape or scores.ndim != 2 or scores.shape[1] != len(LABELS):
        raise ValueError("E28 label-score and family-probability matrices do not align")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("E28 family probabilities must be finite and nonnegative")
    for row in range(len(scores)):
        supported = np.isfinite(scores[row])
        if not supported.any():
            raise ValueError(f"E28 row {row} has no checkpoint-supported label")
        floor = float(scores[row, supported].min()) - 1.0
        p = probs[row] / max(float(probs[row].max()), 1e-12)
        # Keep the router's reliability values primary while using family
        # probabilities as a low-order tie-break among supported labels.
        scores[row, supported] += p[supported] * 1e-12
        # The larger fallback gap keeps unsupported labels below every vote.
        scores[row, ~supported] = floor + p[~supported] * 1e-9
    if not np.isfinite(scores).all():
        raise ValueError("E28 Hungarian score matrix must be finite")
    return scores


def route_family_max(hard, reliability, family_probabilities, families):
    """Expose score evidence while retaining v037's independent decision rule."""
    scores = family_max_label_scores(hard, reliability, families)
    pred, chosen = e13.route_predictions(
        "family_max", hard, reliability, family_probabilities, families)
    supported = np.isfinite(scores)
    safe = finite_assignment_scores(scores, family_probabilities)
    if not np.array_equal(pred, safe.argmax(axis=1)):
        # v037 resolves reliability ties with the family-balanced probabilities,
        # then canonical order. This explicit implementation guards parity.
        for row in range(len(pred)):
            best = np.max(scores[row, supported[row]])
            tied = np.flatnonzero(np.isclose(scores[row], best, rtol=0, atol=1e-12))
            safe_tie = min(tied, key=lambda label: (-family_probabilities[row, label], label))
            if pred[row] != safe_tie:
                raise AssertionError("E28 family-max route diverges from original v037 routing")
    return pred, chosen, scores, safe


def _read_e13_oof(root=ROOT):
    """Recreate E13-B fold-held-out scores from its frozen router models/caches."""
    root = Path(root)
    v0 = cascades._v0(root)
    baseline = pd.read_parquet(E13_PREDICTIONS)
    if baseline.id.astype(str).tolist() != v0.id.astype(str).tolist():
        raise ValueError("E13-B saved predictions are not aligned with the current V0 rows")
    if baseline.y_true.astype(str).tolist() != v0.model.astype(str).tolist():
        raise ValueError("E13-B saved truth labels differ from V0")
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any():
        raise ValueError("V0 has labels outside the canonical ordering")
    _, members, probs, _, snapshot = _load_inputs(root, v0, version="trust-list-2-v037")
    result_doc = json.loads(E13_RESULTS.read_text())
    result = next(item for item in result_doc["results"] if item["ensemble"] == "E13-B")
    deployment_path = root / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
    if not deployment_path.is_file():
        raise FileNotFoundError("Frozen E13-B v037 Kaggle manifest is unavailable")
    deployment = json.loads(deployment_path.read_text())
    deployment_roster = list(deployment["deployment"]["checkpoint_ids"])
    if len(deployment_roster) != 15 or set(deployment_roster) != {
            member["candidate_id"] for member in deployment["members"]}:
        raise ValueError("Packaged E13-B v037 deployment roster is not the expected 15 checkpoints")
    id_to_row = {str(value): i for i, value in enumerate(v0.id)}
    predictions = np.empty(len(v0), dtype=np.int64)
    score_matrix = np.full((len(v0), len(LABELS)), np.nan, dtype=np.float64)
    family_probs_full = np.full_like(score_matrix, np.nan)
    fold_assignment = np.full(len(v0), -1, dtype=np.int8)
    roster_union = set()
    fold_rosters = []
    for fold in result["folds"]:
        number = int(fold["fold"])
        fit = np.asarray([id_to_row[str(x)] for x in fold["fit_ids"]], dtype=np.int64)
        held = np.asarray([id_to_row[str(x)] for x in fold["held_ids"]], dtype=np.int64)
        if np.intersect1d(fit, held).size or (fold_assignment[held] >= 0).any():
            raise ValueError(f"E13-B fold {number} overlaps or repeats held rows")
        cp_ids = list(map(str, fold["checkpoint_ids"]))
        fold_rosters.append({"fold": number, "checkpoint_ids": cp_ids})
        roster_union.update(cp_ids)
        missing = set(cp_ids) - set(probs)
        if missing:
            raise FileNotFoundError(f"E13-B v037 probability caches missing: {sorted(missing)}")
        model_path = Path(fold["model_path"])
        if not model_path.is_absolute():
            model_path = root / model_path
        if not model_path.is_file():
            model_path = E13_MODELS / model_path.name
        if not model_path.is_file():
            raise FileNotFoundError(f"E13-B v037 CatBoost router missing: {model_path}")
        family_names = [e13._model_family(members[sid]["model"]) for sid in cp_ids]
        tensor = np.stack([probs[sid] for sid in cp_ids]).astype(np.float64)
        features, _ = e13.build_feature_frame(tensor, cp_ids, family_names,
            truth[fit], fit, held)
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        reliability = model.predict_proba(e13._cat_frame(features))[:, 1].reshape(
            len(held), len(cp_ids)).T
        hard = tensor[:, held].argmax(axis=2)
        family_probs = e13.family_balanced_probabilities(tensor[:, held], family_names)
        pred, _, raw_scores, finite_scores = route_family_max(
            hard, reliability, family_probs, family_names)
        predictions[held] = pred
        score_matrix[held] = finite_scores
        family_probs_full[held] = family_probs
        fold_assignment[held] = number
    if (fold_assignment < 0).any() or not np.isfinite(score_matrix).all():
        raise ValueError("E13-B OOF predictions do not cover every V0 row exactly once")
    saved_pred = pd.Categorical(baseline["E13-B"].astype(str), categories=LABELS).codes
    parity = int(np.count_nonzero(predictions != saved_pred))
    if parity:
        wrong = np.flatnonzero(predictions != saved_pred)[:10]
        raise AssertionError(f"E13-B family-max parity failed on {parity} rows: {wrong.tolist()}")
    return {"v0": v0, "truth": truth, "row_ids": v0.id.astype(str).to_numpy(),
        "prompt_ids": v0.prompt_id.astype(str).to_numpy(), "scores": score_matrix,
        "family_probabilities": family_probs_full, "predictions": predictions,
        "folds": fold_assignment, "roster_union": sorted(roster_union,
            key=lambda sid: int(sid[1:])), "deployment_roster": deployment_roster,
        "fold_rosters": fold_rosters, "trust_list_version": snapshot["version"],
        "independent_parity_rows": parity}


def _metric(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def _comparison(y, baseline, pred):
    return {"corrected": int(((baseline != y) & (pred == y)).sum()),
            "introduced": int(((baseline == y) & (pred != y)).sum()),
            "changed": int((baseline != pred).sum())}


def _grouping_metrics(true_groups, predicted_groups):
    return e19b_evaluate.grouping_metrics(np.asarray(true_groups, dtype=str),
                                          np.asarray(predicted_groups, dtype=str))


def _external_f_models(root=ROOT, out=OUT):
    """Fit/freeze E19-F-C on external E27 train+validation rows only."""
    root, out = Path(root), Path(out)
    external = pd.read_parquet(root / "artifacts/analysis/e27/external_v214_canonical.parquet")
    embeddings = np.load(root / "artifacts/analysis/e27/external_embeddings.npy", mmap_mode="r")
    metadata = json.loads((root / "artifacts/analysis/e27/external_embeddings.json").read_text())
    ids = external.id.astype(str).to_numpy()
    if len(external) != len(embeddings) or metadata.get("ids_sha256") != hashlib.sha256(
            "\n".join(ids).encode()).hexdigest():
        raise ValueError("E27 external embeddings do not align with canonical V2.14 rows")
    fit = np.flatnonzero(external.split.astype(str).isin(("train", "validation")))
    fit_texts = external.text.fillna("").astype(str).to_numpy()[fit]
    fit_ids = ids[fit]
    fit_embeddings = np.asarray(embeddings)[fit]
    fit_groups = external.prompt_canon.astype(str).to_numpy()[fit]
    local_rows = np.arange(len(fit), dtype=np.int64)
    vectorizers = e19f._fit_lexical(fit_texts, local_rows)
    lexical = e19f._transform_lexical(fit_texts, vectorizers)
    _, ordered, pairs, features = e27._candidate_bundle(
        local_rows, fit_ids, fit_texts, fit_embeddings, vectorizers, lexical=lexical)
    y = (fit_groups[ordered[pairs[:, 0]]] == fit_groups[ordered[pairs[:, 1]]]).astype(np.int8)
    model = e19f.fit_similarity_model(ordered, pairs, lexical, fit_embeddings, fit_groups,
                                      c=1.0, seed=e19f.SEED)
    # Fit output is deliberately external-only; serialized model is usable for
    # V0/test grouping without any threshold or label selection at deployment.
    bundle = {"method": "E19-F-C", "threshold": 0.45, "neighbors": 40,
        "max_cluster_size": 18, "pair_model": model,
        "vectorizers": vectorizers, "semantic_model": e19f.MODEL_ID,
        "semantic_revision": e19f.MODEL_REVISION,
        "external_training_rows": int(len(fit)), "external_training_pairs": int(len(pairs)),
        "external_positive_pairs": int(y.sum()),
        "threshold_source": "E27 external validation; frozen, no V0 labels"}
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "e19_f_c_external_grouper.joblib", compress=3)
    return bundle


def _external_d_model(root=ROOT, out=OUT):
    """Fit/freeze E27's E19-D scorer using its external 1% split only."""
    root, out = Path(root), Path(out)
    folder = root / "artifacts/analysis/e27/e19-d"
    frame = pd.read_parquet(root / "artifacts/analysis/e27/external_v214_reconstruction_rotation_01pct.parquet")
    recon_frame = pd.read_parquet(folder / "reconstructed_external.parquet")
    recon_pos = {sid: i for i, sid in enumerate(recon_frame.id.astype(str))}
    order = np.asarray([recon_pos[sid] for sid in frame.id.astype(str)], dtype=np.int64)
    recon = recon_frame.reconstructed_prompt.astype(str).to_numpy()[order]
    topics = recon_frame.topic.fillna("").astype(str).to_numpy()[order]
    details = [e27._as_list(x) for x in recon_frame.distinctive_details.to_numpy()[order]]
    constraints = [e27._as_list(x) for x in recon_frame.constraints.to_numpy()[order]]
    meta = json.loads((folder / "semantic_embeddings.json").read_text())
    semantic_all = np.load(folder / "semantic_embeddings.npy", mmap_mode="r")
    if meta.get("ids_sha256") != hashlib.sha256(
            "\n".join(recon_frame.id.astype(str)).encode()).hexdigest():
        raise ValueError("E27 E19-D external embeddings are stale or misaligned")
    semantic = np.asarray(semantic_all)[order]
    response = frame.text.fillna("").astype(str).to_numpy()
    groups = frame.view_group.astype(str).to_numpy()
    fit = np.flatnonzero(frame.split.astype(str).isin(("train", "validation")))
    spaces = e19b_group.fit_similarity_spaces(recon, response, fit)
    fit_rows, fit_pairs, features = e27._reconstruction_features(
        fit, spaces, recon, response, semantic, topics, details, constraints)
    targets = (groups[fit_rows[fit_pairs[:, 0]]] == groups[fit_rows[fit_pairs[:, 1]]]).astype(np.int8)
    scorer = e19b_group.fit_pair_scorer(features, targets, c=0.1, seed=e27.SEED)
    vectorizers = {name: spaces[name + "_vectorizer"] for name in (
        "reconstruction_char", "reconstruction_word", "response_char", "response_word")}
    bundle = {"method": "E19-D", "threshold": 0.35, "max_cluster_size": 18,
        "neighbors": 40, "pair_scorer": scorer, "vectorizers": vectorizers,
        "semantic_model": "google/embeddinggemma-2",
        "semantic_revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "external_training_rows": int(len(fit)), "external_training_pairs": int(len(fit_pairs)),
        "external_positive_pairs": int(targets.sum()),
        "threshold_source": "E27 external validation; frozen, no V0 labels"}
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "e19_d_external_grouper.joblib", compress=3)
    return bundle


def _fit_f_c_if_needed(root=ROOT, out=OUT):
    path = Path(out) / "e19_f_c_external_grouper.joblib"
    return joblib.load(path) if path.is_file() else _external_f_models(root, out)


def _fit_d_if_needed(root=ROOT, out=OUT):
    path = Path(out) / "e19_d_external_grouper.joblib"
    return joblib.load(path) if path.is_file() else _external_d_model(root, out)


def _predict_f_groups(method, ids, texts, embeddings, bundle):
    ids = np.asarray(ids, dtype=str)
    texts = np.asarray(texts, dtype=str)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not (len(ids) == len(texts) == len(embeddings)) or len(set(ids.tolist())) != len(ids):
        raise ValueError("E28 E19-F grouping inputs are misaligned or IDs are duplicated")
    lexical = e19f._transform_lexical(texts, bundle["vectorizers"])
    _, ordered, pairs, features = e27._candidate_bundle(
        np.arange(len(ids)), ids, texts, embeddings, bundle["vectorizers"], lexical=lexical)
    score = (features[:, 0] if method == "E19-F-A" else
             e19f.score_similarity_model(bundle["pair_model"], features))
    groups, _ = e19f._cluster_fc(ids[ordered], pairs, score, bundle["threshold"])
    result = np.empty(len(ids), dtype=object)
    result[ordered] = groups
    return result


def _predict_d_groups(ids, texts, recon_frame, semantic, bundle):
    ids = np.asarray(ids, dtype=str)
    recon_frame = recon_frame.copy()
    if recon_frame.id.astype(str).tolist() != ids.tolist() or len(semantic) != len(ids):
        raise ValueError("E28 E19-D reconstructed prompt/embedding row order mismatch")
    recon = recon_frame.reconstructed_prompt.astype(str).to_numpy()
    topics = recon_frame.topic.fillna("").astype(str).to_numpy()
    details = [e27._as_list(x) for x in recon_frame.distinctive_details.to_numpy()]
    constraints = [e27._as_list(x) for x in recon_frame.constraints.to_numpy()]
    spaces = e19b_group.transform_similarity_spaces(recon, texts, bundle["vectorizers"])
    rows, pairs, features = e27._reconstruction_features(np.arange(len(ids)), spaces,
        recon, texts, semantic, topics, details, constraints)
    pred, _, _ = e27._pair_cluster(ids[rows], pairs, features,
        bundle["pair_scorer"], bundle["threshold"])
    result = np.empty(len(ids), dtype=object)
    result[rows] = pred
    return result


def _group_and_score(method, rows, context, v0_embeddings, f_c, d_model):
    v0 = context["v0"]
    ids = context["row_ids"][rows]
    texts = v0.text.fillna("").astype(str).to_numpy()[rows]
    if method.startswith("E19-F"):
        bundle = f_c if method == "E19-F-C" else {
            "threshold": 0.85, "neighbors": 40,
            "vectorizers": f_c["vectorizers"]}
        groups = _predict_f_groups(method, ids, texts, v0_embeddings[rows], bundle)
    else:
        rec_all = pd.read_parquet(ROOT / "artifacts/analysis/e19d/reconstructed_prompts.parquet")
        if rec_all.id.astype(str).tolist() != context["row_ids"].tolist():
            raise ValueError("E19-D V0 reconstruction cache does not align with V0 rows")
        sem_all = np.load(ROOT / "artifacts/analysis/e19d/semantic_embeddings.npy", mmap_mode="r")
        groups = _predict_d_groups(ids, texts, rec_all.iloc[rows].reset_index(drop=True),
                                   np.asarray(sem_all)[rows], d_model)
    return groups


def evaluate(root=ROOT, output_dir=OUT, *, include_d=True, scenarios=50):
    started = time.perf_counter()
    root, output_dir = Path(root), Path(output_dir)
    context = _read_e13_oof(root)
    v0 = context["v0"]
    y, base, scores = context["truth"], context["predictions"], context["scores"]
    ids = context["row_ids"]
    # Reuse E19-F's exact response embeddings; E27 uses this pinned model/revision.
    emb = np.load(root / "artifacts/analysis/e19f/response_embeddings.npy", mmap_mode="r")
    if len(emb) != len(v0):
        raise ValueError("E19-F cached V0 response embeddings have stale row count")
    f_c = _fit_f_c_if_needed(root, output_dir)
    d_model = _fit_d_if_needed(root, output_dir) if include_d else None
    methods = ["E19-F-C", "E19-F-A"] + (["E19-D"] if include_d else [])
    all_groups = {}
    result = {"experiment": "E28", "trust_list_version": context["trust_list_version"],
        "checkpoint_count": 15, "checkpoint_roster_union": context["roster_union"],
        "frozen_deployment_roster": context["deployment_roster"],
        "oof_fold_rosters": context["fold_rosters"],
        "independent_parity_rows": context["independent_parity_rows"],
        "baseline": {**_metric(y, base), "protocol": "E13-B saved five-fold prompt-group OOF predictions"},
        "groupers": {}, "scenarios": {}, "runtime_seconds": {}}
    for method in methods:
        method_started = time.perf_counter()
        groups = _group_and_score(method, np.arange(len(v0)), context, emb, f_c, d_model)
        if len(groups) != len(ids) or pd.Series(groups).value_counts().max() > len(LABELS):
            raise ValueError(f"{method} produced misaligned or oversized V0 groups")
        pred = e20.hungarian(scores, groups, ids)
        result["groupers"][method] = {
            "metrics": _metric(y, pred), "comparison_vs_E28_A": _comparison(y, base, pred),
            "grouping_metrics": _grouping_metrics(context["prompt_ids"], groups),
            "group_size_distribution": {str(k): int(v) for k, v in
                pd.Series(groups).value_counts().value_counts().sort_index().items()},
            "parameters": f_c["threshold"] if method == "E19-F-C" else
                (0.85 if method == "E19-F-A" else d_model["threshold"])}
        all_groups[method] = groups
        pd.DataFrame({"id": ids, "predicted_group": groups, "prediction": np.asarray(LABELS)[pred],
                      "fold": context["folds"]}).to_parquet(
            output_dir / f"predictions_{method.replace('-', '_')}_full.parquet", index=False)
        result["runtime_seconds"][method + "_full"] = time.perf_counter() - method_started
    samples = e27.sample_incomplete_scenarios(v0, scenarios=scenarios, seed=SCENARIO_SEED)
    for method in methods:
        method_started = time.perf_counter()
        method_results = {}
        for number, rows in samples.items():
            tic = time.perf_counter()
            groups = _group_and_score(method, rows, context, emb, f_c, d_model)
            pred = e20.hungarian(scores[rows], groups, ids[rows])
            grouping = _grouping_metrics(context["prompt_ids"][rows], groups)
            method_results[str(number)] = {"rows": int(len(rows)),
                "metrics": _metric(y[rows], pred),
                "baseline_metrics": _metric(y[rows], base[rows]),
                "comparison_vs_E28_A": _comparison(y[rows], base[rows], pred),
                "grouping_metrics": grouping,
                "group_size_distribution": {str(k): int(v) for k, v in
                    pd.Series(groups).value_counts().value_counts().sort_index().items()},
                "runtime_seconds": time.perf_counter() - tic}
            if int(number) % 5 == 0 or int(number) == len(samples):
                result["runtime_seconds"][method + "_scenarios_elapsed"] = time.perf_counter() - method_started
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "local_results_partial.json").write_text(
                    json.dumps(result, indent=2) + "\n")
                print(f"E28 {method}: scenario {number}/{len(samples)}; "
                      f"BA={method_results[str(number)]['metrics']['balanced_accuracy']:.4%}",
                      flush=True)
        result["scenarios"][method] = method_results
    result["runtime_seconds"]["total"] = time.perf_counter() - started
    result["baseline_balanced_accuracy_source"] = "E13-B v037 saved OOF: 64.6465%"
    result["v0_usage"] = "Evaluation only: labels and prompt_id used for metrics/scenario sampling; no fit, threshold tuning or model selection."
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "e28_oof_scores.npz", row_ids=ids, y=y,
        labels=np.asarray(LABELS), scores=scores, family_probabilities=context["family_probabilities"],
        independent_predictions=base, folds=context["folds"])
    (output_dir / "local_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-external-groupers", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-e19-d", action="store_true")
    args = parser.parse_args(argv)
    if args.fit_external_groupers:
        fc = _fit_f_c_if_needed()
        d = _external_d_model()
        print(json.dumps({"E19-F-C": {k: v for k, v in fc.items() if k != "pair_model" and k != "vectorizers"},
                          "E19-D": {k: v for k, v in d.items() if k != "pair_scorer" and k != "vectorizers"}},
                         indent=2, default=str))
    if args.evaluate:
        print(json.dumps(evaluate(include_d=not args.no_e19_d), indent=2))
    if not args.fit_external_groupers and not args.evaluate:
        parser.error("select --fit-external-groupers and/or --evaluate")


if __name__ == "__main__":
    main()
