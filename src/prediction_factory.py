"""Deterministic search and reliability lookup over saved model predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from .ensemble_factory import fit_weights


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _resolve_artifact_path(value, root):
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(root)
    direct = root / path
    if direct.exists():
        return direct
    return root / "artifacts" / path


def _read_candidates(artifact_dir, *, min_validation_rows=1, candidate_ids=None, deduplicate=True):
    """Read validation runs, optionally restricting to an explicit champion roster."""
    artifact_dir = Path(artifact_dir)
    root = Path.cwd()
    candidate_ids = set(map(str, candidate_ids)) if candidate_ids is not None else None
    by_key = {}
    problems = []
    for manifest_path in sorted(artifact_dir.glob("*_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
            model = str(manifest.get("model", ""))
            if not model or "ensemble" in model.lower() or "+" in model:
                continue
            split_name = str(manifest.get("split", ""))
            # Older continuation manifests use labels such as `v0_validation`.
            # They are safe to consider because _align_candidates later requires
            # an exact match with the frozen reference IDs and labels.
            if split_name != "validation" and not split_name.endswith("_validation"):
                continue
            prediction_value = manifest.get("prediction_path")
            if not prediction_value:
                continue
            prediction_path = _resolve_artifact_path(prediction_value, root)
            if not prediction_path.is_file():
                continue
            frame = pd.read_parquet(prediction_path)
            if len(frame) < min_validation_rows:
                continue
            class_columns = [column for column in frame if column.startswith("prob::")]
            classes = sorted(column.removeprefix("prob::") for column in class_columns)
            if not classes or not {"id", "y_true", "prompt_id"}.issubset(frame.columns):
                continue
            probability_columns = [f"prob::{label}" for label in classes]
            probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
            if (probabilities.shape != (len(frame), len(classes)) or
                    not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
                    not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
                problems.append(f"invalid probabilities: {prediction_path}")
                continue
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            ordered = frame[["id", "prompt_id", "y_true"]].copy()
            ordered["id"] = ordered.id.astype(str)
            ordered["prompt_id"] = ordered.prompt_id.astype(str)
            ordered["y_true"] = ordered.y_true.astype(str)
            if ordered.id.duplicated().any():
                problems.append(f"duplicate validation IDs: {prediction_path}")
                continue
            order = np.argsort(ordered.id.to_numpy(), kind="stable")
            ordered = ordered.iloc[order].reset_index(drop=True)
            probabilities = probabilities[order]
            recorded = manifest.get("metrics") or {}
            score = float(recorded.get("balanced_accuracy", manifest.get("selected_validation_accuracy", 0.0)) or 0.0)
            # A special Qwen epoch-selection export is a checkpoint variant of Qwen3-0.6B.
            if model == "qwen06_best_epoch03":
                model = "qwen06"
            dataset_version = str(manifest.get("dataset", "unknown"))
            candidate_id = str(manifest.get("experiment_id", manifest_path.stem.removesuffix("_manifest")))
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            key = (model, dataset_version) if deduplicate and candidate_ids is None else candidate_id
            candidate = {
                "candidate_id": candidate_id,
                "model": model,
                "dataset": dataset_version,
                "manifest_path": str(manifest_path.resolve()),
                "checkpoint_path": manifest.get("checkpoint_path") or manifest.get("checkpoint"),
                "model_manifest": manifest,
                "validation_score": score,
                "classes": classes,
                "frame": ordered,
                "probabilities": probabilities,
            }
            prior = by_key.get(key)
            if prior is None or (score, candidate_id) > (prior["validation_score"], prior["candidate_id"]):
                by_key[key] = candidate
        except Exception as exc:
            problems.append(f"{manifest_path.name}: {type(exc).__name__}: {exc}")
    return sorted(by_key.values(), key=lambda item: (item["model"], item["dataset"], item["candidate_id"])), problems


def _reference_candidate(candidates, reference_dataset="v0", preferred_ids=None):
    preferred = set(preferred_ids or [])
    pool = [item for item in candidates if item["candidate_id"] in preferred]
    if pool:
        return next((item for item in pool if item["dataset"] == reference_dataset), pool[0])
    pool = [item for item in candidates if item["dataset"] == reference_dataset] or candidates
    if not pool:
        raise ValueError("no usable validation model predictions found")
    signatures = [_sha({"classes": item["classes"],
                        "rows": item["frame"][["id", "prompt_id", "y_true"]].values.tolist()})
                  for item in pool]
    counts = {key: signatures.count(key) for key in set(signatures)}
    best = min(range(len(pool)), key=lambda index: (-counts[signatures[index]], pool[index]["candidate_id"]))
    return pool[best]


def _align_candidates(candidates, *, reference_dataset="v0", reference_frame=None):
    if not candidates:
        raise ValueError("no usable validation model predictions found")
    classes = candidates[0]["classes"]
    references = [item for item in candidates if item["dataset"] == reference_dataset]
    if reference_frame is not None:
        reference = reference_frame
    elif references:
        reference = references[0]["frame"]
    else:
        reference = candidates[0]["frame"]
    reference_ids = reference.id.to_list()
    reference_set = set(reference_ids)
    y = reference.y_true.to_numpy(dtype=str)
    groups = reference.prompt_id.to_numpy(dtype=str)
    aligned = []
    excluded = []
    for candidate in candidates:
        frame = candidate["frame"]
        if candidate["classes"] != classes:
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "validation classes differ from frozen reference"})
            continue
        if set(frame.id) != reference_set:
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "validation IDs differ from frozen reference"})
            continue
        frame = frame.set_index("id").loc[reference_ids].reset_index()
        if not np.array_equal(frame.y_true.astype(str).to_numpy(), y):
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "validation labels differ from frozen reference"})
            continue
        if not np.array_equal(frame.prompt_id.astype(str).to_numpy(), groups):
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "validation prompt groups differ from frozen reference"})
            continue
        order = frame.id.map({value: index for index, value in enumerate(candidate["frame"].id)}).to_numpy(dtype=int)
        aligned_candidate = dict(candidate)
        aligned_candidate["probabilities"] = candidate["probabilities"][order]
        aligned.append(aligned_candidate)
    if len(aligned) < 1:
        raise ValueError("no candidates align to the frozen validation set")
    if len(set(y)) != len(classes) or not all(np.sum(y == label) > 0 for label in classes):
        raise ValueError("frozen validation set does not contain every model class")
    return classes, y, groups, aligned, excluded, reference_ids


def _reliability_table(candidates, training_predictions, classes, *, minimum_support=10,
                       prior_alpha=1.0, prior_beta=1.0):
    rows = []
    for candidate in candidates:
        training = training_predictions[candidate["candidate_id"]]
        y_train = np.asarray(training["labels"], dtype=str)
        predicted = np.asarray(classes)[np.asarray(training["probabilities"]).argmax(axis=1)]
        for label in classes:
            mask = predicted == label
            predicted_count = int(mask.sum())
            true_positive = int(np.sum(mask & (y_train == label)))
            false_positive = predicted_count - true_positive
            actual_count = int(np.sum(y_train == label))
            false_negative = actual_count - true_positive
            true_negative = len(y_train) - true_positive - false_positive - false_negative
            precision = true_positive / predicted_count if predicted_count else None
            trust = ((true_positive + prior_alpha) /
                     (predicted_count + prior_alpha + prior_beta)) if predicted_count else None
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "model": candidate["model"],
                "training_dataset": candidate["dataset"],
                "predicted_nlm": label,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "predicted_count": predicted_count,
                "actual_count": actual_count,
                "training_rows": int(len(y_train)),
                "precision": precision,
                "smoothed_trust": trust,
                "eligible": predicted_count >= minimum_support,
            })
    return rows


def _lookup_predictions(candidates, classes, table_rows, *, minimum_support=10):
    reliability = {(row["candidate_id"], row["predicted_nlm"]): row
                   for row in table_rows}
    outputs = []
    used_candidate_ids = set()
    for row_index in range(len(candidates[0]["probabilities"])):
        options = []
        for candidate in candidates:
            probabilities = candidate["probabilities"][row_index]
            class_id = int(np.argmax(probabilities))
            label = classes[class_id]
            evidence = reliability[(candidate["candidate_id"], label)]
            if evidence["predicted_count"] < minimum_support:
                continue
            margin = float(np.partition(probabilities, -2)[-1] - np.partition(probabilities, -2)[-2])
            options.append((float(evidence["smoothed_trust"]), float(probabilities[class_id]),
                            margin, candidate["candidate_id"], label, candidate))
        if options:
            options.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
            trust, confidence, margin, _, label, winner = options[0]
            fallback = False
        else:
            winner = max(candidates, key=lambda item: (
                item.get("training_balanced_accuracy", 0.0), item["candidate_id"]))
            probabilities = winner["probabilities"][row_index]
            label = classes[int(np.argmax(probabilities))]
            trust, confidence = None, float(np.max(probabilities))
            margin = None
            fallback = True
        used_candidate_ids.add(winner["candidate_id"])
        outputs.append({"predicted_nlm": label, "candidate_id": winner["candidate_id"],
                        "confidence": confidence, "smoothed_trust": trust,
                        "confidence_margin": margin, "fallback": fallback})
    predicted = np.asarray([item["predicted_nlm"] for item in outputs])
    return outputs, predicted, used_candidate_ids


def _probability_metrics(probabilities, y_true, classes):
    predicted = np.asarray(classes)[np.asarray(probabilities).argmax(axis=1)]
    return {"accuracy": float(accuracy_score(y_true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
            "log_loss": float(log_loss(y_true, probabilities, labels=list(classes)))}


def _forward_select(candidates, y_true, classes):
    """Greedily add candidates by accuracy, then log loss, with stable name ties."""
    if not candidates:
        raise ValueError("forward selection needs at least one candidate")
    candidates = sorted(candidates, key=lambda item: item["candidate_id"])
    remaining = list(candidates)
    individual = []
    for candidate in candidates:
        metric = _probability_metrics(candidate["probabilities"], y_true, classes)
        individual.append({"candidate_id": candidate["candidate_id"], **metric})
    first = min(individual, key=lambda row: (-row["accuracy"], row["log_loss"], row["candidate_id"]))
    selected = [next(item for item in candidates if item["candidate_id"] == first["candidate_id"])]
    remaining = [item for item in remaining if item["candidate_id"] != first["candidate_id"]]
    current_probs = selected[0]["probabilities"].copy()
    current = _probability_metrics(current_probs, y_true, classes)
    steps = [{"step": 1, "added_candidate_id": first["candidate_id"],
              "selected_candidate_ids": [first["candidate_id"]], **current}]
    while remaining:
        trials = []
        for candidate in remaining:
            probs = np.mean(np.stack([item["probabilities"] for item in selected + [candidate]]), axis=0)
            metric = _probability_metrics(probs, y_true, classes)
            trials.append((metric, candidate, probs))
        metric, candidate, probs = min(
            trials, key=lambda trial: (-trial[0]["accuracy"], trial[0]["log_loss"],
                                       len(selected) + 1, trial[1]["candidate_id"]))
        improves = (metric["accuracy"] > current["accuracy"] or
                    (metric["accuracy"] == current["accuracy"] and
                     metric["log_loss"] < current["log_loss"] - 1e-12))
        if not improves:
            break
        selected.append(candidate)
        remaining = [item for item in remaining if item["candidate_id"] != candidate["candidate_id"]]
        current, current_probs = metric, probs
        steps.append({"step": len(selected), "added_candidate_id": candidate["candidate_id"],
                      "selected_candidate_ids": [item["candidate_id"] for item in selected], **current})
    return selected, current_probs, steps, individual


def _weighted_probabilities(candidates, weights):
    return np.einsum("m,mnc->nc", np.asarray(weights, dtype=float),
                     np.stack([item["probabilities"] for item in candidates]))


def _grouped_strategy_comparison(candidates, y_true, groups, classes, champion_probabilities=None,
                                 *, folds=5, l2=.05):
    """Out-of-fold evaluation of selection and weight fitting, grouped by prompt."""
    unique_groups = np.unique(np.asarray(groups, dtype=str))
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError("at least two prompt groups are required for grouped comparison")
    outputs = {"equal_probability_blend": np.zeros((len(y_true), len(classes)), dtype=float),
               "optimized_probability_blend": np.zeros((len(y_true), len(classes)), dtype=float),
               "all_candidate_weighted_blend": np.zeros((len(y_true), len(classes)), dtype=float)}
    if champion_probabilities is not None:
        outputs["previous_champion"] = np.asarray(champion_probabilities).copy()
    fold_records = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y_true)), y_true, groups), 1):
        fold_candidates = [{**candidate, "probabilities": candidate["probabilities"][train_idx]}
                           for candidate in candidates]
        selected, _, _, _ = _forward_select(fold_candidates, y_true[train_idx], classes)
        selected_ids = [item["candidate_id"] for item in selected]
        lookup = {item["candidate_id"]: item for item in candidates}
        selected_valid = [lookup[item] for item in selected_ids]
        equal = np.mean(np.stack([item["probabilities"][valid_idx] for item in selected_valid]), axis=0)
        weights = fit_weights([item["probabilities"] for item in selected],
                              y_true[train_idx], classes, l2=l2)
        weighted = np.einsum("m,mnc->nc", weights,
                             np.stack([item["probabilities"][valid_idx] for item in selected_valid]))
        all_weights = fit_weights([item["probabilities"] for item in fold_candidates],
                                  y_true[train_idx], classes, l2=l2)
        all_candidate_weighted = np.einsum("m,mnc->nc", all_weights,
            np.stack([item["probabilities"][valid_idx] for item in candidates]))
        outputs["equal_probability_blend"][valid_idx] = equal
        outputs["optimized_probability_blend"][valid_idx] = weighted
        outputs["all_candidate_weighted_blend"][valid_idx] = all_candidate_weighted
        fold_records.append({"fold": fold_index, "held_out_prompt_groups": int(len(set(np.asarray(groups)[valid_idx]))),
            "selected_candidate_ids": selected_ids, "optimized_weights": dict(zip(selected_ids, weights.tolist())),
            "all_candidate_weights": dict(zip([item["candidate_id"] for item in candidates], all_weights.tolist()))})
    metrics = {name: _probability_metrics(probabilities, y_true, classes)
               for name, probabilities in outputs.items()}
    return metrics, outputs, fold_records


def _training_reliability_predictions(candidates, *, data_dir="data/processed", output_dir=None):
    """Cache each candidate's predictions on the common V0 training partition."""
    from .pipeline import _load_dataset
    from .inference import (predict_catboost_stylometry, predict_svc_bundle,
                            _predict_neural)
    import joblib

    rows, _ = _load_dataset("v0", data_dir)
    training = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(rows.model.astype(str).unique().tolist())
    labels = training.model.astype(str).to_numpy()
    output_dir = Path(output_dir) if output_dir else Path("artifacts/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256("\n".join(training.id.astype(str)).encode()).hexdigest()[:12]
    results = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        cache_path = output_dir / f"{candidate_id}_train_{cache_key}.parquet"
        if cache_path.is_file():
            cached = pd.read_parquet(cache_path)
            if (cached.id.astype(str).tolist() != training.id.astype(str).tolist() or
                    cached.y_true.astype(str).tolist() != labels.tolist()):
                raise ValueError(f"cached training predictions do not align for {candidate_id}")
            probabilities = cached[[f"prob::{label}" for label in classes]].to_numpy(dtype=np.float64)
        else:
            manifest = candidate["model_manifest"]
            model_name = str(manifest.get("model", candidate["model"]))
            checkpoint = candidate.get("checkpoint_path")
            if not checkpoint:
                raise ValueError(f"candidate {candidate_id} has no checkpoint path")
            checkpoint = _resolve_artifact_path(checkpoint, Path.cwd())
            if model_name == "svc":
                probabilities, returned_classes = predict_svc_bundle(
                    joblib.load(checkpoint), training.text.astype(str).tolist())
            elif model_name == "catboost_stylometry":
                probabilities, returned_classes = predict_catboost_stylometry(
                    checkpoint, training.text.astype(str).tolist(), classes,
                    temperature=manifest.get("calibration_temperature", 1.0))
            else:
                config = manifest.get("config", {})
                configured_batch = int(manifest.get("effective_batch_size") or
                                       config.get("effective_batch_size") or
                                       config.get("batch_size") or 8)
                inference_batch = max(1, min(int(config.get("inference_batch_size", configured_batch)), 4 if model_name in {
                    "qwen4", "llama32_3b", "llama31_8b", "ministral3_8b", "smollm3_3b", "granite4_h_micro",
                    "granite4_350m", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"} else 8))
                probabilities, returned_classes = _predict_neural(
                    model_name, checkpoint, training.text.astype(str).tolist(), classes,
                    batch_size=inference_batch,
                    max_length=int(config.get("max_length", 384)),
                    temperature=float(manifest.get("calibration_temperature", 1.0) or 1.0),
                    base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision"),
                    fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
                    text_only_backbone=bool(config.get("text_only_backbone", False)),
                    quantization=config.get("quantization", manifest.get("quantization")),
                    attention_backend=config.get("attention_backend", manifest.get("attention_backend")))
            if returned_classes != classes:
                indexes = {label: index for index, label in enumerate(returned_classes)}
                probabilities = probabilities[:, [indexes[label] for label in classes]]
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if (probabilities.shape != (len(training), len(classes)) or
                    not np.isfinite(probabilities).all() or
                    not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
                raise ValueError(f"invalid V0 training predictions for {candidate_id}")
            frame = training[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
            for index, label in enumerate(classes):
                frame[f"prob::{label}"] = probabilities[:, index]
            frame.to_parquet(cache_path, index=False)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if (probabilities.shape != (len(training), len(classes)) or
                not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
                not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
            raise ValueError(f"invalid V0 training predictions for {candidate_id}")
        training_score = float(balanced_accuracy_score(
            labels, np.asarray(classes)[probabilities.argmax(axis=1)]))
        results[candidate_id] = {"labels": labels, "probabilities": probabilities,
                                 "training_rows": len(training),
                                 "training_balanced_accuracy": training_score,
                                 "prediction_path": str(cache_path.resolve())}
        candidate["training_balanced_accuracy"] = training_score
        candidate["training_prediction_path"] = str(cache_path.resolve())
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return results, labels, cache_key


def evaluate_prediction_strategies(artifact_dir="artifacts", *, output_dir=None,
                                   reference_dataset="v0", minimum_support=10,
                                   blend_batch_size=64, top_k=100, candidate_ids=None,
                                   data_dir="data/processed", previous_champion=None,
                                   group_folds=5, weight_l2=.05):
    """Select a compact equal or weighted blend from saved validation probabilities."""
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir) if output_dir else artifact_dir / "predictions"
    all_candidates, read_problems = _read_candidates(artifact_dir / "predictions", deduplicate=False)
    reference = _reference_candidate(all_candidates, reference_dataset, candidate_ids)
    reference_frame = reference["frame"]
    if candidate_ids is None:
        candidates = all_candidates
    else:
        requested_ids = set(map(str, candidate_ids))
        candidates = [item for item in all_candidates if item["candidate_id"] in requested_ids]
        missing_ids = requested_ids - {item["candidate_id"] for item in candidates}
        if missing_ids:
            raise FileNotFoundError(f"active candidate manifests or predictions are missing: {sorted(missing_ids)}")
    candidates.sort(key=lambda item: (item["candidate_id"] != reference["candidate_id"], item["candidate_id"]))
    classes, y_true, groups, candidates, excluded, reference_ids = _align_candidates(
        candidates, reference_dataset=reference_dataset, reference_frame=reference_frame)
    if candidate_ids is not None and excluded:
        raise ValueError(f"active candidates do not align to the frozen validation set: {excluded}")
    selected, equal_probabilities, selection_steps, individual_scores = _forward_select(candidates, y_true, classes)
    selected_ids = [item["candidate_id"] for item in selected]
    selected_weights = fit_weights([item["probabilities"] for item in selected], y_true, classes, l2=weight_l2)
    weighted_probabilities = _weighted_probabilities(selected, selected_weights)
    all_candidate_weights = fit_weights([item["probabilities"] for item in candidates], y_true, classes, l2=weight_l2)
    all_candidate_probabilities = _weighted_probabilities(candidates, all_candidate_weights)
    champion = previous_champion or {}
    champion_ids = [str(item.get("candidate_id") if isinstance(item, dict) else item)
                    for item in champion.get("selected_members", [])]
    if not champion_ids:
        champion_ids = list(champion.get("selected_candidate_ids") or [])
    historical = []
    if champion_ids:
        _, _, _, historical, historic_excluded, _ = _align_candidates(
            [item for item in all_candidates if item["candidate_id"] in champion_ids],
            reference_frame=candidates[0]["frame"])
        if historic_excluded:
            raise ValueError(f"previous champion does not align: {historic_excluded}")
    candidate_by_id = {item["candidate_id"]: item for item in historical}
    champion_members = [candidate_by_id[item] for item in champion_ids if item in candidate_by_id]
    if champion_ids and len(champion_members) != len(champion_ids):
        missing = sorted(set(champion_ids) - set(candidate_by_id))
        raise ValueError(f"previous champion candidates are unavailable for comparison: {missing}")
    champion_rule = champion.get("selected_strategy", "equal_probability_blend")
    if champion_rule == "previous_champion":
        legacy_weights = champion.get("selected_weights") or {}
        if legacy_weights and (set(legacy_weights) != set(champion_ids) or
                any(not np.isclose(float(value), 1.0 / len(champion_ids), atol=1e-6)
                    for value in legacy_weights.values())):
            raise ValueError("previous champion lacks its executable strategy for nonuniform weights")
        champion_rule = "equal_probability_blend"
    champion_weights = []
    if champion_members:
        if champion_rule == "equal_probability_blend":
            champion_weights = [1.0 / len(champion_members)] * len(champion_members)
        elif champion_rule == "optimized_probability_blend":
            weight_map = champion.get("selected_weights")
            if not isinstance(weight_map, dict) or set(weight_map) != set(champion_ids):
                raise ValueError("weighted champion requires a weight for each selected member")
            champion_weights = np.asarray([weight_map[item] for item in champion_ids], dtype=float)
            if (not np.isfinite(champion_weights).all() or np.any(champion_weights < 0)
                    or not np.isclose(champion_weights.sum(), 1.0, atol=1e-6)):
                raise ValueError("weighted champion has invalid or unnormalized weights")
            champion_weights = champion_weights.tolist()
        else:
            raise ValueError(f"unsupported previous champion strategy: {champion_rule}")
        champion_probabilities = _weighted_probabilities(champion_members, champion_weights)
    else:
        champion_probabilities = None
    oof_metrics, oof_probabilities, fold_records = _grouped_strategy_comparison(
        candidates, y_true, groups, classes, champion_probabilities=champion_probabilities,
        folds=group_folds, l2=weight_l2)
    sizes = {"previous_champion": len(champion_members),
             "equal_probability_blend": len(selected_ids),
             "optimized_probability_blend": len(selected_ids),
             "all_candidate_weighted_blend": len(candidates)}
    selection_names = {"previous_champion": champion.get("selected_strategy", "previous_champion"),
                       "equal_probability_blend": "equal_probability_blend",
                       "optimized_probability_blend": "optimized_probability_blend"}
    selectable_names = {"previous_champion", "equal_probability_blend", "optimized_probability_blend"}
    winner_name = min((name for name in oof_metrics if name in selectable_names), key=lambda name: (-oof_metrics[name]["accuracy"],
        oof_metrics[name]["log_loss"], sizes[name], selection_names[name], tuple(
            champion_ids if name == "previous_champion" else selected_ids)))
    final_probs = {"previous_champion": champion_probabilities,
                   "equal_probability_blend": equal_probabilities,
                   "optimized_probability_blend": weighted_probabilities}[winner_name]
    full_metrics = {name: _probability_metrics(probs, y_true, classes) for name, probs in {
        "previous_champion": champion_probabilities,
        "equal_probability_blend": equal_probabilities,
        "optimized_probability_blend": weighted_probabilities,
        "all_candidate_weighted_blend": all_candidate_probabilities}.items() if probs is not None}
    # Make final inference payload correspond to the selected strategy.
    if winner_name == "previous_champion" and champion_members:
        final_members = champion_members
        final_weights = champion_weights
    else:
        final_members = selected
        final_weights = (selected_weights.tolist() if winner_name == "optimized_probability_blend"
                         else [1.0 / len(selected)] * len(selected))
    selected_rule = (champion_rule
                     if winner_name == "previous_champion" else winner_name)
    if selected_rule == "previous_champion":
        # Older registry entries can store the comparison label rather than
        # the executable rule. The pre-optimizer champion used equal averaging.
        selected_rule = "equal_probability_blend"
    winner_metrics = oof_metrics[winner_name]
    winner_prediction = np.asarray(classes)[final_probs.argmax(axis=1)]
    validation_ids_sha256 = hashlib.sha256("\n".join(reference_ids).encode()).hexdigest()
    validation_groups_sha256 = _sha(groups.tolist())
    candidate_ids_ordered = [item["candidate_id"] for item in candidates]
    run_hash = _sha({"candidate_ids": candidate_ids_ordered,
                     "validation_ids_sha256": validation_ids_sha256,
                     "optimizer": "forward_group_cv_weighted_all_candidates_v2",
                     "validation_groups_sha256": validation_groups_sha256,
                     "validation_labels_sha256": _sha(y_true.tolist()),
                     "classes": classes,
                     "prediction_sha256": {item["candidate_id"]: hashlib.sha256(
                         np.ascontiguousarray(item["probabilities"]).tobytes()).hexdigest()
                         for item in candidates + historical},
                     "champion_ids": champion_ids, "champion_strategy": champion_rule if champion_ids else None,
                     "champion_weights": champion_weights, "group_folds": group_folds,
                     "weight_l2": weight_l2})[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"prediction_{reference_dataset}_{run_hash}"
    summary_path = output_dir / f"{prefix}_summary.json"
    steps_path = output_dir / f"{prefix}_forward_steps.csv"
    row_path = output_dir / f"{prefix}_validation_predictions.parquet"
    manifest_path = output_dir / f"{prefix}_manifest.json"
    pd.DataFrame(selection_steps).to_csv(steps_path, index=False)
    validation_output = pd.DataFrame({"id": reference_ids, "prompt_id": groups, "y_true": y_true,
        "equal_blend_prediction": np.asarray(classes)[equal_probabilities.argmax(axis=1)],
        "weighted_blend_prediction": np.asarray(classes)[weighted_probabilities.argmax(axis=1)],
        "all_candidate_weighted_prediction": np.asarray(classes)[all_candidate_probabilities.argmax(axis=1)],
        "selected_prediction": winner_prediction})
    if champion_probabilities is not None:
        validation_output["champion_prediction"] = np.asarray(classes)[champion_probabilities.argmax(axis=1)]
    validation_output.to_parquet(row_path, index=False)
    candidate_records = []
    for item in {item["candidate_id"]: item for item in candidates + historical}.values():
        manifest = item["model_manifest"]
        checkpoint = item.get("checkpoint_path")
        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path and not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        candidate_records.append({
            "candidate_id": item["candidate_id"], "model": item["model"],
            "training_dataset": item["dataset"], "source_manifest": item["manifest_path"],
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_filename": checkpoint_path.name if checkpoint_path else None,
            "selected_epoch": manifest.get("selected_epoch"),
            "model_revision": manifest.get("model_revision"), "config": manifest.get("config", {}),
            "calibration_temperature": manifest.get("calibration_temperature", 1.0),
            "classes": classes, "validation_balanced_accuracy": item["validation_score"],
            "individual_accuracy": next((score["accuracy"] for score in individual_scores
                                         if score["candidate_id"] == item["candidate_id"]), None),
        })
    report = {
        "experiment_id": prefix, "module": "prediction_factory", "reference_dataset": reference_dataset,
        "split": "validation", "classes": classes, "validation_rows": len(y_true),
        "validation_ids_sha256": validation_ids_sha256,
        "validation_groups_sha256": validation_groups_sha256,
        "validation_prompt_groups": int(len(set(groups))),
        "candidate_count": len(candidates), "candidate_ids": candidate_ids_ordered,
        "candidates": candidate_records, "excluded_candidates": excluded,
        "read_problems": read_problems,
        "selection_method": "greedy_forward_selection",
        "selection_steps": selection_steps,
        "individual_accuracy": {row["candidate_id"]: row["accuracy"] for row in individual_scores},
        "selected_candidate_ids": [item["candidate_id"] for item in final_members],
        "forward_selected_candidate_ids": selected_ids,
        "all_candidate_weights": dict(zip([item["candidate_id"] for item in candidates], all_candidate_weights.tolist())),
        "selected_weights": dict(zip([item["candidate_id"] for item in final_members], final_weights)),
        "previous_champion_candidate_ids": champion_ids,
        "all_candidate_weights": dict(zip([item["candidate_id"] for item in candidates], all_candidate_weights.tolist())),
        "previous_champion_strategy": champion.get("selected_strategy"),
        "strategy_metrics": {name: {**oof_metrics[name], "ensemble_size": sizes[name]}
                             for name in oof_metrics},
        "full_data_strategy_metrics_exploratory": {name: {**full_metrics[name], "ensemble_size": sizes[name]}
                                                   for name in full_metrics},
        "strategy_fold_details": fold_records,
        "selected_strategy": selected_rule,
        "selected_comparison": winner_name,
        "selected_members": [item["candidate_id"] for item in final_members],
        "selected_validation_accuracy": winner_metrics["accuracy"],
        "selected_validation_balanced_accuracy": winner_metrics["balanced_accuracy"],
        "selected_validation_log_loss": winner_metrics["log_loss"],
        "selected_validation_metric_scope": "prompt_group_out_of_fold",
        "selected_full_data_exploratory_metrics": full_metrics[winner_name],
        "weight_l2": weight_l2,
        "blend_rule": "weighted mean of member class-probability vectors",
        "selection_tiebreak": ["higher_group_oof_accuracy", "lower_group_oof_log_loss", "fewer_models", "strategy_name", "candidate_id_order"],
        "forward_steps_path": str(steps_path), "summary_path": str(summary_path),
        "validation_predictions_path": str(row_path),
        "validation_selection_note": "Group out-of-fold metrics hold out prompt groups during each fold's model selection and weight fitting. Base checkpoint and roster selection may already have used the full validation set, and the previous champion was selected on it; strategy selection among these metrics can also be optimistic. Training-row metrics are descriptive. Only a separate untouched test score is an independent final estimate.",
        "oof_scope_note": "Meta-level cross-fitting only; it does not undo prior checkpoint, roster, or historical champion selection on this validation set.",
    }
    summary_path.write_text(json.dumps({"available_models": len(candidates),
        "selected_models": report["selected_members"], "individual_accuracy": report["individual_accuracy"],
        "strategy_metrics_group_oof": report["strategy_metrics"],
        "strategy_metrics_full_data_exploratory": report["full_data_strategy_metrics_exploratory"],
        "selected_weights": report["selected_weights"], "selected_strategy": selected_rule,
        "all_candidate_weights": report["all_candidate_weights"],
        "selected_comparison": winner_name,
        "forward_selection_steps": selection_steps}, indent=2) + "\n")
    manifest_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    report["manifest_path"] = str(manifest_path)
    return report
