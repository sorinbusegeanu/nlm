"""Deterministic search and reliability lookup over saved model predictions."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss


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


def _read_candidates(artifact_dir, *, min_validation_rows=1, candidate_ids=None):
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
            key = (model, dataset_version)
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


def _align_candidates(candidates, *, reference_dataset="v0", reference_frame=None):
    if not candidates:
        raise ValueError("no usable validation model predictions found")
    classes = candidates[0]["classes"]
    if any(candidate["classes"] != classes for candidate in candidates):
        raise ValueError("candidate model classes do not share one canonical class order")
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
        if set(frame.id) != reference_set:
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "validation IDs differ from frozen reference"})
            continue
        frame = frame.set_index("id").loc[reference_ids].reset_index()
        if not np.array_equal(frame.y_true.astype(str).to_numpy(), y):
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "validation labels differ from frozen reference"})
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


def _search_equal_blends(candidates, y_true, classes, *, batch_size=64, top_k=100):
    """Exhaustively score all non-empty equal blends by balanced accuracy."""
    names = [candidate["candidate_id"] for candidate in candidates]
    probability_stack = np.stack([candidate["probabilities"] for candidate in candidates]).astype(np.float32)
    y_indexes = np.asarray([classes.index(label) for label in y_true], dtype=np.int64)
    best = None
    top = []
    total = 0
    for size in range(1, len(candidates) + 1):
        iterator = itertools.combinations(range(len(candidates)), size)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            indexes = np.asarray(batch, dtype=np.int16)
            averages = probability_stack[indexes].mean(axis=1)
            pred_indexes = averages.argmax(axis=2)
            accuracies = np.mean(pred_indexes == y_indexes[None, :], axis=1)
            recalls = np.stack([
                np.mean(pred_indexes[:, y_indexes == class_index] == class_index, axis=1)
                for class_index in range(len(classes))
            ], axis=1)
            scores = recalls.mean(axis=1)
            for subset, score, accuracy, probs in zip(batch, scores, accuracies, averages):
                total += 1
                member_names = tuple(names[index] for index in subset)
                record = {"members": list(member_names), "accuracy": float(accuracy),
                          "balanced_accuracy": float(score), "size": len(subset)}
                key = (float(score), -len(subset), tuple(member_names))
                if (best is None or key[:2] > best[0][:2] or
                        (key[:2] == best[0][:2] and key[2] < best[0][2])):
                    best = (key, record, probs.copy())
                top.append((key, record))
            if len(top) > 2 * top_k:
                top.sort(key=lambda item: (-item[0][0], -item[0][1], item[0][2]))
                top = top[:top_k]
    top.sort(key=lambda item: (-item[0][0], -item[0][1], item[0][2]))
    return best[1], best[2], [item[1] for item in top[:top_k]], total


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
                inference_batch = max(1, min(configured_batch, 4 if model_name in {
                    "qwen4", "llama32_3b", "smollm3_3b", "granite4_h_micro",
                    "granite4_350m", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"} else 8))
                probabilities, returned_classes = _predict_neural(
                    model_name, checkpoint, training.text.astype(str).tolist(), classes,
                    batch_size=inference_batch,
                    max_length=int(config.get("max_length", 384)),
                    temperature=float(manifest.get("calibration_temperature", 1.0) or 1.0),
                    base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision"),
                    fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
                    text_only_backbone=bool(config.get("text_only_backbone", False)))
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
                                   data_dir="data/processed"):
    """Compare equal blends with a train-fitted per-response reliability selector."""
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir) if output_dir else artifact_dir / "predictions"
    all_candidates, read_problems = _read_candidates(artifact_dir / "predictions")
    references = [item for item in all_candidates if item["dataset"] == reference_dataset]
    reference_frame = references[0]["frame"] if references else None
    if candidate_ids is None:
        candidates = all_candidates
    else:
        requested_ids = set(map(str, candidate_ids))
        candidates = [item for item in all_candidates if item["candidate_id"] in requested_ids]
        missing_ids = requested_ids - {item["candidate_id"] for item in candidates}
        if missing_ids:
            raise FileNotFoundError(f"active candidate manifests or predictions are missing: {sorted(missing_ids)}")
    classes, y_true, groups, candidates, excluded, reference_ids = _align_candidates(
        candidates, reference_dataset=reference_dataset, reference_frame=reference_frame)
    training_predictions, training_labels, training_ids_sha256 = _training_reliability_predictions(
        candidates, data_dir=data_dir, output_dir=output_dir)
    table = _reliability_table(candidates, training_predictions, classes,
                               minimum_support=minimum_support)
    lookup_rows, lookup_pred, used_ids = _lookup_predictions(
        candidates, classes, table, minimum_support=minimum_support)
    lookup_accuracy = float(accuracy_score(y_true, lookup_pred))
    lookup_balanced_accuracy = float(balanced_accuracy_score(y_true, lookup_pred))
    lookup_log_loss = None
    blend_best, blend_probabilities, top_blends, combination_count = _search_equal_blends(
        candidates, y_true, classes, batch_size=blend_batch_size, top_k=top_k)
    blend_pred = np.asarray(classes)[blend_probabilities.argmax(axis=1)]
    blend_accuracy = float(accuracy_score(y_true, blend_pred))
    blend_balanced_accuracy = float(balanced_accuracy_score(y_true, blend_pred))
    blend_log_loss = float(log_loss(y_true, blend_probabilities, labels=classes))

    choices = [
        {"strategy": "per_response_reliability", "accuracy": lookup_accuracy,
         "balanced_accuracy": lookup_balanced_accuracy, "log_loss": lookup_log_loss,
         "members": sorted(used_ids), "prediction": lookup_pred},
        {"strategy": "equal_probability_blend", "accuracy": blend_accuracy,
         "balanced_accuracy": blend_balanced_accuracy, "log_loss": blend_log_loss,
         "members": blend_best["members"], "prediction": blend_pred},
    ]
    choices.sort(key=lambda item: (-item["balanced_accuracy"], len(item["members"]),
                                   item["strategy"], tuple(item["members"])))
    winner = choices[0]
    validation_ids_sha256 = hashlib.sha256("\n".join(reference_ids).encode()).hexdigest()
    candidate_ids_ordered = [item["candidate_id"] for item in candidates]
    run_hash = _sha({"candidate_ids": candidate_ids_ordered,
                     "validation_ids_sha256": validation_ids_sha256,
                     "minimum_support": minimum_support,
                     "prior": [1.0, 1.0], "blend_method": "equal_probability_average"})[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"prediction_{reference_dataset}_{run_hash}"
    trust_path = output_dir / f"{prefix}_trust_table.csv"
    blends_path = output_dir / f"{prefix}_top_blends.csv"
    row_path = output_dir / f"{prefix}_validation_predictions.parquet"
    manifest_path = output_dir / f"{prefix}_manifest.json"
    pd.DataFrame(table).to_csv(trust_path, index=False)
    pd.DataFrame(top_blends).to_csv(blends_path, index=False)
    validation_output = pd.DataFrame({"id": reference_ids, "y_true": y_true,
        "lookup_prediction": lookup_pred, "blend_prediction": blend_pred,
        "selected_prediction": winner["prediction"]})
    for i, row in enumerate(lookup_rows):
        validation_output.loc[i, "lookup_candidate_id"] = row["candidate_id"]
        validation_output.loc[i, "lookup_confidence"] = row["confidence"]
        validation_output.loc[i, "lookup_smoothed_trust"] = row["smoothed_trust"]
        validation_output.loc[i, "lookup_fallback"] = row["fallback"]
    validation_output.to_parquet(row_path, index=False)
    candidate_records = []
    for item in candidates:
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
            "training_balanced_accuracy": item.get("training_balanced_accuracy"),
            "training_prediction_path": item.get("training_prediction_path"),
        })
    metrics = {}
    for choice in choices:
        metrics[choice["strategy"]] = {key: choice[key] for key in
            ("accuracy", "balanced_accuracy", "log_loss", "members")}
    report = {
        "experiment_id": prefix, "module": "prediction_factory", "reference_dataset": reference_dataset,
        "split": "validation", "classes": classes, "validation_rows": len(y_true),
        "validation_ids_sha256": validation_ids_sha256,
        "validation_prompt_groups": int(len(set(groups))),
        "candidate_count": len(candidates), "candidate_ids": candidate_ids_ordered,
        "combination_count": combination_count,
        "candidates": candidate_records, "excluded_candidates": excluded,
        "read_problems": read_problems,
        "reliability_rule": {"fit_split": "v0_train", "fit_ids_sha256": training_ids_sha256,
            "fit_rows": int(len(training_labels)), "metric": "precision_given_prediction",
            "true_positive": "prediction equals true label",
            "false_positive": "prediction differs from true label", "prior_alpha": 1.0,
            "prior_beta": 1.0, "minimum_support": minimum_support,
            "score": "(true_positive + 1) / (predicted_count + 2)",
            "fallback": "highest_training_balanced_accuracy_candidate_argmax"},
        "blend_rule": "uniform mean of member class-probability vectors",
        "strategy_metrics": metrics, "selected_strategy": winner["strategy"],
        "selected_members": winner["members"],
        "selected_validation_accuracy": float(accuracy_score(y_true, winner["prediction"])),
        "selected_validation_balanced_accuracy": winner["balanced_accuracy"],
        "selection_tiebreak": ["higher_balanced_accuracy", "fewer_members", "strategy_name", "candidate_id_order"],
        "trust_table_path": str(trust_path), "top_blends_path": str(blends_path),
        "validation_predictions_path": str(row_path),
        "validation_selection_note": "Reliability statistics use only V0 training rows. The winning strategy is selected from validation results; its score is exploratory and may be optimistic.",
    }
    manifest_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    report["manifest_path"] = str(manifest_path)
    return report
