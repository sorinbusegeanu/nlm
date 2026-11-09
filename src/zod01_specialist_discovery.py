"""Discover high-precision label rules across saved NLM checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .checkpoint_comparison import _validated_probabilities
from .prediction_factory import _align_candidates, _read_candidates
from .trusted_checkpoint_ensemble import ULTRABERT_CHECKPOINTS
from .zod01 import _apply_rules, _label_stats, _learn_rules

MODEL_TITLES = {
    "catboost_stylometry": "CatBoost + Stylometry",
    "embeddinggemma2": "EmbeddingGemma 2",
    "granite4_350m": "Granite 4.0 350M Base",
    "gemma4_e2b": "Gemma 4 E2B",
    "granite4_h_micro": "Granite 4 H Micro",
    "llama32_3b": "Llama 3.2 3B",
    "ministral3_3b": "Ministral 3 3B",
    "qwen06": "Qwen3 0.6B",
    "qwen17": "Qwen3 1.7B",
    "qwen4": "Qwen3 4B",
    "smollm3_3b": "SmolLM3 3B",
    "svc": "SVC",
    "ultrabert380": "UltraBERT-380M",
}
REQUESTED_MODELS = set(MODEL_TITLES) - {"ultrabert380"}
ULTRABERT_IDS = set(ULTRABERT_CHECKPOINTS.values())
MINIMUM_SUPPORT = 10
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0


def _resolve(path_value):
    path = Path(path_value) if path_value else Path()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _inference_for_candidate(candidate, texts, classes):
    """Use the existing model inference APIs to rebuild a V0 cache if needed."""
    from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
    manifest = candidate["model_manifest"]
    model = str(manifest.get("model", candidate["model"]))
    if model == "qwen06_best_epoch03":
        model = "qwen06"
    checkpoint = _resolve(candidate.get("checkpoint_path") or manifest.get("checkpoint_path") or manifest.get("checkpoint"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint unavailable: {checkpoint}")
    temperature = float(manifest.get("calibration_temperature", 1.0) or 1.0)
    if model == "catboost_stylometry":
        probabilities, _ = predict_catboost_stylometry(checkpoint, texts, classes, temperature=temperature)
    elif model == "svc":
        bundle = joblib.load(checkpoint)
        probabilities, _ = predict_svc_bundle(bundle, texts)
    else:
        config = manifest.get("config") or {}
        probabilities, _ = _predict_neural(model, checkpoint, texts, classes,
            max_length=int(config.get("max_length", 384)), temperature=temperature,
            base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision"),
            fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
            text_only_backbone=bool(config.get("text_only_backbone", False)),
            quantization=config.get("quantization", manifest.get("quantization")),
            attention_backend=config.get("attention_backend", manifest.get("attention_backend")))
    return _validated_probabilities(probabilities, len(texts), len(classes), candidate["candidate_id"])


def _candidate_stats(candidate, y, classes, minimum_support):
    probabilities = np.asarray(candidate["probabilities"], dtype=np.float64)
    predictions = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
    result = []
    for label, stats in _label_stats(y, predictions, classes).items():
        count = stats["prediction_count"]
        smoothed = (stats["true_positive"] + PRIOR_ALPHA) / (count + PRIOR_ALPHA + PRIOR_BETA)
        result.append({"candidate_key": candidate["candidate_key"],
            "candidate_id": candidate["candidate_id"], "model": candidate["model"],
            "model_title": MODEL_TITLES.get(candidate["model"], candidate["model"]),
            "dataset": candidate["dataset"], "label": label,
            "true_positive": stats["true_positive"], "false_positive": stats["false_positive"],
            "false_negative": stats["false_negative"], "prediction_count": count,
            "precision": stats["precision"], "smoothed_precision": float(smoothed),
            "recall": stats["recall"], "support_eligible": count >= minimum_support,
            "evaluation_eligible": bool(candidate.get("evaluation_eligible", True)),
            "eligible": count >= minimum_support and bool(candidate.get("evaluation_eligible", True)),
            "evaluation_exclusion_reason": candidate.get("evaluation_exclusion_reason", ""),
            "checkpoint_available": bool(candidate["checkpoint_available"])})
    return predictions, result


def _run_cascade(candidates, y, groups, classes, *, candidate_order, folds=5):
    """Cross-fit ZOD01 ordering for a candidate set without loading checkpoints."""
    n = len(y)
    output = np.empty(n, dtype=object)
    selected = np.empty(n, dtype=object)
    fold_ids = np.full(n, -1, dtype=int)
    splitter = GroupKFold(n_splits=folds)
    by_id = {item["candidate_id"]: item for item in candidates}
    traces = []
    for fold, (fit_idx, test_idx) in enumerate(splitter.split(np.zeros(n), y, groups), start=1):
        train_candidates = [{**by_id[cid], "probabilities": np.asarray(by_id[cid]["probabilities"])[fit_idx]}
                            for cid in candidate_order]
        rules = _learn_rules(train_candidates, y[fit_idx], classes, minimum_support=MINIMUM_SUPPORT,
            prior_alpha=PRIOR_ALPHA, prior_beta=PRIOR_BETA, candidate_order=candidate_order)
        test_candidates = [{**by_id[cid], "probabilities": np.asarray(by_id[cid]["probabilities"])[test_idx]}
                           for cid in candidate_order]
        pred, decisions, trace = _apply_rules(test_candidates, classes, rules,
            minimum_support=MINIMUM_SUPPORT, fallback_candidate_id="D",
            y_true=y[test_idx], fold=fold)
        output[test_idx] = pred
        selected[test_idx] = decisions.selected_candidate_id.to_numpy()
        fold_ids[test_idx] = fold
        for row in trace:
            traces.append(row)
    return output, selected, fold_ids, pd.DataFrame(traces)


def _run_additive_precision_cascade(base_candidates, all_candidates, y, groups, classes,
                                   *, base_order, all_order, minimum_precision, folds=5):
    """Add only fit-fold high-precision new rules ahead of the original A-D cascade."""
    n = len(y)
    output = np.empty(n, dtype=object)
    selected = np.empty(n, dtype=object)
    split_ids = np.full(n, -1, dtype=int)
    base_ids = set(base_order)
    full_by_id = {item["candidate_id"]: item for item in all_candidates}
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(np.zeros(n), y, groups), start=1):
        fit_all = [{**full_by_id[cid], "probabilities": np.asarray(full_by_id[cid]["probabilities"])[fit_idx]}
                   for cid in all_order]
        new_rules = _learn_rules(fit_all, y[fit_idx], classes, minimum_support=MINIMUM_SUPPORT,
            prior_alpha=PRIOR_ALPHA, prior_beta=PRIOR_BETA, candidate_order=all_order)
        additions = [rule for rule in new_rules if rule["candidate_id"] not in base_ids
                     and rule["eligible"] and rule["smoothed_precision"] >= minimum_precision]
        fit_base = [{**full_by_id[cid], "probabilities": np.asarray(full_by_id[cid]["probabilities"])[fit_idx]}
                    for cid in base_order]
        base_rules = _learn_rules(fit_base, y[fit_idx], classes, minimum_support=MINIMUM_SUPPORT,
            prior_alpha=PRIOR_ALPHA, prior_beta=PRIOR_BETA, candidate_order=base_order)
        combined_rules = [*additions, *base_rules]
        for rank, rule in enumerate(combined_rules, start=1):
            rule["rule_rank"] = rank
        held = [{**full_by_id[cid], "probabilities": np.asarray(full_by_id[cid]["probabilities"])[held_idx]}
                for cid in all_order]
        pred, decisions, _ = _apply_rules(held, classes, combined_rules,
            minimum_support=MINIMUM_SUPPORT, fallback_candidate_id="D", y_true=y[held_idx], fold=fold)
        output[held_idx] = pred
        selected[held_idx] = decisions.selected_candidate_id.to_numpy()
        split_ids[held_idx] = fold
    return output, selected, split_ids


def run_zod01_specialist_discovery(*, artifact_dir="artifacts/predictions",
        data_dir="data/processed", output_dir="artifacts/analysis/zod01_specialist_discovery",
        report_path="docs/zod01_specialist_discovery.md", minimum_support=MINIMUM_SUPPORT):
    from .pipeline import _load_dataset
    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.eq("validation"), ["id", "prompt_id", "model", "text"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["y_true"] = validation.pop("model").astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540 or validation.prompt_id.nunique() != 30:
        raise ValueError("expected frozen V0 validation with 540 rows and 30 prompt groups")
    classes = sorted(validation.y_true.unique().tolist())

    # Keep every trained run/config as a distinct candidate, including seed and
    # continuation variants. Add UltraBERT A-D strictly as comparison rules.
    selected_manifest_ids = set(ULTRABERT_IDS)
    for path in sorted(Path(artifact_dir).glob("*_manifest.json")):
        try:
            manifest = json.loads(path.read_text())
        except Exception:
            continue
        model = str(manifest.get("model", ""))
        if model == "qwen06_best_epoch03":
            model = "qwen06"
        if model in REQUESTED_MODELS and manifest.get("experiment_id"):
            selected_manifest_ids.add(str(manifest["experiment_id"]))
    raw_candidates, problems = _read_candidates(Path(artifact_dir), min_validation_rows=1,
        candidate_ids=selected_manifest_ids, deduplicate=False)
    candidates_by_id = {item["candidate_id"]: item for item in raw_candidates}
    expected_ultra = ULTRABERT_IDS - set(candidates_by_id)
    if expected_ultra:
        raise FileNotFoundError(f"missing UltraBERT baseline predictions: {sorted(expected_ultra)}")

    output_dir = Path(output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    ref = validation[["id", "prompt_id", "y_true"]].copy()
    v0_profile = json.loads((Path(data_dir) / "dataset_profile.json").read_text())
    v0_dataset_ids = set(rows.id.astype(str))
    frozen_validation_ids = set(validation.id.astype(str))
    v0_data_hash = str(v0_profile.get("dataset_sha256", ""))
    v0_split_hash = str(v0_profile.get("split_sha256", ""))
    evaluated = []
    excluded = []
    ordered_ids = sorted(candidates_by_id)
    for candidate_id in ordered_ids:
        candidate = candidates_by_id[candidate_id]
        model = candidate["model"]
        if model == "qwen06_best_epoch03":
            model = "qwen06"
        candidate["model"] = model
        candidate["candidate_id"] = candidate_id
        candidate["dataset"] = str(candidate["dataset"])
        candidate["model_manifest"]["model"] = model
        manifest = candidate["model_manifest"]
        candidate["evaluation_eligible"] = True
        candidate["validation_training_overlap"] = 0
        candidate["evaluation_exclusion_reason"] = ""
        # Historical alternate splits over the V0 source can put frozen
        # validation rows in the checkpoint's training partition.
        if (str(manifest.get("dataset_sha256", "")) == v0_data_hash
                and str(manifest.get("split_sha256", "")) != v0_split_hash):
            cached_validation_ids = set(candidate["frame"].id.astype(str))
            inferred_train_ids = v0_dataset_ids - cached_validation_ids
            if len(inferred_train_ids) == int(manifest.get("train_rows", -1)):
                overlap = len(inferred_train_ids & frozen_validation_ids)
                candidate["validation_training_overlap"] = overlap
                if overlap:
                    candidate["evaluation_eligible"] = False
                    candidate["evaluation_exclusion_reason"] = (
                        f"training partition contains {overlap} of the frozen V0 validation rows")
            else:
                candidate["evaluation_eligible"] = False
                candidate["evaluation_exclusion_reason"] = (
                    "alternate V0 split training IDs could not be verified")
        try:
            _, _, _, aligned_list, align_errors, ids = _align_candidates([candidate],
                reference_dataset="v0", reference_frame=ref)
        except Exception as exc:
            aligned_list, align_errors = [], [{"reason": f"{type(exc).__name__}: {exc}"}]
        cache_status = "cached"
        if aligned_list:
            aligned = aligned_list[0]
            probabilities = aligned["probabilities"]
        else:
            prior_v0_cache = prediction_dir / f"{candidate_id}_validation.parquet"
            if prior_v0_cache.is_file():
                prior = pd.read_parquet(prior_v0_cache)
                prior["id"] = prior.id.astype(str)
                if (set(prior.id) == set(validation.id) and not prior.id.duplicated().any()
                        and np.array_equal(prior.set_index("id").loc[validation.id, "y_true"].astype(str).to_numpy(),
                                           validation.y_true.astype(str).to_numpy())
                        and np.array_equal(prior.set_index("id").loc[validation.id, "prompt_id"].astype(str).to_numpy(),
                                           validation.prompt_id.astype(str).to_numpy())):
                    prior = prior.set_index("id").loc[validation.id].reset_index()
                    probabilities = _validated_probabilities(prior[[f"prob::{label}" for label in classes]].to_numpy(),
                        len(validation), len(classes), candidate_id)
                    cache_status = "reused_generated_v0_cache"
                    ids = validation.id.tolist()
                else:
                    prior_v0_cache.unlink()
            if cache_status != "reused_generated_v0_cache":
                cache_status = "inferred_on_frozen_v0"
                try:
                    probabilities = _inference_for_candidate(candidate, validation.text.astype(str).tolist(), classes)
                except Exception as exc:
                    excluded.append({"candidate_id": candidate_id, "model": model,
                        "dataset": candidate["dataset"], "reason": f"inference failed: {type(exc).__name__}: {exc}",
                        "alignment": align_errors})
                    continue
                ids = validation.id.tolist()
        probabilities = _validated_probabilities(probabilities, len(validation), len(classes), candidate_id)
        candidate["probabilities"] = probabilities
        candidate["classes"] = classes
        candidate["ids"] = list(map(str, ids))
        candidate["prompt_ids"] = validation.prompt_id.tolist()
        candidate["cache_status"] = cache_status
        checkpoint_path = _resolve(candidate.get("checkpoint_path") or
                                   candidate["model_manifest"].get("checkpoint_path") or
                                   candidate["model_manifest"].get("checkpoint"))
        candidate["checkpoint_available"] = checkpoint_path.exists()
        candidate["checkpoint_basename"] = checkpoint_path.name if candidate["checkpoint_available"] else "missing"
        standardized = pd.DataFrame({"id": validation.id, "prompt_id": validation.prompt_id,
                                     "y_true": validation.y_true})
        for index, label in enumerate(classes):
            standardized[f"prob::{label}"] = probabilities[:, index]
        standardized["prediction"] = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
        saved_prediction = prediction_dir / f"{candidate_id}_validation.parquet"
        standardized.to_parquet(saved_prediction, index=False)
        candidate["saved_prediction_path"] = str(saved_prediction)
        candidate["validation_accuracy"] = float(np.mean(standardized.prediction.to_numpy() == validation.y_true.to_numpy()))
        candidate["selected_epoch"] = manifest.get("selected_epoch")
        candidate["requested_epochs"] = (manifest.get("config") or {}).get("epochs")
        epoch_history = manifest.get("validation_metrics_by_epoch") or []
        candidate["recorded_training_epochs"] = len(epoch_history)
        candidate["selected_iteration"] = manifest.get("selected_iteration")
        candidate["manifest_selected_validation_accuracy"] = manifest.get("selected_validation_accuracy")
        candidate["seed"] = manifest.get("seed")
        candidate["experiment_name"] = manifest.get("experiment_name", "")
        candidate["checkpoint_path_resolved"] = str(checkpoint_path)
        evaluated.append(candidate)

    # Stable compact IDs make the exhaustive ranking navigable; exact IDs stay
    # in the inventory table and JSON/CSV artifacts.
    non_ultra = sorted([c for c in evaluated if c["candidate_id"] not in ULTRABERT_IDS],
                       key=lambda c: (c["model"], c["dataset"], c["candidate_id"]))
    ultra = {alias: next(c for c in evaluated if c["candidate_id"] == experiment_id)
             for alias, experiment_id in ULTRABERT_CHECKPOINTS.items()}
    for index, candidate in enumerate(non_ultra, start=1):
        candidate["candidate_key"] = f"N{index:04d}"
    for alias, candidate in ultra.items():
        candidate["candidate_key"] = alias
    ranking_candidates = [*non_ultra, *[ultra[key] for key in ULTRABERT_CHECKPOINTS]]
    all_stats, checkpoint_accuracy = [], {}
    probability_lookup = {}
    for candidate in ranking_candidates:
        prediction, stats = _candidate_stats(candidate, validation.y_true.to_numpy(), classes, minimum_support)
        all_stats.extend(stats)
        checkpoint_accuracy[candidate["candidate_id"]] = float(np.mean(prediction == validation.y_true.to_numpy()))
        probability_lookup[candidate["candidate_id"]] = candidate["probabilities"]
    rank = pd.DataFrame(all_stats)
    rank["rank"] = pd.NA
    eligible = rank.loc[rank.eligible].sort_values(
        ["smoothed_precision", "precision", "prediction_count", "candidate_key", "label"],
        ascending=[False, False, False, True, True], kind="stable").copy()
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    rank = rank.drop(columns="rank").merge(eligible[["candidate_id", "label", "rank"]],
        on=["candidate_id", "label"], how="left", validate="one_to_one")
    rank = rank.sort_values(["rank", "candidate_key", "label"], na_position="last", kind="stable")

    # Best individual model per label and comparison to strongest UltraBERT rule.
    best_overall, best_new, ultra_best, outperform = {}, {}, {}, []
    for label in classes:
        eligible_label = eligible[eligible.label.eq(label)]
        best_overall[label] = eligible_label.iloc[0].to_dict() if not eligible_label.empty else None
        new_label = eligible_label[~eligible_label.candidate_id.isin(ULTRABERT_IDS)]
        best_new[label] = new_label.iloc[0].to_dict() if not new_label.empty else None
        old_label = eligible_label[eligible_label.candidate_id.isin(ULTRABERT_IDS)]
        ultra_best[label] = old_label.iloc[0].to_dict() if not old_label.empty else None
        if best_new[label] is not None and ultra_best[label] is not None:
            new_rule, old_rule = best_new[label], ultra_best[label]
            if (new_rule["smoothed_precision"], new_rule["precision"]) > (old_rule["smoothed_precision"], old_rule["precision"]):
                outperform.append({"label": label, "candidate_key": new_rule["candidate_key"],
                    "candidate_id": new_rule["candidate_id"], "model_title": new_rule["model_title"],
                    "dataset": new_rule["dataset"], "precision": float(new_rule["precision"]),
                    "smoothed_precision": float(new_rule["smoothed_precision"]),
                    "recall": float(new_rule["recall"]), "prediction_count": int(new_rule["prediction_count"]),
                    "ultrabert_candidate_key": old_rule["candidate_key"],
                    "ultrabert_precision": float(old_rule["precision"]),
                    "ultrabert_smoothed_precision": float(old_rule["smoothed_precision"]),
                    "ultrabert_prediction_count": int(old_rule["prediction_count"])})

    precision_thresholds = {}
    for source, frame in (("new_checkpoints", eligible[~eligible.candidate_id.isin(ULTRABERT_IDS)]),
                          ("all_checkpoints", eligible)):
        precision_thresholds[source] = {}
        for threshold in (.80, .90):
            rows_at_threshold = frame[frame.precision >= threshold]
            labels_at_threshold = sorted(rows_at_threshold.label.unique().tolist())
            precision_thresholds[source][f"precision_ge_{int(threshold*100)}"] = {
                "rule_count": int(len(rows_at_threshold)), "label_count": len(labels_at_threshold),
                "labels": labels_at_threshold}

    # Directly test whether adding specialists improves the existing ZOD01
    # cascade under the same prompt-group folds.
    baseline_order = list(ULTRABERT_CHECKPOINTS)
    baseline_candidates = [{**ultra[alias], "candidate_id": alias} for alias in baseline_order]
    cascade_non_ultra = [candidate for candidate in non_ultra if candidate.get("evaluation_eligible", True)]
    expanded_order = baseline_order + [candidate["candidate_key"] for candidate in cascade_non_ultra]
    expanded_candidates = [{**ultra[alias], "candidate_id": alias} for alias in baseline_order] + [
        {**candidate, "candidate_id": candidate["candidate_key"]} for candidate in cascade_non_ultra]
    baseline_oof, _, folds, baseline_trace = _run_cascade(baseline_candidates,
        validation.y_true.to_numpy(), validation.prompt_id.to_numpy(), classes,
        candidate_order=baseline_order)
    expanded_oof, expanded_selected, _, expanded_trace = _run_cascade(expanded_candidates,
        validation.y_true.to_numpy(), validation.prompt_id.to_numpy(), classes,
        candidate_order=expanded_order)
    thresholded_cascades = {}
    thresholded_outputs = {}
    for precision_floor in (.70, .75, .80, .85, .90):
        pred, selected, _ = _run_additive_precision_cascade(baseline_candidates,
            expanded_candidates, validation.y_true.to_numpy(), validation.prompt_id.to_numpy(),
            classes, base_order=baseline_order, all_order=expanded_order,
            minimum_precision=precision_floor)
        correct_mask = pred == validation.y_true.to_numpy()
        thresholded_outputs[precision_floor] = (pred, selected)
        thresholded_cascades[f"{precision_floor:.2f}"] = {
            "accuracy": float(np.mean(correct_mask)), "correct": int(correct_mask.sum()),
            "selected_additional_rows": int(sum(candidate_key not in set(baseline_order)
                for candidate_key in selected)),
            "corrected_vs_ultrabert": int(np.sum((baseline_oof != validation.y_true.to_numpy()) & correct_mask)),
            "regressed_vs_ultrabert": int(np.sum((baseline_oof == validation.y_true.to_numpy()) & ~correct_mask)),
            "per_label": _label_stats(validation.y_true.to_numpy(), pred, classes)}
    baseline_metrics = {"accuracy": float(np.mean(baseline_oof == validation.y_true.to_numpy())),
        "correct": int(np.sum(baseline_oof == validation.y_true.to_numpy())),
        "per_label": _label_stats(validation.y_true.to_numpy(), baseline_oof, classes)}
    expanded_metrics = {"accuracy": float(np.mean(expanded_oof == validation.y_true.to_numpy())),
        "correct": int(np.sum(expanded_oof == validation.y_true.to_numpy())),
        "per_label": _label_stats(validation.y_true.to_numpy(), expanded_oof, classes),
        "selected_candidate_counts": pd.Series(expanded_selected).value_counts().to_dict()}
    cascade_summary = {"ultrabert_A_to_D": baseline_metrics, "expanded": expanded_metrics,
        "thresholded_additions": thresholded_cascades,
        "delta_accuracy": expanded_metrics["accuracy"] - baseline_metrics["accuracy"],
        "corrected_vs_ultrabert": int(np.sum((baseline_oof != validation.y_true.to_numpy()) & (expanded_oof == validation.y_true.to_numpy()))),
        "regressed_vs_ultrabert": int(np.sum((baseline_oof == validation.y_true.to_numpy()) & (expanded_oof != validation.y_true.to_numpy())))}

    output_dir.mkdir(parents=True, exist_ok=True)
    rank.to_csv(output_dir / "ranked_checkpoint_label_rules.csv", index=False)
    inventory = []
    for c in sorted(evaluated, key=lambda item: (item["candidate_key"], item["candidate_id"])):
        cfg = c["model_manifest"].get("config") or {}
        inventory.append({"candidate_key": c["candidate_key"], "experiment_id": c["candidate_id"],
            "model": c["model"], "model_title": MODEL_TITLES.get(c["model"], c["model"]),
            "training_dataset": c["dataset"], "validation_accuracy": c["validation_accuracy"],
            "selected_epoch": c.get("selected_epoch"), "requested_epochs": c.get("requested_epochs"),
            "recorded_training_epochs": c.get("recorded_training_epochs"),
            "selected_iteration": c.get("selected_iteration"),
            "manifest_selected_validation_accuracy": c.get("manifest_selected_validation_accuracy"),
            "seed": c.get("seed"), "checkpoint_basename": c["checkpoint_basename"],
            "checkpoint_available": c["checkpoint_available"],
            "evaluation_eligible": c.get("evaluation_eligible", True),
            "validation_training_overlap": c.get("validation_training_overlap", 0),
            "evaluation_exclusion_reason": c.get("evaluation_exclusion_reason", ""),
            "prediction_source": c["cache_status"],
            "checkpoint_path": c["checkpoint_path_resolved"], "prediction_path": c["saved_prediction_path"],
            "manifest_path": c["manifest_path"], "configured_base_model": cfg.get("checkpoint")})
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(output_dir / "checkpoint_inventory.csv", index=False)
    fold_oof = pd.DataFrame({"id": validation.id, "prompt_id": validation.prompt_id,
        "y_true": validation.y_true, "fold": folds, "zod01_ultrabert_prediction": baseline_oof,
        "zod01_expanded_prediction": expanded_oof, "expanded_selected_candidate": expanded_selected})
    for floor, (pred, selected) in thresholded_outputs.items():
        key = f"{int(floor*100)}"
        fold_oof[f"zod01_add_precision_{key}_prediction"] = pred
        fold_oof[f"zod01_add_precision_{key}_selected"] = selected
    fold_oof.to_parquet(output_dir / "cascade_comparison_oof.parquet", index=False)
    baseline_trace.to_csv(output_dir / "zod01_ultrabert_rule_trace.csv", index=False)
    expanded_trace.to_csv(output_dir / "zod01_expanded_rule_trace.csv", index=False)
    profile_path = Path(data_dir) / "profile_v0.json"
    if not profile_path.exists():
        profile_path = Path(data_dir) / "dataset_profile.json"
    try:
        split_hash = json.loads(profile_path.read_text()).get("split_sha256")
    except Exception:
        split_hash = None
    metrics = {"validation_rows": len(validation), "classes": classes,
        "prompt_groups": int(validation.prompt_id.nunique()), "minimum_support": minimum_support,
        "prior_alpha": PRIOR_ALPHA, "prior_beta": PRIOR_BETA,
        "validation_split_sha256": split_hash,
        "validation_ids_sha256": hashlib.sha256("\n".join(validation.id).encode()).hexdigest(),
        "candidate_count": len(evaluated), "new_candidate_count": len(non_ultra),
        "evaluation_eligible_candidate_count": len(cascade_non_ultra) + len(ultra),
        "validation_leak_candidates": [{"candidate_key": c["candidate_key"],
            "candidate_id": c["candidate_id"], "training_validation_overlap": c.get("validation_training_overlap", 0),
            "reason": c.get("evaluation_exclusion_reason", "")}
            for c in evaluated if not c.get("evaluation_eligible", True)],
        "excluded_candidates": excluded, "prediction_factory_issues": problems,
        "precision_threshold_coverage": precision_thresholds,
        "best_checkpoint_per_label": {label: _to_json(best_overall[label]) for label in classes},
        "best_new_checkpoint_per_label": {label: _to_json(best_new[label]) for label in classes},
        "best_ultrabert_checkpoint_per_label": {label: _to_json(ultra_best[label]) for label in classes},
        "new_specialists_outperforming_ultrabert": outperform,
        "zod01_cascade_comparison": cascade_summary,
        "ranked_rules_csv": str((output_dir / "ranked_checkpoint_label_rules.csv").resolve()),
        "inventory_csv": str((output_dir / "checkpoint_inventory.csv").resolve()),
        "prediction_dir": str(prediction_dir.resolve())}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_report(Path(report_path), inventory_frame, rank, metrics)
    return {"report_path": str(report_path), "candidate_count": len(evaluated),
        "new_candidate_count": len(non_ultra), "evaluation_eligible_candidate_count": len(cascade_non_ultra) + len(ultra),
        "rule_count": int(len(rank)),
        "eligible_rule_count": int(rank.eligible.sum()), "inferred_candidates": int(sum(c["cache_status"] == "inferred_on_frozen_v0" for c in evaluated)),
        "precision_threshold_coverage": precision_thresholds,
        "cascade_accuracy": expanded_metrics["accuracy"], "baseline_cascade_accuracy": baseline_metrics["accuracy"],
        "delta_cascade_accuracy": cascade_summary["delta_accuracy"],
        "excluded_candidates": excluded}


def _to_json(value):
    if value is None:
        return None
    return {key: (item.item() if isinstance(item, np.generic) else item)
            for key, item in value.items() if key in {"candidate_key", "candidate_id", "model_title", "dataset",
                "label", "precision", "smoothed_precision", "recall", "prediction_count", "true_positive", "false_positive"}}


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value): return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_report(path, inventory, ranking, metrics):
    lines = ["# ZOD01 Specialist Discovery — Existing NLM Checkpoints", "",
        "## Scope and validation", "",
        f"Evaluated {metrics['candidate_count']} distinct checkpoint runs ({metrics['new_candidate_count']} requested non-UltraBERT runs plus UltraBERT A–D) on the frozen V0 validation set: {metrics['validation_rows']} rows, {metrics['prompt_groups']} prompt groups and {len(metrics['classes'])} labels. Existing aligned probability caches were reused. Nonmatching cached validations were regenerated on frozen V0 from their saved checkpoints. No training was run.", "",
        "All checkpoints have previously been compared or selected using validation data, and this specialist ranking selects rules on those same 540 labels. Treat precision and cascade scores as development diagnostics, not independent generalization estimates. Alternate splits of the V0 source that trained on frozen V0 validation rows remain inventoried but are excluded from ranked rules and cascade evaluation.", "",
        f"Rule rank uses Beta(1,1) smoothed precision `(TP+1)/(prediction_count+2)` and minimum support {metrics['minimum_support']}. The threshold counts below use raw observed precision with the same minimum support. Split SHA-256: `{metrics.get('validation_split_sha256')}`; validation ID SHA-256: `{metrics['validation_ids_sha256']}`.", "",
        "## Checkpoint inventory", "",
        "The checkpoint key is used in the global ranking; the exact experiment ID and checkpoint basename/path are in the inventory CSV. Accuracy is recomputed from aligned V0 probabilities. Epoch history and selected CatBoost iteration are recorded in the inventory CSV. The overlap column counts frozen V0 validation rows used in training.", "",
        "| Key | Model | Training data | V0 accuracy | Selected epoch | Checkpoint present | V0 train overlap | Prediction source | Experiment ID |", "|---|---|---|---:|---:|---|---:|---|---|"]
    for _, row in inventory.iterrows():
        epoch = "—" if pd.isna(row.selected_epoch) else str(int(row.selected_epoch))
        present = "yes" if bool(row.checkpoint_available) else "no (cached probabilities only)"
        short_id = str(row.experiment_id)
        lines.append(f"| {row.candidate_key} | {row.model_title} | {row.training_dataset} | {row.validation_accuracy:.1%} | {epoch} | {present} | {int(row.validation_training_overlap)} | {row.prediction_source} | `{short_id}` |")
    if metrics["validation_leak_candidates"]:
        lines.extend(["", "Excluded from precision ranking and cascade because their fit partitions overlap frozen validation:"])
        for item in metrics["validation_leak_candidates"]:
            lines.append(f"- `{item['candidate_id']}`: {item['training_validation_overlap']} frozen V0 validation rows in its training partition.")
    lines.extend(["", "## Precision coverage", "",
        "Counts require at least 10 top-1 predictions for a checkpoint-label pair. A label counts once if any eligible checkpoint meets the raw precision threshold.", "",
        "| Candidate set | Labels with ≥80% precision | Labels with ≥90% precision |", "|---|---:|---:|"])
    for key, title in (("new_checkpoints", "Additional checkpoints"), ("all_checkpoints", "Additional + UltraBERT A–D")):
        values = metrics["precision_threshold_coverage"][key]
        lines.append(f"| {title} | {values['precision_ge_80']['label_count']}/18 ({', '.join(values['precision_ge_80']['labels']) or 'none'}) | {values['precision_ge_90']['label_count']}/18 ({', '.join(values['precision_ge_90']['labels']) or 'none'}) |")
    lines.extend(["", "## Global ranked checkpoint–label rules", "",
        f"All {int(ranking.eligible.sum())} supported rules are listed in descending smoothed precision. The complete table is included here; unsupported pairs remain in `ranked_checkpoint_label_rules.csv` with `eligible=false` and no rank.", "",
        "| Rank | Key | Model | Dataset | Label | Precision | Smoothed precision | Recall | TP | FP | Predictions |", "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for _, row in ranking[ranking.eligible].sort_values("rank", kind="stable").iterrows():
        lines.append(f"| {int(row['rank'])} | {row['candidate_key']} | {row['model_title']} | {row['dataset']} | {row['label']} | {row['precision']:.1%} | {row['smoothed_precision']:.1%} | {row['recall']:.1%} | {int(row['true_positive'])} | {int(row['false_positive'])} | {int(row['prediction_count'])} |")
    lines.extend(["", "## Best precision checkpoint per label", "",
        "Chosen from all evaluation-eligible checkpoints by smoothed precision, then raw precision.", "",
        "| Label | Key | Model | Dataset | Precision (smoothed) | Recall | n |", "|---|---|---|---|---:|---:|---:|"])
    for label in metrics["classes"]:
        best = metrics["best_checkpoint_per_label"].get(label)
        if best:
            lines.append(f"| {label} | {best['candidate_key']} | {best['model_title']} | {best['dataset']} | {best['precision']:.1%} ({best['smoothed_precision']:.1%}) | {best['recall']:.1%} | {best['prediction_count']} |")
        else:
            lines.append(f"| {label} | — | — | — | — | — | — |")
    lines.extend(["", "## Best additional checkpoint vs UltraBERT", "",
        "The new-vs-UltraBERT comparison uses smoothed precision first, then raw precision. 'Outperforms' means the best additional checkpoint's smoothed precision exceeds the best supported UltraBERT A–D rule for that label.", "",
        "| Label | Best additional checkpoint | Precision (smoothed) | Recall | n | Best UltraBERT | Precision (smoothed) | n | New improves? |", "|---|---|---:|---:|---:|---|---:|---:|---|"])
    for label in metrics["classes"]:
        new = metrics["best_new_checkpoint_per_label"].get(label)
        old = metrics["best_ultrabert_checkpoint_per_label"].get(label)
        if new:
            new_name = f"{new['candidate_key']} {new['model_title']} ({new['dataset']})"
            npres, nsmooth, nrec, nsup = new["precision"], new["smoothed_precision"], new["recall"], new["prediction_count"]
        else:
            new_name, npres, nsmooth, nrec, nsup = "—", 0., 0., 0., 0
        if old:
            old_name = old["candidate_key"]
            opres, osmooth, osup = old["precision"], old["smoothed_precision"], old["prediction_count"]
        else:
            old_name, opres, osmooth, osup = "—", 0., 0., 0
        wins = bool(new and old and nsmooth > osmooth)
        lines.append(f"| {label} | {new_name} | {npres:.1%} ({nsmooth:.1%}) | {nrec:.1%} | {nsup} | {old_name} | {opres:.1%} ({osmooth:.1%}) | {osup} | {'yes' if wins else 'no'} |")
    lines.extend(["", "### Additional specialists exceeding UltraBERT precision", ""])
    if metrics["new_specialists_outperforming_ultrabert"]:
        lines.extend(["| Label | New specialist | Dataset | Precision (smoothed) | Recall | n | UltraBERT best | Precision (smoothed) |", "|---|---|---|---:|---:|---:|---|---:|"])
        for row in metrics["new_specialists_outperforming_ultrabert"]:
            lines.append(f"| {row['label']} | {row['candidate_key']} {row['model_title']} | {row['dataset']} | {row['precision']:.1%} ({row['smoothed_precision']:.1%}) | {row['recall']:.1%} | {row['prediction_count']} | {row['ultrabert_candidate_key']} | {row['ultrabert_precision']:.1%} ({row['ultrabert_smoothed_precision']:.1%}) |")
    else:
        lines.append("No additional checkpoint exceeded the strongest supported UltraBERT rule on smoothed precision.")
    cascade = metrics["zod01_cascade_comparison"]
    lines.extend(["", "## Cross-fitted cascade check", "",
        "As a direct sanity check, the original A–D cascade and the expanded candidate cascade were both fit and evaluated with identical five-fold prompt-group splits. Each fold fits rule order on four groups partitions and predicts the fifth.", "",
        "| Candidate set | Correct | Accuracy | Delta vs A–D | Corrected vs A–D | Regressed vs A–D |", "|---|---:|---:|---:|---:|---:|"])
    lines.append(f"| UltraBERT A–D | {cascade['ultrabert_A_to_D']['correct']}/540 | {cascade['ultrabert_A_to_D']['accuracy']:.1%} | — | — | — |")
    lines.append(f"| All evaluation-eligible checkpoints | {cascade['expanded']['correct']}/540 | {cascade['expanded']['accuracy']:.1%} | {cascade['delta_accuracy']:+.1%} | {cascade['corrected_vs_ultrabert']} | {cascade['regressed_vs_ultrabert']} |")
    for floor, result in cascade["thresholded_additions"].items():
        delta = result["accuracy"] - cascade["ultrabert_A_to_D"]["accuracy"]
        lines.append(f"| Add new rules ≥{float(floor):.0%} smoothed precision, then A–D | {result['correct']}/540 | {result['accuracy']:.1%} | {delta:+.1%} | {result['corrected_vs_ultrabert']} | {result['regressed_vs_ultrabert']} |")
    lines.extend(["", "The expanded cascade includes only evaluation-eligible candidates. The cascade check is an added grouped development diagnostic. Exact per-row predictions and selected checkpoint keys are in `cascade_comparison_oof.parquet`; rule traces are in `zod01_ultrabert_rule_trace.csv` and `zod01_expanded_rule_trace.csv`.", "",
        "## Artifacts and interpretation", "",
        "Reusable V0 probability files are stored in `artifacts/analysis/zod01_specialist_discovery/predictions/`. `checkpoint_inventory.csv` maps compact keys to exact experiment IDs, checkpoint paths, training dataset, selected epoch, prediction source and V0 accuracy. `ranked_checkpoint_label_rules.csv` contains every checkpoint-label pair, including unsupported pairs. `metrics.json` contains per-label best rules, threshold coverage and grouped cascade metrics.", "",
        "A high-precision rule may have low recall and only cover the subset of rows that its checkpoint predicts as that label. Precision estimates use 540 previously inspected/used validation labels; adding models can improve rule coverage, but these estimates can be optimistic and require confirmation on unseen examples.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
