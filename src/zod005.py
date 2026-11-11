"""Precision-ranked checkpoint cascade built from the V2 checkpoint trust list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


TRUST_LIST = Path("docs/trust-list-2-v001.csv")
TRUST_METADATA = Path("docs/trust-list-2-v001.json")
PREDICTIONS_DIR = Path("artifacts/analysis/trust_list_2_v0")
OUTPUT_DIR = Path("artifacts/analysis/zod005/v001")
SEED_CHECKPOINTS = ("N0086", "N0081", "N0072")
MINIMUM_PRECISION = 0.90
MINIMUM_SUPPORT = 30


def _metrics(y_true, prediction, classes):
    y = np.asarray(y_true, dtype=str)
    pred = np.asarray(prediction, dtype=str)
    matrix = confusion_matrix(y, pred, labels=list(classes))
    per_label = {}
    for i, label in enumerate(classes):
        tp = int(matrix[i, i])
        predicted = int(matrix[:, i].sum())
        actual = int(matrix[i, :].sum())
        per_label[label] = {
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
            "support": actual,
            "predicted": predicted,
            "correct": tp,
        }
    return {
        "rows": int(len(y)),
        "correct": int(np.sum(y == pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_label": per_label,
    }


def _sort_rules(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    result = rows.copy()
    result["precision"] = pd.to_numeric(result.precision, errors="raise")
    result["prediction_count"] = pd.to_numeric(result.prediction_count, errors="raise").astype(int)
    result["SeqID"] = pd.to_numeric(result.SeqID, errors="raise").astype(int)
    result = result.loc[(result.precision >= MINIMUM_PRECISION) &
                        (result.prediction_count >= MINIMUM_SUPPORT)]
    # SeqID is the trust-list's stable ordering for exact precision/support ties.
    return result.sort_values(["precision", "prediction_count", "SeqID"],
                              ascending=[False, False, True], kind="stable").reset_index(drop=True)


def _apply_rules(y_true, classes, top1_by_id, rules, fallback_id):
    y = np.asarray(y_true, dtype=str)
    output = np.empty(len(y), dtype=object)
    source = np.empty(len(y), dtype=object)
    assigned = np.zeros(len(y), dtype=bool)
    trace = []
    for order, row in enumerate(rules.to_dict("records"), start=1):
        candidate_id, label = str(row["experiment_id"]), str(row["target_label"])
        if candidate_id not in top1_by_id:
            raise KeyError(f"missing predictions for eligible rule checkpoint {candidate_id}")
        take = (~assigned) & (top1_by_id[candidate_id] == label)
        indices = np.flatnonzero(take)
        output[indices] = label
        source[indices] = candidate_id
        assigned[indices] = True
        correct = int(np.sum(y[indices] == label))
        trace.append({
            "order": order, "SeqID": int(row["SeqID"]), "sequence_id": str(row["sequence_id"]),
            "experiment_id": candidate_id, "model": str(row["model"]),
            "training_dataset": str(row["training_dataset"]), "label": label,
            "precision": float(row["precision"]), "prediction_count": int(row["prediction_count"]),
            "assigned": int(len(indices)), "correct": correct,
            "incorrect": int(len(indices) - correct),
            "incremental_precision": correct / len(indices) if len(indices) else None,
            "cumulative_rule_coverage": float(assigned.mean()), "remaining": int((~assigned).sum()),
        })
    fallback_rows = np.flatnonzero(~assigned)
    output[fallback_rows] = top1_by_id[fallback_id][fallback_rows]
    source[fallback_rows] = fallback_id
    fallback_correct = int(np.sum(output[fallback_rows] == y[fallback_rows]))
    return output.astype(str), source.astype(str), assigned, trace, {
        "assigned": int(len(fallback_rows)), "correct": fallback_correct,
        "incorrect": int(len(fallback_rows) - fallback_correct),
    }


def _load_v0(data_dir="data/processed"):
    from .pipeline import _load_dataset

    rows, _ = _load_dataset("v0", data_dir)
    if "model" not in rows or "id" not in rows:
        raise ValueError("V0 dataset must include id and model columns")
    frame = rows[["id", "model"]].copy()
    frame["id"] = frame.id.astype(str)
    frame["y_true"] = frame.pop("model").astype(str)
    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    if len(frame) != 2699 or frame.id.duplicated().any():
        raise ValueError(f"expected 2,699 unique full-V0 rows, got {len(frame)}")
    return frame


def _load_predictions(trust, validation, cache_dir=PREDICTIONS_DIR, *,
                      inferred_cache_dir=None, infer_missing=True):
    classes = sorted(validation.y_true.unique().tolist())
    top1 = {}
    cache_status = {}
    inferred_root = Path(inferred_cache_dir) if inferred_cache_dir else Path(cache_dir)
    inferred_root.mkdir(parents=True, exist_ok=True)
    for candidate_id in sorted(trust.experiment_id.astype(str).unique()):
        path = Path(cache_dir) / f"{candidate_id}.parquet"
        if not path.is_file():
            path = inferred_root / f"{candidate_id}.parquet"
        metadata_path = path.with_suffix(".json")
        if path.is_file():
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            cache_key = metadata.get("cache_key", {})
            if (cache_key.get("experiment_id") not in (None, candidate_id) or
                    cache_key.get("v0_sha256") not in (None, trust.validation_split_sha256.iloc[0]) or
                    int(metadata.get("rows", len(validation))) != len(validation)):
                raise ValueError(f"stale or incompatible prediction cache for {candidate_id}: {path}")
            frame = pd.read_parquet(path)
            if not {"id", "y_true"}.issubset(frame.columns):
                raise ValueError(f"prediction cache is missing row identity columns: {path}")
            frame["id"] = frame.id.astype(str)
            if frame.id.duplicated().any() or set(frame.id) != set(validation.id):
                raise ValueError(f"prediction cache IDs do not match full V0: {candidate_id}")
            frame = frame.set_index("id").loc[validation.id]
            if not np.array_equal(frame.y_true.astype(str).to_numpy(), validation.y_true.to_numpy()):
                raise ValueError(f"prediction cache label alignment mismatch: {candidate_id}")
            values = frame.get("y_pred")
            if values is None:
                columns = [f"prob::{label}" for label in classes]
                if not set(columns).issubset(frame.columns):
                    raise ValueError(f"prediction cache has no top-1 or complete probability columns: {candidate_id}")
                values = np.asarray(classes, dtype=object)[frame[columns].to_numpy(dtype=float).argmax(axis=1)]
            top1[candidate_id] = np.asarray(values, dtype=str)
            cache_status[candidate_id] = ("inferred_and_cached"
                if cache_key.get("inferred_for_zod005") else "reused")
            continue
        if not infer_missing:
            cache_status[candidate_id] = "missing"
            continue
        row = trust.loc[trust.experiment_id.astype(str).eq(candidate_id)].iloc[0]
        prediction = _infer_checkpoint(row, validation)
        top1[candidate_id] = prediction
        cache_status[candidate_id] = "inferred_and_cached"
        out = pd.DataFrame({"id": validation.id, "y_true": validation.y_true, "y_pred": prediction})
        inferred_path = inferred_root / f"{candidate_id}.parquet"
        inferred_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(inferred_path, index=False)
        cache_key = {
            "experiment_id": candidate_id,
            "v0_sha256": trust.validation_split_sha256.iloc[0],
            "rows": len(validation), "inferred_for_zod005": True,
        }
        inferred_path.with_suffix(".json").write_text(
            json.dumps({"cache_key": cache_key, "rows": len(validation)}, indent=2) + "\n")
    return top1, cache_status, classes


def _infer_checkpoint(trust_row, validation):
    """Infer a missing V0 cache using the checkpoint manifest and NLM predictors."""
    import gc
    from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle

    model = str(trust_row.model)
    checkpoint = Path(str(trust_row.checkpoint_path))
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint unavailable for missing V0 cache: {checkpoint}")
    texts = validation.text.astype(str).tolist() if "text" in validation else None
    if texts is None:
        from .pipeline import _load_dataset
        rows, _ = _load_dataset("v0", "data/processed")
        data = rows.set_index(rows.id.astype(str)).loc[validation.id]
        texts = data.text.fillna("").astype(str).tolist()
    classes = sorted(validation.y_true.unique().tolist())
    manifest_path = Path("artifacts/predictions") / f"{trust_row.experiment_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    config = manifest.get("config", {})
    if model == "svc":
        import joblib
        probabilities, model_classes = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probabilities, model_classes = predict_catboost_stylometry(checkpoint, texts, classes)
    else:
        probabilities, model_classes = _predict_neural(
            model, checkpoint, texts, classes,
            batch_size=int(config.get("batch_size", 8) or 8),
            max_length=int(config.get("max_length", 384) or 384),
            base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision"),
            text_only_backbone=bool(config.get("text_only_backbone", False)),
        )
    class_order = list(map(str, model_classes))
    ordered = np.asarray(probabilities)[:, [class_order.index(label) for label in classes]]
    prediction = np.asarray(classes, dtype=object)[ordered.argmax(axis=1)].astype(str)
    del probabilities
    gc.collect()
    return prediction


def _compare_candidate(y_true, classes, current_prediction, current_source, trial_prediction,
                       trial_source, candidate_id, rules, trial_coverage, current_coverage,
                       fallback_id):
    y = np.asarray(y_true, dtype=str)
    was_correct = current_prediction == y
    now_correct = trial_prediction == y
    claimed = trial_source == candidate_id
    changed = current_prediction != trial_prediction
    before_metrics = _metrics(y, current_prediction, classes)
    after_metrics = _metrics(y, trial_prediction, classes)
    per_label_changes = {
        label: {
            "precision_before": before_metrics["per_label"][label]["precision"],
            "precision_after": after_metrics["per_label"][label]["precision"],
            "precision_delta": after_metrics["per_label"][label]["precision"] - before_metrics["per_label"][label]["precision"],
            "recall_before": before_metrics["per_label"][label]["recall"],
            "recall_after": after_metrics["per_label"][label]["recall"],
            "recall_delta": after_metrics["per_label"][label]["recall"] - before_metrics["per_label"][label]["recall"],
        } for label in classes
    }
    fallback_before = int(np.sum(current_source == fallback_id))
    fallback_after = int(np.sum(trial_source == fallback_id))
    return {
        "candidate_claimed_rows": int(claimed.sum()),
        "candidate_claimed_correct_rows": int(np.sum(claimed & now_correct)),
        "candidate_claimed_incorrect_rows": int(np.sum(claimed & ~now_correct)),
        "incremental_precision": float(now_correct[claimed].mean()) if claimed.any() else None,
        "prediction_changed_rows": int(changed.sum()),
        "ownership_changed_rows": int(np.sum(current_source != trial_source)),
        "additional_correct_classifications": int(np.sum(~was_correct & now_correct)),
        "additional_incorrect_classifications": int(np.sum(was_correct & ~now_correct)),
        "metrics": after_metrics,
        "coverage": float(trial_coverage.mean()),
        "coverage_before": float(current_coverage.mean()),
        "coverage_delta": float(trial_coverage.mean() - current_coverage.mean()),
        "fallback_rows_before": fallback_before,
        "fallback_rows": fallback_after,
        "fallback_rows_delta": fallback_after - fallback_before,
        "candidate_rule_count": int(len(rules)),
        "per_label_changes": per_label_changes,
    }


def evaluate_cascade(*, data_dir="data/processed", trust_list_path=TRUST_LIST,
                     trust_metadata_path=TRUST_METADATA, predictions_dir=PREDICTIONS_DIR,
                     output_dir=OUTPUT_DIR, infer_missing=True):
    validation = _load_v0(data_dir)
    trust = pd.read_csv(trust_list_path, dtype=str, keep_default_na=False)
    required = {"SeqID", "sequence_id", "experiment_id", "model", "target_label",
                "training_dataset", "precision", "prediction_count", "balanced_accuracy",
                "overall_accuracy", "checkpoint_path", "checkpoint_available",
                "evaluation_eligible", "validation_training_overlap", "validation_split_sha256"}
    if not required.issubset(trust.columns):
        raise ValueError(f"trust list is missing columns: {sorted(required - set(trust.columns))}")
    trust = trust.loc[trust.validation_split_sha256.astype(str).eq(trust.validation_split_sha256.iloc[0])].copy()
    meta = json.loads(Path(trust_metadata_path).read_text())
    if (meta.get("version") != "trust-list-2-v001" or int(meta.get("validation_rows", 0)) != len(validation)
            or meta.get("validation_sha256") != str(trust.validation_split_sha256.iloc[0])):
        raise ValueError("V2 trust-list metadata does not match the full V0 dataset")
    if not trust.evaluation_eligible.astype(str).str.lower().isin({"true", "1"}).all():
        trust = trust.loc[trust.evaluation_eligible.astype(str).str.lower().isin({"true", "1"})].copy()
    trust["validation_training_overlap"] = pd.to_numeric(trust.validation_training_overlap, errors="raise")
    trust = trust.loc[trust.validation_training_overlap.eq(0)].copy()
    trust["precision"] = pd.to_numeric(trust.precision, errors="raise")
    trust["prediction_count"] = pd.to_numeric(trust.prediction_count, errors="raise")
    trust["balanced_accuracy"] = pd.to_numeric(trust.balanced_accuracy, errors="raise")
    trust["overall_accuracy"] = pd.to_numeric(trust.overall_accuracy, errors="raise")
    trust["checkpoint_available"] = trust.checkpoint_available.astype(str).str.lower().isin({"true", "1"})
    available = trust.groupby("experiment_id", sort=False).checkpoint_available.first()
    trust = trust.loc[trust.experiment_id.isin(available[available].index)].copy()
    top1, cache_status, classes = _load_predictions(trust, validation, Path(predictions_dir),
        inferred_cache_dir=Path(output_dir) / "inferred_validation_predictions",
        infer_missing=infer_missing)
    checkpoint_rows = trust.drop_duplicates("experiment_id").copy()
    checkpoint_rows = checkpoint_rows.loc[checkpoint_rows.experiment_id.isin(top1)]
    checkpoint_rows["balanced_accuracy"] = pd.to_numeric(checkpoint_rows.balanced_accuracy)
    checkpoint_rows["overall_accuracy"] = pd.to_numeric(checkpoint_rows.overall_accuracy)
    checkpoint_rows["_seq_num"] = checkpoint_rows.sequence_id.str.extract(r"N(\d+)", expand=False).astype(int)
    fallback_row = checkpoint_rows.sort_values(
        ["balanced_accuracy", "overall_accuracy", "_seq_num"],
        ascending=[False, False, True], kind="stable").iloc[0]
    fallback_id = str(fallback_row.experiment_id)

    eligible_rules = _sort_rules(trust)
    by_sequence = {str(row.sequence_id): str(row.experiment_id)
                   for row in checkpoint_rows.itertuples(index=False)}
    missing_seeds = set(SEED_CHECKPOINTS) - set(by_sequence)
    if missing_seeds:
        raise ValueError(f"initial ZOD005 checkpoints missing from eligible V2 roster: {sorted(missing_seeds)}")
    selected_sequences = list(SEED_CHECKPOINTS)
    selected_ids = {by_sequence[s] for s in selected_sequences}

    def materialize(ids):
        subset = eligible_rules.loc[eligible_rules.experiment_id.astype(str).isin(ids)]
        return _sort_rules(subset)

    current_rules = materialize(selected_ids)
    current_prediction, current_source, current_coverage, current_trace, current_fallback = _apply_rules(
        validation.y_true.to_numpy(), classes, top1, current_rules, fallback_id)
    current_metrics = _metrics(validation.y_true, current_prediction, classes)
    all_candidates = sorted(set(checkpoint_rows.experiment_id.astype(str)) - selected_ids)
    attempts = []
    accepted_order = []
    iteration = 0
    while all_candidates:
        iteration += 1
        trial_results = []
        for candidate_id in all_candidates:
            candidate_sequences = checkpoint_rows.loc[
                checkpoint_rows.experiment_id.astype(str).eq(candidate_id), "sequence_id"].astype(str).tolist()
            candidate_rules = materialize(selected_ids | {candidate_id})
            pred, source, coverage, trace, fallback = _apply_rules(
                validation.y_true.to_numpy(), classes, top1, candidate_rules, fallback_id)
            metrics = _metrics(validation.y_true, pred, classes)
            delta_key = (metrics["balanced_accuracy"], metrics["accuracy"])
            before_key = (current_metrics["balanced_accuracy"], current_metrics["accuracy"])
            improves = delta_key > before_key
            contribution = _compare_candidate(validation.y_true, classes, current_prediction,
                current_source, pred, source, candidate_id, eligible_rules.loc[
                    eligible_rules.experiment_id.astype(str).eq(candidate_id)], coverage,
                    current_coverage, fallback_id)
            attempt = {
                "iteration": iteration, "candidate_id": candidate_id,
                "sequence_ids": candidate_sequences, "decision": "candidate" if improves else "rejected",
                "decision_reason": "lexicographic_metric_improvement" if improves else "did_not_improve_balanced_accuracy_then_accuracy",
                "rules_after_reordering": int(len(candidate_rules)),
                "before_metrics": current_metrics, "contribution": contribution,
            }
            attempts.append(attempt)
            if improves:
                trial_results.append((delta_key, candidate_id, candidate_sequences,
                                      candidate_rules, pred, source, coverage, trace, fallback,
                                      metrics, attempt))
        if not trial_results:
            break
        trial_results.sort(key=lambda item: (item[0][0], item[0][1], item[1]), reverse=True)
        (_, candidate_id, candidate_sequences, current_rules, current_prediction,
         current_source, current_coverage, current_trace, current_fallback,
         current_metrics, chosen_attempt) = trial_results[0]
        for _, other_id, *_rest, other_attempt in trial_results[1:]:
            other_attempt["decision"] = "rejected"
            other_attempt["decision_reason"] = "lower_metric_gain_than_accepted_candidate"
        chosen_attempt["decision"] = "accepted"
        chosen_attempt["decision_reason"] = "best_available_balanced_accuracy_then_accuracy_gain"
        chosen_attempt["after_metrics"] = current_metrics
        chosen_attempt["cumulative_selected_sequence_ids"] = [*selected_sequences, *candidate_sequences]
        selected_ids.add(candidate_id)
        selected_sequences.extend(s for s in candidate_sequences if s not in selected_sequences)
        accepted_order.append(candidate_id)
        all_candidates.remove(candidate_id)

    # Re-run the final state to ensure saved artifacts match the selected manifest.
    current_rules = materialize(selected_ids)
    final_prediction, final_source, final_coverage, final_trace, fallback_metrics = _apply_rules(
        validation.y_true.to_numpy(), classes, top1, current_rules, fallback_id)
    final_metrics = _metrics(validation.y_true, final_prediction, classes)
    base_metrics = _metrics(validation.y_true,
                             top1[fallback_id], classes)
    rules_accepted = set(selected_ids)
    final_rules_records = current_rules.to_dict("records")
    configuration = {
        "name": "zod005", "version": "v001", "algorithm": "precision_ranked_checkpoint_cascade",
        "trust_list_version": meta["version"], "trust_list_generated_at_utc": meta["generated_at_utc"],
        "validation_scope": meta["validation_scope"], "validation_rows": len(validation),
        "validation_sha256": meta["validation_sha256"],
        "rule_thresholds": {"precision": MINIMUM_PRECISION, "minimum_support": MINIMUM_SUPPORT},
        "seed_sequences": list(SEED_CHECKPOINTS),
        "selected_sequences": selected_sequences,
        "selected_checkpoints": checkpoint_rows.loc[checkpoint_rows.experiment_id.astype(str).isin(rules_accepted),
            ["sequence_id", "experiment_id", "model", "training_dataset", "checkpoint_path"]].drop_duplicates().to_dict("records"),
        "ordered_rules": final_rules_records,
        "fallback": {"sequence_id": str(fallback_row.sequence_id), "experiment_id": fallback_id,
            "model": str(fallback_row.model), "training_dataset": str(fallback_row.training_dataset),
            "checkpoint_path": str(fallback_row.checkpoint_path),
            "validation_balanced_accuracy": float(fallback_row.balanced_accuracy),
            "validation_accuracy": float(fallback_row.overall_accuracy)},
        "metrics": final_metrics,
        "cache_status": cache_status,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "cascade.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (root / "checkpoint_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    pd.DataFrame(final_trace).to_csv(root / "rule_trace.csv", index=False)
    decisions = validation[["id", "y_true"]].copy()
    decisions["prediction"] = final_prediction
    decisions["correct"] = final_prediction == validation.y_true.to_numpy()
    decisions["selected_experiment_id"] = final_source
    decisions["used_fallback"] = final_source == fallback_id
    decisions.to_parquet(root / "validation_predictions.parquet", index=False)
    summary = {
        "rows": len(validation), "correct": final_metrics["correct"],
        "accuracy": final_metrics["accuracy"], "balanced_accuracy": final_metrics["balanced_accuracy"],
        "rule_coverage": float(final_coverage.mean()), "fallback_rows": fallback_metrics["assigned"],
        "fallback_correct": fallback_metrics["correct"], "fallback_incorrect": fallback_metrics["incorrect"],
        "fallback_candidate_id": fallback_id, "fallback_alone": base_metrics,
        "accepted_additions": accepted_order, "attempt_count": len(attempts),
        "trust_list_version": meta["version"], "validation_sha256": meta["validation_sha256"],
        "seed_metrics": _metrics(validation.y_true, _apply_rules(
            validation.y_true.to_numpy(), classes, top1, materialize({by_sequence[s] for s in SEED_CHECKPOINTS}),
            fallback_id)[0], classes),
    }
    (root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "evaluation_report.md").write_text(_render_report(summary, attempts, final_metrics))
    comparisons = compare_existing_ensembles(decisions, data_dir=data_dir)
    (root / "local_validation_comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (root / "local_validation_report.md").write_text(_render_comparison_report(summary, comparisons))
    return {"configuration": configuration, "summary": summary, "attempts": attempts,
            "validation": decisions, "classes": classes}


def _render_report(summary, attempts, final_metrics):
    lines = ["# ZOD005 V2 Trust-List Cascade", "",
        f"Validation: {summary['rows']} full-V0 rows; trust-list `{summary['trust_list_version']}`; "
        f"SHA-256 `{summary['validation_sha256']}`.", "",
        f"Final metrics: accuracy {final_metrics['accuracy']:.2%}; balanced accuracy "
        f"{final_metrics['balanced_accuracy']:.2%}; rule coverage {summary['rule_coverage']:.2%}; "
        f"fallback rows {summary['fallback_rows']}/{summary['rows']}.", "",
        f"Fallback: `{summary['fallback_candidate_id']}` ({summary['fallback_alone']['balanced_accuracy']:.2%} "
        "balanced accuracy alone).", "",
        "## Candidate attempts", "",
        "| Iteration | Checkpoint | Decision | Rules | Claimed | Correct | Incorrect | Incremental precision | Δ accuracy | Δ balanced accuracy |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for attempt in attempts:
        c = attempt["contribution"]
        before = attempt["before_metrics"]
        after = attempt.get("after_metrics", c["metrics"])
        da = after["accuracy"] - before["accuracy"]
        db = after["balanced_accuracy"] - before["balanced_accuracy"]
        ip = c["incremental_precision"]
        lines.append(f"| {attempt['iteration']} | `{attempt['candidate_id']}` | {attempt['decision']} | "
            f"{attempt['rules_after_reordering']} | {c['candidate_claimed_rows']} | "
            f"{c['candidate_claimed_correct_rows']} | {c['candidate_claimed_incorrect_rows']} | "
            f"{'—' if ip is None else f'{ip:.2%}'} | {da:+.2%} | {db:+.2%} |")
    lines.extend(["", "The candidate rule set is ranked again as a whole for every insertion. Candidate-level "
        "claimed-row precision and total cascade metric changes are both computed from aligned row-level predictions.", ""])
    return "\n".join(lines)


def compare_existing_ensembles(zod005_predictions, *, data_dir="data/processed"):
    """Compare saved ZOD outputs on the established 540-row V0 validation slice."""
    from .pipeline import _load_dataset

    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.astype(str).eq("validation"), ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["y_true"] = validation.pop("model").astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540:
        raise ValueError(f"expected the frozen V0 validation slice (540 rows), got {len(validation)}")
    classes = sorted(validation.y_true.unique().tolist())
    z005 = zod005_predictions.set_index("id").loc[validation.id]
    output = [{"ensemble": "zod005", "scope": "frozen V0 validation (subset of full V0)",
               **_metrics(validation.y_true, z005.prediction.astype(str), classes)}]
    artifacts = Path("artifacts/analysis")
    for ensemble, path, column in [
        ("zod01", artifacts / "zod01_18label_cascade/predictions.parquet", "prediction"),
        ("zod02", artifacts / "zod02_precision_ranked/predictions_full_v0.parquet", "zod02_prediction"),
    ]:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if not {"id", column}.issubset(frame.columns):
            continue
        frame["id"] = frame.id.astype(str)
        if set(frame.id) != set(validation.id):
            continue
        frame = frame.set_index("id").loc[validation.id]
        output.append({"ensemble": ensemble, "scope": "frozen V0 validation (540 rows)",
                       **_metrics(validation.y_true, frame[column].astype(str), classes)})

    for ensemble, rule_path, manifest_path in [
        ("zod03", artifacts / "zod03_110/rule_trace.csv", None),
        ("zod04", None, Path("artifacts/kaggle/zod04-models/zod04_manifest.json")),
    ]:
        try:
            if ensemble == "zod03":
                trace = pd.read_csv(rule_path)
                fallback_id = str(trace.loc[trace["rank"].astype(str).eq("Fallback"), "candidate_id"].iloc[0])
                rules = trace.loc[~trace["rank"].astype(str).eq("Fallback") &
                                  ~trace["skipped"].astype(bool)].sort_values("rank", kind="stable")
            else:
                bundle = json.loads(manifest_path.read_text())
                rules = pd.DataFrame(bundle["rules"]).sort_values("order", kind="stable")
                fallback_id = str(bundle["fallback_candidate_id"])
            ids = [str(x) for x in rules.candidate_id] + [fallback_id]
            top1 = {}
            for candidate_id in sorted(set(ids)):
                cache = Path("artifacts/analysis/zod01_specialist_discovery/predictions") / f"{candidate_id}_validation.parquet"
                if not cache.is_file():
                    cache = Path("artifacts/predictions") / f"{candidate_id}_validation.parquet"
                if not cache.is_file():
                    raise FileNotFoundError(str(cache))
                top1[candidate_id] = _load_validation_top1(cache, validation, classes, candidate_id)
            assigned = np.zeros(len(validation), dtype=bool)
            prediction = np.empty(len(validation), dtype=object)
            for row in rules.to_dict("records"):
                candidate_id, label = str(row["candidate_id"]), str(row["label"])
                take = (~assigned) & (top1[candidate_id] == label)
                prediction[take], assigned[take] = label, True
            prediction[~assigned] = top1[fallback_id][~assigned]
            output.append({"ensemble": ensemble,
                "scope": "frozen V0 validation (replayed from saved rules/caches)",
                **_metrics(validation.y_true, prediction.astype(str), classes)})
        except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
            output.append({"ensemble": ensemble, "scope": "unavailable for aligned replay",
                           "reason": str(exc)})
    return output


def _load_validation_top1(path, validation, classes, candidate_id):
    frame = pd.read_parquet(path)
    if not {"id", "y_true", "prompt_id"}.issubset(frame.columns):
        raise ValueError(f"validation cache identity fields missing for {candidate_id}")
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any() or set(frame.id) != set(validation.id):
        raise ValueError(f"validation cache IDs mismatch for {candidate_id}")
    frame = frame.set_index("id").loc[validation.id]
    if not np.array_equal(frame.y_true.astype(str).to_numpy(), validation.y_true.to_numpy()):
        raise ValueError(f"validation cache labels mismatch for {candidate_id}")
    if not np.array_equal(frame.prompt_id.astype(str).to_numpy(), validation.prompt_id.to_numpy()):
        raise ValueError(f"validation cache prompts mismatch for {candidate_id}")
    columns = [f"prob::{label}" for label in classes]
    if set(columns).issubset(frame.columns):
        return np.asarray(classes, dtype=object)[frame[columns].to_numpy(dtype=float).argmax(axis=1)].astype(str)
    if "y_pred" in frame:
        return frame.y_pred.astype(str).to_numpy()
    raise ValueError(f"prediction fields missing for {candidate_id}")


def _render_comparison_report(summary, comparisons):
    lines = ["# ZOD005 Local Validation Comparison", "",
        f"Primary validation: all {summary['rows']} V0 responses, trust-list `{summary['trust_list_version']}` "
        f"(SHA-256 `{summary['validation_sha256']}`). ZOD005 scored {summary['correct']}/{summary['rows']} "
        f"({summary['accuracy']:.2%} accuracy; {summary['balanced_accuracy']:.2%} balanced accuracy).", "",
        "Existing ensembles are compared on the frozen 540-row V0 validation slice when aligned saved "
        "predictions or rule caches are available.", "",
        "| Ensemble | Scope | Rows | Accuracy | Balanced accuracy |", "|---|---|---:|---:|---:|"]
    for item in comparisons:
        if "accuracy" not in item:
            lines.append(f"| {item['ensemble']} | {item['scope']}: {item.get('reason', 'unavailable')} | — | — | — |")
        else:
            lines.append(f"| {item['ensemble']} | {item['scope']} | {item['rows']} | "
                         f"{item['accuracy']:.2%} | {item['balanced_accuracy']:.2%} |")
    return "\n".join(lines) + "\n"


def predict_test(*, config_path=OUTPUT_DIR / "cascade.json", test_path="data/raw/test.csv",
                 output_dir=OUTPUT_DIR / "test"):
    """Run selected checkpoint inference locally and write CSV plus decision statistics."""
    import gc
    from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
    import joblib

    config = json.loads(Path(config_path).read_text())
    test = pd.read_csv(test_path, dtype={"id": "string"})
    if not {"id", "text"}.issubset(test.columns):
        raise ValueError("competition test input must include id and text")
    if test.id.isna().any() or test.id.duplicated().any():
        raise ValueError("competition test IDs must be non-null and unique")
    texts = test.text.fillna("").astype(str).tolist()
    classes = sorted(pd.read_csv(TRUST_LIST, dtype=str).target_label.astype(str).unique().tolist())
    # Use cached test top-1 predictions only when the file is explicitly keyed to these test IDs.
    candidate_records = {str(item["experiment_id"]): item for item in config["selected_checkpoints"]}
    candidate_records[config["fallback"]["experiment_id"]] = config["fallback"]
    test_cache_dir = Path("artifacts/analysis/zod005/test_prediction_cache")
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    inference_stats = {}
    for candidate_id, item in candidate_records.items():
        cache_path = test_cache_dir / f"{candidate_id}.parquet"
        if cache_path.is_file():
            cache = pd.read_parquet(cache_path)
            if (cache.id.astype(str).tolist() == test.id.astype(str).tolist() and
                    cache.get("test_sha256", pd.Series([None])).iloc[0] == _ids_hash(test.id)):
                predictions[candidate_id] = cache.prediction.astype(str).to_numpy()
                inference_stats[candidate_id] = {"cache": "reused", "rows": len(cache)}
                continue
        checkpoint = Path(item["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"selected checkpoint is unavailable: {checkpoint}")
        model = str(item["model"])
        started = __import__("time").monotonic()
        if model == "svc":
            probs, model_classes = predict_svc_bundle(joblib.load(checkpoint), texts)
        elif model == "catboost_stylometry":
            probs, model_classes = predict_catboost_stylometry(checkpoint, texts, classes)
        else:
            manifest_path = Path("artifacts/predictions") / f"{candidate_id}_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            conf = manifest.get("config", {})
            probs, model_classes = _predict_neural(model, checkpoint, texts, classes,
                batch_size=int(conf.get("batch_size", 4) or 4),
                max_length=int(conf.get("max_length", 384) or 384),
                base_checkpoint=conf.get("checkpoint"), revision=manifest.get("model_revision"),
                text_only_backbone=bool(conf.get("text_only_backbone", False)))
        order = [list(map(str, model_classes)).index(label) for label in classes]
        pred = np.asarray(classes, dtype=object)[np.asarray(probs)[:, order].argmax(axis=1)].astype(str)
        predictions[candidate_id] = pred
        pd.DataFrame({"id": test.id.astype(str), "prediction": pred,
                      "test_sha256": _ids_hash(test.id)}).to_parquet(cache_path, index=False)
        inference_stats[candidate_id] = {"cache": "computed", "rows": len(test),
                                         "seconds": round(__import__("time").monotonic() - started, 3)}
        del probs
        gc.collect()

    n = len(test)
    assigned = np.zeros(n, dtype=bool)
    output = np.empty(n, dtype=object)
    source = np.empty(n, dtype=object)
    for order, rule in enumerate(config["ordered_rules"], start=1):
        candidate_id, label = str(rule["experiment_id"]), str(rule["target_label"])
        chosen = (~assigned) & (predictions[candidate_id] == label)
        output[chosen], source[chosen], assigned[chosen] = label, candidate_id, True
    fallback_id = config["fallback"]["experiment_id"]
    output[~assigned] = predictions[fallback_id][~assigned]
    source[~assigned] = fallback_id
    if not set(output.astype(str)).issubset(classes):
        raise ValueError("prediction contains labels outside the 18-label competition vocabulary")
    submission = pd.DataFrame({"id": test.id.astype(str), "model": output.astype(str)})
    if submission.columns.tolist() != ["id", "model"] or len(submission) != len(test):
        raise ValueError("submission schema or row count is invalid")
    if submission.id.tolist() != test.id.astype(str).tolist() or submission.id.duplicated().any():
        raise ValueError("submission IDs do not exactly preserve test order")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    submission.to_csv(root / "submission.csv", index=False)
    submission.to_parquet(root / "submission.parquet", index=False)
    pd.DataFrame({"id": test.id.astype(str), "prediction": output.astype(str),
                  "selected_experiment_id": source.astype(str), "used_fallback": source == fallback_id
                  }).to_parquet(root / "decisions.parquet", index=False)
    stats = {"rows": len(test), "rule_covered": int(assigned.sum()),
        "rule_coverage": float(assigned.mean()), "fallback_rows": int((~assigned).sum()),
        "fallback_rate": float((~assigned).mean()), "schema": submission.columns.tolist(),
        "test_ids_sha256": _ids_hash(test.id), "checkpoints": inference_stats,
        "prediction_cache_dir": str(test_cache_dir)}
    (root / "inference_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _ids_hash(values):
    import hashlib
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data-dir", default="data/processed")
    evaluate.add_argument("--trust-list", default=str(TRUST_LIST))
    evaluate.add_argument("--trust-metadata", default=str(TRUST_METADATA))
    evaluate.add_argument("--predictions-dir", default=str(PREDICTIONS_DIR))
    evaluate.add_argument("--output-dir", default=str(OUTPUT_DIR))
    predict = sub.add_parser("predict")
    predict.add_argument("--config", default=str(OUTPUT_DIR / "cascade.json"))
    predict.add_argument("--test", default="data/raw/test.csv")
    predict.add_argument("--output-dir", default=str(OUTPUT_DIR / "test"))
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_cascade(data_dir=args.data_dir, trust_list_path=args.trust_list,
            trust_metadata_path=args.trust_metadata, predictions_dir=args.predictions_dir,
            output_dir=args.output_dir)
        print(json.dumps(result["summary"], indent=2))
    else:
        print(json.dumps(predict_test(config_path=args.config, test_path=args.test,
                                      output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
