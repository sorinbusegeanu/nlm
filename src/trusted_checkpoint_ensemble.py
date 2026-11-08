"""Model-agnostic trusted checkpoint selection with prompt-group OOF evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

from .checkpoint_comparison import _validated_probabilities, checkpoint_metrics
from .ensemble_factory import ensemble_probabilities
from .prediction_factory import _align_candidates, _read_candidates, _reliability_table


ULTRABERT_CHECKPOINTS = {
    "A": "ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0",
    "B": "ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2",
    "C": "ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0",
    "D": "ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff",
}
DEFAULT_MINIMUM_SUPPORT = 10
DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0


def _class_stats(y_true, predictions, classes):
    y_true = np.asarray(y_true, dtype=str)
    predictions = np.asarray(predictions, dtype=str)
    output = {}
    for label in classes:
        tp = int(np.sum((y_true == label) & (predictions == label)))
        fp = int(np.sum((y_true != label) & (predictions == label)))
        fn = int(np.sum((y_true == label) & (predictions != label)))
        tn = int(len(y_true) - tp - fp - fn)
        support = tp + fn
        predicted_count = tp + fp
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[label] = {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                         "true_negative": tn, "support": support,
                         "predicted_count": predicted_count, "precision": float(precision),
                         "recall": float(recall), "f1": float(f1)}
    return output


def _hard_metrics(y_true, predictions, classes):
    y_true = np.asarray(y_true, dtype=str)
    predictions = np.asarray(predictions, dtype=str)
    per_class = _class_stats(y_true, predictions, classes)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=classes, average="macro", zero_division=0)),
        "total_correct": int(np.sum(y_true == predictions)),
        "total_rows": int(len(y_true)),
        "per_class": per_class,
    }


def _candidate_predictions(candidate, classes):
    probabilities = np.asarray(candidate["probabilities"], dtype=np.float64)
    return np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]


def _probability_summary(y_true, probabilities, classes):
    y = np.asarray(y_true, dtype=str)
    probs = _validated_probabilities(probabilities, len(y), len(classes), "probability summary")
    truth_index = np.asarray([list(classes).index(label) for label in y], dtype=int)
    one_hot = np.eye(len(classes), dtype=float)[truth_index]
    return {"accuracy": float(np.mean(np.asarray(classes, dtype=object)[probs.argmax(axis=1)] == y)),
            "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))}


def select_specialists(candidates, classes, reliability_rows, *, candidate_order=None):
    """Assign each class to its highest-recall checkpoint with deterministic ties."""
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    order_index = {candidate_id: i for i, candidate_id in enumerate(order)}
    lookup = {(row["candidate_id"], row["predicted_nlm"]): row for row in reliability_rows}
    specialists, detail = {}, {}
    for label in classes:
        candidates_for_label = []
        for candidate in candidates:
            row = lookup[(candidate["candidate_id"], label)]
            tp, fp, fn = int(row["true_positive"]), int(row["false_positive"]), int(row["false_negative"])
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            candidates_for_label.append((candidate["candidate_id"], recall, precision, f1, row))
        winner = min(candidates_for_label, key=lambda item: (-item[1], -item[2], -item[3],
                                                              order_index.get(item[0], len(order))))
        specialists[label] = winner[0]
        detail[label] = {"specialist_candidate_id": winner[0], "recall": winner[1],
                         "precision": winner[2], "f1": winner[3],
                         "predicted_count": int(winner[4]["predicted_count"]),
                         "true_positive": int(winner[4]["true_positive"]),
                         "false_positive": int(winner[4]["false_positive"]),
                         "false_negative": int(winner[4]["false_negative"]),
                         "all_candidate_scores": [
                             {"candidate_id": cid, "recall": recall, "precision": precision, "f1": f1,
                              "predicted_count": int(row["predicted_count"])}
                             for cid, recall, precision, f1, row in candidates_for_label]}
    return specialists, detail


def route_candidates(candidates, classes, reliability_rows, specialists, *,
                     minimum_support=DEFAULT_MINIMUM_SUPPORT, prior_alpha=DEFAULT_PRIOR_ALPHA,
                     prior_beta=DEFAULT_PRIOR_BETA, candidate_order=None):
    """Route one row of model-agnostic candidate probabilities to one hard label.

    Priority: (1) proposal is made by that label's assigned specialist, (2) Beta
    smoothed precision, (3) assigned specialist's recall, then input candidate
    order and canonical class order. If every proposal lacks minimum support,
    checkpoint D (or the last candidate when no D ID exists) supplies the label.
    """
    if not candidates:
        raise ValueError("trusted routing needs at least one candidate")
    classes = list(map(str, classes))
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    order_index = {candidate_id: i for i, candidate_id in enumerate(order)}
    class_index = {label: i for i, label in enumerate(classes)}
    reliability = {(row["candidate_id"], row["predicted_nlm"]): row for row in reliability_rows}
    proposals, eligible = [], []
    for candidate in candidates:
        probabilities = np.asarray(candidate["probabilities"], dtype=np.float64)
        if probabilities.shape != (1, len(classes)):
            raise ValueError("route_candidates expects a single-row probability matrix per candidate")
        index = int(probabilities[0].argmax())
        label = classes[index]
        row = reliability[(candidate["candidate_id"], label)]
        specialist_id = specialists[label]
        proposal = {
            "candidate_id": candidate["candidate_id"], "label": label,
            "confidence": float(probabilities[0, index]),
            "predicted_count": int(row["predicted_count"]),
            "smoothed_precision": float(row["smoothed_trust"] if row.get("smoothed_trust") is not None
                else prior_alpha / (prior_alpha + prior_beta)),
            "observed_precision": float(row["precision"] if row.get("precision") is not None else 0.0),
            "specialist_candidate_id": specialist_id,
            "specialist_recall": float(reliability[(specialist_id, label)]["true_positive"] /
                max(1, reliability[(specialist_id, label)]["actual_count"])),
            "is_assigned_specialist": candidate["candidate_id"] == specialist_id,
            "class_order": class_index[label], "candidate_order": order_index.get(candidate["candidate_id"], len(order)),
        }
        proposals.append(proposal)
        if proposal["predicted_count"] >= minimum_support:
            eligible.append(proposal)
    if not eligible:
        fallback_id = next((cid for cid in order if cid == "D"), candidates[-1]["candidate_id"])
        fallback = next((item for item in candidates if item["candidate_id"] == fallback_id), candidates[-1])
        probabilities = np.asarray(fallback["probabilities"], dtype=float)
        label = classes[int(probabilities[0].argmax())]
        proposal = next(item for item in proposals if item["candidate_id"] == fallback["candidate_id"])
        chosen = {**proposal, "fallback": True, "selection_reason": "fallback_to_D_no_proposal_met_minimum_support"}
    else:
        chosen = min(eligible, key=lambda item: (-int(item["is_assigned_specialist"]),
            -item["smoothed_precision"], -item["specialist_recall"],
            item["candidate_order"], item["class_order"]))
        reason = ("assigned_specialist_priority" if chosen["is_assigned_specialist"] else
                  "smoothed_precision_then_specialist_recall")
        chosen = {**chosen, "fallback": False, "selection_reason": reason}
    return chosen, proposals


def _individual_oof_predictions(probabilities, classes):
    return {key: _candidate_predictions(candidate, classes)
            for key, candidate in probabilities.items()}


def run_trusted_checkpoint_ensemble(*, candidates, y_true, groups, classes,
                                    output_dir="artifacts/analysis/trusted_checkpoint_ensemble",
                                    candidate_order=None, minimum_support=DEFAULT_MINIMUM_SUPPORT,
                                    prior_alpha=DEFAULT_PRIOR_ALPHA, prior_beta=DEFAULT_PRIOR_BETA,
                                    folds=5, validation_split_sha256=None, validation_ids=None,
                                    report_path="docs/trusted_checkpoint_ensemble_ultrabert.md"):
    """Cross-fit specialist assignments and trusted routing by prompt group.

    Candidate contract: candidate_id, classes, probabilities [rows, classes],
    and aligned validation row IDs/prompt IDs in `frame` (or explicit `ids` and
    `prompt_ids`). This routine is architecture-agnostic.
    """
    if not candidates:
        raise ValueError("no candidate checkpoints supplied")
    classes = list(map(str, classes))
    y = np.asarray(y_true, dtype=str)
    groups = np.asarray(groups, dtype=str)
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    by_id = {item["candidate_id"]: item for item in candidates}
    if len(by_id) != len(candidates) or set(order) != set(by_id):
        raise ValueError("candidate order must contain each unique candidate ID exactly once")
    candidates = [by_id[candidate_id] for candidate_id in order]
    for candidate in candidates:
        if list(map(str, candidate.get("classes", classes))) != classes:
            raise ValueError(f"{candidate['candidate_id']} class order differs from canonical order")
        _validated_probabilities(candidate["probabilities"], len(y), len(classes), candidate["candidate_id"])
        if "ids" in candidate and validation_ids is not None and list(map(str, candidate["ids"])) != list(map(str, validation_ids)):
            raise ValueError(f"{candidate['candidate_id']} validation IDs are not aligned")
        if "prompt_ids" in candidate and not np.array_equal(np.asarray(candidate["prompt_ids"], dtype=str), groups):
            raise ValueError(f"{candidate['candidate_id']} prompt IDs are not aligned")
    if not (len(y) == len(groups)) or len(set(y)) != len(classes):
        raise ValueError("truth, prompt groups, and classes are inconsistent")
    n_folds = int(folds)
    if len(np.unique(groups)) < n_folds:
        raise ValueError("not enough prompt groups for requested GroupKFold")
    model_predictions = {item["candidate_id"]: _candidate_predictions(item, classes) for item in candidates}
    oof_trusted = np.empty(len(y), dtype=object)
    oof_selected = np.empty(len(y), dtype=object)
    oof_reason = np.empty(len(y), dtype=object)
    oof_specialist = np.empty(len(y), dtype=object)
    oof_smoothed_precision = np.zeros(len(y), dtype=float)
    oof_specialist_recall = np.zeros(len(y), dtype=float)
    oof_fallback = np.zeros(len(y), dtype=bool)
    oof_competing_count = np.zeros(len(y), dtype=int)
    oof_specialist_only = {label: np.empty(len(y), dtype=object) for label in classes}
    route_rows, specialist_assignments, fold_records = [], [], []
    reliability_by_fold = {}
    outer = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, held_idx) in enumerate(outer.split(np.zeros(len(y)), y, groups), start=1):
        training_predictions = {
            candidate["candidate_id"]: {"labels": y[train_idx],
                "probabilities": np.asarray(candidate["probabilities"])[train_idx]}
            for candidate in candidates}
        reliability_rows = _reliability_table(candidates, training_predictions, classes,
            minimum_support=minimum_support, prior_alpha=prior_alpha, prior_beta=prior_beta)
        specialists, specialist_stats = select_specialists(candidates, classes, reliability_rows,
                                                            candidate_order=order)
        reliability_by_fold[str(fold)] = {
            "fit_rows": int(len(train_idx)), "held_out_rows": int(len(held_idx)),
            "fit_prompt_groups": sorted(set(groups[train_idx])),
            "held_out_prompt_groups": sorted(set(groups[held_idx])),
            "minimum_support": minimum_support, "prior_alpha": prior_alpha, "prior_beta": prior_beta,
            "reliability": reliability_rows, "specialist_assignments": specialist_stats}
        for label in classes:
            specialist_assignments.append({"fold": fold, "label": label,
                "specialist_candidate_id": specialists[label],
                **{key: value for key, value in specialist_stats[label].items()
                   if key not in {"all_candidate_scores", "specialist_candidate_id"}}})
            expert = by_id[specialists[label]]
            local = {**expert, "probabilities": np.asarray(expert["probabilities"])[held_idx]}
            oof_specialist_only[label][held_idx] = _candidate_predictions(local, classes)
        held_candidates = [{**candidate, "probabilities": np.asarray(candidate["probabilities"])[held_idx]}
                           for candidate in candidates]
        for local_row, global_row in enumerate(held_idx):
            one_row = [{**candidate, "probabilities": candidate["probabilities"][local_row:local_row+1]}
                       for candidate in held_candidates]
            chosen, proposals = route_candidates(one_row, classes, reliability_rows, specialists,
                minimum_support=minimum_support, prior_alpha=prior_alpha, prior_beta=prior_beta,
                candidate_order=order)
            oof_trusted[global_row] = chosen["label"]
            oof_selected[global_row] = chosen["candidate_id"]
            oof_reason[global_row] = chosen["selection_reason"]
            oof_specialist[global_row] = chosen["specialist_candidate_id"]
            oof_smoothed_precision[global_row] = chosen["smoothed_precision"]
            oof_specialist_recall[global_row] = chosen["specialist_recall"]
            oof_fallback[global_row] = bool(chosen["fallback"])
            distinct = sorted(set(item["label"] for item in proposals))
            oof_competing_count[global_row] = len(distinct)
            route_rows.append({"row_index": int(global_row), "fold": fold,
                "id": str(validation_ids[global_row]) if validation_ids is not None else str(global_row),
                "prompt_id": str(groups[global_row]), "y_true": str(y[global_row]),
                "predicted_label": chosen["label"], "selected_candidate_id": chosen["candidate_id"],
                "assigned_specialist_for_label": chosen["specialist_candidate_id"],
                "selection_reason": chosen["selection_reason"], "fallback": bool(chosen["fallback"]),
                "smoothed_precision": chosen["smoothed_precision"],
                "observed_precision": chosen["observed_precision"],
                "specialist_recall": chosen["specialist_recall"],
                "predicted_count_in_fit": chosen["predicted_count"],
                "distinct_proposal_labels": distinct,
                "proposals": proposals})
        fold_records.append({"fold": fold, "fit_rows": int(len(train_idx)), "held_out_rows": int(len(held_idx)),
            "fit_prompt_groups": int(len(set(groups[train_idx]))),
            "held_out_prompt_groups": int(len(set(groups[held_idx]))),
            "specialist_assignments": specialists})

    # Fold-assigned per-label specialist one-vs-rest outputs (not one joint classifier).
    specialist_oof_stats = {label: _class_stats(y, oof_specialist_only[label], [label])[label]
                            for label in classes}
    trusted_metrics = _hard_metrics(y, oof_trusted, classes)
    d_id = "D" if "D" in model_predictions else candidates[-1]["candidate_id"]
    d_predictions = model_predictions[d_id]
    d_metrics = _hard_metrics(y, d_predictions, classes)
    d_rank_metrics = checkpoint_metrics(y, by_id[d_id]["probabilities"], classes)
    for metric in ("top3_accuracy", "mrr", "log_loss"):
        d_metrics[metric] = d_rank_metrics[metric]
    d_metrics.update(_probability_summary(y, by_id[d_id]["probabilities"], classes))
    bd_id = {"B", "D"}
    if bd_id.issubset(model_predictions):
        b, d = by_id["B"], by_id["D"]
        bd_probs = ensemble_probabilities([b["probabilities"], d["probabilities"]])
        bd_metrics = checkpoint_metrics(y, bd_probs, classes)
        bd_predictions = np.asarray(classes, dtype=object)[bd_probs.argmax(axis=1)]
        bd_metrics.update({"macro_f1": _hard_metrics(y, bd_predictions, classes)["macro_f1"]})
        bd_metrics.update(_probability_summary(y, bd_probs, classes))
    else:
        bd_probs = None
        bd_metrics = None
        bd_predictions = None
    individual_metrics = {candidate_id: _hard_metrics(y, prediction, classes)
                          for candidate_id, prediction in model_predictions.items()}
    oracle_correct = np.logical_or.reduce([model_predictions[candidate_id] == y for candidate_id in order])
    oracle_metrics = {"accuracy": float(oracle_correct.mean()),
                      "total_correct": int(oracle_correct.sum()), "total_rows": int(len(y)),
                      "deployable": False, "definition": "any checkpoint top-1 prediction is correct with hindsight"}

    # Report-only pooled best individual candidates by recall/precision per label.
    # The live router never consults these pooled validation winners.
    pooled_candidate_label = {}
    for label in classes:
        values = []
        for candidate_id in order:
            stat = individual_metrics[candidate_id]["per_class"][label]
            values.append((candidate_id, stat))
        best_recall = min(values, key=lambda item: (-item[1]["recall"], -item[1]["precision"],
            -item[1]["f1"], order.index(item[0])))
        supported = [item for item in values if item[1]["predicted_count"] >= minimum_support]
        best_precision = min(supported or values, key=lambda item: (-item[1]["precision"],
            -item[1]["recall"], -item[1]["f1"], order.index(item[0])))
        pooled_candidate_label[label] = {"best_recall_candidate_id": best_recall[0], **best_recall[1],
            "best_precision_candidate_id": best_precision[0],
            "best_precision": best_precision[1]["precision"],
            "best_precision_recall": best_precision[1]["recall"],
            "best_precision_f1": best_precision[1]["f1"],
            "best_precision_predicted_count": best_precision[1]["predicted_count"]}

    # Most frequently learned expert for the coverage table; OOF values below are
    # from the fold-assigned specialist stream, not a full-validation fit.
    assignment_frame = pd.DataFrame(specialist_assignments)
    summary_assignments = {}
    for label in classes:
        counts = assignment_frame.loc[assignment_frame.label.eq(label), "specialist_candidate_id"].value_counts()
        majority = min(order, key=lambda candidate_id: (-int(counts.get(candidate_id, 0)), order.index(candidate_id)))
        pooled = pooled_candidate_label[label]
        best_recall_candidate = pooled["best_recall_candidate_id"]
        summary_assignments[label] = {"best_recall_candidate_id": best_recall_candidate,
            "best_recall_candidate_metrics": individual_metrics[best_recall_candidate]["per_class"][label],
            "majority_fold_specialist": majority,
            "fold_assignment_counts": {candidate_id: int(counts.get(candidate_id, 0)) for candidate_id in order},
            "specialist_stream_oof": specialist_oof_stats[label]}

    # Build high-accuracy coverage for baseline D, fold-assigned specialists and router.
    threshold_metrics = {}
    for method, stats in (("D", d_metrics["per_class"]),
                          ("best_individual_specialists", specialist_oof_stats),
                          ("trusted_checkpoint_ensemble", trusted_metrics["per_class"])):
        threshold_metrics[method] = {
            "recall_ge_80": sum(value["recall"] >= .80 for value in stats.values()),
            "recall_ge_90": sum(value["recall"] >= .90 for value in stats.values()),
            "precision_ge_80": sum(value["precision"] >= .80 for value in stats.values()),
            "precision_ge_90": sum(value["precision"] >= .90 for value in stats.values()),
            "both_ge_80": sum(value["precision"] >= .80 and value["recall"] >= .80 for value in stats.values()),
            "both_ge_90": sum(value["precision"] >= .90 and value["recall"] >= .90 for value in stats.values()),
        }
    pooled_any = {}
    for label in classes:
        stats = [individual_metrics[candidate_id]["per_class"][label] for candidate_id in order]
        pooled_any[label] = {
            "recall_ge_80": any(value["recall"] >= .80 for value in stats),
            "recall_ge_90": any(value["recall"] >= .90 for value in stats),
            "precision_ge_80": any(value["predicted_count"] >= minimum_support and value["precision"] >= .80 for value in stats),
            "precision_ge_90": any(value["predicted_count"] >= minimum_support and value["precision"] >= .90 for value in stats),
            "both_ge_80": any(value["predicted_count"] >= minimum_support and value["precision"] >= .80 and value["recall"] >= .80 for value in stats),
            "both_ge_90": any(value["predicted_count"] >= minimum_support and value["precision"] >= .90 and value["recall"] >= .90 for value in stats),
        }
    threshold_metrics["any_individual_checkpoint_pooled"] = {
        key: sum(value[key] for value in pooled_any.values())
        for key in next(iter(pooled_any.values()))
    }
    counts_by_selected = {}
    for candidate_id in order:
        mask = oof_selected == candidate_id
        counts_by_selected[candidate_id] = {"selected_rows": int(mask.sum()),
            "correct": int(np.sum(mask & (oof_trusted == y))),
            "incorrect": int(np.sum(mask & (oof_trusted != y))),
            "labels": {label: {"selected": int(np.sum(mask & (oof_trusted == label))),
                "correct": int(np.sum(mask & (oof_trusted == label) & (y == label))),
                "incorrect": int(np.sum(mask & (oof_trusted == label) & (y != label)))} for label in classes}}
    trusted_class_predictions = trusted_metrics["per_class"]
    movement = {"corrected_vs_D": int(np.sum((d_predictions != y) & (oof_trusted == y))),
                "regressed_vs_D": int(np.sum((d_predictions == y) & (oof_trusted != y))),
                "same_prediction_as_D": int(np.sum(oof_trusted == d_predictions)),
                "rows_with_multiple_distinct_proposals": int(np.sum(oof_competing_count > 1)),
                "mean_distinct_proposal_labels": float(np.mean(oof_competing_count)),
                "fallback_rows": int(oof_fallback.sum())}

    metrics = {
        "validation_rows": int(len(y)), "classes": classes, "prompt_groups": int(len(set(groups))),
        "folds": fold_records, "minimum_support": minimum_support,
        "precision_smoothing": {"prior_alpha": prior_alpha, "prior_beta": prior_beta,
                                 "formula": "(TP + alpha) / (predicted_count + alpha + beta)"},
        "baselines": {"D": d_metrics, "B_plus_D_equal_probability": bd_metrics,
                      "four_checkpoint_top1_oracle": oracle_metrics,
                      "individual_checkpoints": individual_metrics},
        "trusted_checkpoint_ensemble": trusted_metrics,
        "specialist_stream_oof_by_label": specialist_oof_stats,
        "specialist_coverage_matrix": summary_assignments,
        "pooled_individual_label_diagnostics": pooled_candidate_label,
        "high_accuracy_label_coverage": threshold_metrics,
        "routing_counts_by_selected_candidate": counts_by_selected,
        "routing_movement": movement,
        "specialist_correct_vs_D": int(np.sum((d_predictions != y) & (oof_trusted == y))),
        "specialist_regressed_vs_D": int(np.sum((d_predictions == y) & (oof_trusted != y))),
        "validation_split_sha256": validation_split_sha256,
        "validation_ids_sha256": hashlib.sha256("\n".join(map(str, validation_ids or range(len(y)))).encode()).hexdigest(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({"id": list(map(str, validation_ids)) if validation_ids is not None else list(map(str, range(len(y)))),
        "prompt_id": groups, "y_true": y, "fold": np.asarray([row["fold"] for row in route_rows], dtype=int),
        "trusted_prediction": oof_trusted, "selected_candidate_id": oof_selected,
        "assigned_specialist_candidate_id": oof_specialist, "selection_reason": oof_reason,
        "selected_smoothed_precision": oof_smoothed_precision,
        "selected_specialist_recall": oof_specialist_recall, "fallback": oof_fallback,
        "distinct_proposal_count": oof_competing_count,
        "prediction_D": d_predictions,
        "prediction_B_plus_D": bd_predictions if bd_predictions is not None else np.full(len(y), "", dtype=object)})
    for candidate_id in order:
        oof[f"prediction::{candidate_id}"] = model_predictions[candidate_id]
        p = np.asarray(by_id[candidate_id]["probabilities"])
        for j, label in enumerate(classes):
            oof[f"prob::{candidate_id}::{label}"] = p[:, j]
    if bd_probs is not None:
        for j, label in enumerate(classes):
            oof[f"prob::B_plus_D::{label}"] = bd_probs[:, j]
    for label in classes:
        oof[f"specialist_prediction::{label}"] = oof_specialist_only[label]
    routes = pd.DataFrame(route_rows)
    routes["proposals_json"] = routes.proposals.map(lambda value: json.dumps(value, sort_keys=True))
    routes = routes.drop(columns="proposals")
    oof_path = output_dir / "oof_predictions.parquet"
    routes_path = output_dir / "routing_decisions.parquet"
    assignments_path = output_dir / "specialist_assignments.csv"
    reliability_path = output_dir / "specialist_reliability.json"
    metrics_path = output_dir / "metrics.json"
    oof.to_parquet(oof_path, index=False)
    routes.to_parquet(routes_path, index=False)
    assignment_frame.to_csv(assignments_path, index=False)
    reliability_path.write_text(json.dumps(reliability_by_fold, indent=2, default=_json_default) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    report_path = Path(report_path)
    _write_report(report_path, metrics, order, validation_ids)
    return {"report_path": str(report_path), "oof_predictions_path": str(oof_path),
            "routing_decisions_path": str(routes_path), "specialist_assignments_path": str(assignments_path),
            "specialist_reliability_path": str(reliability_path), "metrics_path": str(metrics_path),
            "balanced_accuracy": trusted_metrics["balanced_accuracy"],
            "high_accuracy_label_coverage": threshold_metrics,
            "routing_movement": movement}


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _percent(value):
    return f"{100 * float(value):.1f}%"


def _write_report(path, metrics, order, validation_ids):
    label_stats = metrics["trusted_checkpoint_ensemble"]["per_class"]
    d_stats = metrics["baselines"]["D"]["per_class"]
    specialist_stats = metrics["specialist_stream_oof_by_label"]
    specialist_matrix = metrics["specialist_coverage_matrix"]
    assignments = metrics["pooled_individual_label_diagnostics"]
    lines = ["# Trusted Checkpoint Ensemble — UltraBERT", "",
        "## Method and validation boundary", "",
        f"The analysis uses {metrics['validation_rows']} frozen V0 rows, {len(metrics['classes'])} labels and {metrics['prompt_groups']} prompt groups. Five-fold GroupKFold keeps every prompt group wholly in one fold. Specialist assignments, precision smoothing and minimum-support decisions are fitted on the other four folds only; routing is evaluated on the held-out fold. The four checkpoints and their roster were previously selected using V0, so these are development diagnostics, not independent generalization estimates.", "",
        f"Reliability rule: minimum predicted-label support **{metrics['minimum_support']}**. Precision is smoothed with Beta(α={metrics['precision_smoothing']['prior_alpha']:g}, β={metrics['precision_smoothing']['prior_beta']:g}): `(TP+α)/(predicted_count+α+β)`. For each label the specialist is selected by highest fitting-fold recall, then raw precision, then F1; exact ties use candidate order `{', '.join(order)}`. For routing, eligible proposals rank by assigned-specialist flag, smoothed precision, specialist recall, candidate order and canonical label order. If none meet support, route to D.", "",
        f"Split SHA-256: `{metrics.get('validation_split_sha256')}`. Validation ID SHA-256: `{metrics.get('validation_ids_sha256')}`.", "",
        "## A. Specialist coverage matrix", "",
        "The highest-recall checkpoint and best-precision checkpoint columns are descriptive measurements over all 540 saved predictions (the checkpoints were already selected using V0). Fold assignments show the independently re-fit experts used by the OOF router; their OOF specialist-only metrics are reported in section C.", "",
        "| Label | Highest recall checkpoint | Recall | Precision | Best precision checkpoint | Precision | Recall | Fold assignment counts |", "|---|---|---:|---:|---|---:|---:|---|"]
    for label in metrics["classes"]:
        specialist = specialist_matrix[label]
        best_recall = specialist["best_recall_candidate_metrics"]
        pooled = assignments[label]
        recall_mark = " **≥90%**" if best_recall["recall"] >= .90 else " **≥80%**" if best_recall["recall"] >= .80 else ""
        counts_text = ", ".join(f"{key}:{value}" for key, value in specialist["fold_assignment_counts"].items() if value)
        lines.append(f"| {label} | {specialist['best_recall_candidate_id']} | {best_recall['recall']:.1%}{recall_mark} | {best_recall['precision']:.1%} | {pooled['best_precision_candidate_id']} | {pooled['best_precision']:.1%} | {pooled['best_precision_recall']:.1%} | {counts_text} |")
    lines.extend(["", "## B. Trusted ensemble per-label performance", "",
        "TP counts are out of 30 true examples per label. The specialist column shows the most frequently fitted expert, its fold count, and which checkpoint actually supplied this label's routed predictions.", "",
        "| Label | D recall (TP/30) | Best individual specialist recall (TP/30) | Trusted recall (TP/30) | Trusted precision | Selected expert / routed rows |", "|---|---:|---:|---:|---:|---|"])
    for label in metrics["classes"]:
        d = d_stats[label]
        individual = specialist_stats[label]
        trusted = label_stats[label]
        selected_for_label = []
        for candidate_id in order:
            info = metrics["routing_counts_by_selected_candidate"][candidate_id]["labels"][label]
            if info["selected"]:
                selected_for_label.append(f"{candidate_id}:{info['selected']}")
        selected_text = ", ".join(selected_for_label) if selected_for_label else "none"
        best = specialist_matrix[label]
        lines.append(f"| {label} | {d['recall']:.1%} ({d['true_positive']}/30) | {individual['recall']:.1%} ({individual['true_positive']}/30) | {trusted['recall']:.1%} ({trusted['true_positive']}/30) | {trusted['precision']:.1%} | {best['majority_fold_specialist']} ({best['fold_assignment_counts'][best['majority_fold_specialist']]}/5); {selected_text} |")
    lines.extend(["", "## C. High-accuracy label coverage", "",
        "Best individual specialists are selected within each fitting fold and measured on held-out rows. The additional any-individual-checkpoint column is a pooled descriptive diagnostic over all V0 predictions and is optimistic because it selects the best checkpoint per label after seeing these same labels. Precision requires at least the configured minimum predicted-label support. D and trusted-router values are ordinary multiclass one-vs-rest metrics.", "",
        "| Metric | D | Best individual specialists (OOF) | Any individual checkpoint (pooled) | Trusted ensemble (OOF) |", "|---|---:|---:|---:|---:|"])
    captions = [("recall_ge_80", "Labels with recall ≥80%"), ("recall_ge_90", "Labels with recall ≥90%"),
        ("precision_ge_80", "Labels with precision ≥80%"), ("precision_ge_90", "Labels with precision ≥90%"),
        ("both_ge_80", "Labels with both precision and recall ≥80%"),
        ("both_ge_90", "Labels with both precision and recall ≥90%")]
    for key, caption in captions:
        values = metrics["high_accuracy_label_coverage"]
        lines.append(f"| {caption} | {values['D'][key]} | {values['best_individual_specialists'][key]} | {values['any_individual_checkpoint_pooled'][key]} | {values['trusted_checkpoint_ensemble'][key]} |")
    lines.extend(["", "## D. Routing analysis", "",
        "### Checkpoint usage", "",
        "| Checkpoint | Selected predictions | Correct | Incorrect |", "|---|---:|---:|---:|"])
    for candidate_id in order:
        stats = metrics["routing_counts_by_selected_candidate"][candidate_id]
        lines.append(f"| {candidate_id} | {stats['selected_rows']} | {stats['correct']} | {stats['incorrect']} |")
    lines.extend(["", "Rows with conflicting proposals (different top-1 labels): "
        f"{metrics['routing_movement']['rows_with_multiple_distinct_proposals']} / {metrics['validation_rows']}; "
        f"mean distinct proposed labels: {metrics['routing_movement']['mean_distinct_proposal_labels']:.2f}. "
        f"Fallback to D for insufficient support: {metrics['routing_movement']['fallback_rows']} rows.", "",
        f"Compared with D, trusted routing corrected {metrics['routing_movement']['corrected_vs_D']} rows and regressed {metrics['routing_movement']['regressed_vs_D']}; identical top-1 label on {metrics['routing_movement']['same_prediction_as_D']} rows.", "",
        "Per-checkpoint, per-label selected correct/incorrect counts:", ""])
    for candidate_id in order:
        lines.append(f"**{candidate_id}**")
        lines.append("| Label | Routed rows | Correct | Incorrect |")
        lines.append("|---|---:|---:|---:|")
        for label in metrics["classes"]:
            row = metrics["routing_counts_by_selected_candidate"][candidate_id]["labels"][label]
            if row["selected"]:
                lines.append(f"| {label} | {row['selected']} | {row['correct']} | {row['incorrect']} |")
        lines.append("")
    lines.extend(["Specialist assignments were re-fit in each fold. Fold assignment counts appear in the coverage matrix; full fold-by-fold confusion counts are in `specialist_assignments.csv` and `specialist_reliability.json`.", "",
        "## E. Overall performance", "",
        "| Method | Balanced accuracy | Total correct | Macro F1 | Top-3 | MRR | Log loss | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    d = metrics["baselines"]["D"]
    bd = metrics["baselines"]["B_plus_D_equal_probability"]
    trusted = metrics["trusted_checkpoint_ensemble"]
    oracle = metrics["baselines"]["four_checkpoint_top1_oracle"]
    lines.append(f"| D | {d['balanced_accuracy']:.1%} | {d['total_correct']}/540 | {d['macro_f1']:.1%} | {d['top3_accuracy']:.1%} | {d['mrr']:.4f} | {d['log_loss']:.4f} | {d['multiclass_brier']:.4f} |")
    lines.append(f"| B+D equal probability | {bd['balanced_accuracy']:.1%} | {int(round(bd['accuracy']*540))}/540 | {bd['macro_f1']:.1%} | {bd['top3_accuracy']:.1%} | {bd['mrr']:.4f} | {bd['log_loss']:.4f} | {bd['multiclass_brier']:.4f} |")
    lines.append(f"| Trusted checkpoint ensemble | {trusted['balanced_accuracy']:.1%} | {trusted['total_correct']}/540 | {trusted['macro_f1']:.1%} | N/A | N/A | N/A | N/A |")
    lines.append(f"| Four-checkpoint top-1 oracle (theoretical) | {oracle['accuracy']:.1%} | {oracle['total_correct']}/540 | N/A | N/A | N/A | N/A | N/A |")
    lines.extend(["", "## F. Conclusions", "",
        f"1. **Labels at ≥80% recall:** {metrics['high_accuracy_label_coverage']['trusted_checkpoint_ensemble']['recall_ge_80']} of 18 after routing.",
        f"2. **Labels at ≥90% recall:** {metrics['high_accuracy_label_coverage']['trusted_checkpoint_ensemble']['recall_ge_90']} of 18 after routing.",
        "3. **Reliable specialists:** The coverage matrix shows best observed per-label checkpoints; fold assignment counts show which specialists the OOF fitting folds selected.",
        f"4. **Recall/precision trade-off:** trusted routing yields {metrics['high_accuracy_label_coverage']['trusted_checkpoint_ensemble']['both_ge_80']} labels with both precision and recall ≥80%, and {metrics['high_accuracy_label_coverage']['trusted_checkpoint_ensemble']['both_ge_90']} with both ≥90%. The routed predictions' false positives are shown per label above.",
        f"5. **Overall balanced accuracy:** {trusted['balanced_accuracy']:.1%} versus D {d['balanced_accuracy']:.1%} and B+D {bd['balanced_accuracy']:.1%}.",
        f"6. **Main limit:** {metrics['routing_movement']['rows_with_multiple_distinct_proposals']} rows had competing labels; see corrected/regressed counts and class-specific false positives to assess whether conflicts or weak expert coverage dominate.",
        "7. **Other architectures:** Yes. The router consumes candidate IDs, ordered classes and aligned probability matrices; model-specific code is confined to loading the saved candidates.",
        "",
        f"The four-checkpoint oracle reaches {oracle['accuracy']:.1%} by using the true label to choose a correct checkpoint. It is an upper bound, not a prediction method. All results use V0, which had previously informed checkpoint selection.",
        "", "Artifacts: `oof_predictions.parquet`, `routing_decisions.parquet`, `specialist_assignments.csv`, `specialist_reliability.json`, and `metrics.json`.", ""])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ultrabert_trusted_checkpoint_ensemble(*, artifact_dir="artifacts/predictions",
        data_dir="data/processed", output_dir="artifacts/analysis/trusted_checkpoint_ensemble_ultrabert",
        report_path="docs/trusted_checkpoint_ensemble_ultrabert.md",
        minimum_support=DEFAULT_MINIMUM_SUPPORT, prior_alpha=DEFAULT_PRIOR_ALPHA,
        prior_beta=DEFAULT_PRIOR_BETA, registry_path="experiments/registry.sqlite"):
    """Resolve A-D saved prediction files and run model-agnostic trusted routing."""
    from .pipeline import _load_dataset
    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.eq("validation"), ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["y_true"] = validation.pop("model").astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540 or validation.prompt_id.nunique() != 30:
        raise ValueError("expected frozen V0 validation with 540 rows and 30 prompt groups")
    classes = sorted(validation.y_true.unique().tolist())
    candidates, problems = _read_candidates(Path(artifact_dir), min_validation_rows=540,
        candidate_ids=ULTRABERT_CHECKPOINTS.values(), deduplicate=False)
    found = {item["candidate_id"] for item in candidates}
    missing = set(ULTRABERT_CHECKPOINTS.values()) - found
    if missing:
        raise FileNotFoundError(f"missing trusted ensemble candidates {sorted(missing)}; issues={problems}")
    _, y, groups, aligned, excluded, ids = _align_candidates(candidates,
        reference_dataset="v0", reference_frame=validation)
    by_id = {item["candidate_id"]: item for item in aligned}
    ordered_candidates = []
    display_order = []
    for display_id, experiment_id in ULTRABERT_CHECKPOINTS.items():
        candidate = by_id[experiment_id]
        if candidate["model"] != "ultrabert380":
            raise ValueError(f"{display_id} is not an UltraBERT candidate")
        item = dict(candidate)
        item["candidate_id"] = display_id
        item["classes"] = classes
        item["ids"] = ids
        item["prompt_ids"] = groups
        item["experiment_id"] = experiment_id
        item["manifest_path"] = candidate["manifest_path"]
        ordered_candidates.append(item)
        display_order.append(display_id)
    split_hash = json.loads((Path(data_dir) / "dataset_profile.json").read_text())["split_sha256"]
    result = run_trusted_checkpoint_ensemble(candidates=ordered_candidates, y_true=y,
        groups=groups, classes=classes, output_dir=output_dir, candidate_order=display_order,
        minimum_support=minimum_support, prior_alpha=prior_alpha, prior_beta=prior_beta,
        folds=5, validation_split_sha256=split_hash, validation_ids=ids, report_path=report_path)
    # Add exact experiment IDs and source file paths to the machine-readable results.
    paths = {item["candidate_id"]: {"experiment_id": item["experiment_id"],
        "manifest_path": item["manifest_path"],
        "prediction_path": str((Path(artifact_dir) / f"{item['experiment_id']}_validation.parquet").resolve())}
        for item in ordered_candidates}
    metric_path = Path(result["metrics_path"])
    metrics = json.loads(metric_path.read_text())
    metrics["candidate_artifacts"] = paths
    metric_path.write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    result["candidate_artifacts"] = paths
    return result
