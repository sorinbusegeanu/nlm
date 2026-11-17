"""Recall-ranked one-rule-per-label cascades on cached Trust List 2 predictions.

All checkpoint predictions and prompt-group folds are reused from the existing
ensemble analysis. No checkpoint loading, inference, or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from .ensemble_cascade_comparison import (
    CACHE_DIR as PROB_CACHE_DIR,
    EQUAL9_IDS,
    MIN_TRAIN_SUPPORT,
    ROOT,
    _load_inputs,
    _sha256,
)
from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS

TRUST_PATH = Path("docs/trust-list-2-v025.csv")
EXISTING_CASCADE_DIR = Path("artifacts/analysis/ensemble_cascade_comparison")
OUT_DIR = Path("artifacts/analysis/ensemble_recall_cascade")
REPORT_PATH = Path("docs/ensemble_recall_cascade.md")
STRATEGIES = ("R1_highest_recall", "R2_recall_x_prediction_count", "R3_recall_x_true_label_count")


def _score(stat: dict, strategy: str) -> float:
    if strategy == STRATEGIES[0]:
        return float(stat["recall"])
    if strategy == STRATEGIES[1]:
        return float(stat["recall"] * stat["prediction_count"])
    return float(stat["recall"] * stat["support"])


def _candidate_stat(y: np.ndarray, cp_prediction: np.ndarray, rows: np.ndarray, label: str) -> dict:
    predicted = cp_prediction[rows] == label
    truth = y[rows]
    count = int(predicted.sum())
    tp = int(np.sum(predicted & (truth == label)))
    support = int(np.sum(truth == label))
    fp = count - tp
    recall = tp / support if support else 0.0
    precision = tp / count if count else 0.0
    return {"prediction_count": count, "support": support, "tp": tp, "fp": fp,
            "recall": recall, "precision": precision}


def _select_rules(strategy: str, y: np.ndarray, predictions: dict[str, np.ndarray],
                  rows: np.ndarray, checkpoint_model: dict[str, str]) -> list[dict]:
    """Fit one supported specialist per class, then globally rank by its score."""
    selected = []
    for label_index, label in enumerate(LABELS):
        candidates = []
        for checkpoint_id, cp_prediction in predictions.items():
            stat = _candidate_stat(y, cp_prediction, rows, label)
            if stat["prediction_count"] < MIN_TRAIN_SUPPORT:
                continue
            candidates.append((checkpoint_id, stat))
        if not candidates:
            continue

        # Deterministic tie-breaking after the requested primary selection
        # score: recall, precision, then ascending checkpoint sequence ID.
        def candidate_key(item):
            cp_id, stat = item
            return (-_score(stat, strategy), -stat["recall"], -stat["precision"],
                    int(cp_id[1:]))

        cp_id, stat = min(candidates, key=candidate_key)
        selected.append({"checkpoint_id": cp_id, "label": label,
                         "model": checkpoint_model[cp_id], "label_index": label_index,
                         "fit_score": _score(stat, strategy), **{f"fit_{k}": v for k, v in stat.items()}})

    # Global ordering: requested score descending, canonical label order, then
    # checkpoint sequence ID. There is exactly one rule per covered label.
    selected.sort(key=lambda rule: (-rule["fit_score"], rule["label_index"],
                                    int(rule["checkpoint_id"][1:])))
    for pos, rule in enumerate(selected, 1):
        rule["position"] = pos
    return selected


def _apply_and_trace(rules: list[dict], rows: np.ndarray, y: np.ndarray,
                     predictions: dict[str, np.ndarray], fallback: np.ndarray):
    output = fallback[rows].copy()
    remaining = np.ones(len(rows), dtype=bool)
    contributions = []
    for rule in rules:
        fire = remaining & (predictions[rule["checkpoint_id"]][rows] == rule["label"])
        assigned = int(fire.sum())
        correct = int(np.sum(fire & (y[rows] == rule["label"])))
        contributions.append({"position": rule["position"], "checkpoint_id": rule["checkpoint_id"],
                              "model": rule["model"], "label": rule["label"],
                              "fit_score": rule["fit_score"], "fit_recall": rule["fit_recall"],
                              "fit_prediction_count": rule["fit_prediction_count"],
                              "fit_tp": rule["fit_tp"], "assigned_rows": assigned,
                              "correct": correct, "incorrect": assigned - correct})
        output[fire] = rule["label"]
        remaining[fire] = False
    return output, remaining, contributions


def _metrics(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray, scope: str,
             strategy: str, rules: list[dict], fallback_rows: int) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=LABELS, zero_division=0)
    old_correct, new_correct = baseline == y, pred == y
    return {"scope": scope, "strategy": strategy, "rule_count": len(rules),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "correct": int(new_correct.sum()), "rows": len(y),
            "corrected_vs_equal9": int(np.sum(~old_correct & new_correct)),
            "introduced_vs_equal9": int(np.sum(old_correct & ~new_correct)),
            "disagreements_vs_equal9": int(np.sum(pred != baseline)),
            "fallback_rows": int(fallback_rows),
            "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i]),
                                  "correct": int(np.sum((y == label) & (pred == label)))}
                          for i, label in enumerate(LABELS)}}


def run(root: str | Path = ROOT, output_dir: str | Path = OUT_DIR,
        report_path: str | Path = REPORT_PATH) -> dict:
    root = Path(root).resolve()
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    report = Path(report_path); report = report if report.is_absolute() else root / report
    out.mkdir(parents=True, exist_ok=True)

    v0, trust, probabilities, predictions, equal9_prob, equal9_pred, fold_id, _ = _load_inputs(root)
    y = v0.model.to_numpy(dtype=object)
    n = len(v0)
    indices = np.arange(n)
    checkpoint_model = trust.drop_duplicates("sequence_id").set_index("sequence_id").model.astype(str).to_dict()
    cp_ids = sorted(predictions, key=lambda value: int(value[1:]))

    # Verify existing C1/C3 caches are aligned to the same frozen V0 and folds.
    prior = pd.read_parquet(root / EXISTING_CASCADE_DIR / "cascade_predictions.parquet")
    for column, expected in (("id", v0.id.astype(str).to_numpy()),
                             ("prompt_id", v0.prompt_id.astype(str).to_numpy()),
                             ("y_true", y.astype(str))):
        if column not in prior or not np.array_equal(prior[column].astype(str).to_numpy(), expected):
            raise ValueError(f"Existing cascade cache is misaligned at {column}")
    required_prior = ["pred_oof::C1_precision", "pred_oof::C3_precision",
                      "pred_full::C1_precision", "pred_full::C3_precision"]
    if any(col not in prior for col in required_prior):
        raise ValueError("Existing EQUAL9/C1/C3 comparison predictions are incomplete")
    if not np.array_equal(prior["pred_equal9"].astype(str).to_numpy(), equal9_pred.astype(str)):
        raise ValueError("EQUAL9 cached predictions disagree with the probability-based fallback")

    oof = {strategy: np.empty(n, dtype=object) for strategy in STRATEGIES}
    full_predictions = {}
    fold_rules: dict[str, dict[str, list[dict]]] = {}
    fold_rows = []
    fold_contributions = []
    fold_json_dir = out / "folds"
    fold_json_dir.mkdir(exist_ok=True)

    for fold in range(5):
        fit_rows = indices[fold_id != fold]
        held_rows = indices[fold_id == fold]
        fold_rules[str(fold + 1)] = {}
        for strategy in STRATEGIES:
            rules = _select_rules(strategy, y, predictions, fit_rows, checkpoint_model)
            fold_rules[str(fold + 1)][strategy] = rules
            pred, fallback_mask, contributions = _apply_and_trace(
                rules, held_rows, y, predictions, equal9_pred)
            oof[strategy][held_rows] = pred
            fallback_count = int(fallback_mask.sum())
            for row in contributions:
                fold_contributions.append({"fold": fold + 1, "strategy": strategy, **row})
            fb_correct = int(np.sum(fallback_mask & (equal9_pred[held_rows] == y[held_rows])))
            fold_contributions.append({"fold": fold + 1, "strategy": strategy,
                "position": len(rules) + 1, "checkpoint_id": "EQUAL9", "model": "EQUAL9",
                "label": "fallback", "fit_score": None, "fit_recall": None,
                "fit_prediction_count": None, "fit_tp": None, "assigned_rows": fallback_count,
                "correct": fb_correct, "incorrect": fallback_count - fb_correct})
            fold_rows.append({"fold": fold + 1, "strategy": strategy,
                              "fit_rows": len(fit_rows), "held_rows": len(held_rows),
                              "rule_count": len(rules), "fallback_rows": fallback_count,
                              "fallback_correct": fb_correct})
        payload = {"fold": fold + 1, "held_indexes": held_rows.tolist(),
                   "strategies": fold_rules[str(fold + 1)]}
        (fold_json_dir / f"fold_{fold + 1:02d}.json").write_text(json.dumps(payload, indent=2) + "\n")
        np.savez_compressed(fold_json_dir / f"fold_{fold + 1:02d}_predictions.npz",
                            **{key: val[held_rows].astype(str) for key, val in oof.items()})

    pd.DataFrame({"id": v0.id.astype(str), "prompt_id": v0.prompt_id.astype(str),
                  "fold": fold_id + 1}).to_csv(out / "fold_assignments.csv", index=False)

    full_rules, full_contributions = {}, []
    for strategy in STRATEGIES:
        rules = _select_rules(strategy, y, predictions, indices, checkpoint_model)
        full_rules[strategy] = rules
        pred, fallback_mask, contributions = _apply_and_trace(
            rules, indices, y, predictions, equal9_pred)
        full_predictions[strategy] = pred
        for row in contributions:
            full_contributions.append({"strategy": strategy, **row})
        fb_count = int(fallback_mask.sum())
        fb_correct = int(np.sum(fallback_mask & (equal9_pred == y)))
        full_contributions.append({"strategy": strategy, "position": len(rules) + 1,
            "checkpoint_id": "EQUAL9", "model": "EQUAL9", "label": "fallback",
            "fit_score": None, "fit_recall": None, "fit_prediction_count": None,
            "fit_tp": None, "assigned_rows": fb_count, "correct": fb_correct,
            "incorrect": fb_count - fb_correct})

    # Baselines are read from exact existing caches to prevent implementation drift.
    records = []
    for scope, current in (("prompt_group_oof", True), ("full_v0_diagnostic", False)):
        for name in ("EQUAL9", "C1_precision", "C3_precision"):
            if name == "EQUAL9":
                pred = equal9_pred if not current else prior["pred_equal9"].astype(str).to_numpy(dtype=object)
            else:
                suffix = "oof" if current else "full"
                pred = prior[f"pred_{suffix}::{name}"].astype(str).to_numpy(dtype=object)
            m = _metrics(y, pred, equal9_pred, scope, name, [], 0)
            records.append(m)
        for strategy in STRATEGIES:
            pred = oof[strategy] if current else full_predictions[strategy]
            if pd.isna(pred).any() or any(not str(x) for x in pred):
                raise RuntimeError(f"Incomplete predictions for {scope} {strategy}")
            contributions = ([x for x in fold_contributions if x["strategy"] == strategy]
                             if current else [x for x in full_contributions if x["strategy"] == strategy])
            fallback_rows = sum(int(x["assigned_rows"]) for x in contributions if x["label"] == "fallback")
            rules = [rule for fold in fold_rules.values() for rule in fold[strategy]] if current else full_rules[strategy]
            records.append(_metrics(y, pred, equal9_pred, scope, strategy,
                                    rules if not current else rules[:], fallback_rows))

    # For OOF rows, each example is scored once; report mean fold rule count separately.
    metrics = pd.DataFrame([{k: v for k, v in item.items() if k != "per_label"} for item in records])
    per_label_rows = []
    for item in records:
        for label, stats in item["per_label"].items():
            per_label_rows.append({"scope": item["scope"], "strategy": item["strategy"],
                                   "label": label, **stats})
    # The aggregate OOF rule_count is mean active rules per fold, not sum.
    for strategy in STRATEGIES:
        mask = metrics.strategy.eq(strategy) & metrics.scope.eq("prompt_group_oof")
        metrics.loc[mask, "rule_count"] = np.mean([len(fold_rules[str(f)][strategy]) for f in range(1, 6)])
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(out / "per_label_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(fold_contributions).to_csv(out / "oof_rule_contributions.csv", index=False)
    pd.DataFrame(full_contributions).to_csv(out / "full_v0_rule_contributions.csv", index=False)

    prediction_frame = v0[["id", "prompt_id", "model"]].copy().rename(columns={"model": "y_true"})
    prediction_frame["pred_equal9"] = equal9_pred
    for baseline in ("C1_precision", "C3_precision"):
        prediction_frame[f"pred_oof::{baseline}"] = prior[f"pred_oof::{baseline}"].astype(str).to_numpy()
        prediction_frame[f"pred_full::{baseline}"] = prior[f"pred_full::{baseline}"].astype(str).to_numpy()
    for strategy in STRATEGIES:
        prediction_frame[f"pred_oof::{strategy}"] = oof[strategy]
        prediction_frame[f"pred_full::{strategy}"] = full_predictions[strategy]
    prediction_frame.to_parquet(out / "cascade_predictions.parquet", index=False)
    (out / "selected_rules.json").write_text(json.dumps({"cross_fitted": fold_rules,
        "full_v0_diagnostic": full_rules}, indent=2) + "\n")
    pd.DataFrame(fold_contributions).to_csv(out / "rule_firing_summary.csv", index=False)

    config = {"algorithm": "one rule per label, selected and globally ordered by recall-based objective",
        "trust_list_version": "trust-list-2-v025", "trust_list_path": str(TRUST_PATH),
        "trust_sha256": _sha256(root / TRUST_PATH), "prediction_cache": str(PROB_CACHE_DIR / "checkpoint_probabilities.npz"),
        "prediction_cache_sha256": _sha256(root / PROB_CACHE_DIR / "checkpoint_probabilities.npz"),
        "v0_rows": n, "v0_ordered_ids_sha256": hashlib.sha256("\n".join(v0.id.astype(str)).encode()).hexdigest(),
        "prompt_groups": int(v0.prompt_id.nunique()), "checkpoint_ids": cp_ids,
        "checkpoint_count": len(cp_ids), "class_order": LABELS,
        "equal9_checkpoint_ids": EQUAL9_IDS, "fold_source": "ensemble_precision_recall_overlap existing five prompt-group folds",
        "min_training_fold_rule_support": MIN_TRAIN_SUPPORT,
        "selection": {
            STRATEGIES[0]: "maximize recall per label; tie break by precision, then checkpoint sequence ID",
            STRATEGIES[1]: "maximize recall * top-1 prediction count per label; tie break by recall, precision, then checkpoint sequence ID",
            STRATEGIES[2]: "maximize recall * true-label count (TP) per label; tie break by recall, precision, then checkpoint sequence ID"},
        "global_order": "descending strategy score; ties by canonical label order then ascending checkpoint sequence ID",
        "cascade": "each rule claims remaining rows where its checkpoint top-1 equals the rule label; first match wins; EQUAL9 handles leftovers",
        "full_v0_diagnostic_note": "rules are fitted and evaluated on the same full V0 rows; descriptive/in-sample only"}
    (out / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"status": "completed", "trust_sha256": config["trust_sha256"],
                "prediction_cache_sha256": config["prediction_cache_sha256"], "rows": n,
                "folds_completed": 5, "artifact_dir": str(out.relative_to(root)),
                "report_path": str(report.relative_to(root))}
    (out / "analysis_state.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(report, metrics, per_label_rows, fold_rules, full_rules, config, out)
    return manifest


def _write_report(path: Path, metrics: pd.DataFrame, per_label_rows: list[dict],
                  fold_rules: dict, full_rules: dict, config: dict, out: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = metrics[(metrics.scope == "prompt_group_oof") & (metrics.strategy == "EQUAL9")].iloc[0]
    lines = ["# Recall-based cascade experiment", "",
        "Cache-only evaluation using Trust List 2 v025 and the existing full-V0 checkpoint probability predictions. No model inference, base-model training, production-ensemble changes, or Kaggle submission was performed.", "",
        f"Trust-list SHA-256: `{config['trust_sha256']}`. Cached checkpoint predictions: {config['checkpoint_count']}. V0 rows: {config['v0_rows']}; prompt groups: {config['prompt_groups']}. Existing five prompt-group folds and the minimum fitting-fold rule support of {config['min_training_fold_rule_support']} were reused.", "",
        "Cross-fitted results are primary. Although rule selection is fit-fold-only, checkpoints were previously selected using V0, so held-out folds do not remove checkpoint-selection exposure. Full-V0 diagnostics fit and score on the same rows and are optimistic descriptive results.", "",
        f"EQUAL9 baseline: {baseline.accuracy:.4%} accuracy, {baseline.balanced_accuracy:.4%} balanced accuracy ({int(baseline.correct)}/{int(baseline.rows)} correct). C1 and C3 values below are taken from the existing aligned cascade prediction artifacts.", "",
        "## Overall comparison", "",
        "| Scope | Strategy | Rules | Accuracy | Balanced accuracy | Correct | Δ BA vs EQUAL9 | Corrected | Introduced | Disagreements | Fallback rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    display_order = ["EQUAL9", "C1_precision", "C3_precision", *STRATEGIES]
    for scope in ("prompt_group_oof", "full_v0_diagnostic"):
        base = metrics[(metrics.scope == scope) & (metrics.strategy == "EQUAL9")].iloc[0]
        for name in display_order:
            row = metrics[(metrics.scope == scope) & (metrics.strategy == name)].iloc[0]
            delta = (row.balanced_accuracy - base.balanced_accuracy) * 100
            lines.append(f"| {scope} | {name} | {row.rule_count:.1f} | {row.accuracy:.4%} | {row.balanced_accuracy:.4%} | {int(row.correct)}/{int(row.rows)} | {delta:+.3f} pp | {int(row.corrected_vs_equal9)} | {int(row.introduced_vs_equal9)} | {int(row.disagreements_vs_equal9)} | {int(row.fallback_rows)} |")

    lines += ["", "## Rule selection and order", "",
        "R1 selects each label's checkpoint by highest fit-fold recall. R2 selects by recall multiplied by that checkpoint's top-1 prediction count for the label. R3 selects by recall multiplied by the true-label support, which equals TP. Candidates with fewer than 10 matching fit-fold top-1 predictions are excluded. Ties are deterministic: selection ties use recall, precision, then lowest checkpoint sequence; global order ties use canonical label order then checkpoint sequence.", "",
        "### Full-V0 diagnostic rules", ""]
    for strategy in STRATEGIES:
        lines.append(f"**{strategy}**")
        for rule in full_rules[strategy]:
            lines.append(f"{rule['position']}. `{rule['label']}` ← {rule['checkpoint_id']} ({rule['model']}); selection score {rule['fit_score']:.4f}, recall {rule['fit_recall']:.4f}, predictions {rule['fit_prediction_count']}, TP {rule['fit_tp']}.")
        lines.append("")
    lines += ["### Exact fold-specific rule orders", "",
        "Each fold's list was selected only on the other four folds and applied to the held-out fold:", ""]
    for fold in sorted(fold_rules, key=int):
        lines.append(f"**Fold {fold}**")
        for strategy in STRATEGIES:
            order = " → ".join(f"{r['checkpoint_id']}:{r['label']}" for r in fold_rules[fold][strategy])
            lines.append(f"- {strategy}: {order}")
        lines.append("")
    lines += ["Fold JSON also preserves each selected rule's fitting-fold recall, prediction count, TP, score and model ID.", "",
        "## Per-label precision and recall", "",
        "Values below are prompt-group OOF precision / recall. C1 and C3 are included for direct comparison.", "",
        "| Label | EQUAL9 | C1 | C3 | R1 | R2 | R3 |", "|---|---:|---:|---:|---:|---:|---:|"]
    per = {(r["scope"], r["strategy"], r["label"]): r for r in per_label_rows}
    for label in LABELS:
        cells=[]
        for name in ("EQUAL9", "C1_precision", "C3_precision", *STRATEGIES):
            x=per[("prompt_group_oof", name, label)]
            cells.append(f"{x['precision']:.3f} / {x['recall']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Explain whether different criteria yield actual different predicted vectors.
    pred = pd.read_parquet(out / "cascade_predictions.parquet")
    oof_cols = [f"pred_oof::{name}" for name in STRATEGIES]
    full_cols = [f"pred_full::{name}" for name in STRATEGIES]
    lines += ["", "## Distinctness and firing behavior", "",
        "| Pair | OOF differing rows | OOF agreement | Full-V0 differing rows | Full-V0 agreement |", "|---|---:|---:|---:|---:|"]
    for i, a in enumerate(STRATEGIES):
        for b in STRATEGIES[i+1:]:
            oof_diff = int(np.sum(pred[f"pred_oof::{a}"].to_numpy() != pred[f"pred_oof::{b}"].to_numpy()))
            full_diff = int(np.sum(pred[f"pred_full::{a}"].to_numpy() != pred[f"pred_full::{b}"].to_numpy()))
            lines.append(f"| {a} vs {b} | {oof_diff} | {1-oof_diff/len(pred):.4%} | {full_diff} | {1-full_diff/len(pred):.4%} |")
    contributions = pd.read_csv(out / "oof_rule_contributions.csv")
    lines += ["", "Held-out cascade rule assignments (summed across the five folds):", "",
        "| Strategy | Rule-assigned rows | Correct | Incorrect | Fallback rows |", "|---|---:|---:|---:|---:|"]
    for strategy in STRATEGIES:
        sub = contributions[contributions.strategy == strategy]
        rules = sub[sub.label != "fallback"]
        fallback = sub[sub.label == "fallback"]
        lines.append(f"| {strategy} | {int(rules.assigned_rows.sum())} | {int(rules.correct.sum())} | {int(rules.incorrect.sum())} | {int(fallback.assigned_rows.sum())} |")

    lines += ["", "## Interpretation", "",
        "R3 is a control: because true-label support is constant across checkpoints for a given label, maximizing recall × support is exactly maximizing TP. R2 rewards the product of capture rate and firing volume; this can select a high-volume rule even when its recall is lower. A rule's fit-fold score is not its incremental precision after earlier rules have claimed examples; held-out firing counts show the cascade's actual first-match behavior.", "",
        "## Reproducibility", "",
        "- `experiment_config.json`: input hashes, roster, folds, support threshold, and exact tie-breaking.",
        "- `selected_rules.json` and `folds/fold_*.json`: full-data diagnostics and each fit-fold rule sequence.",
        "- `fold_assignments.csv`: exact saved five-fold ID/prompt mapping.",
        "- `cascade_predictions.parquet`: aligned IDs, targets, EQUAL9/C1/C3 baselines, and OOF/full predictions.",
        "- `metrics.csv`, `per_label_metrics.csv`, `fold_metrics.csv`, `oof_rule_contributions.csv`, and `full_v0_rule_contributions.csv`: metrics and rule firing details.",
        "- `analysis_state.json`: completion state and input hashes.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output_dir, args.report), indent=2))


if __name__ == "__main__":
    main()
