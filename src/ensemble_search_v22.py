"""Cross-fitted 4-10 member ensemble search over Trust List 2.2 v0005.

Consumes only cached full-V0 probabilities.  Selection and optimized weights
are fitted within each four-fold training partition; held-out probabilities are
pooled into out-of-fold metrics.  A separate full-V0 search is diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix

from .ensemble_precision_recall_overlap import EQUAL9, _sha as _file_sha
from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS, _read_saved_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("artifacts/analysis/ensemble_search_v22")
REPORT = Path("docs/ensemble_search_v22.md")
TRUST_POINTER = "docs/trust-list-2.2-latest.json"
FOLD_SOURCE = Path("artifacts/analysis/ensemble_precision_recall_overlap")
L2 = 0.05
MIN_SIZE, MAX_SIZE, TOP_K = 4, 10, 3
BEAM_WIDTH, SWAP_SEEDS, SWAP_PASSES = 24, 2, 1


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _id_key(combo: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    return tuple(sorted(int(x[1:]) for x in combo))


def _prob_metrics(y: np.ndarray, p: np.ndarray, combo: list[str], weights: list[float]) -> dict:
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    tp = np.diag(cm); fp = cm.sum(axis=0) - tp; support = cm.sum(axis=1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    correct = pred == y
    return {
        "checkpoint_ids": list(combo), "weights": {sid: float(w) for sid, w in zip(combo, weights)},
        "checkpoint_count": len(combo), "rows": int(len(y)), "correct": int(correct.sum()),
        "accuracy": float(correct.mean()), "balanced_accuracy": float(recall.mean()),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean()),
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(support[i]-tp[i])}
            for i, label in enumerate(LABELS)},
        "confusion_matrix": cm.tolist(), "y_pred": pred.tolist(),
    }


def _rank_key(record: dict) -> tuple:
    return (-record["balanced_accuracy"], -record["accuracy"], record["log_loss"],
            _id_key(record["checkpoint_ids"]))


def _fit_weights(prob: np.ndarray, y: np.ndarray, l2: float = L2) -> np.ndarray:
    """Same regularized log-loss objective as ensemble_factory.fit_weights.

    This uses its analytic gradient to make repeated deterministic subset fits
    efficient; constraints, initialization and shrinkage are otherwise exact.
    ``prob`` is [members, rows, classes].
    """
    m = prob.shape[0]
    target = prob[:, np.arange(len(y)), y].T  # rows x members
    equal = np.full(m, 1.0 / m)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        blend = target @ w
        loss = -np.log(np.clip(blend, 1e-12, 1.0)).mean()
        grad = -(target / np.clip(blend[:, None], 1e-12, None)).mean(axis=0)
        delta = w - equal
        return float(loss + l2 * np.dot(delta, delta)), grad + 2.0 * l2 * delta

    result = minimize(lambda w: objective(w)[0], equal, jac=lambda w: objective(w)[1],
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
                      "jac": lambda w: np.ones(m)}], method="SLSQP",
        options={"maxiter": 100, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"regularized weight fit failed: {result.message}")
    w = np.maximum(result.x, 0.0); w /= w.sum()
    return w


def _blend(matrices: dict[str, np.ndarray], combo: tuple[str, ...], indexes: np.ndarray,
           weights: np.ndarray) -> np.ndarray:
    return np.einsum("m,mnc->nc", weights,
        np.stack([matrices[sid][indexes] for sid in combo], axis=0), optimize=True)


def _search(matrices: dict[str, np.ndarray], y: np.ndarray, fit_idx: np.ndarray,
            method: str, *, beam_width: int = BEAM_WIDTH,
            swap_seeds: int = SWAP_SEEDS, swap_passes: int = SWAP_PASSES,
            return_diagnostics: bool = False) -> dict[int, list[dict]] | tuple[dict[int, list[dict]], dict]:
    candidates = sorted(matrices, key=lambda x: int(x[1:]))
    score_cache: dict[tuple[str, ...], dict] = {}
    def evaluate(raw_combo: tuple[str, ...]) -> dict:
        combo = tuple(sorted(raw_combo, key=lambda x: int(x[1:])))
        if combo in score_cache:
            return score_cache[combo]
        if method == "equal":
            weights = np.full(len(combo), 1.0 / len(combo))
        else:
            weights = _fit_weights(np.stack([matrices[s][fit_idx] for s in combo]), y[fit_idx])
        p = _blend(matrices, combo, fit_idx, weights)
        record = _prob_metrics(y[fit_idx], p, list(combo), weights.tolist())
        score_cache[combo] = record
        return record

    beam: list[tuple[str, ...]] = [(sid,) for sid in candidates]
    # Keep the complete layers below size 4; there are only 1,771 subsets up to
    # size 3 for this 22-member pool, which gives every member a fair start.
    for size in range(2, MIN_SIZE):
        all_layer = set()
        for node in beam:
            for sid in candidates:
                if sid not in node:
                    all_layer.add(tuple(sorted((*node, sid), key=lambda x: int(x[1:]))))
        beam = sorted(all_layer, key=lambda c: _rank_key(evaluate(c)))
        if size == MIN_SIZE - 1:
            beam = beam[:beam_width]

    top_by_size: dict[int, list[dict]] = {}
    for size in range(MIN_SIZE, MAX_SIZE + 1):
        generated: set[tuple[str, ...]] = set()
        for node in beam:
            for sid in candidates:
                if sid not in node:
                    generated.add(tuple(sorted((*node, sid), key=lambda x: int(x[1:]))))
        ranked = sorted(generated, key=lambda c: _rank_key(evaluate(c)))
        seeds = ranked[:swap_seeds]
        local_optima = []
        for seed in seeds:
            current = seed; current_record = evaluate(current)
            for _ in range(swap_passes):
                best_combo, best_record = current, current_record
                for remove in current:
                    for add in candidates:
                        if add in current:
                            continue
                        proposal = tuple(sorted((set(current) - {remove}) | {add},
                                                key=lambda x: int(x[1:])))
                        record = evaluate(proposal)
                        if _rank_key(record) < _rank_key(best_record):
                            best_combo, best_record = proposal, record
                if best_combo == current:
                    break
                current, current_record = best_combo, best_record
            local_optima.append(current)
        keep = set(ranked[:beam_width]) | set(local_optima)
        beam = sorted((c for c in keep if len(c) == size),
                      key=lambda c: _rank_key(evaluate(c)))[:beam_width]
        top_by_size[size] = [evaluate(c) for c in sorted(
            (c for c in keep if len(c) == size), key=lambda c: _rank_key(evaluate(c)))[:TOP_K]]
    if return_diagnostics:
        return top_by_size, {"unique_subsets_scored": len(score_cache),
            "beam_width": beam_width, "swap_seeds": swap_seeds,
            "swap_passes": swap_passes, "method": method}
    return top_by_size


def _load(root: Path, trust_list: str | None, output: Path):
    pointer_path = root / TRUST_POINTER
    pointer = json.loads(pointer_path.read_text())
    trust_path = root / trust_list if trust_list and not Path(trust_list).is_absolute() else Path(trust_list or root / pointer["csv_path"])
    if not trust_path.is_file():
        raise FileNotFoundError(trust_path)
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    unique = trust.drop_duplicates("sequence_id").copy()
    needed = {"sequence_id", "experiment_id", "model", "training_dataset", "checkpoint_path",
              "evaluation_eligible", "rule_eligible", "checkpoint_available",
              "validation_training_overlap", "validation_rows"}
    if needed - set(unique.columns):
        raise ValueError(f"Trust List missing fields: {sorted(needed-set(unique.columns))}")
    if len(unique) != 22:
        raise ValueError(f"expected 22 unique Trust List 2.2 candidates, found {len(unique)}")
    bool_true = lambda s: s.astype(str).str.lower().isin({"true", "1", "yes"})
    eligible = (bool_true(unique.evaluation_eligible) & bool_true(unique.rule_eligible) &
                bool_true(unique.checkpoint_available) &
                unique.validation_training_overlap.astype(str).isin({"0", "False", "false", ""}) &
                (pd.to_numeric(unique.validation_rows, errors="coerce") == 2699))
    if not eligible.all():
        raise ValueError("Trust List 2.2 contains ineligible rows among its 22 candidates")
    v0 = _dataset_rows("v0", root / "data/processed")
    if len(v0) != 2699 or v0.id.astype(str).duplicated().any():
        raise ValueError("V0 must contain exactly 2,699 unique response IDs")
    ids = v0.id.astype(str).tolist(); labels_text = v0.model.astype(str).tolist()
    if set(labels_text) != set(LABELS):
        raise ValueError("V0 canonical labels do not match the expected 18-class order")
    y = np.asarray([LABELS.index(label) for label in labels_text], dtype=np.int64)
    matrices: dict[str, np.ndarray] = {}; cache = {}
    # Add EQUAL9 members absent from the V2.x-only list via the current Trust List 2.
    baseline_trust_path = root / "docs/trust-list-2-v025.csv"
    baseline_trust = pd.read_csv(baseline_trust_path, dtype=str, keep_default_na=False)
    identity = unique.set_index("sequence_id", drop=False)
    base_identity = baseline_trust.drop_duplicates("sequence_id").set_index("sequence_id", drop=False)
    for sid in sorted(set(unique.sequence_id.tolist()) | set(EQUAL9), key=lambda x: int(x[1:])):
        source = identity if sid in identity.index else base_identity
        if sid not in source.index:
            raise KeyError(f"cannot resolve checkpoint {sid} for cached probabilities")
        row = source.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        exp_id = str(row.experiment_id)
        probs, path = _read_saved_probabilities(root, exp_id, ids, labels_text)
        matrices[sid] = probs.astype(np.float64, copy=False)
        cache[sid] = {"experiment_id": exp_id, "model": str(row.model),
                      "training_dataset": str(row.training_dataset),
                      "prediction_path": str(path.relative_to(root)),
                      "prediction_sha256": _file_sha(path)}
    # Reuse the prior five prompt-group assignments exactly.
    held_sets = []
    for fold in range(1, 6):
        record = json.loads((root / FOLD_SOURCE / "folds" / f"fold_{fold:02d}.json").read_text())
        held_sets.append(np.asarray(record["held_indexes"], dtype=np.int64))
    flat = np.concatenate(held_sets)
    if len(flat) != len(v0) or set(flat.tolist()) != set(range(len(v0))):
        raise ValueError("existing five folds do not partition all V0 rows")
    fold_id = np.empty(len(v0), dtype=np.int64)
    prompts = v0.prompt_id.astype(str).to_numpy()
    fold_records = []
    for i, held in enumerate(held_sets):
        fit = np.setdiff1d(np.arange(len(v0)), held, assume_unique=True)
        if set(prompts[fit]) & set(prompts[held]):
            raise ValueError(f"prompt group leakage in existing fold {i+1}")
        fold_id[held] = i + 1
        fold_records.append({"fold": i+1, "fit_indexes": fit, "held_indexes": held})
    return pointer, trust_path, trust, unique, v0, y, labels_text, matrices, cache, fold_id, fold_records


def _write_report(path: Path, results: dict) -> None:
    baseline = results["baseline"]
    lines = ["# Trust List 2.2 ensemble search (4–10 checkpoints)", "",
        f"Trust List: `{results['trust_list_version']}` from `{results['trust_list_path']}`; SHA-256 `{results['trust_list_sha256']}`.",
        "Reproduce with `python -m src.ensemble_search_v22`.",
        f"The candidate pool includes all **{results['candidate_count']}** eligible checkpoints. Probabilities were loaded from cached full-V0 prediction files only; no checkpoints were loaded and no inference or training ran.",
        "Selection used the existing five prompt-group folds. Each fold's selected members and optimized weights were fitted on its other four folds, then applied once to held-out rows. The Trust List roster/checkpoint history and prior checkpoint selection may have seen V0, so cross-fitting only evaluates this ensemble-selection layer; it is not an independent Kaggle estimate.",
        "Optimized weights use the existing ensemble_factory regularized objective: training-fold log loss plus L2 shrinkage toward equal weights, nonnegative simplex constraints. Search uses deterministic beam expansion and one-member local swaps.", "",
        "## Reference and search settings", "",
        f"EQUAL9 full-V0 balanced accuracy: **{baseline['full_v0']['balanced_accuracy']:.4%}** (target reference 65.3898%). Cross-fitted fixed EQUAL9 balanced accuracy: **{baseline['cross_fitted']['balanced_accuracy']:.4%}**.",
        f"Beam width {results['search']['beam_width']}; local-search seeds {results['search']['swap_seeds']}; swap passes {results['search']['swap_passes']}; sizes {MIN_SIZE}–{MAX_SIZE}; top {TOP_K} per method and size.", "",
        "## Cross-fitted results by size and rank", "",
        "Rank refers to a configuration selected using only that fold's fitting rows. Thus member IDs and optimized weights can vary across folds; exact selections are in `fold_selections.json`. Metrics pool the five held-out prediction partitions.", "",
        "| Size | Weight method | Fit-fold rank | OOF balanced accuracy | OOF accuracy | OOF log loss | Fold BA SD | Mean member Jaccard | Δ BA vs EQUAL9 | Corrected / introduced vs EQUAL9 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results["cross_fitted_rows"]:
        lines.append(f"| {row['size']} | {row['method']} | {row['rank']} | {row['balanced_accuracy']:.4%} | {row['accuracy']:.4%} | {row['log_loss']:.6f} | {row['fold_ba_std']*100:.3f} pp | {row['mean_member_jaccard']:.3f} | {(row['balanced_accuracy']-baseline['cross_fitted']['balanced_accuracy'])*100:+.3f} pp | {row['corrected_vs_equal9']} / {row['introduced_vs_equal9']} |")
    lines += ["", "## Full-V0 selected configurations (optimistic diagnostics)", "",
        "Each configuration below was selected and scored on all V0 rows. Use these exact IDs and weights only as development candidates; scores are optimistic after selection.", ""]
    for size in range(MIN_SIZE, MAX_SIZE+1):
        lines += [f"### {size} checkpoints", "", "| Method | Rank | Checkpoint IDs | Weights | Full-V0 BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
        for method in ("equal", "optimized"):
            for rank, rec in enumerate(results["full_v0"][method][str(size)], 1):
                members = ", ".join(rec["checkpoint_ids"])
                weights = ", ".join(f"{sid}:{w:.4f}" for sid, w in rec["weights"].items())
                lines.append(f"| {method} | {rank} | {members} | {weights} | {rec['balanced_accuracy']:.4%} | {rec['accuracy']:.4%} | {rec['log_loss']:.6f} | {rec['corrected_vs_equal9']} / {rec['introduced_vs_equal9']} | {(rec['balanced_accuracy']-baseline['full_v0']['balanced_accuracy'])*100:+.3f} pp |")
        lines.append("")
    lines += ["## Best three distinct cross-fitted candidates", "",
        "Candidate selection is ranked by pooled OOF balanced accuracy across all three fit-fold ranks. Among candidates within 0.5 percentage points of the leader, prediction disagreement is used to prefer complementary procedures. Diversity does not alter the reported accuracy ranking.", "",
        "| Rank | Size | Method | Fit-fold rank | OOF BA | OOF accuracy | OOF log loss | Disagreement vs prior selected |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for i, rec in enumerate(results["best_three_distinct"], 1):
        diversity = "—" if i == 1 else f"{rec['disagreement_vs_previous']:.2%}"
        lines.append(f"| {i} | {rec['size']} | {rec['method']} | {rec['rank']} | {rec['balanced_accuracy']:.4%} | {rec['accuracy']:.4%} | {rec['log_loss']:.6f} | {diversity} |")
        lines.append(f"|  |  | Per-fold selected members: {json.dumps(rec['fold_members'], separators=(',', ':'))} |  |  |  |  |  |")
    lines += ["", "## Per-label metrics", "",
        "Per-label precision, recall and F1 are stored in `per_label_metrics.csv` for every pooled OOF rank/size/method row and every full-V0 top-three configuration. Prediction and confusion matrices are stored in the JSON metrics and Parquet outputs.", "",
        "## Assessment", "",
        f"- **Best cross-fitted procedure:** {results['conclusions']['best_overall']}.",
        f"- **Best equal-weight procedure:** {results['conclusions']['best_equal']}.",
        f"- **Best optimized-weight procedure:** {results['conclusions']['best_optimized']}.",
        f"- **Best full-V0 diagnostic:** {results['conclusions']['best_full_v0']}.",
        f"- **Compared with EQUAL9:** {results['conclusions']['comparison']}",
        f"- **Kaggle recommendation:** {results['conclusions']['kaggle_recommendation']}",
        "No Kaggle submission was made. A submission would be a follow-up decision; full-V0 diagnostics are optimistic, and the cross-fit procedures select different members in different folds.", "",
        "## Reproducibility outputs", "",
        "- `experiment_config.json`: Trust List version/hash, candidate IDs and probability-cache provenance.",
        "- `fold_selections.json`: top-three fold-specific members and weights for every size and method.",
        "- `full_v0_selections.json`: exact full-V0 top-three configurations and weights.",
        "- `metrics.json` and `metrics.csv`: cross-fitted and full-V0 scores, corrections, errors and per-label metrics.",
        "- `folds/fold_XX.npz`: held-out probabilities from fold-specific selections.",
        "- `predictions_oof.parquet`, `predictions_full_v0.parquet`: row-aligned reproducible predictions and probabilities.",
        "- `fold_assignments.csv`: reused five prompt-group fold IDs.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(*, root: str | Path = ROOT, trust_list: str | None = None,
        output_dir: str | Path = OUT, report_path: str | Path = REPORT,
        beam_width: int = BEAM_WIDTH, swap_seeds: int = SWAP_SEEDS,
        swap_passes: int = SWAP_PASSES) -> dict:
    root = Path(root).resolve()
    output = Path(output_dir); output = output if output.is_absolute() else root / output
    report = Path(report_path); report = report if report.is_absolute() else root / report
    output.mkdir(parents=True, exist_ok=True)
    (output / "folds").mkdir(exist_ok=True)
    pointer, trust_path, trust, unique, v0, y, y_text, matrices, cache, fold_id, folds = _load(root, trust_list, output)
    candidate_ids = sorted(unique.sequence_id.tolist(), key=lambda x: int(x[1:]))
    # Search candidates exclude EQUAL9-only members unless they are in the 22-ID v2.2 pool.
    search_matrices = {sid: matrices[sid] for sid in candidate_ids}
    baseline_p = np.mean(np.stack([matrices[sid] for sid in EQUAL9]), axis=0)
    baseline_p /= baseline_p.sum(axis=1, keepdims=True)
    baseline_full = _prob_metrics(y, baseline_p, EQUAL9, [1/len(EQUAL9)]*len(EQUAL9))

    fold_selections: dict[str, dict] = {}; oof_probs = {
        (method, size, rank): np.zeros((len(v0), len(LABELS)), dtype=np.float64)
        for method in ("equal", "optimized") for size in range(MIN_SIZE, MAX_SIZE+1) for rank in range(1, TOP_K+1)}
    fold_by_rank: dict[str, dict] = {}
    for fold_record in folds:
        fold = fold_record["fold"]; fit = fold_record["fit_indexes"]; held = fold_record["held_indexes"]
        selection = {}
        fold_arrays = {"held_indexes": held}
        for method in ("equal", "optimized"):
            selected = _search(search_matrices, y, fit, method, beam_width=beam_width,
                               swap_seeds=swap_seeds, swap_passes=swap_passes)
            selection[method] = {}
            for size, records in selected.items():
                selection[method][str(size)] = []
                for rank in range(1, TOP_K+1):
                    rec = records[rank-1]
                    combo = tuple(rec["checkpoint_ids"]); weights = np.asarray(list(rec["weights"].values()))
                    hp = _blend(matrices, combo, held, weights)
                    oof_probs[(method,size,rank)][held] = hp
                    fold_arrays[f"{method}_{size}_rank{rank}"] = hp.astype(np.float32)
                    selection[method][str(size)].append({k:v for k,v in rec.items() if k not in {"y_pred","confusion_matrix"}})
        fold_selections[str(fold)] = {"fit_rows": int(len(fit)), "held_rows": int(len(held)),
            "fit_prompt_groups": int(v0.iloc[fit].prompt_id.astype(str).nunique()),
            "held_prompt_groups": int(v0.iloc[held].prompt_id.astype(str).nunique()), "selections": selection}
        np.savez_compressed(output / "folds" / f"fold_{fold:02d}.npz", **fold_arrays)
        _atomic_json(output / "folds" / f"fold_{fold:02d}.json", {"fold": fold,
            "held_indexes": held.tolist(), "fit_rows": len(fit), "selections": selection})

    # Full-V0 development-only search, independent for each weighting method.
    all_idx = np.arange(len(v0))
    full_v0: dict[str, dict[str, list[dict]]] = {}
    full_probs = {"EQUAL9": baseline_p}
    for method in ("equal", "optimized"):
        selected = _search(search_matrices, y, all_idx, method, beam_width=beam_width,
                           swap_seeds=swap_seeds, swap_passes=swap_passes)
        full_v0[method] = {}
        for size, records in selected.items():
            full_v0[method][str(size)] = []
            for rank, rec in enumerate(records, 1):
                cleaned = {k:v for k,v in rec.items() if k not in {"y_pred","confusion_matrix"}}
                full_v0[method][str(size)].append(cleaned)
                combo = tuple(rec["checkpoint_ids"]); weights = np.asarray(list(rec["weights"].values()))
                full_probs[f"{method}_{size}_rank{rank}"] = _blend(matrices, combo, all_idx, weights)

    base_oof = baseline_p.copy()
    cross_rows = []; per_label_rows = []; oof_metric_records = {}
    for method in ("equal", "optimized"):
        for size in range(MIN_SIZE, MAX_SIZE+1):
            for rank in range(1, TOP_K+1):
                key = (method,size,rank); p = oof_probs[key]
                rec = _prob_metrics(y, p, [], [])
                base_pred = base_oof.argmax(axis=1); pred = p.argmax(axis=1)
                rec["corrected_vs_equal9"] = int(np.sum((base_pred != y) & (pred == y)))
                rec["introduced_vs_equal9"] = int(np.sum((base_pred == y) & (pred != y)))
                rec["disagreement_vs_equal9"] = int(np.sum(pred != base_pred))
                fold_ba=[]
                for f in range(1,6):
                    held=folds[f-1]["held_indexes"]
                    fold_ba.append(float(_prob_metrics(y[held],p[held],[],[])['balanced_accuracy']))
                rec["fold_balanced_accuracy"] = fold_ba
                combos=[fold_selections[str(f)]["selections"][method][str(size)][rank-1]["checkpoint_ids"] for f in range(1,6)]
                jaccards=[]
                for i in range(len(combos)):
                    for j in range(i+1,len(combos)):
                        a,b=set(combos[i]),set(combos[j]); jaccards.append(len(a&b)/len(a|b))
                rec["mean_member_jaccard"] = float(np.mean(jaccards)) if jaccards else 1.0
                rec["fold_ba_std"] = float(np.std(fold_ba,ddof=0))
                row = {"size": size, "method": method, "rank": rank,
                    **{k:rec[k] for k in ("accuracy","balanced_accuracy","log_loss","corrected_vs_equal9","introduced_vs_equal9","disagreement_vs_equal9","fold_balanced_accuracy","fold_ba_std","mean_member_jaccard")}}
                row["fold_members_and_weights"] = json.dumps([
                    {"fold": f, "checkpoint_ids": fold_selections[str(f)]["selections"][method][str(size)][rank-1]["checkpoint_ids"],
                     "weights": fold_selections[str(f)]["selections"][method][str(size)][rank-1]["weights"]}
                    for f in range(1, 6)], separators=(",", ":"))
                cross_rows.append(row); oof_metric_records[f"{method}_{size}_rank{rank}"] = rec
                for label in LABELS:
                    per_label_rows.append({"scope":"cross_fitted","method":method,"size":size,"rank":rank,
                        "checkpoint_ids_by_fold":json.dumps(combos),
                        "label":label,**rec["per_label"][label]})
    full_metric_records = {}
    for method in ("equal","optimized"):
        for size, records in full_v0[method].items():
            for rank, cfg in enumerate(records,1):
                name=f"{method}_{size}_rank{rank}"; p=full_probs[name]
                rec=_prob_metrics(y,p,cfg["checkpoint_ids"],list(cfg["weights"].values()))
                base_pred=baseline_p.argmax(axis=1); pred=p.argmax(axis=1)
                rec["corrected_vs_equal9"]=int(np.sum((base_pred!=y)&(pred==y)))
                rec["introduced_vs_equal9"]=int(np.sum((base_pred==y)&(pred!=y)))
                rec["disagreement_vs_equal9"]=int(np.sum(pred!=base_pred))
                cfg.update({k:v for k,v in rec.items() if k not in {"y_pred","confusion_matrix","checkpoint_ids","weights","checkpoint_count"}})
                full_metric_records[name]=rec
                for label in LABELS:
                    per_label_rows.append({"scope":"full_v0_diagnostic","method":method,"size":int(size),"rank":rank,
                        "checkpoint_ids_by_fold":json.dumps(cfg["checkpoint_ids"]),"label":label,**rec["per_label"][label]})

    # OOF rank-1 procedures provide the comparable cross-fitted candidate set.
    candidates=[]
    for row in cross_rows:
        p=oof_probs[(row["method"],row["size"],row["rank"])]
        candidates.append({**row,"probabilities":p,"pred":p.argmax(axis=1),
            "fold_members":[fold_selections[str(f)]["selections"][row["method"]][str(row["size"])][row["rank"]-1]["checkpoint_ids"] for f in range(1,6)]})
    candidates.sort(key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["fold_ba_std"],-r["mean_member_jaccard"],-r["accuracy"],r["size"],r["method"]))
    chosen=[]
    for candidate in candidates:
        if not chosen:
            chosen.append(candidate); continue
        if candidate["balanced_accuracy"] < chosen[0]["balanced_accuracy"] - 0.005:
            continue
        diversity=min(float(np.mean(candidate["pred"]!=prev["pred"])) for prev in chosen)
        candidate["disagreement_vs_previous"]=round(diversity,6)
        if all(not np.array_equal(candidate["pred"],prev["pred"]) for prev in chosen):
            chosen.append(candidate)
        if len(chosen)==3: break
    # If all near-best candidates predict identically, fill with best remaining distinct procedures.
    if len(chosen)<3:
        for candidate in candidates:
            if all(not np.array_equal(candidate["pred"],prev["pred"]) for prev in chosen):
                candidate["disagreement_vs_previous"]=min(float(np.mean(candidate["pred"]!=prev["pred"])) for prev in chosen) if chosen else 0.0
                chosen.append(candidate)
            if len(chosen)==3: break
    best_equal=min((r for r in cross_rows if r["method"]=="equal"),key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["fold_ba_std"],-r["mean_member_jaccard"]))
    best_opt=min((r for r in cross_rows if r["method"]=="optimized"),key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["fold_ba_std"],-r["mean_member_jaccard"]))
    # write row-aligned probability bundles
    common={"id":v0.id.astype(str).tolist(),"prompt_id":v0.prompt_id.astype(str).tolist(),"y_true":y_text,
            "fold_id":fold_id.tolist(),"pred::EQUAL9":np.asarray(LABELS,dtype=object)[base_oof.argmax(1)]}
    for i,label in enumerate(LABELS): common[f"prob::EQUAL9::{label}"]=base_oof[:,i]
    for method in ("equal","optimized"):
        for size in range(MIN_SIZE,MAX_SIZE+1):
            for rank in range(1,TOP_K+1):
                p=oof_probs[(method,size,rank)]; name=f"{method}_{size}_rank{rank}"
                common[f"pred::{name}"]=np.asarray(LABELS,dtype=object)[p.argmax(1)]
                for i,label in enumerate(LABELS): common[f"prob::{name}::{label}"]=p[:,i]
    pd.DataFrame(common).to_parquet(output/"predictions_oof.parquet",index=False)
    full_frame={"id":v0.id.astype(str).tolist(),"prompt_id":v0.prompt_id.astype(str).tolist(),"y_true":y_text,
                "pred::EQUAL9":np.asarray(LABELS,dtype=object)[baseline_p.argmax(1)]}
    for i,label in enumerate(LABELS): full_frame[f"prob::EQUAL9::{label}"]=baseline_p[:,i]
    for name,p in full_probs.items():
        if name=="EQUAL9": continue
        full_frame[f"pred::{name}"]=np.asarray(LABELS,dtype=object)[p.argmax(1)]
        for i,label in enumerate(LABELS): full_frame[f"prob::{name}::{label}"]=p[:,i]
    pd.DataFrame(full_frame).to_parquet(output/"predictions_full_v0.parquet",index=False)
    baseline_oof_rec=_prob_metrics(y,base_oof,EQUAL9,[1/len(EQUAL9)]*len(EQUAL9))
    baseline={"checkpoint_ids":EQUAL9,"weights":{x:1/len(EQUAL9) for x in EQUAL9},
        "cross_fitted":baseline_oof_rec,"full_v0":baseline_full}
    best_crossfit = max(cross_rows, key=lambda r:r["balanced_accuracy"])
    if best_crossfit["balanced_accuracy"] > baseline_oof_rec["balanced_accuracy"]:
        kaggle_recommendation = (f"Potentially worth a later submission: {best_crossfit['method']} {best_crossfit['size']}-checkpoint selection improves OOF BA by {(best_crossfit['balanced_accuracy']-baseline_oof_rec['balanced_accuracy'])*100:+.3f} pp; verify selection stability and test predictions first.")
    else:
        kaggle_recommendation = (f"No candidate merits submission from this search: the best is {best_crossfit['method']} {best_crossfit['size']} at {best_crossfit['balanced_accuracy']:.4%} OOF BA, {(best_crossfit['balanced_accuracy']-baseline_oof_rec['balanced_accuracy'])*100:.3f} pp below EQUAL9.")
    best_full_method, best_full_size, best_full_record = max(
        ((method, int(size), rec) for method in ("equal", "optimized")
         for size, records in full_v0[method].items() for rec in records),
        key=lambda item: item[2]["balanced_accuracy"])
    results={"trust_list_version":pointer["version"],"trust_list_path":str(trust_path.relative_to(root)),
        "trust_list_sha256":_file_sha(trust_path),"candidate_count":len(candidate_ids),"candidate_ids":candidate_ids,
        "search":{"beam_width":beam_width,"swap_seeds":swap_seeds,"swap_passes":swap_passes,
                  "l2_shrinkage":L2,"objective":"log_loss + l2 * squared_distance_to_equal_weights"},
        "baseline":baseline,"cross_fitted_rows":cross_rows,"cross_fitted_metrics":oof_metric_records,
        "full_v0":full_v0,"full_v0_metrics":full_metric_records,"fold_selections":fold_selections,
        "best_three_distinct":[{k:v for k,v in row.items() if k not in {"probabilities","pred"}} for row in chosen],
        "conclusions":{"best_overall":f"{best_crossfit['method']} {best_crossfit['size']}-checkpoint fit-fold rank {best_crossfit['rank']} at {best_crossfit['balanced_accuracy']:.4%} OOF BA",
            "best_equal":f"equal {best_equal['size']}-checkpoint fit-fold rank {best_equal['rank']} at {best_equal['balanced_accuracy']:.4%} OOF BA",
            "best_optimized":f"optimized {best_opt['size']}-checkpoint fit-fold rank {best_opt['rank']} at {best_opt['balanced_accuracy']:.4%} OOF BA",
            "best_full_v0":f"{best_full_method} {best_full_size}-checkpoint rank {full_v0[best_full_method][str(best_full_size)].index(best_full_record)+1} at {best_full_record['balanced_accuracy']:.4%} BA ({(best_full_record['balanced_accuracy']-baseline_full['balanced_accuracy'])*100:+.3f} pp vs EQUAL9; optimistic diagnostic)",
            "comparison":f"Best cross-fitted candidate changes balanced accuracy by {(best_crossfit['balanced_accuracy']-baseline_oof_rec['balanced_accuracy'])*100:+.3f} pp versus fixed EQUAL9 OOF ({baseline_oof_rec['balanced_accuracy']:.4%}); full-V0 baseline is {baseline_full['balanced_accuracy']:.4%}.",
            "kaggle_recommendation":kaggle_recommendation}}
    _atomic_json(output/"experiment_config.json",{"trust_list_version":pointer["version"],"trust_list_path":str(trust_path.relative_to(root)),
        "trust_list_sha256":results["trust_list_sha256"],"candidate_ids":candidate_ids,"candidate_sources":cache,
        "baseline":baseline,"fold_source":str(FOLD_SOURCE/"folds"),"fold_assignments":"fold_assignments.csv",
        "search":results["search"],"probability_order":list(LABELS)})
    _atomic_json(output/"fold_selections.json",fold_selections)
    _atomic_json(output/"full_v0_selections.json",full_v0)
    _atomic_json(output/"metrics.json",{"baseline":baseline,"cross_fitted":oof_metric_records,
        "cross_fitted_rows":cross_rows,"full_v0":full_metric_records,"full_v0_selections":full_v0,
        "best_three_distinct":results["best_three_distinct"],"conclusions":results["conclusions"]})
    pd.DataFrame(cross_rows).to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(per_label_rows).to_csv(output/"per_label_metrics.csv",index=False)
    pd.DataFrame({"id":v0.id.astype(str),"prompt_id":v0.prompt_id.astype(str),"fold":fold_id}).to_csv(output/"fold_assignments.csv",index=False)
    _write_report(report,results)
    summary={"report":str(report.relative_to(root)),"output_dir":str(output.relative_to(root)),
        "trust_list_version":pointer["version"],"candidate_count":len(candidate_ids),
        "baseline_oof_balanced_accuracy":baseline_oof_rec["balanced_accuracy"],
        "baseline_full_v0_balanced_accuracy":baseline_full["balanced_accuracy"],
        "best_three_distinct":results["best_three_distinct"],"conclusions":results["conclusions"]}
    print(json.dumps(summary,indent=2,default=str))
    return results


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-list")
    parser.add_argument("--output-dir",default=str(OUT))
    parser.add_argument("--report",default=str(REPORT))
    parser.add_argument("--beam-width",type=int,default=BEAM_WIDTH)
    parser.add_argument("--swap-seeds",type=int,default=SWAP_SEEDS)
    parser.add_argument("--swap-passes",type=int,default=SWAP_PASSES)
    args=parser.parse_args()
    run(trust_list=args.trust_list,output_dir=args.output_dir,report_path=args.report,
        beam_width=args.beam_width,swap_seeds=args.swap_seeds,swap_passes=args.swap_passes)


if __name__=="__main__":
    main()
