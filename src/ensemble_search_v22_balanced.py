"""Search Trust List 2.2 ensembles with direct balanced-accuracy optimization.

The input is the cached full-V0 probability roster. Fold-specific checkpoint
selection and all three weight methods are fit on four prompt-group folds and
evaluated only on the held-out fold. Full-V0 selections are diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .ensemble_precision_recall_overlap import EQUAL9, _sha as _file_sha
from .ensemble_search_v22 import (
    BEAM_WIDTH, L2, MAX_SIZE, MIN_SIZE, OUT, REPORT, SWAP_PASSES, SWAP_SEEDS,
    TOP_K, _atomic_json, _blend, _fit_weights, _id_key, _load, _prob_metrics,
    _rank_key, _search as _search_logloss,
)
from .v2_checkpoint_trust import LABELS

DEFAULT_TRUST = "docs/trust-list-2.2-v0005.csv"
METHODS = ("equal", "log_loss", "balanced_accuracy")
DE_POPULATION = 12
DE_GENERATIONS = 10
DE_TOLERANCE = 0.01


def _simplex(raw: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    total = float(weights.sum())
    return weights / total if total > 1e-15 else np.full(len(weights), 1.0 / len(weights))


def _stable_seed(seed: int, combo: tuple[str, ...], fold: int | str) -> int:
    payload = f"{seed}|{fold}|{'|'.join(combo)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _fit_balanced_weights(prob: np.ndarray, y: np.ndarray, *, seed: int,
                          l2: float = L2, population_size: int = DE_POPULATION,
                          generations: int = DE_GENERATIONS) -> tuple[np.ndarray, dict]:
    """Optimize macro recall on argmax blends with deterministic differential evolution.

    The initial population explicitly contains equal and regularized-log-loss
    weights, plus seeded Dirichlet starts. The scalar objective preserves the
    required lexicographic order: balanced accuracy, accuracy, log loss, then
    squared distance from equal weights. The last term is only a stability
    preference when classification metrics tie.
    """
    members, rows, classes = prob.shape
    equal = np.full(members, 1.0 / members, dtype=np.float64)
    if members == 1:
        return equal, {"success": True, "message": "single member", "nfev": 1,
                       "nit": 0, "population_size": 1, "generations": 0,
                       "starts": ["equal", "log_loss"]}
    logloss = _fit_weights(prob, y, l2=l2)
    rng = np.random.default_rng(seed)
    starts = [equal, _simplex(logloss)]
    starts.extend(rng.dirichlet(np.ones(members) * 1.25) for _ in range(max(0, population_size - 2)))
    init = np.clip(np.stack(starts), 1e-12, 1.0)
    init /= init.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=np.int64)
    supports = np.bincount(y, minlength=classes).astype(np.float64)
    target = np.arange(rows)
    ba_quantum = 1.0 / (classes * max(float(supports.max()), 1.0))
    accuracy_tie_scale = ba_quantum * 0.05
    loss_tie_scale = accuracy_tie_scale / max(rows * 40.0, 1.0)
    simplicity_tie_scale = loss_tie_scale * 0.1
    best_observed = {"balanced_accuracy": -1.0, "accuracy": -1.0,
                     "log_loss": float("inf"), "weights": equal.tolist()}

    def objective(raw: np.ndarray) -> float:
        weights = _simplex(raw)
        blended = np.einsum("m,mnc->nc", weights, prob, optimize=True)
        prediction = blended.argmax(axis=1)
        recalls = np.bincount(y[prediction == y], minlength=classes) / np.maximum(supports, 1)
        ba = float(recalls.mean())
        accuracy = float(np.mean(prediction == y))
        loss = float(-np.log(np.clip(blended[target, y], 1e-15, 1.0)).mean())
        distance = float(np.sum((weights - equal) ** 2))
        candidate = (ba, accuracy, -loss, -distance)
        current = (best_observed["balanced_accuracy"], best_observed["accuracy"],
                   -best_observed["log_loss"], -float(np.sum((np.asarray(best_observed["weights"]) - equal) ** 2)))
        if candidate > current:
            best_observed.update(balanced_accuracy=ba, accuracy=accuracy,
                                 log_loss=loss, weights=weights.tolist())
        # Scales are based on the fitting fold's minimum BA and accuracy steps,
        # so lower-priority terms cannot outweigh a higher-priority change.
        return (-ba + accuracy_tie_scale * (1.0 - accuracy)
                + loss_tie_scale * loss + simplicity_tie_scale * distance)

    result = differential_evolution(
        objective, bounds=[(0.0, 1.0)] * members,
        seed=seed, init=init, maxiter=generations, tol=DE_TOLERANCE, atol=0.0,
        polish=False, updating="immediate", workers=1, mutation=(0.5, 1.0),
        recombination=0.7,
    )
    weights = _simplex(result.x)
    # Keep the best actual lexicographic point, since DE's final raw scalar
    # objective is merely a numerical encoding of that ordering.
    observed = np.asarray(best_observed["weights"], dtype=np.float64)
    best_pair = (best_observed["balanced_accuracy"], best_observed["accuracy"],
                 -best_observed["log_loss"], -float(np.sum((observed - equal) ** 2)))
    blended = np.einsum("m,mnc->nc", weights, prob, optimize=True)
    pred = blended.argmax(axis=1)
    ba = float(np.mean(np.bincount(y[pred == y], minlength=classes) / np.maximum(supports, 1)))
    acc = float(np.mean(pred == y))
    loss = float(-np.log(np.clip(blended[target, y], 1e-15, 1.0)).mean())
    result_pair = (ba, acc, -loss, -float(np.sum((weights - equal) ** 2)))
    if best_pair > result_pair:
        weights = observed
    return weights, {
        "success": bool(result.success), "message": str(result.message),
        "nfev": int(result.nfev), "nit": int(result.nit),
        "population_size": int(len(init)), "generations": int(result.nit),
        "seed": int(seed), "starts": ["equal", "regularized_log_loss", "seeded_dirichlet"],
        "best_objective_balanced_accuracy": float(best_observed["balanced_accuracy"]),
        "best_objective_accuracy": float(best_observed["accuracy"]),
        "best_objective_log_loss": float(best_observed["log_loss"]),
        "weight_distance_from_equal": float(np.sum((weights - equal) ** 2)),
        "l2_tie_stability": float(l2),
    }


def _search_balanced(matrices: dict[str, np.ndarray], y: np.ndarray, indexes: np.ndarray,
                     *, seed: int, fold_key: int | str,
                     beam_width: int = BEAM_WIDTH, swap_seeds: int = SWAP_SEEDS,
                     swap_passes: int = SWAP_PASSES, population_size: int = DE_POPULATION,
                     generations: int = DE_GENERATIONS) -> tuple[dict[int, list[dict]], dict]:
    """Same deterministic beam/swap member search, scoring each set by fitted BA weights."""
    candidates = sorted(matrices, key=lambda x: int(x[1:]))
    cache: dict[tuple[str, ...], dict] = {}
    started = time.perf_counter()

    def evaluate(raw: tuple[str, ...]) -> dict:
        combo = tuple(sorted(raw, key=lambda x: int(x[1:])))
        if combo in cache:
            return cache[combo]
        member_prob = np.stack([matrices[s][indexes] for s in combo], axis=0)
        if len(combo) == 1:
            weights, diag = np.ones(1), {"success": True, "nfev": 1, "nit": 0,
                "population_size": 1, "generations": 0, "seed": _stable_seed(seed, combo, fold_key),
                "starts": ["equal", "regularized_log_loss"]}
        else:
            weights, diag = _fit_balanced_weights(member_prob, y[indexes],
                seed=_stable_seed(seed, combo, fold_key), population_size=population_size,
                generations=generations)
        p = np.einsum("m,mnc->nc", weights, member_prob, optimize=True)
        rec = _prob_metrics(y[indexes], p, list(combo), weights.tolist())
        rec["optimizer"] = diag
        cache[combo] = rec
        return rec

    beam: list[tuple[str, ...]] = [(sid,) for sid in candidates]
    for size in range(2, MIN_SIZE):
        layer = set()
        for node in beam:
            for sid in candidates:
                if sid not in node:
                    layer.add(tuple(sorted((*node, sid), key=lambda x: int(x[1:]))))
        beam = sorted(layer, key=lambda c: _rank_key(evaluate(c)))
        if size == MIN_SIZE - 1:
            beam = beam[:beam_width]

    top: dict[int, list[dict]] = {}
    for size in range(MIN_SIZE, MAX_SIZE + 1):
        generated = set()
        for node in beam:
            for sid in candidates:
                if sid not in node:
                    generated.add(tuple(sorted((*node, sid), key=lambda x: int(x[1:]))))
        ranked = sorted(generated, key=lambda c: _rank_key(evaluate(c)))
        local = []
        for current in ranked[:swap_seeds]:
            current_rec = evaluate(current)
            for _ in range(swap_passes):
                best, best_rec = current, current_rec
                for remove in current:
                    for add in candidates:
                        if add in current:
                            continue
                        proposal = tuple(sorted((set(current) - {remove}) | {add},
                                                key=lambda x: int(x[1:])))
                        rec = evaluate(proposal)
                        if _rank_key(rec) < _rank_key(best_rec):
                            best, best_rec = proposal, rec
                if best == current:
                    break
                current, current_rec = best, best_rec
            local.append(current)
        keep = set(ranked[:beam_width]) | set(local)
        beam = sorted((c for c in keep if len(c) == size),
                      key=lambda c: _rank_key(evaluate(c)))[:beam_width]
        top[size] = [evaluate(c) for c in sorted((c for c in keep if len(c) == size),
                    key=lambda c: _rank_key(evaluate(c)))[:TOP_K]]
    diagnostics = {"unique_subsets_scored": len(cache), "elapsed_seconds": time.perf_counter() - started,
        "beam_width": beam_width, "swap_seeds": swap_seeds, "swap_passes": swap_passes,
        "weight_optimizer": "scipy differential_evolution on normalized nonnegative simplex weights",
        "population_size": population_size, "max_generations": generations,
        "sum_unique_member_set_evaluations": int(sum(len(v) for v in top.values()))}
    return top, diagnostics


def _clean(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in {"y_pred", "confusion_matrix"}}


def _add_delta(rec: dict, base_pred: np.ndarray, y: np.ndarray) -> dict:
    pred = np.asarray(rec["y_pred"], dtype=np.int64)
    rec["corrected_vs_equal9"] = int(np.sum((base_pred != y) & (pred == y)))
    rec["introduced_vs_equal9"] = int(np.sum((base_pred == y) & (pred != y)))
    rec["disagreement_vs_equal9"] = int(np.sum(pred != base_pred))
    return rec


def _write_report(path: Path, result: dict) -> None:
    baseline = result["baseline"]
    lines = ["# Trust List 2.2 balanced-accuracy ensemble search", "",
        f"Trust List `{result['trust_list_version']}`: `{result['trust_list_path']}` (SHA-256 `{result['trust_list_sha256']}`).",
        "All base predictions are cached full-V0 probabilities. No model inference, training, Kaggle submission, production-ensemble or training-queue changes were made.",
        "Five prompt-group folds are reused from the existing ensemble analysis. For each fold, member selection and weights use only the other four folds; the held-out rows are scored once. Earlier checkpoint and roster selection may have used V0, so OOF figures are development diagnostics, not independent generalization estimates.", "",
        "## Baseline reproduction", "",
        f"EQUAL9 full-V0 balanced accuracy: **{baseline['full_v0']['balanced_accuracy']:.4%}** (reference 65.3898%); accuracy {baseline['full_v0']['accuracy']:.4%}; log loss {baseline['full_v0']['log_loss']:.6f}.",
        f"Fixed EQUAL9 OOF balanced accuracy: **{baseline['cross_fitted']['balanced_accuracy']:.4%}**. All 22 v0005 checkpoint IDs were loaded and considered.",
        f"Equal and log-loss search parameters were reused unchanged from `src/ensemble_search_v22.py` (beam width {BEAM_WIDTH}, swap seeds {SWAP_SEEDS}, swap passes {SWAP_PASSES}, size {MIN_SIZE}–{MAX_SIZE}). Their best scores are compared to the existing report in `baseline_comparison.json`.", "",
        "## OOF top three by size and method", "",
        "Rows are rank-ordered by fitting-fold balanced accuracy, then accuracy and log loss. The table reports pooled held-out predictions; each fold's exact IDs and weights are in `fold_selections.json`.", "",
        "| Size | Method | Rank | OOF balanced accuracy | OOF accuracy | OOF log loss | Δ BA vs EQUAL9 | Corrected / introduced | Mean member Jaccard |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["cross_fitted_rows"]:
        lines.append(f"| {r['size']} | {r['method']} | {r['rank']} | {r['balanced_accuracy']:.4%} | {r['accuracy']:.4%} | {r['log_loss']:.6f} | {(r['balanced_accuracy']-baseline['cross_fitted']['balanced_accuracy'])*100:+.3f} pp | {r['corrected_vs_equal9']} / {r['introduced_vs_equal9']} | {r['mean_member_jaccard']:.3f} |")
    lines += ["", "### Fold-specific members and weights", "",
        "Each cell was selected using only that fold's four fitting partitions. Weights are the learned values for that fold.", "",
        "| Size | Method | Rank | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |", "|---:|---|---:|---|---|---|---|---|"]
    for method in METHODS:
        for size in range(MIN_SIZE, MAX_SIZE + 1):
            for rank in range(1, TOP_K + 1):
                cells = []
                for fold in range(1, 6):
                    rec = result["fold_selections"][str(fold)]["methods"][method][str(size)][rank - 1]
                    weights = rec["weights"]
                    cells.append(", ".join(f"{sid}:{weights[sid]:.4f}" for sid in rec["checkpoint_ids"]))
                lines.append(f"| {size} | {method} | {rank} | " + " | ".join(cells) + " |")
    lines += ["", "## Full-V0 top three by size and method", "",
        "These are development-only, in-sample selection diagnostics and are optimistic. The table includes exact members and weights for reproducibility.", ""]
    for size in range(MIN_SIZE, MAX_SIZE + 1):
        lines += [f"### {size} checkpoints", "", "| Method | Rank | Checkpoints | Weights | BA | Accuracy | Log loss | Corrected / introduced | Δ BA vs EQUAL9 |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            for rank, r in enumerate(result["full_v0"][method][str(size)], 1):
                members = ", ".join(r["checkpoint_ids"])
                weights = ", ".join(f"{sid}={w:.4f}" for sid, w in r["weights"].items())
                lines.append(f"| {method} | {rank} | {members} | {weights} | {r['balanced_accuracy']:.4%} | {r['accuracy']:.4%} | {r['log_loss']:.6f} | {r['corrected_vs_equal9']} / {r['introduced_vs_equal9']} | {(r['balanced_accuracy']-baseline['full_v0']['balanced_accuracy'])*100:+.3f} pp |")
        lines.append("")
    lines += ["## Three strongest distinct cross-fitted candidates", "",
        "All candidates below are ranked by OOF balanced accuracy. Distinctness is measured by disagreement of pooled held-out predictions.", "",
        "| Rank | Size | Method | OOF BA | OOF accuracy | OOF log loss | Disagreement vs previous |", "|---:|---:|---|---:|---:|---:|---:|"]
    for i, r in enumerate(result["best_three_distinct"], 1):
        disagreement = "—" if i == 1 else f"{r['disagreement_vs_previous']:.2%}"
        lines.append(f"| {i} | {r['size']} | {r['method']} | {r['balanced_accuracy']:.4%} | {r['accuracy']:.4%} | {r['log_loss']:.6f} | {disagreement} |")
        lines.append(f"|  |  | Fold members: `{json.dumps(r['fold_members'], separators=(',', ':'))}` |  |  |  |  |")
    lines += ["", "## Per-label performance, corrections and regressions", "",
        "The table compares fixed EQUAL9 with the best OOF balanced-accuracy and log-loss-weighted candidates. Per-label results for every size, method and rank, including full-V0 diagnostics, are in `per_label_metrics.csv`.", "",
        "| Label | EQUAL9 P / R / F1 | Best BA candidate P / R / F1 | Best log-loss candidate P / R / F1 |", "|---|---:|---:|---:|"]
    best_balanced = max((r for r in result["cross_fitted_rows"] if r["method"] == "balanced_accuracy"),
                        key=lambda r: (r["balanced_accuracy"], r["accuracy"], -r["log_loss"]))
    best_logloss = max((r for r in result["cross_fitted_rows"] if r["method"] == "log_loss"),
                       key=lambda r: (r["balanced_accuracy"], r["accuracy"], -r["log_loss"]))
    equal9_labels = baseline["cross_fitted"]["per_label"]
    balanced_labels = result["cross_fitted_metrics"][f"balanced_accuracy_{best_balanced['size']}_rank{best_balanced['rank']}"]["per_label"]
    logloss_labels = result["cross_fitted_metrics"][f"log_loss_{best_logloss['size']}_rank{best_logloss['rank']}"]["per_label"]
    for label in LABELS:
        columns = []
        for metric_set in (equal9_labels, balanced_labels, logloss_labels):
            label_metrics = metric_set[label]
            columns.append(f"{label_metrics['precision']:.3f} / {label_metrics['recall']:.3f} / {label_metrics['f1']:.3f}")
        lines.append(f"| {label} | " + " | ".join(columns) + " |")
    lines += ["", "Error corrections/introductions versus EQUAL9 are in each OOF/full-V0 metric row and the row-level Parquet predictions.", "",
        "## Weight and member stability", "",
        "`weight_stability.csv` reports the five-fold selected-member frequency and each member's weight mean/standard deviation (absent members have weight zero). `member_stability.csv` contains pairwise fold Jaccard scores and per-label checkpoint selection frequencies.", "",
        "## Search coverage and convergence", "",
        f"All {result['candidate_count']} Trust List candidates were included. Unique subsets evaluated:", "",
        "| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Full V0 | Total |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        counts = [int(x["unique_subsets_scored"]) for x in result["search_coverage"][method]]
        lines.append(f"| {method} | " + " | ".join(f"{x:,}" for x in counts[:5])
                     + f" | {counts[5]:,} | {sum(counts):,} |")
    optimizer_rows = list(result["optimizer_convergence"].values())
    converged = sum(bool(x.get("success")) for x in optimizer_rows)
    nfev_values = [int(x.get("nfev", 0)) for x in optimizer_rows]
    failures = [key for key, value in result["optimizer_convergence"].items() if not value.get("success")]
    lines += ["", f"Balanced-weight differential evolution used equal and regularized log-loss starts plus seeded Dirichlet starts (population {DE_POPULATION}, maximum {DE_GENERATIONS} generations, fixed seeds). It converged in **{converged}/{len(optimizer_rows)}** reported top configurations; median function evaluations were **{np.median(nfev_values):.0f}** (range {min(nfev_values)}–{max(nfev_values)}). Non-converged records: {', '.join(failures) if failures else 'none'}. Full per-configuration status and counts are in `optimizer_convergence.json`.",
        "The derivative-free optimizer uses macro-average recall from blended-probability argmax as its primary objective, accuracy then log loss as tie-breaks, and a very small distance-to-equal regularizer only for remaining ties.", "",
        "## Assessment", "",
        f"- Best OOF candidate: **{result['conclusions']['best_oof']}**.",
        f"- Balanced-accuracy search versus log-loss weight search: {result['conclusions']['balanced_vs_log_loss']}",
        f"- Submission recommendation: {result['conclusions']['recommendation']}",
        "- Full-V0 metrics are development diagnostics because the same V0 data participates in checkpoint/trust-list history and full-V0 candidate selection.", "",
        "## Reproducibility artifacts", "",
        "- `experiment_config.json`: trust-list, candidate and cache hashes, folds, optimizer settings and seeds.",
        "- `fold_selections.json`, `full_v0_selections.json`: exact checkpoint memberships and weights.",
        "- `predictions_oof.parquet`, `predictions_full_v0.parquet`: row-aligned probabilities and predicted labels.",
        "- `metrics.json`, `metrics.csv`, `per_label_metrics.csv`: aggregate and per-class scores.",
        "- `optimizer_convergence.json`, `search_coverage.json`, `weight_stability.csv`, `member_stability.csv`: reproducible search diagnostics.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(*, root: str | Path = Path(__file__).resolve().parents[1],
        trust_list: str = DEFAULT_TRUST,
        output_dir: str | Path = "artifacts/analysis/ensemble_search_v22_balanced",
        report_path: str | Path = "docs/ensemble_search_v22_balanced.md",
        beam_width: int = BEAM_WIDTH, swap_seeds: int = SWAP_SEEDS,
        swap_passes: int = SWAP_PASSES, seed: int = 2026,
        population_size: int = DE_POPULATION, generations: int = DE_GENERATIONS) -> dict:
    root = Path(root).resolve()
    output = Path(output_dir); output = output if output.is_absolute() else root / output
    report = Path(report_path); report = report if report.is_absolute() else root / report
    output.mkdir(parents=True, exist_ok=True); (output / "folds").mkdir(exist_ok=True)
    pointer, trust_path, trust, unique, v0, y, y_text, matrices, cache, fold_id, folds = _load(root, trust_list, output)
    if not str(pointer.get("version", "")).endswith("v0005") or not trust_path.name.endswith("v0005.csv"):
        raise ValueError(f"latest Trust List pointer is {pointer.get('version')}, expected v0005")
    candidate_ids = sorted(unique.sequence_id.tolist(), key=lambda x: int(x[1:]))
    if len(candidate_ids) != 22:
        raise ValueError(f"expected all 22 v0005 checkpoints, found {len(candidate_ids)}")
    search_matrices = {sid: matrices[sid] for sid in candidate_ids}
    baseline_p = np.mean(np.stack([matrices[sid] for sid in EQUAL9]), axis=0)
    baseline_p /= baseline_p.sum(axis=1, keepdims=True)
    baseline_full = _prob_metrics(y, baseline_p, EQUAL9, [1 / len(EQUAL9)] * len(EQUAL9))
    oof_probs = {(m, n, r): np.zeros((len(v0), len(LABELS)), dtype=np.float64)
                 for m in METHODS for n in range(MIN_SIZE, MAX_SIZE + 1) for r in range(1, TOP_K + 1)}
    selections: dict[str, dict] = {}; coverage = {m: [] for m in METHODS}; optimizer_records = {}
    for fold_record in folds:
        fold, fit, held = fold_record["fold"], fold_record["fit_indexes"], fold_record["held_indexes"]
        selections[str(fold)] = {"fit_rows": int(len(fit)), "held_rows": int(len(held)), "methods": {}}
        arrays = {"held_indexes": held}
        equal_selected, equal_diag = _search_logloss(search_matrices, y, fit, "equal", beam_width=beam_width,
            swap_seeds=swap_seeds, swap_passes=swap_passes, return_diagnostics=True)
        logloss_selected, logloss_diag = _search_logloss(search_matrices, y, fit, "optimized", beam_width=beam_width,
            swap_seeds=swap_seeds, swap_passes=swap_passes, return_diagnostics=True)
        balanced_selected, balanced_diag = _search_balanced(search_matrices, y, fit, seed=seed,
            fold_key=fold, beam_width=beam_width, swap_seeds=swap_seeds, swap_passes=swap_passes,
            population_size=population_size, generations=generations)
        coverage["equal"].append({"fold": fold, **equal_diag})
        coverage["log_loss"].append({"fold": fold, **logloss_diag})
        coverage["balanced_accuracy"].append({"fold": fold, **balanced_diag})
        selected_by_method = {"equal": equal_selected, "log_loss": logloss_selected,
                              "balanced_accuracy": balanced_selected}
        for method, by_size in selected_by_method.items():
            selections[str(fold)]["methods"][method] = {}
            for size, records in by_size.items():
                selections[str(fold)]["methods"][method][str(size)] = []
                for rank in range(1, TOP_K + 1):
                    rec = records[rank - 1]
                    combo = tuple(rec["checkpoint_ids"])
                    weights = np.asarray(list(rec["weights"].values()), dtype=np.float64)
                    hp = _blend(matrices, combo, held, weights)
                    oof_probs[(method, size, rank)][held] = hp
                    arrays[f"{method}_{size}_rank{rank}"] = hp.astype(np.float32)
                    cleaned = _clean(rec)
                    selections[str(fold)]["methods"][method][str(size)].append(cleaned)
                    if method == "balanced_accuracy":
                        optimizer_records[f"fold{fold}_{size}_rank{rank}"] = rec.get("optimizer", {})
        np.savez_compressed(output / "folds" / f"fold_{fold:02d}.npz", **arrays)
        _atomic_json(output / "folds" / f"fold_{fold:02d}.json", {
            "fold": fold, "held_indexes": held.tolist(), "selections": selections[str(fold)]})

    full_v0: dict[str, dict[str, list[dict]]] = {m: {} for m in METHODS}
    full_probs = {"EQUAL9": baseline_p}
    full_per_label = []
    all_idx = np.arange(len(v0))
    full_balanced_diag = {}
    for method in METHODS:
        if method == "balanced_accuracy":
            selected, diag = _search_balanced(search_matrices, y, all_idx, seed=seed,
                fold_key="full_v0", beam_width=beam_width, swap_seeds=swap_seeds,
                swap_passes=swap_passes, population_size=population_size, generations=generations)
            full_balanced_diag = diag
        else:
            selected, diag = _search_logloss(search_matrices, y, all_idx,
                "equal" if method == "equal" else "optimized", beam_width=beam_width,
                swap_seeds=swap_seeds, swap_passes=swap_passes, return_diagnostics=True)
            coverage[method].append({"fold": "full_v0", **diag})
        full_v0[method] = {}
        for size, records in selected.items():
            full_v0[method][str(size)] = []
            for rank, rec in enumerate(records, 1):
                combo = tuple(rec["checkpoint_ids"])
                weights = np.asarray(list(rec["weights"].values()), dtype=np.float64)
                p = _blend(matrices, combo, all_idx, weights)
                full_probs[f"{method}_{size}_rank{rank}"] = p
                metric = _add_delta(_prob_metrics(y, p, list(combo), weights.tolist()),
                                    baseline_p.argmax(1), y)
                cfg = _clean(rec); cfg.update({k: v for k, v in metric.items()
                    if k not in {"y_pred", "confusion_matrix", "checkpoint_ids", "weights", "checkpoint_count"}})
                full_v0[method][str(size)].append(cfg)
                for label in LABELS:
                    full_per_label.append({"scope": "full_v0_diagnostic", "method": method,
                        "size": int(size), "rank": rank, "label": label,
                        **metric["per_label"][label]})
                if method == "balanced_accuracy":
                    optimizer_records[f"full_{size}_rank{rank}"] = rec.get("optimizer", {})
    coverage["balanced_accuracy"].append({"fold": "full_v0", **full_balanced_diag})

    base_pred = baseline_p.argmax(1)
    baseline_oof = _prob_metrics(y, baseline_p, EQUAL9, [1 / len(EQUAL9)] * len(EQUAL9))
    rows = []; per_label = []; oof_records = {}; stability_rows = []; member_rows = []
    for method in METHODS:
        for size in range(MIN_SIZE, MAX_SIZE + 1):
            for rank in range(1, TOP_K + 1):
                p = oof_probs[(method, size, rank)]
                rec = _add_delta(_prob_metrics(y, p, [], []), base_pred, y)
                fold_bas = []
                combos = []
                all_fold_weights = []
                for fold in range(1, 6):
                    held = folds[fold - 1]["held_indexes"]
                    fold_bas.append(_prob_metrics(y[held], p[held], [], [])["balanced_accuracy"])
                    selected_rec = selections[str(fold)]["methods"][method][str(size)][rank - 1]
                    combos.append(selected_rec["checkpoint_ids"])
                    all_fold_weights.append(selected_rec["weights"])
                jaccards = []
                for a in range(5):
                    for b in range(a + 1, 5):
                        aa, bb = set(combos[a]), set(combos[b])
                        jaccards.append(len(aa & bb) / len(aa | bb))
                rec["fold_balanced_accuracy"] = fold_bas
                rec["fold_ba_std"] = float(np.std(fold_bas))
                rec["mean_member_jaccard"] = float(np.mean(jaccards))
                row = {"method": method, "size": size, "rank": rank,
                    **{k: rec[k] for k in ("balanced_accuracy", "accuracy", "log_loss", "corrected_vs_equal9",
                        "introduced_vs_equal9", "disagreement_vs_equal9", "fold_ba_std", "mean_member_jaccard")}}
                rows.append(row); oof_records[f"{method}_{size}_rank{rank}"] = rec
                for label in LABELS:
                    per_label.append({"scope": "cross_fitted", "method": method, "size": size,
                        "rank": rank, "label": label, **rec["per_label"][label]})
                for sid in candidate_ids:
                    values = [float(w.get(sid, 0.0)) for w in all_fold_weights]
                    support = sum(sid in c for c in combos)
                    stability_rows.append({"method": method, "size": size, "rank": rank,
                        "sequence_id": sid, "selected_folds": support,
                        "selection_frequency": support / 5, "mean_weight_including_absent": float(np.mean(values)),
                        "weight_std_including_absent": float(np.std(values)),
                        "mean_weight_when_selected": float(np.mean([v for v in values if v > 0])) if any(v > 0 for v in values) else 0.0})
                member_rows.append({"method": method, "size": size, "rank": rank,
                    "fold_members": json.dumps(combos, separators=(",", ":")),
                    "pairwise_jaccard_mean": float(np.mean(jaccards)),
                    "pairwise_jaccard_min": float(np.min(jaccards))})

    # Candidate ranking uses OOF balanced accuracy; then accuracy and loss.
    candidates = []
    for r in rows:
        p = oof_probs[(r["method"], r["size"], r["rank"])]
        candidates.append({**r, "pred": p.argmax(1), "probabilities": p,
            "fold_members": [selections[str(f)]["methods"][r["method"]][str(r["size"])][r["rank"]-1]["checkpoint_ids"] for f in range(1, 6)]})
    candidates.sort(key=lambda r: (-r["balanced_accuracy"], -r["accuracy"], r["log_loss"], r["size"], r["method"], r["rank"]))
    chosen = []
    for candidate in candidates:
        if chosen and candidate["balanced_accuracy"] < chosen[0]["balanced_accuracy"] - 0.005:
            continue
        if all(not np.array_equal(candidate["pred"], prior["pred"]) for prior in chosen):
            candidate["disagreement_vs_previous"] = (min(float(np.mean(candidate["pred"] != prior["pred"])) for prior in chosen) if chosen else None)
            chosen.append(candidate)
        if len(chosen) == 3:
            break
    # Still deliver three distinct procedures when fewer than three lie within
    # 0.5 pp of the leader; the remaining candidates stay ranked by OOF BA.
    if len(chosen) < 3:
        for candidate in candidates:
            if all(not np.array_equal(candidate["pred"], prior["pred"]) for prior in chosen):
                candidate["disagreement_vs_previous"] = (min(
                    float(np.mean(candidate["pred"] != prior["pred"])) for prior in chosen) if chosen else None)
                chosen.append(candidate)
            if len(chosen) == 3:
                break

    cross_best = max(rows, key=lambda r: (r["balanced_accuracy"], r["accuracy"], -r["log_loss"]))
    method_best = {m: max((r for r in rows if r["method"] == m),
        key=lambda r: (r["balanced_accuracy"], r["accuracy"], -r["log_loss"])) for m in METHODS}
    full_candidates = [(m, int(n), rank, rec) for m in METHODS for n, recs in full_v0[m].items()
                       for rank, rec in enumerate(recs, 1)]
    best_full = max(full_candidates, key=lambda x: (x[3]["balanced_accuracy"], x[3]["accuracy"], -x[3]["log_loss"]))
    delta_balanced = method_best["balanced_accuracy"]["balanced_accuracy"] - method_best["log_loss"]["balanced_accuracy"]
    if cross_best["balanced_accuracy"] > baseline_oof["balanced_accuracy"] + 1e-12:
        recommendation = f"Balanced-accuracy candidate improves EQUAL9 OOF BA by {(cross_best['balanced_accuracy']-baseline_oof['balanced_accuracy'])*100:+.3f} pp; candidate for a separate review, subject to member stability and V0-selection exposure."
    else:
        recommendation = f"No submission recommended: best OOF candidate is {(cross_best['balanced_accuracy']-baseline_oof['balanced_accuracy'])*100:.3f} pp versus EQUAL9."
    results = {
        "trust_list_version": pointer["version"], "trust_list_path": str(trust_path.relative_to(root)),
        "trust_list_sha256": _file_sha(trust_path), "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "baseline": {"cross_fitted": baseline_oof, "full_v0": baseline_full},
        "cross_fitted_rows": rows, "cross_fitted_metrics": oof_records,
        "per_label_rows": per_label + full_per_label,
        "full_v0": full_v0, "fold_selections": selections,
        "best_three_distinct": [{k: v for k, v in x.items() if k not in {"pred", "probabilities"}} for x in chosen],
        "optimizer_convergence": optimizer_records, "search_coverage": coverage,
        "conclusions": {
            "best_oof": f"{cross_best['method']} size {cross_best['size']} rank {cross_best['rank']} at {cross_best['balanced_accuracy']:.4%} OOF BA",
            "balanced_vs_log_loss": f"Best balanced-optimized OOF BA {method_best['balanced_accuracy']['balanced_accuracy']:.4%} versus best log-loss-weight OOF BA {method_best['log_loss']['balanced_accuracy']:.4%} ({delta_balanced*100:+.3f} pp).",
            "recommendation": recommendation,
            "best_full_v0": f"{best_full[0]} size {best_full[1]} rank {best_full[2]} at {best_full[3]['balanced_accuracy']:.4%} (in-sample development diagnostic)",
        },
    }

    # Row-aligned probabilities for all three methods, sizes and top ranks.
    pred_frame = {"id": v0.id.astype(str), "prompt_id": v0.prompt_id.astype(str),
        "y_true": y_text, "fold_id": fold_id,
        "pred::EQUAL9": np.asarray(LABELS, dtype=object)[base_pred]}
    for i, label in enumerate(LABELS): pred_frame[f"prob::EQUAL9::{label}"] = baseline_p[:, i]
    for m in METHODS:
        for size in range(MIN_SIZE, MAX_SIZE + 1):
            for rank in range(1, TOP_K + 1):
                p = oof_probs[(m, size, rank)]; name = f"{m}_{size}_rank{rank}"
                pred_frame[f"pred::{name}"] = np.asarray(LABELS, dtype=object)[p.argmax(1)]
                for i, label in enumerate(LABELS): pred_frame[f"prob::{name}::{label}"] = p[:, i]
    pd.DataFrame(pred_frame).to_parquet(output / "predictions_oof.parquet", index=False)
    full_frame = {"id": v0.id.astype(str), "prompt_id": v0.prompt_id.astype(str), "y_true": y_text,
        "pred::EQUAL9": np.asarray(LABELS, dtype=object)[base_pred]}
    for i, label in enumerate(LABELS): full_frame[f"prob::EQUAL9::{label}"] = baseline_p[:, i]
    for name, p in full_probs.items():
        if name == "EQUAL9": continue
        full_frame[f"pred::{name}"] = np.asarray(LABELS, dtype=object)[p.argmax(1)]
        for i, label in enumerate(LABELS): full_frame[f"prob::{name}::{label}"] = p[:, i]
    pd.DataFrame(full_frame).to_parquet(output / "predictions_full_v0.parquet", index=False)

    best_equal_oof = max((r for r in rows if r["method"] == "equal"), key=lambda r: r["balanced_accuracy"])
    best_logloss_oof = max((r for r in rows if r["method"] == "log_loss"), key=lambda r: r["balanced_accuracy"])
    best_logloss_full = max((r for vals in full_v0["log_loss"].values() for r in vals), key=lambda r: r["balanced_accuracy"])
    baseline_compare = {"equal9_full_v0_target_balanced_accuracy": 0.653898,
        "equal9_full_v0_actual_balanced_accuracy": baseline_full["balanced_accuracy"],
        "equal9_full_v0_match": abs(baseline_full["balanced_accuracy"] - 0.653898) < 0.00005,
        "existing_report": "docs/ensemble_search_v22.md",
        "existing_report_equal9_full_v0_ba": 0.653898,
        "existing_report_best_equal_oof_ba": 0.616473,
        "actual_best_equal_oof_ba": best_equal_oof["balanced_accuracy"],
        "best_equal_oof_match": abs(best_equal_oof["balanced_accuracy"] - 0.616473) < 0.00005,
        "existing_report_best_log_loss_oof_ba": 0.621265,
        "actual_best_log_loss_oof_ba": best_logloss_oof["balanced_accuracy"],
        "best_log_loss_oof_match": abs(best_logloss_oof["balanced_accuracy"] - 0.621265) < 0.00005,
        "existing_report_best_log_loss_full_v0_ba": 0.629068,
        "actual_best_log_loss_full_v0_ba": best_logloss_full["balanced_accuracy"],
        "best_log_loss_full_v0_match": abs(best_logloss_full["balanced_accuracy"] - 0.629068) < 0.00005,
        "equal_search_reused_existing_code": True,
        "log_loss_search_reused_existing_code": True,
        "equal_logloss_scores": {m: {str(n): [
            {"rank": rank, "balanced_accuracy": rec["balanced_accuracy"],
             "accuracy": rec["accuracy"], "log_loss": rec["log_loss"]}
            for rank, rec in enumerate(full_v0[m][str(n)], 1)]
            for n in range(MIN_SIZE, MAX_SIZE + 1)} for m in ("equal", "log_loss")}}
    _atomic_json(output / "baseline_comparison.json", baseline_compare)
    _atomic_json(output / "experiment_config.json", {
        "trust_list_version": pointer["version"], "trust_list_path": results["trust_list_path"],
        "trust_list_sha256": results["trust_list_sha256"], "candidate_ids": candidate_ids,
        "candidate_sources": cache, "baseline": results["baseline"],
        "fold_source": "artifacts/analysis/ensemble_precision_recall_overlap/folds",
        "fold_assignment_path": str((output / "fold_assignments.csv").relative_to(root)),
        "search": {"methods": list(METHODS), "sizes": [MIN_SIZE, MAX_SIZE], "beam_width": beam_width,
            "swap_seeds": swap_seeds, "swap_passes": swap_passes, "seed": seed,
            "balanced_objective": "macro recall after argmax; tie by accuracy, log loss, then distance to equal weights",
            "optimizer": "scipy differential_evolution", "population_size": population_size,
            "generations": generations, "l2_tie_stability": L2},
        "class_order": list(LABELS)})
    _atomic_json(output / "fold_selections.json", selections)
    _atomic_json(output / "full_v0_selections.json", full_v0)
    _atomic_json(output / "optimizer_convergence.json", optimizer_records)
    _atomic_json(output / "search_coverage.json", coverage)
    _atomic_json(output / "metrics.json", {"baseline": results["baseline"],
        "cross_fitted": oof_records, "cross_fitted_rows": rows, "full_v0": full_v0,
        "best_three_distinct": results["best_three_distinct"], "conclusions": results["conclusions"]})
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(per_label + full_per_label).to_csv(output / "per_label_metrics.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output / "weight_stability.csv", index=False)
    pd.DataFrame(member_rows).to_csv(output / "member_stability.csv", index=False)
    pd.DataFrame({"id": v0.id.astype(str), "prompt_id": v0.prompt_id.astype(str), "fold": fold_id}).to_csv(output / "fold_assignments.csv", index=False)
    _write_report(report, results)
    summary = {"report": str(report.relative_to(root)), "output_dir": str(output.relative_to(root)),
        "trust_list_version": pointer["version"], "candidate_count": len(candidate_ids),
        "equal9_full_v0_balanced_accuracy": baseline_full["balanced_accuracy"],
        "equal9_oof_balanced_accuracy": baseline_oof["balanced_accuracy"],
        "best_three_distinct": results["best_three_distinct"], "conclusions": results["conclusions"]}
    print(json.dumps(summary, indent=2, default=str))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-list", default=DEFAULT_TRUST)
    parser.add_argument("--output-dir", default="artifacts/analysis/ensemble_search_v22_balanced")
    parser.add_argument("--report", default="docs/ensemble_search_v22_balanced.md")
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--swap-seeds", type=int, default=SWAP_SEEDS)
    parser.add_argument("--swap-passes", type=int, default=SWAP_PASSES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--population-size", type=int, default=DE_POPULATION)
    parser.add_argument("--generations", type=int, default=DE_GENERATIONS)
    args = parser.parse_args()
    run(trust_list=args.trust_list, output_dir=args.output_dir, report_path=args.report,
        beam_width=args.beam_width, swap_seeds=args.swap_seeds, swap_passes=args.swap_passes,
        seed=args.seed, population_size=args.population_size, generations=args.generations)


if __name__ == "__main__":
    main()
