"""E33 negative-evidence and candidate-label elimination analysis.

All base-model probabilities and E31 group assignments are cached. Rule
statistics and the learned elimination classifier are prompt-group cross-fit;
the 50 ``v0_group`` scenarios are reused V0 robustness diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from . import ensemble_catalog as catalog
from .e20 import hungarian
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e33"
E31_DIR = ROOT / "artifacts/analysis/e31"
SCORE_CACHE = E31_DIR / "roster_scenario_evaluation/predictions_and_scores.npz"
SCENARIO_CACHE = E31_DIR / "roster_scenario_evaluation/scenario_predictions.parquet"
E31_SELECTION = E31_DIR / "selection_results.json"
E31_MANIFEST = E31_DIR / "e31_manifest.json"
NEGATIVE_PREDICTIONS_DOC = ROOT / "docs/negative_preds.md"
THREADS = 8
SEED = 2033
VARIANTS = ("E33-A", "E33-B", "E33-C", "E33-D", "E33-E", "E33-F")
METHODS = ("Independent", "E19-F-A", "E19-F-C", "E19-D")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_positions(probabilities: np.ndarray) -> np.ndarray:
    """Return 1-based label ranks: checkpoint x row x label."""
    order = np.argsort(-probabilities, axis=2, kind="stable")
    ranks = np.empty_like(order, dtype=np.int8)
    cp_ix = np.arange(order.shape[0])[:, None, None]
    row_ix = np.arange(order.shape[1])[None, :, None]
    ranks[cp_ix, row_ix, order] = np.arange(1, order.shape[2] + 1, dtype=np.int8)[None, None, :]
    return ranks


def _folds_from_e31(ids: np.ndarray, prompts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    obj = json.loads(E31_SELECTION.read_text(encoding="utf-8"))
    lookup = {str(value): i for i, value in enumerate(ids)}
    folds = []
    seen: list[int] = []
    for record in obj["outer_folds"]:
        fit = np.asarray([lookup[str(value)] for value in record["fit_ids"]], dtype=np.int64)
        held = np.asarray([lookup[str(value)] for value in record["held_ids"]], dtype=np.int64)
        if set(prompts[fit]) & set(prompts[held]):
            raise ValueError(f"E31 fold {record['fold']} leaks prompt groups")
        folds.append((fit, held))
        seen.extend(held.tolist())
    if len(folds) != 5 or sorted(seen) != list(range(len(ids))):
        raise ValueError("E31 outer folds do not form an exact five-fold partition")
    return folds


def _fit_rank_stats(rank_position: np.ndarray, y: np.ndarray, train_rows: np.ndarray,
                    *, baseline_pred: np.ndarray | None = None) -> dict:
    n_cp = rank_position.shape[0]
    n_labels = len(LABELS)
    prior = np.bincount(y[train_rows], minlength=n_labels).astype(float)
    prior /= prior.sum()
    support = np.zeros((n_cp, 3, n_labels), dtype=np.int32)
    correct = np.zeros_like(support)
    for cp in range(n_cp):
        for rank in range(1, 4):
            rp = rank_position[cp, train_rows]
            for label in range(n_labels):
                mask = rp[:, label] == rank
                count = int(mask.sum())
                support[cp, rank - 1, label] = count
                if count:
                    correct[cp, rank - 1, label] = int(np.sum(y[train_rows][mask] == label))
    precision = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support > 0)
    # A fixed prior strength gives a deterministic Beta-binomial shrinkage to
    # the training-fold label prior for positive/negative reliability features.
    smooth = (correct + 30.0 * prior[None, None, :]) / (support + 30.0)
    stats = {"prior": prior, "support": support, "correct": correct,
             "precision": precision, "smoothed_precision": smooth,
             "ratio": np.divide(precision, prior[None, None, :],
                                out=np.zeros_like(precision), where=prior[None, None, :] > 0),
             "smoothed_ratio": np.divide(smooth, prior[None, None, :],
                                          out=np.ones_like(smooth), where=prior[None, None, :] > 0)}
    if baseline_pred is not None:
        cs = np.zeros_like(support)
        cc = np.zeros_like(correct)
        base = baseline_pred[train_rows]
        for cp in range(n_cp):
            for rank in range(1, 4):
                rp = rank_position[cp, train_rows]
                for label in range(n_labels):
                    mask = (base == label) & (rp[:, label] == rank)
                    cs[cp, rank - 1, label] = int(mask.sum())
                    if mask.any():
                        cc[cp, rank - 1, label] = int(np.sum(y[train_rows][mask] == label))
        stats["conditional_support"] = cs
        stats["conditional_correct"] = cc
        stats["conditional_precision"] = np.divide(cc, cs, out=np.zeros_like(cc, dtype=float), where=cs > 0)
    return stats


def _negative_rules(stats: dict, families: list[str]) -> dict:
    precision, support, prior = stats["precision"], stats["support"], stats["prior"]
    errors = 1.0 - precision
    family_index = np.asarray(families, dtype=object)
    single = np.zeros_like(support, dtype=bool)
    for c in range(len(families)):
        single[c] = (support[c] >= 30) & (errors[c] >= 0.97) & (precision[c] < prior[None, :])
    family_trigger_count = np.zeros((3, len(LABELS)), dtype=np.int16)
    for family in sorted(set(families)):
        family_trigger_count += np.any(single[family_index == family], axis=0).astype(np.int16)
    consensus = family_trigger_count >= 2
    conditional = np.zeros_like(support, dtype=bool)
    if "conditional_support" in stats:
        conditional = ((stats["conditional_support"] >= 20)
                       & ((1.0 - stats["conditional_precision"]) >= 0.95))
    return {"single": single, "consensus": consensus, "conditional": conditional,
            "family_trigger_count": family_trigger_count}


def _candidate_exclusions(rank_position: np.ndarray, rows: np.ndarray, stats: dict,
                          rules: dict, families: list[str], baseline_pred: np.ndarray,
                          variant: str) -> np.ndarray:
    out = np.zeros((len(rows), len(LABELS)), dtype=bool)
    family_names = sorted(set(families))
    if variant == "E33-C":
        for cp in range(len(families)):
            for rank in range(1, 4):
                for label in range(len(LABELS)):
                    if rules["conditional"][cp, rank - 1, label]:
                        hit = ((baseline_pred[rows] == label)
                               & (rank_position[cp, rows, label] == rank))
                        out[hit, label] = True
        return out
    family_hits = {family: np.zeros_like(out) for family in family_names}
    for cp, family in enumerate(families):
        for rank in range(1, 4):
            for label in range(len(LABELS)):
                if rules["single"][cp, rank - 1, label]:
                    hit = rank_position[cp, rows, label] == rank
                    out[hit, label] = True
                    family_hits[family][hit, label] = True
    if variant == "E33-B":
        family_count = np.zeros_like(out, dtype=np.int16)
        for hit in family_hits.values():
            family_count += hit.astype(np.int16)
        return family_count >= 2
    return out


def _weighted_factors(rank_position: np.ndarray, rows: np.ndarray, stats: dict,
                      families: list[str], *, combined: bool) -> np.ndarray:
    nrow = len(rows)
    factors = np.ones((nrow, len(LABELS)), dtype=np.float64)
    by_family: dict[str, np.ndarray] = {}
    for family in sorted(set(families)):
        cp_indices = [i for i, f in enumerate(families) if f == family]
        evidence_logs = np.zeros((nrow, len(LABELS)), dtype=np.float64)
        evidence_count = np.zeros((nrow, len(LABELS)), dtype=np.int16)
        strongest_negative = np.ones((nrow, len(LABELS)), dtype=np.float64)
        negative_present = np.zeros((nrow, len(LABELS)), dtype=bool)
        for cp in cp_indices:
            for rank in range(1, 4):
                active = (rank_position[cp, rows] == rank)
                if combined:
                    supported = stats["support"][cp, rank - 1] >= 30
                    ratio = stats["smoothed_ratio"][cp, rank - 1]
                else:
                    supported = ((stats["support"][cp, rank - 1] >= 30)
                                 & (stats["precision"][cp, rank - 1] < stats["prior"]))
                    ratio = stats["ratio"][cp, rank - 1]
                mask = active & supported[None, :]
                if not mask.any():
                    continue
                safe_ratio = np.clip(ratio, 1e-12, 1e12)
                if combined:
                    evidence_logs += np.where(mask, np.log(safe_ratio)[None, :], 0.0)
                    evidence_count += mask.astype(np.int16)
                else:
                    strongest_negative = np.where(mask,
                        np.minimum(strongest_negative, safe_ratio[None, :]), strongest_negative)
                    negative_present |= mask
        # D retains the strongest negative signal in each family. E averages
        # log-reliability within family so extra correlated checkpoints do not
        # multiply their influence solely through family size.
        if combined:
            family_log = np.divide(evidence_logs, evidence_count,
                                   out=np.zeros_like(evidence_logs), where=evidence_count > 0)
        else:
            family_log = np.where(negative_present, np.log(strongest_negative), 0.0)
        by_family[family] = family_log
    for family_log in by_family.values():
        factors *= np.exp(np.clip(family_log, -20, 20))
    return factors


def _build_candidate_features(rows: np.ndarray, stats: dict, *, probabilities: np.ndarray,
                              rank_position: np.ndarray, base_scores: np.ndarray,
                              families: list[str]) -> pd.DataFrame:
    n_cp, _, _ = probabilities.shape
    n = len(rows)
    score_order = np.argsort(-base_scores[rows], axis=1, kind="stable")
    score_rank = np.empty_like(score_order)
    score_rank[np.arange(n)[:, None], score_order] = np.arange(1, len(LABELS) + 1)[None, :]
    base_sorted = np.sort(base_scores[rows], axis=1)
    margin = base_sorted[:, -1] - base_sorted[:, -2]
    records = []
    family_pos = np.zeros((n, len(LABELS)), dtype=np.int16)
    family_neg = np.zeros_like(family_pos)
    cp_ratio = np.ones((n_cp, n, len(LABELS)), dtype=np.float32)
    cp_support = np.zeros((n_cp, n, len(LABELS)), dtype=np.float32)
    for cp in range(n_cp):
        for label in range(len(LABELS)):
            ranks = rank_position[cp, rows, label]
            valid = ranks <= 3
            rr = ranks[valid].astype(int) - 1
            idx = np.flatnonzero(valid)
            if len(idx):
                cp_support[cp, idx, label] = stats["support"][cp, rr, label]
                supported = stats["support"][cp, rr, label] >= 30
                cp_ratio[cp, idx[supported], label] = stats["smoothed_ratio"][cp, rr[supported], label]
    for family in sorted(set(families)):
        cp_ix = [i for i, f in enumerate(families) if f == family]
        eligible = cp_support[cp_ix] >= 30
        active_pos = np.any((cp_ratio[cp_ix] > 1.0) & eligible, axis=0)
        active_neg = np.any((cp_ratio[cp_ix] < 1.0) & eligible, axis=0)
        family_pos += active_pos.astype(np.int16)
        family_neg += active_neg.astype(np.int16)
    # Candidate-major blocks preserve a deterministic row/label alignment.
    labels = np.arange(len(LABELS), dtype=np.int64)
    for row_offset, row in enumerate(rows):
        for label in labels:
            rec = {"candidate_label": LABELS[label],
                   "e31_score": float(base_scores[row, label]),
                   "e31_rank": int(score_rank[row_offset, label]),
                   "e31_top_margin": float(margin[row_offset]),
                   "positive_family_count": int(family_pos[row_offset, label]),
                   "negative_family_count": int(family_neg[row_offset, label])}
            for cp, checkpoint_id in enumerate(_BUILD_CP_IDS):
                rec[f"prob::{checkpoint_id}"] = float(probabilities[cp, row, label])
                rec[f"rank::{checkpoint_id}"] = int(rank_position[cp, row, label])
                rec[f"reliability::{checkpoint_id}"] = float(cp_ratio[cp, row_offset, label])
                rec[f"support::{checkpoint_id}"] = float(cp_support[cp, row_offset, label])
            records.append(rec)
    frame = pd.DataFrame.from_records(records)
    frame["candidate_label"] = frame["candidate_label"].astype(str)
    if not np.isfinite(frame.drop(columns=["candidate_label"]).to_numpy(dtype=float)).all():
        raise ValueError("E33 candidate features contain non-finite values")
    return frame


_BUILD_CP_IDS: list[str] = []


def _fit_e33_f(fit_rows: np.ndarray, held_rows: np.ndarray, *, ids: np.ndarray,
               prompts: np.ndarray, y: np.ndarray, probabilities: np.ndarray,
               rank_position: np.ndarray, base_scores: np.ndarray,
               families: list[str], baseline_pred: np.ndarray, fold_number: int,
               output_dir: Path) -> tuple[np.ndarray, dict]:
    # Cross-fit the reliability features used to construct the meta-classifier's
    # training matrix. Each training response's reliability estimates exclude
    # its prompt group.
    group_kfold = GroupKFold(n_splits=4)
    x_blocks, target_blocks = [], []
    for inner_fit_local, inner_held_local in group_kfold.split(
            fit_rows, y[fit_rows], prompts[fit_rows]):
        inner_fit = fit_rows[inner_fit_local]
        inner_held = fit_rows[inner_held_local]
        if set(prompts[inner_fit]) & set(prompts[inner_held]):
            raise ValueError("E33-F inner folds overlap prompt groups")
        stats = _fit_rank_stats(rank_position, y, inner_fit)
        features = _build_candidate_features(inner_held, stats,
            probabilities=probabilities, rank_position=rank_position,
            base_scores=base_scores, families=families)
        target = np.tile((y[inner_held, None] != np.arange(len(LABELS))[None, :]).astype(np.int8).ravel(), 1)
        x_blocks.append(features)
        target_blocks.append(target)
    train_x = pd.concat(x_blocks, ignore_index=True)
    train_y = np.concatenate(target_blocks)
    if len(train_x) != len(train_y) or train_y.min() != 0 or train_y.max() != 1:
        raise ValueError("E33-F cross-fit training feature/target alignment failed")

    outer_stats = _fit_rank_stats(rank_position, y, fit_rows)
    valid_x = _build_candidate_features(held_rows, outer_stats,
        probabilities=probabilities, rank_position=rank_position,
        base_scores=base_scores, families=families)
    model = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=5.0,
        auto_class_weights="Balanced", random_seed=SEED + fold_number,
        thread_count=THREADS, verbose=False, allow_writing_files=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_x, train_y, cat_features=["candidate_label"])
    p_wrong = model.predict_proba(valid_x)[:, list(model.classes_).index(1)]
    p_correct = (1.0 - p_wrong).reshape(len(held_rows), len(LABELS))
    adjusted = base_scores[held_rows] * p_correct
    model.save_model(str(output_dir / f"e33_f_fold{fold_number}.cbm"))
    metadata = {"fold": fold_number, "training_rows": int(len(fit_rows)),
        "oof_training_candidate_rows": int(len(train_x)), "held_rows": int(len(held_rows)),
        "iterations": 400, "depth": 6, "learning_rate": 0.05,
        "threads": THREADS, "positive_target": "candidate label is incorrect",
        "model_path": f"artifacts/analysis/e33/e33_f_fold{fold_number}.cbm"}
    return adjusted, metadata


def _augment_hungarian(scores: np.ndarray, excluded: np.ndarray,
                       row_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Restore excluded candidates by original-score order until a matching exists."""
    allowed = ~excluded.copy()
    restored = 0

    def maximum_matching() -> tuple[int, list[int]]:
        label_to_row = [-1] * len(LABELS)

        def visit(row: int, visited: set[int]) -> bool:
            candidates = np.flatnonzero(allowed[row])
            candidates = sorted(candidates.tolist(), key=lambda c: (-scores[row, c], c))
            for label in candidates:
                if label in visited:
                    continue
                visited.add(label)
                prior_row = label_to_row[label]
                if prior_row < 0 or visit(prior_row, visited):
                    label_to_row[label] = row
                    return True
            return False

        size = 0
        for row in range(len(allowed)):
            if visit(row, set()):
                size += 1
        row_to_label = [-1] * len(allowed)
        for label, row in enumerate(label_to_row):
            if row >= 0:
                row_to_label[row] = label
        return size, row_to_label

    while True:
        size, row_to_label = maximum_matching()
        if size == len(allowed):
            break
        unmatched = [r for r, c in enumerate(row_to_label) if c < 0]
        if not unmatched:
            raise RuntimeError("E33 matching feasibility computation did not identify an unmatched row")
        row = unmatched[0]
        candidates = [c for c in range(len(LABELS)) if not allowed[row, c]]
        if not candidates:
            raise RuntimeError("E33 cannot restore a feasible candidate assignment")
        label = min(candidates, key=lambda c: (-scores[row, c], c))
        allowed[row, label] = True
        restored += 1

    adjusted = np.where(allowed, scores, -1e12)
    pred = hungarian(adjusted, np.zeros(len(adjusted), dtype=str), row_ids)
    # The regular shared decoder works by group; this helper is called per group,
    # so a zero-valued group vector is the correct one-group assignment.
    return pred, restored


def _evaluate_scenario_scores(scenario_rows: pd.DataFrame, row_lookup: dict[str, int],
                              scores: np.ndarray, exclusions: np.ndarray | None,
                              baseline_pred: np.ndarray) -> tuple[dict, list[dict]]:
    rows = np.asarray([row_lookup[str(x)] for x in scenario_rows.original_id], dtype=np.int64)
    truth = pd.Categorical(scenario_rows.truth.astype(str), categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any() or not np.array_equal(truth, _SCENARIO_TRUTH[rows]):
        raise ValueError("E33 scenario truth is misaligned with the fixed V0 rows")
    scenario_scores = scores[rows]
    scenario_exclusions = (np.zeros_like(scenario_scores, dtype=bool) if exclusions is None
                           else exclusions[rows])
    scenario_scores = scenario_scores.copy()
    all_removed = scenario_exclusions.all(axis=1)
    scenario_exclusions[all_removed] = False
    scenario_scores[scenario_exclusions] = -1e12
    results: dict[str, dict] = {}
    predictions: list[dict] = []
    independent = np.argmax(scenario_scores, axis=1)
    results["Independent"] = _metric(truth, independent)
    predictions.extend({"scenario_id": str(scenario_rows.scenario_id.iloc[i]),
        "scenario_row_id": str(scenario_rows.scenario_row_id.iloc[i]),
        "original_id": str(scenario_rows.original_id.iloc[i]), "truth": LABELS[truth[i]],
        "prediction": LABELS[independent[i]], "method": "Independent"} for i in range(len(rows)))
    for method in METHODS[1:]:
        group_col = f"group::{method}"
        groups = scenario_rows[group_col].astype(str).to_numpy()
        pred = np.full(len(rows), -1, dtype=np.int64)
        restored_total = 0
        for group in sorted(set(groups)):
            local = np.flatnonzero(groups == group)
            if exclusions is None:
                pred[local] = hungarian(scenario_scores[local], groups[local],
                                         scenario_rows.scenario_row_id.astype(str).to_numpy()[local])
            else:
                group_pred, restored = _augment_hungarian(
                    scores[rows[local]], exclusions[rows[local]],
                    scenario_rows.scenario_row_id.astype(str).to_numpy()[local])
                pred[local] = group_pred
                restored_total += restored
        if (pred < 0).any():
            raise RuntimeError("E33 did not predict every scenario row")
        results[method] = {**_metric(truth, pred), "restored_candidates_for_feasibility": restored_total}
        predictions.extend({"scenario_id": str(scenario_rows.scenario_id.iloc[i]),
            "scenario_row_id": str(scenario_rows.scenario_row_id.iloc[i]),
            "original_id": str(scenario_rows.original_id.iloc[i]), "truth": LABELS[truth[i]],
            "prediction": LABELS[pred[i]], "method": method} for i in range(len(rows)))
    return results, predictions


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction))}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = catalog.cascades._v0(ROOT)
    _, members, all_probs, excluded, trust_snapshot = catalog._load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    if excluded:
        raise ValueError(f"E31 baseline trust roster has unavailable caches: {excluded}")
    manifest = json.loads(E31_MANIFEST.read_text(encoding="utf-8"))
    roster = list(map(str, manifest["roster"]))
    if len(roster) != 40 or len(set(roster)) != 40 or set(roster) - set(all_probs):
        raise ValueError("E31 manifest does not resolve to 40 cached checkpoints")
    ids = v0.id.astype(str).to_numpy()
    prompts = v0.prompt_id.astype(str).to_numpy()
    y = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if len(v0) != 2699 or (y < 0).any() or len(set(prompts)) != 150:
        raise ValueError("Expected full V0 with 2,699 rows and 150 prompt groups")
    probabilities = np.stack([all_probs[sid] for sid in roster]).astype(np.float32)
    global _BUILD_CP_IDS
    _BUILD_CP_IDS = roster
    families = [_model_family(members[sid]["model"]) for sid in roster]
    if len(families) != 40:
        raise ValueError("E31 family metadata is incomplete")
    rank_position = _rank_positions(probabilities)
    score_cache = np.load(SCORE_CACHE, allow_pickle=True)
    if (score_cache["row_ids"].astype(str).tolist() != ids.tolist()
            or score_cache["classes"].astype(str).tolist() != list(LABELS)):
        raise ValueError("E31 score cache row/class order differs from full V0")
    base_scores = score_cache["score_40"].astype(np.float64)
    base_pred = score_cache["independent_40"].astype(np.int64)
    if base_scores.shape != (2699, 18) or not np.array_equal(base_scores.argmax(1), base_pred):
        raise ValueError("E31 independent predictions do not match its cached score matrix")
    folds = _folds_from_e31(ids, prompts)
    scenario = pd.read_parquet(ROOT / "data/processed/dataset_v0_group.parquet")
    scenario_preds = pd.read_parquet(SCENARIO_CACHE)
    scenario_preds = scenario_preds.loc[scenario_preds.checkpoint_count.eq(40)].copy()
    group_map = {}
    for method in METHODS[1:]:
        subset = scenario_preds.loc[scenario_preds.configuration.eq(method)]
        if len(subset) != 25000 or subset.scenario_row_id.duplicated().any():
            raise ValueError(f"Cached E31 {method} groups are incomplete or duplicated")
        group_map[method] = subset.set_index("scenario_row_id").predicted_group.astype(str).to_dict()
    scenario_frames = []
    id_set = set(ids)
    for sid, frame in scenario.groupby("scenario_id", sort=True):
        frame = frame.reset_index(drop=True).copy()
        if len(frame) != 500 or frame.original_id.astype(str).duplicated().any():
            raise ValueError(f"Invalid v0_group scenario {sid}")
        if set(frame.original_id.astype(str)) - id_set:
            raise ValueError(f"Scenario {sid} has IDs outside V0")
        frame["truth"] = frame.model.astype(str)
        for method in METHODS[1:]:
            try:
                frame[f"group::{method}"] = [group_map[method][str(rid)] for rid in frame.scenario_row_id]
            except KeyError as exc:
                raise ValueError(f"E31 cached group IDs lack scenario row {exc}") from exc
        scenario_frames.append((str(sid), frame))
    if len(scenario_frames) != 50:
        raise ValueError(f"Expected exactly 50 scenarios, got {len(scenario_frames)}")
    row_lookup = {rid: i for i, rid in enumerate(ids)}
    global _SCENARIO_TRUTH
    _SCENARIO_TRUTH = y
    out_scores = {key: np.zeros_like(base_scores) for key in VARIANTS}
    out_exclusions = {key: np.zeros_like(base_scores, dtype=bool) for key in VARIANTS[:3]}
    rule_audit = []
    learned_folds = []
    fold_rule_counts = []
    for fold_number, (fit_rows, held_rows) in enumerate(folds, 1):
        stats = _fit_rank_stats(rank_position, y, fit_rows, baseline_pred=base_pred)
        rules = _negative_rules(stats, families)
        fold_rule_counts.append({"fold": fold_number,
            "A_single_rules": int(rules["single"].sum()),
            "B_consensus_label_rank_rules": int(rules["consensus"].sum()),
            "C_conditional_rules": int(rules["conditional"].sum())})
        for variant in VARIANTS[:3]:
            out_exclusions[variant][held_rows] = _candidate_exclusions(
                rank_position, held_rows, stats, rules, families, base_pred, variant)
            out_scores[variant][held_rows] = base_scores[held_rows]
        factors_d = _weighted_factors(rank_position, held_rows, stats, families, combined=False)
        factors_e = _weighted_factors(rank_position, held_rows, stats, families, combined=True)
        out_scores["E33-D"][held_rows] = base_scores[held_rows] * factors_d
        out_scores["E33-E"][held_rows] = base_scores[held_rows] * factors_e
        for label in range(len(LABELS)):
            for cp in range(len(roster)):
                for rank in range(1, 4):
                    sup = int(stats["support"][cp, rank - 1, label])
                    if sup:
                        rule_audit.append({"fold": fold_number, "checkpoint_id": roster[cp],
                            "family": families[cp], "rank": rank, "label": LABELS[label],
                            "support": sup, "correct": int(stats["correct"][cp, rank - 1, label]),
                            "precision": float(stats["precision"][cp, rank - 1, label]),
                            "label_prior": float(stats["prior"][label]),
                            "negative_rule_A_B": bool(rules["single"][cp, rank - 1, label]),
                            "conditional_support": int(stats["conditional_support"][cp, rank - 1, label]),
                            "conditional_precision": float(stats["conditional_precision"][cp, rank - 1, label]),
                            "conditional_rule_C": bool(rules["conditional"][cp, rank - 1, label])})
        f_scores, f_meta = _fit_e33_f(fit_rows, held_rows, ids=ids, prompts=prompts,
            y=y, probabilities=probabilities, rank_position=rank_position,
            base_scores=base_scores, families=families, baseline_pred=base_pred,
            fold_number=fold_number, output_dir=OUT)
        out_scores["E33-F"][held_rows] = f_scores
        learned_folds.append(f_meta)
        print(f"E33 fold {fold_number}/5 complete; A rules={fold_rule_counts[-1]['A_single_rules']}, "
              f"B consensus={fold_rule_counts[-1]['B_consensus_label_rank_rules']}, "
              f"C rules={fold_rule_counts[-1]['C_conditional_rules']}", flush=True)

    # Retain baseline matrices, and ensure every OOF variant score is populated.
    if not all(np.isfinite(s).all() for s in out_scores.values()):
        raise ValueError("E33 produced incomplete/non-finite OOF scores")
    predictions = []
    scenario_metrics: dict[str, dict[str, list[float]]] = {"E31 baseline": {m: [] for m in METHODS}}
    base_scenario = scenario_preds.copy()
    scenario_lookup_frames = {sid: frame for sid, frame in scenario_frames}
    for _, row in base_scenario.iterrows():
        pass
    for scenario_id, frame in scenario_frames:
        rows = np.asarray([row_lookup[str(rid)] for rid in frame.original_id], dtype=np.int64)
        true = pd.Categorical(frame.truth.astype(str), categories=LABELS).codes.astype(np.int64)
        baseline_frame = scenario_preds.loc[scenario_preds.scenario_id.eq(scenario_id)]
        base_metrics = {}
        for method in METHODS:
            part = baseline_frame.loc[baseline_frame.configuration.eq(method)]
            if len(part) != 500:
                raise ValueError(f"E31 baseline predictions incomplete for {scenario_id}/{method}")
            part = part.set_index("scenario_row_id").loc[
                frame.scenario_row_id.astype(str)].reset_index()
            if not np.array_equal(part.truth.astype(str).to_numpy(), frame.truth.astype(str).to_numpy()):
                raise ValueError(f"E31 baseline row truth mismatch in {scenario_id}/{method}")
            if method == "Independent":
                pred = pd.Categorical(part.prediction.astype(str), categories=LABELS).codes
            else:
                pred = pd.Categorical(part.prediction.astype(str), categories=LABELS).codes
            if (pred < 0).any():
                raise ValueError("E31 baseline contains unknown model labels")
            value = _metric(true, pred)
            base_metrics[method] = value
            scenario_metrics["E31 baseline"][method].append(value["balanced_accuracy"])
            predictions.extend({"scenario_id": scenario_id, "scenario_row_id": str(frame.scenario_row_id.iloc[i]),
                "original_id": str(frame.original_id.iloc[i]), "truth": LABELS[true[i]],
                "prediction": LABELS[pred[i]], "experiment": "E31 baseline", "method": method}
                for i in range(len(frame)))
        for variant in VARIANTS:
            scores = out_scores[variant]
            exclusions = out_exclusions.get(variant)
            # Hard exclusions are validated/fallback-resolved in the decoder.
            variant_result, variant_predictions = _evaluate_scenario_scores(
                frame, row_lookup, scores, exclusions, base_pred)
            scenario_metrics.setdefault(variant, {m: [] for m in METHODS})
            for method in METHODS:
                val = variant_result[method]
                scenario_metrics[variant][method].append(val["balanced_accuracy"])
            for record in variant_predictions:
                record["experiment"] = variant
                predictions.append(record)

    summary = {}
    for experiment, by_method in scenario_metrics.items():
        summary[experiment] = {}
        for method, vals in by_method.items():
            a = np.asarray(vals, dtype=np.float64)
            summary[experiment][method] = {"mean_balanced_accuracy": float(a.mean()),
                "scenario_count": int(len(a)), "scenario_values": a.tolist(),
                "std_balanced_accuracy": float(a.std()),
                "min_balanced_accuracy": float(a.min()),
                "p10_balanced_accuracy": float(np.quantile(a, 0.10))}
    e31_saved = json.loads((E31_DIR / "roster_scenario_evaluation/results.json").read_text())
    for method in METHODS:
        saved = float(e31_saved["summary"]["40"][method]["mean_balanced_accuracy"])
        measured = summary["E31 baseline"][method]["mean_balanced_accuracy"]
        if not np.isclose(saved, measured, rtol=0.0, atol=1e-12):
            raise AssertionError(f"E31 {method} baseline reproduction failed: {measured} != {saved}")

    # Build rule audit for human-readable reproducibility and deployable artifacts.
    pd.DataFrame(rule_audit).to_parquet(OUT / "fold_rule_statistics.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(OUT / "scenario_predictions.parquet", index=False)
    np.savez_compressed(OUT / "oof_adjusted_scores.npz", baseline=base_scores,
        **{key.replace("-", "_"): value for key, value in out_scores.items()},
        **{f"exclusion_{key.replace('-', '_')}": value for key, value in out_exclusions.items()},
        row_ids=ids, classes=np.asarray(LABELS))
    pd.DataFrame(fold_rule_counts).to_csv(OUT / "fold_rule_counts.csv", index=False)
    result = {"experiment": "E33 Negative Evidence and Candidate Label Elimination",
        "status": "Scored", "baseline": "E31 fixed 40-checkpoint ensemble",
        "roster": roster, "families": families, "scenario_count": 50,
        "rows_per_scenario": 500, "methods": list(METHODS),
        "summary": summary, "fold_rule_counts": fold_rule_counts,
        "learned_classifier_folds": learned_folds,
        "selection_folds": [{"fit_rows": len(a), "held_rows": len(b)} for a,b in folds],
        "rule_thresholds": {"A_B": {"support": 30, "wrong_rate": 0.97,
            "precision_below_class_prior": True}, "C": {"conditional_support": 20,
            "conditional_wrong_rate": 0.95}, "D": {"support": 30,
            "precision_below_class_prior": True}, "E": {"support": 30,
            "prior_smoothing_strength": 30}, "F": {"inner_group_folds": 4,
            "iterations": 400, "depth": 6, "learning_rate": 0.05}},
        "inputs": {"trust_list_version": trust_snapshot["version"],
            "trust_list_sha256": trust_snapshot["sha256"],
            "negative_predictions_doc_sha256": _sha(NEGATIVE_PREDICTIONS_DOC),
            "e31_manifest_sha256": _sha(E31_MANIFEST),
            "e31_score_cache_sha256": _sha(SCORE_CACHE),
            "e31_scenario_cache_sha256": _sha(SCENARIO_CACHE),
            "v0_group_sha256": _sha(ROOT / "data/processed/dataset_v0_group.parquet")},
        "notes": ["docs/negative_preds.md provides the full-V0 rank/label analysis and rule definitions; E33 recomputes each statistic inside the E31 prompt folds rather than reusing full-V0 rates.",
            "All rules and learned model training are prompt-group cross-fit using the saved E31 five outer folds.",
            "E33-F training features additionally use four-fold prompt-group OOF reliability features within each outer training split.",
            "E31 scores and its fixed deployment router are preserved exactly; rule/statistic fitting does not refit the base E31 router.",
            "The 50 v0_group scenarios reuse V0 rows and are correlated, in-sample robustness diagnostics rather than independent validation.",
            "Identical cached E19-F-A, E19-F-C and E19-D group assignments are used for every experiment."]}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(result)
    return result


def _write_report(result: dict) -> None:
    labels = ["E31 baseline", *VARIANTS]
    table = []
    for experiment in labels:
        cells = [result["summary"][experiment][method]["mean_balanced_accuracy"] for method in METHODS]
        table.append(cells)
    maxima = np.max(np.asarray(table), axis=0)
    lines = [
        "# E33 — Negative Evidence and Candidate Label Elimination", "",
        "| Experiment | Independent | E19-F-A + Hungarian | E19-F-C + Hungarian | E19-D + Hungarian |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment, values in zip(labels, table):
        cells = [f"**{v*100:.4f}%**" if np.isclose(v, maxima[i], atol=1e-12, rtol=0)
                 else f"{v*100:.4f}%" for i, v in enumerate(values)]
        lines.append(f"| {experiment} | " + " | ".join(cells) + " |")
    lines += ["", "## Evaluation notes", "",
        "All seven rows use the same fixed E31 40-checkpoint roster, V0 rows, 50 scenario memberships and cached E19-F-A/E19-F-C/E19-D reconstructed groups. The baseline is reproduced from E31's cached 40-checkpoint scores and scenario predictions before reporting variants. `docs/negative_preds.md` was used as the full-V0 rank/label analysis reference; E33 recomputes all rule rates within each training fold instead of reusing its full-V0 rates.",
        "", "The 50 scenarios reuse V0 responses; treat the values as a correlated robustness comparison, not independent validation. Rules use prompt-group cross-fitting. E33-F trains its candidate-error classifier on inner prompt-group OOF reliability features, then scores each outer held fold. E31's frozen base score matrix is unchanged.",
        "", "Hard candidate exclusions retain the original E31 scores if every candidate is removed. For Hungarian assignment, excluded labels are restored in descending original E31 score order only until a complete unique-label matching becomes feasible. D and E apply multiplicative reliability factors before independent decoding or assignment. E33-F multiplies E31 scores by the predicted candidate-correctness probability.",
        "", "E33-A/B rule thresholds: support ≥30, wrong rate ≥97%, and precision below the training-fold class prior. E33-C conditions support/wrong rate on E31 selecting that candidate (≥20 and ≥95%). E33-D uses empirical precision/prior ratios for supported negative rules. E33-E uses a prior-shrunk precision estimate (prior strength 30). E33-F uses CatBoost with 400 iterations, depth 6, learning rate 0.05, balanced class weights, and 8 CPU threads.",
        "", "Artifacts: `artifacts/analysis/e33/metrics.json`, `fold_rule_statistics.parquet`, `fold_rule_counts.csv`, `oof_adjusted_scores.npz`, `scenario_predictions.parquet`, and the five `e33_f_fold*.cbm` models.", ""]
    (ROOT / "docs/e33.md").write_text("\n".join(lines), encoding="utf-8")


def _update_catalogue(status: str, result: dict | None = None, error: str | None = None) -> None:
    path = ROOT / "configs/ensemble_catalog_v2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [item for item in data.get("experiments", []) if item.get("id") == "E33"]
    if len(entries) != 1:
        raise ValueError(f"expected exactly one E33 catalogue entry, found {len(entries)}")
    entry = entries[0]
    entry.update({"status": status, "current_stage": status,
                  "progress_source": "artifacts/analysis/e33/metrics.json"})
    if error:
        entry["error"] = error
    if result is not None:
        entry["status"] = "Scored"
        entry["current_stage"] = "Complete"
        entry["result_source"] = "artifacts/analysis/e33/metrics.json"
        entry["scenario_results"] = {
            experiment: {method: values["mean_balanced_accuracy"]
                         for method, values in methods.items()}
            for experiment, methods in result["summary"].items()}
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    _update_catalogue("Running")
    try:
        result = run()
        _update_catalogue("Scored", result)
        print(json.dumps({"summary": {exp: {m: data["mean_balanced_accuracy"]
            for m, data in methods.items()} for exp, methods in result["summary"].items()},
            "output": str(OUT), "report": "docs/e33.md"}, indent=2))
        return 0
    except Exception as exc:
        _update_catalogue("Failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
