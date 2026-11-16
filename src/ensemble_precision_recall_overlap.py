"""Trust List 2 precision/recall overlap analysis and cross-fitted ensembles.

This is an offline analysis command. It consumes saved full-V0 probability
artifacts and does not load checkpoints, train models, or change production
ensemble state.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS, _read_saved_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("artifacts/analysis/ensemble_precision_recall_overlap")
REPORT = Path("docs/ensemble_precision_recall_overlap.md")
EQUAL9 = ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
N_SPLITS = 5
MAX_PER_LABEL = 3
F1_NEAR_BEST = 0.05


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _truth(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _false(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"false", "0", "no", "n", ""})


def _metrics(tp: int, fp: int, positives: int) -> dict:
    fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "prediction_count": int(tp + fp)}


def _candidate_stats(ids: list[str], matrices: dict[str, np.ndarray], y: np.ndarray,
                     class_i: int, indexes: np.ndarray) -> dict[str, dict]:
    truth = y[indexes] == class_i
    stats = {}
    for sid in ids:
        pred = matrices[sid][indexes].argmax(axis=1) == class_i
        tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth))
        stats[sid] = _metrics(tp, fp, int(truth.sum()))
    return stats


def _retained_for_label(ids: list[str], matrices: dict[str, np.ndarray], y: np.ndarray,
                        class_i: int, indexes: np.ndarray) -> tuple[list[str], dict]:
    stats = _candidate_stats(ids, matrices, y, class_i, indexes)
    # Deterministic top checkpoints for each requested criterion; all exact ties
    # are kept. Then keep every checkpoint within five F1 percentage points.
    chosen: set[str] = set()
    for metric in ("f1", "precision", "recall"):
        best = max(stats[sid][metric] for sid in ids)
        chosen.update(sid for sid in ids if np.isclose(stats[sid][metric], best, atol=1e-12))
    best_f1 = max(stats[sid]["f1"] for sid in ids)
    chosen.update(sid for sid in ids if stats[sid]["f1"] >= best_f1 - F1_NEAR_BEST - 1e-12)
    return sorted(chosen, key=lambda x: int(x[1:])), stats


def _combo_stats(combo: tuple[str, ...], matrices: dict[str, np.ndarray], y: np.ndarray,
                 class_i: int, indexes: np.ndarray) -> dict:
    truth = y[indexes] == class_i
    masks = np.stack([matrices[sid][indexes].argmax(axis=1) == class_i for sid in combo])
    union = masks.any(axis=0)
    counts = masks.sum(axis=0)
    tp = int(np.sum(union & truth)); fp = int(np.sum(union & ~truth))
    m = _metrics(tp, fp, int(truth.sum()))
    m.update({"checkpoint_ids": list(combo), "shared_tp": int(np.sum((counts >= 2) & truth)),
              "unique_tp": int(np.sum((counts == 1) & truth)),
              "shared_fp": int(np.sum((counts >= 2) & ~truth)),
              "unique_fp": int(np.sum((counts == 1) & ~truth))})
    # Deterministic contribution order: strongest standalone F1 first, then
    # repeatedly the largest TP gain with FP cost as tie-breaker.
    remaining = set(combo); order: list[str] = []; assigned = np.zeros(len(indexes), dtype=bool)
    contributions = []
    while remaining:
        options = []
        for sid in remaining:
            mask = masks[combo.index(sid)] & ~assigned
            add_tp = int(np.sum(mask & truth)); add_fp = int(np.sum(mask & ~truth))
            sm = _candidate_stats([sid], matrices, y, class_i, indexes)[sid]
            options.append((sid, mask, add_tp, add_fp, sm))
        sid, mask, add_tp, add_fp, _ = min(options, key=lambda z: (-z[2], z[3], int(z[0][1:])))
        before = int(assigned.sum()); assigned |= mask
        order.append(sid); remaining.remove(sid)
        contributions.append({"checkpoint_id": sid, "incremental_tp": add_tp,
            "incremental_fp": add_fp, "incremental_precision": add_tp/(add_tp+add_fp) if add_tp+add_fp else 0.0,
            "newly_covered_rows": int(assigned.sum())-before})
    m["contribution_order"] = order; m["incremental_contributions"] = contributions
    return m


def _best_label_combos(retained: list[str], matrices: dict[str, np.ndarray], y: np.ndarray,
                       class_i: int, indexes: np.ndarray) -> list[dict]:
    rows = []
    for size in range(1, min(MAX_PER_LABEL, len(retained)) + 1):
        for combo in itertools.combinations(retained, size):
            rows.append(_combo_stats(combo, matrices, y, class_i, indexes))
    rows.sort(key=lambda r: (-r["f1"], -r["precision"], -r["recall"],
                             len(r["checkpoint_ids"]), tuple(int(x[1:]) for x in r["checkpoint_ids"])))
    return rows


def _select(indexes: np.ndarray, ids: list[str], matrices: dict[str, np.ndarray],
            y: np.ndarray) -> dict:
    retained_by_label: dict[str, list[str]] = {}
    stats_by_label: dict[str, dict] = {}
    combos_by_label: dict[str, list[dict]] = {}
    for ci, label in enumerate(LABELS):
        retained, stats = _retained_for_label(ids, matrices, y, ci, indexes)
        # Keep EQUAL9 members available as candidates for every target label.
        retained = sorted(set(retained) | (set(EQUAL9) & set(ids)), key=lambda x: int(x[1:]))
        retained_by_label[label] = retained; stats_by_label[label] = stats
        combos_by_label[label] = _best_label_combos(retained, matrices, y, ci, indexes)
    selected_label_combo = {label: combos_by_label[label][0] for label in LABELS}
    selected_global = sorted(set(EQUAL9) | {
        sid for item in selected_label_combo.values() for sid in item["checkpoint_ids"]
    }, key=lambda x: int(x[1:]))
    return {"retained_by_label": retained_by_label, "stats_by_label": stats_by_label,
            "combos_by_label": combos_by_label, "selected_label_combo": selected_label_combo,
            "selected_global": selected_global}


def _weights_for_label(label: str, selected: dict, method: str, y: np.ndarray,
                       indexes: np.ndarray, matrices: dict[str, np.ndarray]) -> dict[str, float]:
    combo = selected["selected_label_combo"][label]
    members = combo["checkpoint_ids"]
    if not members: return {}
    if method == "f1":
        raw = [selected["stats_by_label"][label][sid]["f1"] for sid in members]
    elif method == "complementarity":
        stats = combo["incremental_contributions"]
        raw = []
        by_id = {row["checkpoint_id"]: row for row in stats}
        for sid in members:
            base = selected["stats_by_label"][label][sid]
            inc = by_id[sid]
            # Reliability rewards correct marginal coverage and penalizes new
            # false positives; each term is learned solely on fitting rows.
            marginal_precision = (inc["incremental_tp"] + 1.0) / (
                inc["incremental_tp"] + inc["incremental_fp"] + 2.0)
            marginal_coverage = inc["incremental_tp"] / max(1, int(np.sum(y[indexes] == LABELS.index(label))))
            raw.append(base["precision"] * base["recall"] * marginal_precision * marginal_coverage)
    else:
        raise ValueError(method)
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or raw.sum() <= 0: raw = np.ones(len(members), dtype=float)
    raw /= raw.sum()
    return dict(zip(members, raw.tolist()))


def _make_probabilities(strategy: str, selected: dict, ids: list[str], matrices: dict[str, np.ndarray],
                        y: np.ndarray, indexes: np.ndarray,
                        fit_indexes: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    n = len(indexes); c = len(LABELS)
    fit_indexes = indexes if fit_indexes is None else fit_indexes
    if strategy == "equal9":
        out = np.mean([matrices[sid][indexes] for sid in EQUAL9], axis=0)
        return out / out.sum(axis=1, keepdims=True), {"checkpoint_ids": EQUAL9, "weights": {x: 1/len(EQUAL9) for x in EQUAL9}}
    if strategy == "equal_selected":
        members = selected["selected_global"]
        out = np.mean([matrices[sid][indexes] for sid in members], axis=0)
        return out / out.sum(axis=1, keepdims=True), {"checkpoint_ids": members, "weights": {x: 1/len(members) for x in members}}
    method = "f1" if strategy == "per_label_f1" else "complementarity"
    out = np.zeros((n, c), dtype=np.float64); weight_data = {}
    for ci, label in enumerate(LABELS):
        weights = _weights_for_label(label, selected, method, y, fit_indexes, matrices)
        weight_data[label] = weights
        for sid, weight in weights.items(): out[:, ci] += weight * matrices[sid][indexes, ci]
    sums = out.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= 0
    if bad.any():
        fallback = np.mean([matrices[sid][indexes[bad]] for sid in EQUAL9], axis=0)
        out[bad] = fallback
        sums = out.sum(axis=1, keepdims=True)
    return out / np.maximum(sums, 1e-15), {"per_label_weights": weight_data}


def _evaluate(y: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    tp = np.diag(cm); fp = cm.sum(axis=0) - tp; support = cm.sum(axis=1)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {"rows": int(len(y)), "correct": int(np.sum(pred == y)), "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(recall.mean()),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1)).mean()),
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(support[i]-tp[i])}
            for i, label in enumerate(LABELS)},
        "confusion_matrix": cm.tolist(), "y_pred": pred}


def _comparison(base_pred: np.ndarray, pred: np.ndarray, y: np.ndarray) -> dict:
    b = base_pred == y; p = pred == y
    return {"corrected": int((~b & p).sum()), "introduced": int((b & ~p).sum()),
            "disagreements": int((base_pred != pred).sum()),
            "disagreement_rate": float(np.mean(base_pred != pred))}


def _load_inputs(root: Path, trust_file: str | None, output: Path, resume: bool):
    pointer = json.loads((root / "docs/trust-list-2-latest.json").read_text())
    trust_path = Path(trust_file) if trust_file else root / pointer["csv_path"]
    if not trust_path.is_absolute(): trust_path = root / trust_path
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    eligible = trust.loc[_truth(trust.evaluation_eligible) & _truth(trust.rule_eligible) &
        _truth(trust.checkpoint_available) & _false(trust.validation_training_overlap) &
        (pd.to_numeric(trust.validation_rows, errors="coerce") == 2699)].copy()
    identity = eligible.sort_values("sequence_id", kind="stable").drop_duplicates("sequence_id").set_index("sequence_id")
    ids = sorted(set(identity.index.astype(str)) | set(EQUAL9), key=lambda x: int(x[1:]))
    missing = set(ids) - set(identity.index.astype(str))
    if missing:
        extra = trust.loc[trust.sequence_id.astype(str).isin(missing)].drop_duplicates("sequence_id").set_index("sequence_id")
        identity = pd.concat([identity, extra])
    v0 = _dataset_rows("v0", root / "data/processed")
    if len(v0) != 2699: raise ValueError(f"Expected 2699 V0 rows, got {len(v0)}")
    # Canonical stable reference order, then require exact ID/label/prompt alignment for every matrix.
    v0 = v0.sort_values("id", kind="stable").reset_index(drop=True)
    row_ids = v0.id.astype(str).tolist(); y_text = v0.model.astype(str).to_numpy()
    if set(y_text) != set(LABELS): raise ValueError("V0 canonical label set mismatch")
    label_index = {label: i for i, label in enumerate(LABELS)}
    y = np.asarray([label_index[x] for x in y_text], dtype=np.int64)
    cache_file = output / "checkpoint_probabilities.npz"
    meta_file = output / "prediction_cache_manifest.json"
    trust_hash = _sha(trust_path)
    if resume and cache_file.is_file() and meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        if (meta.get("trust_sha256") == trust_hash and
                meta.get("v0_ids_sha256") == hashlib.sha256("\n".join(row_ids).encode()).hexdigest() and
                meta.get("row_alignment_verified") is True):
            loaded = np.load(cache_file)
            matrices = {sid: loaded[f"p_{sid}"] for sid in ids}
            return trust_path, trust_hash, pointer, trust, eligible, identity, v0, row_ids, y_text, y, matrices, meta
    matrices = {}; sources = {}
    for sid in ids:
        row = identity.loc[sid]
        exp_id = str(row.experiment_id)
        probs, path = _read_saved_probabilities(root, exp_id, row_ids, y_text.tolist())
        schema_columns = set(pq.read_schema(path).names)
        align_columns = ["id", "y_true"] + (["prompt_id"] if "prompt_id" in schema_columns else [])
        cache_frame = pd.read_parquet(path, columns=align_columns)
        if (cache_frame.id.astype(str).tolist() != row_ids or
                cache_frame.y_true.astype(str).tolist() != y_text.tolist()):
            raise ValueError(f"prediction IDs or labels do not align with V0 for {sid}: {path}")
        if "prompt_id" in cache_frame and cache_frame.prompt_id.astype(str).tolist() != v0.prompt_id.astype(str).tolist():
            raise ValueError(f"prediction prompt IDs do not align with V0 for {sid}: {path}")
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != (len(v0), len(LABELS)) or not np.isfinite(probs).all():
            raise ValueError(f"Invalid probability matrix {sid} from {path}")
        matrices[sid] = probs; sources[sid] = {"experiment_id": exp_id,
            "prediction_path": str(Path(path).relative_to(root) if Path(path).is_absolute() else path),
            "prediction_sha256": _sha(Path(path)) if Path(path).is_file() else None}
    np.savez_compressed(cache_file, **{f"p_{sid}": x for sid, x in matrices.items()})
    meta = {"trust_sha256": trust_hash, "trust_list_version": pointer.get("version"),
        "v0_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
        "class_order": LABELS, "checkpoint_ids": ids, "sources": sources,
        "row_alignment_verified": True}
    _atomic_json(meta_file, meta)
    return trust_path, trust_hash, pointer, trust, eligible, identity, v0, row_ids, y_text, y, matrices, meta


def run(*, root: str | Path = ROOT, trust_list: str | None = None,
        output_dir: str | Path = OUT, report_path: str | Path = REPORT,
        resume: bool = True, n_splits: int = N_SPLITS) -> dict:
    root = Path(root).resolve(); output = Path(output_dir)
    if not output.is_absolute(): output = root / output
    output.mkdir(parents=True, exist_ok=True)
    report = Path(report_path)
    if not report.is_absolute(): report = root / report
    (trust_path, trust_hash, pointer, trust, eligible, identity, v0, row_ids,
     y_text, y, matrices, cache_meta) = _load_inputs(root, trust_list, output, resume)
    ids = list(matrices)
    baseline_probs = np.mean([matrices[sid] for sid in EQUAL9], axis=0)
    baseline_probs /= baseline_probs.sum(axis=1, keepdims=True)
    baseline_pred = baseline_probs.argmax(axis=1)
    groups = v0.prompt_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    oof = {key: np.zeros_like(baseline_probs, dtype=np.float64) for key in
           ("equal9", "equal_selected", "per_label_f1", "complementarity")}
    fold_records = []; fold_rules = []; fold_weights = []
    split_list = list(splitter.split(np.zeros(len(y)), y, groups))
    code_hash = _sha(Path(__file__).resolve())
    fold_cache_dir = output / "folds"
    fold_cache_dir.mkdir(exist_ok=True)
    for fold_i, (fit, held) in enumerate(split_list, 1):
        fold_npz = fold_cache_dir / f"fold_{fold_i:02d}.npz"
        fold_json = fold_cache_dir / f"fold_{fold_i:02d}.json"
        if resume and fold_npz.is_file() and fold_json.is_file():
            saved = json.loads(fold_json.read_text())
            if (saved.get("code_sha256") == code_hash and saved.get("trust_sha256") == trust_hash and
                    saved.get("v0_ids_sha256") == cache_meta["v0_ids_sha256"] and
                    saved.get("held_indexes") == held.tolist()):
                arrays = np.load(fold_npz)
                for strategy in oof: oof[strategy][held] = arrays[strategy]
                fold_records.append(saved["fold_record"])
                fold_rules.extend(saved["fold_rules"])
                fold_weights.extend(saved["fold_weights"])
                continue
        selected = _select(fit, ids, matrices, y)
        fold_probabilities = {}
        for strategy in oof:
            if strategy == "equal9":
                probs, details = _make_probabilities("equal9", selected, ids, matrices, y, held)
            elif strategy == "equal_selected":
                probs, details = _make_probabilities(strategy, selected, ids, matrices, y, held)
            else:
                # Fit weights on training groups and score held-out rows only.
                probs, details = _make_probabilities(strategy, selected, ids, matrices, y, held,
                                                     fit_indexes=fit)
            oof[strategy][held] = probs
            fold_probabilities[strategy] = probs
        fold_record = {"fold": fold_i, "fit_rows": len(fit), "heldout_rows": len(held),
            "fit_prompt_groups": sorted(set(groups[fit])), "heldout_prompt_groups": sorted(set(groups[held])),
            "selected_global_checkpoints": selected["selected_global"]}
        fold_records.append(fold_record)
        this_rules=[]; this_weights=[]
        for label in LABELS:
            combo = selected["selected_label_combo"][label]
            this_rules.append({"fold": fold_i, "label": label, **combo})
            for method in ("f1", "complementarity"):
                this_weights.append({"fold": fold_i,"label":label,"method":method,
                    "weights":_weights_for_label(label,selected,method,y,fit,matrices)})
        fold_rules.extend(this_rules); fold_weights.extend(this_weights)
        np.savez_compressed(fold_npz, **fold_probabilities)
        _atomic_json(fold_json,{"code_sha256":code_hash,"trust_sha256":trust_hash,
            "v0_ids_sha256":cache_meta["v0_ids_sha256"],"held_indexes":held.tolist(),
            "fold_record":fold_record,"fold_rules":this_rules,"fold_weights":this_weights})
    # Full-V0 selections are explicitly diagnostic, not independent estimates.
    all_idx = np.arange(len(y))
    full_selection_path = output / "selected_rules_full_v0.json"
    full_selection_meta = output / "full_selection_manifest.json"
    full_selected = None
    if resume and full_selection_path.is_file() and full_selection_meta.is_file():
        saved_meta = json.loads(full_selection_meta.read_text())
        if saved_meta.get("code_sha256") == code_hash and saved_meta.get("trust_sha256") == trust_hash:
            full_selected = json.loads(full_selection_path.read_text())
    if full_selected is None:
        full_selected = _select(all_idx, ids, matrices, y)
        _atomic_json(full_selection_path, full_selected)
        _atomic_json(full_selection_meta,{"code_sha256":code_hash,"trust_sha256":trust_hash,
            "v0_ids_sha256":cache_meta["v0_ids_sha256"]})
    full_probs = {}; full_details = {}
    for strategy in oof:
        if strategy == "equal9": full_probs[strategy], full_details[strategy] = _make_probabilities(strategy, full_selected, ids, matrices, y, all_idx)
        elif strategy == "equal_selected": full_probs[strategy], full_details[strategy] = _make_probabilities(strategy, full_selected, ids, matrices, y, all_idx)
        else: full_probs[strategy], full_details[strategy] = _make_probabilities(strategy, full_selected, ids, matrices, y, all_idx)
    filtered_pool = sorted(set().union(*(set(v) for v in full_selected["retained_by_label"].values())),
                           key=lambda x: int(x[1:]))
    results = {"cross_fitted": {}, "full_v0_diagnostic": {}, "folds": fold_records,
        "trust_list_version": pointer.get("version"), "trust_list_path": str(trust_path.relative_to(root)),
        "trust_sha256": trust_hash, "v0_rows": len(y), "prompt_groups": int(pd.Series(groups).nunique()),
        "checkpoint_count": len(ids), "candidate_ids": ids,
        "candidate_count_before_filter": len(ids), "candidate_count_after_filter": len(filtered_pool),
        "filtered_candidate_ids": filtered_pool,
        "methodology": {"max_checkpoints_per_label_rule": MAX_PER_LABEL,
            "f1_near_best_cutoff": F1_NEAR_BEST, "selection_metric": "union top-1-label F1; ties precision, recall, fewer checkpoints, ID order",
            "equal_strategy": "EQUAL9 plus union of each label's fitting-data selected 1-3 checkpoint rule, equal global probability weights",
            "f1_weights": "selected per-label rule members weighted by their fitting-fold per-label F1, normalized per label",
            "complementarity_weights": "per-label precision*recall*marginal_precision*marginal_TP_coverage, normalized; marginal TP/FP computed in deterministic incremental rule order",
            "folds": n_splits, "splitter": "GroupKFold(prompt_id)"}}
    for strategy in oof:
        for scope, probs in (("cross_fitted", oof[strategy]), ("full_v0_diagnostic", full_probs[strategy])):
            m = _evaluate(y, probs); pred = m.pop("y_pred")
            m.update(_comparison(baseline_pred, pred, y)); results[scope][strategy] = m
    per_label_ensemble_rows=[]
    for scope in ("cross_fitted", "full_v0_diagnostic"):
        for strategy in oof:
            for label, metrics in results[scope][strategy]["per_label"].items():
                per_label_ensemble_rows.append({"scope":scope,"strategy":strategy,"label":label,**metrics})
    pd.DataFrame(per_label_ensemble_rows).to_csv(output/"per_label_ensemble_metrics.csv",index=False)
    # Full-data per-label candidate and exact-overlap combination inventory.
    overlap_rows=[]; candidate_rows=[]
    for ci,label in enumerate(LABELS):
        retained = full_selected["retained_by_label"][label]
        stats = full_selected["stats_by_label"][label]
        for sid in ids:
            s=stats[sid]; candidate_rows.append({"label":label,"checkpoint_id":sid,"retained_for_label":sid in retained,**s})
        for rank,item in enumerate(full_selected["combos_by_label"][label],1):
            overlap_rows.append({"label":label,"rank":rank,**item})
    pd.DataFrame(candidate_rows).to_csv(output/"per_label_checkpoint_metrics.csv",index=False)
    pd.DataFrame(overlap_rows).to_csv(output/"label_overlap_combinations.csv",index=False)
    _atomic_json(output/"fold_selected_rules.json",{"folds":fold_records,"rules":fold_rules})
    _atomic_json(output/"selected_weights.json",{"fold_weights":fold_weights,
        "full_v0":{method:{label:_weights_for_label(label,full_selected,method,y,all_idx,matrices)
            for label in LABELS} for method in ("f1","complementarity")},
        "full_v0_equal_checkpoints":full_selected["selected_global"]})
    pred_frame=pd.DataFrame({"id":row_ids,"prompt_id":groups,"y_true":y_text})
    for strategy in oof:
        pred_frame[f"pred_oof::{strategy}"]=np.asarray(LABELS,dtype=object)[oof[strategy].argmax(axis=1)]
        pred_frame[f"pred_full::{strategy}"]=np.asarray(LABELS,dtype=object)[full_probs[strategy].argmax(axis=1)]
        for ci,label in enumerate(LABELS):
            pred_frame[f"prob_oof::{strategy}::{label}"]=oof[strategy][:,ci]
            pred_frame[f"prob_full::{strategy}::{label}"]=full_probs[strategy][:,ci]
    pred_frame.to_parquet(output/"row_predictions.parquet",index=False)
    _atomic_json(output/"metrics.json",results)
    _atomic_json(output/"experiment_config.json",{"trust_list_version":pointer.get("version"),
        "trust_list_path":str(trust_path.relative_to(root)),"trust_list_sha256":trust_hash,
        "v0_rows":len(y),"v0_ids_sha256":cache_meta["v0_ids_sha256"],"class_order":LABELS,
        "eligible_checkpoint_count":len(eligible.sequence_id.unique()),"baseline":"EQUAL9",
        "equal9_ids":EQUAL9,"checkpoint_sources":cache_meta["sources"],"methodology":results["methodology"]})
    lines=["# Precision, recall and overlap-based ensemble optimization","",
        f"Trust List 2: `{pointer.get('version')}` (`{trust_path.relative_to(root)}`), SHA-256 `{trust_hash}`. V0: {len(y):,} rows, {len(LABELS)} classes, {len(set(groups))} prompt groups.",
        "All prediction matrices were loaded from existing full-V0 artifacts. Row IDs and true labels were checked against V0; prompt IDs were also checked when present, and otherwise attached from V0 by exact row-ID alignment. The full-V0 diagnostic selects and scores on the same rows; only the prompt-group cross-fitted results are held out within this analysis. Checkpoint development/selection may still have used V0, so neither scope is an independent Kaggle estimate.","",
        "## Candidate selection","",
        f"Eligible checkpoint count before filtering: **{len(ids)}**. Filtered candidate union: **{len(filtered_pool)}**. The candidate set includes each label's tied best-F1, best-precision and best-recall checkpoints, all candidates within 5 percentage points of best F1, and every EQUAL9 member.","",
        "Per-checkpoint counts and metrics are in `per_label_checkpoint_metrics.csv`. The exact top-1 prediction overlap of all 1–3 member combinations per label is in `label_overlap_combinations.csv`; columns include shared/unique TP and FP plus incremental contributions.","",
        "## Ensemble comparison","",
        "| Scope | Strategy | Accuracy | Balanced accuracy | Log loss | Corrected vs EQUAL9 | Introduced vs EQUAL9 | Disagreements |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for scope,title in (("cross_fitted","Prompt-group OOF"),("full_v0_diagnostic","Full-V0 diagnostic")):
        for strategy in oof:
            m=results[scope][strategy]
            lines.append(f"| {title} | {strategy} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['log_loss']:.6f} | {m['corrected']} | {m['introduced']} | {m['disagreements']} |")
    base=results["full_v0_diagnostic"]["equal9"]
    lines += ["",f"EQUAL9 full-V0 reference: **{base['balanced_accuracy']:.4%} balanced accuracy**, {base['accuracy']:.4%} accuracy, log loss {base['log_loss']:.6f}.","",
        "## Best label-specific rules and overlap","",
        "The following are the best full-V0 label rules by union F1 (diagnostic only). Use `label_overlap_combinations.csv` for all combinations and fold-specific selections. Strategy-specific per-label precision, recall and F1 for both scopes are in `per_label_ensemble_metrics.csv`.","",
        "| Label | Checkpoints | TP | FP | Precision | Recall | F1 | Shared TP/FP | Unique TP/FP | Incremental TP/FP by rule |","|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for label in LABELS:
        item=full_selected["selected_label_combo"][label]
        inc="; ".join(f"{x['checkpoint_id']} +{x['incremental_tp']}/+{x['incremental_fp']}" for x in item["incremental_contributions"])
        lines.append(f"| {label} | {', '.join(item['checkpoint_ids'])} | {item['tp']} | {item['fp']} | {item['precision']:.1%} | {item['recall']:.1%} | {item['f1']:.1%} | {item['shared_tp']}/{item['shared_fp']} | {item['unique_tp']}/{item['unique_fp']} | {inc} |")
    lines += ["","## Per-label cross-fitted precision, recall and F1","",
        "Each cell is precision / recall / F1 from pooled held-out prompt groups. Full-V0 diagnostic per-label metrics are in `per_label_ensemble_metrics.csv`.","",
        "| Label | EQUAL9 | Equal selected | Per-label F1 | Complementarity |","|---|---|---|---|---|"]
    for label in LABELS:
        cells=[]
        for strategy in oof:
            m=results["cross_fitted"][strategy]["per_label"][label]
            cells.append(f"{m['precision']:.1%} / {m['recall']:.1%} / {m['f1']:.1%}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["","## Recommendation","",
        "EQUAL9 remains the strongest of these strategies on prompt-group OOF balanced accuracy. None of the new strategies beats its 65.3898% OOF reference, so this experiment does not recommend replacing EQUAL9 for Kaggle. Full-V0 per-label F1 weighting looks better only on the same rows used to select candidates and weights; treat that as selection optimism, not evidence of a generalization gain.","",
        "## Reproducibility files","",
        "- `experiment_config.json`: trust-list/V0 hashes, checkpoint sources and method definitions.",
        "- `checkpoint_probabilities.npz` and `prediction_cache_manifest.json`: cached aligned input matrices and provenance.",
        "- `per_label_checkpoint_metrics.csv`: precision/recall/F1/TP/FP for every candidate and label.",
        "- `per_label_ensemble_metrics.csv`: per-label precision/recall/F1 for each strategy in both evaluation scopes.",
        "- `label_overlap_combinations.csv`: exact 1–3 checkpoint overlap and incremental contribution tables.",
        "- `fold_selected_rules.json` and `selected_weights.json`: cross-fit rules and learned weights.",
        "- `row_predictions.parquet` and `metrics.json`: OOF/full diagnostic outputs and metrics.",""]
    report.write_text("\n".join(lines),encoding="utf-8")
    state={"status":"completed","trust_list_version":pointer.get("version"),"trust_sha256":trust_hash,
        "v0_ids_sha256":cache_meta["v0_ids_sha256"],"folds_completed":n_splits,"output_dir":str(output.relative_to(root))}
    _atomic_json(output/"analysis_state.json",state)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trust-list",default=None);p.add_argument("--output-dir",default=str(OUT))
    p.add_argument("--report",default=str(REPORT));p.add_argument("--folds",type=int,default=N_SPLITS)
    p.add_argument("--no-resume",action="store_true")
    a=p.parse_args()
    result=run(trust_list=a.trust_list,output_dir=a.output_dir,report_path=a.report,
        resume=not a.no_resume,n_splits=a.folds)
    print(json.dumps({"report":a.report,"output_dir":a.output_dir,
        "cross_fitted":{k:{m:result["cross_fitted"][k][m] for m in ("accuracy","balanced_accuracy","log_loss","corrected","introduced","disagreements")} for k in result["cross_fitted"]},
        "full_v0_diagnostic":{k:{m:result["full_v0_diagnostic"][k][m] for m in ("accuracy","balanced_accuracy","log_loss","corrected","introduced","disagreements")} for k in result["full_v0_diagnostic"]}},indent=2))


if __name__ == "__main__": main()
