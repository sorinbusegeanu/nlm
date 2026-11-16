"""Search equal-weight ensembles from a per-label-accuracy-filtered Trust List 2 pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS, _read_saved_probabilities

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
OUT = Path("artifacts/analysis/ensemble_search_filtered")
REPORT = Path("docs/ensemble_search_filtered.md")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _false(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"false", "0", "no", "n", ""})


def _sort_key(record: dict) -> tuple:
    return (-float(record["balanced_accuracy"]), -float(record["accuracy"]),
            float(record["log_loss"]), tuple(record["checkpoint_ids"]))


def _result(y_idx: np.ndarray, support: np.ndarray, matrix: np.ndarray,
            combo: tuple[str, ...]) -> dict:
    # Normalize each blended row to remove harmless floating-point sum drift.
    probs = matrix / len(combo)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    correct = pred == y_idx
    counts = np.bincount(y_idx * len(LABELS) + pred,
                         minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
    recall = np.diag(counts) / np.maximum(support, 1)
    precision = np.diag(counts) / np.maximum(counts.sum(axis=0), 1)
    loss = -np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)).mean()
    return {"checkpoint_ids": list(combo), "checkpoint_count": len(combo),
            "balanced_accuracy": float(recall.mean()), "accuracy": float(correct.mean()),
            "log_loss": float(loss), "correct": int(correct.sum()),
            "per_label_precision": precision.tolist(), "per_label_recall": recall.tolist()}


def _write_report(path: Path, version: str, trust_path: Path, trust_sha: str,
                  threshold: float, best_by_label: dict, retained_by_label: dict,
                  candidate_count_before: int, candidate_pool: list[str],
                  total_combos: int, evaluated_count: int, scored_size_count: int,
                  search: dict, baseline: dict, baseline_per_label: dict,
                  top_overall: list[dict], top_by_size: dict, pairwise: list[dict],
                  beam_width: int, swap_seeds: int, swap_passes: int) -> None:
    lines = ["# Accuracy-filtered Trust List 2 ensemble search", "",
        f"Trust List: `{trust_path.relative_to(ROOT)}` (`{version}`), SHA-256 `{trust_sha}`.",
        "V0 contains 2,699 rows. ‘Per-label accuracy’ is the Trust List recall for that label: correct predictions divided by actual examples of that label.",
        "Eligibility required a published full-V0 row, an eligible rule, an available checkpoint and no training/validation overlap.",
        "Filtering and ensemble optimization both use full V0, so all scores and selections are development diagnostics and are selection-optimistic for Kaggle.", "",
        "## Filtering", "",
        f"Eligible checkpoints before filtering: **{candidate_count_before}**. Retained pool: **{len(candidate_pool)}**. Threshold: best per-label recall minus **{threshold:.1%}** (inclusive).", "",
        "| Label | Best per-label accuracy | Cutoff | Retained checkpoints (accuracy) |", "|---|---:|---:|---|"]
    for label in LABELS:
        best = best_by_label[label]
        retained = retained_by_label[label]
        rendered = ", ".join(f"{x['sequence_id']} ({x['recall']:.1%})" for x in retained)
        lines.append(f"| {label} | {best:.1%} | {best-threshold:.1%} | {rendered} |")
    lines += ["", f"Candidate pool ({len(candidate_pool)}): `{', '.join(candidate_pool)}`.", "",
        "## Search coverage", "",
        f"Exact combinations for sizes 5–10: **{total_combos:,}**. Unique combinations scored by the search: **{scored_size_count:,}** ({scored_size_count/max(1,total_combos):.2%} coverage). The full journal includes **{evaluated_count:,}** unique scores, including singleton-to-four-member beam nodes and the EQUAL9 reference.",
        f"Method: deterministic beam search (width {beam_width}) with up to {swap_seeds} local-search seeds per size and {swap_passes} rounds of one-checkpoint swaps. `evaluated_combinations.jsonl` and `search_state.json` make the run resumable.", "",
        "## Best 10 overall", "",
        "| Rank | Size | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, record in enumerate(top_overall, 1):
        lines.append(f"| {rank} | {record['checkpoint_count']} | {', '.join(record['checkpoint_ids'])} | {record['balanced_accuracy']:.4%} | {record['accuracy']:.4%} | {record['log_loss']:.6f} | {(record['balanced_accuracy']-baseline['balanced_accuracy'])*100:+.3f} pp | {record['corrected_vs_baseline']} | {record['introduced_vs_baseline']} |")
    lines += ["", "## Best 3 for each ensemble size", ""]
    for size in range(5, 11):
        lines += [f"### {size} checkpoints", "", "| Rank | Checkpoints | Balanced accuracy | Accuracy | Log loss | Δ BA vs EQUAL9 | Corrected | Introduced |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for rank, record in enumerate(top_by_size[size], 1):
            lines.append(f"| {rank} | {', '.join(record['checkpoint_ids'])} | {record['balanced_accuracy']:.4%} | {record['accuracy']:.4%} | {record['log_loss']:.6f} | {(record['balanced_accuracy']-baseline['balanced_accuracy'])*100:+.3f} pp | {record['corrected_vs_baseline']} | {record['introduced_vs_baseline']} |")
        lines.append("")
    lines += ["## Per-label recall and precision", "",
        "The following table compares EQUAL9 with the best overall searched ensemble. Full per-label precision and recall for the best ten and each size's top three are in `per_label_metrics.csv` and `evaluation_metrics.json`.", "",
        "| Label | EQUAL9 recall / precision | Best-search recall / precision | Δ recall / precision |", "|---|---:|---:|---:|"]
    best = top_overall[0]
    for i, label in enumerate(LABELS):
        p0, r0 = baseline_per_label[label]["precision"], baseline_per_label[label]["recall"]
        p1, r1 = best["per_label_precision"][i], best["per_label_recall"][i]
        lines.append(f"| {label} | {r0:.1%} / {p0:.1%} | {r1:.1%} / {p1:.1%} | {(r1-r0)*100:+.1f} / {(p1-p0)*100:+.1f} pp |")
    lines += ["", "## Prediction disagreement among the best three", "",
        "Pairwise top-1 disagreement on full V0 for each size's best three. The full disagreement matrices and prediction artifacts are saved under the analysis output directory.", "",
        "| Size | Ensemble pair | Rows different | Disagreement |", "|---:|---|---:|---:|"]
    for row in pairwise:
        lines.append(f"| {row['size']} | #{row['rank_left']} vs #{row['rank_right']} | {row['different']} | {row['rate']:.2%} |")
    lines += ["", "## Kaggle candidates", "",
        f"The highest V0-scoring searched ensemble is `{', '.join(top_overall[0]['checkpoint_ids'])}` ({top_overall[0]['balanced_accuracy']:.4%} balanced accuracy). The current EQUAL9 baseline is {baseline['balanced_accuracy']:.4%}.",
        "Use the best-ranked configurations as candidates for later testing, while treating differences on this same V0 set as optimistic. Do not interpret V0 balanced accuracy as a reliable Kaggle score.", "",
        "## Reproducibility artifacts", "",
        "- `experiment_configs.json`: dataset/version, eligibility/filtering rules, exact selected checkpoint lists and equal weights, search parameters, cache paths and hashes.",
        "- `evaluation_metrics.json`: baseline, best ten, best three per size and per-label metrics.",
        "- `ensemble_predictions.parquet`: row-aligned predictions and probability vectors for baseline, best ten and per-size top three.",
        "- `per_label_metrics.csv`: per-label precision/recall for all reported ensembles.",
        "- `evaluated_combinations.jsonl`: resumable score journal.",
        "- `search_state.json`: last completed beam size and search state.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(*, root: str | Path = ROOT, trust_list: str | None = None,
        output_dir: str | Path = OUT, report_path: str | Path = REPORT,
        threshold: float = 0.05, beam_width: int = 256,
        swap_seeds: int = 12, swap_passes: int = 5, resume: bool = True) -> dict:
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute(): output_dir = root / output_dir
    report_path = Path(report_path)
    if not report_path.is_absolute(): report_path = root / report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = root / "docs/trust-list-2-latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    trust_path = Path(trust_list) if trust_list else root / pointer["csv_path"]
    if not trust_path.is_absolute(): trust_path = root / trust_path
    if not trust_path.is_file(): raise FileNotFoundError(f"Trust List 2 CSV missing: {trust_path}")
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    required = {"sequence_id", "experiment_id", "model", "training_dataset", "checkpoint_path",
                "target_label", "recall", "evaluation_eligible", "rule_eligible",
                "checkpoint_available", "validation_training_overlap", "validation_rows"}
    if required - set(trust.columns): raise ValueError(f"Trust List missing: {sorted(required-set(trust.columns))}")
    mask = (_truthy(trust.evaluation_eligible) & _truthy(trust.rule_eligible) &
            _truthy(trust.checkpoint_available) & _false(trust.validation_training_overlap) &
            (pd.to_numeric(trust.validation_rows, errors="coerce") == 2699))
    eligible = trust.loc[mask].copy()
    eligible["recall_num"] = pd.to_numeric(eligible.recall, errors="coerce")
    if eligible.recall_num.isna().any(): raise ValueError("eligible Trust List rows have missing recall")
    cp_identity_cols = ["experiment_id", "model", "training_dataset", "checkpoint_path"]
    cp_metadata = eligible.groupby("sequence_id", sort=False)[cp_identity_cols].nunique(dropna=False)
    if (cp_metadata > 1).any().any(): raise ValueError("a sequence ID maps to multiple checkpoint identities")
    candidate_count_before = int(eligible.sequence_id.nunique())
    best_by_label, retained_by_label, pool = {}, {}, set()
    for label in LABELS:
        rows = eligible.loc[eligible.target_label.eq(label)].copy()
        if rows.empty: raise ValueError(f"no eligible candidates for label {label}")
        best = float(rows.recall_num.max())
        cutoff = best - threshold
        retained = rows.loc[rows.recall_num >= cutoff].sort_values(
            ["recall_num", "sequence_id"], ascending=[False, True], kind="stable")
        best_by_label[label] = best
        retained_by_label[label] = [{"sequence_id": str(r.sequence_id), "recall": float(r.recall_num),
                                     "experiment_id": str(r.experiment_id)} for r in retained.itertuples()]
        pool.update(retained.sequence_id.astype(str))
    candidate_pool = sorted(pool, key=lambda x: int(x[1:]))
    identity = (eligible.sort_values("sequence_id", kind="stable")
                .drop_duplicates("sequence_id").set_index("sequence_id"))
    for sid in candidate_pool + [x for x in BASELINE if x not in pool]:
        if sid not in identity.index:
            rows = trust.loc[trust.sequence_id.astype(str).eq(sid)]
            if rows.empty: raise KeyError(f"checkpoint {sid} (including baseline) missing from Trust List")
            identity.loc[sid] = rows.iloc[0]
    # Checkpoints are loaded by exact experiment identity; caches are aligned to sorted V0 IDs.
    v0 = _dataset_rows("v0", root / "data/processed")
    if len(v0) != 2699: raise ValueError(f"expected 2699 V0 rows, got {len(v0)}")
    ids = v0.id.astype(str).tolist(); y_text = v0.model.astype(str).tolist()
    y_lookup = {label: i for i, label in enumerate(LABELS)}
    if set(y_text) != set(LABELS): raise ValueError("V0 does not contain the canonical 18 labels")
    y_idx = np.fromiter((y_lookup[v] for v in y_text), dtype=np.int64, count=len(y_text))
    support = np.bincount(y_idx, minlength=len(LABELS))
    matrix_by_id, cache_info = {}, {}
    for sid in candidate_pool + [x for x in BASELINE if x not in candidate_pool]:
        row = identity.loc[sid]
        checkpoint = Path(str(row.checkpoint_path))
        if not checkpoint.is_absolute(): checkpoint = root / checkpoint
        exp_id = str(row.experiment_id)
        probs, prediction_path = _read_saved_probabilities(root, exp_id, ids, y_text)
        matrix_by_id[sid] = probs.astype(np.float32)
        cache_info[sid] = {"experiment_id": exp_id, "model": str(row.model),
            "training_dataset": str(row.training_dataset), "checkpoint_path": str(checkpoint),
            "checkpoint_path_exists": checkpoint.exists(),
            "checkpoint_sha256": _sha(checkpoint) if checkpoint.is_file() else None,
            "prediction_path": str(prediction_path.relative_to(root)),
            "prediction_sha256": _sha(prediction_path)}

    score_journal = output_dir / "evaluated_combinations.jsonl"
    scored: dict[tuple[str, ...], dict] = {}
    if resume and score_journal.is_file():
        with score_journal.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                item = json.loads(line)
                combo = tuple(item["checkpoint_ids"])
                scored[combo] = item
    journal = score_journal.open("a", encoding="utf-8", buffering=1)
    new_since_sync = 0
    def score(combo_values: tuple[str, ...]) -> dict:
        nonlocal new_since_sync
        combo = tuple(sorted(combo_values, key=lambda x: int(x[1:])))
        if combo in scored: return scored[combo]
        total = np.zeros_like(matrix_by_id[combo[0]], dtype=np.float32)
        for sid in combo: total += matrix_by_id[sid]
        record = _result(y_idx, support, total, combo)
        journal.write(json.dumps(record, separators=(",", ":")) + "\n")
        scored[combo] = record
        new_since_sync += 1
        if new_since_sync >= 100:
            journal.flush(); new_since_sync = 0
        return record
    def better(a: dict, b: dict) -> bool:
        return _sort_key(a) < _sort_key(b)
    def refresh_state(status: str, size: int, beam: list[tuple[str, ...]]) -> None:
        journal.flush()
        _atomic_json(output_dir / "search_state.json", {"status": status,
            "trust_list": str(trust_path.relative_to(root)), "trust_list_version": pointer.get("version"),
            "threshold": threshold, "candidate_pool": candidate_pool,
            "beam_width": beam_width, "swap_seeds_per_size": swap_seeds,
            "swap_passes": swap_passes, "completed_size": size,
            "beam": [list(x) for x in beam], "unique_combinations_scored": len(scored)})

    refresh_state("running", 0, [])
    baseline_combo = tuple(BASELINE)
    baseline_record = score(baseline_combo)
    # Beam starts with all individual candidates, allowing any checkpoint to seed a combination.
    singleton = [score((sid,)) for sid in candidate_pool]
    beam = [tuple(x["checkpoint_ids"]) for x in sorted(singleton, key=_sort_key)[:beam_width]]
    refresh_state("running", 1, beam)
    for size in range(2, 11):
        generated = set()
        for node in beam:
            for sid in candidate_pool:
                if sid not in node:
                    generated.add(tuple(sorted((*node, sid), key=lambda x: int(x[1:]))))
        expansion_records = [score(combo) for combo in sorted(generated, key=lambda c: tuple(int(x[1:]) for x in c))]
        ranked = sorted(expansion_records, key=_sort_key)
        # One-for-one local swaps from the strongest current beam nodes improve combinations
        # that forward extension can miss, while keeping runtime bounded and reproducible.
        seeds = [tuple(x["checkpoint_ids"]) for x in ranked[:swap_seeds]]
        local_optima = []
        for seed in seeds:
            current = seed
            current_record = scored[current]
            for _ in range(swap_passes):
                best_record = current_record
                best_combo = current
                outside = [sid for sid in candidate_pool if sid not in current]
                for remove in current:
                    for add in outside:
                        proposal = tuple(sorted((set(current) - {remove}) | {add},
                                                key=lambda x: int(x[1:])))
                        candidate_record = score(proposal)
                        if better(candidate_record, best_record):
                            best_record, best_combo = candidate_record, proposal
                if best_combo == current: break
                current, current_record = best_combo, best_record
            local_optima.append(current)
        keep = set(beam) if size == 1 else set()
        for record in ranked[:beam_width]: keep.add(tuple(record["checkpoint_ids"]))
        for combo in local_optima: keep.add(combo)
        beam = [tuple(r["checkpoint_ids"]) for r in sorted(
            (scored[c] for c in keep if len(c) == size), key=_sort_key)[:beam_width]]
        refresh_state("running", size, beam)
    baseline_metrics = _result(y_idx, support,
        sum((matrix_by_id[sid] for sid in BASELINE), np.zeros_like(matrix_by_id[BASELINE[0]], dtype=np.float32)),
        tuple(sorted(BASELINE, key=lambda x: int(x[1:]))))
    # Include delta and corrected/introduced counts relative to EQUAL9 for report candidates.
    base_sum = sum((matrix_by_id[sid] for sid in BASELINE), np.zeros_like(matrix_by_id[BASELINE[0]], dtype=np.float32))
    base_p = base_sum / len(BASELINE); base_p /= base_p.sum(axis=1, keepdims=True)
    base_pred = base_p.argmax(axis=1); base_correct = base_pred == y_idx
    def enrich(record: dict) -> dict:
        combo = tuple(record["checkpoint_ids"])
        total = sum((matrix_by_id[sid] for sid in combo), np.zeros_like(matrix_by_id[combo[0]], dtype=np.float32))
        metrics = _result(y_idx, support, total, combo)
        pred = (total / len(combo)).argmax(axis=1)
        correct = pred == y_idx
        metrics.update({"delta_balanced_accuracy_pp_vs_baseline": (metrics["balanced_accuracy"]-baseline_metrics["balanced_accuracy"])*100,
            "corrected_vs_baseline": int((~base_correct & correct).sum()),
            "introduced_vs_baseline": int((base_correct & ~correct).sum()),
            "disagreement_count_vs_baseline": int((pred != base_pred).sum()),
            "disagreement_rate_vs_baseline": float((pred != base_pred).mean())})
        return metrics
    pool_set = set(candidate_pool)
    # The EQUAL9 reference is scored for comparison, but it is not a searched
    # candidate unless all of its members independently survived filtering.
    search_records = [
        r for combo, r in scored.items()
        if 5 <= len(combo) <= 10 and set(combo) <= pool_set
    ]
    top_overall_raw = sorted(search_records, key=_sort_key)[:10]
    top_overall = [enrich(r) for r in top_overall_raw]
    top_by_size = {size:[enrich(r) for r in sorted((x for x in search_records if len(x["checkpoint_ids"])==size), key=_sort_key)[:3]]
                   for size in range(5,11)}
    # Baseline metrics already includes per-label precision/recall.
    baseline_result = enrich({"checkpoint_ids":list(baseline_combo)})
    # Keep the computed baseline tied to the actual exact probability average.
    baseline_result["balanced_accuracy"] = baseline_metrics["balanced_accuracy"]
    all_reported = {}
    for record in top_overall + [r for values in top_by_size.values() for r in values]:
        all_reported[tuple(record["checkpoint_ids"])] = record
    unique_reported = list(all_reported.values())
    selected = [baseline_result] + unique_reported
    prediction_columns = {"id":ids,"prompt_id":v0.prompt_id.astype(str).tolist(),"y_true":y_text}
    per_label_rows = []
    for record in selected:
        name = "EQUAL9" if record["checkpoint_ids"] == BASELINE else "+".join(record["checkpoint_ids"])
        combo = record["checkpoint_ids"]
        total = sum((matrix_by_id[sid] for sid in combo), np.zeros_like(matrix_by_id[combo[0]], dtype=np.float32))
        probs = total / len(combo); probs /= probs.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        prediction_columns[f"pred::{name}"] = np.asarray(LABELS, dtype=object)[pred]
        prediction_columns[f"correct::{name}"] = pred == y_idx
        for i,label in enumerate(LABELS):
            prediction_columns[f"prob::{name}::{label}"] = probs[:,i]
            p0 = baseline_result["per_label_precision"][i]
            r0 = baseline_result["per_label_recall"][i]
            per_label_rows.append({"ensemble":name,"checkpoint_ids":" ".join(combo),"label":label,
                "precision":record["per_label_precision"][i],"recall":record["per_label_recall"][i],
                "precision_delta_pp_vs_EQUAL9":(record["per_label_precision"][i]-p0)*100,
                "recall_delta_pp_vs_EQUAL9":(record["per_label_recall"][i]-r0)*100})
    pd.DataFrame(prediction_columns).to_parquet(output_dir/"ensemble_predictions.parquet",index=False)
    pd.DataFrame(per_label_rows).to_csv(output_dir/"per_label_metrics.csv",index=False)
    total_combos = sum(math.comb(len(candidate_pool), k) for k in range(5, 11))
    scored_size_count = sum(1 for c in scored if 5 <= len(c) <= 10)
    top_pairwise=[]
    for size, records in top_by_size.items():
        for i in range(len(records)):
            for j in range(i+1,len(records)):
                left=records[i]["checkpoint_ids"]; right=records[j]["checkpoint_ids"]
                def predict(c):
                    s=sum((matrix_by_id[sid] for sid in c),np.zeros_like(matrix_by_id[c[0]],dtype=np.float32))
                    return (s/len(c)).argmax(axis=1)
                different=int((predict(left)!=predict(right)).sum())
                top_pairwise.append({"size":size,"rank_left":i+1,"rank_right":j+1,"different":different,"rate":different/len(y_idx)})
    best_by_label_score={label:best_by_label[label] for label in LABELS}
    configs={"dataset":"v0","validation_rows":2699,"class_order":list(LABELS),
        "trust_list_path":str(trust_path.relative_to(root)),"trust_list_version":pointer.get("version"),
        "trust_list_sha256":_sha(trust_path),"eligibility":{
            "evaluation_eligible":True,"rule_eligible":True,"checkpoint_available":True,
            "validation_training_overlap":False,"validation_rows":2699},
        "per_label_accuracy_definition":"Trust List recall: TP / actual rows for label; used as per-class accuracy.",
        "threshold_pp":threshold,"candidate_count_before_filter":candidate_count_before,
        "candidate_pool":candidate_pool,"retained_by_label":retained_by_label,
        "checkpoint_sources":cache_info,"baseline":{"name":"EQUAL9","checkpoint_ids":BASELINE,
            "weights":{sid:1/len(BASELINE) for sid in BASELINE}},
        "search":{"strategy":"deterministic beam search plus one-for-one checkpoint swaps",
            "beam_width":beam_width,"swap_seeds_per_size":swap_seeds,"swap_passes":swap_passes,
            "all_combinations_sizes_5_to_10":total_combos,"unique_combinations_scored_total":len(scored),
            "unique_size_5_to_10_scored":scored_size_count,"resumable":True},
        "selected_ensembles":{"best_overall":top_overall,"best_by_size":top_by_size},
        "baseline_metrics":baseline_result,"pairwise_disagreement_best_three_by_size":top_pairwise}
    _atomic_json(output_dir/"experiment_configs.json",configs)
    _atomic_json(output_dir/"evaluation_metrics.json",{"baseline":baseline_result,
        "best_overall":top_overall,"best_by_size":top_by_size,"pairwise_disagreement_best_three_by_size":top_pairwise,
        "candidate_count_before_filter":candidate_count_before,"candidate_pool":candidate_pool,
        "combinations_total":total_combos,"combinations_scored":len(scored),
        "size_5_to_10_scored":scored_size_count,"per_label_filter_best_accuracy":best_by_label_score})
    _write_report(report_path,str(pointer.get("version")),trust_path,_sha(trust_path),threshold,
        best_by_label,retained_by_label,candidate_count_before,candidate_pool,total_combos,len(scored),
        scored_size_count,configs,baseline_result,{label:{"precision":baseline_result["per_label_precision"][i],
            "recall":baseline_result["per_label_recall"][i]} for i,label in enumerate(LABELS)},
        top_overall,top_by_size,top_pairwise,beam_width,swap_seeds,swap_passes)
    refresh_state("completed",10,beam)
    journal.close()
    print(json.dumps({"report":str(report_path),"output_dir":str(output_dir),
        "trust_list_version":pointer.get("version"),"candidate_count_before":candidate_count_before,
        "candidate_count_after":len(candidate_pool),"combinations_total":total_combos,
        "combinations_scored":len(scored),"size_5_to_10_scored":scored_size_count,
        "baseline_balanced_accuracy":baseline_result["balanced_accuracy"],
        "best_overall":top_overall[0] if top_overall else None,
        "best_by_size":{k:v[0] if v else None for k,v in top_by_size.items()}},indent=2,default=str))
    return configs


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-list")
    parser.add_argument("--output-dir",default=str(OUT))
    parser.add_argument("--report",default=str(REPORT))
    parser.add_argument("--threshold",type=float,default=0.05)
    parser.add_argument("--beam-width",type=int,default=256)
    parser.add_argument("--swap-seeds",type=int,default=12)
    parser.add_argument("--swap-passes",type=int,default=5)
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    run(trust_list=args.trust_list,output_dir=args.output_dir,report_path=args.report,
        threshold=args.threshold,beam_width=args.beam_width,swap_seeds=args.swap_seeds,
        swap_passes=args.swap_passes,resume=not args.no_resume)


if __name__=="__main__":
    main()
