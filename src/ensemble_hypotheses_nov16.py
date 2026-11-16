"""Cache-only evaluation of the November 16 Trust List 2 ensemble hypotheses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             log_loss, precision_recall_fscore_support)

from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS, _read_saved_probabilities

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
EXPERIMENTS = {
    "EQUAL9": BASELINE,
    "E1_replace_ultrabert": ["N0105", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"],
    "E2_add_diversity": ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048", "N0105", "N0106", "N0081"],
    "E3_refresh_neural": ["N0105", "N0005", "N0081", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048", "N0106"],
}
EXPECTED_BASELINE_BA = 0.653898
OUT = Path("artifacts/analysis/ensemble_hypotheses_nov16")
REPORT = Path("docs/ensemble_hypotheses_nov16.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trust_source(root: Path, trust_path: str | None) -> tuple[Path, dict]:
    if trust_path:
        path = Path(trust_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Trust List 2 export not found: {path}")
        return path, {"source": "explicit_file", "latest_pointer": None}
    pointer_path = root / "docs/trust-list-2-latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = root / pointer["csv_path"]
    if not path.is_file():
        raise FileNotFoundError(f"latest Trust List 2 CSV missing: {path}")
    return path, {"source": "repository_latest_pointer", "latest_pointer": str(pointer_path.relative_to(root)),
                  "latest_pointer_metadata": pointer}


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, probabilities, labels=LABELS)),
        "correct": int((y == predicted).sum()),
        "predictions": predicted,
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i]),
                               "correct": int(((y == label) & (predicted == label)).sum())}
                       for i, label in enumerate(LABELS)},
    }


def _report(metrics: dict, pairwise: list[dict], trust_meta: dict, trust_path: Path,
            cache_records: dict, output_path: Path) -> str:
    baseline = metrics["EQUAL9"]
    lines = [
        "# November 16 ensemble hypotheses", "",
        "Cache-only equal-probability evaluation on all 2,699 V0 rows. No model training, inference, production ensemble changes, or Kaggle submissions were performed.", "",
        f"**Trust-list source:** `{trust_path.relative_to(ROOT)}` ({trust_meta.get('source')}); SHA-256 `{sha256(trust_path)}`.",
    ]
    if trust_meta.get("source") != "explicit_file":
        lines.append("The named `2026-11-16T18-56_export.csv` was not present locally. Checkpoint IDs were resolved against the repository latest pointer, which pointed to `trust-list-2-v025`; all 12 requested IDs resolved uniquely there. The baseline reproduced the supplied score.")
    lines += [
        "", "The Kaggle accuracy column below is an estimate using V0 balanced accuracy as a proxy; it is not a Kaggle score or a calibrated prediction of leaderboard accuracy. These V0 scores are development diagnostics because the checkpoint trust scores and checkpoint selection use V0.", "",
        "## Results", "",
        "| Experiment | Checkpoints | V0 accuracy | V0 balanced accuracy | Δ BA vs EQUAL9 | Log loss | Corrected | Introduced | Prediction changes vs EQUAL9 | Kaggle accuracy estimate (V0 BA) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in EXPERIMENTS:
        m = metrics[name]
        if name == "EQUAL9":
            delta = corrected = introduced = disagreement = "—"
        else:
            delta = f"{(m['balanced_accuracy']-baseline['balanced_accuracy'])*100:+.3f} pp"
            corrected = str(m["corrected_vs_baseline"])
            introduced = str(m["introduced_vs_baseline"])
            disagreement = f"{m['disagreement_count']} ({m['disagreement_rate']:.2%})"
        lines.append(f"| {name} | {len(EXPERIMENTS[name])} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {delta} | {m['log_loss']:.6f} | {corrected} | {introduced} | {disagreement} | {m['balanced_accuracy']:.4%} |")
    lines += ["", "Equal weights were used in every row: `1/N` per checkpoint.", "", "## Per-label changes", "",
              "Recall and precision changes are percentage points versus EQUAL9.", "",
              "| Model label | EQUAL9 recall / precision | E1 Δ recall / precision | E2 Δ recall / precision | E3 Δ recall / precision |",
              "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        b = baseline["per_label"][label]
        cells = [f"{b['recall']:.1%} / {b['precision']:.1%}"]
        for name in ("E1_replace_ultrabert", "E2_add_diversity", "E3_refresh_neural"):
            v = metrics[name]["per_label"][label]
            cells.append(f"{(v['recall']-b['recall'])*100:+.1f} / {(v['precision']-b['precision'])*100:+.1f} pp")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Prediction similarity", "", "Pairwise disagreement counts compare top-1 labels on identical V0 rows.", "",
              "| Pair | Rows differing | Disagreement |", "|---|---:|---:|"]
    for row in pairwise:
        lines.append(f"| {row['left']} vs {row['right']} | {row['different']} | {row['rate']:.2%} |")
    lines += ["", "## Recommendation", "",
              f"EQUAL9 reproduces the supplied baseline: {baseline['balanced_accuracy']:.4%} balanced accuracy, {baseline['correct']}/2,699 correct."]
    ranked = sorted((name for name in EXPERIMENTS if name != "EQUAL9"),
                    key=lambda name: (-metrics[name]["balanced_accuracy"], name))
    best = ranked[0]
    lines.append(f"Best V0 diagnostic among the three hypotheses: **{best}**, {metrics[best]['balanced_accuracy']:.4%} balanced accuracy ({(metrics[best]['balanced_accuracy']-baseline['balanced_accuracy'])*100:+.3f} pp vs baseline).")
    lines.append(f"The alternatives are prediction-distinct: they change {metrics['E1_replace_ultrabert']['disagreement_rate']:.1%}, {metrics['E2_add_diversity']['disagreement_rate']:.1%}, and {metrics['E3_refresh_neural']['disagreement_rate']:.1%} of rows versus EQUAL9. E1 vs E2 disagree on {pairwise[3]['rate']:.1%}, E1 vs E3 on {pairwise[4]['rate']:.1%}, and E2 vs E3 on {pairwise[5]['rate']:.1%}.")
    lines.append("Recommendation: do not submit these three variants based on this evaluation; all score below EQUAL9 on V0 balanced accuracy (by 0.776, 1.075, and 0.818 pp). Their predictions differ substantially, but did not improve this metric. V0 is not an independent leaderboard estimate, so a Kaggle submission could still test distribution shift, but these results do not favor spending submissions on them.")
    lines += ["", "## Reproducibility", "", f"- Machine-readable configuration, trust mappings and source cache hashes: `{(output_path / 'experiment_configs.json').relative_to(ROOT)}`.",
              f"- Metrics and per-label values: `{(output_path / 'evaluation_metrics.json').relative_to(ROOT)}`.",
              f"- Row-aligned ensemble predictions and probabilities: `{(output_path / 'v0_ensemble_predictions.parquet').relative_to(ROOT)}`.",
              f"- Checkpoint cache inventory: `{(output_path / 'checkpoint_cache_inventory.csv').relative_to(ROOT)}`.",
              "- All predictions were checked for exact ordered V0 IDs, matching true labels, canonical 18-label ordering, finite normalized probabilities, and valid checkpoint paths.", ""]
    return "\n".join(lines)


def run(*, root: str | Path = ROOT, trust_list: str | None = None,
        output_dir: str | Path = OUT, report_path: str | Path = REPORT) -> dict:
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = root / report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    trust_path, trust_meta = _trust_source(root, trust_list)
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    required = {"sequence_id", "experiment_id", "model", "training_dataset", "checkpoint_path"}
    if required - set(trust.columns):
        raise ValueError(f"Trust List 2 missing columns: {sorted(required-set(trust.columns))}")
    id_rows = {}
    for sid in sorted({sid for members in EXPERIMENTS.values() for sid in members}):
        rows = trust.loc[trust.sequence_id.astype(str).eq(sid)]
        if rows.empty:
            raise KeyError(f"checkpoint ID {sid} is absent from {trust_path}")
        unique = rows[["experiment_id", "model", "training_dataset", "checkpoint_path"]].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"checkpoint ID {sid} maps to inconsistent artifacts")
        rec = unique.iloc[0].to_dict()
        checkpoint = Path(rec["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"{sid} checkpoint path is missing: {checkpoint}")
        id_rows[sid] = {**rec, "resolved_checkpoint_path": str(checkpoint)}

    v0 = _dataset_rows("v0", root / "data/processed")
    if len(v0) != 2699 or not set(v0.model.astype(str)).issubset(set(LABELS)):
        raise ValueError(f"unexpected V0 rows or labels: {len(v0)}")
    ids = v0.id.astype(str).tolist()
    y = v0.model.astype(str).to_numpy(dtype=object)
    probabilities, cache_info = {}, {}
    for sid, rec in id_rows.items():
        matrix, cache_path = _read_saved_probabilities(root, rec["experiment_id"], ids, y.tolist())
        if matrix.shape != (2699, 18) or not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"invalid cached probabilities for {sid}")
        probabilities[sid] = matrix
        cache_info[sid] = {**rec, "prediction_path": str(cache_path.relative_to(root)),
                           "prediction_sha256": sha256(cache_path)}

    ensemble_probs, metrics = {}, {}
    for name, members in EXPERIMENTS.items():
        matrix = np.mean(np.stack([probabilities[sid] for sid in members]), axis=0)
        matrix /= matrix.sum(axis=1, keepdims=True)
        ensemble_probs[name] = matrix
        metrics[name] = _metrics(y, matrix)
    base = metrics["EQUAL9"]
    if not np.isclose(base["balanced_accuracy"], EXPECTED_BASELINE_BA, atol=0.00002):
        raise ValueError(f"EQUAL9 failed supplied baseline reproduction: {base['balanced_accuracy']:.8%}")
    base_pred = base["predictions"]
    for name in EXPERIMENTS:
        pred = metrics[name]["predictions"]
        if name == "EQUAL9":
            metrics[name].update({"delta_accuracy_pp_vs_baseline": 0.0,
                "delta_balanced_accuracy_pp_vs_baseline": 0.0,"corrected_vs_baseline":0,
                "introduced_vs_baseline":0,"disagreement_count":0,"disagreement_rate":0.0})
        else:
            old_correct = base_pred == y
            new_correct = pred == y
            metrics[name].update({
                "delta_accuracy_pp_vs_baseline": (metrics[name]["accuracy"]-base["accuracy"])*100,
                "delta_balanced_accuracy_pp_vs_baseline": (metrics[name]["balanced_accuracy"]-base["balanced_accuracy"])*100,
                "corrected_vs_baseline": int((~old_correct & new_correct).sum()),
                "introduced_vs_baseline": int((old_correct & ~new_correct).sum()),
                "disagreement_count": int((pred != base_pred).sum()),
                "disagreement_rate": float((pred != base_pred).mean()),
            })
        # Omit arrays from the machine-readable metrics table; save them in Parquet.
        metrics[name]["predictions"] = pred

    pairwise=[]
    names=list(EXPERIMENTS)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            a=metrics[left]["predictions"]; b=metrics[right]["predictions"]
            diff=int((a!=b).sum()); pairwise.append({"left":left,"right":right,"different":diff,"rate":diff/len(y)})
    per_label_delta={}
    for label in LABELS:
        b=base["per_label"][label]
        per_label_delta[label]={name:{"precision":metrics[name]["per_label"][label]["precision"],
                                      "recall":metrics[name]["per_label"][label]["recall"],
                                      "precision_delta_pp":(metrics[name]["per_label"][label]["precision"]-b["precision"])*100,
                                      "recall_delta_pp":(metrics[name]["per_label"][label]["recall"]-b["recall"])*100,
                                      "correct":metrics[name]["per_label"][label]["correct"]}
                                  for name in EXPERIMENTS}

    safe_metrics={name:{k:v for k,v in m.items() if k!="predictions"} for name,m in metrics.items()}
    config={"dataset":"v0","validation_rows":len(v0),"class_order":list(LABELS),
        "trust_list_path":str(trust_path.relative_to(root)),"trust_list_sha256":sha256(trust_path),
        "trust_source_metadata":trust_meta,"checkpoint_ids":{sid:rec for sid,rec in id_rows.items()},
        "prediction_cache_sources":cache_info,"experiments":{name:{"checkpoint_ids":members,
            "weights":{sid:1/len(members) for sid in members},"strategy":"arithmetic mean of 18-class probabilities"}
            for name,members in EXPERIMENTS.items()},
        "validation_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "method":"saved full-V0 probabilities only; no inference or training"}
    (output_dir/"experiment_configs.json").write_text(json.dumps(config,indent=2,default=str)+"\n",encoding="utf-8")
    (output_dir/"evaluation_metrics.json").write_text(json.dumps({"experiments":safe_metrics,"per_label":per_label_delta,
        "pairwise_prediction_disagreement":pairwise,"baseline_expected_balanced_accuracy":EXPECTED_BASELINE_BA},indent=2,default=str)+"\n",encoding="utf-8")
    pd.DataFrame([{"sequence_id":sid,**rec} for sid,rec in cache_info.items()]).to_csv(output_dir/"checkpoint_cache_inventory.csv",index=False)
    columns={"id":ids,"prompt_id":v0.prompt_id.astype(str).tolist(),"y_true":y.astype(str)}
    for name,matrix in ensemble_probs.items():
        columns[f"pred::{name}"]=metrics[name]["predictions"].astype(str)
        columns[f"correct::{name}"]=metrics[name]["predictions"]==y
        for i,label in enumerate(LABELS): columns[f"prob::{name}::{label}"]=matrix[:,i]
    pd.DataFrame(columns).to_parquet(output_dir/"v0_ensemble_predictions.parquet",index=False)
    report=_report(metrics,pairwise,trust_meta,trust_path,cache_info,output_dir)
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(report,encoding="utf-8")
    summary={name:{k:safe_metrics[name][k] for k in ("accuracy","balanced_accuracy","log_loss","correct","delta_balanced_accuracy_pp_vs_baseline","corrected_vs_baseline","introduced_vs_baseline","disagreement_count","disagreement_rate")} for name in EXPERIMENTS}
    print(json.dumps({"report":str(report_path),"output_dir":str(output_dir),"results":summary,
                      "trust_list":str(trust_path),"trust_list_sha256":config["trust_list_sha256"]},indent=2))
    return {"metrics":safe_metrics,"per_label":per_label_delta,"pairwise":pairwise,"config":config}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-list",help="Trust List 2 CSV; defaults to repository latest pointer")
    parser.add_argument("--output-dir",default=str(OUT))
    parser.add_argument("--report",default=str(REPORT))
    args=parser.parse_args()
    run(trust_list=args.trust_list,output_dir=args.output_dir,report_path=args.report)


if __name__ == "__main__":
    main()
