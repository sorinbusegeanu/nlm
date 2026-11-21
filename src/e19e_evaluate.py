"""Write a three-way, same-fold E19-B/D/E prompt-group comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    ("exact_group_recovery", "Exact original-group recovery", "pct"),
    ("exact_groups", "Exact groups recovered", "count"),
    ("response_exact_grouping", "Response-level exact grouping", "pct"),
    ("exact_responses", "Responses grouped exactly", "count"),
    ("pairwise_precision", "Pairwise precision", "pct"),
    ("pairwise_recall", "Pairwise recall", "pct"),
    ("pairwise_f1", "Pairwise F1", "pct"),
    ("cluster_purity", "Cluster purity", "pct"),
    ("bcubed_precision", "B-cubed precision", "pct"),
    ("bcubed_recall", "B-cubed recall", "pct"),
    ("bcubed_f1", "B-cubed F1", "pct"),
    ("predicted_group_count", "Reconstructed group count", "int"),
    ("incorrectly_merged_groups", "Incorrectly merged groups", "int"),
    ("split_original_groups", "Split original groups", "int"),
    ("singleton_responses", "Singleton responses", "int"),
]


def _metrics(root: Path, experiment: str):
    path = root / f"artifacts/analysis/{experiment.lower()}/metrics.json"
    data = json.loads(path.read_text())
    if int(data.get("rows", -1)) != 2699:
        raise ValueError(f"{experiment} does not contain all 2,699 OOF rows")
    metric = data["aggregate_oof"]
    if metric["rows"] != 2699 or metric["true_group_count"] != 150:
        raise ValueError(f"{experiment} grouping metrics use a different V0 population")
    return data


def render(root=ROOT):
    root = Path(root)
    data = {name: _metrics(root, name) for name in ("e19b", "e19d", "e19e")}
    names = ("e19b", "e19d", "e19e")
    labels = {"e19b": "E19-B · Qwen3.5-2B", "e19d": "E19-D · Qwen3.5-4B",
              "e19e": "E19-E · Qwen3.5-9B Q4_K_M"}
    lines = ["# E19-E — Qwen3.5-9B Prompt Reconstruction Comparison", "",
        "All three experiments are scored on the same 2,699 V0 responses, 150 true prompt groups, and the same five saved E13-B prompt-group outer folds. Grouping uses the same E19-B pipeline, fold-local training and threshold selection, and EmbeddingGemma2 semantic features. The model reconstructs each response independently; prompt IDs are used only for fold construction and scoring.", "",
        "## Aggregate out-of-fold results", "",
        "| Metric | E19-B · Qwen3.5-2B | E19-D · Qwen3.5-4B | E19-E · Qwen3.5-9B Q4_K_M |",
        "|---|---:|---:|---:|"]
    for key, label, kind in METRICS:
        cells = []
        for name in names:
            m = data[name]["aggregate_oof"]
            if kind == "pct":
                value = f"{100*m[key]:.2f}%"
            elif kind == "count":
                denom = m["true_group_count"] if key == "exact_groups" else m["rows"]
                value = f"{m[key]}/{denom}"
            else:
                value = str(m[key])
            cells.append(value)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Outer folds", "",
        "| Fold | Rows | E19-B pair P / R / F1 | E19-D pair P / R / F1 | E19-E pair P / R / F1 | Exact groups E19-B / E19-D / E19-E |",
        "|---:|---:|---:|---:|---:|---:|"]
    fold_sets = [data[n]["outer_folds"] for n in names]
    for fb, fd, fe in zip(*fold_sets):
        assert fb["fold"] == fd["fold"] == fe["fold"]
        fmt = lambda fold: " / ".join(f"{fold['metrics'][k]:.4f}" for k in
                                      ("pairwise_precision", "pairwise_recall", "pairwise_f1"))
        exact = " / ".join(f"{f['metrics']['exact_groups']}/{f['metrics']['true_group_count']}"
                           for f in (fb, fd, fe))
        lines.append(f"| {fb['fold']} | {fb['metrics']['rows']} | {fmt(fb)} | {fmt(fd)} | {fmt(fe)} | {exact} |")
    e19e_config = json.loads((root / "artifacts/analysis/e19e/configuration.json").read_text())
    reconstruct = json.loads((root / "artifacts/analysis/e19e/reconstruction_manifest.json").read_text())
    lines += ["", "## E19-E run details", "",
        f"- Model: `{reconstruct['model_id']}` revision `{reconstruct['model_revision']}`.",
        f"- Quantization: Q4_K_M GGUF; SHA256 `{reconstruct['model_revision'].split(':')[-1]}`.",
        f"- Generation: deterministic (`temperature=0`, seed {reconstruct['seed']}), non-thinking mode.",
        f"- Requested batch ladder: `{reconstruct['batch_ladder']}`; effective parallelism: `{[x['parallelism'] for x in reconstruct['effective_parallelism_history'] if x.get('status') == 'started']}`.",
        f"- Reconstruction coverage: {reconstruct['successes']}/{reconstruct['rows']} successful, {reconstruct['failures']} failed.",
        f"- Shared semantic model: `{e19e_config['semantic_model']}` at `{e19e_config['semantic_revision']}`.",
        "- Evaluation: the existing five E13-B outer prompt-group folds with inner, training-only scorer/threshold selection. The all-V0 grouping is a separate diagnostic and is excluded from this comparison.", "",
        "E19-E is slightly below E19-D on aggregate exact recovery and pairwise F1, while remaining above E19-B on pairwise F1. All three runs achieved 100% pairwise precision; remaining errors are split groups rather than merges.", "",
        "Detailed OOF assignments, per-fold metrics, full-V0 diagnostic, and examples are in `artifacts/analysis/e19e/metrics.json`, `oof_group_predictions.parquet`, `oof_pair_diagnostics.parquet`, and `diagnostic_examples.json`. Reconstruction cache and server logs are also retained in that directory."]
    report = "\n".join(lines) + "\n"
    out = root / "artifacts/analysis/e19e"
    (out / "comparison.json").write_text(json.dumps({
        "experiments": {n: data[n] for n in names},
        "metrics_table": [{"metric": label, **{n: data[n]["aggregate_oof"][key] for n in names}}
                          for key, label, _ in METRICS]}, indent=2))
    (out / "e19e_comparison.md").write_text(report)
    (root / "docs/e19e.md").write_text(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(render(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
