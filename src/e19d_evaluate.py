"""Evaluate E19-D grouping and compare it with the E19-B Qwen3.5-2B baseline."""
from __future__ import annotations

import json
from pathlib import Path

from . import e19b_evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e19d"
BASELINE = ROOT / "artifacts/analysis/e19b/metrics.json"
DOC = ROOT / "docs/e19d.md"


def _format_value(key, metric):
    if key in {"exact_group_recovery", "response_exact_grouping", "pairwise_precision",
               "pairwise_recall", "pairwise_f1", "cluster_purity", "bcubed_precision",
               "bcubed_recall", "bcubed_f1"}:
        return f"{100*metric[key]:.2f}%"
    if key == "exact_groups":
        return f"{metric[key]}/{metric['true_group_count']}"
    if key == "exact_responses":
        return f"{metric[key]}/{metric['rows']}"
    return str(metric[key])


def _record_runtime_config(out):
    import importlib.metadata
    import torch
    config_path=Path(out)/"configuration.json"
    config=json.loads(config_path.read_text())
    config["runtime"]={
        "torch":torch.__version__,
        "cuda_runtime":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers":importlib.metadata.version("transformers"),
        "flash_linear_attention":importlib.metadata.version("flash-linear-attention"),
        "linear_attention_kernel":"flash-linear-attention FLA 0.6.0",
    }
    config_path.write_text(json.dumps(config,indent=2))
    return config


def _comparison_report(e19d, e19b):
    keys = [
        ("exact_group_recovery", "Exact original-group recovery"),
        ("exact_groups", "Exact groups recovered"),
        ("response_exact_grouping", "Response-level exact grouping"),
        ("exact_responses", "Responses grouped exactly"),
        ("pairwise_precision", "Pairwise precision"),
        ("pairwise_recall", "Pairwise recall"),
        ("pairwise_f1", "Pairwise F1"),
        ("cluster_purity", "Cluster purity"),
        ("bcubed_precision", "B-cubed precision"),
        ("bcubed_recall", "B-cubed recall"),
        ("bcubed_f1", "B-cubed F1"),
        ("predicted_group_count", "Reconstructed group count"),
        ("incorrectly_merged_groups", "Incorrectly merged groups"),
        ("split_original_groups", "Split original groups"),
        ("singleton_responses", "Singleton responses"),
    ]
    lines = ["# E19-D — Qwen3.5-4B Prompt Reconstruction Comparison", "",
        "E19-D repeats E19-B with Qwen3.5-4B prompt reconstruction. The grouping algorithm, semantic embedding model, saved E13-B outer prompt-group folds, inner fitting/tuning, thresholds, and pairwise metrics are unchanged.", "",
        "## Aggregate out-of-fold comparison", "",
        "| Metric | E19-B (Qwen3.5-2B) | E19-D (Qwen3.5-4B) | Δ (4B − 2B) |",
        "|---|---:|---:|---:|"]
    a, b = e19b["aggregate_oof"], e19d["aggregate_oof"]
    for key, label in keys:
        av, bv = _format_value(key, a), _format_value(key, b)
        delta = (f"{100*(b[key]-a[key]):+.2f} pp" if key in {
            "exact_group_recovery", "response_exact_grouping", "pairwise_precision",
            "pairwise_recall", "pairwise_f1", "cluster_purity", "bcubed_precision",
            "bcubed_recall", "bcubed_f1"} else f"{b[key]-a[key]:+d}")
        lines.append(f"| {label} | {av} | {bv} | {delta} |")
    lines += ["", "## Outer-fold comparison", "",
        "| Fold | Rows | E19-B pair P / R / F1 | E19-D pair P / R / F1 | E19-B exact groups | E19-D exact groups |",
        "|---:|---:|---:|---:|---:|---:|"]
    for fa, fb in zip(e19b["outer_folds"], e19d["outer_folds"]):
        assert fa["fold"] == fb["fold"]
        ma, mb = fa["metrics"], fb["metrics"]
        fmt = lambda m: f"{m['pairwise_precision']:.4f} / {m['pairwise_recall']:.4f} / {m['pairwise_f1']:.4f}"
        lines.append(f"| {fa['fold']} | {mb['rows']} | {fmt(ma)} | {fmt(mb)} | "
                     f"{ma['exact_groups']}/{ma['true_group_count']} | {mb['exact_groups']}/{mb['true_group_count']} |")
    runtime=e19d["configuration"].get("runtime",{})
    lines += ["", "## E19-D protocol and details", "",
        f"Model: `{e19d['semantic_model']['model']}` is used for grouping embeddings; reconstruction model: `{e19d['configuration']['reconstruction_model']}` at revision `{e19d['configuration']['reconstruction_revision']}`. Reconstruction used batch size {e19d['configuration']['reconstruction_settings']['batch_size']} with dynamic batch splitting on CUDA OOM. Runtime: Transformers {runtime.get('transformers')}, Torch {runtime.get('torch')} (CUDA {runtime.get('cuda_runtime')}), GPU `{runtime.get('gpu')}`, FLA {runtime.get('flash_linear_attention')}.", "",
        f"All {e19d['rows']} V0 rows were held out exactly once under E19-B's saved five prompt-group folds. E19-D's aggregate pairwise precision is {100*b['pairwise_precision']:.2f}% and recall is {100*b['pairwise_recall']:.2f}%. The all-V0 fitted clustering is diagnostic only and is not included in this comparison.", "",
        "Detailed fold metrics, all-V0 diagnostic, B-cubed metrics, group-size distributions, selected thresholds, and diagnostic examples are in `artifacts/analysis/e19d/`. E19-B baseline values are loaded from its saved OOF metrics. No base classifiers were trained.", "",
        "## E19-D detailed report", ""]
    return "\n".join(lines)


def run(root=ROOT):
    root = Path(root)
    out = root / "artifacts/analysis/e19d"
    out.mkdir(parents=True, exist_ok=True)
    current = e19b_evaluate.run(root,
        reconstruct_path=out / "reconstructed_prompts.parquet",
        embeddings_path=out / "semantic_embeddings.npy",
        output_dir=out, experiment="E19-D", docs_path=None)
    baseline_path = root / "artifacts/analysis/e19b/metrics.json"
    baseline = json.loads(baseline_path.read_text())
    write_report(root, current=current, baseline=baseline)
    return current


def write_report(root=ROOT, *, current=None, baseline=None):
    root = Path(root)
    out = root / "artifacts/analysis/e19d"
    if current is None:
        current = json.loads((out / "metrics.json").read_text())
    if baseline is None:
        baseline = json.loads((root / "artifacts/analysis/e19b/metrics.json").read_text())
    config=_record_runtime_config(out)
    current = dict(current)
    current["configuration"] = config
    comparison = {"baseline_experiment": "E19-B Qwen3.5-2B",
        "experiment": "E19-D Qwen3.5-4B", "baseline_metrics": baseline,
        "e19d_metrics": current}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    detail_path = out / "e19d_detail.md"
    detail_metrics=json.loads((out/"metrics.json").read_text())
    predictions=__import__("pandas").read_parquet(out/"oof_group_predictions.parquet")
    pair_df=__import__("pandas").read_parquet(out/"oof_pair_diagnostics.parquet")
    e19b_evaluate._write_report(detail_metrics,predictions,pair_df,out/"diagnostics.json",
                                detail_path,experiment="E19-D")
    detail = detail_path.read_text()
    report = _comparison_report(current, baseline) + "\n\n" + detail
    detail_path.write_text(report)
    (root / "docs/e19d.md").write_text(report)
    return report


if __name__ == "__main__":
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--thresholds", help="comma-separated pair-score thresholds")
    parser.add_argument("--report-only", action="store_true",
                        help="render comparisons from already saved E19-B and E19-D metrics")
    args = parser.parse_args()
    if args.report_only:
        report=write_report(args.root)
        print(report.split("## E19-D protocol and details",1)[0])
        raise SystemExit(0)
    thresholds = e19b_evaluate.THRESHOLDS if not args.thresholds else tuple(
        float(x) for x in args.thresholds.split(","))
    # Run directly with shared fold-local E19-B evaluation settings, then add
    # the paired baseline comparison and report.
    OUT.mkdir(parents=True, exist_ok=True)
    result = e19b_evaluate.run(args.root,
        reconstruct_path=OUT / "reconstructed_prompts.parquet",
        embeddings_path=OUT / "semantic_embeddings.npy", output_dir=OUT,
        threshold_grid=thresholds, experiment="E19-D", docs_path=None)
    baseline = json.loads((args.root / "artifacts/analysis/e19b/metrics.json").read_text())
    report=write_report(args.root,current=result,baseline=baseline)
    print(json.dumps(result["aggregate_oof"],indent=2))
