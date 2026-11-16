#!/usr/bin/env python3
"""Reproduce the V1/V2 UltraBERT degradation audit and score V2 on an independent holdout.

This is read-only with respect to model/data inputs. It writes reports and tables
only under artifacts/analysis/ultrabert_v2_degradation and docs/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS = {
    "V1": "ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0",
    "V2.0": "ultrabert380_v2.0_valv0_datad12bb8be_split870e47f9_seed2026_cfgfe74ea98_codeab9f6d2",
    "V2.1": "ultrabert380_v2.1_valv0_datac59bc6ca_split6330072d_seed2026_cfgaf9f028b_codeab9f6d2",
}
CHECKPOINTS = {
    version: Path("artifacts/runs") / exp / "checkpoints/ultrabert380_foldvalidation"
    for version, exp in EXPERIMENTS.items() if version != "V1"
}
LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
V1_V0_PREDICTIONS = Path("artifacts/analysis/trust_list_2_v0") / f"{EXPERIMENTS['V1']}.parquet"


def norm_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def sha256_ids(values) -> str:
    return hashlib.sha256("\n".join(sorted(pd.Series(values).astype(str))).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_rows(frame: pd.DataFrame, labels: list[str], *, version: str,
                prediction_column: str = "y_pred") -> list[dict]:
    truth = frame.y_true.astype(str)
    pred = frame[prediction_column].astype(str)
    rows = []
    for label in labels:
        actual = truth.eq(label)
        guessed = pred.eq(label)
        tp = int((actual & guessed).sum())
        support, predicted = int(actual.sum()), int(guessed.sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else (0.0 if support else np.nan))
        rows.append({"version": version, "label": label, "support": support,
                     "predicted_count": predicted, "true_positive": tp,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def score_summary(frame: pd.DataFrame, labels: list[str], *, version: str,
                  prediction_column: str = "y_pred") -> dict:
    truth = frame.y_true.astype(str)
    pred = frame[prediction_column].astype(str)
    class_recalls = [float((pred[truth.eq(label)] == label).mean()) for label in labels]
    return {"version": version, "rows": len(frame), "accuracy": float((truth == pred).mean()),
            "balanced_accuracy": float(np.mean(class_recalls)),
            "macro_f1": float(np.mean([row["f1"] for row in metric_rows(
                frame, labels, version=version, prediction_column=prediction_column)]))}


def token_length_stats(tokenizer, frame: pd.DataFrame, name: str, out_dir: Path,
                       batch_size: int = 256) -> tuple[pd.DataFrame, list[dict]]:
    texts = frame.text.fillna("").astype(str).tolist()
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], truncation=False,
                            padding=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    if len(lengths) != len(frame):
        raise ValueError(f"tokenizer returned {len(lengths)} lengths for {name} ({len(frame)} rows)")
    scored = frame[["id", "model"] + (["source_type"] if "source_type" in frame else [])].copy()
    scored["tokens"] = lengths
    out = []
    groups = [("all", scored)]
    if "source_type" in scored:
        groups.extend((str(key), part) for key, part in scored.groupby("source_type", dropna=False))
    for group_name, part in groups:
        series = part.tokens
        out.append({"dataset": name, "source_type": group_name, "rows": len(part),
                    "p50_tokens": float(series.quantile(.50)),
                    "p90_tokens": float(series.quantile(.90)),
                    "p95_tokens": float(series.quantile(.95)),
                    "p99_tokens": float(series.quantile(.99)),
                    "max_tokens": int(series.max()),
                    **{f"rows_gt_{limit}_tokens": int((series > limit).sum())
                       for limit in (384, 512, 1024, 1500)}})
    scored["dataset"] = name
    scored.to_parquet(out_dir / f"token_lengths_{name.replace('.', '_')}.parquet", index=False)
    return scored, out


def infer_checkpoint(version: str, texts: list[str], checkpoint: Path,
                     max_length: int, batch_size: int = 4) -> tuple[list[str], dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the planned inference-only holdout scoring")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True).to("cuda").eval()
    labels_by_id = model.config.id2label
    labels = [str(labels_by_id[str(i)] if str(i) in labels_by_id else labels_by_id[i])
              for i in range(len(labels_by_id))]
    max_length = int(max_length)
    predictions: list[str] = []
    active_batch = max(1, batch_size)
    fallbacks = 0
    started = time.monotonic()
    cursor = 0
    with torch.inference_mode():
        while cursor < len(texts):
            batch = texts[cursor:cursor + active_batch]
            try:
                tokens = tokenizer(batch, truncation=True, max_length=max_length,
                                   padding=True, return_tensors="pt")
                tokens = {key: value.to("cuda") for key, value in tokens.items()}
                logits = model(**tokens).logits
                ids = logits.argmax(dim=-1).cpu().tolist()
                predictions.extend(labels[index] for index in ids)
                cursor += len(batch)
            except torch.cuda.OutOfMemoryError:
                if active_batch == 1:
                    raise
                del tokens
                torch.cuda.empty_cache()
                active_batch = max(1, active_batch // 2)
                fallbacks += 1
    elapsed = time.monotonic() - started
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, {"elapsed_seconds": elapsed, "max_length": max_length,
                         "final_batch_size": active_batch, "oom_fallbacks": fallbacks}


def read_predictions(version: str, v0_ids: set[str], v0_labels: dict[str, str]) -> pd.DataFrame:
    if version == "V1":
        frame = pd.read_parquet(V1_V0_PREDICTIONS)
        frame = frame[["id", "y_true", "y_pred"]].copy()
    else:
        path = Path("artifacts/predictions") / f"{EXPERIMENTS[version]}_validation.parquet"
        saved = pd.read_parquet(path)
        score_columns = [f"score::{label}" for label in LABELS]
        if not set(score_columns).issubset(saved.columns):
            raise ValueError(f"{path} does not contain the expected V0 label scores")
        frame = saved[["id", "y_true", *score_columns]].copy()
        scores = frame[score_columns].to_numpy(dtype=np.float64)
        frame["y_pred"] = np.asarray(LABELS)[scores.argmax(axis=1)]
        frame = frame[["id", "y_true", "y_pred"]]
    if set(frame.id.astype(str)) != v0_ids or len(frame) != len(v0_ids):
        raise ValueError(f"{version} cached predictions do not exactly cover the V0 rows")
    if not frame.apply(lambda row: v0_labels.get(str(row.id)) == str(row.y_true), axis=1).all():
        raise ValueError(f"{version} cached truth labels do not align with the current V0 artifact")
    return frame


def build_overlap(datasets: dict[str, pd.DataFrame], v0: pd.DataFrame,
                  holdout: pd.DataFrame) -> dict:
    norm_sets = {name: set(frame.text.fillna("").astype(str).map(norm_text)) - {""}
                 for name, frame in datasets.items()}
    prompt_sets = {name: set(frame.prompt_id.fillna("").astype(str)) - {""}
                   for name, frame in datasets.items()}
    norm_v0 = set(v0.text.fillna("").astype(str).map(norm_text)) - {""}
    prompt_v0 = set(v0.prompt_id.fillna("").astype(str)) - {""}
    norm_holdout = set(holdout.text.fillna("").astype(str).map(norm_text)) - {""}
    prompt_holdout = set(holdout.prompt_id.fillna("").astype(str)) - {""}
    def label_map(frame: pd.DataFrame) -> dict[str, set[str]]:
        part = pd.DataFrame({"text": frame.text.fillna("").astype(str).map(norm_text),
                             "model": frame.model.astype(str)})
        return part[part.text.ne("")].groupby("text").model.agg(lambda values: set(values)).to_dict()
    label_maps = {name: label_map(frame) for name, frame in datasets.items()}
    result = {"normalization": "NFKC + casefold + collapsed whitespace",
              "dataset_rows": {name: len(frame) for name, frame in datasets.items()},
              "pairwise": {}, "versus_v0": {}, "versus_v1_5_holdout": {}}
    for left, right in (("V1", "V2.0"), ("V1", "V2.1"), ("V2.0", "V2.1")):
        result["pairwise"][f"{left}_vs_{right}"] = {
            "shared_normalized_texts": len(norm_sets[left] & norm_sets[right]),
            "shared_prompt_ids": len(prompt_sets[left] & prompt_sets[right]),
            "same_label_shared_texts": sum(1 for text in norm_sets[left] & norm_sets[right]
                                            if label_maps[left][text] & label_maps[right][text]),
            "cross_label_conflict_groups": sum(1 for text in norm_sets[left] & norm_sets[right]
                                                if any(a != b for a in label_maps[left][text]
                                                       for b in label_maps[right][text])),
        }
    for name in ("V2.0", "V2.1"):
        result["versus_v0"][name] = {
            "shared_normalized_texts": len(norm_sets[name] & norm_v0),
            "shared_prompt_ids": len(prompt_sets[name] & prompt_v0),
        }
        result["versus_v1_5_holdout"][name] = {
            "shared_normalized_texts": len(norm_sets[name] & norm_holdout),
            "shared_prompt_ids": len(prompt_sets[name] & prompt_holdout),
        }
    return result


def provider_conflicts(datasets: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Flag only explicit, canonical-provider contradictions; unknown proxy rows stay unclassified."""
    expected = {"claude-": "anthropic", "gemini-": "google", "glm-": "zai",
                "gpt-": "openai", "qwen": "alibaba", "kimi": "moonshot"}
    aliases = {"anthropic": ("anthropic",), "google": ("google",),
               "zai": ("zai", "z.ai", "zhipu"), "openai": ("openai",),
               "alibaba": ("alibaba",), "moonshot": ("moonshot",)}
    rows = []
    for name, frame in datasets.items():
        if "provider" not in frame:
            continue
        for item in frame[["id", "model", "provider", "source_dataset"]].dropna(subset=["provider"]).itertuples(index=False):
            label = str(item.model).casefold()
            expected_provider = next((provider for prefix, provider in expected.items()
                                      if label.startswith(prefix)), None)
            provider = str(item.provider).casefold()
            if expected_provider and provider.strip() not in {"", "n/s"} and \
                    "kaggle model proxy" not in provider and \
                    not any(alias in provider for alias in aliases[expected_provider]):
                rows.append({"dataset": name, "id": item.id, "label": item.model,
                             "provider": item.provider, "expected_provider_family": expected_provider,
                             "source_dataset": item.source_dataset,
                             "audit_status": "candidate contradiction; requires source-level review"})
    result = pd.DataFrame(rows, columns=["dataset", "id", "label", "provider",
        "expected_provider_family", "source_dataset", "audit_status"])
    result.to_csv(out_dir / "provider_conflict_candidates.csv", index=False)
    return result


def data_audits(datasets: dict[str, pd.DataFrame], out_dir: Path,
                profiles: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    class_rows, source_rows, evidence_rows, duplicate_rows, diversity_rows = [], [], [], [], []
    evidence_fields = ["prompt_evidence", "panel_labels", "match_weight", "provider",
                       "model_version", "source_model_attribution", "route"]
    details = {}
    for name, frame in datasets.items():
        class_counts = frame.groupby("model", dropna=False).size()
        for label, count in class_counts.items():
            class_rows.append({"dataset": name, "model": label, "rows": int(count),
                               "share": float(count / len(frame))})
        if "source_dataset" in frame:
            source_frame = frame[["model", "source_dataset"]].copy()
            source_frame["source_dataset"] = source_frame.source_dataset.fillna("").astype(str)
            source_frame = source_frame[source_frame.source_dataset.str.strip().ne("")]
            for label, label_rows in source_frame.groupby("model"):
                counts = label_rows.source_dataset.value_counts()
                shares = counts / counts.sum()
                diversity_rows.append({"dataset": name, "model": label,
                    "rows_with_source": int(counts.sum()), "unique_sources": int(counts.size),
                    "top_source": counts.index[0], "top_source_rows": int(counts.iloc[0]),
                    "top_source_share": float(shares.iloc[0]),
                    "top3_source_share": float(shares.iloc[:3].sum()),
                    "source_hhi": float((shares ** 2).sum())})
        family = (frame.source_type.fillna("<missing>").astype(str)
                  if "source_type" in frame else frame.source_dataset.fillna("<missing>").astype(str))
        grouped = frame.assign(_source_family=family).groupby("_source_family", dropna=False)
        for source_type, rows in grouped:
            record = {"dataset": name, "source_type": source_type, "rows": len(rows),
                      "share": len(rows) / len(frame)}
            record.update({f"has_{field}_rows": int(rows[field].notna().sum()) if field in rows else 0
                           for field in evidence_fields})
            source_rows.append(record)
            for field in evidence_fields:
                if field in rows:
                    values = rows[field].notna()
                    if pd.api.types.is_object_dtype(rows[field]) or pd.api.types.is_string_dtype(rows[field]):
                        values &= rows[field].astype(str).str.strip().ne("")
                    evidence_rows.append({"dataset": name, "source_type": source_type,
                                          "field": field, "present_rows": int(values.sum()),
                                          "rows": len(rows), "coverage": float(values.mean())})
        normalized = frame.text.fillna("").astype(str).map(norm_text)
        duplicate_groups = pd.DataFrame({"normalized": normalized, "model": frame.model.astype(str)})
        duplicate_groups = duplicate_groups[duplicate_groups.normalized.ne("")].groupby("normalized").agg(
            rows=("model", "size"), labels=("model", "nunique"))
        duplicate_rows.append({"dataset": name, "normalized_duplicate_groups": int((duplicate_groups.rows > 1).sum()),
                               "cross_label_conflict_groups": int((duplicate_groups.labels > 1).sum()),
                               "extra_duplicate_rows": int((duplicate_groups.rows - 1).clip(lower=0).sum())})
        details[name] = {"rows": len(frame), "labels": int(frame.model.nunique()),
                         "min_class_count": int(class_counts.min()),
                         "max_class_count": int(class_counts.max()),
                         "profile_rows": profiles[name].get("rows"),
                         "profile_dataset_sha256": profiles[name].get("dataset_sha256"),
                         "source_types": (frame.source_type.value_counts(dropna=False).to_dict()
                                          if "source_type" in frame else None),
                         "selected_rows_already_in_v1": profiles[name].get("selected_rows_already_in_v1"),
                         "unique_source_datasets": int(frame.source_dataset.nunique(dropna=True)) if "source_dataset" in frame else None,
                         "evidence_coverage": {field: float(frame[field].notna().mean())
                                               for field in evidence_fields if field in frame},
                         "unverified_gemini37_included": profiles[name].get("unverified_gemini37_included_by_user_request", False),
                         "attribution_caveat": profiles[name].get("attribution_caveat")}
    pd.DataFrame(class_rows).to_csv(out_dir / "class_counts.csv", index=False)
    pd.DataFrame(source_rows).to_csv(out_dir / "source_composition_and_evidence.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(out_dir / "evidence_field_coverage.csv", index=False)
    pd.DataFrame(diversity_rows).to_csv(out_dir / "source_diversity_by_label.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(out_dir / "duplicate_audit.csv", index=False)
    return details, pd.DataFrame(duplicate_rows)


def write_report(root: Path, out_dir: Path, report_path: Path, summary: dict,
                 v0_metrics: pd.DataFrame, holdout_metrics: pd.DataFrame,
                 token_stats: pd.DataFrame, dataset_details: dict,
                 overlap: dict, duplicates: pd.DataFrame, curves: pd.DataFrame,
                 provider_conflict_count: int) -> None:
    example_labels = ["gpt-6-astra", "gemini-3.1-pro-preview", "claude-opus-4-6", "qwen3.8-max"]
    selected = v0_metrics[v0_metrics.label.isin(example_labels)]
    precision_table = selected.pivot(index="label", columns="version", values="precision").reindex(example_labels)
    recall_table = selected.pivot(index="label", columns="version", values="recall").reindex(example_labels)
    overall = pd.DataFrame(summary["v0_overall"])
    lines = [
        "# UltraBERT V2.0/V2.1 degradation investigation", "",
        "## Executive summary", "",
        "The V1/V2 gap is reproduced on the same 2,699 V0 rows. The supplied per-label values are precision, not recall. All three source datasets are exactly class-balanced, so simple class imbalance is ruled out.",
        "", "The strongest shared configuration lead is sequence length: V1 recorded 384 tokens, while V2.0 and V2.1 automatically selected 1,595 from the longest combined train/validation response. V2's memory probe then reduced its training microbatch from 8 to 4 (with gradient accumulation 2). V2 validation loss and accuracy peak at epoch 2 and deteriorate afterward while train loss continues falling, consistent with overfitting or unstable optimization; this does not by itself isolate the cause.",
        "", "V2.1 includes 2,135 unverified Gemini 3.7 rows (71.2% of that class), but V2.0 without those rows degrades similarly. That addition is a specific V2.1 risk, not an explanation for the common V2 drop. The old V1 manifest also points to a 51,497-row data snapshot whose hash is not present in the current V1 or V1.5 profiles, limiting exact data-construction attribution.",
        "", "## Common V0 comparison", "",
        "Evaluation uses the same V0 IDs and labels for all cached checkpoints. V2 epoch selection used the full V0 set; its metrics are therefore selection-contaminated and should not be read as an unbiased final estimate.", "",
        overall.to_markdown(index=False, floatfmt=".4f"), "",
        "| Label | V1 precision | V2.0 precision | V2.1 precision | V1 recall | V2.0 recall | V2.1 recall |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in example_labels:
        p = precision_table.loc[label]
        r = recall_table.loc[label]
        lines.append(f"| {label} | {p['V1']:.1%} | {p['V2.0']:.1%} | {p['V2.1']:.1%} | {r['V1']:.1%} | {r['V2.0']:.1%} | {r['V2.1']:.1%} |")
    lines.extend(["", "Full per-label precision/recall/support/predicted-count metrics: [`v0_per_label_metrics.csv`](../artifacts/analysis/ultrabert_v2_degradation/v0_per_label_metrics.csv). Confusion matrices and error-pair summaries are in the same analysis directory.",
        "", "## Data construction and label evidence", "",
        "Class counts are exact and uniform:", ""])
    for name, details in dataset_details.items():
        lines.append(f"- **{name}:** {details['rows']:,} rows, {details['labels']} labels, {details['min_class_count']:,}–{details['max_class_count']:,} rows/class; profile SHA256 `{details['profile_dataset_sha256']}`.")
    evidence_rows = []
    for name, details in dataset_details.items():
        for field, coverage in details.get("evidence_coverage", {}).items():
            evidence_rows.append({"dataset": name, "field": field, "coverage": coverage})
    if evidence_rows:
        evidence_summary = pd.DataFrame(evidence_rows).pivot(index="field", columns="dataset", values="coverage")
        evidence_display = evidence_summary.map(lambda value: f"{value:.1%}" if pd.notna(value) else "n/a")
        lines.extend(["", "Evidence field coverage (non-missing rows; source-specific breakdown is in the CSV):", "",
                      evidence_display.to_markdown(), ""])
    lines.extend(["", f"Normalized text overlap is measured using NFKC, casefolding, and collapsed whitespace. V2.0 and V2.1 share {overlap['pairwise']['V2.0_vs_V2.1']['shared_normalized_texts']:,} unique normalized responses. V2 training rows have {overlap['versus_v0']['V2.0']['shared_normalized_texts']} / {overlap['versus_v0']['V2.1']['shared_normalized_texts']} normalized-response overlaps with V0 and {overlap['versus_v0']['V2.0']['shared_prompt_ids']} / {overlap['versus_v0']['V2.1']['shared_prompt_ids']} prompt-ID overlaps.",
        f"V2.0/V2.1 have {overlap['versus_v1_5_holdout']['V2.0']['shared_normalized_texts']} / {overlap['versus_v1_5_holdout']['V2.1']['shared_normalized_texts']} normalized-text overlaps and {overlap['versus_v1_5_holdout']['V2.0']['shared_prompt_ids']} / {overlap['versus_v1_5_holdout']['V2.1']['shared_prompt_ids']} prompt-ID overlaps with the V1.5 540-row holdout.",
        "", f"V2.1's unverified Gemini 3.7 cohort is explicitly marked unverified in the source profile; no attribution is inferred from missing metadata alone. A conservative provider-family check flagged {provider_conflict_count} explicit candidate contradictions for source-level review; this is a review queue, not proof of mislabeling. Source composition, per-label source concentration, attribution-field coverage, duplicate groups, and cross-label conflicts are in `source_composition_and_evidence.csv`, `source_diversity_by_label.csv`, `evidence_field_coverage.csv`, `duplicate_audit.csv`, and `provider_conflict_candidates.csv`.",
        "", "## Sequence length and training dynamics", "",
        token_stats.to_markdown(index=False, floatfmt=".1f"), "",
        curves[[c for c in ("version", "epoch", "step", "train_loss", "eval_loss",
                            "eval_accuracy", "eval_balanced_accuracy") if c in curves]].to_markdown(
                                index=False, floatfmt=".4f"), "",
        "Token-length counts use the saved UltraBERT tokenizer and do not truncate. Only 628/51,606 V2.0 rows (1.22%) and 661/54,000 V2.1 rows (1.22%) exceed V1's 384-token limit; 3 examples in each exceed 1,024. Nearly all rows above 384 tokens come from the V1.3 source family. The 1,595 limit is set by a single long-tail response, so its direct truncation effect applies to a small minority, though those long rows can still change memory use and batch probing.",
        "The V1 run logged 3,185 optimizer steps/epoch and ended at step 9,555. V2.0 used 6,451 steps/epoch and selected epoch 2 at step 12,902; V2.1 used 6,750 steps/epoch and selected epoch 2 at step 13,500. With the configured gradient accumulation of 2, the probed V2 microbatch 4 implies an optimizer batch of 8, versus V1's configured 8×2=16. This is a substantial training-dynamics confound alongside sequence length.",
        "", "## Independent V2 transfer check", "",
        "The saved V2 checkpoints were run once on the V1.5 prompt-group holdout, which has zero normalized-text and prompt-ID overlap with V2 training. V1 selected its own checkpoint on this holdout, so these results are reported for V2 transfer only, not as an unbiased head-to-head comparison.", "",
        holdout_metrics.to_markdown(index=False, floatfmt=".4f"), "",
        "Per-label holdout results and predictions are in `v1_5_holdout_per_label_metrics.csv` and `v1_5_holdout_predictions.parquet`.",
        "", "## Limits and next causal test", "",
        "This audit establishes reproducible associations, not a causal winner among data source, sequence length, and optimizer dynamics. V2.0 and V2.1 share most normalized response text, and both use the same newer code path; their similar V0 scores narrow the shared cause but do not prove it. The V1 model's original training data snapshot cannot be reconstructed from the current profiles.",
        "", "If this evidence does not settle the cause, the next experiment should train matched checkpoints with a sealed prompt-group evaluation set and vary one factor at a time: 384 versus automatic sequence length; V2.0 versus V2.1 source composition; and V2.1 with versus without the unverified Gemini 3.7 rows. Do not use the sealed evaluation groups for epoch selection.",
        "", "## Reproducibility", "",
        f"Analysis script: [`analyze_ultrabert_v2_degradation.py`](../scripts/analyze_ultrabert_v2_degradation.py). Output directory: `{out_dir}`. Input manifests, dataset profile hashes, inference runtimes, batch sizes, and overlap checks are saved in `audit_summary.json`.", "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/analysis/ultrabert_v2_degradation")
    parser.add_argument("--report", default="docs/ultrabert_v2_degradation.md")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    root = Path.cwd()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = root / args.report

    profile_paths = {"V1": Path("data/processed/dataset_profile_v1.json"),
                     "V1.5": Path("data/processed/dataset_profile_v1.5.json"),
                     "V2.0": Path("data/processed/dataset_profile_v2.0.json"),
                     "V2.1": Path("data/processed/dataset_profile_v2.1.json")}
    profiles = {name: json.loads(path.read_text()) for name, path in profile_paths.items()}
    datasets = {
        "V1": pd.read_parquet("data/processed/dataset_v1.parquet"),
        "V2.0": pd.read_parquet("data/processed/dataset_v2.0.parquet"),
        "V2.1": pd.read_parquet("data/processed/dataset_v2.1.parquet"),
    }
    v0 = pd.read_parquet("data/processed/dataset_v0.parquet")
    v1_5 = pd.read_parquet("data/processed/dataset_v1.5.parquet")
    v1_5_holdout = v1_5[v1_5.split.astype(str).eq("validation")].copy().reset_index(drop=True)
    if len(v0) != 2699 or v1_5_holdout.shape[0] != 540:
        raise ValueError(f"unexpected validation row counts: V0={len(v0)}, V1.5 holdout={len(v1_5_holdout)}")
    if set(v0.model.astype(str)) != set(LABELS) or set(v1_5_holdout.model.astype(str)) != set(LABELS):
        raise ValueError("V0 or V1.5 holdout class set differs from the frozen 18 labels")
    v0_ids = set(v0.id.astype(str))
    v0_truth = dict(zip(v0.id.astype(str), v0.model.astype(str)))

    # Same V0 rows and labels for the three existing prediction caches.
    v0_frames = [read_predictions(version, v0_ids, v0_truth) for version in ("V1", "V2.0", "V2.1")]
    v0_overall = [score_summary(frame, LABELS, version=version)
                  for frame, version in zip(v0_frames, ("V1", "V2.0", "V2.1"))]
    v0_per_label = pd.DataFrame([row for frame, version in zip(
        v0_frames, ("V1", "V2.0", "V2.1"))
        for row in metric_rows(frame, LABELS, version=version)])
    v0_per_label.to_csv(out_dir / "v0_per_label_metrics.csv", index=False)
    for frame, version in zip(v0_frames, ("V1", "V2.0", "V2.1")):
        matrix = pd.crosstab(frame.y_true, frame.y_pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
        matrix.to_csv(out_dir / f"v0_confusion_{version.replace('.', '_')}.csv")
        pairs = (frame[frame.y_true.ne(frame.y_pred)].groupby(["y_true", "y_pred"]).size()
                 .sort_values(ascending=False).rename("errors").reset_index())
        pairs.insert(0, "version", version)
        pairs.to_csv(out_dir / f"v0_error_pairs_{version.replace('.', '_')}.csv", index=False)
    pd.DataFrame(v0_overall).to_csv(out_dir / "v0_overall_metrics.csv", index=False)

    data_details, duplicate_frame = data_audits(datasets, out_dir,
                                                 {"V1": profiles["V1"], "V2.0": profiles["V2.0"], "V2.1": profiles["V2.1"]})
    provider_conflict_frame = provider_conflicts(datasets, out_dir)
    overlap = build_overlap(datasets, v0, v1_5_holdout)
    (out_dir / "dataset_overlap.json").write_text(json.dumps(overlap, indent=2) + "\n")

    # Input identities and recorded checkpoint/training configurations.
    manifest_summary = {}
    curve_by_epoch: dict[tuple[str, int], dict] = {}
    run_summary_rows = []
    manifests = {}
    for version, experiment_id in EXPERIMENTS.items():
        path = Path("artifacts/predictions") / f"{experiment_id}_manifest.json"
        manifest = json.loads(path.read_text())
        manifests[version] = manifest
        manifest_summary[version] = {key: manifest.get(key) for key in (
            "experiment_id", "dataset", "training_dataset", "validation_dataset",
            "train_rows", "validation_rows", "training_dataset_rows", "training_dataset_sha256",
            "dataset_sha256", "split_sha256", "train_id_sha256", "id_sha256",
            "validation_ids_sha256", "code_revision", "model_revision", "config",
            "selected_epoch", "selected_validation_accuracy", "checkpoint_selection_metric",
            "effective_batch_size", "requested_batch_size", "fit_seconds", "checkpoint_size_bytes")}
        if version in CHECKPOINTS:
            manifest_summary[version]["checkpoint_model_safetensors_sha256"] = sha256_file(
                CHECKPOINTS[version] / "model.safetensors")
        run_dir = Path("artifacts/runs") / experiment_id / "checkpoints/ultrabert380_foldvalidation"
        training_log = run_dir / "training_log.json"
        if training_log.is_file():
            for record in json.loads(training_log.read_text()):
                if "epoch" in record:
                    epoch = int(record["epoch"])
                    row = curve_by_epoch.setdefault((version, epoch), {"version": version, "epoch": epoch})
                    for key in ("loss", "grad_norm", "learning_rate", "step", "eval_loss",
                                "eval_accuracy", "eval_balanced_accuracy"):
                        if key in record:
                            row[{"loss": "train_loss"}.get(key, key)] = record[key]
                if "train_runtime" in record or "train_loss" in record:
                    run_summary_rows.append({"version": version, **record})
        for epoch in manifest.get("validation_metrics_by_epoch", []):
            row = curve_by_epoch.setdefault((version, int(epoch["epoch"])),
                                             {"version": version, "epoch": int(epoch["epoch"])})
            for key in ("step", "eval_loss", "eval_accuracy", "eval_balanced_accuracy",
                        "validation_loss_raw", "validation_loss_calibrated", "calibration_temperature"):
                if key in epoch:
                    row[key] = epoch[key]
    current_profile = {"V1": profiles["V1"], "V1.5": profiles["V1.5"]}
    historical_split = pd.read_parquet("data/processed/splits_v1.5.parquet")
    train_split_hash = sha256_ids(historical_split.loc[historical_split.split.eq("train"), "id"])
    validation_split_hash = sha256_ids(historical_split.loc[historical_split.split.eq("validation"), "id"])
    manifest_summary["V1_snapshot_comparison"] = {
        "manifest_dataset_sha256": manifests["V1"].get("dataset_sha256"),
        "manifest_total_rows": int(manifests["V1"].get("train_rows", 0)) + int(manifests["V1"].get("validation_rows", 0)),
        "current_v1_profile_sha256": current_profile["V1"].get("dataset_sha256"),
        "current_v1_profile_rows": current_profile["V1"].get("rows"),
        "current_v1_5_profile_sha256": current_profile["V1.5"].get("dataset_sha256"),
        "current_v1_5_profile_rows": current_profile["V1.5"].get("rows"),
        "train_id_hash_matches_splits_v1_5": manifests["V1"].get("train_id_sha256") == train_split_hash,
        "validation_id_hash_matches_splits_v1_5": manifests["V1"].get("id_sha256") == validation_split_hash,
        "note": "Historical model dataset hash is not present in current V1/V1.5 profiles; split ID hashes can be compared separately.",
    }
    (out_dir / "audit_summary.json").write_text(json.dumps({
        "manifests": manifest_summary, "datasets": data_details,
        "v0_overall": v0_overall, "overlap": overlap,
        "duplicate_audit": duplicate_frame.to_dict("records"),
        "provider_conflict_candidate_rows": len(provider_conflict_frame),
        "holdout_definition": {"dataset": "V1.5", "rows": len(v1_5_holdout),
                               "ids_sha256": sha256_ids(v1_5_holdout.id),
                               "prompt_groups": int(v1_5_holdout.prompt_id.nunique()),
                               "class_counts": v1_5_holdout.model.value_counts().to_dict()},
    }, indent=2, default=str) + "\n")
    curves_frame = pd.DataFrame(curve_by_epoch.values()).sort_values(["version", "epoch"])
    curves_frame.to_csv(out_dir / "training_curves.csv", index=False)
    pd.DataFrame(run_summary_rows).to_csv(out_dir / "training_run_summary.csv", index=False)

    # Token length distributions on each training corpus and the independent holdout.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS["V2.0"], local_files_only=True,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_stats_rows = []
    for name in ("V1", "V2.0", "V2.1"):
        _, stats = token_length_stats(tokenizer, datasets[name], name, out_dir)
        token_stats_rows.extend(stats)
    _, holdout_token_stats = token_length_stats(tokenizer, v1_5_holdout, "V1.5_holdout", out_dir)
    token_stats_rows.extend(holdout_token_stats)
    _, v0_token_stats = token_length_stats(tokenizer, v0, "V0", out_dir)
    token_stats_rows.extend(v0_token_stats)
    token_stats = pd.DataFrame(token_stats_rows)
    token_stats.to_csv(out_dir / "token_length_summary.csv", index=False)

    # One inference-only pass per V2 checkpoint; V1 is intentionally not scored on
    # its own checkpoint-selection holdout as an unbiased comparator.
    holdout_pred_frames, holdout_metrics_rows, runtimes = [], [], {}
    holdout_texts = v1_5_holdout.text.fillna("").astype(str).tolist()
    for version in ("V2.0", "V2.1"):
        predictions, runtime = infer_checkpoint(version, holdout_texts, CHECKPOINTS[version],
                                                manifests[version]["config"]["max_length"], args.batch_size)
        runtimes[version] = runtime
        pred_frame = v1_5_holdout[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
        pred_frame["y_pred"] = predictions
        pred_frame.insert(0, "version", version)
        holdout_pred_frames.append(pred_frame)
        holdout_metrics_rows.extend(metric_rows(pred_frame, LABELS, version=version))
        overall = score_summary(pred_frame, LABELS, version=version)
        overall["elapsed_seconds"] = runtime["elapsed_seconds"]
        holdout_metrics_rows.append({"version": version, "label": "__OVERALL__",
                                     "support": len(pred_frame), "predicted_count": len(pred_frame),
                                     "true_positive": None, "precision": overall["accuracy"],
                                     "recall": overall["balanced_accuracy"], "f1": overall["macro_f1"],
                                     "elapsed_seconds": runtime["elapsed_seconds"]})
    holdout_predictions = pd.concat(holdout_pred_frames, ignore_index=True)
    holdout_predictions.to_parquet(out_dir / "v1_5_holdout_predictions.parquet", index=False)
    holdout_metrics = pd.DataFrame(holdout_metrics_rows)
    holdout_metrics.to_csv(out_dir / "v1_5_holdout_per_label_metrics.csv", index=False)
    (out_dir / "holdout_inference_runtime.json").write_text(json.dumps(runtimes, indent=2) + "\n")

    # Put prediction metrics in a compact overall table for the report.
    holdout_overall = holdout_metrics[holdout_metrics.label.eq("__OVERALL__")][
        ["version", "support", "precision", "recall", "f1", "elapsed_seconds"]].rename(
            columns={"support": "rows", "precision": "accuracy", "recall": "balanced_accuracy", "f1": "macro_f1"})
    write_report(root, out_dir, report_path, {"v0_overall": v0_overall},
                 v0_per_label, holdout_overall, token_stats[token_stats.source_type.eq("all")],
                 data_details, overlap, duplicate_frame, curves_frame,
                 len(provider_conflict_frame))
    print(f"Report: {report_path}")
    print(f"Supporting artifacts: {out_dir}")
    print("V0 overall:")
    print(pd.DataFrame(v0_overall).to_string(index=False))
    print("V1.5 holdout overall:")
    print(holdout_overall.to_string(index=False))


if __name__ == "__main__":
    main()
