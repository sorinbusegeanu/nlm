"""CLI orchestration for immutable V0 data, model, ensemble, and export runs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import balanced_accuracy_score

from .data_factory import (assemble_openpool_mix, assemble_v1_5, build_v1_3_external_dataset,
                           curate_openpool, curate_v1_all_balanced_deduped,
                           curate_v1_all_clean_compact, prepare, _normal_text,
                           validate_data, make_splits)
from .evaluation import (align_predictions, complementarity, evaluate, paired_group_bootstrap)
from .ensemble_factory import ensemble_probabilities
from .feature_factory import load_or_fit, stylometry
from .model_factory import (FROZEN_BACKBONE_MODELS, calculate_max_token_length,
                            train_encoder_fold,
                            train_frozen_backbone_fold, train_qwen_lora_fold,
                            train_catboost_stylometry, train_svc, select_epoch_view_indices)
from .registry import Registry
from .v2_0_chunks800 import main as build_v2_0_chunks800_dataset

MODEL_CONFIGS = {"svc": "configs/models/svc.yaml", "ultrabert380": "configs/models/ultrabert380.yaml",
    "phi35_mini": "configs/models/phi35_mini.yaml",
    "qwen06": "configs/models/qwen06.yaml", "qwen17": "configs/models/qwen17.yaml", "qwen4": "configs/models/qwen4.yaml",
    "qwen8": "configs/models/qwen8.yaml",
    "minicpm5_2b": "configs/models/minicpm5_2b.yaml",
    "hunyuan_1_8b": "configs/models/hunyuan_1_8b.yaml",
    "llama32_3b": "configs/models/llama32_3b.yaml", "smollm3_3b": "configs/models/smollm3_3b.yaml",
    "llama31_8b": "configs/models/llama31_8b_qlora.yaml",
    "granite4_h_micro": "configs/models/granite4_h_micro.yaml",
    "granite4_350m": "configs/models/granite4_350m.yaml",
    "embeddinggemma2": "configs/models/embeddinggemma2.yaml",
    "gemma4_e2b": "configs/models/gemma4_e2b.yaml",
    "gemma4_e4b": "configs/models/gemma4_e4b.yaml",
    "ministral3_3b": "configs/models/ministral3_3b.yaml",
    "ministral3_8b": "configs/models/ministral3_8b_qlora.yaml",
    "catboost_stylometry": "configs/models/catboost_stylometry.yaml"}


class _TeeTextStream:
    """Write training output to both its original stream and a per-run log."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", None) or "utf-8"

    @property
    def errors(self):
        return getattr(self.terminal, "errors", None) or "replace"

    def write(self, value):
        self.log_file.write(value)
        self.log_file.flush()
        return self.terminal.write(value)

    def flush(self):
        self.log_file.flush()
        self.terminal.flush()

    def isatty(self):
        return False

    def writable(self):
        return True

    def fileno(self):
        return self.terminal.fileno()
DEFAULT_MVP_MODELS = ["svc", "catboost_stylometry", "ultrabert380", "qwen06"]
ENSEMBLE_MODEL_CHOICES = [*MODEL_CONFIGS, "qwen06_best_epoch03"]


def _revision():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: return "unknown"


def _read_yaml(path):
    with open(path, encoding="utf-8") as stream: return yaml.safe_load(stream) or {}


_OBSOLETE_TRAINING_CONFIG_FIELDS = {
    "calibration", "calibration_cv", "calibration_enabled", "calibration_full_dataset",
    "calibration_max_samples", "calibration_seed", "temperature_calibration",
}


def _strip_obsolete_calibration_config(config):
    for key in _OBSOLETE_TRAINING_CONFIG_FIELDS:
        # SVC sigmoid calibration is an explicit, train-only option. Other
        # post-training calibration settings remain obsolete.
        if key in {"calibration", "calibration_cv"} and config.get("model") == "svc":
            continue
        config.pop(key, None)
    return config


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _ensemble_member_labels(members):
    display_names = {"svc": "SVC", "ultrabert380": "UltraBERT-380M", "qwen06": "Qwen3-0.6B",
        "qwen06_best_epoch03": "Qwen3-0.6B best checkpoint (epoch 3)",
        "phi35_mini": "Phi-3.5 Mini",
        "qwen17": "Qwen3-1.7B", "qwen4": "Qwen3-4B", "qwen8": "Qwen3-8B QLoRA", "llama32_3b": "Llama-3.2-3B", "llama31_8b": "Llama-3.1-8B QLoRA",
        "minicpm5_2b": "MiniCPM5-2B-Base", "hunyuan_1_8b": "Hunyuan-1.8B-Pretrain",
        "smollm3_3b": "SmolLM3-3B",
        "granite4_h_micro": "Granite 4.0 H Micro Base",
        "granite4_350m": "Granite 4.0 350M Base",
        "embeddinggemma2": "EmbeddingGemma 2", "gemma4_e2b": "Gemma 4 E2B", "gemma4_e4b": "Gemma 4 E4B",
        "ministral3_3b": "Ministral 3 3B", "ministral3_8b": "Ministral 3 8B QLoRA", "catboost_stylometry": "CatBoost + Stylometry"}
    model_ids = sorted(display_names, key=len, reverse=True)
    labels = []
    for name in members:
        model_id = next((item for item in model_ids if name == item or name.startswith(item + "_")), None)
        labels.append(display_names.get(model_id, name.split("_", 1)[0]))
    return labels


def _ensemble_model_label(members):
    return " + ".join(_ensemble_member_labels(members)) + " ensemble"


def _ensemble_variant_label(name):
    train_weighted = name.startswith("train_weighted:")
    if train_weighted:
        name = name.removeprefix("train_weighted:")
    labels = _ensemble_member_labels(name.split("+"))
    method = "Train-fitted weighted blend" if train_weighted else "Equal-weight blend"
    return f"{method}: {' + '.join(labels)}"


def prepare_data(config):
    cfg = _read_yaml(config)
    return prepare(cfg["train_path"], cfg["output_dir"], cfg.get("test_path"), seed=cfg.get("seed", 2026),
                   validation_fraction=cfg.get("validation_fraction", .2), candidates=cfg.get("split_candidates", 1000),
                   expected_classes=cfg.get("expected_classes", 18), dataset_version=cfg.get("dataset_version", "v0"))


def curate_openpool_data(config):
    cfg = _read_yaml(config)
    return curate_openpool(cfg["source_path"], cfg.get("output_dir", "data/processed"),
        dataset_version=cfg.get("dataset_version", "v1"),
        official_reference_path=cfg.get("official_reference_path"), seed=cfg.get("seed", 2026),
        expected_classes=cfg.get("expected_classes", 18), rows_per_class=cfg.get("rows_per_class", 2711),
        validation_dataset=cfg.get("validation_dataset", "v0"),
        selection_method=cfg.get("selection_method", "source_stratified"),
        base_dataset_path=cfg.get("base_dataset_path"),
        additional_rows_by_class=cfg.get("additional_rows_by_class"))


def curate_v1_all_balanced_deduped_data(config):
    cfg = _read_yaml(config)
    source_path = Path(cfg.get("source_path", "data/processed/dataset_v1-all.parquet"))
    # The checked-in V1-all artifact predates the requested hyphenated alias.
    if not source_path.exists() and source_path.name == "dataset_v1-all.parquet":
        alias = source_path.with_name("dataset_v1_all.parquet")
        if alias.exists():
            source_path = alias
    profile = curate_v1_all_balanced_deduped(
        source_path, cfg.get("output_dir", "data/processed"),
        dataset_version=cfg.get("dataset_version", "v1-all_balanced_deduped"),
        validation_dataset=cfg.get("validation_dataset", "v0"),
        seed=cfg.get("seed", 2026),
        near_duplicate_threshold=cfg.get("near_duplicate_threshold", 0.90),
        char_ngram_size=cfg.get("char_ngram_size", 5),
        minhash_permutations=cfg.get("minhash_permutations", 64),
        minimum_template_chars=cfg.get("minimum_template_chars", 100),
        template_min_repeats=cfg.get("template_min_repeats", 5),
        source_median_multiplier=cfg.get("source_median_multiplier", 3.0))
    report_path = Path(cfg.get("report_path", "docs/v1_all_balanced_deduped.md"))
    _write_v1_all_curation_report(profile, report_path)
    profile["report_path"] = str(report_path)
    return profile


def curate_v1_all_clean_compact_data(config):
    cfg = _read_yaml(config)
    source_path = Path(cfg.get("source_path", "data/processed/dataset_v1_all.parquet"))
    if not source_path.exists() and source_path.name == "dataset_v1-all.parquet":
        alias = source_path.with_name("dataset_v1_all.parquet")
        if alias.exists():
            source_path = alias
    result = curate_v1_all_clean_compact(
        source_path, cfg.get("output_dir", "data/processed"),
        dataset_version=cfg.get("dataset_version", "v1-all"),
        validation_dataset=cfg.get("validation_dataset", "v0"),
        seed=cfg.get("seed", 2026),
        minhash_permutations=cfg.get("minhash_permutations", 64),
        compact_char_similarity=cfg.get("compact_char_similarity", 0.92),
        compact_word_similarity=cfg.get("compact_word_similarity", 0.85),
        tfidf_diagnostic_sample=cfg.get("tfidf_diagnostic_sample", 3000))
    report_path = Path(cfg.get("report_path", "docs/v1_all_clean_compact.md"))
    _write_v1_all_clean_compact_report(result, report_path,
        cfg.get("output_dir", "data/processed"),
        dataset_version=cfg.get("dataset_version", "v1-all"),
        display_name=cfg.get("display_name", cfg.get("dataset_version", "V1-all")))
    result["report_path"] = str(report_path)
    return result


def build_v1_3_external_data(config):
    """Assemble V1.3 from the two new raw corpora, then run clean and compact."""
    cfg = _read_yaml(config)
    output_dir = cfg.get("output_dir", "data/processed")
    dataset_version = cfg.get("dataset_version", "v1.3")
    base_profile = build_v1_3_external_dataset(
        cfg.get("glm_primary_files", []),
        cfg.get("glm_provenance_path", "data/raw/external_candidates/glm53_flash/data/provenance_r34.parquet"),
        cfg.get("gpt_traces_path", "data/raw/external_candidates/gpt56_sol_luna/data/train.parquet"),
        output_dir, dataset_version=dataset_version,
        validation_dataset=cfg.get("validation_dataset", "v0"),
        max_response_chars=cfg.get("max_response_chars", 1900))
    clean_result = curate_v1_all_clean_compact(
        Path(output_dir) / f"dataset_{dataset_version}.parquet", output_dir,
        dataset_version=dataset_version,
        validation_dataset=cfg.get("validation_dataset", "v0"),
        expected_classes=cfg.get("expected_classes", 3),
        seed=cfg.get("seed", 2026),
        minhash_permutations=cfg.get("minhash_permutations", 64),
        compact_char_similarity=cfg.get("compact_char_similarity", 0.92),
        compact_word_similarity=cfg.get("compact_word_similarity", 0.85),
        tfidf_diagnostic_sample=cfg.get("tfidf_diagnostic_sample", 3000))
    clean_result["source_dataset_profile"] = base_profile
    report_path = Path(cfg.get("report_path", "docs/v1_3_clean_compact.md"))
    display_name = cfg.get("display_name", "V1.3")
    _write_v1_all_clean_compact_report(clean_result, report_path, output_dir,
                                       dataset_version=dataset_version, display_name=display_name)
    return {"dataset": base_profile, "clean": clean_result["clean"],
            "compact": clean_result["compact"], "report_path": str(report_path),
            "verified_near_duplicate_pairs": clean_result["verified_near_duplicate_pairs"],
            "clean_stats": clean_result["clean_stats"],
            "compact_stats": clean_result["compact_stats"]}


def _write_v1_all_clean_compact_report(result, path, data_dir, *,
                                       dataset_version="v1-all", display_name="V1-all"):
    before, clean, compact = result["before_stats"], result["clean_stats"], result["compact_stats"]
    clean_profile, compact_profile = result["clean"], result["compact"]
    clean_version, compact_version = f"{dataset_version}_clean", f"{dataset_version}_compact"
    clean_label, compact_label = f"{display_name} clean", f"{display_name} compact"
    input_path = f"{data_dir}/dataset_{dataset_version}.parquet"
    lines = [f"# {display_name} Clean and Compact Datasets", "",
        f"Input: `{input_path}`. The source build may apply its configured response-length limit; clean removes only explicit defects; compact starts from clean and removes verified same-label near duplicates. Neither clean nor compact uses class quotas, classifier outputs, or validation scores.", "",
        "## Dataset overview", "",
        f"| Metric | {display_name} | Clean | Compact |", "|---|---:|---:|---:|"]
    for label, key in [("Total rows", "rows"), ("Unique prompts", "prompt_groups"), ("Unique sources", "unique_sources")]:
        lines.append(f"| {label} | {before[key]:,} | {clean[key]:,} | {compact[key]:,} |")
    lines.extend([
        f"| Exact/normalized duplicate responses removed in Clean | — | {clean_profile['removals_by_reason'].get('normalized_duplicate_response', 0):,} | — |",
        f"| Verified near-duplicate pairs | — | — | {result['verified_near_duplicate_pairs']:,} |",
        f"| Model-label conflict rows removed in Clean | — | {clean_profile['removals_by_reason'].get('conflicting_labels_for_identical_response', 0):,} | — |", "",
        "## Per-model row counts", "", f"| Model | {display_name} | Clean | Compact | Clean retained | Compact retained from Clean |", "|---|---:|---:|---:|---:|---:|"])
    source_profile = result.get("source_dataset_profile", {})
    if source_profile.get("max_response_chars") is not None:
        lines.insert(3, f"Before this table, the data factory removed {source_profile['response_length_filter_removed_rows']:,} source rows with responses longer than {source_profile['max_response_chars']:,} characters; per-label removals: " +
                     ", ".join(f"`{label}` {count:,}" for label, count in source_profile.get("response_length_filter_removed_by_model", {}).items()) + ".")
    for model in sorted(result["class_counts_before"]):
        a = result["class_counts_before"][model]; b = result["class_counts_clean"][model]; c = result["class_counts_compact"][model]
        lines.append(f"| {model} | {a:,} | {b:,} | {c:,} | {b/max(1,a):.1%} | {c/max(1,b):.1%} |")
    lines.extend(["", "## Removal audit", "", "### Clean", "", "| Reason | Rows removed |", "|---|---:|"])
    for reason, count in sorted(clean_profile["removals_by_reason"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{reason}` | {count:,} |")
    clean_audit_path = Path(data_dir) / f"{clean_version}_removals.csv"
    compact_audit_path = Path(data_dir) / f"{compact_version}_removals.csv"
    for title, audit_path in (("Clean examples", clean_audit_path), ("Compact examples", compact_audit_path)):
        lines.extend(["", f"### {title}", ""])
        audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
        if audit.empty:
            lines.append("No rows were removed in this pass.")
            continue
        for reason, group in audit.groupby("removal_reason", sort=True):
            lines.append(f"**`{reason}` — {len(group):,} rows.** Example rows (up to five):")
            examples = group.head(5)
            for _, row in examples.iterrows():
                extra = f"; representative `{row.get('representative_id', '')}`" if pd.notna(row.get("representative_id")) and str(row.get("representative_id")) else ""
                lines.append(f"- `{row.get('id', 'N/A')}`{extra}: {row.get('evidence', 'measured removal rule')}.")
    lines.extend(["", "## Redundancy diagnostics", "",
        f"Near-duplicate detection unions character 5-gram and word-trigram MinHash candidates (candidate thresholds: character {result['similarity_thresholds']['minhash_candidate_thresholds']['character']:.2f}, word {result['similarity_thresholds']['minhash_candidate_thresholds']['word']:.2f}), then verifies exact character 5-gram Jaccard ≥ {result['similarity_thresholds']['character_5gram_jaccard']:.2f} and word 3-gram Jaccard ≥ {result['similarity_thresholds']['word_3gram_jaccard']:.2f}. Verified pairs: **{result['verified_near_duplicate_pairs']:,}**; connected clusters removed: **{result['near_duplicate_clusters']:,}**. Cluster sizes: `{result['near_duplicate_cluster_sizes']}`.",
        f"Of {result['minhash_candidate_pair_count']:,} same-label candidate pairs, character 5-gram Jaccard P50/P90/max = {result['candidate_similarity']['character_5gram_jaccard']['p50']}, {result['candidate_similarity']['character_5gram_jaccard']['p90']}, {result['candidate_similarity']['character_5gram_jaccard']['max']}; word 3-gram Jaccard P50/P90/max = {result['candidate_similarity']['word_3gram_jaccard']['p50']}, {result['candidate_similarity']['word_3gram_jaccard']['p90']}, {result['candidate_similarity']['word_3gram_jaccard']['max']}. Verified-pair medians = {result['verified_similarity']['character_5gram_jaccard']['p50']} (character) and {result['verified_similarity']['word_3gram_jaccard']['p50']} (word); pairs sharing a prompt ID: {result['verified_pairs_same_prompt']:,}.",
        f"Repeated normalized templates occurring at least three times: {result['template_diagnostic']['templates_repeated_3plus']:,} templates / {result['template_diagnostic']['rows_in_repeated_templates']:,} rows. Prompt groups with multiple same-label rows: {result['prompt_redundancy_diagnostic']['model_prompt_groups_with_multiple_rows']:,} groups / {result['prompt_redundancy_diagnostic']['rows_in_repeated_model_prompt_groups']:,} rows. These diagnostic counts did not independently cause removals.",
        f"TF-IDF diagnostic: {result['tfidf_diagnostic']['method']}; sample {result['tfidf_diagnostic']['sample_size']:,}; nearest-neighbor cosine similarity P50/P90/max = {result['tfidf_diagnostic']['nearest_cosine_p50']}, {result['tfidf_diagnostic']['nearest_cosine_p90']}, {result['tfidf_diagnostic']['nearest_cosine_max']}. Diagnostic only; not used to select rows.", "",
        "## Response characteristics", "", f"| Statistic | {display_name} | Clean | Compact |", "|---|---:|---:|---:|"])
    for key, label, fmt in [("length_chars", "Character length median (P10–P90)", ".0f"),
                            ("length_words", "Word count median (P10–P90)", ".0f")]:
        values = [before[key], clean[key], compact[key]]
        lines.append(f"| {label} | " + " | ".join(f"{v['0.5']:{fmt}} ({v['0.1']:{fmt}}–{v['0.9']:{fmt}})" for v in values) + " |")
    for key, label in [("mean_lexical_diversity", "Mean type-token ratio"), ("markdown_fraction", "Markdown/structured formatting fraction"),
                       ("code_block_fraction", "Code block fraction"), ("url_fraction", "URL fraction")]:
        lines.append(f"| {label} | " + " | ".join(f"{x[key]:.2%}" for x in (before, clean, compact)) + " |")
    for key, label in [("paragraph_count_median", "Median paragraphs per response"),
                       ("repeated_openings_10plus", "Opening templates repeated ≥10 times"),
                       ("repeated_endings_10plus", "Ending templates repeated ≥10 times")]:
        lines.append(f"| {label} | " + " | ".join(f"{x[key]:,.1f}" if isinstance(x[key], float) else f"{x[key]:,}" for x in (before, clean, compact)) + " |")
    clean_representative_pct = (f"{100 * clean_profile['removed_rows_with_representative'] / clean_profile['removed_rows']:.1f}%"
                                if clean_profile["removed_rows"] else "N/A (no rows removed)")
    compact_representative_pct = (f"{100 * compact_profile['removed_rows_with_representative'] / compact_profile['removed_rows']:.1f}%"
                                   if compact_profile["removed_rows"] else "N/A (no rows removed)")
    lines.extend(["", "## Integrity and validation", "",
        f"All {len(clean_profile['classes'])} labels in the source remain. Compact is a subset of clean. Both profiles are checked against the frozen V0 validation set (split hash `{clean_profile['validation_split_sha256']}`), with zero validation ID or prompt-group overlap. These Parquet artifacts contain training rows only.",
        "",
        f"{display_name} removals by rule: " + ", ".join(f"`{key}` {value:,}" for key, value in clean_profile["removals_by_reason"].items()) + ".",
        f"Removed rows with a retained representative: Clean {clean_representative_pct}; Compact {compact_representative_pct}. Similarity is evidence of redundancy, not proof that either response is defective or that training accuracy will improve.",
        "",
        "### Ambiguous findings retained",
        "",
        "Only explicit text markers and available source metadata were treated as defects; plausible but unmarked truncation, unusual writing, and uncertain provenance were retained. `model_version` conflicts were checked only when the value exactly matched another canonical label. No attempt was made to verify that the claimed model generated a response.",
        "MinHash is an approximate candidate index. Zero verified pairs means no candidate pair passed both configured thresholds; it is not proof that no near-duplicates exist. TF-IDF diagnostics cover only the deterministic 3,000-row sample and did not affect selection.",
        "", f"Profiles: `dataset_profile_{clean_version}.json`, `dataset_profile_{compact_version}.json`. Audits: `{clean_audit_path}`, `{compact_audit_path}`."])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_v1_all_curation_report(profile, path):
    lines = ["# V1-all Balanced and Deduped Dataset", "",
             f"Input: `{profile['source_dataset_path']}`  ",
             f"Output: `data/processed/dataset_{profile['dataset_version']}.parquet`  ",
             f"Method: `{profile['curation_method']}`; seed {profile['seed']}; word-trigram MinHash candidates verified with character {profile['char_ngram_size']}-gram Jaccard at threshold {profile['near_duplicate_threshold']:.2f}.", "",
             "## Dataset size and reduction", "",
             "| Metric | V1-all | New dataset | Change |", "|---|---:|---:|---:|"]
    original_rows = sum(v["original_count"] for v in profile["class_summary"].values())
    original_sources = profile["unique_sources_before"]
    selected_sources = profile["unique_sources"]
    lines.extend([
        f"| Total rows | {original_rows:,} | {profile['rows']:,} | {profile['rows']-original_rows:+,} ({100*profile['rows']/max(1,original_rows):.1f}% retained) |",
        f"| Unique prompts | {profile['unique_prompts_before']:,} | {profile['unique_prompts_after']:,} | {profile['unique_prompts_after']-profile['unique_prompts_before']:+,} |",
        f"| Unique sources | {original_sources:,} | {selected_sources:,} | {selected_sources-original_sources:+,} |",
        f"| Near-duplicate clusters | N/A | {profile['near_duplicate_clusters']:,} | Verified pairs: {profile['near_duplicate_pairs_verified']:,} |", "",
        "## Per-class summary", "",
        "| Model | Original | Retained | Retained % | Unique prompts | Sources |", "|---|---:|---:|---:|---:|---:|"])
    for label, row in sorted(profile["class_summary"].items(), key=lambda kv: (-kv[1]["retained_count"], kv[0])):
        lines.append(f"| {label} | {row['original_count']:,} | {row['retained_count']:,} | {row['retained_fraction']:.1%} | {row['unique_prompts_retained']:,} | {row['sources_retained']} |")
    lines.extend(["", "## Filtering summary", "", "| Removal reason | Rows |", "|---|---:|"])
    for reason, count in sorted(profile["removals_by_reason"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count:,} |")
    lines.extend(["", f"Rows removed: **{profile['removed_rows']:,}**. Suspicious exact cross-label conflicts: **{profile['suspicious_exact_label_conflict_rows']:,} rows**; near-duplicate cross-label clusters flagged for review: **{profile['suspicious_near_duplicate_cross_label_clusters']:,}**.",
        "", "| Length percentile (characters) | V1-all | New dataset |", "|---|---:|---:|"])
    for q, name in zip(["0.0", "0.1", "0.5", "0.9", "1.0"], ["Min", "P10", "Median", "P90", "Max"]):
        lines.append(f"| {name} | {profile['length_chars_before'][q]:,.0f} | {profile['length_chars_after'][q]:,.0f} |")
    before = profile["source_concentration_before"]
    after = profile["source_concentration_after"]
    lines.extend(["", "| Source concentration | V1-all | New dataset |", "|---|---:|---:|"])
    for metric, title in [("largest_source_fraction", "Mean largest-source share across classes"), ("top3_source_fraction", "Mean top-three-source share across classes")]:
        b = sum(v[metric] for v in before.values()) / len(before)
        a = sum(v[metric] for v in after.values()) / len(after)
        lines.append(f"| {title} | {b:.1%} | {a:.1%} |")
    lines.extend(["", "## Assessment", "",
        f"Selection was label-preserving and used no model predictions or validation scores. Exact duplicate text was checked across labels; word-trigram MinHash candidates were verified by character n-gram Jaccard. The reported zero near-duplicate clusters means no candidate pair passed verification; MinHash candidate generation is approximate and does not prove that no near-duplicates exist. Long repeated templates and statistical prompt outliers were reduced within each label. The adaptive source cap is max(Q3, min(Tukey upper fence, {profile.get('source_median_multiplier', 3.0):g} times the class median source count)). There are no equal class quotas. Source-cap selection round-robins response-length, formatting and lexical-richness bins while preferring complete provenance.",
        "", f"All 18 labels remain. Validation isolation: {profile['validation_rows']} frozen V0 rows, split hash `{profile['validation_split_sha256']}`, ID overlap {profile['validation_id_overlap']}, prompt overlap {profile['validation_prompt_overlap']}. The V0 validation data and the V1-all source artifact were read-only.",
        "", "Removal audit: `data/processed/" + Path(profile["removal_audit_path"]).name + "`. Profile: `data/processed/dataset_profile_" + profile["dataset_version"] + ".json`."])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble_training_dataset(config):
    cfg = _read_yaml(config)
    return assemble_openpool_mix(cfg.get("input_dir", "data/processed"),
        cfg.get("output_dir", "data/processed"),
        dataset_version=cfg.get("dataset_version", "v1.5"),
        openpool_version=cfg.get("openpool_version", "v1"),
        expected_classes=cfg.get("expected_classes", 18))


def _profile_path(data_dir, dataset):
    name = "dataset_profile.json" if dataset == "v0" else f"dataset_profile_{dataset}.json"
    return Path(data_dir) / name


def _effective_profile(data_dir, dataset, rows):
    profile = json.loads(_profile_path(data_dir, dataset).read_text())
    is_training_only = (profile.get("training_only") is True or
                        str(profile.get("curation_method", "")).startswith("openpool_v11_hard_defect_"))
    if dataset != "v0" and is_training_only and profile.get("validation_dataset_version") == "v0":
        v0_profile = json.loads(_profile_path(data_dir, "v0").read_text())
        train = rows.loc[rows.split.eq("train")].sort_values("id", kind="stable")
        valid = rows.loc[rows.split.eq("validation")].sort_values("id", kind="stable")
        components = {"training_dataset_sha256": profile["dataset_sha256"],
                      "validation_dataset_sha256": v0_profile["dataset_sha256"],
                      "v0_validation_split_sha256": v0_profile["split_sha256"],
                      "training_ids_sha256": hashlib.sha256("\n".join(sorted(train.id.astype(str))).encode()).hexdigest(),
                      "validation_ids_sha256": hashlib.sha256("\n".join(sorted(valid.id.astype(str))).encode()).hexdigest()}
        profile = dict(profile)
        profile.update({"split_sha256": _sha(components), "training_rows": len(train),
                        "validation_rows": len(valid), "effective_dataset_components": components})
    return profile


def _load_dataset(dataset, data_dir):
    """Load the historical split view for analysis utilities."""
    data_dir = Path(data_dir)
    rows = pd.read_parquet(data_dir / f"dataset_{dataset}.parquet")
    if dataset != "v0" and "split" not in rows.columns:
        profile_path = _profile_path(data_dir, dataset)
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            is_training_only = (profile.get("training_only") is True or
                str(profile.get("curation_method", "")).startswith("openpool_v11_hard_defect_"))
            if is_training_only and profile.get("validation_dataset_version") == "v0":
                v0_rows, _ = _load_dataset("v0", data_dir)
                validation = v0_rows.loc[v0_rows.split.eq("validation")].copy()
                if len(validation) != int(profile.get("validation_rows", -1)):
                    raise ValueError(f"{dataset} reference validation row count differs from V0 assignments")
                if profile.get("validation_split_sha256") != json.loads(
                        _profile_path(data_dir, "v0").read_text()).get("split_sha256"):
                    raise ValueError(f"{dataset} references a different V0 split artifact")
                external = rows.copy()
                external["split"] = "train"
                combined = pd.concat([external, validation], ignore_index=True, sort=False)
                if combined.id.duplicated().any():
                    raise ValueError("training dataset and referenced V0 validation IDs overlap")
                if set(external.prompt_id.astype(str)) & set(validation.prompt_id.astype(str)):
                    raise ValueError("training dataset and referenced V0 validation prompts overlap")
                return combined.sort_values("id", kind="stable").reset_index(drop=True), data_dir
    assignments_path = data_dir / f"splits_{dataset}.parquet"
    if "split" in rows.columns:
        if assignments_path.is_file():
            assignments = pd.read_parquet(assignments_path)
            if rows.id.duplicated().any() or assignments.id.duplicated().any():
                raise ValueError("dataset and split assignment IDs must be unique")
            inline = rows.set_index("id").split.astype(str).sort_index()
            assigned = assignments.set_index("id").split.astype(str).sort_index()
            if not inline.equals(assigned):
                raise ValueError("inline split values differ from the split assignment artifact")
        return rows, data_dir
    if not assignments_path.is_file():
        raise FileNotFoundError(f"split assignments are missing for {dataset}")
    assignments = pd.read_parquet(assignments_path)
    if rows.id.duplicated().any() or assignments.id.duplicated().any():
        raise ValueError("dataset and split assignment IDs must be unique")
    if set(rows.id.astype(str)) != set(assignments.id.astype(str)):
        raise ValueError("dataset and split assignment ID sets differ")
    rows = rows.merge(assignments[["id", "split"]], on="id", how="left", validate="one_to_one")
    if rows["split"].isna().any():
        raise ValueError("every dataset row needs a split assignment")
    return rows, data_dir


def _dataset_rows(dataset, data_dir):
    """Load every row in a dataset artifact, ignoring any historical split column."""
    rows = pd.read_parquet(Path(data_dir) / f"dataset_{dataset}.parquet")
    required = {"id", "prompt_id", "text", "model"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {sorted(missing)}")
    if (rows.id.duplicated().any() or rows.model.isna().any() or rows.text.isna().any()
            or rows.prompt_id.isna().any()):
        raise ValueError(f"{dataset} has duplicate IDs or missing response/label values")
    return rows.drop(columns=["split"], errors="ignore").sort_values(
        "id", kind="stable").reset_index(drop=True)


def _partition_sha256(rows):
    content = rows[[column for column in ("id", "prompt_id", "text", "model")
                    if column in rows]].astype(str).to_csv(index=False, lineterminator="\n")
    digest = hashlib.sha256(content.encode("utf-8"))
    if "training_views" in rows:
        for index, views in enumerate(rows.training_views):
            digest.update(str(index).encode())
            digest.update(json.dumps(list(views), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if "training_view_ids" in rows:
                digest.update(json.dumps(list(rows.iloc[index].training_view_ids), separators=(",", ":")).encode())
    return digest.hexdigest()


def _original_response_rows(rows: pd.DataFrame, *, training: bool) -> pd.DataFrame:
    """Collapse a multi-view artifact to one original row, retaining its views."""
    if "view_type" not in rows or "original_response_id" not in rows:
        return rows
    original_ids = rows.original_response_id.astype(str)
    full_mask = rows.view_type.astype(str).eq("full_response")
    full_counts = original_ids[full_mask].value_counts()
    missing_or_duplicate = full_counts[full_counts.ne(1)]
    if len(missing_or_duplicate):
        raise ValueError(f"multi-view dataset needs one full response for each original; invalid groups: {len(missing_or_duplicate)}")
    full = rows.loc[full_mask].copy()
    if training:
        consistency = rows.assign(_original_id=original_ids).groupby("_original_id", sort=False).agg(
            labels=("model", "nunique"), prompts=("prompt_id", "nunique"))
        inconsistent = consistency.index[consistency["labels"].ne(1) | consistency["prompts"].ne(1)]
        if len(inconsistent):
            raise ValueError(f"multi-view rows disagree on label or prompt provenance for {len(inconsistent)} originals")
        chunks = rows.loc[~full_mask].copy()
        chunks["original_response_id"] = chunks.original_response_id.astype(str)
        chunks = chunks.sort_values(["original_response_id", "char_start", "chunk_id"], kind="stable")
        text_by_id = chunks.groupby("original_response_id", sort=False).text.agg(lambda values: list(map(str, values)))
        chunk_id_by_id = chunks.groupby("original_response_id", sort=False).chunk_id.agg(lambda values: list(map(str, values)))
        full_ids = full.original_response_id.astype(str)
        full["training_views"] = [ [str(text), *text_by_id.get(oid, [])]
                                   for oid, text in zip(full_ids, full.text)]
        full["training_view_ids"] = [["full", *chunk_id_by_id.get(oid, [])] for oid in full_ids]
    full["id"] = full.original_response_id.astype(str)
    return full.sort_values("id", kind="stable").reset_index(drop=True)


def select_single_training_views(training_views, view_ids=None, *, seed=2026, epoch=0):
    """Pick one reproducible view per original for single-fit classifiers."""
    return select_epoch_view_indices([len(views) for views in training_views], seed=seed,
                                     epoch=epoch, view_ids=view_ids)


def _continuation_source_matches(source, training_dataset, validation_dataset,
                                 train, valid, training_dataset_sha256, *,
                                 allow_cross_dataset_source=False):
    source_training_dataset = source.get("training_dataset", source.get("dataset"))
    same_training_dataset = source_training_dataset == training_dataset
    if not same_training_dataset and not allow_cross_dataset_source:
        return False
    if source.get("validation_dataset", source.get("validation_dataset_version")) != validation_dataset:
        return False
    train_hash, valid_hash = _partition_sha256(train), _partition_sha256(valid)
    if same_training_dataset and (source.get("training_dataset_sha256") or
                                  source.get("validation_dataset_sha256")):
        return (source.get("training_dataset_sha256") == train_hash
                and source.get("validation_dataset_sha256") == valid_hash)
    # Cross-dataset continuation is permitted only when the checkpoint's full
    # validation partition is exactly the target run's validation partition.
    expected_ids = hashlib.sha256("\n".join(sorted(valid.id.astype(str))).encode()).hexdigest()
    expected_prompt_ids = hashlib.sha256(
        "\n".join(sorted(valid.prompt_id.astype(str))).encode()).hexdigest()
    expected_normalized_text = hashlib.sha256(
        "\n".join(sorted(valid.text.astype(str).map(_normal_text))).encode()).hexdigest()
    legacy_validation = source.get("full_v0_validation") or {}
    recorded_ids = (source.get("validation_ids_sha256") or
                    source.get("checkpoint_selection_ids_sha256") or
                    legacy_validation.get("ids_sha256"))
    recorded_prompt_ids = (source.get("validation_prompt_ids_sha256") or
                           source.get("checkpoint_selection_prompt_ids_sha256"))
    recorded_text = (source.get("validation_normalized_text_sha256") or
                     source.get("checkpoint_selection_normalized_text_sha256") or
                     legacy_validation.get("normalized_text_sha256"))
    if int(source.get("validation_rows", -1)) != len(valid):
        return False
    if recorded_ids != expected_ids:
        return False
    if recorded_prompt_ids is not None and recorded_prompt_ids != expected_prompt_ids:
        return False
    if recorded_text is not None and recorded_text != expected_normalized_text:
        # Legacy manifests hash ID and normalized text together.
        legacy_text = hashlib.sha256("\n".join(sorted(
            f"{row.id}\t{_normal_text(row.text)}"
            for row in valid[["id", "text"]].itertuples(index=False))).encode()).hexdigest()
        if recorded_text != legacy_text:
            return False
    if same_training_dataset and source.get("dataset_sha256") != training_dataset_sha256:
        return False
    return (source.get("validation_dataset_sha256") in {None, valid_hash})


def _training_partitions(training_dataset, validation_dataset, data_dir):
    train = _dataset_rows(training_dataset, data_dir)
    valid = _dataset_rows(validation_dataset, data_dir)
    if "view_type" in train:
        train = _original_response_rows(train, training=True)
    if "view_type" in valid:
        valid = _original_response_rows(valid, training=False)
    # The official V0 artifact records the historical prompt-grouped holdout.
    # When V0 is named on both sides, interpret those names as its two roles.
    if training_dataset == validation_dataset == "v0":
        source, _ = _load_dataset("v0", Path(data_dir))
        if "split" not in source:
            raise ValueError("V0 train/validation roles require the historical split assignments")
        train = source[source.split.astype(str).eq("train")].drop(
            columns=["split"], errors="ignore").sort_values("id", kind="stable").reset_index(drop=True)
        valid = source[source.split.astype(str).eq("validation")].drop(
            columns=["split"], errors="ignore").sort_values("id", kind="stable").reset_index(drop=True)
        if train.empty or valid.empty:
            raise ValueError("historical V0 split must have non-empty train and validation partitions")
    if set(train.model.astype(str)) != set(valid.model.astype(str)):
        raise ValueError("training and validation datasets must contain the same classes")
    checks = (("ID", set(train.id.astype(str)), set(valid.id.astype(str))),
              ("normalized response", set(train.text.astype(str).map(_normal_text)),
               set(valid.text.astype(str).map(_normal_text))),
              ("prompt group", set(train.prompt_id.astype(str)), set(valid.prompt_id.astype(str))))
    for name, left, right in checks:
        overlap = left & right
        if overlap:
            raise ValueError(f"training and validation datasets overlap on {name}: {len(overlap)}")
    for column in ("prompt_canon", "prompt", "prompt_text"):
        if column in train and column in valid:
            left = set(train[column].fillna("").astype(str).map(_normal_text)) - {""}
            right = set(valid[column].fillna("").astype(str).map(_normal_text)) - {""}
            overlap = left & right
            if overlap:
                raise ValueError(f"training and validation datasets overlap on normalized prompt content: {len(overlap)}")
    return train, valid


def _release_completed_training_model(result):
    """Free training-only model objects before the post-training prediction pass."""
    if isinstance(result, dict):
        result.pop("model", None)
        result.pop("tokenizer", None)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _refresh_ensemble_after_training(manifest, manifest_path, *, artifact_dir, data_dir,
                                     refresh_prediction=False, refresh_checkpoint_catalog=False):
    """Refresh selected downstream artifacts only when explicitly requested."""
    project_root = Path(__file__).resolve().parents[1]
    prediction_dir = Path(artifact_dir) / "predictions"
    started = time.perf_counter()
    if refresh_prediction:
        from .ensemble_registry import refresh_prediction_factory
        manifest["prediction_factory_update"] = refresh_prediction_factory(
            artifact_dir=artifact_dir, data_dir=data_dir, reference_dataset="v0",
            completed_manifest=manifest_path, evaluate_if_unchanged=False,
            registry_path=project_root / "docs" / "ensembles.md",
            roster_path=prediction_dir / "active_model_roster.json")
    else:
        manifest["prediction_factory_update"] = {"status": "skipped", "reason": "disabled"}
    if refresh_checkpoint_catalog:
        try:
            from .checkpoint_catalog import refresh_catalog
            manifest["checkpoint_catalog_update"] = refresh_catalog(
                artifact_root=artifact_dir, data_dir=data_dir)
        except Exception as exc:
            manifest["checkpoint_catalog_update"] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    else:
        manifest["checkpoint_catalog_update"] = {"status": "skipped", "reason": "disabled"}
    manifest.setdefault("runtime_breakdown", {})["downstream_refresh_seconds"] = time.perf_counter() - started
    manifest_path = Path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    metrics_path = manifest_path.with_name(manifest_path.name.removesuffix("_manifest.json") + "_metrics.json")
    metrics_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {"prediction_factory_update": manifest["prediction_factory_update"],
            "checkpoint_catalog_update": manifest["checkpoint_catalog_update"],
            "downstream_refresh_seconds": manifest["runtime_breakdown"]["downstream_refresh_seconds"]}


def _append_v2_checkpoint_trust(manifest, manifest_path):
    """Append saved full-V0 training metrics to the separate v2 trust list."""
    try:
        from .v2_checkpoint_trust import append_training_run
        result = append_training_run(manifest, manifest_path,
                                     root=Path(__file__).resolve().parents[1])
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    manifest["v2_checkpoint_trust_update"] = result
    manifest_path = Path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    metrics_path = manifest_path.with_name(
        manifest_path.name.removesuffix("_manifest.json") + "_metrics.json")
    metrics_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return result


def _post_training_runtime(result):
    """Normalize post-fit stage durations across model families."""
    runtime = dict(result.get("runtime_breakdown") or {})
    runtime.setdefault("best_checkpoint_restoration_seconds", 0.0)
    runtime.setdefault("final_validation_inference_seconds", result.get("inference_seconds"))
    runtime.setdefault("checkpoint_serialization_seconds", runtime.get("checkpoint_save_seconds", 0.0))
    runtime.setdefault("manifest_prediction_export_seconds", 0.0)
    runtime.setdefault("downstream_refresh_seconds", 0.0)
    runtime.setdefault("checkpoint_snapshot_seconds", runtime.get("best_checkpoint_state_snapshot_seconds", 0.0))
    runtime.setdefault("best_checkpoint_state_snapshot_seconds", runtime.get("checkpoint_snapshot_seconds", 0.0))
    runtime.setdefault("effective_inference_batch_size", result.get("effective_inference_batch_size"))
    runtime.setdefault("final_validation_rows_per_second", result.get("final_validation_rows_per_second"))
    runtime.setdefault("inference_oom_fallbacks", result.get("inference_oom_fallbacks", 0))
    runtime.setdefault("prediction_export_seconds", 0.0)
    runtime.setdefault("manifest_export_seconds", 0.0)
    runtime.setdefault("total_post_training_seconds", 0.0)
    runtime.setdefault("model_post_training_seconds", 0.0)
    return runtime


def _register_completed_manifest(manifest, registry_path):
    """Validate published outputs and idempotently make the manifest discoverable."""
    prediction_path = Path(manifest.get("prediction_path", ""))
    if not prediction_path.is_file():
        raise FileNotFoundError(f"completed experiment prediction artifact is missing: {prediction_path}")
    if manifest.get("prediction_sha256"):
        digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if digest != manifest["prediction_sha256"]:
            raise ValueError("completed experiment prediction artifact does not match its manifest hash")
    for key in ("selected_checkpoint_path", "best_checkpoint_path", "checkpoint_path"):
        checkpoint = manifest.get(key)
        if checkpoint and not Path(checkpoint).exists():
            raise FileNotFoundError(f"completed experiment {key} is missing: {checkpoint}")
    experiment_id = manifest.get("experiment_id")
    model = manifest.get("model")
    metrics = manifest.get("validation_metrics", manifest.get("metrics"))
    if not experiment_id or not model or not isinstance(metrics, dict):
        raise ValueError("completed experiment manifest is missing registration metadata")
    Registry(registry_path).add(experiment_id, model, manifest.get("split", "validation"),
        manifest.get("config", {}), metrics, dataset_hash=manifest.get("dataset_sha256"),
        split_hash=manifest.get("split_sha256"), model_revision=manifest.get("model_revision"),
        code_revision=manifest.get("code_revision"), seed=manifest.get("seed"),
        runtime_seconds=manifest.get("fit_seconds"), artifact_path=str(prediction_path),
        config_hash=manifest.get("config_sha256"),
        description=manifest.get("description", f"{model} experiment on {manifest.get('training_dataset', manifest.get('dataset'))}."))


def _model_cfg(model, config_path=None):
    path = Path(config_path or MODEL_CONFIGS[model])
    if not path.exists(): raise FileNotFoundError(f"model config not found: {path}")
    cfg = _read_yaml(path)
    _strip_obsolete_calibration_config(cfg)
    cfg.setdefault("model", model)
    if cfg["model"] != model: raise ValueError(f"model config {path} declares {cfg['model']!r}, expected {model!r}")
    cfg["_config_path"] = str(path)
    if model not in {"svc", "catboost_stylometry"}:
        cfg.setdefault("inference_batch_size", max(8, int(cfg.get("batch_size", 1))))
    if model not in {"svc", "catboost_stylometry"} and (not cfg.get("checkpoint") or not cfg.get("revision")):
        raise ValueError(f"{model} needs an explicit checkpoint and immutable revision in {path}")
    if model not in {"svc", "catboost_stylometry"} and not re.fullmatch(r"[0-9a-fA-F]{40}", str(cfg["revision"])):
        raise ValueError(f"{model} revision in {path} must be a full 40-character commit hash, not a branch or tag")
    if model == "svc" and float(cfg.get("C", .5)) <= 0:
        raise ValueError(f"SVC config {path} needs C > 0")
    if model == "catboost_stylometry" and (int(cfg.get("iterations", 0)) < 1
            or not 1 <= int(cfg.get("depth", 0)) <= 16 or int(cfg.get("thread_count", 0)) < 1):
        raise ValueError(f"CatBoost config {path} needs positive iterations/thread_count and depth in [1, 16]")
    if model not in {"svc", "catboost_stylometry"} and (int(cfg.get("epochs", 1)) < 1 or int(cfg.get("batch_size", 1)) < 1):
        raise ValueError(f"model config {path} needs positive epochs and batch_size")
    if model not in {"svc", "catboost_stylometry"} and int(cfg.get("inference_batch_size", cfg.get("batch_size", 1))) < 1:
        raise ValueError(f"model config {path} needs positive inference_batch_size")
    if cfg.get("auto_batch_size") and (int(cfg.get("max_batch_size", cfg.get("batch_size", 1))) < 1
            or not 0 < float(cfg.get("batch_probe_safety_fraction", 0.95)) <= 1):
        raise ValueError(f"model config {path} needs max_batch_size >= 1 and batch_probe_safety_fraction in (0, 1]")
    return cfg


def _train_fold(model, train, valid, classes, cfg, output_dir, seed, fold, cache_dir, *,
                cache_fold=None, checkpoint_selection_metric="balanced_accuracy"):
    has_views = "training_views" in train.columns
    training_views = train.training_views.tolist() if has_views else None
    training_view_ids = train.training_view_ids.tolist() if has_views else None
    dynamic_views = has_views and (model == "ultrabert380" or (
        model not in {"svc", "catboost_stylometry"} and
        not (model in FROZEN_BACKBONE_MODELS and cfg.get("training_mode", "frozen") == "frozen")))
    static_sampling = None
    if has_views and not dynamic_views:
        selected, counts = select_single_training_views(training_views, training_view_ids, seed=seed)
        train = train.copy()
        train["text"] = [views[choice] for views, choice in zip(training_views, selected)]
        static_sampling = {"policy": "one deterministic view per original response for single-fit classifier",
                           "seed": int(seed), "epoch": 0, "original_responses": len(train),
                           "selected_view_counts": counts}
    if model == "svc":
        result = train_svc(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
                           c=cfg.get("C", .5), seed=seed,
                           feature_config=cfg.get("features", {"views": ["char", "word"]}),
                           cache_dir=cache_dir, fold_token=cache_fold or f"fold-{fold}",
                           validation_labels=valid.model.tolist(),
                           calibration=cfg.get("calibration"),
                           calibration_cv=cfg.get("calibration_cv", 3))
        checkpoint = Path(output_dir) / "checkpoints" / f"svc_fold{fold}.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_started = time.perf_counter()
        joblib.dump({"model": result["model"], "raw_model": result["raw_model"], "features": result["features"], "classes": classes, "config": cfg}, checkpoint)
        runtime = dict(result.get("runtime_breakdown") or {})
        runtime["checkpoint_serialization_seconds"] = time.perf_counter() - checkpoint_started
        result["runtime_breakdown"] = runtime
        result["checkpoint"] = str(checkpoint)
    elif model == "catboost_stylometry":
        checkpoint = Path(output_dir) / "checkpoints" / f"catboost_stylometry_fold{fold}"
        feature_recipe = cfg.get("feature_recipe")
        feature_config = cfg.get("features") or {
            "views": ["stylometry_raw"], "stylometry_recipe": feature_recipe}
        feature_config = {**feature_config,
                          "stylometry_recipe": feature_config.get("stylometry_recipe", feature_recipe)}
        x_train, x_valid, feature_transform, cache_key = load_or_fit(
            train.text.tolist(), valid.text.tolist(), config=feature_config,
            cache_dir=cache_dir, fold_token=cache_fold or f"fold-{fold}")
        result = train_catboost_stylometry(train.text.tolist(), train.model.tolist(),
            valid.text.tolist(), classes, output_dir=checkpoint,
            iterations=cfg.get("iterations", 800), depth=cfg.get("depth", 7),
            learning_rate=cfg.get("learning_rate", .05), l2_leaf_reg=cfg.get("l2_leaf_reg", 5),
            thread_count=cfg.get("thread_count", 8), seed=seed,
            validation_labels=valid.model.tolist(),
            early_stopping_rounds=cfg.get("early_stopping_rounds", 100),
            feature_recipe=feature_config.get("stylometry_recipe"),
            feature_matrices=(x_train.toarray(), x_valid.toarray(),
                              feature_transform.stylometry_feature_names),
            checkpoint_selection_metric=checkpoint_selection_metric)
        result["cache_key"] = cache_key
        result["checkpoint"] = str(checkpoint)
    elif model == "ultrabert380":
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_encoder_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 20), batch_size=cfg.get("batch_size", 16),
            inference_batch_size=cfg.get("inference_batch_size"), max_length=cfg["max_length"],
            learning_rate=cfg.get("learning_rate", 2e-5), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            target_effective_batch_size=cfg.get("target_effective_batch_size"),
            local_files_only=cfg.get("local_files_only", True),
            validation_labels=valid.model.tolist(),
            learning_rate_decay=cfg.get("learning_rate_decay"),
            optimizer=cfg.get("optimizer", "adamw_torch"),
            weight_decay=cfg.get("weight_decay", 0.0),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
            warmup_ratio=cfg.get("warmup_ratio", 0.0),
            warmup_epochs=cfg.get("warmup_epochs"),
            early_stopping_patience=cfg.get("early_stopping_patience", 2),
            auto_batch_size=cfg.get("auto_batch_size", False),
            max_batch_size=cfg.get("max_batch_size", 1024),
            batch_probe_safety_fraction=cfg.get("batch_probe_safety_fraction", 0.95),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            select_best_checkpoint=(checkpoint_selection_metric == "balanced_accuracy"
                or cfg.get("checkpoint_selection") == "highest_validation_accuracy"),
            checkpoint_selection_metric=checkpoint_selection_metric,
            save_epoch_checkpoints=cfg.get("save_epoch_checkpoints", False),
            trainer_resume_checkpoint=cfg.get("trainer_resume_checkpoint"),
            training_views=training_views, training_view_ids=training_view_ids)
        result["checkpoint"] = str(checkpoint)
    elif model in FROZEN_BACKBONE_MODELS and cfg.get("training_mode", "frozen") == "frozen":
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_frozen_backbone_fold(model, train.text.tolist(), train.model.tolist(),
            valid.text.tolist(), classes, checkpoint=cfg["checkpoint"], revision=cfg["revision"],
            output_dir=checkpoint, max_length=cfg["max_length"],
            batch_size=cfg.get("batch_size", 8),
            inference_batch_size=cfg.get("inference_batch_size", cfg.get("batch_size", 8)),
            c=cfg.get("C", .5), seed=seed,
            fix_mistral_regex=cfg.get("fix_mistral_regex", False),
            validation_labels=valid.model.tolist(),
            train_row_ids=train.id.astype(str).tolist(),
            validation_row_ids=valid.id.astype(str).tolist(),
            embedding_cache_dir=Path(cache_dir).parent / "frozen_embeddings",
            trust_remote_code=cfg.get("trust_remote_code", False))
        result["checkpoint"] = str(checkpoint)
    elif model in ({"qwen06", "qwen17", "qwen4", "qwen8", "minicpm5_2b", "hunyuan_1_8b", "llama32_3b", "llama31_8b", "ministral3_8b", "smollm3_3b", "granite4_h_micro", "granite4_350m", "phi35_mini"}
                   | FROZEN_BACKBONE_MODELS):
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_qwen_lora_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 20), batch_size=cfg.get("batch_size", 4), max_length=cfg["max_length"],
            learning_rate=cfg.get("learning_rate", 1e-4), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            target_effective_batch_size=cfg.get("target_effective_batch_size"),
            lora_rank=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            learning_rate_decay=cfg.get("learning_rate_decay", 1.0),
            auto_batch_size=cfg.get("auto_batch_size", False),
            max_batch_size=cfg.get("max_batch_size", 1024),
            batch_probe_safety_fraction=cfg.get("batch_probe_safety_fraction", 0.95),
            resume_checkpoint=cfg.get("resume_checkpoint"),
            validation_labels=valid.model.tolist(),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            local_files_only=cfg.get("local_files_only", True),
            readout_hidden_size=cfg.get("readout_hidden_size"),
            task_prefix=cfg.get("task_prefix", ""),
            fix_mistral_regex=cfg.get("fix_mistral_regex", False),
            lora_exclude_modules=cfg.get("lora_exclude_modules"),
            early_stopping_patience=cfg.get("early_stopping_patience", 2),
            text_only_backbone=cfg.get("text_only_backbone", False),
            lr_scheduler_type=cfg.get("lr_scheduler_type"),
            warmup_ratio=cfg.get("warmup_ratio", 0.0),
            warmup_epochs=cfg.get("warmup_epochs"),
            optimizer=cfg.get("optimizer", "adamw_torch"),
            weight_decay=cfg.get("weight_decay", 0.0),
            trust_remote_code=cfg.get("trust_remote_code", False),
            quantization=cfg.get("quantization", "none"),
            attention_backend=cfg.get("attention_backend"),
            checkpoint_selection_metric=checkpoint_selection_metric,
            save_epoch_checkpoints=cfg.get("save_epoch_checkpoints", False),
            inference_batch_size=cfg.get("inference_batch_size"),
            training_views=training_views, training_view_ids=training_view_ids)
        result["checkpoint"] = str(checkpoint)
    else: raise ValueError(f"unknown model: {model}")
    if static_sampling is not None:
        result["training_view_sampling"] = static_sampling
    return result


def _dataset_max_length(model, cfg, rows, *, checkpoint=None, revision=None):
    """Count all training and validation responses with the training tokenizer."""
    frozen = model in FROZEN_BACKBONE_MODELS and cfg.get("training_mode", "frozen") == "frozen"
    if "training_views" in rows:
        texts = []
        for row in rows.itertuples(index=False):
            views = getattr(row, "training_views", None)
            if isinstance(views, (list, tuple)):
                texts.extend(str(view) for view in views)
            else:
                texts.append(str(row.text))
        rows = pd.DataFrame({"text": texts})
    return calculate_max_token_length(
        model, rows.text.astype(str).tolist(), checkpoint=checkpoint or cfg["checkpoint"],
        revision=revision if revision is not None else cfg.get("revision"),
        task_prefix=cfg.get("task_prefix", ""), use_processor=frozen,
        trust_remote_code=cfg.get("trust_remote_code", model == "ultrabert380"),
        fix_mistral_regex=cfg.get("fix_mistral_regex", False),
        local_files_only=cfg.get("local_files_only", True))


def _profile_max_length_override(model, cfg, profile):
    """Use the pinned token-window limit for datasets built with that tokenizer."""
    if not profile.get("chunking_strategy"):
        return None
    if model != "ultrabert380":
        return None
    tokenizer_id = profile.get("tokenizer_id")
    tokenizer_revision = profile.get("tokenizer_revision")
    if cfg.get("checkpoint") and cfg["checkpoint"] != tokenizer_id:
        raise ValueError("training checkpoint differs from the token-chunk tokenizer")
    if cfg.get("revision") and cfg["revision"] != tokenizer_revision:
        raise ValueError("training tokenizer revision differs from the token-chunk tokenizer")
    maximum = int(profile["max_encoded_length"])
    observed = int(profile.get("maximum_observed_encoded_length", maximum))
    if observed > maximum:
        raise ValueError(f"chunk profile has encoded length {observed} over its limit {maximum}")
    return maximum


def _resolve_checkpoint_path(path, artifact_dir):
    """Resolve historical manifest checkpoint paths without double-prefixing artifacts."""
    raw = Path(path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([Path(artifact_dir) / raw, Path(artifact_dir).parent / raw])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0]


def _save_epoch_validation_predictions(result, valid, classes, prediction_dir, experiment_id):
    """Persist each epoch's already-computed validation logits and probabilities."""
    epochs = result.get("validation_predictions_by_epoch") or []
    if not epochs:
        return []
    prediction_dir = Path(prediction_dir) / f"{experiment_id}_epoch_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    base = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    records = []
    for item in epochs:
        epoch = int(item["epoch"])
        scores = np.asarray(item["scores"], dtype=np.float64)
        if scores.shape != (len(valid), len(classes)) or not np.isfinite(scores).all():
            raise ValueError(f"epoch {epoch} validation predictions are not aligned to validation rows/classes")
        shifted = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        frame = base.copy()
        frame["validation_epoch"] = epoch
        for index, label in enumerate(classes):
            frame[f"score::{label}"] = scores[:, index]
            frame[f"prob::{label}"] = probabilities[:, index]
        path = prediction_dir / f"epoch-{epoch:03d}.parquet"
        import io
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
        path.write_bytes(payload)
        epoch_checkpoints = result.get("epoch_checkpoints") or []
        checkpoint_path = (epoch_checkpoints[epoch - 1]
                           if epoch <= len(epoch_checkpoints) else None)
        if checkpoint_path is None and int(result.get("best_epoch") or -1) == epoch:
            checkpoint_path = (result.get("selected_checkpoint_path") or
                               result.get("best_checkpoint_path") or result.get("best_checkpoint"))
        if checkpoint_path and not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"epoch {epoch} checkpoint link does not resolve: {checkpoint_path}")
        records.append({"epoch": epoch, "path": str(path), "rows": len(frame),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                        "class_order": list(classes)})
    return records


def _align_saved_validation_predictions(source_manifest, valid, classes, artifact_dir):
    """Return saved source probabilities in exactly the supplied validation order."""
    raw = source_manifest.get("prediction_path")
    if not raw:
        raise ValueError("source checkpoint manifest has no saved validation predictions")
    path = Path(raw)
    if not path.is_file() and not path.is_absolute():
        path = Path(artifact_dir).parent / path
    if not path.is_file():
        raise FileNotFoundError(f"source validation predictions not found: {path}")
    frame = pd.read_parquet(path)
    required = {"id", "prompt_id", "y_true", *(f"prob::{label}" for label in classes)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"saved source predictions are missing columns: {sorted(missing)}")
    if frame.id.astype(str).duplicated().any():
        raise ValueError("saved source predictions contain duplicate IDs")
    expected = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    expected["id"] = expected.id.astype(str)
    expected["prompt_id"] = expected.prompt_id.astype(str)
    expected["y_true"] = expected.y_true.astype(str)
    frame = frame.copy()
    frame["id"] = frame.id.astype(str)
    frame["prompt_id"] = frame.prompt_id.astype(str)
    frame["y_true"] = frame.y_true.astype(str)
    aligned = expected.merge(frame[["id", "prompt_id", "y_true", *(f"prob::{label}" for label in classes)]],
                             on="id", how="left", validate="one_to_one", suffixes=("", "_saved"),
                             indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("source prediction file does not contain every V0 holdout ID")
    for column in ("prompt_id", "y_true"):
        if not aligned[column].astype(str).equals(aligned[f"{column}_saved"].astype(str)):
            raise ValueError(f"source prediction metadata differs from V0 holdout for {column}")
    probabilities = aligned[[f"prob::{label}" for label in classes]].to_numpy(dtype=np.float64)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("saved source validation probabilities are invalid")
    return probabilities, path.resolve()


def kaggle_v0_finetune(*, from_experiment,
                       data_dir="data/processed", artifact_dir="artifacts",
                       epochs=10, learning_rate=2e-6, early_stopping_patience=2,
                       seed=2026):
    """Fine-tune a saved UltraBERT on V0 train rows for a Kaggle-only artifact.

    This deliberately bypasses the general experiment registry, checkpoint catalog,
    prediction factory, and Trust List 2. It initializes model weights from the
    source checkpoint with a fresh optimizer and scheduler.
    """
    if int(epochs) != 10 or float(learning_rate) != 2e-6 or int(early_stopping_patience) != 2:
        raise ValueError("the Kaggle V0 fine-tune is pinned to 10 epochs, LR 2e-6, and patience 2")
    manifests_dir = Path(artifact_dir) / "predictions"
    source_manifest_path = manifests_dir / f"{from_experiment}_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"source experiment manifest not found: {source_manifest_path}")
    source = json.loads(source_manifest_path.read_text())
    if source.get("model") != "ultrabert380" or source.get("training_dataset", source.get("dataset")) != "v2.14":
        raise ValueError("source must be a full V2.14 UltraBERT checkpoint")
    if int(source.get("validation_rows", 0)) != 2699:
        raise ValueError("source checkpoint must have been selected using all 2,699 V0 rows")
    raw_checkpoint = source.get("best_checkpoint_path") or source.get("selected_checkpoint_path") or source.get("checkpoint_path")
    if not raw_checkpoint:
        raise FileNotFoundError("source manifest has no selected checkpoint path")
    source_checkpoint = _resolve_checkpoint_path(raw_checkpoint, artifact_dir)
    nested_best = source_checkpoint / "best_checkpoint"
    if nested_best.is_dir():
        source_checkpoint = nested_best
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")

    train, valid = _training_partitions("v0", "v0", data_dir)
    if len(train) != 2159 or len(valid) != 540:
        raise ValueError(f"expected the frozen V0 2159/540 split, got {len(train)}/{len(valid)}")
    classes = sorted(train.model.astype(str).unique().tolist())
    if source.get("classes") != classes:
        raise ValueError("source checkpoint class ordering differs from V0")
    source_probabilities, source_prediction_path = _align_saved_validation_predictions(
        source, valid, classes, artifact_dir)
    train_hash, valid_hash = _partition_sha256(train), _partition_sha256(valid)
    split_hash = _sha(["kaggle_v0_finetune", train_hash, valid_hash])
    inherited = source.get("config", {})
    config = {
        "model": "ultrabert380", "dataset": "v0", "training_dataset": "v0",
        "validation_dataset": "v0", "initialization": "weights_only_fresh_optimizer",
        "source_experiment": from_experiment, "source_checkpoint": str(source_checkpoint),
        "epochs": 10, "early_stopping_patience": 2, "learning_rate": 2e-6,
        "lr_scheduler_type": "linear", "warmup_ratio": 0.0,
        "optimizer": inherited.get("optimizer", "adamw_torch"),
        "weight_decay": float(inherited.get("weight_decay", 0.0)),
        "batch_size": int(inherited.get("batch_size", 8)),
        "gradient_accumulation_steps": int(inherited.get("gradient_accumulation_steps", 1)),
        "target_effective_batch_size": inherited.get("target_effective_batch_size"),
        "auto_batch_size": bool(inherited.get("auto_batch_size", True)),
        "max_batch_size": int(inherited.get("max_batch_size", 1024)),
        "batch_probe_safety_fraction": float(inherited.get("batch_probe_safety_fraction", 0.95)),
        "gradient_checkpointing": bool(inherited.get("gradient_checkpointing", True)),
        "max_length": int(inherited.get("max_length", 581)),
        "checkpoint_selection_metric": "balanced_accuracy",
        "checkpoint_selection": "highest_validation_balanced_accuracy",
        "selection_split": "V0 prompt-grouped validation holdout",
        "seed": int(seed),
    }
    config_hash = _sha(config)
    experiment_id = (f"kaggle_only_ultrabert380_v0_from{hashlib.sha256(from_experiment.encode()).hexdigest()[:8]}_"
                    f"split{split_hash[:8]}_cfg{config_hash[:8]}")
    root = Path(artifact_dir) / "kaggle_only"
    run_dir = root / "runs" / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"Kaggle-only run already exists: {run_dir}")
    checkpoint_output = run_dir / "checkpoints" / "ultrabert380_v0_finetuned"
    print(f"Starting Kaggle-only V0 fine-tune {experiment_id}: train={len(train)}, validation={len(valid)}", flush=True)
    result = train_encoder_fold(
        train.text.astype(str).tolist(), train.model.astype(str).tolist(),
        valid.text.astype(str).tolist(), classes,
        checkpoint=source.get("checkpoint_id") or str(source_checkpoint),
        revision=None, resume_checkpoint=source_checkpoint, output_dir=checkpoint_output,
        epochs=10, batch_size=config["batch_size"],
        inference_batch_size=max(8, config["batch_size"]), max_length=config["max_length"],
        learning_rate=2e-6, learning_rate_decay=None, seed=seed,
        lr_scheduler_type="linear", warmup_ratio=0.0,
        optimizer=config["optimizer"], weight_decay=config["weight_decay"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        target_effective_batch_size=config["target_effective_batch_size"],
        auto_batch_size=config["auto_batch_size"], max_batch_size=config["max_batch_size"],
        batch_probe_safety_fraction=config["batch_probe_safety_fraction"],
        gradient_checkpointing=config["gradient_checkpointing"],
        validation_labels=valid.model.astype(str).tolist(), select_best_checkpoint=True,
        local_files_only=True, early_stopping_patience=2,
        checkpoint_selection_metric="balanced_accuracy", copy_resume_state=False)

    new_probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    if new_probabilities.shape != source_probabilities.shape:
        raise ValueError("fine-tuned predictions do not align with source holdout predictions")
    y_true = valid.model.astype(str).to_numpy()
    source_metrics = evaluate(y_true, source_probabilities, classes, source_probabilities)
    new_metrics = evaluate(y_true, new_probabilities, classes, new_probabilities)
    source_pred = np.asarray(classes)[source_probabilities.argmax(axis=1)]
    new_pred = np.asarray(classes)[new_probabilities.argmax(axis=1)]
    source_metrics["accuracy"] = float(np.mean(source_pred == y_true))
    new_metrics["accuracy"] = float(np.mean(new_pred == y_true))
    for metrics in (source_metrics, new_metrics):
        for key in [name for name in metrics if str(name).startswith("calibrated_")]:
            metrics.pop(key, None)
    corrected = (source_pred != y_true) & (new_pred == y_true)
    regressed = (source_pred == y_true) & (new_pred != y_true)
    output_predictions = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    output_predictions["source_prediction"] = source_pred
    output_predictions["finetuned_prediction"] = new_pred
    output_predictions["source_correct"] = source_pred == y_true
    output_predictions["finetuned_correct"] = new_pred == y_true
    for idx, label in enumerate(classes):
        output_predictions[f"source_prob::{label}"] = source_probabilities[:, idx]
        output_predictions[f"prob::{label}"] = new_probabilities[:, idx]
    output_predictions_path = run_dir / "v0_holdout_predictions.parquet"
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_predictions.to_parquet(output_predictions_path, index=False)
    epoch_predictions = _save_epoch_validation_predictions(
        result, valid, classes, run_dir / "epoch_predictions", experiment_id)
    best_checkpoint = Path(result["best_checkpoint_path"]).resolve()
    manifest = {
        "experiment_id": experiment_id, "experiment_name": "Kaggle-only V0 UltraBERT fine-tune",
        "scope": "kaggle_only_not_in_trust_list_2", "model": "ultrabert380",
        "dataset": "v0", "training_dataset": "v0", "validation_dataset": "v0",
        "train_rows": len(train), "validation_rows": len(valid),
        "train_id_sha256": hashlib.sha256("\n".join(train.id.astype(str)).encode()).hexdigest(),
        "validation_id_sha256": hashlib.sha256("\n".join(valid.id.astype(str)).encode()).hexdigest(),
        "training_dataset_sha256": train_hash, "validation_dataset_sha256": valid_hash,
        "split_sha256": split_hash, "classes": classes, "config": config,
        "config_sha256": config_hash, "seed": int(seed),
        "source_experiment": from_experiment,
        "source_checkpoint_path": str(source_checkpoint),
        "source_prediction_path": str(source_prediction_path),
        "source_validation_metrics": source_metrics, "validation_metrics": new_metrics,
        "validation_metrics_by_epoch": result.get("validation_metrics_by_epoch", []),
        "validation_predictions_by_epoch": epoch_predictions,
        "selected_epoch": result.get("best_epoch"),
        "selected_validation_balanced_accuracy": result.get("best_validation_accuracy"),
        "checkpoint_selection_metric": "balanced_accuracy",
        "best_checkpoint_path": str(best_checkpoint),
        "checkpoint_path": str(best_checkpoint),
        "prediction_path": str(output_predictions_path),
        "corrected_rows": int(corrected.sum()), "regressed_rows": int(regressed.sum()),
        "corrected_ids": output_predictions.loc[corrected, "id"].astype(str).tolist(),
        "regressed_ids": output_predictions.loc[regressed, "id"].astype(str).tolist(),
        "early_stopping": result.get("early_stopping"),
        "fit_seconds": result.get("fit_seconds"),
        "batch_size_probe": result.get("batch_size_probe"),
        "effective_batch_size": result.get("effective_batch_size"),
        "selected_physical_batch_size": result.get("selected_physical_batch_size"),
        "gradient_accumulation_steps": result.get("gradient_accumulation_steps"),
        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
        "code_revision": _revision(),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "comparison_note": "Development diagnostic: source checkpoint selection used all V0, including these 540 rows.",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    _release_completed_training_model(result)
    return manifest


def build_features(dataset="v1", data_dir="data/processed", config_path="configs/features/default.yaml", cache_dir="artifacts/features", *, validation_dataset="v0"):
    train, valid = _training_partitions(dataset, validation_dataset, data_dir)
    cfg = _read_yaml(config_path)
    x_train, x_val, features, key = load_or_fit(
        train.text, valid.text, config=cfg, cache_dir=cache_dir,
        fold_token=f"{dataset}-{_partition_sha256(train)}-{validation_dataset}-{_partition_sha256(valid)}")
    report = [{"cache_key": key, "train_shape": list(x_train.shape),
               "validation_shape": list(x_val.shape)}]
    output = Path(cache_dir) / f"{dataset}_to_{validation_dataset}_features_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"training_dataset": dataset, "validation_dataset": validation_dataset,
        "training_rows": len(train), "validation_rows": len(valid), "feature_config": cfg,
        "split": "explicit_datasets", "partitions": report}, indent=2) + "\n")
    return report


def train_models(dataset="v1", model="svc", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, config_path=None, feature_config_path=None, seed=2026, folds=None, config_overrides=None, refresh_prediction=False, refresh_checkpoint_catalog=False, validation_dataset="v0", max_length=None):
    """Train on every row in dataset and validate on every row in validation_dataset."""
    if folds is not None:
        raise ValueError("fold selection was removed; specify explicit training and validation datasets")
    cfg = _model_cfg(model, config_path)
    if config_overrides:
        cfg.update(config_overrides)
        _strip_obsolete_calibration_config(cfg)
        _validate_training_overrides(model, cfg)
    if model in {"svc", "catboost_stylometry"} and feature_config_path:
        cfg["features"] = _read_yaml(feature_config_path)
    if max_length is not None:
        max_length = int(max_length)
        if max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if model in {"svc", "catboost_stylometry"}:
            raise ValueError("--max-length applies to tokenizer-based models, not SVC or CatBoost")
        cfg["max_length"] = max_length
    train, valid = _training_partitions(dataset, validation_dataset, data_dir)
    classes = sorted(train.model.astype(str).unique().tolist())
    train_profile = json.loads(_profile_path(Path(data_dir), dataset).read_text())
    if model not in {"svc", "catboost_stylometry"}:
        if max_length is None:
            profile_max_length = _profile_max_length_override(model, cfg, train_profile)
            if profile_max_length is not None:
                cfg["max_length"] = profile_max_length
            else:
                cfg["max_length"] = _dataset_max_length(model, cfg, pd.concat([train, valid], ignore_index=True))
    train_hash = _partition_sha256(train)
    valid_hash = _partition_sha256(valid)
    split_hash = _sha({"training_dataset": dataset, "training_sha256": train_hash,
                       "validation_dataset": validation_dataset, "validation_sha256": valid_hash})
    profile = {"dataset_sha256": train_profile.get("dataset_sha256", train_hash),
               "split_sha256": split_hash,
               "validation_ids_sha256": hashlib.sha256("\n".join(sorted(valid.id.astype(str))).encode()).hexdigest(),
               "validation_prompt_ids_sha256": hashlib.sha256("\n".join(sorted(valid.prompt_id.astype(str))).encode()).hexdigest(),
               "validation_normalized_text_sha256": hashlib.sha256("\n".join(sorted(valid.text.astype(str).map(_normal_text))).encode()).hexdigest(),
               "validation_prompt_content_sha256": None}
    code_revision = _revision()
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_hash = _sha(config_snapshot)
    experiment_id = f"{model}_{dataset}_val{validation_dataset}_data{train_hash[:8]}_split{split_hash[:8]}_seed{seed}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    run_artifact_dir = Path(artifact_dir) / "runs" / experiment_id
    pred_dir = Path(artifact_dir) / "predictions"
    prediction_path = pred_dir / f"{experiment_id}_validation.parquet"
    manifest_path = pred_dir / f"{experiment_id}_manifest.json"
    if manifest_path.exists() and prediction_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _register_completed_manifest(manifest, registry_path)
        _append_v2_checkpoint_trust(manifest, manifest_path)
        return manifest
    if run_artifact_dir.exists():
        raise FileExistsError(f"incomplete immutable experiment exists at {run_artifact_dir}; remove that generated run directory to retry")
    metric = "balanced_accuracy"
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    console_log_path = run_artifact_dir / "training_console.log"
    with console_log_path.open("a", encoding="utf-8", buffering=1) as console_log:
        stdout = _TeeTextStream(sys.stdout, console_log)
        stderr = _TeeTextStream(sys.stderr, console_log)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print("TRAINING_RUN " + json.dumps({"pid": os.getpid(),
                "model": model, "dataset": dataset, "started_at": time.time()}), flush=True)
            result = _train_fold(model, train, valid, classes, cfg, run_artifact_dir, seed, "validation", Path(artifact_dir)/"features",
                                 cache_fold=f"{dataset}-{train_hash}-{validation_dataset}-{valid_hash}",
                                 checkpoint_selection_metric=metric)
    training_view_sampling = result.get("training_view_sampling")
    if training_view_sampling is not None:
        sampling_path = run_artifact_dir / "training_view_sampling.json"
        sampling_path.write_text(json.dumps({"training_dataset": dataset,
            "training_dataset_sha256": train_hash, "sampling": training_view_sampling}, indent=2) + "\n")
    export_started = time.perf_counter()
    post_training_started = export_started
    runtime_breakdown = _post_training_runtime(result)
    scores, probabilities = result["scores"], result["probabilities"]
    if (not np.isfinite(scores).all() or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise RuntimeError("V0 validation output is incomplete or non-finite")
    metrics = evaluate(valid.model.astype(str), scores, classes, probabilities)
    if result.get("validation_loss_raw") is not None:
        metrics["validation_loss_raw"] = float(result["validation_loss_raw"])
    for key in [name for name in metrics if str(name).startswith("calibrated_")]:
        metrics.pop(key, None)
    pred = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for i, label in enumerate(classes):
        pred[f"score::{label}"] = scores[:, i]
        pred[f"prob::{label}"] = probabilities[:, i]
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_export_started = time.perf_counter()
    pred.to_parquet(prediction_path, index=False)
    epoch_prediction_records = _save_epoch_validation_predictions(
        result, valid, classes, pred_dir, experiment_id)
    runtime_breakdown["prediction_export_seconds"] = time.perf_counter() - prediction_export_started
    checkpoint = Path(result["checkpoint"])
    artifact_size = sum(f.stat().st_size for f in checkpoint.rglob("*") if f.is_file()) if checkpoint.is_dir() else checkpoint.stat().st_size
    validation_ids = sorted(valid.id.astype(str))
    manifest_export_started = time.perf_counter()
    manifest = {"experiment_id": experiment_id, "experiment_name": config_snapshot.get("experiment_name"),
        "model": model, "dataset": dataset,
        "split": "validation", "training_dataset": dataset, "validation_dataset": validation_dataset,
        "training_dataset_rows": len(train), "validation_dataset_rows": len(valid),
        "training_dataset_sha256": train_hash, "validation_dataset_sha256": valid_hash,
        "checkpoint_selection_scope": validation_dataset,
        "classes": classes, "train_rows": len(train), "validation_rows": len(valid),
        "train_id_sha256": hashlib.sha256("\n".join(train.id.astype(str)).encode()).hexdigest(),
        "id_sha256": hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),
        "checkpoint_selection_rows": len(valid),
        "checkpoint_selection_ids_sha256": hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),
        "checkpoint_selection_prompt_groups": int(valid.prompt_id.nunique()),
        "checkpoint_selection_prompt_ids_sha256": profile.get("validation_prompt_ids_sha256"),
        "checkpoint_selection_normalized_text_sha256": profile.get("validation_normalized_text_sha256"),
        "checkpoint_selection_prompt_content_sha256": profile.get("validation_prompt_content_sha256"),
        "dataset_sha256": profile["dataset_sha256"], "split_sha256": profile["split_sha256"],
        "code_revision": code_revision, "model_revision": cfg.get("revision"), "checkpoint_id": cfg.get("checkpoint"),
        "config": config_snapshot, "config_sha256": config_hash, "seed": seed,
        "quantization": result.get("quantization", cfg.get("quantization", "none")),
        "quantization_config": result.get("quantization_config"),
        "attention_backend": result.get("attention_backend", cfg.get("attention_backend")),
        "fit_seconds": float(result["fit_seconds"]), "inference_seconds": result.get("inference_seconds"),
        "calibration_method": result.get("calibration_method"),
        "runtime_breakdown": runtime_breakdown,
        "training_rows_per_second": (len(train) / float(runtime_breakdown.get(
            "training_excluding_validation_seconds") or result.get("fit_seconds") or 1.0)),
        "training_view_sampling": training_view_sampling,
        "training_view_sampling_artifact": (str(run_artifact_dir / "training_view_sampling.json")
                                             if training_view_sampling is not None else None),
        "inference_rows_per_second": (len(valid) / result["inference_seconds"]
            if result.get("inference_seconds") else None),
        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
        "peak_gpu_memory_reserved_bytes": result.get("peak_gpu_memory_reserved_bytes"),
        "checkpoint_size_bytes": artifact_size,
        "validation_loss_raw": result.get("validation_loss_raw"),
        "requested_batch_size": result.get("requested_batch_size"),
        "effective_batch_size": result.get("effective_batch_size"),
        "selected_physical_batch_size": result.get("selected_physical_batch_size",
            result.get("effective_batch_size")),
        "effective_optimizer_batch_size": result.get("effective_optimizer_batch_size"),
        "target_effective_batch_size": cfg.get("target_effective_batch_size"),
        "gradient_accumulation_steps": result.get("gradient_accumulation_steps",
            cfg.get("gradient_accumulation_steps", 1)),
        "oom_recovery": result.get("oom_recovery"),
        "requested_inference_batch_size": result.get("requested_inference_batch_size", cfg.get("inference_batch_size")),
        "effective_inference_batch_size": result.get("effective_inference_batch_size"),
        "batch_size_probe": result.get("batch_size_probe"),
        "lr_scheduler_type": result.get("lr_scheduler_type", cfg.get("lr_scheduler_type", "epoch_decay")),
        "warmup_ratio": result.get("warmup_ratio", cfg.get("warmup_ratio", 0.0)),
        "validation_metrics_by_epoch": result.get("validation_metrics_by_epoch", []),
        "validation_predictions_by_epoch": epoch_prediction_records,
        "validation_metrics_by_iteration": result.get("validation_metrics_by_iteration", []),
        "validation_metrics": metrics,
        "checkpoint_selection_metric": result.get("checkpoint_selection_metric") or
            ("single_fit_no_epoch_selection" if model in {"svc"} or
             (model in FROZEN_BACKBONE_MODELS and cfg.get("training_mode", "frozen") == "frozen")
             else f"validation_{metric}"),
        "selected_validation_balanced_accuracy": result.get("best_validation_balanced_accuracy",
            result.get("selected_validation_balanced_accuracy", metrics.get("balanced_accuracy"))),
        "early_stopping_scope": validation_dataset,
        "early_stopping": result.get("early_stopping"),
        "epoch_checkpoints": result.get("epoch_checkpoints", []),
        "selected_epoch": result.get("best_epoch"),
        "selected_iteration": result.get("selected_iteration"),
        "selected_validation_accuracy": result.get("best_validation_accuracy",
            result.get("selected_validation_accuracy", metrics.get("accuracy"))),
        "baseline_validation_metrics": result.get("baseline_validation_metrics"),
        "checkpoint_scoring_method": result.get("checkpoint_scoring_method"),
        "final_validation_reused_existing_predictions": result.get("final_validation_reused_existing_predictions",
            result.get("final_validation_reused_epoch_predictions")),
        "feature_count": result.get("feature_count"),
        "feature_names": result.get("feature_names"),
        "feature_recipe": result.get("feature_recipe", cfg.get("feature_recipe")),
        "feature_cache_key": result.get("cache_key"), "checkpoint_path": str(checkpoint),
        "selected_checkpoint_path": result.get("selected_checkpoint_path") or
            result.get("best_checkpoint_path") or str(checkpoint),
        "best_checkpoint_path": result.get("best_checkpoint_path") or
            result.get("selected_checkpoint_path") or str(checkpoint),
        "prediction_path": str(prediction_path), "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "validation_dataset_provenance": {"name": validation_dataset, "rows": len(valid),
            "sha256": valid_hash, "ids_sha256": profile.get("validation_ids_sha256"),
            "prompt_groups": int(valid.prompt_id.nunique())},
        "metrics": metrics, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    experiment_name = config_snapshot.get("experiment_name")
    description = (f"{experiment_name}: " if experiment_name else "") + (
        f"{model} trained on all {len(train)} {dataset} rows; validation and checkpoint selection used all "
        f"{len(valid)} {validation_dataset} rows (balanced accuracy).")
    manifest["description"] = description
    _register_completed_manifest(manifest, registry_path)
    runtime_breakdown["manifest_prediction_export_seconds"] = time.perf_counter() - export_started
    runtime_breakdown["manifest_export_seconds"] = time.perf_counter() - manifest_export_started
    manifest["runtime_breakdown"] = runtime_breakdown
    (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _release_completed_training_model(result)
    if refresh_prediction or refresh_checkpoint_catalog:
        if validation_dataset == "v0":
            _refresh_ensemble_after_training(manifest, manifest_path,
                artifact_dir=artifact_dir, data_dir=data_dir,
                refresh_prediction=refresh_prediction,
                refresh_checkpoint_catalog=refresh_checkpoint_catalog)
        else:
            manifest["prediction_factory_update"] = {"status": "skipped",
                "reason": "downstream rosters and catalog require V0 validation rows"}
            manifest["checkpoint_catalog_update"] = {"status": "skipped",
                "reason": "downstream rosters and catalog require V0 validation rows"}
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    else:
        manifest["prediction_factory_update"] = {"status": "skipped", "reason": "disabled"}
        manifest["checkpoint_catalog_update"] = {"status": "skipped", "reason": "disabled"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runtime_breakdown["total_post_training_seconds"] = (
        float(runtime_breakdown.get("model_post_training_seconds", 0.0)) +
        time.perf_counter() - post_training_started)
    manifest["runtime_breakdown"] = runtime_breakdown
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    _append_v2_checkpoint_trust(manifest, manifest_path)
    return manifest


TRAINING_OVERRIDE_FIELDS = (
    "epochs", "early_stopping_patience", "learning_rate", "learning_rate_decay", "lr_scheduler_type", "warmup_ratio",
    "optimizer", "weight_decay", "batch_size", "gradient_accumulation_steps",
    "target_effective_batch_size",
    "lora_rank", "lora_alpha", "lora_target_modules", "quantization", "attention_backend", "C",
    "calibration", "calibration_cv",
    "iterations", "depth", "l2_leaf_reg", "early_stopping_rounds", "inference_batch_size", "save_epoch_checkpoints",
    "auto_batch_size", "max_batch_size", "batch_probe_safety_fraction", "gradient_checkpointing",
    "trainer_resume_checkpoint",
)

CONTINUATION_CONFIG_FIELDS = set(TRAINING_OVERRIDE_FIELDS) | {
    "auto_batch_size", "max_batch_size", "batch_probe_safety_fraction",
    "target_effective_batch_size",
    "gradient_checkpointing", "task_prefix", "text_only_backbone",
    "trust_remote_code", "fix_mistral_regex", "readout_hidden_size",
    "lora_exclude_modules", "resume_checkpoint", "trainer_resume_checkpoint",
}


def _continuation_config(model, parent_config=None, custom_config_path=None, explicit_overrides=None):
    """Apply continuation recipe precedence: defaults, parent, explicit YAML, CLI/UI."""
    cfg = _model_cfg(model)
    cfg.update({key: value for key, value in (parent_config or {}).items()
                if key in CONTINUATION_CONFIG_FIELDS})
    if custom_config_path:
        custom = _read_yaml(custom_config_path)
        if custom.get("model", model) != model:
            raise ValueError(f"continuation config must declare model {model!r}")
        cfg.update({key: value for key, value in custom.items() if not key.startswith("_")})
        cfg["_config_path"] = str(custom_config_path)
    cfg.update(explicit_overrides or {})
    _strip_obsolete_calibration_config(cfg)
    cfg["model"] = model
    _validate_training_overrides(model, cfg)
    return cfg


def _matching_continuation_source(model, dataset, validation_dataset, data_dir,
                                  artifact_dir, from_experiment=None, *,
                                  allow_cross_dataset_source=False):
    train, valid = _training_partitions(dataset, validation_dataset, data_dir)
    profile = json.loads(_profile_path(Path(data_dir), dataset).read_text())
    candidates = []
    manifests_dir = Path(artifact_dir) / "predictions"
    if allow_cross_dataset_source and not from_experiment:
        raise ValueError("cross-dataset continuation requires an explicit --from-experiment source")
    source_paths = ([manifests_dir / f"{from_experiment}_manifest.json"]
                    if allow_cross_dataset_source else
                    manifests_dir.glob(f"{model}_{dataset}_*_manifest.json"))
    for path in source_paths:
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        if (record.get("model") == model and _continuation_source_matches(
                record, dataset, validation_dataset, train, valid,
                profile.get("dataset_sha256"),
                allow_cross_dataset_source=allow_cross_dataset_source)):
            candidates.append((path, record))
    if from_experiment:
        candidates = [(path, item) for path, item in candidates
                      if item.get("experiment_id") == from_experiment]
    if not candidates:
        raise FileNotFoundError(f"no matching {model} validation run found to continue")
    return max(candidates, key=lambda item: item[1].get("created_at_utc", ""))


def _validate_training_overrides(model, cfg):
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "learning_rate_decay"):
        if name in cfg and cfg[name] is not None and float(cfg[name]) < 0:
            raise ValueError(f"{name} must not be negative")
    for name in ("epochs", "early_stopping_patience", "batch_size", "gradient_accumulation_steps", "lora_rank", "lora_alpha", "max_length"):
        if name in cfg and int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("inference_batch_size",):
        if name in cfg and int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg.get("target_effective_batch_size") is not None and int(cfg["target_effective_batch_size"]) < 1:
        raise ValueError("target_effective_batch_size must be positive")
    if "max_batch_size" in cfg and int(cfg["max_batch_size"]) < 1:
        raise ValueError("max_batch_size must be positive")
    if "batch_probe_safety_fraction" in cfg and not 0 < float(cfg["batch_probe_safety_fraction"]) <= 1:
        raise ValueError("batch_probe_safety_fraction must be in (0, 1]")
    if cfg.get("lr_scheduler_type") not in {None, "linear", "cosine", "constant", "epoch_decay"}:
        raise ValueError("lr_scheduler_type must be linear, cosine, constant, or epoch_decay")
    if cfg.get("quantization", "none") not in {"none", "qlora4"}:
        raise ValueError("quantization must be 'none' or 'qlora4'")
    if cfg.get("attention_backend") not in {None, "sdpa", "flash_attention_2"}:
        raise ValueError("attention_backend must be 'sdpa', 'flash_attention_2', or unset")
    if cfg.get("optimizer") == "paged_adamw_8bit" and cfg.get("quantization", "none") != "qlora4":
        raise ValueError("paged_adamw_8bit is reserved for qlora4 configurations")
    if model == "svc" and float(cfg.get("C", 0.5)) <= 0:
        raise ValueError("SVC overrides need C > 0")
    if model == "catboost_stylometry" and (int(cfg.get("iterations", 0)) < 1
            or not 1 <= int(cfg.get("depth", 0)) <= 16
            or int(cfg.get("early_stopping_rounds", 1)) < 1):
        raise ValueError("CatBoost overrides need iterations > 0, depth in [1, 16], and early_stopping_rounds > 0")


def _confirm_training_config(model, cfg, *, accept_defaults=False):
    """Require an explicit choice before a CLI training run starts."""
    fields = [key for key in TRAINING_OVERRIDE_FIELDS if key in cfg]
    print(f"Training configuration for {model}:")
    for key in fields:
        print(f"  {key}: {cfg[key]}")
    if accept_defaults:
        print("Using the configured values (--accept-config-defaults).")
        return {}
    if not sys.stdin.isatty():
        raise RuntimeError("Training requires a config choice. Re-run interactively or pass --accept-config-defaults to affirm these values.")
    answer = input("Override any values before training? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Using configured values.")
        return {}
    overrides = {}
    print("Enter a replacement value for each setting to change; press Enter to keep the shown value.")
    for key in fields:
        raw = input(f"{key} [{cfg[key]}]: ").strip()
        if raw:
            overrides[key] = yaml.safe_load(raw)
    updated = {**cfg, **overrides}
    _validate_training_overrides(model, updated)
    print("Training with overridden values:")
    for key in fields:
        print(f"  {key}: {updated[key]}")
    return overrides


def continue_qwen(dataset="v1", data_dir="data/processed", artifact_dir="artifacts",
                  registry_path="experiments/registry.sqlite", *, from_experiment=None, epochs=None, seed=2026,
                  config_path=None, config_overrides=None, model="qwen06", validation_dataset="v0",
                  refresh_prediction=False, refresh_checkpoint_catalog=False):
    """Continue a saved Qwen LoRA model and validate after every additional epoch."""
    from .model_factory import train_qwen_lora_fold
    if model not in {"qwen06", "qwen17", "qwen4", "qwen8"}:
        raise ValueError("continue-qwen supports qwen06, qwen17, qwen4, and qwen8")
    train, valid = _training_partitions(dataset, validation_dataset, data_dir)
    training_views = train.training_views.tolist() if "training_views" in train.columns else None
    training_view_ids = train.training_view_ids.tolist() if "training_view_ids" in train.columns else None
    rows = pd.concat([train, valid], ignore_index=True)
    train_hash, valid_hash = _partition_sha256(train), _partition_sha256(valid)
    training_profile = json.loads(_profile_path(Path(data_dir), dataset).read_text())
    profile = {"dataset_sha256": train_hash, "split_sha256": _sha([dataset, train_hash, validation_dataset, valid_hash])}
    manifests_dir = Path(artifact_dir)/"predictions"
    source_path, source = _matching_continuation_source(
        model, dataset, validation_dataset, data_dir, artifact_dir, from_experiment)
    source_checkpoint = _resolve_checkpoint_path(source.get("best_checkpoint_path") or
                                                  source["checkpoint_path"], artifact_dir)
    if not source_checkpoint.is_dir(): raise FileNotFoundError(f"Qwen continuation checkpoint not found: {source_checkpoint}")
    explicit = dict(config_overrides or {})
    if epochs is not None:
        explicit["epochs"] = epochs
    cfg = _continuation_config(model, source.get("config", {}), config_path, explicit)
    epochs = int(cfg.get("epochs", 4))
    # A continuation must use the exact base model revision that produced its
    # adapter, even if the local model config has since changed.
    cfg["checkpoint"] = source.get("checkpoint_id", cfg["checkpoint"])
    cfg["revision"] = source.get("model_revision", cfg["revision"])
    source_quantization = source.get("quantization", source.get("config", {}).get("quantization", "none"))
    configured_quantization = cfg.get("quantization", "none")
    if source_quantization != configured_quantization:
        raise ValueError(f"continuation quantization mismatch: source={source_quantization}, config={configured_quantization}")
    cfg["max_length"] = _dataset_max_length(model, cfg, rows)
    if epochs < 1: raise ValueError("additional epochs must be positive")
    cfg["epochs"] = epochs
    cfg["validation_dataset"] = validation_dataset
    cfg["checkpoint_selection_metric"] = "balanced_accuracy"
    snapshot = {k:v for k,v in cfg.items() if not k.startswith("_")}
    if training_views is not None:
        snapshot["training_view_sampling_policy"] = "one random view per original response per epoch"
    snapshot["continued_from_experiment"] = source["experiment_id"]
    snapshot["additional_epochs"] = epochs
    config_hash = _sha(snapshot); code_revision = _revision()
    source_key = hashlib.sha256(source["experiment_id"].encode()).hexdigest()[:8]
    experiment_id = f"{model}_{dataset}_continue{epochs}_from{source_key}_seed{seed}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    run_dir = Path(artifact_dir)/"runs"/experiment_id
    if run_dir.exists(): raise FileExistsError(f"continuation run already exists: {run_dir}; choose a new seed or clean generated run")
    checkpoint = run_dir/"checkpoints"/f"{model}_continued"
    result = train_qwen_lora_fold(train.text.tolist(),train.model.tolist(),valid.text.tolist(),
        sorted(rows.model.astype(str).unique()),checkpoint=cfg["checkpoint"],revision=cfg["revision"],
        output_dir=checkpoint,epochs=epochs,batch_size=cfg.get("batch_size",4),max_length=cfg["max_length"],
        inference_batch_size=cfg.get("inference_batch_size"),
        learning_rate=cfg.get("learning_rate",1e-4),seed=seed,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",1),
        target_effective_batch_size=cfg.get("target_effective_batch_size"),
        lora_rank=cfg.get("lora_rank",16),lora_alpha=cfg.get("lora_alpha",32),
        lora_target_modules=cfg.get("lora_target_modules",["q_proj","k_proj","v_proj","o_proj"]),
        auto_batch_size=cfg.get("auto_batch_size",False),
        max_batch_size=cfg.get("max_batch_size",cfg.get("batch_size",4)),
        batch_probe_safety_fraction=cfg.get("batch_probe_safety_fraction",0.95),
        resume_checkpoint=source_checkpoint,validation_labels=valid.model.tolist(),
        local_files_only=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing",False),
        learning_rate_decay=cfg.get("learning_rate_decay",1.0),
        lr_scheduler_type=cfg.get("lr_scheduler_type"),
        warmup_ratio=cfg.get("warmup_ratio",0.0),
        optimizer=cfg.get("optimizer", "adamw_torch"),
        weight_decay=cfg.get("weight_decay", 0.0),
        early_stopping_patience=cfg.get("early_stopping_patience", 2),
        quantization=cfg.get("quantization", "none"),
        attention_backend=cfg.get("attention_backend"),
        checkpoint_selection_metric="balanced_accuracy",
        save_epoch_checkpoints=cfg.get("save_epoch_checkpoints", False),
        training_views=training_views, training_view_ids=training_view_ids)
    training_view_sampling = result.get("training_view_sampling")
    sampling_path = None
    if training_view_sampling is not None:
        sampling_path = run_dir / "training_view_sampling.json"
        sampling_path.write_text(json.dumps({"training_dataset": dataset,
            "training_dataset_sha256": train_hash, "sampling": training_view_sampling}, indent=2) + "\n")
    classes=sorted(rows.model.astype(str).unique())
    metrics=evaluate(valid.model.astype(str),result["scores"],classes,result["probabilities"])
    if result.get("validation_loss_raw") is not None:
        metrics["validation_loss_raw"] = float(result["validation_loss_raw"])
    for key in [name for name in metrics if str(name).startswith("calibrated_")]:
        metrics.pop(key, None)
    metrics["validation_metrics_by_epoch"]=result["validation_metrics_by_epoch"]
    export_started = time.perf_counter()
    post_training_started = export_started
    runtime_breakdown = _post_training_runtime(result)
    pred=valid[["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy()
    for i,label in enumerate(classes): pred[f"score::{label}"]=result["scores"][:,i];pred[f"prob::{label}"]=result["probabilities"][:,i]
    pred_dir=Path(artifact_dir)/"predictions";pred_dir.mkdir(parents=True,exist_ok=True)
    prediction_path=pred_dir/f"{experiment_id}_validation.parquet"
    prediction_export_started=time.perf_counter();pred.to_parquet(prediction_path,index=False)
    epoch_prediction_records = _save_epoch_validation_predictions(
        result, valid, classes, pred_dir, experiment_id)
    runtime_breakdown["prediction_export_seconds"]=time.perf_counter()-prediction_export_started
    manifest_export_started=time.perf_counter()
    manifest={"experiment_id":experiment_id,"model":model,"dataset":dataset,"training_dataset":dataset,
        "validation_dataset":validation_dataset,"split":"validation",
        "continued_from_experiment":source["experiment_id"],"continued_from_checkpoint":str(source_checkpoint),
        "additional_epochs":epochs,"total_epochs":int(source.get("config",{}).get("epochs",0))+epochs,
        "train_rows":len(train),"validation_rows":len(valid),"training_dataset_rows":len(train),
        "validation_dataset_rows":len(valid),"training_dataset_sha256":train_hash,
        "validation_dataset_sha256":valid_hash,"train_id_sha256":hashlib.sha256("\n".join(train.id.astype(str)).encode()).hexdigest(),
        "id_sha256":hashlib.sha256("\n".join(valid.id.astype(str)).encode()).hexdigest(),
        "classes":classes,"config":snapshot,"config_sha256":config_hash,
        "dataset_sha256":profile["dataset_sha256"],"split_sha256":profile["split_sha256"],
        "code_revision":code_revision,"model_revision":cfg["revision"],"checkpoint_id":cfg["checkpoint"],"seed":seed,
        "quantization":result.get("quantization",cfg.get("quantization","none")),
        "quantization_config":result.get("quantization_config"),
        "attention_backend":result.get("attention_backend",cfg.get("attention_backend")),
        "fit_seconds":result["fit_seconds"],"inference_seconds":result["inference_seconds"],
        "runtime_breakdown":runtime_breakdown,
        "training_view_sampling": training_view_sampling,
        "training_view_sampling_artifact": str(sampling_path) if sampling_path else None,
        "training_rows_per_second": (len(train) / float(runtime_breakdown.get(
            "training_excluding_validation_seconds") or result.get("fit_seconds") or 1.0)),
        "checkpoint_scoring_method":result.get("checkpoint_scoring_method"),
        "final_validation_reused_existing_predictions":result.get("final_validation_reused_epoch_predictions"),
        "early_stopping":result.get("early_stopping"),
        "validation_metrics_by_epoch":result["validation_metrics_by_epoch"],
        "validation_loss_raw": result.get("validation_loss_raw"),
        "validation_predictions_by_epoch":epoch_prediction_records,
        "requested_batch_size":result.get("requested_batch_size"),
        "effective_batch_size":result.get("effective_batch_size"),
        "selected_physical_batch_size":result.get("selected_physical_batch_size",result.get("effective_batch_size")),
        "effective_optimizer_batch_size":result.get("effective_optimizer_batch_size"),
        "target_effective_batch_size":cfg.get("target_effective_batch_size"),
        "peak_gpu_memory_bytes":result.get("peak_gpu_memory_bytes"),
        "peak_gpu_memory_reserved_bytes":result.get("peak_gpu_memory_reserved_bytes"),
        "gradient_accumulation_steps":result.get("gradient_accumulation_steps",
            snapshot.get("gradient_accumulation_steps", 1)),
        "oom_recovery":result.get("oom_recovery"),
        "requested_inference_batch_size":result.get("requested_inference_batch_size",cfg.get("inference_batch_size")),
        "effective_inference_batch_size":result.get("runtime_breakdown",{}).get("effective_inference_batch_size"),
        "batch_size_probe":result.get("batch_size_probe"),
        "epoch_checkpoints":result.get("epoch_checkpoints",[]),
        "checkpoint_path":str(checkpoint),"prediction_path":str(prediction_path),"metrics":metrics,
        "selected_checkpoint_path":result.get("selected_checkpoint_path") or result.get("best_checkpoint_path"),
        "best_checkpoint_path":result.get("best_checkpoint_path") or result.get("selected_checkpoint_path"),
        "created_at_utc":pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path=pred_dir/f"{experiment_id}_manifest.json";manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    Registry(registry_path).add(experiment_id,model,"validation",snapshot,metrics,
        dataset_hash=profile["dataset_sha256"],split_hash=profile["split_sha256"],model_revision=cfg["revision"],
        code_revision=code_revision,seed=seed,runtime_seconds=result["fit_seconds"],artifact_path=str(prediction_path),
        config_hash=config_hash,description=f"Continued Qwen LoRA from {source['experiment_id']} for {epochs} epochs; validation metrics recorded each epoch.")
    runtime_breakdown["manifest_prediction_export_seconds"] = time.perf_counter() - export_started
    runtime_breakdown["manifest_export_seconds"] = time.perf_counter() - manifest_export_started
    manifest["runtime_breakdown"] = runtime_breakdown
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    _release_completed_training_model(result)
    if refresh_prediction or refresh_checkpoint_catalog:
        if validation_dataset == "v0":
            _refresh_ensemble_after_training(manifest, manifest_path, artifact_dir=artifact_dir,
                data_dir=data_dir, refresh_prediction=refresh_prediction,
                refresh_checkpoint_catalog=refresh_checkpoint_catalog)
        else:
            manifest["runtime_breakdown"]["downstream_refresh_seconds"] = 0.0
            manifest["prediction_factory_update"] = {"status":"skipped", "reason":"V0 validation required"}
            manifest["checkpoint_catalog_update"] = {"status":"skipped", "reason":"V0 validation required"}
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
            (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    else:
        manifest["prediction_factory_update"] = {"status":"skipped", "reason":"disabled"}
        manifest["checkpoint_catalog_update"] = {"status":"skipped", "reason":"disabled"}
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    runtime_breakdown["total_post_training_seconds"]=(
        float(runtime_breakdown.get("model_post_training_seconds",0.0))+
        time.perf_counter()-post_training_started)
    manifest["runtime_breakdown"]=runtime_breakdown
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    _append_v2_checkpoint_trust(manifest, manifest_path)
    return manifest


def continue_encoder(dataset="v1", model="ultrabert380", data_dir="data/processed",
                     artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *,
                     from_experiment, epochs=None, learning_rate=None, validation_dataset="v0",
                     learning_rate_decay=None, seed=2026, optimizer=None,
                     weight_decay=None, config_path=None, config_overrides=None, refresh_prediction=False,
                     refresh_checkpoint_catalog=False, allow_cross_dataset_source=False):
    """Validate a saved encoder checkpoint, continue training, and keep the best epoch.

    Baseline validation runs before the first optimizer step and is included in the
    run manifest. This path currently supports full-finetuned sequence encoders.
    """
    if model != "ultrabert380":
        raise ValueError("continue-encoder currently supports only ultrabert380; use the model-specific continuation path for LoRA models")
    manifests_dir = Path(artifact_dir) / "predictions"
    source_path = manifests_dir / f"{from_experiment}_manifest.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"source experiment manifest not found: {source_path}")
    source = json.loads(source_path.read_text())
    if source.get("model") != model:
        raise ValueError(f"source experiment is {source.get('model')}, expected {model}")
    # Training manifests commonly point at the checkpoint container, while
    # `train_encoder_fold(select_best_checkpoint=True)` saves the selected
    # epoch in its nested best_checkpoint directory. Prefer an explicit best
    # path, then discover that nested checkpoint before resuming.
    raw_checkpoint = source.get("best_checkpoint_path") or source.get("checkpoint_path")
    if not raw_checkpoint:
        raise FileNotFoundError("source experiment manifest has no checkpoint path")
    source_checkpoint = _resolve_checkpoint_path(raw_checkpoint, artifact_dir)
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")
    nested_best = source_checkpoint / "best_checkpoint"
    if nested_best.is_dir():
        source_checkpoint = nested_best

    train, valid = _training_partitions(dataset, validation_dataset, data_dir)
    training_views = train.training_views.tolist() if "training_views" in train.columns else None
    training_view_ids = train.training_view_ids.tolist() if "training_view_ids" in train.columns else None
    rows = pd.concat([train, valid], ignore_index=True)
    train_hash, valid_hash = _partition_sha256(train), _partition_sha256(valid)
    training_profile = json.loads(_profile_path(Path(data_dir), dataset).read_text())
    if not _continuation_source_matches(source, dataset, validation_dataset,
                                        train, valid, training_profile.get("dataset_sha256"),
                                        allow_cross_dataset_source=allow_cross_dataset_source):
        raise ValueError("source checkpoint dataset content hashes differ from requested continuation data")
    profile = {"dataset_sha256": train_hash, "split_sha256": _sha([dataset, train_hash, validation_dataset, valid_hash])}
    classes = sorted(train.model.astype(str).unique().tolist())
    if source.get("classes") != classes:
        raise ValueError("source checkpoint class labels differ from the target dataset")

    source_key = hashlib.sha256(from_experiment.encode()).hexdigest()[:8]
    explicit = dict(config_overrides or {})
    for key, value in (("epochs", epochs), ("learning_rate", learning_rate),
                       ("learning_rate_decay", learning_rate_decay), ("optimizer", optimizer),
                       ("weight_decay", weight_decay)):
        if value is not None:
            explicit[key] = value
    cfg = _continuation_config(model, source.get("config", {}), config_path, explicit)
    epochs = int(cfg.get("epochs", 20))
    learning_rate = float(cfg.get("learning_rate", 5e-5))
    learning_rate_decay = (float(cfg["learning_rate_decay"])
                           if cfg.get("learning_rate_decay") is not None else None)
    optimizer = str(cfg.get("optimizer", "adamw_torch"))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if (epochs < 1 or learning_rate <= 0 or
            (learning_rate_decay is not None and not 0 < learning_rate_decay <= 1)):
        raise ValueError("epochs and learning_rate must be positive; optional decay must be in (0, 1]")
    config_snapshot = {
        **{key: value for key, value in cfg.items() if not key.startswith("_")},
        "model": model, "dataset": dataset, "training_dataset": dataset,
        "continued_from_experiment": from_experiment,
        "epochs": int(epochs), "learning_rate": float(learning_rate),
        "learning_rate_decay": (float(learning_rate_decay)
                                if learning_rate_decay is not None else None),
        "seed": int(seed),
        "optimizer": optimizer, "weight_decay": float(weight_decay),
        "early_stopping_patience": cfg.get("early_stopping_patience", 2),
        "batch_size": int(cfg.get("batch_size", 8)),
        "auto_batch_size": bool(cfg.get("auto_batch_size", False)),
        "max_batch_size": int(cfg.get("max_batch_size", 1024)),
        "batch_probe_safety_fraction": float(cfg.get("batch_probe_safety_fraction", 0.95)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
        "max_length": (_profile_max_length_override(model, cfg, training_profile)
            if training_profile.get("chunking_strategy") else _dataset_max_length(model, {
                **cfg, "checkpoint": str(source_checkpoint), "revision": None,
                "task_prefix": "", "training_mode": "full", "trust_remote_code": True,
            }, pd.concat([train, valid], ignore_index=True), checkpoint=str(source_checkpoint), revision=None)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.0)),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "linear"),
        "baseline_validation_before_training": True,
        "checkpoint_selection_metric": "balanced_accuracy",
        "validation_dataset": validation_dataset,
        "checkpoint_selection": "highest_validation_balanced_accuracy",
        "save_epoch_checkpoints": bool(cfg.get("save_epoch_checkpoints", False)),
        "inference_batch_size": int(cfg.get("inference_batch_size", max(8, int(cfg.get("batch_size", 8))))),
    }
    if training_views is not None:
        config_snapshot["training_view_sampling_policy"] = "one random view per original response per epoch"
    config_hash = _sha(config_snapshot)
    experiment_id = (f"{model}_{dataset}_from{source_key}_data{profile['dataset_sha256'][:8]}_"
                    f"split{profile['split_sha256'][:8]}_seed{seed}_cfg{config_hash[:8]}_code{_revision()[:7]}")
    run_dir = Path(artifact_dir) / "runs" / experiment_id
    prediction_path = manifests_dir / f"{experiment_id}_validation.parquet"
    manifest_path = manifests_dir / f"{experiment_id}_manifest.json"
    if manifest_path.exists() and prediction_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _register_completed_manifest(manifest, registry_path)
        _append_v2_checkpoint_trust(manifest, manifest_path)
        return manifest
    if run_dir.exists():
        raise FileExistsError(f"incomplete continuation run exists at {run_dir}")

    result = train_encoder_fold(
        train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
        checkpoint=source.get("checkpoint_id") or str(source_checkpoint),
        revision=source.get("model_revision"), resume_checkpoint=source_checkpoint,
        output_dir=run_dir / "checkpoints" / f"{model}_continued",
        epochs=epochs, batch_size=config_snapshot["batch_size"],
        inference_batch_size=config_snapshot["inference_batch_size"],
        max_length=config_snapshot["max_length"], learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay, seed=seed,
        lr_scheduler_type=config_snapshot["lr_scheduler_type"],
        warmup_ratio=config_snapshot["warmup_ratio"],
        optimizer=optimizer, weight_decay=weight_decay,
        gradient_accumulation_steps=config_snapshot["gradient_accumulation_steps"],
        target_effective_batch_size=config_snapshot.get("target_effective_batch_size"),
        auto_batch_size=config_snapshot["auto_batch_size"],
        max_batch_size=config_snapshot["max_batch_size"],
        batch_probe_safety_fraction=config_snapshot["batch_probe_safety_fraction"],
        gradient_checkpointing=config_snapshot["gradient_checkpointing"],
        validation_labels=valid.model.tolist(),
        select_best_checkpoint=True, local_files_only=True,
        early_stopping_patience=cfg.get("early_stopping_patience", 2),
        checkpoint_selection_metric="balanced_accuracy",
        save_epoch_checkpoints=config_snapshot["save_epoch_checkpoints"],
        training_views=training_views, training_view_ids=training_view_ids)
    training_view_sampling = result.get("training_view_sampling")
    sampling_path = None
    if training_view_sampling is not None:
        sampling_path = run_dir / "training_view_sampling.json"
        sampling_path.write_text(json.dumps({"training_dataset": dataset,
            "training_dataset_sha256": train_hash, "sampling": training_view_sampling}, indent=2) + "\n")
    metrics = evaluate(valid.model.astype(str), result["scores"], classes, result["probabilities"])
    if result.get("validation_loss_raw") is not None:
        metrics["validation_loss_raw"] = float(result["validation_loss_raw"])
    for key in [name for name in metrics if str(name).startswith("calibrated_")]:
        metrics.pop(key, None)
    export_started = time.perf_counter()
    post_training_started = export_started
    runtime_breakdown = _post_training_runtime(result)
    pred = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for i, label in enumerate(classes):
        pred[f"score::{label}"] = result["scores"][:, i]
        pred[f"prob::{label}"] = result["probabilities"][:, i]
    manifests_dir.mkdir(parents=True, exist_ok=True)
    prediction_export_started = time.perf_counter()
    pred.to_parquet(prediction_path, index=False)
    epoch_prediction_records = _save_epoch_validation_predictions(
        result, valid, classes, manifests_dir, experiment_id)
    runtime_breakdown["prediction_export_seconds"] = time.perf_counter() - prediction_export_started
    checkpoint = run_dir / "checkpoints" / f"{model}_continued"
    manifest_export_started = time.perf_counter()
    manifest = {
        "experiment_id": experiment_id, "model": model, "dataset": dataset,
        "training_dataset": dataset, "validation_dataset": validation_dataset, "split": "validation",
        "continued_from_experiment": from_experiment,
        "continued_from_checkpoint": str(source_checkpoint),
        "baseline_validation_metrics": result["baseline_validation_metrics"],
        "additional_epochs": int(epochs), "learning_rate": float(learning_rate),
        "learning_rate_decay": (float(learning_rate_decay)
                                if learning_rate_decay is not None else None),
        "validation_metrics_by_epoch": result["validation_metrics_by_epoch"],
        "validation_predictions_by_epoch": epoch_prediction_records,
        "early_stopping": result.get("early_stopping"),
        "selected_epoch": result["best_epoch"],
        "selected_validation_accuracy": result["best_validation_accuracy"],
        "checkpoint_selection_metric": result["checkpoint_selection_metric"],
        "checkpoint_scoring_method": result.get("checkpoint_scoring_method"),
        "final_validation_reused_existing_predictions": result.get("final_validation_reused_epoch_predictions"),
        "train_rows": len(train), "validation_rows": len(valid),
        "training_dataset_rows": len(train), "validation_dataset_rows": len(valid),
        "training_dataset_sha256": train_hash, "validation_dataset_sha256": valid_hash,
        "train_id_sha256": hashlib.sha256("\n".join(train.id.astype(str)).encode()).hexdigest(),
        "id_sha256": hashlib.sha256("\n".join(valid.id.astype(str)).encode()).hexdigest(),
        "classes": classes, "dataset_sha256": profile["dataset_sha256"], "split_sha256": profile["split_sha256"],
        "code_revision": _revision(), "model_revision": source.get("model_revision"),
        "checkpoint_id": source.get("checkpoint_id"), "config": config_snapshot,
        "config_sha256": config_hash, "seed": seed,
        "fit_seconds": result["fit_seconds"], "inference_seconds": result["inference_seconds"],
        "runtime_breakdown": runtime_breakdown,
        "training_view_sampling": training_view_sampling,
        "training_view_sampling_artifact": str(sampling_path) if sampling_path else None,
        "training_rows_per_second": (len(train) / float(runtime_breakdown.get(
            "training_excluding_validation_seconds") or result.get("fit_seconds") or 1.0)),
        "requested_batch_size": result.get("requested_batch_size"),
        "effective_batch_size": result.get("effective_batch_size"),
        "selected_physical_batch_size": result.get("selected_physical_batch_size",
            result.get("effective_batch_size")),
        "effective_optimizer_batch_size": result.get("effective_optimizer_batch_size"),
        "target_effective_batch_size": config_snapshot.get("target_effective_batch_size"),
        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
        "peak_gpu_memory_reserved_bytes": result.get("peak_gpu_memory_reserved_bytes"),
        "gradient_accumulation_steps": result.get("gradient_accumulation_steps",
            config_snapshot.get("gradient_accumulation_steps", 1)),
        "oom_recovery": result.get("oom_recovery"),
        "requested_inference_batch_size": result.get("requested_inference_batch_size", config_snapshot["inference_batch_size"]),
        "effective_inference_batch_size": result.get("effective_inference_batch_size"),
        "batch_size_probe": result.get("batch_size_probe"),
        "validation_loss_raw": result.get("validation_loss_raw"),
        "checkpoint_path": str(checkpoint), "best_checkpoint_path": result["best_checkpoint_path"],
        "selected_checkpoint_path": result.get("selected_checkpoint_path") or result["best_checkpoint_path"],
        "prediction_path": str(prediction_path), "metrics": metrics,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (manifests_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["description"] = f"Continued {model} from {from_experiment} for {epochs} epochs after baseline validation."
    _register_completed_manifest(manifest, registry_path)
    runtime_breakdown["manifest_prediction_export_seconds"] = time.perf_counter() - export_started
    runtime_breakdown["manifest_export_seconds"] = time.perf_counter() - manifest_export_started
    manifest["runtime_breakdown"] = runtime_breakdown
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (manifests_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _release_completed_training_model(result)
    if refresh_prediction or refresh_checkpoint_catalog:
        if validation_dataset == "v0":
            _refresh_ensemble_after_training(manifest, manifest_path, artifact_dir=artifact_dir,
                data_dir=data_dir, refresh_prediction=refresh_prediction,
                refresh_checkpoint_catalog=refresh_checkpoint_catalog)
        else:
            manifest["runtime_breakdown"]["downstream_refresh_seconds"] = 0.0
            manifest["prediction_factory_update"] = {"status":"skipped", "reason":"V0 validation required"}
            manifest["checkpoint_catalog_update"] = {"status":"skipped", "reason":"V0 validation required"}
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
            (manifests_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    else:
        manifest["prediction_factory_update"] = {"status":"skipped", "reason":"disabled"}
        manifest["checkpoint_catalog_update"] = {"status":"skipped", "reason":"disabled"}
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        (manifests_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    runtime_breakdown["total_post_training_seconds"] = (
        float(runtime_breakdown.get("model_post_training_seconds", 0.0)) +
        time.perf_counter() - post_training_started)
    manifest["runtime_breakdown"] = runtime_breakdown
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (manifests_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    _append_v2_checkpoint_trust(manifest, manifest_path)
    return manifest


def _load_validation(dataset, artifact_dir):
    files = sorted(p for p in (Path(artifact_dir) / "predictions").glob("*_validation.parquet")
                   if f"_{dataset}_" in p.name and not p.name.startswith("ensemble_"))
    if not files: raise FileNotFoundError(f"no validation prediction files found for dataset {dataset}")
    frames = {p.name.removesuffix("_validation.parquet"): pd.read_parquet(p) for p in files}
    first = next(iter(frames.values())); classes = sorted(c.removeprefix("prob::") for c in first if c.startswith("prob::"))
    aligned = align_predictions({name: frame.rename(columns={f"prob::{c}": c for c in classes}) for name, frame in frames.items()}, classes)
    manifests = {name: json.loads((Path(artifact_dir)/"predictions"/f"{name}_manifest.json").read_text()) for name in frames}
    dataset_hashes = {m.get("dataset_sha256") for m in manifests.values()}
    split_hashes = {m.get("split_sha256") for m in manifests.values()}
    if len(dataset_hashes) != 1 or len(split_hashes) != 1: raise ValueError("validation artifacts have inconsistent dataset or split hashes")
    for name, manifest in manifests.items():
        if manifest.get("classes") != classes: raise ValueError(f"validation class order differs for {name}")
    base = first.sort_values("id", kind="stable").reset_index(drop=True)
    if not np.array_equal(base.id.astype(str).to_numpy(), aligned.id.astype(str).to_numpy()): raise ValueError("validation IDs do not align")
    for name, frame in frames.items():
        aligned_meta = frame.sort_values("id", kind="stable").reset_index(drop=True)
        for column in ("y_true", "prompt_id"):
            if not np.array_equal(base[column].astype(str).to_numpy(), aligned_meta[column].astype(str).to_numpy()):
                raise ValueError(f"validation metadata mismatch for {name}.{column}")
    y = base.y_true.astype(str).to_numpy(); groups = base.prompt_id.to_numpy()
    probabilities = {name: np.column_stack([aligned[f"{name}::{c}"].to_numpy() for c in classes]) for name in frames}
    return base, classes, y, groups, probabilities, manifests, next(iter(dataset_hashes)), next(iter(split_hashes))


def _load_training_predictions(dataset, artifact_dir, manifests):
    """Predict the fit rows with each trained checkpoint; never read validation labels here."""
    from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
    rows, _ = _load_dataset(dataset, Path(__file__).resolve().parents[1]/"data/processed")
    train = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(rows.model.astype(str).unique())
    output = {}; y = train.model.astype(str).to_numpy()
    for name, manifest in manifests.items():
        path = Path(artifact_dir)/"predictions"/f"{name}_train.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if not np.array_equal(frame.id.astype(str).to_numpy(), train.id.astype(str).to_numpy()):
                raise ValueError(f"stored train predictions do not align for {name}")
            probabilities = np.column_stack([frame[f"prob::{cls}"].to_numpy() for cls in classes])
        elif manifest["model"] == "svc":
            bundle = joblib.load(manifest["checkpoint_path"])
            probabilities, _ = predict_svc_bundle(bundle, train.text.tolist())
        elif manifest["model"] == "catboost_stylometry":
            probabilities, _ = predict_catboost_stylometry(manifest["checkpoint_path"],
                train.text.tolist(), classes, temperature=manifest.get("calibration_temperature", 1.0))
        else:
            probabilities, _ = _predict_neural(manifest["model"], manifest["checkpoint_path"], train.text.tolist(), classes,
                max_length=manifest.get("config", {}).get("max_length", 384),
                temperature=manifest.get("calibration_temperature", 1.0),
                base_checkpoint=manifest.get("config", {}).get("checkpoint"), revision=manifest.get("model_revision"),
                fix_mistral_regex=manifest.get("config", {}).get("fix_mistral_regex", False),
                text_only_backbone=manifest.get("config", {}).get("text_only_backbone", False),
                quantization=manifest.get("config", {}).get("quantization", manifest.get("quantization")),
                attention_backend=manifest.get("config", {}).get("attention_backend", manifest.get("attention_backend")))
        if probabilities.shape != (len(train), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid train predictions for {name}")
        if not path.exists():
            frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
            for i, cls in enumerate(classes): frame[f"prob::{cls}"] = probabilities[:, i]
            frame.to_parquet(path, index=False)
        output[name] = probabilities
    return y, output


def _train_only_weights(models, y, classes, *, grid_steps=10, shrinkage=0.01):
    """Choose a simplex blend on train-fit predictions, with a small equal-weight prior."""
    names = list(models); n = len(names); equal = np.full(n, 1/n)
    best = None
    for cuts in itertools.product(range(grid_steps + 1), repeat=n):
        if sum(cuts) != grid_steps: continue
        weights = np.asarray(cuts, dtype=float) / grid_steps
        probabilities = np.average(np.stack([models[name] for name in names]), axis=0, weights=weights)
        score = balanced_accuracy_score(y, np.asarray(classes)[probabilities.argmax(1)])
        penalty = shrinkage * float(np.sum((weights - equal) ** 2))
        key = (score - penalty, -float(np.sum((weights-equal)**2)))
        if best is None or key > best[0]: best = (key, weights, float(score))
    if best is None: raise RuntimeError("could not find a train-only ensemble weight vector")
    return dict(zip(names, best[1].tolist())), best[2]


def evaluate_ensembles(dataset="v0", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, bootstrap_replicates=2000, seed=2026, model_names=None, force=False):
    """Compare equal-weight blends on the shared validation partition (exploratory)."""
    started = time.perf_counter()
    base, classes, y, groups, models, manifests, dataset_hash, split_hash = _load_validation(dataset, artifact_dir)
    if model_names:
        requested = list(model_names)
        available = {}
        for model_id in requested:
            if model_id in models:
                available[model_id] = model_id
                continue
            matches = [name for name in models if manifests[name].get("model") == model_id]
            if matches:
                available[model_id] = max(matches, key=lambda name: manifests[name].get("created_at_utc", ""))
        missing = set(requested) - set(available)
        if missing: raise ValueError(f"requested models have no validation artifacts: {sorted(missing)}")
        selected = [available[model_id] for model_id in requested]
        models = {name: models[name] for name in selected}
    names = list(models)
    if len(names) < 2: raise ValueError("ensemble evaluation requires at least two model validation files")
    manifests = {name: manifests[name] for name in names}
    train_y, train_models = _load_training_predictions(dataset, artifact_dir, manifests)
    weights, train_weighted_accuracy = _train_only_weights(train_models, train_y, classes)
    reports = {}; predictions = {n: np.asarray(classes)[p.argmax(1)] for n, p in models.items()}
    for size in range(2, len(names) + 1):
        for subset in itertools.combinations(names, size):
            prob = ensemble_probabilities([models[n] for n in subset]); label = "+".join(subset)
            metric = evaluate(y, np.log(np.clip(prob, 1e-12, 1)), classes, prob); metric["members"] = list(subset)
            predictions[label] = np.asarray(classes)[prob.argmax(1)]
            reports[label] = {"metrics": metric, "probabilities": prob}
    weighted_name = "train_weighted:" + "+".join(names)
    weighted_prob = np.average(np.stack([models[name] for name in names]), axis=0,
                               weights=[weights[name] for name in names])
    weighted_metrics = evaluate(y, np.log(np.clip(weighted_prob, 1e-12, 1)), classes, weighted_prob)
    weighted_metrics["members"] = names
    predictions[weighted_name] = np.asarray(classes)[weighted_prob.argmax(1)]
    reports[weighted_name] = {"metrics": weighted_metrics, "probabilities": weighted_prob,
                              "weights": weights, "weight_source": "train_fit_predictions_only",
                              "train_balanced_accuracy": train_weighted_accuracy}
    base_metrics = {name: evaluate(y, np.log(np.clip(prob, 1e-12, 1)), classes, prob) for name, prob in models.items()}
    best_single = max(names, key=lambda name: base_metrics[name]["balanced_accuracy"])
    model_costs = {name: {key: manifests[name].get(key) for key in ("fit_seconds", "inference_seconds", "peak_gpu_memory_bytes", "checkpoint_size_bytes")} for name in names}
    for label, record in reports.items():
        record["paired_bootstrap_vs_best_single"] = paired_group_bootstrap(y, predictions[best_single], predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)
        record["per_class_recall_delta_vs_best_single"] = {cls: record["metrics"]["per_class_recall"][cls] - base_metrics[best_single]["per_class_recall"][cls] for cls in classes}
        members = record["metrics"]["members"]; marginal = {}
        if label == weighted_name:
            equal_prob = ensemble_probabilities([models[item] for item in names])
            equal_pred = np.asarray(classes)[equal_prob.argmax(1)]
            marginal["vs_equal_weight"] = {"delta_balanced_accuracy": float(record["metrics"]["balanced_accuracy"] - balanced_accuracy_score(y, equal_pred)),
                "paired_bootstrap": paired_group_bootstrap(y, equal_pred, predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)}
        elif len(members) > 1:
            for member in members:
                previous = [item for item in members if item != member]
                prev_prob = ensemble_probabilities([models[item] for item in previous]); prev_pred = np.asarray(classes)[prev_prob.argmax(1)]
                marginal[member] = {"delta_balanced_accuracy": float(record["metrics"]["balanced_accuracy"] - balanced_accuracy_score(y, prev_pred)),
                    "paired_bootstrap": paired_group_bootstrap(y, prev_pred, predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)}
        record["marginal_contribution"] = marginal
        record["inference_cost_seconds"] = sum((model_costs[n]["inference_seconds"] or 0) for n in members)
    pred_dir = Path(artifact_dir)/"predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    output_columns = {column: base[column].to_numpy() for column in ("id", "prompt_id", "y_true")}
    for label, record in reports.items():
        for i, cls in enumerate(classes):
            output_columns[f"{label}::score::{cls}"] = np.log(np.clip(record["probabilities"][:, i], 1e-12, 1))
            output_columns[f"{label}::prob::{cls}"] = record["probabilities"][:, i]
    output = pd.DataFrame(output_columns)
    serializable = {k: {kk: vv for kk, vv in v.items() if kk != "probabilities"} for k, v in reports.items()}
    best = max(reports, key=lambda n: reports[n]["metrics"]["balanced_accuracy"])
    best_metrics = reports[best]["metrics"]; code_revision = _revision()
    member_key = _sha({name: manifests[name].get("prediction_sha256") for name in names})[:10]
    model_label = _ensemble_model_label(names)
    report = {"experiment_id": f"ensemble_{dataset}_seed{seed}_members{member_key}_code{code_revision[:7]}",
        "model": model_label, "dataset": dataset, "split": "validation", "classes": classes, "members": names,
        "validation_rows": len(y),
        "best_single": best_single, "best_ensemble": best, "balanced_accuracy": best_metrics["balanced_accuracy"],
        "calibrated_balanced_accuracy": best_metrics.get("calibrated_balanced_accuracy"), "macro_f1": best_metrics["macro_f1"],
        "log_loss": best_metrics.get("log_loss"), "per_class_recall": best_metrics["per_class_recall"],
        "confusion_matrix": best_metrics["confusion_matrix"], "model_costs": model_costs, "models": base_metrics,
        "ensembles": serializable, "complementarity_all": complementarity(y, predictions), "seed": seed,
        "train_weighting": {"weights": weights, "train_balanced_accuracy": train_weighted_accuracy,
            "source": "in-sample predictions on train rows; validation labels were not used to fit weights",
            "grid_step": 1/10, "equal_weight_shrinkage": 0.01},
        "code_revision": code_revision, "dataset_sha256": dataset_hash, "split_sha256": split_hash,
        "validation_selection_note": "Blend comparisons use the same validation rows as the base model scores; treat them as exploratory, not an unbiased final estimate."}
    report["runtime_seconds"] = time.perf_counter() - started
    path = pred_dir/f"{report['experiment_id']}_validation.parquet"; metrics_path=pred_dir/f"{report['experiment_id']}_metrics.json"
    if metrics_path.exists() and path.exists() and not force: return json.loads(metrics_path.read_text())
    output.to_parquet(path,index=False); report["prediction_path"] = str(path)
    metrics_path.write_text(json.dumps(report,indent=2)+"\n")
    Registry(registry_path).add(report["experiment_id"], model_label, "validation", {"members":names,"seed":seed}, report,
        dataset_hash=dataset_hash, split_hash=split_hash, code_revision=code_revision, seed=seed,
        runtime_seconds=report["runtime_seconds"], artifact_path=str(path),
        description=f"Validation blend comparison of {' + '.join(_ensemble_member_labels(names))} on {dataset} (exploratory).")
    return report


def restart_mvp(*, models=None, seed=2026, bootstrap_replicates=2000, data_config="configs/data/official_v0.yaml",
                feature_config="configs/features/default.yaml", registry_path="experiments/registry.sqlite"):
    """Clear generated experiment state and rebuild the complete MVP from raw CSVs."""
    root = Path(__file__).resolve().parents[1]
    data_cfg = _read_yaml(root / data_config)
    feature_cfg_path = root / feature_config
    if not feature_cfg_path.is_file():
        raise FileNotFoundError(f"feature config not found: {feature_cfg_path}")
    feature_cfg = _read_yaml(feature_cfg_path)
    selected = list(models or DEFAULT_MVP_MODELS)
    unknown = set(selected) - set(MODEL_CONFIGS)
    if unknown:
        raise ValueError(f"unknown restart model(s): {sorted(unknown)}")
    for name in selected:
        _model_cfg(name, root / MODEL_CONFIGS[name])
    train_path = Path(data_cfg["train_path"])
    test_path = Path(data_cfg["test_path"]) if data_cfg.get("test_path") else None
    train_path = train_path if train_path.is_absolute() else root / train_path
    test_path = test_path if test_path is None or test_path.is_absolute() else root / test_path
    configured_output = Path(data_cfg.get("output_dir", "data/processed"))
    configured_output = configured_output if configured_output.is_absolute() else root / configured_output
    if configured_output.resolve() != (root / "data/processed").resolve():
        raise ValueError("restart data output must be data/processed; generated state has not been cleared")
    if data_cfg.get("dataset_version", "v0") != "v0":
        raise ValueError("restart only rebuilds the official V0 dataset; generated state has not been cleared")
    if not train_path.is_file() or (test_path is not None and not test_path.is_file()):
        raise FileNotFoundError("restart input CSV is missing; generated state has not been cleared")
    raw_train = pd.read_csv(train_path, dtype={"id": "string", "prompt_id": "string", "model": "string"})
    raw_test = (pd.read_csv(test_path, dtype={"id": "string"}) if test_path is not None else None)
    errors = validate_data(raw_train, raw_test, expected_classes=data_cfg.get("expected_classes", 18))
    if errors:
        raise ValueError("invalid restart inputs; generated state has not been cleared:\n- " + "\n- ".join(errors))
    assignments = make_splits(raw_train, seed=seed,
        validation_fraction=data_cfg.get("validation_fraction", .2),
        candidates=data_cfg.get("split_candidates", 1000))
    if set(assignments.split.astype(str)) != {"train", "validation"}:
        raise ValueError("restart split plan is incomplete; generated state has not been cleared")
    assigned = assignments
    expected_labels = set(raw_train.model.astype(str))
    for role in ("train", "validation"):
        partition = assigned[assigned.split.eq(role)]
        if set(partition.model.astype(str)) != expected_labels:
            raise ValueError(f"restart {role} split omits labels; generated state has not been cleared")
    processed = (root / "data/processed").resolve()
    artifacts = (root / "artifacts").resolve()
    protected = [(root / "data/raw").resolve(), (Path.home()/".cache/huggingface").resolve()]
    for target in (processed, artifacts):
        if target in protected or root not in target.parents:
            raise RuntimeError(f"refusing to clean unexpected path: {target}")
    registry = (root / registry_path).resolve()
    if root not in registry.parents or registry in protected:
        raise RuntimeError(f"refusing to remove unexpected registry path: {registry}")
    for target in (processed, artifacts):
        if target.exists():
            for child in target.iterdir():
                if child.is_symlink() or not child.is_dir(): child.unlink()
                else: shutil.rmtree(child)
        else:
            target.mkdir(parents=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(registry) + suffix).unlink(missing_ok=True)
    data_cfg["seed"] = seed
    profile = prepare(root / data_cfg["train_path"], root / data_cfg["output_dir"],
        root / data_cfg["test_path"] if data_cfg.get("test_path") else None,
        seed=seed, validation_fraction=data_cfg.get("validation_fraction", .2),
        candidates=data_cfg.get("split_candidates", 1000), expected_classes=data_cfg.get("expected_classes", 18))
    feature_report = build_features("v0", root / "data/processed", feature_cfg_path,
        root / "artifacts/features", validation_dataset="v0")
    model_reports = [train_models("v0", name, root/"data/processed", root/"artifacts", registry,
        config_path=root/MODEL_CONFIGS[name], feature_config_path=root/feature_config if name == "svc" else None,
        seed=seed, validation_dataset="v0")
        for name in selected]
    ensemble_report = evaluate_ensembles("v0", root/"artifacts", registry, bootstrap_replicates=bootstrap_replicates,
        seed=seed, model_names=selected)
    return {"cleaned": [str(processed), str(artifacts), str(registry)], "dataset_profile": profile,
        "feature_report": feature_report, "models": model_reports, "ensemble": ensemble_report}


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m src.pipeline");sub=parser.add_subparsers(dest="command",required=True)
    restart=sub.add_parser("restart", help="clean generated state and rebuild data, features, models, and ensembles")
    restart.add_argument("--models",nargs="+",choices=MODEL_CONFIGS,default=list(DEFAULT_MVP_MODELS))
    restart.add_argument("--seed",type=int,default=2026);restart.add_argument("--bootstrap-replicates",type=int,default=2000)
    d=sub.add_parser("data").add_subparsers(dest="action",required=True);p=d.add_parser("prepare");p.add_argument("--config",default="configs/data/official_v0.yaml")
    oc=d.add_parser("curate-openpool");oc.add_argument("--config",default="configs/data/openpool_v1.yaml")
    vd=d.add_parser("curate-balanced-deduped", help="quality and diversity curation for V1-all without class quotas")
    vd.add_argument("--config",default="configs/data/v1_all_balanced_deduped.yaml")
    vc=d.add_parser("curate-clean-compact", help="create defect-only V1-all clean and redundancy-reduced compact datasets")
    vc.add_argument("--config",default="configs/data/v1_all_clean_compact.yaml")
    v13=d.add_parser("build-v1-3", help="assemble V1.3 from the downloaded GLM/GPT corpora and run clean plus compact")
    v13.add_argument("--config",default="configs/data/v1_3_external.yaml")
    v2c=d.add_parser("build-v2-0-chunks800", help="derive deterministic 800-character views from V2.0")
    v2c.add_argument("--seed",type=int,default=2026);v2c.add_argument("--force",action="store_true")
    v2t=d.add_parser("build-v2-token-chunks", help="derive 200- and 300-token UltraBERT training datasets from V2.0")
    v2t.add_argument("--force",action="store_true");v2t.add_argument("--batch-size",type=int,default=256)
    mix=d.add_parser("assemble");mix.add_argument("--config",default="configs/data/mixed_v1_5.yaml")
    f=sub.add_parser("features").add_subparsers(dest="action",required=True);b=f.add_parser("build");b.add_argument("--dataset",default="v1");b.add_argument("--validation-dataset",default="v0");b.add_argument("--data-dir",default="data/processed");b.add_argument("--config",default="configs/features/default.yaml")
    m=sub.add_parser("models").add_subparsers(dest="action",required=True)
    t=m.add_parser("train");t.add_argument("--model",choices=MODEL_CONFIGS,default="svc");t.add_argument("--dataset",default="v1");t.add_argument("--data-dir",default="data/processed");t.add_argument("--artifact-dir",default="artifacts");t.add_argument("--config");t.add_argument("--features-config");t.add_argument("--seed",type=int,default=2026);t.add_argument("--validation-dataset",default="v0",help="dataset artifact used in full for validation (default: v0)");t.add_argument("--max-length",type=int,help="maximum tokenizer sequence length in tokens for training and validation");t.add_argument("--refresh-prediction",action="store_true",help="refresh the active prediction roster if its inputs changed");t.add_argument("--refresh-checkpoint-catalog",action="store_true",help="refresh the checkpoint-label catalog after training");t.add_argument("--skip-prediction-refresh",action="store_true",help=argparse.SUPPRESS);t.add_argument("--accept-config-defaults",action="store_true",help="noninteractive confirmation to use the model config values")
    cq=m.add_parser("continue-qwen",help="continue the latest Qwen LoRA checkpoint and validate after every epoch")
    cq.add_argument("--dataset",default="v1");cq.add_argument("--validation-dataset",default="v0");cq.add_argument("--epochs",type=int);cq.add_argument("--seed",type=int,default=2026)
    cq.add_argument("--model",choices=["qwen06","qwen17","qwen4","qwen8"],default="qwen06")
    cq.add_argument("--from-experiment");cq.add_argument("--config");cq.add_argument("--accept-config-defaults",action="store_true")
    cq.add_argument("--refresh-prediction",action="store_true");cq.add_argument("--refresh-checkpoint-catalog",action="store_true")
    ce=m.add_parser("continue-encoder",help="baseline-validate and continue a saved UltraBERT checkpoint")
    ce.add_argument("--model",choices=["ultrabert380"],default="ultrabert380")
    ce.add_argument("--dataset",default="v1");ce.add_argument("--validation-dataset",default="v0");ce.add_argument("--from-experiment",required=True)
    ce.add_argument("--epochs",type=int);ce.add_argument("--learning-rate",type=float)
    ce.add_argument("--learning-rate-decay",type=float);ce.add_argument("--seed",type=int,default=2026)
    ce.add_argument("--config")
    ce.add_argument("--data-dir",default="data/processed");ce.add_argument("--artifact-dir",default="artifacts")
    ce.add_argument("--registry",default="experiments/registry.sqlite");ce.add_argument("--accept-config-defaults",action="store_true")
    ce.add_argument("--allow-cross-dataset-source",action="store_true",
        help="continue from a model trained on a different dataset when its V0 validation set is identical")
    ce.add_argument("--refresh-prediction",action="store_true");ce.add_argument("--refresh-checkpoint-catalog",action="store_true")
    kv=m.add_parser("kaggle-v0-finetune",help="fine-tune the latest full-V2.14 UltraBERT on V0 train rows; isolated from Trust List 2")
    kv.add_argument("--from-experiment",required=True,help="source V2.14 UltraBERT experiment ID")
    kv.add_argument("--data-dir",default="data/processed");kv.add_argument("--artifact-dir",default="artifacts")
    kv.add_argument("--epochs",type=int,default=10);kv.add_argument("--learning-rate",type=float,default=2e-6)
    kv.add_argument("--early-stopping-patience",type=int,default=2);kv.add_argument("--seed",type=int,default=2026)
    ts=m.add_parser("catboost-two-stage",help="run sequential V1-all to V0 CatBoost feature experiments")
    ts.add_argument("--data-dir",default="data/processed")
    ts.add_argument("--artifact-dir",default="artifacts")
    ts.add_argument("--registry",default="experiments/registry.sqlite")
    ts.add_argument("--ensemble-registry",default="docs/ensembles.md")
    ts.add_argument("--config",default=MODEL_CONFIGS["catboost_stylometry"])
    ts.add_argument("--validation-dataset",default="v0")
    ts.add_argument("--seed",type=int,default=2026)
    ts.add_argument("--stage1-iterations",type=int,default=800)
    ts.add_argument("--stage2-iterations",type=int,default=200)
    ts.add_argument("--run-id",help="optional unique run identifier; defaults to current UTC timestamp")
    cc=m.add_parser("compare-checkpoints",help="compare two same-model checkpoints on frozen V0 validation")
    cc.add_argument("--previous",required=True,help="previous experiment ID or model manifest path")
    cc.add_argument("--new",required=True,help="new experiment ID or model manifest path")
    cc.add_argument("--dataset",default="v0",choices=["v0"],help="frozen validation dataset")
    cc.add_argument("--data-dir",default="data/processed")
    cc.add_argument("--artifact-dir",default="artifacts")
    cc.add_argument("--registry",default="experiments/registry.sqlite")
    cc.add_argument("--output",default="docs/checkpoint_comparison.md")
    ccs=m.add_parser("compare-checkpoint-set",help="compare 3+ same-model checkpoints on frozen V0 validation")
    ccs.add_argument("--checkpoint",action="append",required=True,metavar="LABEL=REFERENCE",
        help="display label and experiment ID or manifest path; repeat in desired report order")
    ccs.add_argument("--dataset",default="v0",choices=["v0"])
    ccs.add_argument("--data-dir",default="data/processed")
    ccs.add_argument("--artifact-dir",default="artifacts")
    ccs.add_argument("--registry",default="experiments/registry.sqlite")
    ccs.add_argument("--output",default="docs/checkpoint_comparison.md")
    ccs.add_argument("--bootstrap-replicates",type=int,default=2000)
    ccs.add_argument("--seed",type=int,default=2026)
    e=sub.add_parser("ensemble").add_subparsers(dest="action",required=True);q=e.add_parser("evaluate");q.add_argument("--dataset",default="v0");q.add_argument("--artifact-dir",default="artifacts");q.add_argument("--bootstrap-replicates",type=int,default=2000);q.add_argument("--models",nargs="+",choices=ENSEMBLE_MODEL_CHOICES);q.add_argument("--force",action="store_true")
    pred=sub.add_parser("prediction").add_subparsers(dest="action",required=True)
    pe=pred.add_parser("evaluate",help="compare active model champions with equal and train-fitted reliability blends")
    pe.add_argument("--dataset",default="v0",help="frozen validation dataset used to anchor candidate rows")
    pe.add_argument("--artifact-dir",default="artifacts")
    pe.add_argument("--minimum-support",type=int,default=10)
    pe.add_argument("--blend-batch-size",type=int,default=64)
    pe.add_argument("--top-k",type=int,default=100)
    pr=pred.add_parser("refresh",help="sync the active champion roster and record a new ensemble version when it changes")
    pr.add_argument("--dataset",default="v0")
    pr.add_argument("--data-dir",default="data/processed")
    pr.add_argument("--artifact-dir",default="artifacts")
    pr.add_argument("--manifest",help="completed training manifest to verify alignment")
    pr.add_argument("--only-if-roster-changed",action="store_true")
    pp=pred.add_parser("package",help="package the selected prediction rule and required checkpoints")
    pp.add_argument("--manifest",required=True)
    pp.add_argument("--output",default="artifacts/kaggle/prediction_bundle.zip")
    pp.add_argument("--external-base-models",action="store_true",
        help="leave pretrained base weights out of the bundle and record Kaggle input references")
    px=pred.add_parser("export",help="run a packaged prediction bundle on test.csv")
    px.add_argument("--bundle",required=True)
    px.add_argument("--test",default="data/raw/test.csv")
    px.add_argument("--output",default="submission.parquet")
    x=sub.add_parser("export",help="export predictions from an explicitly selected saved checkpoint")
    x.add_argument("--model",choices=MODEL_CONFIGS,default="svc")
    x.add_argument("--checkpoint",required=True,
        help="selected SVC bundle file or training manifest JSON (required for CatBoost/neural models)")
    x.add_argument("--test",default="data/raw/test.csv");x.add_argument("--output",default="submission.parquet")
    k=sub.add_parser("package-kaggle",help="package an explicitly selected saved checkpoint manifest for Kaggle")
    k.add_argument("--manifest",required=True,help="selected checkpoint's training manifest JSON")
    k.add_argument("--output",default="artifacts/kaggle/model_package.zip")
    analysis = sub.add_parser("analysis").add_subparsers(dest="action", required=True)
    ua = analysis.add_parser("ultrabert-four-checkpoints", help="run grouped OOF fusion analysis for saved UltraBERT checkpoints A-D")
    ua.add_argument("--artifact-dir", default="artifacts/predictions")
    ua.add_argument("--data-dir", default="data/processed")
    ua.add_argument("--output-dir", default="artifacts/analysis/ultrabert_four_checkpoint")
    ua.add_argument("--report", default="docs/ultrabert_4checkpoints_4experiments.md")
    ta = analysis.add_parser("trusted-checkpoint-ensemble", help="run grouped OOF trusted specialist routing for saved UltraBERT checkpoints A-D")
    ta.add_argument("--artifact-dir", default="artifacts/predictions")
    ta.add_argument("--data-dir", default="data/processed")
    ta.add_argument("--output-dir", default="artifacts/analysis/trusted_checkpoint_ensemble_ultrabert")
    ta.add_argument("--report", default="docs/trusted_checkpoint_ensemble_ultrabert.md")
    ta.add_argument("--minimum-support", type=int, default=10)
    ta.add_argument("--prior-alpha", type=float, default=1.0)
    ta.add_argument("--prior-beta", type=float, default=1.0)
    za = analysis.add_parser("zod01", help="run sequential precision-ranked checkpoint rules with prompt-group OOF evaluation")
    za.add_argument("--artifact-dir", default="artifacts/predictions")
    za.add_argument("--data-dir", default="data/processed")
    za.add_argument("--output-dir", default="artifacts/analysis/zod01_ultrabert")
    za.add_argument("--report", default="docs/zod01_ultrabert.md")
    za.add_argument("--minimum-support", type=int, default=10)
    za.add_argument("--prior-alpha", type=float, default=1.0)
    za.add_argument("--prior-beta", type=float, default=1.0)
    zs = analysis.add_parser("zod01-specialist-discovery", help="rank per-label precision specialists across saved model checkpoints")
    zs.add_argument("--artifact-dir", default="artifacts/predictions")
    zs.add_argument("--data-dir", default="data/processed")
    zs.add_argument("--output-dir", default="artifacts/analysis/zod01_specialist_discovery")
    zs.add_argument("--report", default="docs/zod01_specialist_discovery.md")
    zs.add_argument("--minimum-support", type=int, default=10)
    zf = analysis.add_parser("zod01-fixed-18", help="run the fixed user-specified 18-label ZOD01 cascade")
    zf.add_argument("--data-dir", default="data/processed")
    zf.add_argument("--inventory", default="artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv")
    zf.add_argument("--predictions-dir", default="artifacts/analysis/zod01_specialist_discovery/predictions")
    zf.add_argument("--registry", default="experiments/registry.sqlite")
    zf.add_argument("--output-dir", default="artifacts/analysis/zod01_18label_cascade")
    zf.add_argument("--report", default="docs/zod01_18label_cascade.md")
    z4 = analysis.add_parser("zod04", help="extract frozen UltraBERT layer features and compare logistic readouts")
    z4.add_argument("--artifact-dir", default="artifacts")
    z4.add_argument("--data-dir", default="data/processed")
    z4.add_argument("--output-dir", default="artifacts/analysis/zod_04")
    z4.add_argument("--report", default="docs/zod_04.md")
    z4.add_argument("--experiment-id", default=None,
        help="UltraBERT checkpoint experiment; defaults to the existing V1-all -> V0 best checkpoint")
    z4.add_argument("--batch-size", type=int, default=8)
    z4.add_argument("--n-splits", type=int, default=5)
    z4.add_argument("--seed", type=int, default=2026)
    z4.add_argument("--refresh-features", action="store_true",
        help="re-extract hidden states even if a compatible feature cache exists")
    tc = analysis.add_parser("trust-list-v2-complementarity",
        help="compare global trust precision with incremental cascade value using saved full-V0 predictions")
    tc.add_argument("--trust-list", default="docs/trust-list-2-v001.csv")
    tc.add_argument("--trust-metadata", default="docs/trust-list-2-v001.json")
    tc.add_argument("--predictions-dir", default="artifacts/analysis/trust_list_2_v0")
    tc.add_argument("--artifact-dir", default="artifacts/predictions")
    tc.add_argument("--data-dir", default="data/processed")
    tc.add_argument("--output-dir", default="artifacts/analysis/trust_list_2_v0")
    tc.add_argument("--report", default="docs/trust_list_v2_complementarity.md")
    tc.add_argument("--minimum-incremental-support", type=int, default=10)
    tc.add_argument("--fallback-sequence-id", default="N0081")
    tw = analysis.add_parser("trust-list-v2-weighted-vote",
        help="evaluate per-label reliability-weighted voting from the latest Trust List 2 using saved full-V0 predictions")
    tw.add_argument("--data-dir", default="data/processed")
    tw.add_argument("--predictions-dir", default="artifacts/analysis/trust_list_2_v0")
    tw.add_argument("--artifact-dir", default="artifacts/predictions")
    tw.add_argument("--output-dir", default="artifacts/analysis/trust_list_2_weighted_vote")
    tw.add_argument("--report", default="docs/trust_list_2_weighted_vote.md")
    ts = analysis.add_parser("trust-list-v2-checkpoint-subsets",
        help="greedy-select Trust List 2 v012 checkpoint subsets and evaluate by prompt-group OOF using saved predictions")
    ts.add_argument("--data-dir", default="data/processed")
    ts.add_argument("--predictions-dir", default="artifacts/analysis/trust_list_2_v0")
    ts.add_argument("--artifact-dir", default="artifacts/predictions")
    ts.add_argument("--output-dir", default="artifacts/analysis/trust_list_2_checkpoint_subsets")
    ts.add_argument("--report", default="docs/trust_list_2_checkpoint_subset_selection.md")
    ts.add_argument("--folds", type=int, default=5)
    a=parser.parse_args(argv)
    if a.command=="restart": result=restart_mvp(models=a.models,seed=a.seed,bootstrap_replicates=a.bootstrap_replicates)
    elif a.command=="data" and a.action=="prepare": result=prepare_data(a.config)
    elif a.command=="data" and a.action=="curate-openpool": result=curate_openpool_data(a.config)
    elif a.command=="data" and a.action=="curate-balanced-deduped": result=curate_v1_all_balanced_deduped_data(a.config)
    elif a.command=="data" and a.action=="curate-clean-compact": result=curate_v1_all_clean_compact_data(a.config)
    elif a.command=="data" and a.action=="build-v1-3": result=build_v1_3_external_data(a.config)
    elif a.command=="data" and a.action=="build-v2-0-chunks800":
        argv=["--seed",str(a.seed)]+(["--force"] if a.force else [])
        result=build_v2_0_chunks800_dataset(argv)
    elif a.command=="data" and a.action=="build-v2-token-chunks":
        from .v2_0_token_chunks import build_all
        result=build_all(force=a.force, batch_size=a.batch_size)
    elif a.command=="data" and a.action=="assemble": result=assemble_training_dataset(a.config)
    elif a.command=="features": result=build_features(a.dataset,a.data_dir,a.config,validation_dataset=a.validation_dataset)
    elif a.command=="models" and a.action=="train":
        cfg = _model_cfg(a.model, a.config)
        if a.model == "svc" and a.features_config:
            cfg["features"] = _read_yaml(a.features_config)
        overrides = _confirm_training_config(a.model, cfg, accept_defaults=a.accept_config_defaults)
        result=train_models(a.dataset,a.model,a.data_dir,a.artifact_dir,config_path=a.config,
            feature_config_path=a.features_config,seed=a.seed,config_overrides=overrides,
            refresh_prediction=a.refresh_prediction and not a.skip_prediction_refresh,
            refresh_checkpoint_catalog=a.refresh_checkpoint_catalog,
            validation_dataset=a.validation_dataset,max_length=a.max_length)
    elif a.command=="models" and a.action=="catboost-two-stage":
        from .catboost_two_stage import run_catboost_two_stage
        cfg = _model_cfg("catboost_stylometry", a.config)
        result=run_catboost_two_stage(load_dataset=_load_dataset,effective_profile=_effective_profile,
            validation_dataset=a.validation_dataset,data_dir=a.data_dir,artifact_dir=a.artifact_dir,registry_path=a.registry,
            ensemble_registry_path=a.ensemble_registry,
            config=cfg,seed=a.seed,stage1_iterations=a.stage1_iterations,
            stage2_iterations=a.stage2_iterations,run_id=a.run_id)
    elif a.command=="models" and a.action=="compare-checkpoints":
        from .checkpoint_comparison import compare_checkpoints
        result=compare_checkpoints(a.previous,a.new,data_dir=a.data_dir,
            artifact_dir=a.artifact_dir,registry_path=a.registry,
            output_path=a.output,dataset=a.dataset)
    elif a.command=="models" and a.action=="compare-checkpoint-set":
        from .checkpoint_comparison import compare_checkpoint_set
        checkpoint_map={}
        for item in a.checkpoint:
            if "=" not in item:
                raise ValueError(f"--checkpoint must be LABEL=REFERENCE; got {item!r}")
            label, reference=item.split("=",1)
            if not label.strip() or not reference.strip():
                raise ValueError(f"--checkpoint must have nonempty label and reference; got {item!r}")
            if label.strip() in checkpoint_map:
                raise ValueError(f"duplicate checkpoint label: {label.strip()}")
            checkpoint_map[label.strip()]=reference.strip()
        result=compare_checkpoint_set(checkpoint_map,data_dir=a.data_dir,
            artifact_dir=a.artifact_dir,registry_path=a.registry,
            output_path=a.output,dataset=a.dataset,
            bootstrap_replicates=a.bootstrap_replicates,seed=a.seed)
    elif a.command=="analysis" and a.action=="ultrabert-four-checkpoints":
        from .ultrabert_checkpoint_ensembles import run_ultrabert_four_checkpoint_experiments
        result=run_ultrabert_four_checkpoint_experiments(artifact_dir=a.artifact_dir,
            data_dir=a.data_dir,output_dir=a.output_dir,report_path=a.report)
    elif a.command=="analysis" and a.action=="trusted-checkpoint-ensemble":
        from .trusted_checkpoint_ensemble import run_ultrabert_trusted_checkpoint_ensemble
        result=run_ultrabert_trusted_checkpoint_ensemble(artifact_dir=a.artifact_dir,
            data_dir=a.data_dir,output_dir=a.output_dir,report_path=a.report,
            minimum_support=a.minimum_support,prior_alpha=a.prior_alpha,prior_beta=a.prior_beta)
    elif a.command=="analysis" and a.action=="zod01":
        from .zod01 import run_ultrabert_zod01
        result=run_ultrabert_zod01(artifact_dir=a.artifact_dir,data_dir=a.data_dir,
            output_dir=a.output_dir,report_path=a.report,minimum_support=a.minimum_support,
            prior_alpha=a.prior_alpha,prior_beta=a.prior_beta)
    elif a.command=="analysis" and a.action=="zod01-specialist-discovery":
        from .zod01_specialist_discovery import run_zod01_specialist_discovery
        result=run_zod01_specialist_discovery(artifact_dir=a.artifact_dir,data_dir=a.data_dir,
            output_dir=a.output_dir,report_path=a.report,minimum_support=a.minimum_support)
    elif a.command=="analysis" and a.action=="zod01-fixed-18":
        from .zod01_fixed_cascade import run_fixed_zod01_cascade
        result=run_fixed_zod01_cascade(data_dir=a.data_dir,inventory_path=a.inventory,
            predictions_dir=a.predictions_dir,registry_path=a.registry,
            output_dir=a.output_dir,report_path=a.report)
    elif a.command=="analysis" and a.action=="zod04":
        from .zod_04 import run_zod04
        result=run_zod04(data_dir=a.data_dir,artifact_dir=a.artifact_dir,
            output_dir=a.output_dir,report_path=a.report,experiment_id=a.experiment_id,
            batch_size=a.batch_size,n_splits=a.n_splits,seed=a.seed,
            refresh_features=a.refresh_features)
    elif a.command=="analysis" and a.action=="trust-list-v2-complementarity":
        from .v2_checkpoint_complementarity import run_v2_checkpoint_complementarity
        result=run_v2_checkpoint_complementarity(trust_list=a.trust_list,
            trust_metadata=a.trust_metadata,cache_dir=a.predictions_dir,
            artifact_dir=a.artifact_dir,data_dir=a.data_dir,output_dir=a.output_dir,
            report_path=a.report,minimum_incremental_support=a.minimum_incremental_support,
            fallback_sequence_id=a.fallback_sequence_id)
    elif a.command=="analysis" and a.action=="trust-list-v2-weighted-vote":
        from .trust_list2_weighted_vote import run_trust_list2_weighted_vote
        result=run_trust_list2_weighted_vote(data_dir=a.data_dir,
            cache_dir=a.predictions_dir,artifact_dir=a.artifact_dir,
            output_dir=a.output_dir,report_path=a.report)
    elif a.command=="analysis" and a.action=="trust-list-v2-checkpoint-subsets":
        from .trust_list2_weighted_vote import run_trust_list2_checkpoint_subsets
        result=run_trust_list2_checkpoint_subsets(data_dir=a.data_dir,
            cache_dir=a.predictions_dir,artifact_dir=a.artifact_dir,
            output_dir=a.output_dir,report_path=a.report,n_splits=a.folds)
    elif a.command=="models" and a.action=="continue-qwen":
        _, parent = _matching_continuation_source(a.model, a.dataset, a.validation_dataset,
            "data/processed", "artifacts", a.from_experiment)
        explicit = {"epochs": a.epochs} if a.epochs is not None else {}
        cfg = _continuation_config(a.model, parent.get("config", {}), a.config, explicit)
        overrides = _confirm_training_config(a.model, cfg, accept_defaults=a.accept_config_defaults)
        effective = {**cfg, **overrides}
        result=continue_qwen(a.dataset,validation_dataset=a.validation_dataset,from_experiment=a.from_experiment,
            epochs=int(effective["epochs"]),seed=a.seed,config_path=a.config,
            config_overrides=effective,model=a.model,
            refresh_prediction=a.refresh_prediction,
            refresh_checkpoint_catalog=a.refresh_checkpoint_catalog)
    elif a.command=="models" and a.action=="continue-encoder":
        _, parent = _matching_continuation_source(a.model, a.dataset, a.validation_dataset,
            a.data_dir, a.artifact_dir, a.from_experiment,
            allow_cross_dataset_source=a.allow_cross_dataset_source)
        explicit={key:value for key,value in (("epochs",a.epochs),("learning_rate",a.learning_rate),
            ("learning_rate_decay",a.learning_rate_decay)) if value is not None}
        cfg = _continuation_config(a.model, parent.get("config", {}), a.config, explicit)
        overrides = _confirm_training_config(a.model, cfg, accept_defaults=a.accept_config_defaults)
        effective = {**cfg, **overrides}
        result=continue_encoder(a.dataset,a.model,a.data_dir,a.artifact_dir,a.registry,
            validation_dataset=a.validation_dataset,from_experiment=a.from_experiment,
            epochs=int(effective.get("epochs",20)),
            learning_rate=float(effective.get("learning_rate",5e-5)),
            learning_rate_decay=effective.get("learning_rate_decay",0.9),seed=a.seed,
            optimizer=effective.get("optimizer","adamw_torch"),
            weight_decay=float(effective.get("weight_decay",0.0)),config_path=a.config,
            config_overrides=effective,
            allow_cross_dataset_source=a.allow_cross_dataset_source,
            refresh_prediction=a.refresh_prediction,
            refresh_checkpoint_catalog=a.refresh_checkpoint_catalog)
    elif a.command=="models" and a.action=="kaggle-v0-finetune":
        result=kaggle_v0_finetune(from_experiment=a.from_experiment,
            data_dir=a.data_dir,artifact_dir=a.artifact_dir,epochs=a.epochs,
            learning_rate=a.learning_rate,early_stopping_patience=a.early_stopping_patience,
            seed=a.seed)
    elif a.command=="ensemble": result=evaluate_ensembles(a.dataset,a.artifact_dir,bootstrap_replicates=a.bootstrap_replicates,model_names=a.models,force=a.force)
    elif a.command=="prediction":
        if a.action=="evaluate":
            from .ensemble_registry import read_active_roster, read_ensemble_registry
            from .prediction_factory import evaluate_prediction_strategies
            active = read_active_roster(Path(a.artifact_dir) / "predictions" / "active_model_roster.json")
            history = read_ensemble_registry()
            candidate_ids = ([item["candidate_id"] for item in active["candidates"]]
                             if active else None)
            result=evaluate_prediction_strategies(a.artifact_dir,reference_dataset=a.dataset,
                minimum_support=a.minimum_support,blend_batch_size=a.blend_batch_size,top_k=a.top_k,
                candidate_ids=candidate_ids,
                previous_champion=history["ensembles"][-1] if history["ensembles"] else None)
        elif a.action=="refresh":
            from .ensemble_registry import refresh_prediction_factory
            result=refresh_prediction_factory(artifact_dir=a.artifact_dir,data_dir=a.data_dir,
                reference_dataset=a.dataset,completed_manifest=a.manifest,
                evaluate_if_unchanged=not a.only_if_roster_changed)
        elif a.action=="package":
            from .inference import package_prediction_bundle
            result=package_prediction_bundle(a.manifest,a.output,
                include_base_models=not a.external_base_models)
        else:
            from .inference import export_prediction_bundle
            result=export_prediction_bundle(a.bundle,a.test,a.output)
    elif a.command=="export":
        from .inference import export_submission
        result=export_submission(a.model,a.checkpoint,a.test,a.output)
    elif a.command=="package-kaggle":
        from .inference import package_offline_model
        result=package_offline_model(a.manifest,a.output)
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":main()
