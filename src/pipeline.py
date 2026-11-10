"""CLI orchestration for immutable V0 data, model, ensemble, and export runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
                           build_full_v0_external_scope,
                           curate_openpool, curate_v1_all_balanced_deduped,
                           curate_v1_all_clean_compact, prepare, _normal_text)
from .evaluation import (align_predictions, complementarity, evaluate, paired_group_bootstrap)
from .ensemble_factory import ensemble_probabilities
from .feature_factory import load_or_fit, stylometry
from .model_factory import (FROZEN_BACKBONE_MODELS, calculate_max_token_length,
                            train_encoder_fold,
                            train_frozen_backbone_fold, train_qwen_lora_fold,
                            train_catboost_stylometry, train_svc)
from .registry import Registry

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
DEFAULT_MVP_MODELS = ["svc", "catboost_stylometry", "ultrabert380", "qwen06"]
ENSEMBLE_MODEL_CHOICES = [*MODEL_CONFIGS, "qwen06_best_epoch03"]


def _revision():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: return "unknown"


def _read_yaml(path):
    with open(path, encoding="utf-8") as stream: return yaml.safe_load(stream) or {}


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


def _is_training_only_profile(profile):
    """Whether an artifact is an external-only training dataset."""
    is_curated_openpool = str(profile.get("curation_method", "")).startswith("openpool_v11_hard_defect_")
    return is_curated_openpool or profile.get("training_only") is True


def _effective_profile(data_dir, dataset, rows, validation_scope=None):
    if validation_scope == "full_v0":
        scoped_path = Path(data_dir) / f"dataset_profile_{dataset}_full_v0.json"
        if not scoped_path.is_file():
            raise FileNotFoundError(f"full_v0 validation profile was not prepared for {dataset}")
        profile = json.loads(scoped_path.read_text(encoding="utf-8"))
        expected_validation = int(rows.split.eq("validation").sum())
        expected_training = int(rows.split.eq("train").sum())
        if (expected_validation != int(profile.get("validation_rows", -1))
                or expected_training != int(profile.get("training_rows", -1))):
            raise ValueError(f"{dataset} full_v0 profile row counts do not match scoped rows")
        return profile
    profile = json.loads(_profile_path(data_dir, dataset).read_text())
    if dataset == "v0":
        return profile
    if not _is_training_only_profile(profile):
        return profile
    v0_profile = json.loads(_profile_path(data_dir, "v0").read_text())
    training = rows.loc[rows.split.eq("train")].sort_values("id", kind="stable")
    validation = rows.loc[rows.split.eq("validation")].sort_values("id", kind="stable")
    training_ids = sorted(training.id.astype(str))
    validation_ids = sorted(validation.id.astype(str))
    validation_content = validation[[c for c in ("id", "prompt_id", "text", "model") if c in validation]]
    validation_content_hash = hashlib.sha256(
        validation_content.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    components = {"openpool_dataset_sha256": profile["dataset_sha256"],
                  "v0_validation_split_sha256": v0_profile["split_sha256"],
                  "v0_validation_ids_sha256": hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),
                  "v0_validation_content_sha256": validation_content_hash,
                  "openpool_training_ids_sha256": hashlib.sha256("\n".join(training_ids).encode()).hexdigest()}
    profile = dict(profile)
    profile["openpool_dataset_sha256"] = profile["dataset_sha256"]
    profile["split_sha256"] = _sha(components)
    profile["training_dataset_version"] = dataset
    profile["validation_dataset_version"] = "v0"
    profile["training_rows"] = int(len(training))
    profile["validation_rows"] = int(len(validation))
    profile["effective_dataset_components"] = components
    return profile


def _load_dataset(dataset, data_dir, *, validation_scope=None):
    data_dir = Path(data_dir)
    if validation_scope not in (None, "frozen_v0_540", "full_v0"):
        raise ValueError("validation_scope must be 'frozen_v0_540' or 'full_v0'")
    if validation_scope is None and dataset != "v0":
        candidate_profile_path = _profile_path(data_dir, dataset)
        if candidate_profile_path.is_file():
            candidate_profile = json.loads(candidate_profile_path.read_text(encoding="utf-8"))
            external_only = _is_training_only_profile(candidate_profile)
            has_legacy_v0_reference = (candidate_profile.get("validation_dataset_version") == "v0"
                                       and bool(candidate_profile.get("validation_split_sha256")))
            if external_only and not has_legacy_v0_reference:
                validation_scope = "full_v0"
    if validation_scope == "full_v0":
        source_profile_path = _profile_path(data_dir, dataset)
        source_profile = json.loads(source_profile_path.read_text(encoding="utf-8"))
        if not _is_training_only_profile(source_profile):
            raise ValueError(
                f"validation_scope=full_v0 requires an external-only training dataset; "
                f"{dataset} is not marked training_only. Use its external-only source dataset "
                "(for example v2_openpool instead of mixed v2).")
        external = pd.read_parquet(data_dir / f"dataset_{dataset}.parquet")
        v0_rows, _ = _load_dataset("v0", data_dir)
        v0_profile = json.loads(_profile_path(data_dir, "v0").read_text(encoding="utf-8"))
        expected_v0_rows = 2699
        if int(v0_profile.get("rows", -1)) != expected_v0_rows:
            raise ValueError(
                f"official V0 profile must contain {expected_v0_rows} rows; "
                f"found {v0_profile.get('rows')}")
        rows, assignments, scope = build_full_v0_external_scope(
            external, v0_rows,
            expected_classes=len(v0_profile.get("classes", [])),
            expected_v0_rows=expected_v0_rows)
        scope.update({
            "dataset_version": dataset,
            "dataset_sha256": source_profile["dataset_sha256"],
            "external_dataset_sha256": source_profile["dataset_sha256"],
            "source_profile_path": str(source_profile_path),
            "v0_dataset_sha256": v0_profile["dataset_sha256"],
            "v0_source_split_sha256": v0_profile.get("split_sha256"),
            "validation_split_sha256": scope["split_sha256"],
            "validation_dataset_sha256": v0_profile["dataset_sha256"],
            "validation_ids_sha256": scope["validation_ids_sha256"],
            "validation_hashes": {
                "ids_sha256": scope["validation_ids_sha256"],
                "normalized_text_sha256": scope["validation_normalized_text_sha256"],
                "prompt_ids_sha256": scope["validation_prompt_ids_sha256"],
                "prompt_content_sha256": scope["validation_prompt_content_sha256"],
                "training_ids_sha256": scope["training_ids_sha256"],
                "training_normalized_text_sha256": scope["training_dataset_normalized_text_sha256"],
                "training_prompt_content_sha256": scope["training_dataset_prompt_content_sha256"],
                "scope_split_sha256": scope["split_sha256"],
            },
            "classes": v0_profile.get("classes", []),
            "training_only": True,
        })
        profile_path = data_dir / f"dataset_profile_{dataset}_full_v0.json"
        split_path = data_dir / f"splits_{dataset}_full_v0.parquet"
        if profile_path.exists() or split_path.exists():
            if not (profile_path.is_file() and split_path.is_file()):
                raise FileExistsError(f"partial full_v0 scope artifacts exist for {dataset}")
            old_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            old_assignments = pd.read_parquet(split_path)
            if (old_profile != scope or
                    not old_assignments.equals(assignments)):
                raise FileExistsError(f"full_v0 scope artifacts already exist with different inputs for {dataset}")
        else:
            assignments.to_parquet(split_path, index=False)
            profile_path.write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
        return rows, data_dir
    if dataset != "v0" and _is_training_only_profile(json.loads(_profile_path(data_dir, dataset).read_text())):
        openpool = pd.read_parquet(data_dir / f"dataset_{dataset}.parquet")
        v0_rows, _ = _load_dataset("v0", data_dir)
        openpool_profile = json.loads(_profile_path(data_dir, dataset).read_text())
        v0_profile = json.loads(_profile_path(data_dir, "v0").read_text())
        if openpool.id.duplicated().any():
            raise ValueError("Open Pool training IDs must be unique")
        v0_validation = v0_rows.loc[v0_rows.split.eq("validation")].copy()
        validation_ids = sorted(v0_validation.id.astype(str))
        validation_ids_hash = hashlib.sha256("\n".join(validation_ids).encode()).hexdigest()
        if len(v0_validation) != int(openpool_profile.get("validation_rows", -1)):
            raise ValueError(f"{dataset} profile validation row count differs from the frozen V0 validation set")
        if openpool_profile.get("validation_split_sha256") != v0_profile.get("split_sha256"):
            raise ValueError(f"{dataset} profile references a different V0 validation split")
        if openpool_profile.get("validation_ids_sha256") != validation_ids_hash:
            raise ValueError(f"{dataset} profile validation IDs differ from the frozen V0 validation set")
        if set(openpool.id.astype(str)) & set(v0_validation.id.astype(str)):
            raise ValueError("Open Pool IDs overlap the frozen V0 validation IDs")
        if set(openpool.model.astype(str).unique()) != set(v0_validation.model.astype(str).unique()):
            raise ValueError("Open Pool and V0 label sets differ")
        openpool = openpool.copy()
        openpool["split"] = "train"
        combined = pd.concat([openpool, v0_validation], ignore_index=True, sort=False)
        if combined.id.duplicated().any():
            raise ValueError("Open Pool and frozen V0 validation IDs must be unique")
        train_prompts = set(openpool.prompt_id.astype(str))
        validation_prompts = set(v0_validation.prompt_id.astype(str))
        if train_prompts & validation_prompts:
            raise ValueError("Open Pool training rows overlap frozen V0 validation prompt groups")
        return combined.sort_values("id", kind="stable").reset_index(drop=True), data_dir
    rows = pd.read_parquet(data_dir / f"dataset_{dataset}.parquet")
    assignments = pd.read_parquet(data_dir / f"splits_{dataset}.parquet")
    if rows.id.duplicated().any() or assignments.id.duplicated().any(): raise ValueError("dataset and split assignment IDs must be unique")
    if set(rows.id.astype(str)) != set(assignments.id.astype(str)): raise ValueError("dataset and split assignment ID sets differ")
    if "split" in rows.columns:
        inline = rows.set_index("id").split.astype(str).sort_index()
        assigned = assignments.set_index("id").split.astype(str).sort_index()
        if not inline.equals(assigned):
            raise ValueError("inline split values differ from the split assignment artifact")
        rows = rows.drop(columns="split")
    rows = rows.merge(assignments[["id", "split"]], on="id", how="left", validate="one_to_one")
    if rows["split"].isna().any() or set(rows.split) != {"train", "validation"}: raise ValueError("rows need train and validation split assignments")
    return rows, data_dir


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


def _refresh_ensemble_after_training(manifest, manifest_path, *, artifact_dir, data_dir):
    """Run prediction selection before the training command returns to its queue."""
    from .ensemble_registry import refresh_prediction_factory
    project_root = Path(__file__).resolve().parents[1]
    prediction_dir = Path(artifact_dir) / "predictions"
    update = refresh_prediction_factory(
        artifact_dir=artifact_dir, data_dir=data_dir, reference_dataset="v0",
        completed_manifest=manifest_path, evaluate_if_unchanged=True,
        registry_path=project_root / "docs" / "ensembles.md",
        roster_path=prediction_dir / "active_model_roster.json")
    manifest["prediction_factory_update"] = update
    try:
        from .checkpoint_catalog import refresh_catalog
        manifest["checkpoint_catalog_update"] = refresh_catalog(
            artifact_root=artifact_dir, data_dir=data_dir)
    except Exception as exc:
        # Catalog maintenance must not turn a completed training run into a failed run.
        manifest["checkpoint_catalog_update"] = {
            "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    manifest_path = Path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    metrics_path = manifest_path.with_name(manifest_path.name.removesuffix("_manifest.json") + "_metrics.json")
    metrics_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return update


def _model_cfg(model, config_path=None):
    path = Path(config_path or MODEL_CONFIGS[model])
    if not path.exists(): raise FileNotFoundError(f"model config not found: {path}")
    cfg = _read_yaml(path)
    cfg.setdefault("model", model)
    if cfg["model"] != model: raise ValueError(f"model config {path} declares {cfg['model']!r}, expected {model!r}")
    cfg["_config_path"] = str(path)
    if model not in {"svc", "catboost_stylometry"} and (not cfg.get("checkpoint") or not cfg.get("revision")):
        raise ValueError(f"{model} needs an explicit checkpoint and immutable revision in {path}")
    if model not in {"svc", "catboost_stylometry"} and not re.fullmatch(r"[0-9a-fA-F]{40}", str(cfg["revision"])):
        raise ValueError(f"{model} revision in {path} must be a full 40-character commit hash, not a branch or tag")
    if model == "svc" and (float(cfg.get("C", .5)) <= 0 or int(cfg.get("calibration_cv", 3)) < 2):
        raise ValueError(f"SVC config {path} needs C > 0 and calibration_cv >= 2")
    if model == "catboost_stylometry" and (int(cfg.get("iterations", 0)) < 1
            or not 1 <= int(cfg.get("depth", 0)) <= 16 or int(cfg.get("thread_count", 0)) < 1):
        raise ValueError(f"CatBoost config {path} needs positive iterations/thread_count and depth in [1, 16]")
    if model not in {"svc", "catboost_stylometry"} and (int(cfg.get("epochs", 1)) < 1 or int(cfg.get("batch_size", 1)) < 1):
        raise ValueError(f"model config {path} needs positive epochs and batch_size")
    if cfg.get("auto_batch_size") and (int(cfg.get("max_batch_size", cfg.get("batch_size", 1))) < 1
            or not 0 < float(cfg.get("batch_probe_safety_fraction", 0.95)) <= 1):
        raise ValueError(f"model config {path} needs max_batch_size >= 1 and batch_probe_safety_fraction in (0, 1]")
    return cfg


def _train_fold(model, train, valid, classes, cfg, output_dir, seed, fold, cache_dir, *,
                cache_fold=None, checkpoint_selection_metric="accuracy"):
    if model == "svc":
        result = train_svc(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
                           c=cfg.get("C", .5), calibration_cv=cfg.get("calibration_cv", 3), seed=seed,
                           feature_config=cfg.get("features", {"views": ["char", "word"]}),
                           cache_dir=cache_dir, fold_token=cache_fold or f"fold-{fold}",
                           validation_labels=valid.model.tolist())
        checkpoint = Path(output_dir) / "checkpoints" / f"svc_fold{fold}.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": result["model"], "raw_model": result["raw_model"], "features": result["features"], "classes": classes, "config": cfg}, checkpoint)
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
            temperature_calibration=cfg.get("temperature_calibration", True),
            validation_labels=valid.model.tolist(),
            early_stopping_rounds=cfg.get("early_stopping_rounds", 100),
            feature_recipe=feature_config.get("stylometry_recipe"),
            feature_matrices=(x_train.toarray(), x_valid.toarray(),
                              feature_transform.stylometry_feature_names),
            checkpoint_selection_metric=checkpoint_selection_metric,
            calibration_max_samples=cfg.get("calibration_max_samples", 5000),
            calibration_seed=cfg.get("calibration_seed", seed),
            calibration_full_dataset=cfg.get("calibration_full_dataset", False))
        result["cache_key"] = cache_key
        result["checkpoint"] = str(checkpoint)
    elif model == "ultrabert380":
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_encoder_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 3), batch_size=cfg.get("batch_size", 16), max_length=cfg["max_length"],
            learning_rate=cfg.get("learning_rate", 2e-5), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
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
            select_best_checkpoint=(checkpoint_selection_metric == "balanced_accuracy"
                or cfg.get("checkpoint_selection") == "highest_validation_accuracy"),
            checkpoint_selection_metric=checkpoint_selection_metric,
            calibration_max_samples=cfg.get("calibration_max_samples", 5000),
            calibration_seed=cfg.get("calibration_seed", seed),
            calibration_full_dataset=cfg.get("calibration_full_dataset", False))
        result["checkpoint"] = str(checkpoint)
    elif model in FROZEN_BACKBONE_MODELS and cfg.get("training_mode", "frozen") == "frozen":
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_frozen_backbone_fold(model, train.text.tolist(), train.model.tolist(),
            valid.text.tolist(), classes, checkpoint=cfg["checkpoint"], revision=cfg["revision"],
            output_dir=checkpoint, max_length=cfg["max_length"],
            batch_size=cfg.get("batch_size", 8), c=cfg.get("C", .5),
            calibration_cv=cfg.get("calibration_cv", 3), seed=seed,
            fix_mistral_regex=cfg.get("fix_mistral_regex", False),
            validation_labels=valid.model.tolist(),
            train_row_ids=train.id.astype(str).tolist(),
            validation_row_ids=valid.id.astype(str).tolist(),
            embedding_cache_dir=Path(cache_dir).parent / "frozen_embeddings")
        result["checkpoint"] = str(checkpoint)
    elif model in ({"qwen06", "qwen17", "qwen4", "qwen8", "minicpm5_2b", "hunyuan_1_8b", "llama32_3b", "llama31_8b", "ministral3_8b", "smollm3_3b", "granite4_h_micro", "granite4_350m", "phi35_mini"}
                   | FROZEN_BACKBONE_MODELS):
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_qwen_lora_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 3), batch_size=cfg.get("batch_size", 4), max_length=cfg["max_length"],
            learning_rate=cfg.get("learning_rate", 1e-4), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
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
            calibration_max_samples=cfg.get("calibration_max_samples", 5000),
            calibration_seed=cfg.get("calibration_seed", seed),
            calibration_full_dataset=cfg.get("calibration_full_dataset", False))
        result["checkpoint"] = str(checkpoint)
    else: raise ValueError(f"unknown model: {model}")
    return result


def _dataset_max_length(model, cfg, rows, *, checkpoint=None, revision=None):
    """Count all training and validation responses with the training tokenizer."""
    frozen = model in FROZEN_BACKBONE_MODELS and cfg.get("training_mode", "frozen") == "frozen"
    return calculate_max_token_length(
        model, rows.text.astype(str).tolist(), checkpoint=checkpoint or cfg["checkpoint"],
        revision=revision if revision is not None else cfg.get("revision"),
        task_prefix=cfg.get("task_prefix", ""), use_processor=frozen,
        trust_remote_code=cfg.get("trust_remote_code", model == "ultrabert380"),
        fix_mistral_regex=cfg.get("fix_mistral_regex", False),
        local_files_only=cfg.get("local_files_only", True))


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


def _full_v0_continuation_rows(rows):
    """Return every selected external row and the complete official V0 validation set."""
    train = rows.loc[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    valid = rows.loc[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    if train.empty:
        raise ValueError("continuation training dataset has no training rows")
    if len(valid) != 2699:
        raise ValueError(f"full V0 continuation validation requires all 2,699 rows, found {len(valid)}")
    return train, valid


def _require_full_v0_checkpoint(source):
    if source.get("validation_scope") != "full_v0" or int(source.get("validation_rows", 0)) != 2699:
        raise ValueError("continuation requires a source checkpoint selected against all 2,699 V0 rows")


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
        frame.to_parquet(path, index=False)
        epoch_checkpoints = result.get("epoch_checkpoints") or []
        checkpoint_path = (epoch_checkpoints[epoch - 1]
                           if epoch <= len(epoch_checkpoints) else None)
        if checkpoint_path is None and int(result.get("best_epoch") or -1) == epoch:
            checkpoint_path = result.get("best_checkpoint_path") or result.get("best_checkpoint")
        records.append({"epoch": epoch, "path": str(path), "rows": len(frame),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                        "class_order": list(classes),
                        "probability_calibration": "uncalibrated_softmax_from_validation_logits"})
    return records


def build_features(dataset="v0", data_dir="data/processed", config_path="configs/features/default.yaml", cache_dir="artifacts/features"):
    rows, _ = _load_dataset(dataset, data_dir)
    cfg = _read_yaml(config_path)
    train = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    valid = rows[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    x_train, x_val, features, key = load_or_fit(train.text, valid.text, config=cfg, cache_dir=cache_dir, fold_token=f"{dataset}-train-validation")
    report = [{"cache_key": key, "train_shape": list(x_train.shape), "validation_shape": list(x_val.shape)}]
    output = Path(cache_dir) / f"{dataset}_features_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": dataset, "feature_config": cfg, "split": "train_validation", "partitions": report}, indent=2) + "\n")
    return report


def train_models(dataset="v0", model="svc", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, config_path=None, feature_config_path=None, seed=2026, folds=None, config_overrides=None, refresh_prediction=True, validation_scope=None):
    """Train all external rows and validate each epoch on full V0 when available."""
    if folds is not None:
        raise ValueError("fold selection was removed; this pipeline uses one fixed train/validation split")
    cfg = _model_cfg(model, config_path)
    if config_overrides:
        cfg.update(config_overrides)
        _validate_training_overrides(model, cfg)
    external_only = False
    if dataset != "v0":
        source_profile_path = _profile_path(Path(data_dir), dataset)
        if source_profile_path.is_file():
            external_only = _is_training_only_profile(
                json.loads(source_profile_path.read_text(encoding="utf-8")))
    if dataset != "v0" and not external_only:
        raise ValueError(
            f"{dataset} is not an external-only dataset, so it cannot use all V0 for isolated validation. "
            "Use an external-only source such as v1, v1.1, v2_openpool, or v2.0.")
    requested_scope = validation_scope or cfg.get("validation_scope")
    if external_only and requested_scope not in (None, "full_v0"):
        raise ValueError("external-only datasets must use validation_scope=full_v0")
    scope = "full_v0" if external_only else (requested_scope or "frozen_v0_540")
    if scope not in {"frozen_v0_540", "full_v0"}:
        raise ValueError("validation_scope must be 'frozen_v0_540' or 'full_v0'")
    if validation_scope is not None or cfg.get("validation_scope") is not None:
        cfg["validation_scope"] = scope
    if scope == "full_v0":
        cfg["validation_scope"] = scope
    if model in {"svc", "catboost_stylometry"} and feature_config_path:
        cfg["features"] = _read_yaml(feature_config_path)
    rows, data_dir = _load_dataset(dataset, data_dir,
        validation_scope="full_v0" if scope == "full_v0" else None)
    if model not in {"svc", "catboost_stylometry"}:
        cfg["max_length"] = _dataset_max_length(model, cfg, rows)
    profile = _effective_profile(data_dir, dataset, rows,
                                 validation_scope="full_v0" if scope == "full_v0" else None)
    code_revision = _revision()
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_hash = _sha(config_snapshot)
    experiment_id = f"{model}_{dataset}_data{profile['dataset_sha256'][:8]}_split{profile['split_sha256'][:8]}_seed{seed}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    run_artifact_dir = Path(artifact_dir) / "runs" / experiment_id
    pred_dir = Path(artifact_dir) / "predictions"
    prediction_suffix = "full_v0" if scope == "full_v0" else "validation"
    prediction_path = pred_dir / f"{experiment_id}_{prediction_suffix}.parquet"
    manifest_path = pred_dir / f"{experiment_id}_manifest.json"
    if manifest_path.exists() and prediction_path.exists():
        existing = json.loads(manifest_path.read_text())
        if scope != "full_v0" and refresh_prediction and "prediction_factory_update" not in existing:
            _refresh_ensemble_after_training(existing, manifest_path,
                artifact_dir=artifact_dir, data_dir=data_dir)
        return existing
    if run_artifact_dir.exists(): raise FileExistsError(f"incomplete immutable experiment exists at {run_artifact_dir}; remove that generated run directory to retry")
    classes = sorted(rows.model.astype(str).unique().tolist())
    train = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    valid = rows[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    if set(train.prompt_id.astype(str)) & set(valid.prompt_id.astype(str)):
        raise RuntimeError("prompt-group leakage between training and validation")
    if scope == "full_v0":
        for column, normalize in (("id", lambda values: values.astype(str)),
                                  ("text", lambda values: values.astype(str).map(_normal_text)),
                                  ("prompt_id", lambda values: values.astype(str))):
            overlap = set(normalize(train[column])) & set(normalize(valid[column]))
            if overlap:
                raise RuntimeError(f"external training and full V0 validation overlap on {column}: {len(overlap)}")
        if "prompt_canon" in train and "prompt_canon" in valid:
            train_prompts = set(train.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}
            validation_prompts = set(valid.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}
            if train_prompts & validation_prompts:
                raise RuntimeError(
                    "external training and full V0 validation overlap on normalized prompt text: "
                    f"{len(train_prompts & validation_prompts)}")
    metric = "balanced_accuracy" if scope == "full_v0" else "accuracy"
    result = _train_fold(model, train, valid, classes, cfg, run_artifact_dir, seed, "validation", Path(artifact_dir)/"features",
                         cache_fold=f"{dataset}-train-validation",
                         checkpoint_selection_metric=metric)
    scores, probabilities = result["scores"], result["probabilities"]
    if (not np.isfinite(scores).all() or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise RuntimeError("V0 validation output is incomplete or non-finite")
    metrics = evaluate(valid.model.astype(str), scores, classes, probabilities)
    for name in ("validation_loss_raw", "validation_loss_calibrated"):
        if result.get(name) is not None:
            metrics[name] = float(result[name])
    metrics["calibration_temperature"] = result.get("calibration_temperature")
    for key in ("calibration_sample_size", "calibration_full_dataset", "calibration_seed",
                "calibration_raw_loss", "calibration_calibrated_loss"):
        if result.get(key) is not None:
            metrics[key] = result[key]
    pred = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for i, label in enumerate(classes):
        pred[f"score::{label}"] = scores[:, i]
        pred[f"prob::{label}"] = probabilities[:, i]
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(prediction_path, index=False)
    epoch_prediction_records = _save_epoch_validation_predictions(
        result, valid, classes, pred_dir, experiment_id)
    checkpoint = Path(result["checkpoint"])
    artifact_size = sum(f.stat().st_size for f in checkpoint.rglob("*") if f.is_file()) if checkpoint.is_dir() else checkpoint.stat().st_size
    validation_ids = sorted(valid.id.astype(str))
    manifest = {"experiment_id": experiment_id, "experiment_name": config_snapshot.get("experiment_name"),
        "model": model, "dataset": dataset,
        "split": "validation", "validation_scope": scope,
        "validation_dataset_version": "v0" if scope == "full_v0" else profile.get("validation_dataset_version", "v0"),
        "checkpoint_selection_scope": "full_v0" if scope == "full_v0" else "frozen_v0_540",
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
        "runtime_breakdown": result.get("runtime_breakdown"),
        "inference_rows_per_second": (len(valid) / result["inference_seconds"]
            if result.get("inference_seconds") else None),
        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"), "checkpoint_size_bytes": artifact_size,
        "calibration_method": result.get("calibration_method", "sigmoid train-only CV" if model == "svc" else "train-only scalar temperature"),
        "calibration_temperature": result.get("calibration_temperature"),
        "calibration_sample_size": result.get("calibration_sample_size"),
        "calibration_full_dataset": result.get("calibration_full_dataset"),
        "calibration_seed": result.get("calibration_seed"),
        "calibration_raw_loss": result.get("calibration_raw_loss"),
        "calibration_calibrated_loss": result.get("calibration_calibrated_loss"),
        "validation_loss_raw": result.get("validation_loss_raw"),
        "validation_loss_calibrated": result.get("validation_loss_calibrated"),
        "requested_batch_size": result.get("requested_batch_size"),
        "effective_batch_size": result.get("effective_batch_size"),
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
        "early_stopping_scope": "full_v0" if scope == "full_v0" else "frozen_v0_540",
        "early_stopping": result.get("early_stopping"),
        "epoch_checkpoints": result.get("epoch_checkpoints", []),
        "selected_epoch": result.get("best_epoch"),
        "selected_iteration": result.get("selected_iteration"),
        "selected_validation_accuracy": result.get("best_validation_accuracy",
            result.get("selected_validation_accuracy", metrics.get("accuracy"))),
        "baseline_validation_metrics": result.get("baseline_validation_metrics"),
        "checkpoint_scoring_method": result.get("checkpoint_scoring_method"),
        "feature_count": result.get("feature_count"),
        "feature_names": result.get("feature_names"),
        "feature_recipe": result.get("feature_recipe", cfg.get("feature_recipe")),
        "feature_cache_key": result.get("cache_key"), "checkpoint_path": str(checkpoint),
        "prediction_path": str(prediction_path), "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "full_v0_validation": ({
            "dataset_version": "v0", "rows": len(valid),
            "prompt_groups": int(valid.prompt_id.nunique()),
            "ids_sha256": profile.get("validation_ids_sha256"),
            "normalized_text_sha256": profile.get("validation_normalized_text_sha256"),
            "prompt_ids_sha256": profile.get("validation_prompt_ids_sha256"),
            "prompt_content_sha256": profile.get("validation_prompt_content_sha256"),
            "scope_split_sha256": profile.get("split_sha256"),
            "used_for_epoch_metrics_early_stopping_and_checkpoint_choice": True,
        } if scope == "full_v0" else None),
        "metrics": metrics, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    experiment_name = config_snapshot.get("experiment_name")
    description = (f"{experiment_name}: " if experiment_name else "") + (
        f"{model} trained on all {len(train)} {dataset} rows; epoch validation and checkpoint selection used all "
        f"{len(valid)} official V0 rows (balanced accuracy)." if scope == "full_v0" else
        f"{model} trained on {len(train)} rows and validated on {len(valid)} prompt-grouped rows (80/20).")
    Registry(registry_path).add(manifest["experiment_id"], model, "validation", manifest["config"], metrics,
        dataset_hash=manifest["dataset_sha256"], code_revision=manifest["code_revision"], seed=seed,
        runtime_seconds=manifest["fit_seconds"], artifact_path=str(prediction_path), config_hash=manifest["config_sha256"],
        split_hash=manifest["split_sha256"], model_revision=manifest["model_revision"],
        description=description)
    (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _release_completed_training_model(result)
    if refresh_prediction and scope != "full_v0":
        _refresh_ensemble_after_training(manifest, manifest_path,
            artifact_dir=artifact_dir, data_dir=data_dir)
    elif scope == "full_v0":
        # Existing prediction-factory roster artifacts use the historical
        # frozen-540 scope. A full-V0 result must not silently enter that path.
        manifest["prediction_factory_update"] = {
            "status": "skipped", "reason": "full_v0 validation scope is not a 540-row roster candidate"}
        manifest["checkpoint_catalog_update"] = {
            "status": "skipped", "reason": "full_v0 validation scope requires catalog scope support"}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        (pred_dir / f"{experiment_id}_metrics.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


TRAINING_OVERRIDE_FIELDS = (
    "epochs", "early_stopping_patience", "learning_rate", "learning_rate_decay", "lr_scheduler_type", "warmup_ratio",
    "optimizer", "weight_decay", "batch_size", "gradient_accumulation_steps",
    "lora_rank", "lora_alpha", "lora_target_modules", "quantization", "attention_backend", "C", "calibration_cv",
    "iterations", "depth", "l2_leaf_reg", "early_stopping_rounds",
)


def _validate_training_overrides(model, cfg):
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "learning_rate_decay"):
        if name in cfg and cfg[name] is not None and float(cfg[name]) < 0:
            raise ValueError(f"{name} must not be negative")
    for name in ("epochs", "early_stopping_patience", "batch_size", "gradient_accumulation_steps", "lora_rank", "lora_alpha"):
        if name in cfg and int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg.get("lr_scheduler_type") not in {None, "linear", "cosine", "constant", "epoch_decay"}:
        raise ValueError("lr_scheduler_type must be linear, cosine, constant, or epoch_decay")
    if cfg.get("quantization", "none") not in {"none", "qlora4"}:
        raise ValueError("quantization must be 'none' or 'qlora4'")
    if cfg.get("attention_backend") not in {None, "sdpa", "flash_attention_2"}:
        raise ValueError("attention_backend must be 'sdpa', 'flash_attention_2', or unset")
    if cfg.get("optimizer") == "paged_adamw_8bit" and cfg.get("quantization", "none") != "qlora4":
        raise ValueError("paged_adamw_8bit is reserved for qlora4 configurations")
    if model == "svc" and (float(cfg.get("C", 0.5)) <= 0 or int(cfg.get("calibration_cv", 3)) < 2):
        raise ValueError("SVC overrides need C > 0 and calibration_cv >= 2")
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


def continue_qwen(dataset="v0", data_dir="data/processed", artifact_dir="artifacts",
                  registry_path="experiments/registry.sqlite", *, from_experiment=None, epochs=4, seed=2026,
                  config_path=None, config_overrides=None, model="qwen06"):
    """Continue a saved Qwen LoRA model and validate after every additional epoch."""
    from .model_factory import train_qwen_lora_fold
    if model not in {"qwen06", "qwen17", "qwen4", "qwen8"}:
        raise ValueError("continue-qwen supports qwen06, qwen17, qwen4, and qwen8")
    cfg = _model_cfg(model, config_path)
    if config_overrides:
        cfg.update(config_overrides)
        _validate_training_overrides(model, cfg)
    rows, data_dir = _load_dataset(dataset, data_dir, validation_scope="full_v0")
    profile = _effective_profile(data_dir, dataset, rows, validation_scope="full_v0")
    manifests_dir = Path(artifact_dir)/"predictions"
    candidates = []
    for path in manifests_dir.glob(f"{model}_{dataset}_*_manifest.json"):
        record = json.loads(path.read_text())
        if (record.get("model") == model and record.get("split_sha256") == profile["split_sha256"]
                and record.get("validation_scope") == "full_v0"
                and int(record.get("validation_rows", 0)) == 2699):
            candidates.append((path, record))
    if from_experiment:
        candidates = [(p, m) for p, m in candidates if m.get("experiment_id") == from_experiment]
    if not candidates: raise FileNotFoundError("no matching Qwen validation run found to continue")
    source_path, source = max(candidates, key=lambda item: item[1].get("created_at_utc", ""))
    source_checkpoint = _resolve_checkpoint_path(source.get("best_checkpoint_path") or
                                                  source["checkpoint_path"], artifact_dir)
    if not source_checkpoint.is_dir(): raise FileNotFoundError(f"Qwen continuation checkpoint not found: {source_checkpoint}")
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
    cfg["validation_scope"] = "full_v0"
    cfg["checkpoint_selection_metric"] = "balanced_accuracy"
    snapshot = {k:v for k,v in cfg.items() if not k.startswith("_")}
    snapshot["continued_from_experiment"] = source["experiment_id"]
    snapshot["additional_epochs"] = epochs
    config_hash = _sha(snapshot); code_revision = _revision()
    source_key = hashlib.sha256(source["experiment_id"].encode()).hexdigest()[:8]
    experiment_id = f"{model}_{dataset}_continue{epochs}_from{source_key}_seed{seed}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    run_dir = Path(artifact_dir)/"runs"/experiment_id
    if run_dir.exists(): raise FileExistsError(f"continuation run already exists: {run_dir}; choose a new seed or clean generated run")
    train, valid = _full_v0_continuation_rows(rows)
    checkpoint = run_dir/"checkpoints"/f"{model}_continued"
    result = train_qwen_lora_fold(train.text.tolist(),train.model.tolist(),valid.text.tolist(),
        sorted(rows.model.astype(str).unique()),checkpoint=cfg["checkpoint"],revision=cfg["revision"],
        output_dir=checkpoint,epochs=epochs,batch_size=cfg.get("batch_size",4),max_length=cfg["max_length"],
        learning_rate=cfg.get("learning_rate",1e-4),seed=seed,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",1),
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
        calibration_max_samples=cfg.get("calibration_max_samples", 5000),
        calibration_seed=cfg.get("calibration_seed", seed),
        calibration_full_dataset=cfg.get("calibration_full_dataset", False))
    classes=sorted(rows.model.astype(str).unique())
    metrics=evaluate(valid.model.astype(str),result["scores"],classes,result["probabilities"])
    for name in ("validation_loss_raw", "validation_loss_calibrated"):
        if result.get(name) is not None:
            metrics[name] = float(result[name])
    metrics["calibration_temperature"] = result.get("calibration_temperature")
    for key in ("calibration_sample_size", "calibration_full_dataset", "calibration_seed",
                "calibration_raw_loss", "calibration_calibrated_loss"):
        if result.get(key) is not None:
            metrics[key] = result[key]
    metrics["validation_metrics_by_epoch"]=result["validation_metrics_by_epoch"]
    pred=valid[["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy()
    for i,label in enumerate(classes): pred[f"score::{label}"]=result["scores"][:,i];pred[f"prob::{label}"]=result["probabilities"][:,i]
    pred_dir=Path(artifact_dir)/"predictions";pred_dir.mkdir(parents=True,exist_ok=True)
    prediction_path=pred_dir/f"{experiment_id}_validation.parquet";pred.to_parquet(prediction_path,index=False)
    epoch_prediction_records = _save_epoch_validation_predictions(
        result, valid, classes, pred_dir, experiment_id)
    manifest={"experiment_id":experiment_id,"model":model,"dataset":dataset,"split":"full_v0",
        "validation_scope":"full_v0","validation_dataset_version":"v0",
        "continued_from_experiment":source["experiment_id"],"continued_from_checkpoint":str(source_checkpoint),
        "additional_epochs":epochs,"total_epochs":int(source.get("config",{}).get("epochs",0))+epochs,
        "train_rows":len(train),"validation_rows":len(valid),"classes":classes,"config":snapshot,
        "config_sha256":config_hash,"dataset_sha256":profile["dataset_sha256"],"split_sha256":profile["split_sha256"],
        "code_revision":code_revision,"model_revision":cfg["revision"],"checkpoint_id":cfg["checkpoint"],"seed":seed,
        "quantization":result.get("quantization",cfg.get("quantization","none")),
        "quantization_config":result.get("quantization_config"),
        "attention_backend":result.get("attention_backend",cfg.get("attention_backend")),
        "fit_seconds":result["fit_seconds"],"inference_seconds":result["inference_seconds"],
        "runtime_breakdown":result.get("runtime_breakdown"),
        "early_stopping":result.get("early_stopping"),
        "calibration_temperature":result["calibration_temperature"],"validation_metrics_by_epoch":result["validation_metrics_by_epoch"],
        "calibration_sample_size":result.get("calibration_sample_size"),
        "calibration_full_dataset":result.get("calibration_full_dataset"),
        "calibration_seed":result.get("calibration_seed"),
        "calibration_raw_loss":result.get("calibration_raw_loss"),
        "calibration_calibrated_loss":result.get("calibration_calibrated_loss"),
        "validation_loss_raw": result.get("validation_loss_raw"),
        "validation_loss_calibrated": result.get("validation_loss_calibrated"),
        "validation_predictions_by_epoch":epoch_prediction_records,
        "requested_batch_size":result.get("requested_batch_size"),
        "effective_batch_size":result.get("effective_batch_size"),
        "batch_size_probe":result.get("batch_size_probe"),
        "epoch_checkpoints":result.get("epoch_checkpoints",[]),
        "checkpoint_path":str(checkpoint),"prediction_path":str(prediction_path),"metrics":metrics,
        "created_at_utc":pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path=pred_dir/f"{experiment_id}_manifest.json";manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    Registry(registry_path).add(experiment_id,model,"validation",snapshot,metrics,
        dataset_hash=profile["dataset_sha256"],split_hash=profile["split_sha256"],model_revision=cfg["revision"],
        code_revision=code_revision,seed=seed,runtime_seconds=result["fit_seconds"],artifact_path=str(prediction_path),
        config_hash=config_hash,description=f"Continued Qwen LoRA from {source['experiment_id']} for {epochs} epochs; validation metrics recorded each epoch.")
    _release_completed_training_model(result)
    _refresh_ensemble_after_training(manifest, manifest_path,
        artifact_dir=artifact_dir, data_dir=data_dir)
    return manifest


def continue_encoder(dataset="v0", model="ultrabert380", data_dir="data/processed",
                     artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *,
                     from_experiment, epochs=10, learning_rate=5e-5,
                     learning_rate_decay=0.9, seed=2026, optimizer="adamw_torch",
                     weight_decay=0.0):
    """Validate a saved encoder checkpoint, continue training, and keep the best epoch.

    Baseline validation runs before the first optimizer step and is included in the
    run manifest. This path currently supports full-finetuned sequence encoders.
    """
    if model != "ultrabert380":
        raise ValueError("continue-encoder currently supports only ultrabert380; use the model-specific continuation path for LoRA models")
    if epochs < 1 or learning_rate <= 0 or not 0 < learning_rate_decay <= 1:
        raise ValueError("epochs, learning_rate, and learning_rate_decay must be positive; decay must be <= 1")
    manifests_dir = Path(artifact_dir) / "predictions"
    source_path = manifests_dir / f"{from_experiment}_manifest.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"source experiment manifest not found: {source_path}")
    source = json.loads(source_path.read_text())
    if source.get("model") != model:
        raise ValueError(f"source experiment is {source.get('model')}, expected {model}")
    _require_full_v0_checkpoint(source)
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

    rows, data_dir = _load_dataset(dataset, data_dir, validation_scope="full_v0")
    profile = _effective_profile(data_dir, dataset, rows, validation_scope="full_v0")
    train, valid = _full_v0_continuation_rows(rows)
    classes = sorted(rows.model.astype(str).unique().tolist())
    if source.get("classes") != classes:
        raise ValueError("source checkpoint class labels differ from the target dataset")

    source_key = hashlib.sha256(from_experiment.encode()).hexdigest()[:8]
    cfg = source.get("config", {})
    config_snapshot = {
        "model": model, "dataset": dataset, "training_dataset": dataset,
        "continued_from_experiment": from_experiment,
        "epochs": int(epochs), "learning_rate": float(learning_rate),
        "learning_rate_decay": float(learning_rate_decay), "seed": int(seed),
        "optimizer": optimizer, "weight_decay": float(weight_decay),
        "early_stopping_patience": cfg.get("early_stopping_patience", 2),
        "batch_size": int(source.get("config", {}).get("batch_size", 8)),
        "auto_batch_size": bool(source.get("config", {}).get("auto_batch_size", False)),
        "max_batch_size": int(source.get("config", {}).get("max_batch_size", 1024)),
        "batch_probe_safety_fraction": float(source.get("config", {}).get("batch_probe_safety_fraction", 0.95)),
        "gradient_accumulation_steps": int(source.get("config", {}).get("gradient_accumulation_steps", 1)),
        "max_length": _dataset_max_length(model, {
            **cfg, "checkpoint": str(source_checkpoint), "revision": None,
            "task_prefix": "", "training_mode": "full", "trust_remote_code": True,
        }, rows, checkpoint=str(source_checkpoint), revision=None),
        "warmup_ratio": float(source.get("config", {}).get("warmup_ratio", 0.0)),
        "lr_scheduler_type": source.get("config", {}).get("lr_scheduler_type", "linear"),
        "baseline_validation_before_training": True,
        "checkpoint_selection_metric": "balanced_accuracy",
        "validation_scope": "full_v0",
        "checkpoint_selection": "highest_validation_balanced_accuracy",
    }
    config_hash = _sha(config_snapshot)
    experiment_id = (f"{model}_{dataset}_from{source_key}_data{profile['dataset_sha256'][:8]}_"
                    f"split{profile['split_sha256'][:8]}_seed{seed}_cfg{config_hash[:8]}_code{_revision()[:7]}")
    run_dir = Path(artifact_dir) / "runs" / experiment_id
    prediction_path = manifests_dir / f"{experiment_id}_validation.parquet"
    manifest_path = manifests_dir / f"{experiment_id}_manifest.json"
    if manifest_path.exists() and prediction_path.exists():
        existing = json.loads(manifest_path.read_text())
        if "prediction_factory_update" not in existing:
            _refresh_ensemble_after_training(existing, manifest_path,
                artifact_dir=artifact_dir, data_dir=data_dir)
        return existing
    if run_dir.exists():
        raise FileExistsError(f"incomplete continuation run exists at {run_dir}")

    result = train_encoder_fold(
        train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
        checkpoint=source.get("checkpoint_id") or str(source_checkpoint),
        revision=source.get("model_revision"), resume_checkpoint=source_checkpoint,
        output_dir=run_dir / "checkpoints" / f"{model}_continued",
        epochs=epochs, batch_size=config_snapshot["batch_size"],
        max_length=config_snapshot["max_length"], learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay, seed=seed,
        optimizer=optimizer, weight_decay=weight_decay,
        gradient_accumulation_steps=config_snapshot["gradient_accumulation_steps"],
        validation_labels=valid.model.tolist(),
        select_best_checkpoint=True, local_files_only=True,
        early_stopping_patience=cfg.get("early_stopping_patience", 2),
        checkpoint_selection_metric="balanced_accuracy",
        calibration_max_samples=cfg.get("calibration_max_samples", 5000),
        calibration_seed=cfg.get("calibration_seed", seed),
        calibration_full_dataset=cfg.get("calibration_full_dataset", False))
    metrics = evaluate(valid.model.astype(str), result["scores"], classes, result["probabilities"])
    for name in ("validation_loss_raw", "validation_loss_calibrated"):
        if result.get(name) is not None:
            metrics[name] = float(result[name])
    metrics["calibration_temperature"] = result.get("calibration_temperature")
    for key in ("calibration_sample_size", "calibration_full_dataset", "calibration_seed",
                "calibration_raw_loss", "calibration_calibrated_loss"):
        if result.get(key) is not None:
            metrics[key] = result[key]
    pred = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for i, label in enumerate(classes):
        pred[f"score::{label}"] = result["scores"][:, i]
        pred[f"prob::{label}"] = result["probabilities"][:, i]
    manifests_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(prediction_path, index=False)
    epoch_prediction_records = _save_epoch_validation_predictions(
        result, valid, classes, manifests_dir, experiment_id)
    checkpoint = run_dir / "checkpoints" / f"{model}_continued"
    manifest = {
        "experiment_id": experiment_id, "model": model, "dataset": dataset,
        "training_dataset": dataset, "split": "full_v0", "validation_scope": "full_v0",
        "continued_from_experiment": from_experiment,
        "continued_from_checkpoint": str(source_checkpoint),
        "baseline_validation_metrics": result["baseline_validation_metrics"],
        "additional_epochs": int(epochs), "learning_rate": float(learning_rate),
        "learning_rate_decay": float(learning_rate_decay),
        "validation_metrics_by_epoch": result["validation_metrics_by_epoch"],
        "validation_predictions_by_epoch": epoch_prediction_records,
        "early_stopping": result.get("early_stopping"),
        "selected_epoch": result["best_epoch"],
        "selected_validation_accuracy": result["best_validation_accuracy"],
        "checkpoint_selection_metric": result["checkpoint_selection_metric"],
        "checkpoint_scoring_method": result.get("checkpoint_scoring_method"),
        "train_rows": len(train), "validation_rows": len(valid), "classes": classes,
        "dataset_sha256": profile["dataset_sha256"], "split_sha256": profile["split_sha256"],
        "code_revision": _revision(), "model_revision": source.get("model_revision"),
        "checkpoint_id": source.get("checkpoint_id"), "config": config_snapshot,
        "config_sha256": config_hash, "seed": seed,
        "fit_seconds": result["fit_seconds"], "inference_seconds": result["inference_seconds"],
        "runtime_breakdown": result.get("runtime_breakdown"),
        "calibration_temperature": result["calibration_temperature"],
        "calibration_sample_size": result.get("calibration_sample_size"),
        "calibration_full_dataset": result.get("calibration_full_dataset"),
        "calibration_seed": result.get("calibration_seed"),
        "calibration_raw_loss": result.get("calibration_raw_loss"),
        "calibration_calibrated_loss": result.get("calibration_calibrated_loss"),
        "validation_loss_raw": result.get("validation_loss_raw"),
        "validation_loss_calibrated": result.get("validation_loss_calibrated"),
        "checkpoint_path": str(checkpoint), "best_checkpoint_path": result["best_checkpoint_path"],
        "prediction_path": str(prediction_path), "metrics": metrics,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (manifests_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Registry(registry_path).add(experiment_id, model, "validation", config_snapshot, metrics,
        dataset_hash=profile["dataset_sha256"], split_hash=profile["split_sha256"],
        model_revision=source.get("model_revision"), code_revision=manifest["code_revision"],
        seed=seed, runtime_seconds=manifest["fit_seconds"], artifact_path=str(prediction_path),
        config_hash=config_hash,
        description=f"Continued {model} from {from_experiment} for {epochs} epochs after baseline validation.")
    _release_completed_training_model(result)
    _refresh_ensemble_after_training(manifest, manifest_path,
        artifact_dir=artifact_dir, data_dir=data_dir)
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
    processed = (root / "data/processed").resolve()
    artifacts = (root / "artifacts").resolve()
    protected = [(root / "data/raw").resolve(), (Path.home()/".cache/huggingface").resolve()]
    for target in (processed, artifacts):
        if target in protected or root not in target.parents:
            raise RuntimeError(f"refusing to clean unexpected path: {target}")
    for target in (processed, artifacts):
        if target.exists():
            for child in target.iterdir():
                if child.is_symlink() or not child.is_dir(): child.unlink()
                else: shutil.rmtree(child)
        else:
            target.mkdir(parents=True)
    registry = (root / registry_path).resolve()
    if root not in registry.parents or registry in protected:
        raise RuntimeError(f"refusing to remove unexpected registry path: {registry}")
    for suffix in ("", "-wal", "-shm"):
        Path(str(registry) + suffix).unlink(missing_ok=True)
    data_cfg = _read_yaml(root / data_config)
    data_cfg["seed"] = seed
    profile = prepare(root / data_cfg["train_path"], root / data_cfg["output_dir"],
        root / data_cfg["test_path"] if data_cfg.get("test_path") else None,
        seed=seed, validation_fraction=data_cfg.get("validation_fraction", .2),
        candidates=data_cfg.get("split_candidates", 1000), expected_classes=data_cfg.get("expected_classes", 18))
    feature_report = build_features("v0", root / "data/processed", root / feature_config, root / "artifacts/features")
    selected = models or list(DEFAULT_MVP_MODELS)
    model_reports = [train_models("v0", name, root/"data/processed", root/"artifacts", registry,
        config_path=root/MODEL_CONFIGS[name], feature_config_path=root/feature_config if name == "svc" else None, seed=seed)
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
    mix=d.add_parser("assemble");mix.add_argument("--config",default="configs/data/mixed_v1_5.yaml")
    f=sub.add_parser("features").add_subparsers(dest="action",required=True);b=f.add_parser("build");b.add_argument("--dataset",default="v0");b.add_argument("--data-dir",default="data/processed");b.add_argument("--config",default="configs/features/default.yaml")
    m=sub.add_parser("models").add_subparsers(dest="action",required=True)
    t=m.add_parser("train");t.add_argument("--model",choices=MODEL_CONFIGS,default="svc");t.add_argument("--dataset",default="v0");t.add_argument("--data-dir",default="data/processed");t.add_argument("--artifact-dir",default="artifacts");t.add_argument("--config");t.add_argument("--features-config");t.add_argument("--seed",type=int,default=2026);t.add_argument("--validation-scope",choices=["frozen_v0_540","full_v0"],help="for external-only datasets, train on every dataset row and validate/select checkpoints on all 2,699 V0 rows");t.add_argument("--skip-prediction-refresh",action="store_true",help="save the model run without changing the active prediction roster or ensemble history");t.add_argument("--accept-config-defaults",action="store_true",help="noninteractive confirmation to use the model config values")
    cq=m.add_parser("continue-qwen",help="continue the latest Qwen LoRA checkpoint and validate after every epoch")
    cq.add_argument("--dataset",default="v0");cq.add_argument("--epochs",type=int,default=4);cq.add_argument("--seed",type=int,default=2026)
    cq.add_argument("--model",choices=["qwen06","qwen17","qwen4","qwen8"],default="qwen06")
    cq.add_argument("--from-experiment");cq.add_argument("--config");cq.add_argument("--accept-config-defaults",action="store_true")
    ce=m.add_parser("continue-encoder",help="baseline-validate and continue a saved UltraBERT checkpoint")
    ce.add_argument("--model",choices=["ultrabert380"],default="ultrabert380")
    ce.add_argument("--dataset",default="v0");ce.add_argument("--from-experiment",required=True)
    ce.add_argument("--epochs",type=int,default=10);ce.add_argument("--learning-rate",type=float,default=5e-5)
    ce.add_argument("--learning-rate-decay",type=float,default=0.9);ce.add_argument("--seed",type=int,default=2026)
    ce.add_argument("--data-dir",default="data/processed");ce.add_argument("--artifact-dir",default="artifacts")
    ce.add_argument("--registry",default="experiments/registry.sqlite");ce.add_argument("--accept-config-defaults",action="store_true")
    ts=m.add_parser("catboost-two-stage",help="run sequential V1-all to V0 CatBoost feature experiments")
    ts.add_argument("--data-dir",default="data/processed")
    ts.add_argument("--artifact-dir",default="artifacts")
    ts.add_argument("--registry",default="experiments/registry.sqlite")
    ts.add_argument("--ensemble-registry",default="docs/ensembles.md")
    ts.add_argument("--config",default=MODEL_CONFIGS["catboost_stylometry"])
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
    a=parser.parse_args(argv)
    if a.command=="restart": result=restart_mvp(models=a.models,seed=a.seed,bootstrap_replicates=a.bootstrap_replicates)
    elif a.command=="data" and a.action=="prepare": result=prepare_data(a.config)
    elif a.command=="data" and a.action=="curate-openpool": result=curate_openpool_data(a.config)
    elif a.command=="data" and a.action=="curate-balanced-deduped": result=curate_v1_all_balanced_deduped_data(a.config)
    elif a.command=="data" and a.action=="curate-clean-compact": result=curate_v1_all_clean_compact_data(a.config)
    elif a.command=="data" and a.action=="build-v1-3": result=build_v1_3_external_data(a.config)
    elif a.command=="data" and a.action=="assemble": result=assemble_training_dataset(a.config)
    elif a.command=="features": result=build_features(a.dataset,a.data_dir,a.config)
    elif a.command=="models" and a.action=="train":
        cfg = _model_cfg(a.model, a.config)
        if a.model == "svc" and a.features_config:
            cfg["features"] = _read_yaml(a.features_config)
        overrides = _confirm_training_config(a.model, cfg, accept_defaults=a.accept_config_defaults)
        result=train_models(a.dataset,a.model,a.data_dir,a.artifact_dir,config_path=a.config,
            feature_config_path=a.features_config,seed=a.seed,config_overrides=overrides,
            refresh_prediction=not a.skip_prediction_refresh, validation_scope=a.validation_scope)
    elif a.command=="models" and a.action=="catboost-two-stage":
        from .catboost_two_stage import run_catboost_two_stage
        cfg = _model_cfg("catboost_stylometry", a.config)
        result=run_catboost_two_stage(load_dataset=_load_dataset,effective_profile=_effective_profile,
            data_dir=a.data_dir,artifact_dir=a.artifact_dir,registry_path=a.registry,
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
    elif a.command=="models" and a.action=="continue-qwen":
        cfg = _model_cfg(a.model, a.config); cfg["epochs"] = a.epochs
        overrides = _confirm_training_config(a.model, cfg, accept_defaults=a.accept_config_defaults)
        result=continue_qwen(a.dataset,from_experiment=a.from_experiment,
            epochs=int(overrides.get("epochs", a.epochs)),seed=a.seed,config_path=a.config,
            config_overrides=overrides,model=a.model)
    elif a.command=="models" and a.action=="continue-encoder":
        cfg = _model_cfg(a.model); cfg.update({"epochs":a.epochs,"learning_rate":a.learning_rate,
                                                "learning_rate_decay":a.learning_rate_decay})
        overrides = _confirm_training_config(a.model, cfg, accept_defaults=a.accept_config_defaults)
        result=continue_encoder(a.dataset,a.model,a.data_dir,a.artifact_dir,a.registry,
            from_experiment=a.from_experiment,epochs=int(overrides.get("epochs",a.epochs)),
            learning_rate=float(overrides.get("learning_rate",a.learning_rate)),
            learning_rate_decay=float(overrides.get("learning_rate_decay",a.learning_rate_decay)),seed=a.seed,
            optimizer=overrides.get("optimizer",cfg.get("optimizer","adamw_torch")),
            weight_decay=float(overrides.get("weight_decay",cfg.get("weight_decay",0.0))))
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
