#!/usr/bin/env python3
"""Prepare frozen V0-EXT and complementary V3.0 from Open Pool v16."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/open_pool/v16/ntl_open_pool.parquet"
V0 = ROOT / "data/processed/dataset_v0.parquet"
OUT_EXT = ROOT / "data/processed/dataset_v0_ext.parquet"
OUT_V3 = ROOT / "data/processed/dataset_v3.0.parquet"
PROFILE_EXT = ROOT / "data/processed/dataset_profile_v0_ext.json"
PROFILE_V3 = ROOT / "data/processed/dataset_profile_v3.0.json"
REPORT = ROOT / "docs/v3.0.md"
SEED = 2026


def norm(value):
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def group_key(row):
    canon = norm(row.prompt_canon)
    if canon:
        return "canon:" + canon
    return "fallback:" + json.dumps([norm(getattr(row, c)) for c in
        ("source_dataset", "source_file", "prompt_group")], ensure_ascii=False, separators=(",", ":"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_series(series):
    h = hashlib.sha256()
    for value in series.astype(str):
        h.update(value.encode()); h.update(b"\n")
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace already-frozen output artifacts")
    args = parser.parse_args()
    outputs = [OUT_EXT, OUT_V3, PROFILE_EXT, PROFILE_V3, REPORT]
    if not args.force and any(p.exists() for p in outputs):
        raise FileExistsError("frozen V0-EXT/V3.0 output exists; inspect it or pass --force explicitly")
    raw = pd.read_parquet(SOURCE).reset_index(drop=True)
    v0 = pd.read_parquet(V0).reset_index(drop=True)
    source_columns = list(raw.columns)
    required = {"id", "text", "model", "prompt_canon", "prompt_group", "source_dataset", "source_file", "n_words"}
    if not required.issubset(raw.columns):
        raise ValueError(f"source lacks required columns {sorted(required - set(raw.columns))}")
    if len(raw) != 231036 or v0.shape[0] != 2699:
        raise ValueError(f"unexpected input row counts: source={len(raw)}, V0={len(v0)}")

    raw["_group"] = [group_key(r) for r in raw.itertuples(index=False)]
    # The training/evaluation pipeline requires prompt_id; this is the same stable
    # canonical/fallback key used by the V2.GR.OR16 Open Pool integration.
    raw["prompt_id"] = raw["_group"]
    raw["_textnorm"] = raw.text.map(norm)
    v0_ids = set(v0.id.astype(str))
    v0_texts = set(v0.text.map(norm))
    v0_prompts = set(v0.prompt_id.map(norm))
    raw_group_parts = raw.prompt_group.map(norm)
    raw_canon_parts = raw.prompt_canon.map(norm)
    v0_group_overlap = raw_group_parts.isin(v0_prompts) | raw_canon_parts.isin(v0_prompts)
    v0_overlap = raw.id.astype(str).isin(v0_ids) | raw._textnorm.isin(v0_texts) | v0_group_overlap
    eligible = raw.loc[~v0_overlap & raw.n_words.between(40, 250, inclusive="both")].copy()
    if eligible.empty:
        raise ValueError("no eligible source rows after V0 overlap and length filters")

    # Fit a lightweight CPU topic proxy from response text because the source contains
    # neither source prompt text nor explicit topic annotations. Cluster a seeded sample,
    # then assign the rest in chunks to keep peak memory modest.
    rng = np.random.default_rng(SEED)
    vectorizer = TfidfVectorizer(max_features=12000, min_df=2, max_df=0.9,
                                 stop_words="english", ngram_range=(1, 2), dtype=np.float32)
    sample_size = min(10000, len(eligible))
    sample_pos = rng.choice(len(eligible), size=sample_size, replace=False)
    sample_text = eligible.text.iloc[sample_pos].astype(str).tolist()
    sample_vectors = vectorizer.fit_transform(sample_text)
    n_topics = min(64, max(18, len(eligible) // 1000))
    topic_model = MiniBatchKMeans(n_clusters=n_topics, random_state=SEED,
                                  batch_size=2048, n_init=1, max_iter=20)
    topic_model.fit(sample_vectors)
    topic_ids = np.empty(len(eligible), dtype=np.int16)
    for start in range(0, len(eligible), 8000):
        stop = min(start + 8000, len(eligible))
        vectors = vectorizer.transform(eligible.text.iloc[start:stop].astype(str).tolist())
        topic_ids[start:stop] = topic_model.predict(vectors)
    eligible["_topic"] = topic_ids
    # Reserve the original seeded row-tie draw in the random stream so the frozen
    # selection remains reproducible if only later selection code is refactored.
    rng.random(len(eligible))
    topic_by_group = eligible.groupby("_group", sort=False)._topic.agg(
        lambda values: int(values.value_counts().index[0]))
    grouped = {name: group for name, group in eligible.groupby("_group", sort=False)}
    groups_by_source_topic: dict[str, dict[int, list[str]]] = {}
    for group_name, group in grouped.items():
        topic = int(topic_by_group[group_name])
        for source in group.source_dataset.astype(str).unique():
            groups_by_source_topic.setdefault(source, {}).setdefault(topic, []).append(group_name)
    for by_topic in groups_by_source_topic.values():
        for names in by_topic.values():
            rng.shuffle(names)
    selected_indices, selected_groups = [], set()
    contributor_counts: dict[str, int] = {}
    topic_counts = {topic: 0 for topic in range(n_topics)}
    topic_positions = {(source, topic): 0 for source, by_topic in groups_by_source_topic.items()
                       for topic in by_topic}
    # Round-robin among contributors with available unique groups, then balance their
    # response-text topic clusters. Every selected row has a distinct known prompt group.
    while len(selected_indices) < 4000:
        available_sources, available_by_source = [], {}
        for candidate_source, by_topic in groups_by_source_topic.items():
            topics = [topic for topic, names in by_topic.items()
                      if topic_positions[(candidate_source, topic)] < len(names)]
            if topics:
                available_sources.append(candidate_source)
                available_by_source[candidate_source] = topics
        if not available_sources:
            break
        min_source_count = min(contributor_counts.get(s, 0) for s in available_sources)
        tied_sources = [s for s in available_sources if contributor_counts.get(s, 0) == min_source_count]
        source = str(rng.choice(tied_sources))
        available_topics = available_by_source[source]
        min_topic_count = min(topic_counts[t] for t in available_topics)
        tied_topics = [t for t in available_topics if topic_counts[t] == min_topic_count]
        topic = int(rng.choice(tied_topics))
        names = groups_by_source_topic[source][topic]
        position_key = (source, topic)
        while topic_positions[position_key] < len(names) and names[topic_positions[position_key]] in selected_groups:
            topic_positions[position_key] += 1
        if topic_positions[position_key] >= len(names):
            continue
        group_name = names[topic_positions[position_key]]
        topic_positions[position_key] += 1
        group = grouped[group_name]
        source_rows = group.loc[group.source_dataset.astype(str).eq(source)]
        if source_rows.empty:
            raise ValueError("source-group index points to a group without the contributor")
        row_id = source_rows.index[int(rng.integers(len(source_rows)))]
        selected_indices.append(row_id)
        selected_groups.add(group_name)
        topic_counts[topic] += 1
        contributor_counts[source] = contributor_counts.get(source, 0) + 1
    if len(selected_indices) != 4000:
        raise ValueError(f"only selected {len(selected_indices)} eligible prompt groups; need 4,000")
    ext = raw.loc[selected_indices].drop(columns=["_group", "_textnorm"], errors="ignore").copy()
    # Original source row order is preserved in artifacts.
    ext = ext.sort_index().reset_index(drop=True)

    ext_ids, ext_texts = set(ext.id.astype(str)), set(ext.text.map(norm))
    ext_group_overlap = raw._group.isin(selected_groups)
    v3_mask = ~raw._group.isin(selected_groups) & ~raw.id.astype(str).isin(v0_ids | ext_ids) & ~raw._textnorm.isin(v0_texts | ext_texts) & ~v0_group_overlap
    v3 = raw.loc[v3_mask].drop(columns=["_group", "_textnorm"], errors="ignore").reset_index(drop=True)
    if len(ext) != 4000 or ext.model.nunique() != 18 or ext.model.value_counts().nunique() == 1:
        raise ValueError("V0-EXT must contain exactly 4,000 rows, all 18 labels and unequal label counts")
    if ext.n_words.min() < 40 or ext.n_words.max() > 250 or ext.id.duplicated().any():
        raise ValueError("V0-EXT length or ID integrity check failed")
    if len(v3) + int(ext_group_overlap.sum()) != len(raw):
        # V0 overlap rows form a separate removal category; report exact row accounting below.
        if len(v3) + int(ext_group_overlap.sum()) + int((~ext_group_overlap & v0_overlap).sum()) != len(raw):
            raise ValueError("Open Pool row accounting failed")

    # Cross-split and V0 checks.
    checks = {
        "v0_ext_v0_id_overlap": len(set(ext.id.astype(str)) & v0_ids),
        "v0_ext_v0_normalized_text_overlap": len(set(ext.text.map(norm)) & v0_texts),
        "v0_ext_v0_prompt_overlap": int(v0_group_overlap.iloc[selected_indices].sum()),
        "v3_v0_prompt_overlap": int(v0_group_overlap.loc[v3_mask].sum()),
        "v3_v0_id_overlap": len(set(v3.id.astype(str)) & v0_ids),
        "v3_v0_normalized_text_overlap": len(set(v3.text.map(norm)) & v0_texts),
        "v3_v0_ext_id_overlap": len(set(v3.id.astype(str)) & ext_ids),
        "v3_v0_ext_normalized_text_overlap": len(set(v3.text.map(norm)) & ext_texts),
        "v3_v0_ext_prompt_overlap": len(set(raw.loc[v3_mask, "_group"]) & selected_groups),
    }
    if any(checks[k] for k in checks if k.endswith("overlap")):
        raise ValueError("overlap verification failed: " + json.dumps(checks, sort_keys=True))

    # Persist exact original columns, without internal helper fields.
    OUT_EXT.parent.mkdir(parents=True, exist_ok=True)
    ext.to_parquet(OUT_EXT, index=False, compression="zstd")
    v3.to_parquet(OUT_V3, index=False, compression="zstd")
    source_counts = raw.source_dataset.value_counts().sort_index()
    selected_prompt_counts = ext.source_dataset.value_counts().sort_index()
    accounting = {
        "source_rows": len(raw), "v0_ext_rows": len(ext), "v3_rows": len(v3),
        "removed_v0_overlap_rows": int(v0_overlap.sum()),
        "removed_v0_ext_prompt_group_rows": int(ext_group_overlap.sum()),
        "accounting_identity": "source = V3.0 + all rows in V0-EXT prompt groups + V0-overlapping rows outside those groups",
        "accounting_total": len(v3) + int(ext_group_overlap.sum()) + int((~ext_group_overlap & v0_overlap).sum()),
        "source_contributors": int(raw.source_dataset.nunique()),
        "v0_ext_contributors": int(ext.source_dataset.nunique()),
        "v0_ext_prompt_groups": int(len(selected_groups)),
        "eligible_source_rows_40_250_words_no_v0_overlap": int(len(eligible)),
    }
    labels = sorted(raw.model.unique())
    profile_ext = {"dataset_id":"v0_ext", "name":"V0-EXT — frozen external validation", "rows":len(ext), "seed":SEED,
        "source":"Open Pool v16 original main parquet", "source_path":str(SOURCE.relative_to(ROOT)), "source_sha256":sha(SOURCE),
        "dataset_sha256":sha(OUT_EXT), "row_id_sha256":digest_series(ext.id), "normalized_text_sha256":digest_series(ext.text.map(norm)),
        "label_counts":{k:int(v) for k,v in ext.model.value_counts().reindex(labels,fill_value=0).items()},
        "contributor_counts":{k:int(v) for k,v in ext.source_dataset.value_counts().sort_index().items()},
        "selection":{"word_range_inclusive":[40,250], "prompt_group_rule":"canon:<NFKC/casefold/whitespace prompt_canon>; fallback:<normalized source_dataset, source_file, prompt_group>",
          "one_response_per_prompt_group":True,"topic_diversity":f"greedy contributor round-robin and balancing across {n_topics} MiniBatchKMeans clusters fitted on TF-IDF response text; proxy because the source has no prompt text or topic labels", "topic_cluster_counts":{str(k):int(v) for k,v in sorted(topic_counts.items())}, "contributor_policy":"round-robin among contributors with available unique prompt groups", "seed":SEED},
        "overlap_checks":checks, "accounting":accounting, "preserved_source_columns":source_columns,
        "added_pipeline_columns":["prompt_id"], "frozen":True}
    profile_v3 = {"dataset_id":"v3.0", "name":"V3.0 — Open Pool v16 excluding V0-EXT prompt groups", "rows":len(v3),
        "source":"Open Pool v16 original main parquet", "source_path":str(SOURCE.relative_to(ROOT)), "source_sha256":sha(SOURCE),
        "dataset_sha256":sha(OUT_V3), "row_id_sha256":digest_series(v3.id), "normalized_text_sha256":digest_series(v3.text.map(norm)),
        "label_counts":{k:int(v) for k,v in v3.model.value_counts().reindex(labels,fill_value=0).items()},
        "contributor_counts":{k:int(v) for k,v in v3.source_dataset.value_counts().sort_index().items()},
        "overlap_checks":checks, "accounting":accounting, "preserved_source_columns":source_columns,
        "added_pipeline_columns":["prompt_id"],
        "validation_dataset_ids":["v0","v0_ext"]}
    PROFILE_EXT.write_text(json.dumps(profile_ext,indent=2,sort_keys=True)+"\n")
    PROFILE_V3.write_text(json.dumps(profile_v3,indent=2,sort_keys=True)+"\n")
    matrix = pd.crosstab(v3.source_dataset, v3.model).reindex(columns=labels,fill_value=0)
    matrix["Row total"] = matrix.sum(axis=1)
    matrix.loc["Column total"] = matrix.sum(axis=0)
    lines = ["# V3.0 and V0-EXT dataset preparation", "", "Prepared deterministically from the original Open Pool v16 main parquet. V0-EXT is frozen (seed 2026); no model training was run.", "",
      "## Dataset sizes and row accounting", "", "| Quantity | Rows |", "|---|---:|",
      f"| Open Pool v16 source | {len(raw):,} |", f"| V0-EXT selected responses | {len(ext):,} |",
      f"| Rows in V0-EXT prompt groups removed from V3.0 (including selected rows) | {int(ext_group_overlap.sum()):,} |",
      f"| Additional V0-overlap rows removed outside selected groups | {int((~ext_group_overlap & v0_overlap).sum()):,} |",
      f"| V3.0 retained responses | {len(v3):,} |", f"| Accounting total | {accounting['accounting_total']:,} |", "",
      "Rows in a V0-EXT prompt group include the 4,000 selected rows and any sibling responses. All siblings are excluded from V3.0.", "",
      "## Label distributions", "", "| Label | V0 | V0-EXT | V3.0 |", "|---|---:|---:|---:|"]
    c0=v0.model.value_counts(); ce=ext.model.value_counts(); c3=v3.model.value_counts()
    for label in labels: lines.append(f"| {label} | {int(c0.get(label,0)):,} | {int(ce.get(label,0)):,} | {int(c3.get(label,0)):,} |")
    lines += ["", "## Contributor distributions", "", "| source_dataset | V0-EXT | V3.0 |", "|---|---:|---:|"]
    ce_src=ext.source_dataset.value_counts(); c3_src=v3.source_dataset.value_counts()
    for source in sorted(set(ce_src.index)|set(c3_src.index)): lines.append(f"| {source} | {int(ce_src.get(source,0)):,} | {int(c3_src.get(source,0)):,} |")
    lines += ["", "## V3.0 contributor–label availability matrix", "", matrix.to_markdown(), "",
      "## Integrity and overlap verification", "", "| Check | Overlap rows |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k,v in checks.items()]
    lines += ["", f"- V0-EXT has exactly {len(ext):,} unique responses, all 18 labels, unequal class frequencies, and word counts from {int(ext.n_words.min())} to {int(ext.n_words.max())}.",
      f"- V0-EXT covers {ext.source_dataset.nunique()} of {raw.source_dataset.nunique()} source contributors and {len(selected_groups):,} distinct known prompt groups.",
      f"- V0-EXT uses {sum(v > 0 for v in topic_counts.values())} of {n_topics} response-text topic clusters; these are TF-IDF/MiniBatchKMeans proxies, not source-provided topic labels.",
      f"- V0-EXT selected {ext.source_dataset.nunique()} contributors; contributor counts range from {int(ext.source_dataset.value_counts().min())} to {int(ext.source_dataset.value_counts().max())}. Non-empty cluster sizes range from {min(v for v in topic_counts.values() if v)} to {max(topic_counts.values())}.",
      f"- Source SHA-256: `{profile_ext['source_sha256']}`.", f"- V0-EXT SHA-256: `{profile_ext['dataset_sha256']}`.", f"- V3.0 SHA-256: `{profile_v3['dataset_sha256']}`.",
      "- Prompt-group overlap uses canonical `prompt_canon` when available and the source-dataset/source-file/prompt-group tuple otherwise. Response text overlap uses NFKC normalization, case folding, and whitespace collapse.",
      "- All 30 original Open Pool columns and values are preserved; the pipeline-required `prompt_id` key is added using canonical prompt IDs or the stable source-dataset/source-file/prompt-group fallback.",
      "- The source has no explicit topic field or original prompt text. Topic diversity is proxied with TF-IDF/MiniBatchKMeans clusters over response text; source prompt semantics cannot be independently verified.", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"accounting":accounting,"overlap_checks":checks,"source_sha256":profile_ext["source_sha256"],
      "v0_ext_sha256":profile_ext["dataset_sha256"],"v3_sha256":profile_v3["dataset_sha256"],
      "v0_ext_labels":profile_ext["label_counts"],"v0_ext_contributors":profile_ext["contributor_counts"]},indent=2))


if __name__ == "__main__":
    main()
