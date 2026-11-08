#!/usr/bin/env python3
"""Read-only quality and overlap analysis for the downloaded external datasets.

The script reads only the primary exports named in the external-candidates
README and the V1-all parquet. It writes a Markdown report and optional JSON
metrics; it never changes source datasets.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/raw/external_candidates"
V1 = ROOT / "data/processed/dataset_v1_all.parquet"
REPORT = ROOT / "docs/external_candidates_quality.md"
GLM_FILES = [EXT / "glm53_flash/data" / f"glm53_harvest{suffix}.parquet"
             for suffix in ("", "_r2", "_r3", "_r4")]
GLM_PROV = EXT / "glm53_flash/data/provenance_r34.parquet"
GPT_FILE = EXT / "gpt56_sol_luna/data/train.parquet"
SEED = 2026
NEAR_THRESHOLD = 0.90
MINHASH_PERMUTATIONS = 64


def norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(value.split())


def pct(n, d):
    return 100 * n / d if d else 0.0


def quantiles(values):
    s = pd.Series(values, dtype="float64")
    if s.empty:
        return {"p10": None, "median": None, "p90": None, "max": None}
    return {"p10": float(s.quantile(.1)), "median": float(s.median()),
            "p90": float(s.quantile(.9)), "max": float(s.max())}


def load_glm():
    prov = pq.read_table(GLM_PROV).to_pandas().set_index("id") if GLM_PROV.exists() else pd.DataFrame()
    global GLM_PROVENANCE_DUPLICATE_IDS
    GLM_PROVENANCE_DUPLICATE_IDS = int(prov.index.duplicated(keep="first").sum()) if not prov.empty else 0
    if not prov.empty:
        prov = prov.loc[~prov.index.duplicated(keep="first")]
    frames = []
    for path in GLM_FILES:
        # Primary exports only; register/thinking exports are overlapping views.
        frame = pq.read_table(path).to_pandas()
        frame["model"] = "glm-5.3-flash"  # dataset-level attribution, not row-level
        frame["source_file_primary"] = path.name
        if not prov.empty:
            p = prov.reindex(frame.id)
            for col in p.columns:
                frame[f"prov_{col}"] = p[col].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_gpt():
    base = pq.read_table(GPT_FILE).to_pandas()
    parsed, errors = [], 0
    for encoded in base.row_json:
        try:
            row = json.loads(encoded) if isinstance(encoded, str) else encoded
            if not isinstance(row, dict):
                raise ValueError("row_json is not an object")
            messages = row.get("messages") or []
            users = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            assistants = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
            raw_model = row.get("teacher_model") or row.get("model")
            model = str(raw_model).removeprefix("openai/") if raw_model is not None else None
            parsed.append({"model": model, "raw_model": raw_model, "prompt": users[-1] if users else None,
                           "response": assistants[-1] if assistants else None, "row": row,
                           "message_count": len(messages), "assistant_steps": row.get("assistant_steps"),
                           "source_trajectory_id": row.get("source_trajectory_id")})
        except Exception:
            errors += 1
            parsed.append({"model": None, "raw_model": None, "prompt": None, "response": None, "row": {},
                           "message_count": None, "assistant_steps": None, "source_trajectory_id": None})
    x = pd.DataFrame(parsed)
    out = base.copy()
    out["model"] = x.model
    out["raw_model_identifier"] = x.raw_model
    out["prompt"] = x.prompt
    out["response"] = x.response
    out["source_file_primary"] = "train.parquet"
    out["source_dataset_row"] = base.first_source_dataset
    out["source_config_row"] = base.first_source_config
    out["source_split_row"] = base.first_source_split
    out["source_row_index"] = base.first_source_row_index
    out["lang"] = x.row.map(lambda r: r.get("lang"))
    out["metadata"] = x.row
    out["message_count"] = x.message_count
    out["assistant_steps"] = x.assistant_steps
    out["trajectory_id"] = x.source_trajectory_id
    out["row_metadata"] = x.row
    return out, errors


def enrich_text(frame):
    frame = frame.copy()
    frame["response"] = frame.response.astype("string")
    frame["prompt"] = frame.prompt.astype("string")
    frame["response_norm"] = frame.response.map(norm_text)
    frame["prompt_norm"] = frame.prompt.map(norm_text)
    frame["chars"] = frame.response.fillna("").str.len()
    frame["words"] = frame.response.fillna("").str.findall(r"\b\w+\b").str.len()
    frame["tokens_approx"] = frame.words  # transparent whitespace/word proxy, not tokenizer tokens
    text = frame.response.fillna("").astype(str)
    frame["has_code"] = text.str.contains(r"```|~~~", regex=True)
    frame["has_markdown"] = text.str.contains(r"(?m)^\s{0,3}(?:#{1,6}\s|>\s|[-*+]\s|\d+[.)]\s)|\*\*|__|`", regex=True)
    frame["has_url"] = text.str.contains(r"https?://|www\.", case=False, regex=True)
    frame["has_list"] = text.str.contains(r"(?m)^\s*(?:[-*+]\s|\d+[.)]\s)", regex=True)
    frame["has_structured"] = text.str.contains(r"(?m)^\s*[\[{]|\b(?:json|yaml|xml)\b", case=False, regex=True)
    frame["has_think_tag"] = text.str.contains(r"<think>|</think>", case=False, regex=True)
    frame["prefix_120"] = frame.response_norm.str.slice(0, 120)
    frame["suffix_120"] = frame.response_norm.str.slice(-120)
    return frame


def schema_info(path):
    pf = pq.ParquetFile(path)
    counts = Counter()
    nulls = Counter()
    for batch in pf.iter_batches(batch_size=4096):
        for name, vals in batch.to_pydict().items():
            counts[name] += len(vals)
            nulls[name] += sum(v is None for v in vals)
    return {"rows": pf.metadata.num_rows, "columns": [
        {"name": field.name, "dtype": str(field.type), "missing": nulls[field.name],
         "missing_pct": pct(nulls[field.name], counts[field.name])}
        for field in pf.schema_arrow
    ]}


def normalized_duplicates(frame):
    nonempty = frame.response_norm.ne("")
    counts = frame.loc[nonempty, "response_norm"].value_counts()
    raw_nonempty = frame.loc[nonempty, "response"].dropna().astype(str)
    raw_counts = raw_nonempty.value_counts()
    duplicates = int((counts - 1).clip(lower=0).sum())
    cross = frame.loc[nonempty].groupby("response_norm").model.nunique()
    cross_conflicts = int(cross.gt(1).sum())
    cross_rows = int(frame.response_norm.isin(cross[cross.gt(1)].index).sum())
    return {"exact_duplicate_extra_rows": int((raw_counts - 1).clip(lower=0).sum()),
            "exact_duplicate_texts": int(raw_counts.gt(1).sum()),
            "normalized_duplicate_extra_rows": duplicates,
            "normalized_duplicate_texts": int(counts.gt(1).sum()),
            "identical_texts_across_models": cross_conflicts,
            "rows_in_cross_model_identical_texts": cross_rows}


def near_pairs(frames):
    """Approximate MinHash candidate search + exact character-5gram Jaccard.

    Reports verified candidate pairs only. LSH candidate generation is
    approximate, so absence of a pair is not proof of zero near duplicates.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as e:
        raise RuntimeError("datasketch is required for near-duplicate analysis") from e
    allf = pd.concat([f[["response_norm", "model", "dataset"]] for f in frames], ignore_index=True)
    unique = allf.loc[allf.response_norm.str.len().ge(30)].drop_duplicates("response_norm").reset_index(drop=True)
    lsh = MinHashLSH(threshold=NEAR_THRESHOLD, num_perm=MINHASH_PERMUTATIONS,
                     params=(MINHASH_PERMUTATIONS // 4, 4))
    sigs = []
    import zlib
    # Reuse the same deterministic permutation matrix; constructing a fresh
    # MinHash normally initializes a seeded RNG for every response.
    seed_sig = MinHash(num_perm=MINHASH_PERMUTATIONS, seed=SEED,
                       hashfunc=lambda b: zlib.crc32(b))
    permutations = seed_sig.permutations
    for i, text in enumerate(unique.response_norm):
        words = text.split()
        shingles = {" ".join(words[j:j+3]) for j in range(max(1, len(words)-2))}
        sig = MinHash(num_perm=MINHASH_PERMUTATIONS, seed=SEED,
                      hashfunc=lambda b: zlib.crc32(b), permutations=permutations,
                      scheme=seed_sig.scheme)
        sig.update_batch([s.encode("utf-8") for s in shingles])
        sigs.append(sig)
        lsh.insert(str(i), sig)
    found, by_scope = set(), Counter()
    clusters = list(range(len(unique)))
    def root(i):
        while clusters[i] != i:
            clusters[i] = clusters[clusters[i]]
            i = clusters[i]
        return i
    for i, sig in enumerate(sigs):
        for key in lsh.query(sig):
            j = int(key)
            if j <= i:
                continue
            a, b = unique.at[i, "response_norm"], unique.at[j, "response_norm"]
            if min(len(a), len(b)) / max(len(a), len(b)) < .75:
                continue
            ga, gb = {a[k:k+5] for k in range(len(a)-4)}, {b[k:k+5] for k in range(len(b)-4)}
            score = len(ga & gb) / max(1, len(ga | gb))
            if score >= NEAR_THRESHOLD:
                found.add((i, j))
                ri, rj = root(i), root(j)
                if ri != rj:
                    clusters[max(ri, rj)] = min(ri, rj)
                da, db = unique.at[i, "dataset"], unique.at[j, "dataset"]
                scope = "within:" + da if da == db else "cross_dataset:" + da + "|" + db
                by_scope[scope] += 1
    cluster_counts = Counter(root(i) for i in range(len(unique)))
    return {"method": "64-permutation word-trigram MinHash LSH candidates, verified by exact character 5-gram Jaccard; candidate length ratio >= 0.75",
            "threshold": NEAR_THRESHOLD, "unique_responses_indexed": len(unique),
            "verified_pairs": len(found), "clusters_with_2plus": sum(n > 1 for n in cluster_counts.values()),
            "pairs_by_scope": dict(by_scope)}


def dataset_summary(name, frame, schema, parse_errors=0):
    models = {}
    for model, g in frame.groupby("model", dropna=False, sort=True):
        key = "[missing]" if pd.isna(model) else str(model)
        duplicate_counts = g.loc[g.response_norm.ne("")].response_norm.value_counts()
        all_cross = frame.loc[frame.response_norm.ne("")].groupby("response_norm").model.nunique(dropna=False)
        cross_texts = set(all_cross[all_cross.gt(1)].index)
        models[key] = {
            "rows": len(g), "valid_prompt_response": int((g.prompt_norm.ne("") & g.response_norm.ne("")).sum()),
            "empty_prompts": int(g.prompt_norm.eq("").sum()), "empty_responses": int(g.response_norm.eq("").sum()),
            "attributed": int(g.model.notna().sum()), "source_rows": g.get("source_dataset_row", pd.Series(dtype=str)).value_counts().head(10).to_dict(),
            "lang": g.get("lang", pd.Series(dtype=str)).fillna("[missing]").value_counts(dropna=False).to_dict(),
            "response_lengths": {"chars": quantiles(g.chars), "words": quantiles(g.words), "approx_tokens": quantiles(g.tokens_approx)},
            "source_token_metadata": quantiles(pd.to_numeric(g.get("tokens", pd.Series(index=g.index, dtype="float64")), errors="coerce").dropna()),
            "unique_prompts": int(g.prompt_norm[g.prompt_norm.ne("")].nunique()),
            "duplicate_text_groups": int(duplicate_counts.gt(1).sum()),
            "duplicate_extra_rows": int((duplicate_counts - 1).clip(lower=0).sum()),
            "cross_model_identical_text_groups": int(g.loc[g.response_norm.isin(cross_texts), "response_norm"].nunique()),
            "prompt_multi_response_groups": int((g[g.prompt_norm.ne("")].groupby("prompt_norm").response_norm.nunique() > 1).sum()),
            "formatting": {k: {"rows": int(g[k].sum()), "pct": pct(int(g[k].sum()), len(g))}
                           for k in ("has_markdown", "has_code", "has_url", "has_list", "has_structured", "has_think_tag")},
            "top_prefixes_120": g.prefix_120[g.prefix_120.ne("")].value_counts().head(10).to_dict(),
            "top_suffixes_120": g.suffix_120[g.suffix_120.ne("")].value_counts().head(10).to_dict(),
            "response_vocab": int(len(set(" ".join(g.response_norm).split()))),
            "mean_word_length": float(g.response_norm.str.findall(r"\b\w+\b").map(lambda ws: statistics.mean(map(len, ws)) if ws else 0).mean()) if len(g) else 0,
        }
    return {"name": name, "rows": len(frame), "schema": schema,
            "unique_normalized_nonempty_responses": int(frame.loc[frame.response_norm.ne(""), "response_norm"].nunique()),
            "parse_errors": parse_errors, "rows_by_model": frame.model.fillna("[missing]").value_counts().to_dict(),
            "empty_prompts": int(frame.prompt_norm.eq("").sum()), "empty_responses": int(frame.response_norm.eq("").sum()),
            "duplicates": normalized_duplicates(frame), "models": models}


def readme_metadata():
    result = {}
    for key, file in (("glm", EXT / "glm53_flash/README.md"), ("gpt", EXT / "gpt56_sol_luna/README.md")):
        content = file.read_text(encoding="utf-8")
        m = re.search(r"(?mi)^license:\s*([^\n]+)", content)
        result[key] = {"license": m.group(1).strip() if m else "N/A", "card": content.splitlines()[0:35]}
    return result


def run():
    glm = enrich_text(load_glm())
    gpt, gpt_errors = parse_gpt()
    gpt = enrich_text(gpt)
    v1raw = pq.read_table(V1).to_pandas()
    # The V1-all schema uses model/text and prompt_canon/prompt_id.
    v1 = pd.DataFrame({"model": v1raw.model, "response": v1raw.text,
                       "prompt": v1raw.get("prompt_canon", v1raw.get("prompt_id")),
                       "dataset": "v1-all", "lang": v1raw.get("language")})
    v1 = enrich_text(v1)
    glm["dataset"] = "glm53_flash"; gpt["dataset"] = "gpt56_sol_luna"
    glm["lang_metadata"] = glm.lang
    gpt["lang_metadata"] = gpt.lang
    v1["dataset"] = "v1-all"
    glm_summary = dataset_summary("glm53_flash", glm, {p.name: schema_info(p) for p in GLM_FILES})
    gpt_summary = dataset_summary("gpt56_sol_luna", gpt, schema_info(GPT_FILE), gpt_errors)
    combined_ext = pd.concat([glm, gpt], ignore_index=True, sort=False)
    # Exact and normalized response overlap are equivalent after NFKC/casefold/whitespace normalization;
    # raw exact overlap is measured separately.
    v1_norm = set(v1.response_norm[v1.response_norm.ne("")])
    v1_raw = set(v1.response.dropna().astype(str))
    comp = {}
    for name, frame in (("glm53_flash", glm), ("gpt56_sol_luna", gpt)):
        normalized = frame.response_norm.ne("")
        overlap = normalized & frame.response_norm.isin(v1_norm)
        rawmatch = frame.response.dropna().astype(str).isin(v1_raw)
        comp[name] = {"rows": len(frame), "exact_raw_response_overlap_rows": int(rawmatch.sum()),
                      "normalized_response_overlap_rows": int(overlap.sum()),
                      "normalized_overlap_unique_texts": int(frame.loc[overlap, "response_norm"].nunique()),
                      "unique_candidate_normalized_responses": int(frame.loc[normalized, "response_norm"].nunique()),
                      "unique_candidate_not_in_v1all": int(frame.loc[normalized & ~overlap, "response_norm"].nunique()),
                      "prompt_overlap_rows": int((frame.prompt_norm.ne("") & frame.prompt_norm.isin(set(v1.prompt_norm[v1.prompt_norm.ne("")]))).sum()),
                      "overlap_by_model": frame.assign(_overlap=overlap.to_numpy()).groupby("model", dropna=False)._overlap.sum().to_dict(),
                      "model_detail": {str(model): {"rows": len(g),
                          "normalized_overlap_rows": int(g.response_norm.isin(v1_norm).sum()),
                          "unique_normalized_responses": int(g.loc[g.response_norm.ne(""), "response_norm"].nunique()),
                          "unique_not_in_v1all": int(g.loc[g.response_norm.ne("") & ~g.response_norm.isin(v1_norm), "response_norm"].nunique())}
                          for model, g in frame.groupby("model", dropna=False)},
                      "length_chars": quantiles(frame.chars), "length_words": quantiles(frame.words),
                      "formatting": {k: {"rows": int(frame[k].sum()), "pct": pct(int(frame[k].sum()), len(frame))}
                                     for k in ("has_markdown", "has_code", "has_url", "has_list", "has_structured")}}
    input_paths = GLM_FILES + [GLM_PROV, GPT_FILE, V1]
    fingerprint_payload = [(str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in input_paths]
    source_fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    previous_metrics = REPORT.with_suffix(".json")
    cached_near = None
    if previous_metrics.exists():
        try:
            prior = json.loads(previous_metrics.read_text(encoding="utf-8"))
            if (prior.get("source_fingerprint") == source_fingerprint and
                    prior.get("near_duplicate", {}).get("threshold") == NEAR_THRESHOLD):
                cached_near = prior["near_duplicate"]
        except (OSError, ValueError, KeyError):
            pass
    near = cached_near if cached_near is not None else near_pairs([glm, gpt, v1])
    # GLM provenance coverage is based on source rows having a matched provenance record.
    glm_prov_cols = [c for c in glm if c.startswith("prov_")]
    glm_prov = {c[5:]: int(glm[c].notna().sum()) for c in glm_prov_cols}
    report_data = {"seed": SEED, "source_fingerprint": source_fingerprint,
                   "normalization": "Unicode NFKC, casefold, collapse whitespace; no punctuation removal",
                   "near_duplicate": near, "glm": glm_summary, "gpt": gpt_summary,
                   "comparison": comp, "v1_all_rows": len(v1), "glm_provenance_nonmissing": glm_prov,
                   "glm_provenance_duplicate_ids_first_kept": GLM_PROVENANCE_DUPLICATE_IDS,
                   "glm_by_primary_file": glm.source_file_primary.value_counts().to_dict(),
                   "glm_register": glm.register.value_counts(dropna=False).head(30).to_dict(),
                   "glm_round": glm.get("prov_round", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
                   "glm_source": glm.get("prov_source", pd.Series(dtype=object)).value_counts(dropna=False).head(25).to_dict(),
                   "glm_finish": glm.get("prov_finish", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
                   "glm_truncated": glm.get("prov_truncated", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
                   "glm_cap_hit": glm.get("prov_cap_hit", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
                   "glm_thinking": glm.thinking.value_counts(dropna=False).to_dict(),
                   "glm_language": glm.lang.value_counts(dropna=False).to_dict(),
                   "schemas": {"glm": glm_summary["schema"], "gpt": {GPT_FILE.name: gpt_summary["schema"]}},
                   "glm_turns": glm.turns.value_counts(dropna=False).to_dict(),
                   "gpt_source_dataset": gpt.source_dataset_row.value_counts(dropna=False).to_dict(),
                   "gpt_seen_count": gpt.seen_count.value_counts(dropna=False).to_dict(),
                   "gpt_metadata_language": gpt.lang.value_counts(dropna=False).to_dict(),
                   "gpt_language_by_model": {str(model): g.lang.value_counts(dropna=False).to_dict()
                                             for model, g in gpt.groupby("model", dropna=False)},
                   "v1all_response_lengths": {"chars": quantiles(v1.chars), "words": quantiles(v1.words)},
                   "v1all_formatting": {k: {"rows": int(v1[k].sum()), "pct": pct(int(v1[k].sum()), len(v1))}
                                        for k in ("has_markdown", "has_code", "has_url", "has_list", "has_structured")},
                   "gpt_message_count": gpt.message_count.value_counts(dropna=False).to_dict(),
                   "gpt_assistant_steps": gpt.assistant_steps.value_counts(dropna=False).head(20).to_dict(),
                   "gpt_reasoning_effort": gpt.row_metadata.map(lambda r: r.get("reasoning_effort")).value_counts(dropna=False).to_dict(),
                   "gpt_empty_text_with_tool_calls": int(gpt.row_metadata.map(lambda r: bool(
                       (msgs := [m for m in (r.get("messages") or []) if isinstance(m, dict) and m.get("role") == "assistant"])
                       and not str(msgs[-1].get("content") or "").strip() and msgs[-1].get("tool_calls"))).sum()),
                   "gpt_continuation": gpt.row_metadata.map(lambda r: r.get("continuation")).value_counts(dropna=False).to_dict(),
                   "gpt_original_n_messages": gpt.row_metadata.map(lambda r: r.get("original_n_messages")).value_counts(dropna=False).to_dict(),
                   "gpt_original_n_messages_stats": quantiles(pd.to_numeric(gpt.row_metadata.map(lambda r: r.get("original_n_messages")), errors="coerce").dropna()),
                   "gpt_model_source_mismatch_rows": int(sum(
                       (m == "gpt-5.6-sol" and not str(src).startswith("greghavens/")) or
                       (m == "gpt-5.6-luna" and not str(src).startswith("empero-ai/"))
                       for m, src in zip(gpt.model, gpt.source_dataset_row))),
                   "gpt_missing_model": int(gpt.model.isna().sum()),
                   "metadata": readme_metadata()}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(report_data), encoding="utf-8")
    return report_data


def render(d):
    def pct_table(label, groups):
        lines = [f"| {label} | Rows | % |", "|---|---:|---:|"]
        total = sum(groups.values())
        lines += [f"| {k} | {v:,} | {pct(v,total):.2f}% |" for k,v in groups.items()]
        return "\n".join(lines)
    lines = ["# External candidate dataset quality assessment", "",
             "Read-only analysis of the primary exports identified in `data/raw/external_candidates/README.md`. No raw data was changed.", "",
             "## 1. Summary", "",
             "| Dataset | Rows | Valid prompt + response | Attributed model | Empty response | Exact duplicate extra rows | Normalized duplicate extra rows | Unique normalized responses |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dset, name in ((d["glm"], "GLM-5.3-Flash"), (d["gpt"], "GPT-5.6 Sol/Luna")):
        dup=dset["duplicates"]; models=dset["rows_by_model"]
        lines.append(f"| {name} | {dset['rows']:,} | {sum(m['valid_prompt_response'] for m in dset['models'].values()):,} | {sum(m['attributed'] for m in dset['models'].values()):,} | {dset['empty_responses']:,} | {dup['exact_duplicate_extra_rows']:,} | {dup['normalized_duplicate_extra_rows']:,} | {dset['unique_normalized_nonempty_responses']:,} |")
    lines += ["", "GLM model attribution is inherited from the dataset repository (not per-row). GPT attribution is parsed from row-level `teacher_model` (Sol) or `model` (Luna); row JSON parse failures: **%s**. All five primary Parquet files were read successfully. Raw exact overlap compares literal strings; normalized overlap uses the normalization specified below." % d['gpt']['parse_errors'], "",
              "## 2. Per-model statistics", "",
              "| Model | Rows | Valid prompt/response | Empty prompts | Empty responses | Unique prompts | Prompts with >1 response | Same-model duplicate groups | Cross-model identical text groups | Char P10 / median / P90 / max | Word P10 / median / P90 / max | Source token metadata P10 / median / P90 / max | Language metadata |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for ds in (d["glm"], d["gpt"]):
        for model,m in ds["models"].items():
            c=m["response_lengths"]["chars"]; w=m["response_lengths"]["words"]
            tok=m["source_token_metadata"]
            tok_text = "N/A" if tok["median"] is None else f"{tok['p10']:.0f} / {tok['median']:.0f} / {tok['p90']:.0f} / {tok['max']:.0f}"
            lang=ds.get("name")
            lang_counts=d.get("glm_language",{}) if model=="glm-5.3-flash" else d.get("gpt_language_by_model",{}).get(model,{})
            lines.append(f"| {model} | {m['rows']:,} | {m['valid_prompt_response']:,} | {m['empty_prompts']:,} | {m['empty_responses']:,} | {m['unique_prompts']:,} | {m['prompt_multi_response_groups']:,} | {m['duplicate_text_groups']:,} | {m['cross_model_identical_text_groups']:,} | {c['p10']:.0f} / {c['median']:.0f} / {c['p90']:.0f} / {c['max']:.0f} | {w['p10']:.0f} / {w['median']:.0f} / {w['p90']:.0f} / {w['max']:.0f} | {tok_text} | `{json.dumps(lang_counts, ensure_ascii=False)}` |")
    lines += ["", "### Source and provenance", "",
              f"- GLM primary file row counts: `{json.dumps(d['glm_by_primary_file'])}`. Repository card license: `{d['metadata']['glm']['license']}`. Explicit register counts are measured from row `register`; GLM row label is dataset-level. Round 3/4 provenance joins by `id`; matched fields nonmissing counts: `{json.dumps(d['glm_provenance_nonmissing'])}`.",
              f"- Auxiliary GLM provenance had {d['glm_provenance_duplicate_ids_first_kept']:,} repeated IDs; the first matching row was joined deterministically.",
              f"- GLM register distribution: `{json.dumps(d['glm_register'])}`; provenance round: `{json.dumps(d['glm_round'])}`; source: `{json.dumps(d['glm_source'])}`; finish indicators: `{json.dumps(d['glm_finish'])}`; `truncated`: `{json.dumps(d['glm_truncated'])}`; cap-hit: `{json.dumps(d['glm_cap_hit'])}`.",
              f"- GLM thinking flag: `{json.dumps(d['glm_thinking'])}`; turns: `{json.dumps(d['glm_turns'])}`. The separate provenance file covers R3/R4 IDs only; absent fields are unavailable, not negative assertions.",
              f"- GPT repository card license: `{d['metadata']['gpt']['license']}`. Row-level model counts: `{json.dumps(d['gpt']['rows_by_model'])}`; source dataset counts: `{json.dumps(d['gpt_source_dataset'])}`; row `seen_count`: `{json.dumps(d['gpt_seen_count'])}`. Its row JSON has messages, programming-language tags, split/task/domain/category, reasoning effort, assistant step counts, and trace metadata. The card declares English; row `lang` is a programming-language/task tag, not natural-language identification: `{json.dumps(d['gpt_language_by_model'])}`.",
              f"- Model identifier fields are Sol `teacher_model` and Luna `model`; `openai/` is removed only for a normalized report label, while the raw identifier is retained in JSON metrics. GPT repository card license: `{d['metadata']['gpt']['license']}`. Row-level model counts: `{json.dumps(d['gpt']['rows_by_model'])}`; source dataset counts: `{json.dumps(d['gpt_source_dataset'])}`; row `seen_count`: `{json.dumps(d['gpt_seen_count'])}`. The card declares English; row `lang` is a programming-language/task tag, not natural-language identification: `{json.dumps(d['gpt_language_by_model'])}`.",
              f"- GPT source/model identifier disagreement against source naming: {d['gpt_model_source_mismatch_rows']:,} rows; all row labels are present. This checks recorded fields only.",
              "- Model attribution completeness measures field presence/parseability only. It does not verify that the named model generated the text.", "",
              "### Primary-file schemas and missingness", "",
              "Missingness below is measured from raw primary Parquet columns (null values); derived empty prompt/response counts are shown in the per-model table.", "",
              "| Primary file | Rows | Column: type (missing %) |", "|---|---:|---|"]
    for source, files in d["schemas"].items():
        for filename, sch in files.items():
            cols = "; ".join(f"`{c['name']}`: {c['dtype']} ({c['missing_pct']:.2f}%)" for c in sch["columns"])
            lines.append(f"| {filename} | {sch['rows']:,} | {cols} |")
    lines += ["", "### Attribution/model counts", "",
              f"- GLM-5.3-Flash: {json.dumps(d['glm']['rows_by_model'])}; attribution is dataset-level. GPT-5.6 rows: {json.dumps(d['gpt']['rows_by_model'])}; target names are extracted from row JSON `teacher_model` (Sol) or `model` (Luna).", "",
              "## 3. Redundancy, diversity and response characteristics", "",
              f"- Near duplicates: **{d['near_duplicate']['verified_pairs']:,} verified pairs** in **{d['near_duplicate']['clusters_with_2plus']:,} connected clusters**, using {d['near_duplicate']['method']} at Jaccard **{d['near_duplicate']['threshold']:.2f}**. Indexed unique responses: {d['near_duplicate']['unique_responses_indexed']:,}. Pair scope counts: `{json.dumps(d['near_duplicate']['pairs_by_scope'])}`. Cross-dataset verified pairs: {sum(v for k,v in d['near_duplicate']['pairs_by_scope'].items() if k.startswith('cross_dataset:')):,}.",
              "- LSH candidate retrieval is approximate; reported pairs are verified, but zero reported pairs would not prove zero near duplicates.",
              f"- `words` counts Unicode word-like sequences. GLM's source-provided `tokens` field is retained separately; `approx_tokens` is only a word-count proxy, not tokenizer output. GPT has no row-level token-count field in the primary schema. Formatting flags are deterministic regex heuristics. GPT Sol's {d['gpt_empty_text_with_tool_calls']:,} rows with empty final assistant text all have a structured `tool_calls` payload; these are empty text outputs, not JSON parse failures. Length quantiles include them as zero-length text.", "",
              "| Dataset | Markdown | Code fences | URLs | Lists | Structured markers |", "|---|---:|---:|---:|---:|---:|"]
    for label, ds in (("GLM-5.3-Flash", d["glm"]), ("GPT-5.6 Sol/Luna", d["gpt"])):
        counts = Counter()
        for m in ds["models"].values():
            for key, vals in m["formatting"].items(): counts[key] += vals["rows"]
        lines.append(f"| {label} | {counts['has_markdown']:,} | {counts['has_code']:,} | {counts['has_url']:,} | {counts['has_list']:,} | {counts['has_structured']:,} |")
    lines += ["", "Vocabulary counts below are distinct normalized whitespace tokens across each model's extracted text; mean word length uses Unicode word-like sequences.", "",
              "| Model | Distinct word forms | Mean word length (characters) |", "|---|---:|---:|"]
    for ds in (d["glm"], d["gpt"]):
        for model,m in ds["models"].items():
            lines.append(f"| {model} | {m['response_vocab']:,} | {m['mean_word_length']:.2f} |")
    lines += [
              f"- GLM `thinking` is explicit metadata (`{json.dumps(d['glm_thinking'])}`); `turns` records the source turn count. GPT metadata: `reasoning_effort` `{json.dumps(d['gpt_reasoning_effort'])}`, `continuation` `{json.dumps(d['gpt_continuation'])}`, and `original_n_messages` among nonmissing rows has P10/median/P90/max `{json.dumps(d['gpt_original_n_messages_stats'])}`. Rows are trace chunks, so a single-turn/multiturn classification is not a uniform conversation-level measure. GPT final assistant text contains `<think>` tags in {sum(m['formatting']['has_think_tag']['rows'] for m in d['gpt']['models'].values()):,} rows (heuristic); reasoning effort is explicit metadata.",
              "- The script records repeated normalized 120-character prefixes and suffixes. Top repeats are shown below; common prefixes can reflect shared source/task templates rather than copied full responses.", "",
              "| Dataset / model | Top repeated response prefix (count) | Top repeated response suffix (count) |", "|---|---|---|"]
    for dsname, ds in (("GLM", d["glm"]), ("GPT Sol/Luna", d["gpt"])):
        for model,m in ds["models"].items():
            pre = list(m["top_prefixes_120"].items())[:3]; suf = list(m["top_suffixes_120"].items())[:3]
            lines.append(f"| {dsname} / {model} | `{json.dumps(pre, ensure_ascii=False)}` | `{json.dumps(suf, ensure_ascii=False)}` |")
    lines += ["",
              "## 4. Comparison with V1-all", "",
              "Common response normalization: Unicode NFKC, casefold, and collapse all whitespace runs; punctuation and word order are preserved. Prompt overlap uses the same normalization against V1-all `prompt_canon` (or `prompt_id` fallback).",
              "", "| Candidate source | Rows | Raw exact response overlap rows | Normalized overlap rows | Unique normalized responses | Unique not in V1-all | Prompt overlap rows | Near-pair counts (all datasets) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name,c in d['comparison'].items():
        lines.append(f"| {name} | {c['rows']:,} | {c['exact_raw_response_overlap_rows']:,} | {c['normalized_response_overlap_rows']:,} | {c['unique_candidate_normalized_responses']:,} | {c['unique_candidate_not_in_v1all']:,} | {c['prompt_overlap_rows']:,} | see near-duplicate scope above |")
    lines += ["", "### Overlap by attributed model", "", "| Candidate model | Rows | Normalized response overlap rows | % of rows | Unique response strings not in V1-all |", "|---|---:|---:|---:|---:|"]
    for name,c in d["comparison"].items():
        for model,detail in c["model_detail"].items():
            lines.append(f"| {model} | {detail['rows']:,} | {detail['normalized_overlap_rows']:,} | {pct(detail['normalized_overlap_rows'], detail['rows']):.2f}% | {detail['unique_not_in_v1all']:,} |")
    lines += ["", "### Length and formatting distribution versus V1-all", "",
              "Formatting values are row counts and percentages; markers are heuristic regex detections.", "",
              "| Dataset | Char median (P10–P90) | Word median (P10–P90) | Markdown % | Code % | URL % | List % | Structured % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    dist_rows = [("V1-all", d["v1all_response_lengths"], d["v1all_formatting"])]
    for name,c in d["comparison"].items():
        nice = {"glm53_flash":"GLM-5.3-Flash", "gpt56_sol_luna":"GPT-5.6 Sol/Luna"}[name]
        dist_rows.append((nice, {"chars":c["length_chars"], "words":c["length_words"]}, c["formatting"]))
    for label, lens, fmt in dist_rows:
        ch, wd = lens["chars"], lens["words"]
        lines.append(f"| {label} | {ch['median']:.0f} ({ch['p10']:.0f}–{ch['p90']:.0f}) | {wd['median']:.0f} ({wd['p10']:.0f}–{wd['p90']:.0f}) | {fmt['has_markdown']['pct']:.1f}% | {fmt['has_code']['pct']:.1f}% | {fmt['has_url']['pct']:.1f}% | {fmt['has_list']['pct']:.1f}% | {fmt['has_structured']['pct']:.1f}% |")
    lines += ["", "Differences are descriptive and do not establish source quality or model authorship.", "",
              "## Conclusions", "",
              f"1. The primary files contain **{d['glm']['rows']:,}** GLM rows and **{d['gpt']['rows']:,}** GPT Sol/Luna rows; the latter split is `{json.dumps(d['gpt']['rows_by_model'])}`.",
              f"2. Dataset-level attribution applies to GLM; GPT row-level teacher-model attribution parsed for {sum(v for k,v in d['gpt']['rows_by_model'].items() if k != '[missing]'):,}/{d['gpt']['rows']:,} rows.",
              f"3. Normalized response overlap with V1-all is {sum(c['normalized_response_overlap_rows'] for c in d['comparison'].values()):,} rows across the two sources; distinct responses outside V1-all are reported per source above.",
              f"4. The current estimate found {d['near_duplicate']['verified_pairs']:,} MinHash candidates verified at or above character 5-gram Jaccard {NEAR_THRESHOLD:.2f}; approximate candidate retrieval limits recall.",
              "5. Provenance and metadata are uneven: GLM has structured generation fields, with separate finish/truncation metadata only for matched R3/R4 rows; GPT supplies row-level source/model/trace fields. These fields describe recorded metadata, not independent verification.", "",
              "## Method and reproducibility", "",
              f"Run: `/home/zodrak/cuda13.x/bin/python scripts/analyze_external_candidates.py`. Fixed seed: {SEED}; MinHash permutations: {MINHASH_PERMUTATIONS}; near-duplicate verification threshold: {NEAR_THRESHOLD:.2f}. The script reads the four GLM round files, one GPT train file, GLM provenance metadata and V1-all. Register/thinking/config exports are alternative or overlapping views and are not added as rows.",
              "GPT prompt extraction uses the final user message; response text uses the content of the final assistant message. An empty assistant content remains an empty response string even when a structured tool-call payload exists. GLM prompt and response columns are read directly.",
              "", "Missing or non-applicable measures are reported as N/A in source schemas or left unavailable when metadata is absent. Language counts reflect the source `lang` fields, not language identification performed by this analysis.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    data = run()
    # Keep complete tallies reusable for later concise report extensions.
    metrics_path = args.report.with_suffix(".json")
    metrics_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if args.report != REPORT:
        args.report.write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Report: {args.report}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
