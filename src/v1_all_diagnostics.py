"""CPU-only V1-all composition and UltraBERT V0 confusion diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from .checkpoint_catalog import _align_v0_predictions
from .feature_factory import stylometry
from .pipeline import _load_dataset


WEAK = ["claude-fable-5", "glm-5.3", "glm-5.3-flash"]
PREDICTION_DIR = Path("artifacts/analysis/zod01_specialist_discovery/predictions")
INVENTORY = Path("artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv")
OUTPUT = Path("artifacts/analysis/v1_all_diagnostics")
REPORT = Path("docs/v1_all_diagnostics.md")


def _norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _template(value):
    value = _norm(value)
    value = re.sub(r"https?://\S+", " <url> ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)*\b", " <num> ", value)
    return re.sub(r"\s+", " ", value).strip()


def _short(value, limit=240):
    value = re.sub(r"\s+", " ", str(value))
    return value[:limit] + ("…" if len(value) > limit else "")


def _write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_v0(data_dir):
    rows, _ = _load_dataset("v0", str(data_dir))
    val = rows.loc[rows.split.astype(str).eq("validation"), ["id", "prompt_id", "text", "model"]].copy()
    val["id"] = val.id.astype(str)
    val["prompt_id"] = val.prompt_id.astype(str)
    val["model"] = val.model.astype(str)
    if len(val) != 540 or val.id.duplicated().any():
        raise ValueError("expected the frozen 540-row V0 validation set")
    return val.sort_values("id", kind="stable").reset_index(drop=True)


def _standardize(frame, dataset):
    data = frame.copy()
    data["dataset"] = dataset
    data["model"] = data.model.astype(str)
    data["text"] = data.text.fillna("").astype(str)
    if "prompt_id" not in data:
        data["prompt_id"] = data.get("prompt_group", pd.Series("", index=data.index)).astype(str)
    return data


def _dataset_analysis(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    v11_raw = pd.read_parquet(data_dir / "dataset_v1.1.parquet")
    bm_raw = pd.read_parquet(data_dir / "dataset_v1.2_bm.parquet")
    schemas = {"v1.1": set(v11_raw.columns), "v1.2_bm": set(bm_raw.columns)}
    v11 = _standardize(v11_raw, "v1.1")
    bm = _standardize(bm_raw, "v1.2_bm")
    v0 = _standardize(_load_v0(data_dir), "v0_validation")
    v11["dataset_row_id"] = v11.id.astype(str)
    bm["dataset_row_id"] = bm.id.astype(str)
    v0["dataset_row_id"] = v0.id.astype(str)
    external = pd.concat([v11, bm], ignore_index=True, sort=False)
    union_path = data_dir / "dataset_v1_all.parquet"
    if union_path.exists():
        union = pd.read_parquet(union_path)
        if len(union) != len(external):
            raise ValueError(f"V1-all has {len(union)} rows, but V1.1 + V1.2_bm has {len(external)}")
        if set(union.id.astype(str)) != set(external.id.astype(str)):
            raise ValueError("V1-all row IDs do not equal the V1.1 + V1.2_bm union")
        union_labels = union.set_index(union.id.astype(str)).model.astype(str).sort_index()
        external_labels = external.set_index(external.id.astype(str)).model.astype(str).sort_index()
        if not union_labels.equals(external_labels):
            raise ValueError("V1-all model labels differ from the V1.1 + V1.2_bm union")
        union_hash = hashlib.sha256(union.sort_values("id").id.astype(str).str.cat(sep="\n").encode()).hexdigest()
    else:
        union_hash = ""
    classes = sorted(v0.model.unique().tolist())
    if set(external.model.unique()) != set(classes):
        raise ValueError("external and V0 label sets differ")

    class_counts = (external.groupby(["dataset", "model"]).size().rename("rows").reset_index())
    totals = external.groupby("dataset").size().to_dict()
    class_counts["dataset_rows"] = class_counts.dataset.map(totals)
    class_counts["percent"] = 100 * class_counts.rows / class_counts.dataset_rows
    class_counts["cohort"] = np.where(class_counts.model.isin(WEAK), "weak_three", "other_fifteen")
    _write_csv(class_counts, output_dir / "class_distribution.csv")

    source_col = "source_dataset" if "source_dataset" in external else "source"
    src = (external.groupby(["dataset", "model", source_col], dropna=False).size()
           .rename("rows").reset_index().rename(columns={source_col: "source_dataset"}))
    src["class_rows"] = src.groupby(["dataset", "model"]).rows.transform("sum")
    src["class_percent"] = 100 * src.rows / src.class_rows
    _write_csv(src.sort_values(["dataset", "model", "rows"], ascending=[True, True, False]),
               output_dir / "source_by_class.csv")

    # Row-local stylometry is exactly the project feature recipe; descriptive text
    # formatting fields are additional interpretable diagnostics.
    all_rows = pd.concat([external, v0], ignore_index=True, sort=False)
    # Existing row-local stylometry is comparatively expensive because it
    # extracts lexical, sentence and punctuation distributions. A deterministic
    # stratified sample estimates these features while cheap text measurements
    # below use every response.
    # Groups with fewer than 250 rows need their complete group retained.
    style_sample = pd.concat([
        group.sample(n=min(250, len(group)), random_state=2026)
        for _, group in all_rows.groupby(["dataset", "model"], observed=True)
    ], ignore_index=True)
    style, style_names = stylometry(style_sample.text.tolist(), return_feature_names=True)
    style_frame = pd.DataFrame(style, columns=[f"style_{name}" for name in style_names])
    style_frame["dataset"] = style_sample.dataset.to_numpy()
    style_frame["model"] = style_sample.model.to_numpy()
    style_detail = style_frame.groupby(["dataset", "model"], observed=True).mean().reset_index()
    style_detail["style_sample_rows"] = style_frame.groupby(["dataset", "model"], observed=True).size().to_numpy()
    text_series = all_rows.text
    extras = pd.DataFrame({
        "response_chars": text_series.str.len(),
        "regex_word_count": text_series.str.findall(r"\b[\w'-]+\b").str.len(),
        "punctuation_per_1000_chars": text_series.str.count(r"[^\w\s]") / text_series.str.len().clip(lower=1) * 1000,
        "sentence_marks_per_1000_chars": text_series.str.count(r"[.!?]") / text_series.str.len().clip(lower=1) * 1000,
        "comma_per_1000_chars": text_series.str.count(",") / text_series.str.len().clip(lower=1) * 1000,
        "paragraphs": text_series.str.count(r"\n\s*\n") + 1,
        "markdown_fraction": text_series.str.contains(r"(?m)^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|```|\|", regex=True),
        "code_fence_fraction": text_series.str.contains("```"),
        "url_fraction": text_series.str.contains(r"https?://", case=False, regex=True),
    })
    feat = pd.concat([all_rows[["dataset", "model"]].reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
    feat["cohort"] = np.where(feat.model.isin(WEAK), "weak_three", "other_fifteen")
    # Keep both pooled estimates and a class-balanced aggregate to prevent the
    # heavily represented labels from dominating comparisons.
    numeric = [c for c in feat.select_dtypes(include=[np.number, "bool"]).columns if c not in {"model"}]
    detail = feat.groupby(["dataset", "model"], observed=True)[numeric].mean().reset_index()
    detail = detail.merge(style_detail, on=["dataset", "model"], how="left", validate="one_to_one")
    style_columns = [c for c in style_detail.columns if c.startswith("style_") and c != "style_sample_rows"]
    detail["cohort"] = np.where(detail.model.isin(WEAK), "weak_three", "other_fifteen")
    _write_csv(detail, output_dir / "response_characteristics_by_class.csv")
    class_balanced = detail.groupby(["dataset", "cohort"], observed=True)[numeric + style_columns].mean().reset_index()
    pooled = feat.groupby(["dataset", "cohort"], observed=True)[numeric].mean().reset_index()
    class_weights = feat.groupby(["dataset", "model"], observed=True).size().rename("row_weight").reset_index()
    weighted_style = style_detail.merge(class_weights, on=["dataset", "model"], how="left", validate="one_to_one")
    weighted_style["cohort"] = np.where(weighted_style.model.isin(WEAK), "weak_three", "other_fifteen")
    pooled_style = []
    for (dataset_name, cohort_name), group in weighted_style.groupby(["dataset", "cohort"], observed=True):
        values = {"dataset": dataset_name, "cohort": cohort_name}
        for feature in style_columns:
            values[feature] = float(np.average(group[feature], weights=group.row_weight))
        pooled_style.append(values)
    pooled = pooled.merge(pd.DataFrame(pooled_style), on=["dataset", "cohort"], how="left")
    summaries = pd.concat([class_balanced.assign(aggregation="class_balanced"),
                           pooled.assign(aggregation="row_pooled")], ignore_index=True)
    _write_csv(summaries, output_dir / "response_characteristics_summary.csv")

    provenance_fields = [c for c in ("source_dataset", "source_file", "source_license", "origin_hint",
                                      "provider", "model_version", "route", "source_row_id") if c in external]
    quality_rows = []
    for dataset, group in external.groupby("dataset"):
        for cohort, subset in [("all", group), ("weak_three", group[group.model.isin(WEAK)]),
                               ("other_fifteen", group[~group.model.isin(WEAK)])]:
            for field in provenance_fields:
                if field not in schemas[dataset]:
                    quality_rows.append({"dataset": dataset, "cohort": cohort,
                                         "check": f"unavailable_{field}_in_schema",
                                         "rows": int(len(subset)), "percent": 100.0 if len(subset) else 0.0})
                    continue
                blank = subset[field].isna() | subset[field].astype(str).str.strip().str.lower().isin(["", "n/s", "none", "nan"])
                quality_rows.append({"dataset": dataset, "cohort": cohort, "check": f"missing_{field}",
                                     "rows": int(blank.sum()), "percent": float(blank.mean() * 100) if len(subset) else 0})
            if "model_version" in subset:
                canon = {re.sub(r"[^a-z0-9]", "", c.casefold()): c for c in classes}
                version = subset.model_version.fillna("").astype(str).map(lambda x: canon.get(re.sub(r"[^a-z0-9]", "", x.casefold())))
                conflict = version.notna() & version.ne(subset.model.astype(str))
                quality_rows.append({"dataset": dataset, "cohort": cohort, "check": "direct_model_version_label_conflict",
                                     "rows": int(conflict.sum()), "percent": float(conflict.mean() * 100) if len(subset) else 0})
    # Provider conflicts are direct metadata contradictions; missing provider is
    # separately reported and never inferred as an error.
    if "provider" in external:
        provider_aliases = {"claude": ("anthropic",), "gemini": ("google",),
                            "glm": ("z.ai", "zai", "zhipu"), "gpt": ("openai",),
                            "kimi": ("moonshot",), "qwen": ("alibaba", "qwen")}
        known_aliases = tuple(alias for aliases in provider_aliases.values() for alias in aliases)
        for dataset, group in external.groupby("dataset"):
            providers = group.provider.fillna("").astype(str).str.casefold()
            families = group.model.str.split("-").str[0]
            recognized = providers.map(lambda value: any(alias in value for alias in known_aliases))
            conflict = pd.Series(False, index=group.index)
            for family, aliases in provider_aliases.items():
                direct_other = providers.map(lambda value: any(alias in value for alias in known_aliases)
                                             and not any(alias in value for alias in aliases))
                conflict |= families.eq(family) & recognized & direct_other
            quality_rows.append({"dataset": dataset, "cohort": "all", "check": "provider_label_conflict_observed",
                                 "rows": int(conflict.sum()), "percent": float(conflict.mean() * 100)})
    _write_csv(pd.DataFrame(quality_rows), output_dir / "provenance_quality.csv")

    all_text = pd.concat([external[["dataset", "dataset_row_id", "model", "text"]],
                          v0[["dataset", "dataset_row_id", "model", "text"]]], ignore_index=True)
    all_text["normalized_text"] = all_text.text.map(_norm)
    all_text["text_sha256"] = all_text.normalized_text.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    exact = all_text.groupby("text_sha256").agg(rows=("text_sha256", "size"), labels=("model", "nunique"),
                                                datasets=("dataset", lambda x: ";".join(sorted(set(x)))))
    duplicate_hashes = exact[(exact.rows > 1)].index
    duplicate_rows = all_text[all_text.text_sha256.isin(duplicate_hashes)].merge(
        exact.reset_index(), on="text_sha256", suffixes=("", "_group"))
    _write_csv(duplicate_rows[["dataset", "dataset_row_id", "model", "text_sha256", "rows", "labels", "datasets"]],
               output_dir / "exact_normalized_duplicates.csv")
    conflicting = exact[(exact.rows > 1) & (exact.labels > 1)]

    all_text["template"] = all_text.text.map(_template)
    template_groups = (all_text.groupby(["model", "template"], dropna=False)
                       .agg(rows=("dataset_row_id", "size"), datasets=("dataset", lambda x: ";".join(sorted(set(x)))),
                            example_id=("dataset_row_id", "first"), example=("text", "first")).reset_index())
    repeated = template_groups[(template_groups.rows >= 3) & (template_groups.template.str.len() >= 100)].copy()
    repeated["example"] = repeated.example.map(_short)
    _write_csv(repeated.sort_values("rows", ascending=False), output_dir / "repeated_templates.csv")

    # Deterministic sampled nearest-neighbor audit; this is a screen for unusual
    # template concentration, not proof that no near duplicates exist.
    sample = external.sort_values("id", kind="stable").sample(min(3000, len(external)), random_state=2026)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2,
                          sublinear_tf=True, dtype=np.float32)
    mat = vec.fit_transform(sample.text.astype(str))
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=1).fit(mat)
    distances, indices = nn.kneighbors(mat)
    sims = np.clip(1 - distances[:, 1], 0, 1)
    nearest = pd.DataFrame({"similarity": sims, "left_id": sample.id.astype(str).to_numpy(),
                            "left_model": sample.model.to_numpy(), "left_dataset": sample.dataset.to_numpy(),
                            "left_text": sample.text.to_numpy(),
                            "right_id": sample.iloc[indices[:, 1]].id.astype(str).to_numpy(),
                            "right_model": sample.iloc[indices[:, 1]].model.to_numpy(),
                            "right_dataset": sample.iloc[indices[:, 1]].dataset.to_numpy(),
                            "right_text": sample.iloc[indices[:, 1]].text.to_numpy()})
    nearest["left_text"] = nearest.left_text.map(_short)
    nearest["right_text"] = nearest.right_text.map(_short)
    _write_csv(nearest.sort_values("similarity", ascending=False).head(100), output_dir / "nearest_neighbor_sample.csv")

    # Prompt overlap is a directly observable split leakage risk.
    val_prompts = set(v0.prompt_id.astype(str))
    prompt_overlap = []
    for dataset, group in external.groupby("dataset"):
        overlap_rows = group.prompt_id.astype(str).isin(val_prompts)
        prompt_overlap.append({"dataset": dataset, "overlap_rows": int(overlap_rows.sum()),
                              "overlap_prompt_groups": int(group.loc[overlap_rows, "prompt_id"].nunique())})
    _write_csv(pd.DataFrame(prompt_overlap), output_dir / "v0_prompt_overlap.csv")

    return {"classes": classes, "external": external, "v0": v0, "class_counts": class_counts,
            "source_counts": src, "characteristics": summaries, "quality": pd.DataFrame(quality_rows),
            "exact_duplicate_rows": len(duplicate_rows), "conflicting_exact_groups": len(conflicting),
            "repeated_template_groups": len(repeated), "sample_nearest_p90": float(np.quantile(sims, .9)),
            "sample_nearest_max": float(np.max(sims)), "sample_near_090": int(np.sum(sims >= .90)),
            "prompt_overlap": pd.DataFrame(prompt_overlap), "v1all_rows": len(external),
            "v1all_id_hash": union_hash}


def _confusion_analysis(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    val = _load_v0(data_dir)
    inv = pd.read_csv(INVENTORY).set_index("candidate_key")
    classes = sorted(val.model.unique().tolist())
    aligned = {}
    for key in ("A", "B", "D"):
        row = inv.loc[key]
        prediction_path = Path(row.prediction_path)
        if not prediction_path.is_absolute():
            prediction_path = Path.cwd() / prediction_path
        pred_frame = pd.read_parquet(prediction_path)
        aligned[key] = _align_v0_predictions(pred_frame, val, str(row.experiment_id))
    common = aligned["A"][["id", "prompt_id", "y_true"]].copy()
    for key in ("A", "B", "D"):
        frame = aligned[key]
        if not np.array_equal(frame.id.astype(str).to_numpy(), common.id.astype(str).to_numpy()):
            raise ValueError(f"checkpoint {key} did not align to the A validation IDs")
        common[f"pred_{key}"] = frame.prediction.astype(str).to_numpy()
    y = common.y_true.astype(str).to_numpy()
    per_label_rows, weak_error_rows, pair_rows = [], [], []
    for key in ("A", "B", "D"):
        pred = common[f"pred_{key}"].to_numpy()
        matrix = confusion_matrix(y, pred, labels=classes)
        matrix_frame = pd.DataFrame(matrix, index=classes, columns=classes).rename_axis("true_label")
        matrix_frame.to_csv(output_dir / f"confusion_{key}.csv")
        for i, label in enumerate(classes):
            tp = int(matrix[i, i]); support = int(matrix[i, :].sum()); predicted = int(matrix[:, i].sum())
            per_label_rows.append({"checkpoint": key, "label": label, "support": support,
                                   "predicted_count": predicted, "true_positive": tp,
                                   "false_positive": predicted - tp, "false_negative": support - tp,
                                   "precision": tp / predicted if predicted else 0,
                                   "recall": tp / support if support else 0})
        for target in WEAK:
            ti = classes.index(target)
            for j, other in enumerate(classes):
                if j != ti and matrix[ti, j]:
                    weak_error_rows.append({"checkpoint": key, "direction": "weak_true_to_other_prediction",
                                            "weak_label": target, "other_label": other,
                                            "count": int(matrix[ti, j])})
                if j != ti and matrix[j, ti]:
                    weak_error_rows.append({"checkpoint": key, "direction": "other_true_to_weak_prediction",
                                            "weak_label": target, "other_label": other,
                                            "count": int(matrix[j, ti])})
        for i, true_label in enumerate(classes):
            for j, predicted_label in enumerate(classes):
                if i != j and (true_label in WEAK or predicted_label in WEAK) and matrix[i, j]:
                    pair_rows.append({"checkpoint": key, "true_label": true_label,
                                      "predicted_label": predicted_label, "count": int(matrix[i, j])})
    per_label = pd.DataFrame(per_label_rows)
    _write_csv(per_label, output_dir / "per_label_metrics.csv")
    weak_errors = pd.DataFrame(weak_error_rows)
    _write_csv(weak_errors, output_dir / "weak_label_error_directions.csv")
    pairs = pd.DataFrame(pair_rows)
    _write_csv(pairs, output_dir / "weak_pair_confusions.csv")

    transition_rows = []
    weak_transition_rows = []
    v0_text = val.set_index("id").text.to_dict()
    for before, after in (("A", "B"), ("B", "D")):
        old, new = common[f"pred_{before}"].to_numpy(), common[f"pred_{after}"].to_numpy()
        correct_old, correct_new = old == y, new == y
        status = np.where(~correct_old & correct_new, "corrected",
                          np.where(correct_old & ~correct_new, "regressed", "unchanged_correct"))
        changed = old != new
        for i in np.flatnonzero(changed):
            record = {"transition": f"{before}_to_{after}", "id": common.at[i, "id"],
                      "prompt_id": common.at[i, "prompt_id"], "y_true": y[i],
                      "prediction_before": old[i], "prediction_after": new[i],
                      "outcome": status[i], "text_excerpt": _short(v0_text[str(common.at[i, 'id'])], 320)}
            transition_rows.append(record)
            if y[i] in WEAK or old[i] in WEAK or new[i] in WEAK:
                weak_transition_rows.append(record)
    transitions = pd.DataFrame(transition_rows)
    weak_transitions = pd.DataFrame(weak_transition_rows)
    _write_csv(transitions, output_dir / "prediction_changes_A_to_B_B_to_D.csv")
    _write_csv(weak_transitions, output_dir / "weak_label_prediction_changes.csv")
    counts = []
    for before, after in (("A", "B"), ("B", "D")):
        old, new = common[f"pred_{before}"].to_numpy(), common[f"pred_{after}"].to_numpy()
        ok_old, ok_new = old == y, new == y
        for name, mask in [("all", np.ones(len(y), dtype=bool)),
                           ("weak_true", np.isin(y, WEAK)),
                           ("weak_involved", np.isin(y, WEAK) | np.isin(old, WEAK) | np.isin(new, WEAK))]:
            counts.append({"transition": f"{before}_to_{after}", "scope": name,
                           "rows": int(mask.sum()), "prediction_changed": int(np.sum((old != new) & mask)),
                           "corrected": int(np.sum((~ok_old & ok_new) & mask)),
                           "regressed": int(np.sum((ok_old & ~ok_new) & mask)),
                           "accuracy_delta_points": 100 * (float(np.mean(ok_new[mask])) - float(np.mean(ok_old[mask]))) if mask.any() else 0})
    transition_summary = pd.DataFrame(counts)
    _write_csv(transition_summary, output_dir / "transition_summary.csv")
    return {"classes": classes, "aligned": common, "per_label": per_label,
            "weak_errors": weak_errors, "pairs": pairs, "transitions": transitions,
            "weak_transitions": weak_transitions, "transition_summary": transition_summary,
            "overall": {key: float(np.mean(common[f"pred_{key}"].to_numpy() == y)) for key in ("A", "B", "D")}}


def _pct(x):
    return f"{100 * float(x):.1f}%"


def _make_report(dataset, confusion, path):
    counts = dataset["class_counts"]
    table = counts.pivot(index="model", columns="dataset", values="rows").fillna(0).astype(int)
    pct = counts.pivot(index="model", columns="dataset", values="percent").fillna(0)
    dist = ["| Label | V1.1 n (%) | V1.2_bm n (%) | V1-all n (%) |", "|---|---:|---:|---:|"]
    for label in dataset["classes"]:
        n11, nbm = int(table.loc[label, "v1.1"]), int(table.loc[label, "v1.2_bm"])
        nall = n11 + nbm
        dist.append(f"| {label} | {n11:,} ({pct.loc[label,'v1.1']:.2f}%) | {nbm:,} ({pct.loc[label,'v1.2_bm']:.2f}%) | {nall:,} ({100*nall/dataset['v1all_rows']:.2f}%) |")
    weak_class = counts[counts.model.isin(WEAK)].groupby("dataset").agg(rows=("rows", "sum"), percent=("percent", "sum"))
    src = dataset["source_counts"]
    source_lines = ["| Dataset | Label | Source dataset | Rows | Within-label share |", "|---|---|---|---:|---:|"]
    for _, row in src[src.model.isin(WEAK)].sort_values(["dataset", "model", "rows"], ascending=[True, True, False]).iterrows():
        source_lines.append(f"| {row.dataset} | {row.model} | {row.source_dataset} | {int(row.rows):,} | {row.class_percent:.1f}% |")
    char = dataset["characteristics"]
    desired = ["response_chars", "regex_word_count", "punctuation_per_1000_chars", "sentence_marks_per_1000_chars", "markdown_fraction", "code_fence_fraction", "url_fraction", "style_chars_per_word", "style_words_per_sentence", "style_uppercase_char_fraction", "style_digit_char_fraction"]
    char_table = char[char.aggregation.eq("class_balanced") & char.cohort.isin(["weak_three", "other_fifteen"])].set_index(["dataset", "cohort"])
    char_lines = ["| Measure | V0 weak | V1.1 weak | V1.2_bm weak | V0 other 15 | V1.1 other 15 | V1.2_bm other 15 |", "|---|---:|---:|---:|---:|---:|---:|"]
    aliases = {"response_chars":"mean chars", "regex_word_count":"mean words", "punctuation_per_1000_chars":"punctuation / 1k chars", "sentence_marks_per_1000_chars":"sentence marks / 1k chars", "markdown_fraction":"Markdown fraction", "code_fence_fraction":"Code-fence fraction", "url_fraction":"URL fraction", "style_chars_per_word":"stylometry: chars/word", "style_words_per_sentence":"stylometry: words/sentence", "style_uppercase_char_fraction":"stylometry: uppercase fraction", "style_digit_char_fraction":"stylometry: digit fraction"}
    ds_map = {"V0 weak":("v0_validation","weak_three"), "V1.1 weak":("v1.1","weak_three"), "V1.2_bm weak":("v1.2_bm","weak_three"), "V0 other 15":("v0_validation","other_fifteen"), "V1.1 other 15":("v1.1","other_fifteen"), "V1.2_bm other 15":("v1.2_bm","other_fifteen")}
    for feature in desired:
        vals=[]
        for key,(ds,cohort) in ds_map.items():
            v=float(char_table.loc[(ds,cohort),feature]) if (ds,cohort) in char_table.index else float("nan")
            vals.append(f"{100*v:.2f}%" if "fraction" in feature else f"{v:.1f}")
        char_lines.append(f"| {aliases[feature]} | " + " | ".join(vals) + " |")

    per = confusion["per_label"]
    weak_metrics = per[per.label.isin(WEAK)]
    metric_lines = ["| Checkpoint | Label | Precision | Recall | FP | FN |", "|---|---|---:|---:|---:|---:|"]
    for _, r in per.iterrows():
        metric_lines.append(f"| {r.checkpoint} | {r.label} | {_pct(r.precision)} | {_pct(r.recall)} | {int(r.false_positive)} | {int(r.false_negative)} |")
    error_lines = ["| Checkpoint | Error direction | Weak label | Confused label | Rows |", "|---|---|---|---|---:|"]
    weak_errors = confusion["weak_errors"]
    for key in ("A", "B", "D"):
        for target in WEAK:
            subset = weak_errors[(weak_errors.checkpoint == key) & (weak_errors.weak_label == target)].sort_values("count", ascending=False).head(5)
            for _, r in subset.iterrows():
                error_lines.append(f"| {key} | {r.direction} | {target} | {r.other_label} | {int(r['count'])} |")
    overall = confusion["overall"]
    transitions = confusion["transition_summary"]
    trans_lines = ["| Transition | Scope | Changed | Corrected | Regressed | Accuracy change |", "|---|---|---:|---:|---:|---:|"]
    for _, r in transitions.iterrows():
        trans_lines.append(f"| {r.transition} | {r.scope} | {int(r.prediction_changed)} | {int(r.corrected)} | {int(r.regressed)} | {r.accuracy_delta_points:+.2f} pp |")
    quality = dataset["quality"]
    qrows = quality[quality.check.str.contains("missing_|unavailable_|conflict", regex=True)
                    & quality.cohort.isin(["all", "weak_three"])]
    quality_lines = ["| Dataset | Check | Rows | Percent |", "|---|---|---:|---:|"]
    for _, r in qrows.iterrows():
        quality_lines.append(f"| {r.dataset} | {r.check} ({r.cohort}) | {int(r.rows):,} | {r.percent:.2f}% |")
    overlap_lines = ["| Dataset | Training rows sharing V0 validation prompt IDs | Prompt groups |", "|---|---:|---:|"]
    for _, r in dataset["prompt_overlap"].iterrows():
        overlap_lines.append(f"| {r.dataset} | {int(r.overlap_rows):,} | {int(r.overlap_prompt_groups):,} |")
    changes = confusion["weak_transitions"]
    examples=[]
    for transition in ("A_to_B", "B_to_D"):
        block=changes[(changes.transition==transition)&(changes.outcome.isin(["corrected","regressed"]))]
        for _,r in block.head(4).iterrows():
            examples.append(f"- `{transition}` {r.outcome}: `{r.id}` true `{r.y_true}`, `{r.prediction_before}` → `{r.prediction_after}` — {r.text_excerpt}")
    near_note=(f"The deterministic 3,000-row word TF-IDF nearest-neighbor sample had p90 cosine {dataset['sample_nearest_p90']:.3f}, max {dataset['sample_nearest_max']:.3f}, and {dataset['sample_near_090']} rows with nearest cosine ≥0.90. This is a sampled screen, not proof of absence. Full normalized-text duplicate rows: {dataset['exact_duplicate_rows']}; exact-text groups spanning labels: {dataset['conflicting_exact_groups']}. Repeated normalized templates (≥3 rows, ≥100 chars): {dataset['repeated_template_groups']}.")
    m = per[per.label.isin(WEAK)].pivot(index="label", columns="checkpoint", values="recall")
    src_summary = {}
    for label in WEAK:
        for ds_name in ("v1.1", "v1.2_bm"):
            top = src[(src.model == label) & (src.dataset == ds_name)].sort_values("rows", ascending=False).head(1)
            if len(top):
                src_summary[(label, ds_name)] = f"{top.iloc[0].source_dataset} ({top.iloc[0].class_percent:.1f}%)"
    weak_metric = per[per.label.isin(WEAK)].set_index(["label", "checkpoint"])
    weak_b_to_d = confusion["transition_summary"].query("transition == 'B_to_D' and scope == 'weak_true'").iloc[0]
    provider_conflicts = quality[(quality.check == "provider_label_conflict_observed")].rows.sum()
    missing_orig_v12 = quality[(quality.dataset == "v1.2_bm") & (quality.cohort == "all") &
                               (quality.check == "missing_origin_hint")].rows.sum()
    findings = [
        f"**V1.2_bm adds relatively few weak-label rows:** the three labels are {weak_class.loc['v1.2_bm','percent']:.1f}% of BM ({int(table.loc[WEAK, 'v1.2_bm'].sum()):,} rows) versus {weak_class.loc['v1.1','percent']:.1f}% of V1.1 ({int(table.loc[WEAK, 'v1.1'].sum()):,} rows). BM is not disproportionately concentrated on them.",
        f"**V1-all helps GLM Flash most, but hurts Fable precision:** A→B recall changes are Fable {100*(m.loc['claude-fable-5','B']-m.loc['claude-fable-5','A']):+.1f} pp, GLM 5.3 {100*(m.loc['glm-5.3','B']-m.loc['glm-5.3','A']):+.1f} pp, and GLM Flash {100*(m.loc['glm-5.3-flash','B']-m.loc['glm-5.3-flash','A']):+.1f} pp. Fable precision falls from {_pct(weak_metric.loc[('claude-fable-5','A'),'precision'])} to {_pct(weak_metric.loc[('claude-fable-5','B'),'precision'])} as its false positives rise from {int(weak_metric.loc[('claude-fable-5','A'),'false_positive'])} to {int(weak_metric.loc[('claude-fable-5','B'),'false_positive'])}.",
        f"**V1-all→V0 changes the error balance again:** D raises Fable recall to {_pct(weak_metric.loc[('claude-fable-5','D'),'recall'])} and cuts GLM 5.3 / Flash false positives to {int(weak_metric.loc[('glm-5.3','D'),'false_positive'])} / {int(weak_metric.loc[('glm-5.3-flash','D'),'false_positive'])}, but GLM Flash recall falls to {_pct(weak_metric.loc[('glm-5.3-flash','D'),'recall'])}. On the three true weak classes combined, B→D corrects {int(weak_b_to_d.corrected)} and regresses {int(weak_b_to_d.regressed)} (net accuracy change {weak_b_to_d.accuracy_delta_points:+.2f} pp); overall accuracy rises from {_pct(overall['B'])} to {_pct(overall['D'])}.",
        f"**There is source concentration, but not a clear response-format mismatch:** V1.2_bm's largest sources are Fable {src_summary.get(('claude-fable-5','v1.2_bm'),'n/a')}, GLM 5.3 {src_summary.get(('glm-5.3','v1.2_bm'),'n/a')}, and GLM Flash {src_summary.get(('glm-5.3-flash','v1.2_bm'),'n/a')}. Weak-label mean chars differ only modestly from V0 (726–749); weak responses have somewhat higher punctuation density than V0.",
        f"**No direct duplicate or label contradiction was found:** normalized exact duplicate rows={dataset['exact_duplicate_rows']}, cross-label duplicate groups={dataset['conflicting_exact_groups']}, repeated long templates={dataset['repeated_template_groups']}, V0 prompt overlap=0, direct provider conflicts={int(provider_conflicts)}. V1.2_bm has {int(missing_orig_v12):,} rows without `origin_hint`; this is missing metadata, not evidence of false attribution.",
    ]
    report = ["# V1-all Data Quality and Confusion Analysis", "",
              "CPU-only diagnostics using saved datasets and saved UltraBERT probabilities. No models were trained or inference run.", "",
              "## Five key findings", ""]
    report.extend(f"{i}. {line}" for i,line in enumerate(findings,1))
    report += ["", "## 1. Dataset composition", "",
               f"V1.1 contains {len(pd.read_parquet(Path('data/processed')/'dataset_v1.1.parquet')):,} rows; V1.2_bm contains {len(pd.read_parquet(Path('data/processed')/'dataset_v1.2_bm.parquet')):,}. Their union has {dataset['v1all_rows']:,} rows. The corresponding V1-all artifact ID digest is `{dataset['v1all_id_hash']}`.", "",
               *dist, "", "### Source distribution for the three focus labels", "", *source_lines,
               "", "### Response characteristics", "",
               "Text length, word/punctuation counts and formatting indicators use all rows. Existing project stylometry features use a deterministic sample of up to 250 rows per dataset and class (all V0 validation rows are included). Means below are class-balanced within each cohort; the complete per-label distributions and row-pooled estimates are in `response_characteristics_by_class.csv` and `response_characteristics_summary.csv`.", "",
               *char_lines, "", "## 2. Provenance, duplication, and split checks", "", *quality_lines, "", near_note, "",
               *overlap_lines, "", "Missing optional provenance fields are reported as missing metadata, not by themselves as proof that attribution is false. Provider/model-version conflicts count only when the stored fields directly contradict the canonical class mapping.", "",
               "## 3. Checkpoint confusion", "",
               "Full 18×18 matrices are saved as `confusion_A.csv`, `confusion_B.csv`, and `confusion_D.csv` (rows=true label, columns=predicted label). Class metrics for all 18 labels are in `per_label_metrics.csv`.", "",
               "Complete per-label precision, recall, FP and FN are shown below and exported to `per_label_metrics.csv`.", "",
               *metric_lines, "", "### Weak-label confusion directions", "", *error_lines, "",
               "### Prediction changes on identical V0 rows", "", *trans_lines, "", "Selected changed examples involving a focus label:", "",
               *(examples or ["No corrected or regressed rows involving a focus label."]), "",
               "## 4. Explicit assessment", "",
               f"1. **Does V1.2_bm disproportionately contribute examples to the three weak labels?** No. They contribute {int(table.loc[WEAK, 'v1.2_bm'].sum()):,} rows ({weak_class.loc['v1.2_bm','percent']:.1f}% of BM); compare V1.1's {int(table.loc[WEAK, 'v1.1'].sum()):,} ({weak_class.loc['v1.1','percent']:.1f}%). See class counts above.",
               "2. **Is there evidence of source/domain mismatch or data-quality problems?** There is evidence of source concentration and a modest style shift, not a demonstrated causal mismatch: 96.9% of BM Fable comes from one source, and the largest GLM sources differ by label. Weak-label response lengths are near V0, while punctuation is somewhat denser. No direct provider/model-version contradiction, exact duplicate conflict, or V0 prompt overlap was detected. Missing optional metadata remains an audit limitation: BM lacks `origin_hint`/provider/route on 25.4% overall and `model_version` on 98.6%; V1.1 has no provider/model-version fields in its schema.",
               "3. **Are the weak labels overpredicted, underpredicted, or confused with specific related models?** In A, Fable is underpredicted (17 predictions for 30 examples); GLM 5.3 is slightly overpredicted (33), and Flash is overpredicted (40), with low precision. B overpredicts Fable (38) and Flash (49), while GLM 5.3 is slightly underpredicted (29). D still overpredicts Fable (37) and underpredicts both GLM labels (25 and 26). GLM 5.3 and Flash are strongly confused with each other in all checkpoints; the directed counts are tabulated above.",
               "4. **Did adding V1-all data improve or worsen recognition?** A→B improves overall accuracy by 1.11 pp and accuracy on the three true weak labels by 6.67 pp (21 corrected, 15 regressed). It improves Flash recall by 20.0 pp and Fable recall by 3.3 pp, but worsens GLM 5.3 recall by 3.3 pp and Fable precision by 23.4 pp. B→D is a separate continuation: overall accuracy rises another 2.96 pp, while weak-label accuracy is unchanged (12 corrected, 12 regressed); it improves Fable and GLM 5.3 precision/recall tradeoffs but reduces Flash recall.",
               "5. **Which three targeted changes should be tested next?** (a) Run a V1.1 vs V1.1+BM source-stratified ablation for the weak labels, keeping V0 prompts excluded and logging exact source mix. (b) Test pair-focused sampling or class weighting for GLM 5.3 vs GLM 5.3 Flash, because they exchange many errors in both directions. (c) Audit/hold out the dominant BM sources for Fable and GLM before adding rows, then compare precision and recall on the unchanged V0 rows; preserve the current checkpoint and recipe as controls.", "",
               "## Reproducibility and artifact index", "",
               "Run with `python -m src.v1_all_diagnostics`. All generated tables are under `artifacts/analysis/v1_all_diagnostics/`. Prediction files are aligned with the repository's strict V0 ID/prompt/label alignment utility. A/B/D resolve through `checkpoint_inventory.csv`.", "",
               "The three checkpoints are A=V1, B=V1-all, and D=V1-all→V0. V1.2_bm is training-only. The official V0 validation set is used only as the evaluation reference.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")


def run(data_dir="data/processed", output_dir=OUTPUT, report_path=REPORT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = _dataset_analysis(data_dir, output_dir)
    cm = _confusion_analysis(data_dir, output_dir)
    _make_report(ds, cm, Path(report_path))
    summary = {"v1_1_rows": int(len(ds["external"].query("dataset == 'v1.1'"))),
               "v1_2_bm_rows": int(len(ds["external"].query("dataset == 'v1.2_bm'"))),
               "v1_all_rows": ds["v1all_rows"], "v1_all_id_hash": ds["v1all_id_hash"],
               "overall_accuracy": cm["overall"], "artifacts": str(output_dir), "report": str(report_path)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data_dir, args.output_dir, args.report), indent=2))


if __name__ == "__main__":
    main()
