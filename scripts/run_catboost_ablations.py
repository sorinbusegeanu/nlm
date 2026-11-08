"""Run the CatBoost stylometry ablation sequence from docs/CatBoost_ablations.md."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import yaml

from src.data_factory import make_splits
from src.evaluation import evaluate, paired_group_bootstrap
from src.pipeline import _effective_profile, _load_dataset, train_models
from src.registry import Registry


ROOT = Path(__file__).resolve().parents[1]
DATASET = "v1"
OUTPUT = ROOT / "analysis-output" / "catboost_ablations_v1"
BASELINE = ["length", "case", "punctuation", "function_words"]
SINGLE_GROUPS = ["L", "V", "C", "P", "W", "F", "R", "X", "M", "U"]
OMIT_BASE = ["length", "case", "punctuation", "function_words"]
EXTRA_SEEDS = [2027, 2028, 2029]


def _recipe(base=None, groups=None):
    return {"baseline": list(BASELINE if base is None else base), "groups": list(groups or [])}


def _run(tag, recipe, *, dataset=DATASET, data_dir=None, seed=2026, description=None):
    config_dir = OUTPUT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/models/catboost_stylometry.yaml").read_text())
    cfg["feature_recipe"] = recipe
    config_path = config_dir / f"{tag}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    existing_ids = {row["experiment_id"] for row in Registry(ROOT / "experiments/registry.sqlite").rows()}
    manifest = train_models(dataset=dataset, model="catboost_stylometry",
        data_dir=str(data_dir or ROOT / "data/processed"),
        artifact_dir=str(ROOT / "artifacts"),
        registry_path=str(ROOT / "experiments/registry.sqlite"),
        config_path=str(config_path), seed=seed)
    if manifest["experiment_id"] not in existing_ids:
        Registry(ROOT / "experiments/registry.sqlite").add(
            manifest["experiment_id"], "catboost_stylometry", "validation",
            manifest["config"], manifest["metrics"],
            dataset_hash=manifest["dataset_sha256"], code_revision=manifest["code_revision"],
            seed=manifest["seed"], runtime_seconds=manifest["fit_seconds"],
            artifact_path=manifest["prediction_path"], config_hash=manifest["config_sha256"],
            split_hash=manifest["split_sha256"], model_revision=manifest.get("model_revision"),
            description=description or f"CatBoost stylometry ablation {tag}: {recipe}")
    return manifest


def _summary_row(tag, manifest, recipe):
    metrics = manifest["metrics"]
    return {"run": tag, "dataset": manifest["dataset"],
        "balanced_accuracy": metrics["balanced_accuracy"], "accuracy": _ordinary_accuracy(manifest),
        "macro_f1": metrics["macro_f1"], "log_loss": metrics.get("log_loss"),
        "feature_count": manifest.get("feature_count"),
        "fit_seconds": manifest.get("fit_seconds"),
        "inference_seconds": manifest.get("inference_seconds"),
        "selected_iteration": manifest.get("selected_iteration"),
        "recipe": json.dumps(recipe, sort_keys=True),
        "experiment_id": manifest["experiment_id"],
        "prediction_path": manifest["prediction_path"],
        "manifest_path": str(ROOT / "artifacts/predictions" / (manifest["experiment_id"] + "_manifest.json"))}


def _ordinary_accuracy(manifest):
    pred = pd.read_parquet(manifest["prediction_path"])
    prob_cols = [c for c in pred if c.startswith("prob::")]
    labels = [c.removeprefix("prob::") for c in prob_cols]
    guesses = np.asarray(labels)[pred[prob_cols].to_numpy().argmax(axis=1)]
    return float(np.mean(guesses == pred.y_true.astype(str).to_numpy()))


def _read_predictions(manifest):
    frame = pd.read_parquet(manifest["prediction_path"])
    columns = [c for c in frame if c.startswith("prob::")]
    labels = [c.removeprefix("prob::") for c in columns]
    return frame[["id", "prompt_id", "y_true"]].copy(), frame[columns].to_numpy(dtype=float), labels


def _save_a6(manifests, compact_tag):
    from src.registry import Registry
    reg = Registry(ROOT / "experiments/registry.sqlite")
    rows = reg.rows()
    base_models = {"svc", "ultrabert380", "qwen06", "qwen06_best_epoch03", "embeddinggemma2"}
    best = {}
    for row in rows:
        if row["split"] != "validation" or row["model"] not in base_models:
            continue
        if (row["dataset_hash"] != manifests["A0"]["dataset_sha256"]
                or row["split_hash"] != manifests["A0"]["split_sha256"]):
            continue
        score = json.loads(row["metrics_json"]).get("balanced_accuracy")
        if score is None:
            continue
        canonical = "qwen06" if row["model"].startswith("qwen06") else row["model"]
        if canonical not in best or score > best[canonical][0]:
            prediction_path = Path(row["artifact_path"] or "")
            if prediction_path.exists():
                best[canonical] = (score, prediction_path)
    reference, _, classes = _read_predictions(manifests["A0"])
    other_probs = []
    other_names = []
    for name, (_, path) in best.items():
        frame = pd.read_parquet(path)
        if not set(reference.id) == set(frame.id):
            continue
        frame = frame.set_index("id").loc[reference.id].reset_index()
        cols = [f"prob::{label}" for label in classes]
        if not set(cols).issubset(frame.columns):
            continue
        other_probs.append(frame[cols].to_numpy(dtype=float))
        other_names.append(name)
    if not other_probs:
        return {"status": "skipped", "reason": f"no saved {DATASET} model predictions aligned to CatBoost validation IDs"}
    truth = reference.y_true.astype(str).to_numpy()
    true_indexes = np.asarray([classes.index(label) for label in truth], dtype=int)
    groups = reference.prompt_id.astype(str).to_numpy()
    probability_by_name = dict(zip(other_names, other_probs))
    model_sets = {f"only_{name}": [name] for name in other_names}
    model_sets["all_existing_models"] = other_names
    output = {"existing_models": other_names, "validation_rows": len(reference),
              "prompt_groups": int(pd.Series(groups).nunique()), "results": {}}
    for tag in ("A0", compact_tag):
        _, cat_probs, _ = _read_predictions(manifests[tag])
        output["results"][tag] = {}
        for set_name, members in model_sets.items():
            selected_probs = [probability_by_name[name] for name in members]
            baseline_probs = np.mean(selected_probs, axis=0)
            with_cat_probs = np.mean(selected_probs + [cat_probs], axis=0)
            baseline = evaluate(truth, baseline_probs, classes, baseline_probs)
            with_cat = evaluate(truth, with_cat_probs, classes, with_cat_probs)
            output["results"][tag][set_name] = {
                "members": members,
                "without_catboost": {**{k: baseline[k] for k in ("balanced_accuracy", "macro_f1", "log_loss")},
                                     "accuracy": float(np.mean(baseline_probs.argmax(1) == true_indexes))},
                "with_catboost": {**{k: with_cat[k] for k in ("balanced_accuracy", "macro_f1", "log_loss")},
                                  "accuracy": float(np.mean(with_cat_probs.argmax(1) == true_indexes))},
                "balanced_accuracy_delta": with_cat["balanced_accuracy"] - baseline["balanced_accuracy"],
                "correctness_changes": {
                    "fixed_by_catboost": int(np.sum((with_cat_probs.argmax(1) == true_indexes) & (baseline_probs.argmax(1) != true_indexes))),
                    "broken_by_catboost": int(np.sum((with_cat_probs.argmax(1) != true_indexes) & (baseline_probs.argmax(1) == true_indexes)))
                },
                "paired_prompt_group_bootstrap": paired_group_bootstrap(
                    truth, np.asarray(classes)[baseline_probs.argmax(1)],
                    np.asarray(classes)[with_cat_probs.argmax(1)], groups,
                    n_bootstrap=2000, seed=2026),
            }
    return output


def _build_split_data(seed):
    source_dir = ROOT / "data/processed"
    combined, _ = _load_dataset(DATASET, source_dir)
    # Keep the official V0 validation rows out of these secondary splits. A5
    # measures recipe stability by splitting only the rows V1 uses for fitting.
    source = combined.loc[combined.split.eq("train")].drop(columns="split").reset_index(drop=True)
    assignments = make_splits(source, seed=seed, validation_fraction=.2, candidates=1000)
    version = f"catboost_v1_a5_s{seed}"
    split_dir = OUTPUT / "split_data" / version
    split_dir.mkdir(parents=True, exist_ok=True)
    source.to_parquet(split_dir / f"dataset_{version}.parquet", index=False)
    assignments.to_parquet(split_dir / f"splits_{version}.parquet", index=False)
    split_payload = assignments[["id", "split"]].to_csv(index=False, lineterminator="\n").encode()
    base_profile = _effective_profile(source_dir, DATASET, combined)
    profile = {"dataset_version": version, "dataset_sha256": base_profile["dataset_sha256"],
        "split_sha256": __import__("hashlib").sha256(split_payload).hexdigest(),
        "validation_rows": int(assignments.split.eq("validation").sum()), "seed": seed}
    (split_dir / f"dataset_profile_{version}.json").write_text(json.dumps(profile, indent=2) + "\n")
    return version, split_dir


def _load_manifest(tag):
    path = OUTPUT / "manifests" / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _verify_a0_reproduction(manifest):
    """Require the frozen V0 baseline metrics and predictions to reproduce before A1."""
    baseline_config = yaml.safe_load((ROOT / "configs/models/catboost_stylometry.yaml").read_text())
    references = []
    for row in Registry(ROOT / "experiments/registry.sqlite").rows():
        if (row["model"] != "catboost_stylometry" or row["split"] != "validation"
                or row["dataset_hash"] != manifest["dataset_sha256"]
                or row["split_hash"] != manifest["split_sha256"]):
            continue
        config = json.loads(row["config_json"])
        if config.get("feature_recipe") is not None:
            continue
        keys = ("iterations", "depth", "learning_rate", "l2_leaf_reg", "thread_count",
                "temperature_calibration", "early_stopping_rounds")
        if any(config.get(k, baseline_config.get(k)) != baseline_config.get(k) for k in keys):
            continue
        path = Path(row["artifact_path"] or "")
        if path.exists():
            references.append((row, json.loads(row["metrics_json"]), path))
    if not references:
        return {"status": "skipped", "reason":
            "No CatBoost baseline is registered for this exact dataset and split; historical V1 runs used the previous mixed V0+Open Pool loader."}
    row, old_metrics, old_path = max(references, key=lambda item: item[0]["sequence_number"])
    old = pd.read_parquet(old_path).sort_values("id", kind="stable").reset_index(drop=True)
    current = pd.read_parquet(manifest["prediction_path"]).sort_values("id", kind="stable").reset_index(drop=True)
    if not old.id.astype(str).equals(current.id.astype(str)):
        raise RuntimeError("A0 validation IDs differ from the registered baseline")
    if not old.y_true.astype(str).equals(current.y_true.astype(str)):
        raise RuntimeError("A0 validation labels differ from the registered baseline")
    classes = manifest["classes"]
    old_probs = old[[f"prob::{label}" for label in classes]].to_numpy(dtype=float)
    new_probs = current[[f"prob::{label}" for label in classes]].to_numpy(dtype=float)
    same_predictions = bool(np.array_equal(old_probs.argmax(1), new_probs.argmax(1)))
    max_probability_delta = float(np.max(np.abs(old_probs - new_probs)))
    accuracy_delta = float(manifest["metrics"]["balanced_accuracy"] - old_metrics["balanced_accuracy"])
    if not same_predictions or abs(accuracy_delta) > 1e-12 or max_probability_delta > 1e-5:
        raise RuntimeError(f"A0 failed baseline reproduction: balanced-accuracy delta={accuracy_delta:.8f}, max probability delta={max_probability_delta:.8g}, same predictions={same_predictions}; stopping before A1")
    return {"reference_experiment": row["experiment_id"],
        "reference_balanced_accuracy": old_metrics["balanced_accuracy"],
        "a0_balanced_accuracy": manifest["metrics"]["balanced_accuracy"],
        "same_predictions": same_predictions, "max_probability_delta": max_probability_delta}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_recipe = _recipe()
    manifests = {}
    rows = []
    a0 = _run("A0", base_recipe, description=f"CatBoost {DATASET} ablation A0: reproduce 35-feature stylometry baseline")
    a0_reproduction = _verify_a0_reproduction(a0)
    (OUTPUT / "a0_reproduction.json").write_text(json.dumps(a0_reproduction, indent=2) + "\n")
    manifests["A0"] = a0
    rows.append(_summary_row("A0", a0, base_recipe))
    baseline_score = a0["metrics"]["balanced_accuracy"]

    singles = {}
    for group in SINGLE_GROUPS:
        recipe = _recipe(groups=[group])
        tag = f"A1_{group}"
        result = _run(tag, recipe, description=f"CatBoost {DATASET.upper()} ablation A1: baseline plus feature group {group}")
        manifests[tag] = result
        singles[group] = result["metrics"]["balanced_accuracy"]
        rows.append(_summary_row(tag, result, recipe))

    removals = {}
    for group in OMIT_BASE:
        recipe = _recipe(base=[item for item in BASELINE if item != group])
        tag = f"A2_minus_{group}"
        result = _run(tag, recipe, description=f"CatBoost {DATASET.upper()} ablation A2: baseline minus {group}")
        manifests[tag] = result
        removals[group] = result["metrics"]["balanced_accuracy"]
        rows.append(_summary_row(tag, result, recipe))

    promising = [group for group in SINGLE_GROUPS if singles[group] > baseline_score]
    if len(promising) < 2:
        promising = sorted(SINGLE_GROUPS, key=lambda g: (-singles[g], SINGLE_GROUPS.index(g)))[:2]
    combined_recipe = _recipe(groups=promising)
    combo_tag = "A3_combined"
    combined = _run(combo_tag, combined_recipe,
        description=f"CatBoost {DATASET.upper()} ablation A3: promising groups {','.join(promising)}")
    manifests[combo_tag] = combined
    rows.append(_summary_row(combo_tag, combined, combined_recipe))
    for group in promising:
        recipe = _recipe(groups=[g for g in promising if g != group])
        tag = f"A3_minus_{group}"
        result = _run(tag, recipe, description=f"CatBoost {DATASET.upper()} ablation A3: combined promising groups minus {group}")
        manifests[tag] = result
        rows.append(_summary_row(tag, result, recipe))

    candidate_tags = ["A0", *(f"A1_{g}" for g in SINGLE_GROUPS),
        *(f"A2_minus_{g}" for g in OMIT_BASE), combo_tag,
        *(f"A3_minus_{g}" for g in promising)]
    peak_score = max(manifests[tag]["metrics"]["balanced_accuracy"] for tag in candidate_tags)
    compact_pool = [tag for tag in candidate_tags
                    if manifests[tag]["metrics"]["balanced_accuracy"] >= peak_score - 0.01]
    compact_tag = min(compact_pool, key=lambda tag: (
        manifests[tag].get("feature_count", 10**9),
        -manifests[tag]["metrics"]["balanced_accuracy"], tag))
    compact_recipe = manifests[compact_tag]["config"].get("feature_recipe", base_recipe)
    a4 = manifests[compact_tag] if compact_tag == "A0" else _run("A4_compact", compact_recipe,
        description=f"CatBoost {DATASET.upper()} ablation A4: compact recipe selected from {compact_tag}")
    manifests["A4"] = a4
    rows.append(_summary_row("A4_compact", a4, compact_recipe))

    for seed in EXTRA_SEEDS:
        version, split_dir = _build_split_data(seed)
        for tag, recipe in (("A0", base_recipe), ("A4", compact_recipe)):
            run_tag = f"A5_{tag}_s{seed}"
            result = _run(run_tag, recipe, dataset=version, data_dir=split_dir, seed=seed,
                description=f"CatBoost {DATASET.upper()} ablation A5: {tag} recipe, grouped split seed {seed}")
            manifests[run_tag] = result
            rows.append(_summary_row(run_tag, result, recipe))

    a6 = _save_a6(manifests, "A4")
    a0_ref, a0_probs, classes = _read_predictions(manifests["A0"])
    a4_ref, a4_probs, _ = _read_predictions(a4)
    a0_predictions = np.asarray(classes)[a0_probs.argmax(1)]
    a4_predictions = np.asarray(classes)[a4_probs.argmax(1)]
    truth = a0_ref.y_true.astype(str).to_numpy()
    paired = {"status": "ok", "baseline": "A0", "compact": "A4",
        "prompt_groups": int(a0_ref.prompt_id.astype(str).nunique()),
        "correctness_changes": {
            "fixed_by_A4": int(np.sum((a4_predictions == truth) & (a0_predictions != truth))),
            "broken_by_A4": int(np.sum((a4_predictions != truth) & (a0_predictions == truth)))},
        "group_bootstrap": paired_group_bootstrap(
            truth, a0_predictions, a4_predictions, a0_ref.prompt_id.astype(str),
            n_bootstrap=2000, seed=2026)}
    paired["per_class_recall_delta"] = {
        label: a4["metrics"]["per_class_recall"].get(label, 0.0) -
               manifests["A0"]["metrics"]["per_class_recall"].get(label, 0.0)
        for label in classes}
    (OUTPUT / "a0_vs_a4_paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    (OUTPUT / "a6_blends.json").write_text(json.dumps(a6, indent=2) + "\n")
    for tag, manifest in manifests.items():
        (OUTPUT / "manifests").mkdir(parents=True, exist_ok=True)
        (OUTPUT / "manifests" / f"{tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "ablation_summary.csv", index=False)
    summary.to_parquet(OUTPUT / "ablation_summary.parquet", index=False)
    report = ["# CatBoost stylometry ablation results", "",
        f"All A0-A4 runs use the frozen {DATASET.upper()} split and fixed CatBoost settings. For V1, training uses only curated Open Pool rows and validation uses the frozen 540 V0 validation rows. A5 uses three additional grouped 80/20 splits over only the Open Pool rows; all V0 rows remain excluded. Validation-guided recipe selection is exploratory; use repeated-split results and group bootstrap intervals to assess stability.", "",
        f"Baseline A0 balanced accuracy: {baseline_score:.4f}. A4 chose the smallest feature recipe within 0.01 balanced-accuracy points of the best candidate (source: {compact_tag}); it uses {a4.get('feature_count')} features.", "",
        "| Run | Balanced accuracy | Accuracy | Macro F1 | Log loss | Features | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['run']} | {row['balanced_accuracy']:.4f} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['log_loss']:.4f} | {row['feature_count']} | {row['fit_seconds']:.1f} |")
    report += ["", "## A1 single-group changes", "", "| Group | Delta balanced accuracy vs A0 |", "|---|---:|"]
    for group in SINGLE_GROUPS:
        report.append(f"| {group} | {singles[group]-baseline_score:+.4f} |")
    report += ["", "## A2 baseline-subgroup removals", "", "| Removed subgroup | Delta balanced accuracy vs A0 |", "|---|---:|"]
    for group, score in removals.items():
        report.append(f"| {group} | {score-baseline_score:+.4f} |")
    a5_rows = summary[summary.run.str.startswith("A5_")].copy()
    report += ["", "## A5 repeated grouped splits", ""]
    if not a5_rows.empty:
        for recipe_tag in ("A0", "A4"):
            values = a5_rows.loc[a5_rows.run.str.startswith(f"A5_{recipe_tag}_"), "balanced_accuracy"]
            report.append(f"- {recipe_tag}: mean {values.mean():.4f}, standard deviation {values.std(ddof=1):.4f} over {len(values)} additional grouped splits.")
    report += ["", "## A0 baseline reproduction", "", "```json", json.dumps(a0_reproduction, indent=2), "```", "",
        "## A0 versus A4 uncertainty and per-class changes", "", "```json", json.dumps(paired, indent=2), "```", ""]
    report += ["## A0 and A4 confusion matrices", "", "```json", json.dumps({
        tag: {"classes": manifests[tag]["metrics"]["classes"],
              "confusion_matrix": manifests[tag]["metrics"]["confusion_matrix"]}
        for tag in ("A0", "A4")}, indent=2), "```", ""]
    report += ["", "## A6 equal-weight blends", "", "```json", json.dumps(a6, indent=2), "```", "",
        "Raw per-run manifests and ID-aligned probability Parquet files are retained in the experiment artifacts. Full summary: `ablation_summary.csv`."]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(OUTPUT), "baseline": baseline_score,
        "compact_source": compact_tag, "compact_accuracy": a4["metrics"]["balanced_accuracy"],
        "runs": len(rows), "a6": a6}, indent=2))


if __name__ == "__main__":
    main()
