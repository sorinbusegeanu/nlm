"""Prediction-guided reduction of the V2.14 training dataset.

This module only runs inference and deterministic row selection. It never fits a
model. Prediction arrays are cached per checkpoint under artifacts/analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
from .data_factory import _normal_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_v2.14.parquet"
PROFILE_PATH = ROOT / "data/processed/dataset_profile_v2.14.json"
OUT_DIR = ROOT / "artifacts/analysis/v2.14_prediction_filter"
OUT_DATA = ROOT / "data/processed/dataset_v2.14-prediction-filtered.parquet"
OUT_PROFILE = ROOT / "data/processed/dataset_profile_v2.14-prediction-filtered.json"
OUT_REPORT = ROOT / "docs/v2.14_prediction_filter_report.md"
OUT_DECISIONS = OUT_DIR / "row_decisions.csv"

CHECKPOINTS = {
    "ultrabert": {
        "experiment_id": "ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgb9e735ff_code81bce4a",
        "manifest": "artifacts/predictions/ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgb9e735ff_code81bce4a_manifest.json",
    },
    "svc": {
        "experiment_id": "svc_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfg66dc1893_code81bce4a",
        "manifest": "artifacts/predictions/svc_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfg66dc1893_code81bce4a_manifest.json",
    },
    "catboost": {
        "experiment_id": "catboost_stylometry_v1.5_data490fe052_split215fd0d0_seed2026_cfgdb877fc5_code1b9bfc2",
        "manifest": "artifacts/predictions/catboost_stylometry_v1.5_data490fe052_split215fd0d0_seed2026_cfgdb877fc5_code1b9bfc2_manifest.json",
    },
}
CATEGORY_NAMES = [
    "Very easy", "Easy but disputed", "Moderately difficult", "Difficult",
    "Strong consensus error", "Other errors",
]
RETENTION = np.array([0.20, 0.50, 0.60, 1.00, 0.00, 1.00])
RETENTION_80PCT = np.array([0.60, 1.00, 1.00, 1.00, 1.00, 1.00])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifests():
    result = {}
    for name, spec in CHECKPOINTS.items():
        path = ROOT / spec["manifest"]
        manifest = json.loads(path.read_text())
        if manifest.get("experiment_id") != spec["experiment_id"]:
            raise ValueError(f"manifest identity mismatch for {name}: {path}")
        checkpoint = Path(manifest["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result[name] = (manifest, checkpoint, path)
    return result


def _load_cache(name: str, ids: np.ndarray, classes: list[str], manifest: dict,
                dataset_sha256: str, cache_dir: Path = OUT_DIR) -> np.ndarray | None:
    path = cache_dir / f"{name}_probabilities.npz"
    meta_path = cache_dir / f"{name}_probabilities.json"
    if not path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    if (meta.get("experiment_id") != manifest["experiment_id"] or meta.get("classes") != classes
            or meta.get("dataset_sha256") != dataset_sha256):
        return None
    with np.load(path, allow_pickle=False) as cache:
        cached_ids, probs = cache["ids"], cache["probabilities"]
    if not np.array_equal(cached_ids.astype(str), ids.astype(str)):
        return None
    if probs.shape != (len(ids), len(classes)) or not np.isfinite(probs).all():
        return None
    return probs


def _save_cache(name: str, ids: np.ndarray, probs: np.ndarray, classes: list[str], manifest: dict,
                manifest_path: Path, dataset_sha256: str, cache_dir: Path = OUT_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_probabilities.npz"
    np.savez_compressed(path, ids=ids.astype(str), probabilities=probs.astype(np.float32))
    meta = {
        "experiment_id": manifest["experiment_id"], "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_manifest": str(manifest_path.relative_to(ROOT)),
        "checkpoint_manifest_sha256": _sha256(manifest_path), "classes": classes,
        "rows": len(ids), "dataset_sha256": dataset_sha256,
        "probability_path": str(path.relative_to(ROOT)),
    }
    (cache_dir / f"{name}_probabilities.json").write_text(json.dumps(meta, indent=2) + "\n")


def get_predictions(frame: pd.DataFrame, classes: list[str], dataset_sha256: str,
                    cache_dir: Path = OUT_DIR) -> tuple[dict[str, np.ndarray], dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifests = _load_manifests()
    texts, ids = frame.text.astype(str).tolist(), frame.id.astype(str).to_numpy()
    probabilities = {}
    for name in ("svc", "catboost", "ultrabert"):
        manifest, checkpoint, manifest_path = manifests[name]
        probs = _load_cache(name, ids, classes, manifest, dataset_sha256, cache_dir)
        if probs is None:
            print(f"Running {name} inference for {len(ids):,} V2.14 rows", flush=True)
            model_classes = list(manifest["classes"])
            if set(model_classes) != set(classes):
                raise ValueError(f"{name} class roster differs from V2.14 labels")
            if name == "svc":
                import joblib
                bundle = joblib.load(checkpoint)
                probs, returned = predict_svc_bundle(bundle, texts)
            elif name == "catboost":
                probs, returned = predict_catboost_stylometry(
                    checkpoint, texts, model_classes,
                    temperature=float(manifest.get("calibration_temperature") or 1.0))
            else:
                config = manifest.get("config", {})
                probs, returned = _predict_neural(
                    "ultrabert380", checkpoint, texts, model_classes,
                    batch_size=int(config.get("inference_batch_size") or 8),
                    max_length=int(config.get("max_length") or 581),
                    temperature=float(manifest.get("calibration_temperature") or 1.0),
                    return_scores=False)
            if list(returned) != model_classes:
                raise ValueError(f"{name} returned an unexpected class order")
            # Normalize all model outputs into V2.14's canonical class ordering.
            order = [model_classes.index(label) for label in classes]
            probs = np.asarray(probs, dtype=np.float64)[:, order]
            if probs.shape != (len(ids), len(classes)) or not np.isfinite(probs).all():
                raise ValueError(f"invalid {name} probability matrix: {probs.shape}")
            if np.any(probs < -1e-8) or not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
                raise ValueError(f"{name} probability rows are invalid")
            _save_cache(name, ids, probs, classes, manifest, manifest_path, dataset_sha256, cache_dir)
        probabilities[name] = probs
    return probabilities, {name: value[0] for name, value in manifests.items()}


def classify_rows(frame: pd.DataFrame, probabilities: dict[str, np.ndarray], classes: list[str]):
    class_index = {label: i for i, label in enumerate(classes)}
    y = frame.model.astype(str).to_numpy()
    if not set(y) <= set(classes):
        raise ValueError("dataset contains labels outside the model roster")
    true_i = np.fromiter((class_index[v] for v in y), dtype=np.int32, count=len(y))
    pred_i, correct = {}, {}
    ranked = {}
    for name in ("ultrabert", "svc", "catboost"):
        p = probabilities[name]
        # Stable sort preserves canonical label order for probability ties.
        order = np.argsort(-p, axis=1, kind="stable")
        ranked[name] = order
        pred_i[name] = order[:, 0]
        correct[name] = pred_i[name] == true_i
    ub = probabilities["ultrabert"]
    confidence = ub.max(axis=1)
    true_probability = ub[np.arange(len(frame)), true_i]
    true_rank = (ranked["ultrabert"] == true_i[:, None]).argmax(axis=1) + 1
    medians = {}
    for label in classes:
        mask = (y == label) & correct["ultrabert"]
        if not mask.any():
            raise ValueError(f"UltraBERT has no correctly classified rows for median: {label}")
        medians[label] = float(np.median(confidence[mask]))
    high = confidence >= np.fromiter((medians[v] for v in y), dtype=float, count=len(y))
    category = np.full(len(frame), -1, dtype=np.int8)
    remaining = np.ones(len(frame), dtype=bool)

    def assign(mask, cat):
        nonlocal remaining
        take = remaining & mask
        category[take] = cat
        remaining[take] = False

    ub_correct = correct["ultrabert"]
    other_correct = correct["svc"] | correct["catboost"]
    assign(ub_correct & high & other_correct, 0)
    assign(ub_correct & high & ~correct["svc"] & ~correct["catboost"], 1)
    assign(ub_correct & ~high, 2)
    top3 = np.any(ranked["ultrabert"][:, :3] == true_i[:, None], axis=1)
    ub_wrong = ~ub_correct
    assign(ub_wrong & (other_correct | top3), 3)
    consensus = (pred_i["ultrabert"] == pred_i["svc"]) & (pred_i["ultrabert"] == pred_i["catboost"])
    assign(ub_wrong & consensus & (confidence >= 0.90) & ~top3, 4)
    assign(ub_wrong, 5)
    if remaining.any():
        raise AssertionError(f"unclassified rows: {remaining.sum()}")
    return {
        "true_i": true_i, "pred_i": pred_i, "correct": correct, "ranked": ranked,
        "confidence": confidence, "true_probability": true_probability,
        "true_rank": true_rank, "medians": medians, "category": category,
    }


def _stratum_columns(frame: pd.DataFrame):
    words = frame.text.astype(str).map(lambda x: len(re.findall(r"\b\w+\b", x)))
    length = pd.cut(words, bins=[-1, 99, 300, np.inf], labels=["short", "medium", "long"])
    source_col = "source_dataset" if "source_dataset" in frame else None
    source = frame[source_col].fillna("<missing>").astype(str) if source_col else pd.Series("<missing>", index=frame.index)
    return source.to_numpy(), length.astype(str).to_numpy(), words.to_numpy()


def select_rows(frame: pd.DataFrame, categories: np.ndarray, seed: int = 2026,
                retention: np.ndarray | list[float] | tuple[float, ...] | None = None) -> np.ndarray:
    """Apply per-stratum nearest-integer rates with largest-remainder allocation."""
    rates = np.asarray(RETENTION if retention is None else retention, dtype=float)
    if rates.shape != (len(CATEGORY_NAMES),) or not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
        raise ValueError(f"retention must contain {len(CATEGORY_NAMES)} rates in [0, 1]")
    source, lengths, _ = _stratum_columns(frame)
    labels = frame.model.astype(str).to_numpy()
    ids = frame.id.astype(str).to_numpy()
    keep = np.zeros(len(frame), dtype=bool)
    rng = np.random.default_rng(seed)
    for cat, rate in enumerate(rates):
        indexes = np.flatnonzero(categories == cat)
        if rate == 1:
            keep[indexes] = True
            continue
        if rate == 0 or not len(indexes):
            continue
        strata: dict[tuple[str, str, str], list[int]] = {}
        for i in indexes:
            strata.setdefault((labels[i], source[i], lengths[i]), []).append(int(i))
        ordered_keys = sorted(strata)
        sizes = np.array([len(strata[k]) for k in ordered_keys], dtype=int)
        desired = len(indexes) * rate
        target = int(np.floor(desired + 0.5))
        raw = sizes * rate
        quotas = np.floor(raw).astype(int)
        remainder = target - int(quotas.sum())
        fractions = raw - quotas
        ties = rng.random(len(ordered_keys))
        ranking = sorted(range(len(ordered_keys)), key=lambda j: (-fractions[j], ties[j], ordered_keys[j]))
        for j in ranking[:remainder]:
            if quotas[j] < sizes[j]:
                quotas[j] += 1
        # If capped strata prevented allocation, pass remaining slots to the
        # next eligible strata in the same deterministic fractional ranking.
        left = target - int(quotas.sum())
        for j in ranking:
            if left <= 0:
                break
            if quotas[j] < sizes[j]:
                quotas[j] += 1
                left -= 1
        for key, quota in zip(ordered_keys, quotas):
            candidates = np.asarray(sorted(strata[key], key=lambda i: ids[i]), dtype=int)
            if quota:
                selected = rng.choice(candidates, size=int(quota), replace=False)
                keep[selected] = True
    return keep


def _decision_table(frame: pd.DataFrame, classes: list[str], probs: dict[str, np.ndarray], result: dict,
                    retained: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(classes)
    return pd.DataFrame({
        "response_id": frame.id.astype(str), "true_label": frame.model.astype(str),
        "ultrabert_prediction": labels[result["pred_i"]["ultrabert"]],
        "svc_prediction": labels[result["pred_i"]["svc"]],
        "catboost_prediction": labels[result["pred_i"]["catboost"]],
        "ultrabert_top1_confidence": result["confidence"],
        "ultrabert_true_label_probability": result["true_probability"],
        "ultrabert_true_label_rank": result["true_rank"],
        "category": [CATEGORY_NAMES[i] for i in result["category"]],
        "retained": retained,
    })


def _metrics_report(frame, classes, probs, result, keep, cache_meta, output_sha, *,
                    retention: np.ndarray | list[float] | tuple[float, ...] | None = None,
                    dataset_version: str = "v2.14-prediction-filtered", output_data: Path = OUT_DATA,
                    output_profile: Path = OUT_PROFILE, output_decisions: Path = OUT_DECISIONS,
                    output_dir: Path = OUT_DIR, seed: int = 2026):
    rates = np.asarray(RETENTION if retention is None else retention, dtype=float)
    category = result["category"]
    decisions = {}
    for i, name in enumerate(CATEGORY_NAMES):
        original = int((category == i).sum())
        kept = int(((category == i) & keep).sum())
        decisions[name] = (original, kept, original - kept, rates[i])
    lines = [
        f"# {dataset_version} prediction-guided dataset reduction", "",
        "This is a deterministic, inference-only reduction. Models were not retrained.",
        f"Input: `{DATA_PATH.relative_to(ROOT)}` ({len(frame):,} rows).",
        f"Output: `{output_data.relative_to(ROOT) if output_data.is_absolute() else output_data}` ({int(keep.sum()):,} rows).",
        f"Output SHA-256: `{output_sha}`. Selection seed: `{seed}`.", "",
        "## Checkpoints", "",
        "| Model | Experiment | Training dataset | Checkpoint | V0 balanced accuracy |",
        "|---|---|---|---|---:|",
    ]
    for name in ("ultrabert", "svc", "catboost"):
        m = cache_meta[name]
        lines.append(f"| {name} | `{m['experiment_id']}` | {m.get('dataset', 'unknown')} | `{m['checkpoint_path']}` | {m.get('v0_balanced_accuracy', float('nan')):.4f} |")
    lines += ["", "The preferred full-V2.14 UltraBERT checkpoint is Trust List 2 sequence `N0071`; no local checkpoint record labeled `N0122` was found. N0071 is the strongest saved UltraBERT checkpoint trained on full V2.14 by V0 balanced accuracy.",
        "The UltraBERT and SVC accuracy values below are **in-sample**: both checkpoints were trained on the V2.14 rows being scored. They describe the requested prediction-guided selection behavior, not unbiased generalization accuracy. CatBoost was trained on V1.5.", ""]
    lines += ["", "## A. Training-set prediction accuracy", "", "| Model | Correct | Incorrect | Accuracy |", "|---|---:|---:|---:|"]
    for name in ("ultrabert", "svc", "catboost"):
        n_correct = int(result["correct"][name].sum())
        lines.append(f"| {name} | {n_correct:,} | {len(frame)-n_correct:,} | {n_correct/len(frame):.4%} |")
    lines += ["", "## B. Dataset reduction", "", "| Category | Original rows | Retained | Removed | Retention |", "|---|---:|---:|---:|---:|"]
    for name, (original, kept, removed, rate) in decisions.items():
        lines.append(f"| {name} | {original:,} | {kept:,} | {removed:,} | {rate:.0%} |")
    removed = len(frame) - int(keep.sum())
    lines.append(f"| **TOTAL** | **{len(frame):,}** | **{int(keep.sum()):,}** | **{removed:,}** | **{(1-removed/len(frame)):.2%} retained** |")
    lines += ["", "## C. Dataset composition", "", f"Final size is **{int(keep.sum()):,}** rows; reduction is **{removed/len(frame):.2%}**.", "",
        "### Counts by label", "", "| Label | Original | Retained | Removed | Removed % |", "|---|---:|---:|---:|---:|"]
    y = frame.model.astype(str)
    for label in classes:
        original = int((y == label).sum()); kept = int(((y == label) & keep).sum())
        lines.append(f"| {label} | {original:,} | {kept:,} | {original-kept:,} | {(original-kept)/original:.2%} |")
    lines += ["", "### Counts by source dataset", "", "| Source | Original | Retained | Removed |", "|---|---:|---:|---:|"]
    source = frame.source_dataset.fillna("<missing>").astype(str) if "source_dataset" in frame else pd.Series("<missing>", index=frame.index)
    for src, count in source.value_counts().sort_index().items():
        kept = int((keep & source.eq(src).to_numpy()).sum())
        lines.append(f"| {src} | {int(count):,} | {kept:,} | {int(count)-kept:,} |")
    lines += ["", "### UltraBERT confidence and retained/removed accuracy", "",
        "| Subset | Rows | Correct | Accuracy | Mean confidence | Median confidence |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, mask in [("Retained", keep), ("Removed", ~keep)]:
        n = int(mask.sum()); c = int((result["correct"]["ultrabert"] & mask).sum())
        conf = result["confidence"][mask]
        lines.append(f"| {name} | {n:,} | {c:,} | {c/n:.4%} | {conf.mean():.4f} | {np.median(conf):.4f} |")
    confidence = result["confidence"]
    correct_conf = confidence[result["correct"]["ultrabert"]]
    wrong_conf = confidence[~result["correct"]["ultrabert"]]
    lines += ["", "UltraBERT confidence quantiles by correctness (p10 / p25 / p50 / p75 / p90):", "",
              f"- Correct: {', '.join(f'{x:.4f}' for x in np.quantile(correct_conf,[.1,.25,.5,.75,.9]))}",
              f"- Incorrect: {', '.join(f'{x:.4f}' for x in np.quantile(wrong_conf,[.1,.25,.5,.75,.9]))}", "",
              "UltraBERT error counts by true label:", "",
              "| True label | Errors | Most common wrong prediction | Count |", "|---|---:|---|---:|"]
    labels_by_pred = np.asarray(classes)
    true_labels = frame.model.astype(str).to_numpy()
    for label in classes:
        mask = (true_labels == label) & ~result["correct"]["ultrabert"]
        counts = pd.Series(labels_by_pred[result["pred_i"]["ultrabert"][mask]]).value_counts()
        common_label = str(counts.index[0]) if len(counts) else "—"
        common_count = int(counts.iloc[0]) if len(counts) else 0
        lines.append(f"| {label} | {int(mask.sum()):,} | {common_label} | {common_count:,} |")
    lines += ["", f"UltraBERT median confidence by true label among correct rows: {json.dumps(result['medians'], ensure_ascii=False)}", "",
        f"Overall confidence p10/p25/p50/p75/p90: {', '.join(f'{x:.4f}' for x in np.quantile(confidence,[.1,.25,.5,.75,.9]))}.",
        f"Correct-row confidence median: {np.median(correct_conf):.4f}; incorrect-row confidence median: {np.median(wrong_conf):.4f}.", "",
        f"Selection is stratified within each category by true label, `source_dataset`, and word-count bucket (short <100; medium 100–300; long >300). Integer quotas use largest-remainder allocation to match each category's requested retention rate as closely as possible; candidate ordering and seed `{seed}` make sampling reproducible.", "",
        "### Cache and outputs", "",
        f"Prediction caches are reused from `{OUT_DIR.relative_to(ROOT)}/`; row decisions are under `{output_dir.relative_to(ROOT) if output_dir.is_absolute() else output_dir}/`.",
        f"Row decisions CSV: `{output_decisions.relative_to(ROOT) if output_decisions.is_absolute() else output_decisions}`. Dataset profile: `{output_profile.relative_to(ROOT) if output_profile.is_absolute() else output_profile}`.",
        "V2.14 source data was not modified. Only V2.14 external rows were selected; no V0 rows were appended.", ""]
    return "\n".join(lines)


def run(*, infer=True, retention=None, seed: int = 2026,
        dataset_version: str = "v2.14-prediction-filtered", output_data: Path = OUT_DATA,
        output_profile: Path = OUT_PROFILE, output_report: Path = OUT_REPORT,
        output_decisions: Path = OUT_DECISIONS, output_dir: Path = OUT_DIR,
        cache_dir: Path = OUT_DIR, verify_existing_classifications: bool = True):
    # Accept path-like strings as well as Path objects for both CLI and Python callers.
    output_data = Path(output_data)
    output_profile = Path(output_profile)
    output_report = Path(output_report)
    output_decisions = Path(output_decisions)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    rates = np.asarray(RETENTION if retention is None else retention, dtype=float)
    if rates.shape != (len(CATEGORY_NAMES),) or not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
        raise ValueError(f"retention must contain {len(CATEGORY_NAMES)} rates in [0, 1]")
    frame = pd.read_parquet(DATA_PATH)
    profile = json.loads(PROFILE_PATH.read_text())
    if len(frame) != int(profile["rows"]) or len(frame) != 231336:
        raise ValueError(f"V2.14 row count changed: {len(frame)}")
    classes = list(profile["classes"])
    if frame.id.isna().any() or frame.id.duplicated().any() or frame.text.isna().any():
        raise ValueError("V2.14 IDs/text are invalid")
    if frame.model.nunique() != 18 or set(frame.model.astype(str)) != set(classes):
        raise ValueError("V2.14 label roster does not match its profile")
    output_dir.mkdir(parents=True, exist_ok=True)
    if infer:
        probs, manifests = get_predictions(frame, classes, profile["dataset_sha256"], cache_dir)
    else:
        probs, manifests = {}, {name: (_load_manifests()[name][0]) for name in CHECKPOINTS}
        for name in CHECKPOINTS:
            probs[name] = _load_cache(name, frame.id.astype(str).to_numpy(), classes,
                                      manifests[name], profile["dataset_sha256"], cache_dir)
            if probs[name] is None:
                raise FileNotFoundError(f"missing valid {name} prediction cache")
    result = classify_rows(frame, probs, classes)
    if verify_existing_classifications:
        reference_path = OUT_DIR / "row_decisions.csv"
        if reference_path.is_file() and reference_path.resolve() != output_decisions.resolve():
            reference = pd.read_csv(reference_path, usecols=["response_id", "category"])
            if not np.array_equal(reference.response_id.astype(str).to_numpy(), frame.id.astype(str).to_numpy()):
                raise ValueError("existing prediction-filter decision IDs do not align with V2.14 rows")
            actual = np.asarray([CATEGORY_NAMES[i] for i in result["category"]], dtype=object)
            if not np.array_equal(reference.category.astype(str).to_numpy(), actual):
                raise ValueError("recomputed categories differ from the existing cached classification")
    keep = select_rows(frame, result["category"], seed, rates)
    decisions = _decision_table(frame, classes, probs, result, keep)
    output_decisions.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_decisions, index=False)
    filtered = frame.loc[keep].copy().reset_index(drop=True)
    output_data.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_data, index=False)
    output_sha = _sha256(output_data)
    v0_path = DATA_PATH.parent / "dataset_v0.parquet"
    v0_profile_path = DATA_PATH.parent / "dataset_profile.json"
    v0 = pd.read_parquet(v0_path)
    if len(v0) != 2699:
        raise ValueError(f"full V0 validation artifact must contain 2,699 rows; found {len(v0)}")
    v0_ids = set(v0.id.astype(str))
    v0_text = set(v0.text.astype(str).map(_normal_text)) - {""}
    output_ids = set(filtered.id.astype(str))
    output_text = set(filtered.text.astype(str).map(_normal_text)) - {""}
    id_overlap = output_ids & v0_ids
    text_overlap = output_text & v0_text
    train_prompts = set()
    for column in ("prompt_group", "prompt_id"):
        if column in filtered:
            train_prompts |= set(filtered[column].dropna().astype(str)) - {""}
    v0_prompts = set(v0.prompt_id.dropna().astype(str)) - {""}
    prompt_overlap = train_prompts & v0_prompts
    if id_overlap or text_overlap or prompt_overlap:
        raise ValueError(f"V0 isolation failed: ids={len(id_overlap)}, text={len(text_overlap)}, prompts={len(prompt_overlap)}")
    new_profile = dict(profile)
    new_profile.update({
        "dataset_version": dataset_version, "dataset_sha256": output_sha,
        "rows": len(filtered), "parent_dataset": "v2.14",
        "parent_dataset_sha256": profile["dataset_sha256"],
        "prediction_filter": {"seed": int(seed), "retention_rates": dict(zip(CATEGORY_NAMES, rates.tolist())),
                              "category_counts": {name: list(map(int, decisions_count)) for name, decisions_count in _category_counts(result["category"], keep).items()},
                              "checkpoints": {name: {"experiment_id": m["experiment_id"], "checkpoint_path": m["checkpoint_path"]}
                                              for name, m in manifests.items()}},
        "artifacts": {"dataset": str(output_data), "profile": str(output_profile),
                      "report": str(output_report), "row_decisions": str(output_decisions)},
        "training_only": True, "external_source_rows_only": True,
        # Avoid _load_dataset's historical behavior of appending held-out V0
        # rows when this profile points to validation_dataset_version='v0'.
        "validation_dataset_version": None,
        "validation_dataset": "v0",
        "validation_isolation": {
            "validation_dataset": "v0", "validation_rows": len(v0),
            "validation_dataset_sha256": (json.loads(v0_profile_path.read_text()).get("dataset_sha256")
                                           if v0_profile_path.is_file() else None),
            "id_overlap": len(id_overlap), "normalized_text_overlap": len(text_overlap),
            "prompt_group_overlap": len(prompt_overlap),
            "scope": "all 2,699 official V0 examples; V0 rows excluded from this external-only training dataset",
        },
        "notes": list(profile.get("notes", [])) + ["Derived by deterministic prediction-guided selection; no V0 rows were appended."],
    })
    new_profile["class_counts"] = {label: int((filtered.model.astype(str) == label).sum()) for label in classes}
    new_profile["rows_by_source_dataset"] = (filtered.source_dataset.fillna("<missing>").astype(str).value_counts().sort_index().astype(int).to_dict()
                                              if "source_dataset" in filtered else {})
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(json.dumps(new_profile, indent=2, ensure_ascii=False) + "\n")
    cache_meta = {}
    for name, (manifest, checkpoint, manifest_path) in _load_manifests().items():
        cache_meta[name] = {"experiment_id": manifest["experiment_id"], "checkpoint_path": str(checkpoint),
                            "dataset": manifest.get("dataset"),
                            "v0_balanced_accuracy": float(manifest.get("metrics", {}).get("balanced_accuracy", float("nan")))}
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_metrics_report(frame, classes, probs, result, keep, cache_meta, output_sha,
        retention=rates, dataset_version=dataset_version, output_data=output_data,
        output_profile=output_profile, output_decisions=output_decisions, output_dir=output_dir, seed=seed))
    return {"rows": len(frame), "retained": int(keep.sum()), "removed": int((~keep).sum()),
            "dataset": str(output_data), "profile": str(output_profile),
            "report": str(output_report), "decisions": str(output_decisions),
            "category_counts": _category_counts(result["category"], keep),
            "categories": result["category"], "keep": keep}


def _category_counts(category, keep):
    result = {}
    for i, name in enumerate(CATEGORY_NAMES):
        total = int((category == i).sum()); retained = int(((category == i) & keep).sum())
        result[name] = (total, retained, total-retained)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true", help="require complete prediction caches; skip inference")
    parser.add_argument("--dataset-version", default="v2.14-prediction-filtered")
    parser.add_argument("--retention", help="six comma-separated rates in category order; defaults to the original rates")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-data", type=Path, default=OUT_DATA)
    parser.add_argument("--output-profile", type=Path, default=OUT_PROFILE)
    parser.add_argument("--output-report", type=Path, default=OUT_REPORT)
    parser.add_argument("--output-decisions", type=Path, default=OUT_DECISIONS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    retention = None
    if args.retention:
        try:
            retention = np.asarray([float(value.strip()) for value in args.retention.split(",")], dtype=float)
        except ValueError as exc:
            parser.error(f"invalid --retention value: {exc}")
    result = run(infer=not args.cached_only, retention=retention, seed=args.seed,
        dataset_version=args.dataset_version, output_data=args.output_data,
        output_profile=args.output_profile, output_report=args.output_report,
        output_decisions=args.output_decisions, output_dir=args.output_dir,
        cache_dir=args.cache_dir)
    result.pop("categories")
    result.pop("keep")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
