#!/usr/bin/env python3
"""Run the resumable V3 contributor-label training factory."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score

from src import e13_reliability_router as e13
from src.pipeline import _dataset_max_length
from src.model_factory import train_catboost_stylometry, train_encoder_fold, train_svc
from src.trust_list2_weighted_vote import _model_family
from src.v3_contributor_label_factory import (FactoryState, LABELS, MODELS,
    equal_dataset_label_weights, fixed_row_folds, load_queue, pooled_metrics, sha256_file)

DATA = ROOT / "data/processed"
OUT = ROOT / "artifacts/analysis/v3"
TMP = OUT / "tmp"
QUEUE_PATH = OUT / "contributor_label_ranking.csv"


def _frame(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"id", "text", "model"}
    if required - set(frame):
        raise ValueError(f"{path} missing required columns {sorted(required - set(frame))}")
    frame["id"] = frame.id.astype(str)
    frame["model"] = frame.model.astype(str)
    frame["text"] = frame.text.fillna("").astype(str)
    frame["_dataset"] = name
    if frame.id.duplicated().any():
        raise ValueError(f"duplicate IDs in {name}")
    if set(frame.model) != set(LABELS):
        raise ValueError(f"{name} label set differs from the canonical 18")
    return frame


def _train_one_model(model_name: str, train: pd.DataFrame, valid: pd.DataFrame, work: Path) -> np.ndarray:
    work.mkdir(parents=True, exist_ok=True)
    y = train.model.astype(str).tolist()
    yv = valid.model.astype(str).tolist()
    texts, vtexts = train.text.tolist(), valid.text.tolist()
    if model_name == "svc":
        cfg = yaml.safe_load((ROOT / "configs/models/svc.yaml").read_text())
        result = train_svc(texts, y, vtexts, list(LABELS), c=cfg.get("C", .5), seed=2026,
            feature_config=cfg.get("features"), cache_dir=work / "features/svc",
            fold_token="v3-factory", validation_labels=yv,
            calibration=cfg.get("calibration"), calibration_cv=cfg.get("calibration_cv", 3))
        import joblib
        joblib.dump({"model": result["model"], "raw_model": result["raw_model"],
            "features": result["features"], "classes": list(LABELS)}, work / "svc.joblib")
    elif model_name == "catboost_stylometry":
        cfg = yaml.safe_load((ROOT / "configs/models/catboost_stylometry.yaml").read_text())
        result = train_catboost_stylometry(texts, y, vtexts, list(LABELS),
            output_dir=work / "catboost", iterations=cfg.get("iterations", 800),
            depth=cfg.get("depth", 7), learning_rate=cfg.get("learning_rate", .05),
            l2_leaf_reg=cfg.get("l2_leaf_reg", 5), thread_count=cfg.get("thread_count", 8),
            seed=2026, validation_labels=yv, early_stopping_rounds=cfg.get("early_stopping_rounds", 100),
            feature_recipe=cfg.get("feature_recipe"), checkpoint_selection_metric="balanced_accuracy",
            validation_sample_weights=equal_dataset_label_weights(valid))
    elif model_name == "ultrabert380":
        cfg = yaml.safe_load((ROOT / "configs/models/ultrabert380.yaml").read_text())
        max_length = _dataset_max_length("ultrabert380", cfg,
            pd.concat([train, valid], ignore_index=True))
        result = train_encoder_fold(texts, y, vtexts, list(LABELS),
            checkpoint=cfg["checkpoint"], revision=cfg["revision"],
            output_dir=work / "ultrabert380", epochs=20,
            batch_size=cfg.get("batch_size", 8), inference_batch_size=600,
            validation_batch_size=600, max_length=max_length, learning_rate=8e-6,
            seed=2026, gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
            local_files_only=True, validation_labels=yv, lr_scheduler_type="linear",
            early_stopping_patience=2, auto_batch_size=cfg.get("auto_batch_size", True),
            max_batch_size=cfg.get("max_batch_size", 1024),
            batch_probe_safety_fraction=cfg.get("batch_probe_safety_fraction", .95),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            select_best_checkpoint=True, checkpoint_selection_metric="balanced_accuracy",
            validation_dataset_tags=valid._dataset.astype(str).tolist())
    else:
        raise ValueError(f"unknown model worker {model_name}")
    result_file = work / "validation_probabilities.npz"
    np.savez_compressed(result_file, probabilities=np.asarray(result["probabilities"], dtype=np.float32))
    (work / "worker_manifest.json").write_text(json.dumps({"model": model_name,
        "rows": len(train), "validation_rows": len(valid), "seed": 2026,
        "training_config": (cfg | {"seed": 2026}),
        "selected_physical_batch_size": result.get("selected_physical_batch_size",
            result.get("effective_batch_size")),
        "gradient_accumulation_steps": result.get("gradient_accumulation_steps",
            cfg.get("gradient_accumulation_steps", 1)),
        "effective_optimizer_batch_size": result.get("effective_optimizer_batch_size"),
        "batch_size_probe": result.get("batch_size_probe"),
        "validation_batch_size": 600 if model_name == "ultrabert380" else None,
        "inference_batch_size": result.get("effective_inference_batch_size", 600
            if model_name == "ultrabert380" else None),
        "selected_epoch": result.get("best_epoch"),
        "selected_iteration": result.get("selected_iteration")}, indent=2) + "\n")
    return result["probabilities"]


def _train_models(train: pd.DataFrame, valid: pd.DataFrame, work: Path) -> np.ndarray:
    """Run CPU model fits alongside UltraBERT in isolated worker processes."""
    work.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = work / "train.parquet", work / "validation.parquet"
    train.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)
    workers = {}
    for model_name in MODELS:
        model_dir = work / model_name
        log = (work / f"{model_name}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            "--model-worker", model_name, "--train-file", str(train_path),
            "--valid-file", str(valid_path), "--output", str(model_dir)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        workers[model_name] = (proc, log)
    failures = []
    for name, (proc, log) in workers.items():
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append((name, code))
    if failures:
        raise RuntimeError(f"model worker failure(s): {failures}; see logs under {work}")
    return np.stack([np.load(work / name / "validation_probabilities.npz")["probabilities"]
                     for name in MODELS])


def _route_cv(probabilities: np.ndarray, rows: pd.DataFrame, folds: dict[str, int]) -> dict:
    """Nested OOF correctness-router evaluation; all feature statistics are fold local."""
    labels = list(LABELS)
    truth = pd.Categorical(rows.model.astype(str), categories=labels).codes.astype(np.int64)
    if (truth < 0).any():
        raise ValueError("unknown validation label")
    dataset = rows._dataset.astype(str).to_numpy()
    fold_ids = np.asarray([folds[str(x)] for x in rows.id], dtype=np.int8)
    model_ids = list(MODELS)
    families = [_model_family(name) for name in model_ids]
    hard = probabilities.argmax(axis=2)
    routed = np.full(len(rows), -1, dtype=np.int64)
    for outer in range(5):
        fit_outer = np.flatnonzero(fold_ids != outer)
        held_outer = np.flatnonzero(fold_ids == outer)
        x_parts, y_parts, w_parts = [], [], []
        # Train router only on inner-OOF examples from the outer training split.
        for inner in range(5):
            target = fit_outer[fold_ids[fit_outer] == inner]
            fit = fit_outer[fold_ids[fit_outer] != inner]
            if not len(target):
                continue
            features, _ = e13.build_feature_frame(probabilities, model_ids, families,
                truth[fit], fit, target, fit_sample_weights=equal_dataset_label_weights(
                    rows.iloc[fit].assign(_dataset=dataset[fit])))
            x_parts.append(features)
            y_parts.append(e13._targets(hard[:, target], truth[target]))
            w_parts.append(np.repeat(equal_dataset_label_weights(
                rows.iloc[target].assign(_dataset=dataset[target])), len(MODELS)))
        router = e13._new_catboost()
        router.fit(e13._cat_frame(pd.concat(x_parts, ignore_index=True)),
            np.concatenate(y_parts), sample_weight=np.concatenate(w_parts),
            cat_features=list(e13.CATEGORICAL), verbose=False)
        features, _ = e13.build_feature_frame(probabilities, model_ids, families,
            truth[fit_outer], fit_outer, held_outer,
            fit_sample_weights=equal_dataset_label_weights(
                rows.iloc[fit_outer].assign(_dataset=dataset[fit_outer])))
        reliability = e13._score_correctness(router, features).reshape(len(held_outer), len(MODELS)).T
        family_prob = e13.family_balanced_probabilities(probabilities[:, held_outer], families)
        routed[held_outer], _ = e13.route_predictions("family_max", hard[:, held_outer],
            reliability, family_prob, families)
    if (routed < 0).any():
        raise AssertionError("outer-fold routing did not cover every validation row")
    result = pooled_metrics(np.asarray(labels)[routed], rows.model.astype(str).to_numpy(), dataset, labels)
    return result


def _episode_runner(v0: pd.DataFrame, v0ext: pd.DataFrame, baseline: pd.DataFrame,
                    v3: pd.DataFrame, folds: dict[str, int]):
    valid = pd.concat([v0, v0ext], ignore_index=True).reset_index(drop=True)
    base_ids = set(baseline.id.astype(str))

    def run(rank: int, contributor: str, label: str, accepted: list,
            work: Path, prior_baseline):
        if rank == 0:
            train = baseline.copy()
        else:
            selected = {tuple(pair) for pair in accepted}
            selected.add((contributor, label))
            candidate_mask = np.zeros(len(v3), dtype=bool)
            for source_dataset, candidate_label in selected:
                candidate_mask |= (v3.source_dataset.astype(str).to_numpy() == source_dataset) & (
                    v3.model.astype(str).to_numpy() == candidate_label)
            candidate = v3.loc[~v3.id.astype(str).isin(base_ids) & candidate_mask]
            train = pd.concat([baseline, candidate], ignore_index=True).drop_duplicates("id")
        if train.id.astype(str).duplicated().any():
            raise ValueError("candidate training rows have duplicate response IDs")
        probs = _train_models(train, valid, work / "models")
        result = _route_cv(probs, valid, folds)
        return result
    return run


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-worker", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--train-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--valid-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--max-episodes", type=int, default=None,
        help="optional rank limit for a deliberate partial run; resume with the same fixed queue")
    args = parser.parse_args()
    if args.model_worker:
        if not args.train_file or not args.valid_file or not args.output:
            parser.error("model worker requires train-file, valid-file and output")
        _train_one_model(args.model_worker, pd.read_parquet(args.train_file),
                         pd.read_parquet(args.valid_file), args.output)
        return
    if not QUEUE_PATH.is_file():
        raise FileNotFoundError(f"fixed contributor-label queue not found: {QUEUE_PATH}")
    queue = load_queue(QUEUE_PATH)
    baseline = _frame(DATA / "dataset_v3.0_baseline.parquet", "v3.0-baseline")
    v3 = _frame(DATA / "dataset_v3.0.parquet", "v3.0")
    v0 = _frame(DATA / "dataset_v0.parquet", "v0").sort_values("id", kind="stable").reset_index(drop=True)
    v0ext = _frame(DATA / "dataset_v0_ext.parquet", "v0_ext").reset_index(drop=True)
    if set(v0.id.astype(str)) & set(v0ext.id.astype(str)):
        raise ValueError("V0 and V0-EXT response IDs must be disjoint")
    validation = pd.concat([v0, v0ext], ignore_index=True).reset_index(drop=True)
    folds = fixed_row_folds(validation)
    hashes = {name: sha256_file(path) for name, path in {
        "queue": QUEUE_PATH, "baseline": DATA / "dataset_v3.0_baseline.parquet",
        "v3": DATA / "dataset_v3.0.parquet", "v0": DATA / "dataset_v0.parquet",
        "v0_ext": DATA / "dataset_v0_ext.parquet"}.items()}
    state = FactoryState(OUT, queue, hashes, folds)
    runner = _episode_runner(v0, v0ext, baseline, v3, folds)
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            parser.error("--max-episodes must be positive")
        # A queue prefix can be completed in chunks; state retains its original
        # queue hash and next rank. The caller may rerun without the limit later.
        original = state.queue
        state.queue = original.iloc[:args.max_episodes].reset_index(drop=True)
    # Catalogue status is updated only after explicit launch, never during import.
    catalog_path = ROOT / "configs/ensemble_catalog_v2.json"
    catalog = json.loads(catalog_path.read_text())
    entry = next(x for x in catalog["experiments"] if x.get("id") == "V3-CONTRIBUTOR-LABEL-FACTORY")
    entry.update({"status": "Running", "current_stage": f"episode rank {state.state['next_rank']}",
                  "factory_state": "artifacts/analysis/v3/factory_state.json"})
    tmp_catalog = catalog_path.with_suffix(".tmp")
    tmp_catalog.write_text(json.dumps(catalog, indent=2) + "\n")
    os.replace(tmp_catalog, catalog_path)
    try:
        state.run(runner, TMP)
    except Exception as exc:
        entry["status"] = "Failed episode; resumable"
        entry["current_stage"] = str(exc)
        tmp_catalog.write_text(json.dumps(catalog, indent=2) + "\n")
        os.replace(tmp_catalog, catalog_path)
        raise
    entry["status"] = "Complete" if state.state["next_rank"] > len(queue) else "Paused"
    entry["current_stage"] = "Complete" if entry["status"] == "Complete" else "Paused at queue rank " + str(state.state["next_rank"])
    tmp_catalog.write_text(json.dumps(catalog, indent=2) + "\n")
    os.replace(tmp_catalog, catalog_path)


if __name__ == "__main__":
    main()
