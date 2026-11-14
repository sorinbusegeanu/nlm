"""Frozen Trust List 2 v013 top-five submission experiment.

The module reuses the NLM inference loaders and the subset-selection score
construction. It does not train models or alter the active trust list.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from .ensemble_factory import fit_weights
from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
from .trust_list2_weighted_vote import (
    FAMILY_COUNTS,
    _checkpoint_score_matrices,
    _label_prediction,
    _model_family,
    _subset_scores,
)
from .v2_checkpoint_complementarity import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    _is_eligible_fallback,
    _sha256_v0,
)
from .v2_checkpoint_trust import LABELS


TRUST_VERSION = "trust-list-2-v013"
SEQUENCE_IDS = ("N0071", "N0056", "N0005", "N0058", "N0072")
EXCLUDED_SEQUENCE_IDS = {"N0028", "N0029", "N0033", "N0041", "N0043"}
DEFAULT_SUBSET_DIR = Path("artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013")
DEFAULT_OUTPUT_DIR = Path("artifacts/kaggle/trust2_v013_top5")
WEIGHT_L2 = 0.05
MIN_RULE_SUPPORT = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_checkpoint(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _rooted(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "rows": int(len(y_true)),
        "correct": int(np.sum(prediction == y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def _load_frozen_roster(root: Path, subset_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    selection_path = subset_dir / "selected_checkpoints.json"
    trust_path = root / "docs" / "trust-list-2-v013.csv"
    audit_path = subset_dir / "checkpoint_training_overlap_audit.csv"
    if not all(path.is_file() for path in (selection_path, trust_path, audit_path)):
        raise FileNotFoundError("V013 selected list, trust CSV, or training-overlap audit is missing")
    selection = json.loads(selection_path.read_text())
    if selection.get("trust_list_version") != TRUST_VERSION:
        raise ValueError("subset artifact is not from Trust List 2 v013")
    members = selection["full_v0_selection"]["selected_by_size"]["5"]
    sequence_order = [str(item["sequence_id"]) for item in members]
    if tuple(sequence_order) != SEQUENCE_IDS:
        raise ValueError(f"frozen V013 top-five changed: {sequence_order}")
    if any(sequence_id in EXCLUDED_SEQUENCE_IDS for sequence_id in sequence_order):
        raise ValueError("the frozen subset contains a checkpoint excluded from v013")

    trust = pd.read_csv(trust_path)
    if set(trust.trust_list_version.astype(str)) != {TRUST_VERSION}:
        raise ValueError("trust CSV contains an unexpected version")
    checkpoint_rows = (trust.loc[trust.sequence_id.astype(str).isin(sequence_order)]
                       .drop_duplicates("sequence_id").set_index("sequence_id"))
    if set(checkpoint_rows.index.astype(str)) != set(sequence_order):
        raise ValueError("one or more selected checkpoints are missing from the V013 trust list")
    # Trust artifacts are ordered by their selection order, not by the CSV's
    # per-label SeqID column.
    checkpoint_rows = checkpoint_rows.loc[list(sequence_order)].reset_index()
    if not checkpoint_rows.checkpoint_available.astype(str).str.lower().isin({"true", "1"}).all():
        raise FileNotFoundError("a selected checkpoint is marked unavailable in v013")
    if not checkpoint_rows.evaluation_eligible.astype(str).str.lower().isin({"true", "1"}).all():
        raise ValueError("a selected checkpoint is not evaluation eligible")
    if pd.to_numeric(checkpoint_rows.validation_training_overlap, errors="coerce").fillna(-1).ne(0).any():
        raise ValueError("selected checkpoint has a reported V0 training overlap")

    audit = pd.read_csv(audit_path, dtype={"sequence_id": str})
    audited = audit.set_index("sequence_id").reindex(sequence_order)
    if audited.isna().any(axis=None):
        raise ValueError("selected checkpoint is missing from the audited 68-checkpoint roster")
    if pd.to_numeric(audited.v0_training_id_overlap, errors="coerce").fillna(-1).ne(0).any():
        raise ValueError("overlap audit found a V0 training ID overlap")
    if pd.to_numeric(audited.v0_training_normalized_text_overlap, errors="coerce").fillna(-1).ne(0).any():
        raise ValueError("overlap audit found a normalized V0 text overlap")
    if not audited.audit_status.astype(str).str.contains("zero V0 ID/text overlap", case=False).all():
        raise ValueError("selected checkpoint lineage lacks an explicit zero-overlap audit")

    manifests: dict[str, dict] = {}
    for row in checkpoint_rows.itertuples(index=False):
        sequence_id = str(row.sequence_id)
        experiment_id = str(row.experiment_id)
        manifest_path = root / "artifacts" / "predictions" / f"{experiment_id}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"source prediction manifest missing for {sequence_id}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        checkpoint = _rooted(root, row.checkpoint_path)
        manifest_checkpoint = _rooted(root, manifest.get("selected_checkpoint_path") or
                                       manifest.get("best_checkpoint_path") or manifest["checkpoint_path"])
        if checkpoint.resolve() != manifest_checkpoint.resolve() or not checkpoint.exists():
            raise FileNotFoundError(f"V013 checkpoint path does not match the source manifest for {sequence_id}")
        if str(manifest.get("model")) != str(row.model):
            raise ValueError(f"model identity mismatch for {sequence_id}")
        manifests[sequence_id] = {**manifest, "_manifest_path": str(manifest_path),
                                  "_checkpoint_path": str(checkpoint.resolve())}
    return selection, trust, checkpoint_rows, audited.reset_index(), manifests


def _load_v0(root: Path, data_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(data_dir / "dataset_v0.parquet")
    required = {"id", "prompt_id", "text", "model"}
    if required - set(frame):
        raise ValueError(f"V0 lacks required columns {sorted(required - set(frame))}")
    frame = frame.copy()
    for column in required:
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    if len(frame) != 2699 or frame.id.duplicated().any() or set(frame.model) != set(LABELS):
        raise ValueError("V0 must contain 2,699 unique examples and all canonical labels")
    return frame


def _load_probability_cache(experiment_id: str, v0: pd.DataFrame, *, root: Path,
                            cache_dir: Path, artifact_dir: Path) -> tuple[np.ndarray, Path]:
    candidates = (cache_dir / f"{experiment_id}.parquet",
                  artifact_dir / f"{experiment_id}_validation.parquet")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no saved V0 probability cache for {experiment_id}")
    frame = pd.read_parquet(path)
    required = {"id", "y_true", *(f"prob::{label}" for label in LABELS)}
    if missing := required - set(frame.columns):
        raise ValueError(f"prediction cache {path} lacks {sorted(missing)}")
    frame = frame.copy()
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any() or len(frame) != len(v0) or set(frame.id) != set(v0.id):
        raise ValueError(f"prediction cache IDs do not exactly match full V0: {path}")
    aligned = frame.set_index("id").loc[v0.id.tolist()]
    if not np.array_equal(aligned.y_true.astype(str).to_numpy(), v0.model.to_numpy()):
        raise ValueError(f"prediction labels do not align to V0: {path}")
    if "prompt_id" in aligned and not np.array_equal(aligned.prompt_id.astype(str).to_numpy(), v0.prompt_id.to_numpy()):
        raise ValueError(f"prediction prompt IDs do not align to V0: {path}")
    probabilities = aligned[[f"prob::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    _validate_probabilities(probabilities, len(v0), len(LABELS))
    return probabilities, path


def _validate_probabilities(probabilities: np.ndarray, rows: int, classes: int) -> None:
    if probabilities.shape != (rows, classes) or not np.isfinite(probabilities).all():
        raise ValueError("probability matrix has an invalid shape or nonfinite values")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability matrix is not normalized")


def _fit_trust_rule_weights(trust_rows: pd.DataFrame, top1: dict[str, np.ndarray],
                            y_true: np.ndarray, fit_indices: np.ndarray | None = None) -> dict:
    weights = {}
    for row in trust_rows.itertuples(index=False):
        sequence_id, label = str(row.sequence_id), str(row.target_label)
        if fit_indices is None:
            support = int(row.prediction_count)
            precision = float(row.precision)
        else:
            predicted = top1[sequence_id][fit_indices]
            y_fit = y_true[fit_indices]
            mask = predicted == label
            support = int(mask.sum())
            if support < MIN_RULE_SUPPORT:
                continue
            precision = float(np.mean(y_fit[mask] == label))
        weights[(sequence_id, label)] = precision * support / (support + FAMILY_COUNTS)
    return weights


def _trust5_scores(sequence_ids: list[str], trust_rows: pd.DataFrame,
                   top1: dict[str, np.ndarray], weights: dict,
                   family_by_sequence: dict[str, str], fallback_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    matrices = _checkpoint_score_matrices(sequence_ids, top1, weights, tuple(LABELS))
    scores = _subset_scores(sequence_ids, matrices, family_by_sequence)
    has_vote = scores.sum(axis=1) > 0
    prediction = _label_prediction(scores)
    if (~has_vote).any():
        prediction[~has_vote] = top1[fallback_sequence][~has_vote]
    return scores, prediction


def _submission_frame(ids: pd.Series, prediction: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame({"id": ids.astype(str).to_numpy(), "model": prediction.astype(str)})
    if output.id.duplicated().any() or output.id.isna().any() or output.model.isna().any():
        raise ValueError("submission IDs and labels must be complete and unique")
    if not set(output.model).issubset(set(LABELS)):
        raise ValueError("submission contains labels outside the canonical 18-class vocabulary")
    return output


def _predict_one(sequence_id: str, manifest: dict, texts: list[str]) -> np.ndarray:
    from joblib import load

    model = str(manifest["model"])
    checkpoint = manifest["_checkpoint_path"]
    config = manifest.get("config", {})
    if model == "svc":
        probabilities, returned_classes = predict_svc_bundle(load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probabilities, returned_classes = predict_catboost_stylometry(
            checkpoint, texts, LABELS, temperature=float(manifest.get("calibration_temperature", 1.0)))
    else:
        # Granite's optional Mamba CUDA extensions cannot run on CPU. Its
        # Transformers model supplies equivalent reference PyTorch functions;
        # prevent the CUDA-only optional packages from shadowing those functions
        # when this workflow is intentionally run CPU-only.
        torch = importlib.import_module("torch")
        masked_modules = ("causal_conv1d", "mamba_ssm")
        previous_modules = {name: sys.modules.get(name, ...) for name in masked_modules}
        if model == "granite4_h_micro" and not torch.cuda.is_available():
            for name in masked_modules:
                sys.modules[name] = None
        try:
            probabilities, returned_classes = _predict_neural(
                model, checkpoint, texts, LABELS,
                batch_size=int(config.get("inference_batch_size", 8)),
                max_length=int(config.get("max_length", 384)),
                temperature=float(manifest.get("calibration_temperature", 1.0)),
                base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision"),
                fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
                text_only_backbone=bool(config.get("text_only_backbone", False)),
                quantization=config.get("quantization", manifest.get("quantization")),
                attention_backend=config.get("attention_backend", manifest.get("attention_backend")))
        finally:
            for name, previous in previous_modules.items():
                if previous is ...:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
    if list(returned_classes) != list(LABELS):
        index = {label: column for column, label in enumerate(returned_classes)}
        probabilities = probabilities[:, [index[label] for label in LABELS]]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    _validate_probabilities(probabilities, len(texts), len(LABELS))
    return probabilities


def _load_test_predictions(test: pd.DataFrame, members: pd.DataFrame, manifests: dict,
                           output_dir: Path, test_hash: str, *, force: bool = False) -> dict[str, np.ndarray]:
    from joblib import load

    prediction_dir = output_dir / "checkpoint_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for row in members.itertuples(index=False):
        sequence_id, experiment_id = str(row.sequence_id), str(row.experiment_id)
        cache_path = prediction_dir / f"{sequence_id}.parquet"
        meta_path = prediction_dir / f"{sequence_id}.json"
        if cache_path.is_file() and meta_path.is_file() and not force:
            metadata = json.loads(meta_path.read_text())
            if metadata.get("test_sha256") != test_hash:
                raise ValueError(f"test prediction cache for {sequence_id} is for a different test.csv")
            frame = pd.read_parquet(cache_path)
            if frame.id.astype(str).tolist() != test.id.astype(str).tolist():
                raise ValueError(f"cached prediction IDs/order changed for {sequence_id}")
            probs = frame[[f"prob::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
            _validate_probabilities(probs, len(test), len(LABELS))
        else:
            manifest = manifests[sequence_id]
            if manifest["model"] == "svc":
                probabilities, returned_classes = predict_svc_bundle(load(manifest["_checkpoint_path"]), test.text.tolist())
                if list(returned_classes) != list(LABELS):
                    order = {label: i for i, label in enumerate(returned_classes)}
                    probabilities = probabilities[:, [order[label] for label in LABELS]]
                probs = np.asarray(probabilities, dtype=np.float64)
            else:
                probs = _predict_one(sequence_id, manifest, test.text.tolist())
            _validate_probabilities(probs, len(test), len(LABELS))
            frame = pd.DataFrame({"id": test.id.astype(str).to_numpy()})
            for column, label in enumerate(LABELS):
                frame[f"prob::{label}"] = probs[:, column]
            frame.to_parquet(cache_path, index=False)
            meta_path.write_text(json.dumps({
                "sequence_id": sequence_id, "experiment_id": experiment_id,
                "test_sha256": test_hash, "probability_sha256": _sha256(cache_path),
                "rows": int(len(frame)), "classes": list(LABELS),
            }, indent=2) + "\n")
        results[sequence_id] = probs
        # The predictors return no model object, but explicitly clear allocator
        # caches between checkpoints to keep peak VRAM bounded.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return results


def run(*, root: str | Path = ".", test_path: str | Path = "data/raw/test.csv",
        output_dir: str | Path = DEFAULT_OUTPUT_DIR, force_inference: bool = False) -> dict:
    root = Path(root).resolve()
    subset_dir = root / DEFAULT_SUBSET_DIR
    output_dir = _rooted(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection, trust, members, audit, manifests = _load_frozen_roster(root, subset_dir)
    sequence_ids = members.sequence_id.astype(str).tolist()
    v0 = _load_v0(root, root / DEFAULT_DATA_DIR)
    validation_hash = _sha256_v0(v0)
    if validation_hash != selection.get("validation_sha256"):
        raise ValueError("V013 subset validation hash differs from local V0")

    selected_rules = trust.loc[trust.sequence_id.astype(str).isin(sequence_ids)].copy()
    eligible_rules = selected_rules.loc[selected_rules.apply(_is_eligible_fallback, axis=1)].copy()
    eligible_rules["sequence_id"] = eligible_rules.sequence_id.astype(str)
    eligible_rules["target_label"] = eligible_rules.target_label.astype(str)
    if eligible_rules.duplicated(["sequence_id", "target_label"]).any():
        raise ValueError("V013 selected checkpoint-label rules are duplicated")

    cache_dir = root / DEFAULT_CACHE_DIR
    artifact_dir = root / DEFAULT_ARTIFACT_DIR
    v0_probs: dict[str, np.ndarray] = {}
    v0_sources = {}
    for row in members.itertuples(index=False):
        probs, source = _load_probability_cache(str(row.experiment_id), v0, root=root,
                                                cache_dir=cache_dir, artifact_dir=artifact_dir)
        v0_probs[str(row.sequence_id)] = probs
        v0_sources[str(row.sequence_id)] = str(source)
    top1 = {sequence_id: np.asarray(LABELS, dtype=object)[probs.argmax(axis=1)]
            for sequence_id, probs in v0_probs.items()}
    family_by_sequence = {str(row.sequence_id): _model_family(str(row.model))
                          for row in members.itertuples(index=False)}
    y_true = v0.model.to_numpy(dtype=object)

    # A deterministic fallback is needed only when no selected checkpoint
    # proposes a label with an eligible v013 trust rule. Highest V013 overall
    # accuracy wins; selected order breaks ties.
    overall_accuracy = {str(row.sequence_id): float(row.overall_accuracy)
                        for row in members.itertuples(index=False)}
    fallback_sequence = max(sequence_ids, key=lambda value: (overall_accuracy[value], -sequence_ids.index(value)))

    full_rule_weights = _fit_trust_rule_weights(eligible_rules, top1, y_true)
    trust_scores, trust_prediction = _trust5_scores(sequence_ids, eligible_rules, top1,
                                                    full_rule_weights, family_by_sequence,
                                                    fallback_sequence)
    reference_path = subset_dir / "full_v0_predictions" / "k05.parquet"
    reference = pd.read_parquet(reference_path).set_index("id").loc[v0.id.tolist()]
    if not np.array_equal(trust_prediction, reference.y_pred.astype(str).to_numpy(dtype=object)):
        raise ValueError("TRUST5 full-V0 replay does not reproduce the saved V013 k05 predictions")
    trust_full_metrics = _metrics(y_true, trust_prediction)
    if trust_full_metrics["correct"] != 1717 or not np.isclose(trust_full_metrics["accuracy"], 0.6361615413115969, atol=1e-12):
        raise ValueError(f"TRUST5 full-V0 replay did not reproduce 1,717/2,699: {trust_full_metrics}")

    equal_full_probs = np.mean(np.stack([v0_probs[key] for key in sequence_ids]), axis=0)
    equal_full_prediction = np.asarray(LABELS, dtype=object)[equal_full_probs.argmax(axis=1)]
    weighted_full_weights = fit_weights([v0_probs[key] for key in sequence_ids], y_true, LABELS, l2=WEIGHT_L2)
    weighted_full_probs = np.einsum("m,nmc->nc", weighted_full_weights,
                                    np.stack([v0_probs[key] for key in sequence_ids], axis=1))
    weighted_full_prediction = np.asarray(LABELS, dtype=object)[weighted_full_probs.argmax(axis=1)]

    # Identical GroupKFold partitions for all three fixed-roster strategies.
    groups = v0.prompt_id.astype(str).to_numpy()
    oof_predictions = {name: np.full(len(v0), None, dtype=object)
                       for name in ("TRUST5", "EQUAL5", "WEIGHTED5")}
    fold_weights = {}
    splitter = GroupKFold(n_splits=5)
    for fold_id, (fit_idx, held_idx) in enumerate(splitter.split(v0, y_true, groups), start=1):
        if set(groups[fit_idx]) & set(groups[held_idx]):
            raise AssertionError("prompt group leaked across OOF fold")
        fold_top1 = {key: top1[key] for key in sequence_ids}
        fold_rules = _fit_trust_rule_weights(eligible_rules, fold_top1, y_true, fit_idx)
        _, trust_oof = _trust5_scores(sequence_ids, eligible_rules, top1, fold_rules,
                                      family_by_sequence, fallback_sequence)
        oof_predictions["TRUST5"][held_idx] = trust_oof[held_idx]
        equal_held = np.mean(np.stack([v0_probs[key][held_idx] for key in sequence_ids]), axis=0)
        oof_predictions["EQUAL5"][held_idx] = np.asarray(LABELS, dtype=object)[equal_held.argmax(axis=1)]
        weights = fit_weights([v0_probs[key][fit_idx] for key in sequence_ids], y_true[fit_idx],
                              LABELS, l2=WEIGHT_L2)
        weighted_held = np.einsum("m,nmc->nc", weights,
                                  np.stack([v0_probs[key][held_idx] for key in sequence_ids], axis=1))
        oof_predictions["WEIGHTED5"][held_idx] = np.asarray(LABELS, dtype=object)[weighted_held.argmax(axis=1)]
        fold_weights[str(fold_id)] = {key: float(value) for key, value in zip(sequence_ids, weights)}
    if any(pd.isna(values).any() for values in oof_predictions.values()):
        raise ValueError("incomplete grouped OOF predictions")

    # Verify the previously published five-fold subset-selection score against
    # its saved OOF artifact. That historical artifact permits fold-specific
    # checkpoint selection, whereas this experiment freezes the global top 5.
    historical_oof_path = subset_dir / "oof_predictions" / "k05_oof.parquet"
    historical_oof = pd.read_parquet(historical_oof_path).set_index("id").loc[v0.id.tolist()]
    historical_metrics = _metrics(y_true, historical_oof.y_pred.astype(str).to_numpy(dtype=object))
    if historical_metrics["correct"] != 1679 or not np.isclose(historical_metrics["accuracy"], 0.6220822526861801, atol=1e-12):
        raise ValueError("saved V013 five-fold subset-selection OOF artifact failed its recorded score check")

    full_metrics = {
        "TRUST5": trust_full_metrics,
        "EQUAL5": _metrics(y_true, equal_full_prediction),
        "WEIGHTED5": _metrics(y_true, weighted_full_prediction),
    }
    oof_metrics = {name: _metrics(y_true, values) for name, values in oof_predictions.items()}
    oof_metrics["TRUST5_historical_subset_selection"] = historical_metrics

    # Save V0 and OOF diagnostics in the output bundle for transparent replay.
    v0_artifacts = pd.DataFrame({"id": v0.id, "prompt_id": v0.prompt_id, "y_true": y_true})
    for name, values in (("TRUST5", trust_prediction), ("EQUAL5", equal_full_prediction),
                         ("WEIGHTED5", weighted_full_prediction)):
        v0_artifacts[f"pred::{name}"] = values
    for name, values in oof_predictions.items():
        v0_artifacts[f"oof_pred::{name}"] = values
    v0_artifacts.to_parquet(output_dir / "v0_evaluation_predictions.parquet", index=False)

    test_path = _rooted(root, test_path)
    test_bytes_hash = _sha256(test_path)
    test = pd.read_csv(test_path, dtype={"id": "string"})
    if not {"id", "text"}.issubset(test.columns):
        raise ValueError("test.csv must contain id and text")
    if test.id.isna().any() or test.id.duplicated().any() or test.text.isna().any():
        raise ValueError("test IDs/texts must be present; IDs must be unique")
    if test.text.astype(str).str.strip().eq("").any():
        raise ValueError("test text contains empty responses")
    test_probs = _load_test_predictions(test, members, manifests, output_dir,
                                        test_bytes_hash, force=force_inference)
    test_top1 = {key: np.asarray(LABELS, dtype=object)[test_probs[key].argmax(axis=1)]
                 for key in sequence_ids}
    test_rules = _fit_trust_rule_weights(eligible_rules, top1, y_true)
    _, trust_test_prediction = _trust5_scores(sequence_ids, eligible_rules, test_top1,
                                              test_rules, family_by_sequence, fallback_sequence)
    equal_test_probs = np.mean(np.stack([test_probs[key] for key in sequence_ids]), axis=0)
    equal_test_prediction = np.asarray(LABELS, dtype=object)[equal_test_probs.argmax(axis=1)]
    weighted_test_probs = np.einsum("m,nmc->nc", weighted_full_weights,
                                    np.stack([test_probs[key] for key in sequence_ids], axis=1))
    weighted_test_prediction = np.asarray(LABELS, dtype=object)[weighted_test_probs.argmax(axis=1)]

    output_names = {
        "TRUST5": "submission_trust5.parquet",
        "EQUAL5": "submission_equal5.parquet",
        "WEIGHTED5": "submission_weighted5.parquet",
    }
    test_outputs = {"TRUST5": trust_test_prediction, "EQUAL5": equal_test_prediction,
                    "WEIGHTED5": weighted_test_prediction}
    hashes = {}
    for strategy, prediction in test_outputs.items():
        frame = _submission_frame(test.id, prediction)
        if frame.id.astype(str).tolist() != test.id.astype(str).tolist():
            raise ValueError(f"{strategy} output does not preserve test order")
        path = output_dir / output_names[strategy]
        frame.to_parquet(path, index=False)
        check = pd.read_parquet(path)
        if check.columns.tolist() != ["id", "model"] or check.id.astype(str).tolist() != test.id.astype(str).tolist():
            raise ValueError(f"invalid saved submission layout for {strategy}")
        hashes[strategy] = {"filename": output_names[strategy], "sha256": _sha256(path),
                            "rows": int(len(frame)), "columns": frame.columns.tolist(),
                            "prediction_counts": frame.model.value_counts().sort_index().to_dict()}

    source_hashes = {}
    checkpoint_hashes = {}
    v013_meta = json.loads((root / "docs" / "trust-list-2-v013.json").read_text())
    listed_hashes = {str(item["sequence_id"]): item.get("checkpoint_sha256")
                     for item in v013_meta.get("inference_runs", [])}
    for row in members.itertuples(index=False):
        sequence_id = str(row.sequence_id)
        manifest = manifests[sequence_id]
        source_hashes[sequence_id] = _sha256(Path(manifest["_manifest_path"]))
        checkpoint_hashes[sequence_id] = _sha256_checkpoint(Path(manifest["_checkpoint_path"]))

    prediction_hashes = {}
    for sequence_id in sequence_ids:
        path = output_dir / "checkpoint_predictions" / f"{sequence_id}.parquet"
        prediction_hashes[sequence_id] = _sha256(path)
    manifest = {
        "experiment": "trust2_v013_top5",
        "trust_list_version": TRUST_VERSION,
        "trust_list_csv": "docs/trust-list-2-v013.csv",
        "subset_selection_artifact": str((subset_dir / "selected_checkpoints.json").relative_to(root)),
        "validation_sha256": validation_hash,
        "validation_rows": len(v0),
        "checkpoints": [
            {"selection_order": order, "sequence_id": str(row.sequence_id),
             "experiment_id": str(row.experiment_id), "model": str(row.model),
             "training_dataset": str(row.training_dataset),
             "dataset_lineage": str(row.dataset_lineage),
             "checkpoint_path": str(_rooted(root, row.checkpoint_path)),
             "checkpoint_payload_sha256": checkpoint_hashes[str(row.sequence_id)],
             "checkpoint_sha256_from_v013_catalog": listed_hashes.get(str(row.sequence_id)),
             "source_manifest": manifests[str(row.sequence_id)]["_manifest_path"],
             "source_manifest_sha256": source_hashes[str(row.sequence_id)],
             "model_revision": manifests[str(row.sequence_id)].get("model_revision"),
             "inference_config": manifests[str(row.sequence_id)].get("config", {}),
             "audit_status": str(audit.set_index("sequence_id").loc[str(row.sequence_id), "audit_status"]),
             "v0_training_id_overlap": int(audit.set_index("sequence_id").loc[str(row.sequence_id), "v0_training_id_overlap"]),
             "v0_training_normalized_text_overlap": int(audit.set_index("sequence_id").loc[str(row.sequence_id), "v0_training_normalized_text_overlap"]),
             "validation_prediction_cache": v0_sources[str(row.sequence_id)]}
            for order, row in enumerate(members.itertuples(index=False), start=1)
        ],
        "strategies": {
            "TRUST5": {"algorithm": "V013 subset score: per eligible top-1 label precision * support/(support+30), divided by selected checkpoint count for its model family, summed by label; argmax in canonical label order",
                        "eligible_rule_minimum_support": MIN_RULE_SUPPORT,
                        "fallback": {"when": "no selected checkpoint contributes an eligible V013 rule vote",
                                     "checkpoint_sequence_id": fallback_sequence,
                                     "selection": "highest V013 full-V0 overall accuracy among the five; selection order breaks ties"},
                        "validation_full_v0": full_metrics["TRUST5"],
                        "validation_fixed_roster_oof": oof_metrics["TRUST5"],
                        "historical_fold_selected_subset_oof": historical_metrics},
            "EQUAL5": {"algorithm": "equal mean of five aligned canonical 18-class probability vectors",
                       "weights": {key: 0.2 for key in sequence_ids},
                       "validation_full_v0": full_metrics["EQUAL5"],
                       "validation_fixed_roster_oof": oof_metrics["EQUAL5"]},
            "WEIGHTED5": {"algorithm": "src.ensemble_factory.fit_weights; nonnegative simplex; multiclass log-loss plus L2 shrinkage to equal weights",
                          "fit_l2": WEIGHT_L2,
                          "final_weights": {key: float(value) for key, value in zip(sequence_ids, weighted_full_weights)},
                          "fold_weights": fold_weights,
                          "validation_full_v0": full_metrics["WEIGHTED5"],
                          "validation_fixed_roster_oof": oof_metrics["WEIGHTED5"]},
        },
        "test": {"path": str(test_path), "sha256": test_bytes_hash, "rows": int(len(test)),
                 "input_ids_sha256": hashlib.sha256("\n".join(test.id.astype(str)).encode()).hexdigest(),
                 "inference_device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
                 "granite_cpu_kernel_mode": ("Transformers reference PyTorch Mamba/causal-convolution functions"
                                             if not __import__("torch").cuda.is_available() else "CUDA kernels"),
                 "checkpoint_probability_caches": {
                     sequence_id: {"path": f"checkpoint_predictions/{sequence_id}.parquet",
                                   "sha256": prediction_hashes[sequence_id]}
                     for sequence_id in sequence_ids},
                 "submissions": hashes},
        "same_cached_checkpoint_probabilities_for_all_strategies": True,
        "notes": [
            "V013 rules, prediction caches, and the five-checkpoint selection were frozen; no reselection or training was performed.",
            "The 63.62% full-V0 score is a development-set result; V0 predictions/rules and checkpoint selection were evaluated on V0.",
            "Historical 62.21% OOF reproduced from the saved subset-selection OOF artifact; that procedure selected checkpoints separately within folds. Fixed-roster OOF values are reported separately.",
        ],
    }
    manifest_path = output_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args(argv)
    result = run(root=args.root, test_path=args.test, output_dir=args.output,
                 force_inference=args.force_inference)
    print(json.dumps({"output_dir": args.output,
                      "full_v0": {key: value["validation_full_v0"] for key, value in result["strategies"].items()},
                      "fixed_roster_oof": {key: value["validation_fixed_roster_oof"] for key, value in result["strategies"].items()},
                      "submissions": result["test"]["submissions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
