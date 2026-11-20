"""Run and submit ORACLE05 A/B/C on the competition test rows.

The inference pass is CPU-only by default so it can run while the NLM GPU queue
continues. Per-checkpoint test probabilities are cached and submissions are
resumable. The ORACLE02 deployment model and ORACLE05 full-V0 diagnostic rules
are held fixed from their saved artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
TRUST = ROOT / "docs/trust-list-2-v025.csv"
TRUST_VERSION = "trust-list-2-v025"
CACHE_META = ROOT / "artifacts/analysis/ensemble_precision_recall_overlap/prediction_cache_manifest.json"
ORACLE02_MODEL = ROOT / "artifacts/deployment/oracle04/oracle02_catboost.cbm"
ORACLE04_DEPLOYMENT = ROOT / "artifacts/deployment/oracle04/deployment_manifest.json"
ORACLE05_RULES = ROOT / "artifacts/analysis/oracle05/final_rules_full_v0.json"
CLASSES = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]
DESCRIPTIONS = {
    "A_pair": "ORACLE05 A fixed label-pair top-2 rules Trust List 2 v025",
    "B_margin": "ORACLE05 B probability-margin top-2 rules Trust List 2 v025",
    "C_checkpoint": "ORACLE05 C checkpoint-informed top-2 rules Trust List 2 v025",
}
HF_CACHE = Path.home() / ".cache/huggingface/hub"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_snapshot(model_id: str | None, revision: str | None) -> Path | None:
    if not model_id:
        return None
    repo = HF_CACHE / ("models--" + model_id.replace("/", "--"))
    snapshots = repo / "snapshots"
    if revision and (snapshots / revision).is_dir():
        return snapshots / revision
    ref = repo / "refs/main"
    if ref.is_file() and (snapshots / ref.read_text().strip()).is_dir():
        return snapshots / ref.read_text().strip()
    found = sorted(p for p in snapshots.iterdir()) if snapshots.is_dir() else []
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(f"cannot resolve cached base {model_id}@{revision}: {repo}")


def _candidate_probabilities(sid: str, row: pd.Series, texts: list[str], output: Path,
                             classes: list[str]) -> np.ndarray:
    from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle

    raw_checkpoint = Path(str(row.checkpoint_path))
    checkpoint = raw_checkpoint if raw_checkpoint.is_absolute() else ROOT / raw_checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"{sid} checkpoint is missing: {checkpoint}")
    exp = str(row.experiment_id)
    manifest_path = ROOT / "artifacts/predictions" / f"{exp}_manifest.json"
    run_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    cfg = run_manifest.get("config") or {}
    internal_path = checkpoint / "checkpoint_manifest.json" if checkpoint.is_dir() else None
    ckpt_meta = json.loads(internal_path.read_text()) if internal_path and internal_path.is_file() else {}
    model = str(row.model)
    temperature = float(run_manifest.get("calibration_temperature", 1.0) or 1.0)
    if model == "svc":
        probabilities, returned = predict_svc_bundle(joblib.load(checkpoint), texts)
        indexes = {label: i for i, label in enumerate(returned)}
        probabilities = probabilities[:, [indexes[label] for label in classes]]
    elif model == "catboost_stylometry":
        probabilities, _ = predict_catboost_stylometry(checkpoint, texts, classes, temperature=temperature)
    else:
        # Transformers currently prefers the installed CUDA-only causal_conv1d
        # and mamba_ssm packages for Granite H-Micro. On this CPU-only run, hide
        # those optional accelerators so its reference PyTorch path is selected.
        if model == "granite4_h_micro":
            import torch
            if not torch.cuda.is_available():
                sys.modules["causal_conv1d"] = None
                sys.modules["mamba_ssm"] = None
        base = str(cfg.get("checkpoint") or ckpt_meta.get("checkpoint") or "")
        revision = str(run_manifest.get("model_revision") or cfg.get("revision") or ckpt_meta.get("revision") or "") or None
        base_path = _base_snapshot(base, revision)
        model_checkpoint = checkpoint
        if checkpoint.is_dir() and not any((checkpoint / name).is_file() for name in
                                           ("tokenizer.json", "vocab.txt", "spiece.model", "tokenizer.model")):
            if base_path is None:
                raise FileNotFoundError(f"{sid} has no tokenizer and no cached base tokenizer")
            model_checkpoint = output / "checkpoint_views" / sid
            if model_checkpoint.exists():
                shutil.rmtree(model_checkpoint)
            shutil.copytree(checkpoint, model_checkpoint)
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                         "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
                if (base_path / name).is_file():
                    shutil.copy2(base_path / name, model_checkpoint / name)
        max_length = int(ckpt_meta.get("max_length") or cfg.get("max_length") or 384)
        probabilities, _ = _predict_neural(
            model, model_checkpoint, texts, classes, batch_size=1, max_length=max_length,
            temperature=temperature, base_checkpoint=base or None, revision=revision,
            fix_mistral_regex=bool(cfg.get("fix_mistral_regex", False)),
            text_only_backbone=bool(cfg.get("text_only_backbone", False)),
            local_base_model=base_path,
            quantization=str(cfg.get("quantization") or ckpt_meta.get("quantization") or "none"),
            attention_backend=cfg.get("attention_backend") or ckpt_meta.get("attention_backend"),
            code_revision=cfg.get("code_revision") or ckpt_meta.get("code_revision"),
        )
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(texts), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probabilities from {sid}: {probabilities.shape}")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError(f"unnormalized probabilities from {sid}")
    return probabilities


def _load_or_infer(test: pd.DataFrame, output: Path, *, force: bool = False) -> tuple[list[str], dict[str, np.ndarray], pd.DataFrame]:
    from src.v2_checkpoint_trust import LABELS

    trust = pd.read_csv(TRUST, dtype=str, keep_default_na=False)
    if set(trust.trust_list_version) != {TRUST_VERSION}:
        raise ValueError("ORACLE05 requires the frozen Trust List 2 v025 roster")
    meta = json.loads(CACHE_META.read_text())
    if meta.get("trust_list_version") != TRUST_VERSION:
        raise ValueError("checkpoint cache metadata is not Trust List 2 v025")
    ids = list(meta["checkpoint_ids"])
    if len(ids) != 78 or len(set(ids)) != 78:
        raise ValueError(f"expected 78 cached v025 checkpoint IDs, found {len(ids)}")
    by_id = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    if not set(ids).issubset(by_id.index):
        raise ValueError("v025 cache IDs do not align with the frozen trust list")
    test_hash = hashlib.sha256(pd.util.hash_pandas_object(test, index=False).values.tobytes()).hexdigest()
    pred_dir = output / "checkpoint_probabilities"
    pred_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for number, sid in enumerate(ids, 1):
        path = pred_dir / f"{sid}.npy"
        metadata_path = pred_dir / f"{sid}.json"
        valid_cache = False
        if not force and path.is_file() and metadata_path.is_file():
            saved = json.loads(metadata_path.read_text())
            valid_cache = saved.get("test_sha256") == test_hash and saved.get("trust_list_version") == TRUST_VERSION
        if valid_cache:
            probs = np.load(path)
        else:
            started = time.monotonic()
            probs = _candidate_probabilities(sid, by_id.loc[sid], test.text.astype(str).tolist(), output, LABELS)
            np.save(path, probs.astype(np.float32))
            metadata_path.write_text(json.dumps({"sequence_id": sid, "experiment_id": by_id.loc[sid, "experiment_id"],
                "test_sha256": test_hash, "trust_list_version": TRUST_VERSION,
                "probability_sha256": _sha(path), "seconds": time.monotonic() - started}, indent=2) + "\n")
        arrays[sid] = np.asarray(probs, dtype=np.float64)
        print(f"[{number:02d}/78] {sid} {by_id.loc[sid, 'model']} {'cached' if valid_cache else 'inferred'}", flush=True)
    return ids, arrays, by_id


def _build_predictions(test: pd.DataFrame, ids: list[str], matrices: dict[str, np.ndarray],
                       by_id: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    from src.ensemble_precision_recall_overlap import EQUAL9
    from src.oracle02 import _align_proba, _build_features
    from src.oracle05 import _apply_rules, _family
    from src.v2_checkpoint_trust import LABELS

    equal9 = np.mean(np.stack([matrices[sid] for sid in EQUAL9], axis=0), axis=0)
    equal9 /= equal9.sum(axis=1, keepdims=True)
    features, feature_names = _build_features(ids, matrices, equal9)
    cb = CatBoostClassifier()
    cb.load_model(str(ORACLE02_MODEL))
    expected_features = len(cb.feature_names_)
    if expected_features != features.shape[1]:
        raise ValueError(f"ORACLE02 CatBoost expects {expected_features} features; built {features.shape[1]}")
    cb_probs = _align_proba(cb, features)
    deploy = json.loads(ORACLE04_DEPLOYMENT.read_text())
    baseline_info = deploy["baseline"]
    alpha = float(baseline_info["catboost_weight"])
    base = alpha * cb_probs + (1.0 - alpha) * equal9
    base /= base.sum(axis=1, keepdims=True)

    rule_blob = json.loads(ORACLE05_RULES.read_text())
    rule_sets = rule_blob["rules"]
    cp_prob = np.stack([matrices[sid] for sid in ids], axis=0)
    data = {"cp_ids": ids, "cp_prob": cp_prob,
        "cp_pred": cp_prob.argmax(axis=2).T,
        "families": [_family(str(by_id.loc[sid, "model"])) for sid in ids]}
    data["family_names"] = sorted(set(data["families"]))
    predictions = {}
    pred_arrays = {"A_pair": base.copy()}
    for strategy in ("B_margin", "C_checkpoint"):
        updated, fired, names = _apply_rules(base, rule_sets[strategy], data, np.arange(len(test)))
        pred_arrays[strategy] = updated
        row_out = pd.DataFrame({"id": test.id.astype(str), "model": np.asarray(LABELS)[updated.argmax(axis=1)]})
        row_out["rule_fired"] = fired
        row_out["rule"] = names
        predictions[strategy] = row_out
    pred_arrays["A_pair"] = pred_arrays["A_pair"]
    predictions["A_pair"] = pd.DataFrame({"id": test.id.astype(str),
        "model": np.asarray(LABELS)[base.argmax(axis=1)], "rule_fired": False, "rule": ""})
    for strategy in ("A_pair", "B_margin", "C_checkpoint"):
        for i, label in enumerate(LABELS):
            predictions[strategy][f"prob::{label}"] = pred_arrays[strategy][:, i]
    return predictions, {"baseline_feature_count": int(features.shape[1]), "feature_names_sha256": hashlib.sha256(
        "\n".join(feature_names).encode()).hexdigest(), "baseline_blend_catboost_weight": alpha,
        "base_model_probabilities": base, "o05_rule_sets": rule_sets}


def _submit(paths: dict[str, Path], *, competition: str = "name-that-llm") -> list[dict]:
    existing = json.loads(subprocess.check_output(
        ["kaggle", "competitions", "submissions", "-c", competition, "--page-size", "100", "--format", "json"],
        text=True))
    records = []
    for strategy, path in paths.items():
        description = DESCRIPTIONS[strategy]
        match = next((x for x in existing if x.get("description") == description), None)
        if match:
            records.append({"strategy": strategy, "description": description, "submission_id": match["ref"],
                "status": match["status"], "public_score": match.get("publicScore", ""), "reused_existing": True})
            continue
        result = subprocess.run(["kaggle", "competitions", "submit", "-c", competition,
            "-f", str(path), "-m", description], text=True, capture_output=True, check=True)
        print(result.stdout.strip(), flush=True)
        newest = json.loads(subprocess.check_output(
            ["kaggle", "competitions", "submissions", "-c", competition, "--page-size", "100", "--format", "json"],
            text=True))
        match = next((x for x in newest if x.get("description") == description), None)
        records.append({"strategy": strategy, "description": description,
            "submission_id": match.get("ref") if match else None,
            "status": match.get("status") if match else "submitted_unlisted",
            "public_score": match.get("publicScore", "") if match else "", "reused_existing": False,
            "stdout": result.stdout.strip()})
    return records


def run(*, test_path: str | Path, output_dir: str | Path, submit: bool = False, force: bool = False) -> dict:
    output = Path(output_dir)
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    test_path = Path(test_path)
    test = pd.read_csv(test_path)
    if list(test.columns) != ["id", "text"] or test.id.astype(str).duplicated().any():
        raise ValueError("competition test input must have unique IDs and columns [id, text]")
    ids, matrices, by_id = _load_or_infer(test, output, force=force)
    predictions, details = _build_predictions(test, ids, matrices, by_id)
    files = {}
    for strategy, frame in predictions.items():
        submission = frame[["id", "model"]].copy()
        if submission.id.tolist() != test.id.astype(str).tolist() or not set(submission.model).issubset(CLASSES):
            raise ValueError(f"invalid Kaggle schema/order/labels for {strategy}")
        path = output / f"submission_oracle05_{strategy.lower()}.parquet"
        submission.to_parquet(path, index=False)
        files[strategy] = path
    sample = output / "sample_submission.parquet"
    if sample.is_file():
        expected = pd.read_parquet(sample)
        for strategy, path in files.items():
            got = pd.read_parquet(path)
            if got.columns.tolist() != expected.columns.tolist() or got.id.astype(str).tolist() != expected.id.astype(str).tolist():
                raise ValueError(f"submission does not match Kaggle sample schema/order: {strategy}")
    result = {"test_rows": len(test), "checkpoint_count": len(ids), "strategies": {}, "submitted": submit}
    for strategy, frame in predictions.items():
        result["strategies"][strategy] = {"file": str(files[strategy].relative_to(ROOT)),
            "predictions": frame.model.astype(str).tolist(), "rule_triggers": int(frame.rule_fired.sum())}
    if submit:
        result["submissions"] = _submit(files)
    hashes = {strategy: _sha(path) for strategy, path in files.items()}
    manifest = {"experiment": "ORACLE05 A/B/C Kaggle submissions", "trust_list_version": TRUST_VERSION,
        "test_sha256": _sha(test_path), "checkpoint_ids": ids, "checkpoint_count": len(ids),
        "checkpoint_probability_cache_dir": str((output / "checkpoint_probabilities").relative_to(ROOT)),
        "probability_cache_hashes": {sid: _sha(output / "checkpoint_probabilities" / f"{sid}.npy") for sid in ids},
        "submission_sha256": hashes, "submission_records": result.get("submissions", []),
        "baseline_feature_count": details["baseline_feature_count"],
        "feature_names_sha256": details["feature_names_sha256"],
        "baseline_blend_catboost_weight": details["baseline_blend_catboost_weight"],
        "rule_artifact_sha256": _sha(ORACLE05_RULES), "oracle02_model_sha256": _sha(ORACLE02_MODEL),
        "gpu_used": False}
    (output / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="/tmp/oracle05_kaggle/test.csv")
    parser.add_argument("--sample-submission", default="/tmp/oracle05_kaggle/sample_submission.parquet")
    parser.add_argument("--output-dir", default="artifacts/analysis/oracle05/kaggle_submissions")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run checkpoint inference instead of reusing cache")
    args = parser.parse_args()
    sample_dest = ROOT / args.output_dir / "sample_submission.parquet"
    sample_dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(args.sample_submission).is_file(): shutil.copy2(args.sample_submission, sample_dest)
    run(test_path=args.test, output_dir=args.output_dir, submit=args.submit, force=args.force)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.set_num_threads(2)
    main()
