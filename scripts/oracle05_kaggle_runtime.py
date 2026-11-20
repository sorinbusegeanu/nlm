"""Hidden-test inference runtime for the three ORACLE05 strategies."""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oracle04_kaggle_runtime import AssetStore, _base_paths, _materialize_checkpoint


def run(strategy: str, assets: Path, output_path: Path) -> dict:
    if strategy not in {"A_pair", "B_margin", "C_checkpoint"}:
        raise ValueError(f"unknown ORACLE05 strategy: {strategy}")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required for ORACLE05 inference")

    assets = Path(assets)
    input_root = Path("/kaggle/input")
    bundle = json.loads((assets / "oracle04_assets.json").read_text())
    ids = list(bundle["checkpoint_ids"])
    classes = list(bundle["classes"])
    if bundle.get("trust_list_version") != "trust-list-2-v025" or len(ids) != 78:
        raise ValueError("ORACLE05 requires the frozen 78-checkpoint Trust List 2 v025 roster")

    work = Path("/kaggle/temp/oracle05_runtime")
    shutil.rmtree(work, ignore_errors=True)
    src, deploy = work / "src", work / "deployment"
    src.mkdir(parents=True, exist_ok=True)
    deploy.mkdir(parents=True, exist_ok=True)
    store = AssetStore(assets, bundle)
    started = time.monotonic()
    try:
        for name in store.names():
            if name.startswith("source__") and name.endswith(".py"):
                store.extract(name, src / name.removeprefix("source__"))
            elif name.startswith("deployment__"):
                store.extract(name, deploy / name.removeprefix("deployment__"))
        store.extract("oracle05_full_v0_rules.json", work / "final_rules_full_v0.json")

        bases = _base_paths(input_root, store, bundle, work)
        sys.path.insert(0, str(work))
        try:
            from importlib.metadata import PackageNotFoundError, version
            from packaging.version import Version
            if Version(version("torchao")) <= Version("0.16.0"):
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                               check=True, capture_output=True)
        except (PackageNotFoundError, ImportError):
            pass
        try:
            import peft.tuners.lora.torchao as peft_torchao
            peft_torchao.is_torchao_available = lambda: False
        except ImportError:
            pass

        from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
        from src.oracle02 import _align_proba, _build_features
        from src.oracle05 import _apply_rules, _family
        from src.v2_checkpoint_trust import LABELS
        from src.ensemble_precision_recall_overlap import EQUAL9
        import joblib

        if classes != list(LABELS):
            raise ValueError("checkpoint bundle class order differs from canonical label order")
        tests = [input_root / "competitions/name-that-llm/test.csv",
                 input_root / "name-that-llm/test.csv"]
        test_path = next((p for p in tests if p.is_file()), None)
        if test_path is None:
            found = list(input_root.rglob("test.csv"))
            if len(found) != 1:
                raise FileNotFoundError(f"cannot uniquely find runtime test.csv: {found}")
            test_path = found[0]
        test = pd.read_csv(test_path, dtype={"id": "string"})
        if not {"id", "text"}.issubset(test.columns):
            raise ValueError(f"unexpected competition test schema: {test.columns.tolist()}")
        if test.id.isna().any() or test.id.duplicated().any():
            raise ValueError("runtime test IDs must be unique and non-null")
        test["text"] = test.text.fillna("").astype(str)
        texts = test.text.tolist()
        matrices: dict[str, np.ndarray] = {}

        def infer_one(sid: str) -> np.ndarray:
            candidate = bundle["candidates"][sid]
            model = candidate["model"]
            folder = _materialize_checkpoint(store, candidate, sid, work)
            checkpoint = Path(candidate["runtime_checkpoint"])
            temperature = float(candidate.get("temperature", 1.0))
            try:
                if model == "svc":
                    probabilities, returned = predict_svc_bundle(joblib.load(checkpoint), texts)
                elif model == "catboost_stylometry":
                    probabilities, returned = predict_catboost_stylometry(
                        checkpoint, texts, classes, temperature=temperature)
                else:
                    base = bases.get(model)
                    if model != "ultrabert380" and base is None:
                        raise FileNotFoundError(f"base model is not resolved for {model}")
                    batch = min(4, max(1, int(candidate["config"].get("batch_size", 4) or 4)))
                    while True:
                        try:
                            probabilities, returned = _predict_neural(
                                model, checkpoint, texts, classes, batch_size=batch,
                                max_length=int(candidate["config"].get("max_length") or 384),
                                temperature=temperature,
                                base_checkpoint=candidate.get("base_model") or None,
                                revision=candidate.get("base_revision") or None,
                                local_base_model=base,
                                fix_mistral_regex=bool(candidate["config"].get("fix_mistral_regex", False)),
                                text_only_backbone=bool(candidate["config"].get("text_only_backbone", False)),
                                quantization=str(candidate["config"].get("quantization") or "none"),
                                attention_backend=candidate["config"].get("attention_backend"))
                            break
                        except RuntimeError as exc:
                            if "out of memory" not in str(exc).lower() or batch <= 1:
                                raise
                            batch = max(1, batch // 2)
                            gc.collect()
                            torch.cuda.empty_cache()
                col = {str(label): i for i, label in enumerate(returned)}
                out = np.asarray(probabilities, dtype=np.float64)[:, [col[label] for label in classes]]
                if out.shape != (len(test), len(classes)) or not np.isfinite(out).all():
                    raise ValueError(f"invalid probabilities from {sid}: {out.shape}")
                if np.any(out < 0) or not np.allclose(out.sum(axis=1), 1.0, atol=1e-5):
                    raise ValueError(f"unnormalized probabilities from {sid}")
                del probabilities
                gc.collect()
                torch.cuda.empty_cache()
                return out
            finally:
                shutil.rmtree(folder, ignore_errors=True)

        for number, sid in enumerate(ids, 1):
            matrices[sid] = infer_one(sid)
            print({"checkpoint": sid, "model": bundle["candidates"][sid]["model"],
                   "progress": f"{number}/{len(ids)}", "runtime_test_rows": len(test)}, flush=True)

        equal9 = np.mean(np.stack([matrices[sid] for sid in EQUAL9]), axis=0)
        equal9 /= equal9.sum(axis=1, keepdims=True)
        features, feature_names = _build_features(ids, matrices, equal9)
        cb = __import__("catboost").CatBoostClassifier()
        cb.load_model(str(deploy / "oracle02_catboost.cbm"))
        if len(feature_names) != len(cb.feature_names_):
            raise ValueError("ORACLE02 stacked feature count does not match the deployed CatBoost model")
        baseline_meta = json.loads((deploy / "deployment_manifest.json").read_text())["baseline"]
        alpha = float(baseline_meta["catboost_weight"])
        catboost_prob = _align_proba(cb, features)
        baseline_prob = alpha * catboost_prob + (1.0 - alpha) * equal9
        baseline_prob /= baseline_prob.sum(axis=1, keepdims=True)

        if strategy == "A_pair":
            output_prob = baseline_prob
            fired = np.zeros(len(test), dtype=bool)
            rule_names = [""] * len(test)
        else:
            cp_prob = np.stack([matrices[sid] for sid in ids], axis=0)
            families = [_family(bundle["candidates"][sid]["model"]) for sid in ids]
            data = {"cp_ids": ids, "cp_prob": cp_prob,
                    "cp_pred": cp_prob.argmax(axis=2).T,
                    "families": families, "family_names": sorted(set(families))}
            rules_blob = json.loads((work / "final_rules_full_v0.json").read_text())
            rules = rules_blob["rules"][strategy]
            output_prob, fired, rule_names = _apply_rules(
                baseline_prob, rules, data, np.arange(len(test)))

        pred = np.asarray(classes, dtype=object)[output_prob.argmax(axis=1)]
        submission = pd.DataFrame({"id": test.id.astype(str), "model": pred.astype(str)})
        if submission.columns.tolist() != ["id", "model"]:
            raise ValueError("submission columns must be exactly id,model")
        if submission.id.tolist() != test.id.astype(str).tolist():
            raise ValueError("submission lost runtime test ordering")
        if submission.id.isna().any() or submission.id.duplicated().any() or submission.model.isna().any():
            raise ValueError("submission contains null or duplicate values")
        if not submission.model.isin(classes).all():
            raise ValueError("submission contains noncanonical model labels")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_parquet(output_path, index=False)
        summary = {"strategy": strategy, "rows": len(submission),
                   "rule_triggers": int(np.sum(fired)),
                   "rule_counts": {name: int(sum(x == name for x in rule_names)) for name in set(rule_names) if name},
                   "submission": str(output_path),
                   "runtime_seconds": round(time.monotonic() - started, 2)}
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        store.close()
