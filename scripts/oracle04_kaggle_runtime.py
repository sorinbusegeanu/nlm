"""Kaggle-side ORACLE04 runtime inference for dynamically supplied test rows."""
from __future__ import annotations

import gc
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path


class AssetStore:
    """Read assets whether Kaggle exposes the TAR or expands it on upload."""
    prefix = "oracle04_assets/"

    def __init__(self, root: Path, bundle: dict):
        self.tree = root / "oracle04_assets"
        self.archive = None
        archive_path = root / bundle.get("archive", "oracle04-assets.tar")
        if not self.tree.is_dir():
            if not archive_path.is_file():
                raise FileNotFoundError("ORACLE04 archive or extracted asset directory is missing")
            self.archive = tarfile.open(archive_path, "r:")

    def names(self) -> list[str]:
        if self.tree.is_dir():
            return [p.name for p in self.tree.iterdir() if p.is_file()]
        return [name.removeprefix(self.prefix) for name in self.archive.getnames()
                if name.startswith(self.prefix) and name.count("/") == 1]

    def extract(self, name: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.tree.is_dir():
            source_path = self.tree / name
            if not source_path.is_file():
                raise FileNotFoundError(f"missing expanded ORACLE04 asset: {source_path}")
            source = source_path.open("rb")
        else:
            member = self.archive.getmember(self.prefix + name)
            if not member.isfile():
                raise ValueError(f"not a regular asset file: {name}")
            source = self.archive.extractfile(member)
            if source is None:
                raise FileNotFoundError(f"asset payload missing: {name}")
        with source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


def _materialize_checkpoint(store: AssetStore, candidate: dict,
                            sid: str, work: Path) -> Path:
    folder = work / "checkpoints" / sid
    folder.mkdir(parents=True, exist_ok=True)
    for item in candidate["files"]:
        store.extract(item["flat_filename"], folder / item["relative_path"])
    checkpoint = (folder if candidate["checkpoint_type"] == "directory"
                  else folder / candidate["files"][0]["relative_path"])
    candidate["runtime_checkpoint"] = str(checkpoint)
    return folder


def _find_model_input(input_root: Path, markers: list[str]) -> Path:
    for config in input_root.rglob("config.json"):
        if any(marker.lower() in str(config.parent).lower() for marker in markers):
            return config.parent
    raise FileNotFoundError(f"base model not found under Kaggle inputs; searched {markers}")


def _base_paths(input_root: Path, store: AssetStore, bundle: dict,
                work: Path) -> dict[str, Path]:
    bases: dict[str, Path] = {}
    for model, info in bundle.get("bundled_bases", {}).items():
        dest = work / "bases" / model
        for item in info["files"]:
            target = dest / item["relative_path"]
            if not target.exists():
                store.extract(item["flat_filename"], target)
        bases[model] = dest

    models = {candidate["model"] for candidate in bundle["candidates"].values()}
    external = {
        "qwen06": ["qwen3-0.6b"], "qwen17": ["1.7b-base", "qwen3-1.7b"],
        "gemma4_e2b": ["gemma-4-e2b"], "embeddinggemma2": ["embeddinggemma-2"],
        "ministral3_3b": ["ministral-3-3b-base-2512"],
    }
    for model, markers in external.items():
        if model in models:
            bases[model] = _find_model_input(input_root, markers)

    for manifest_name, model in (("llama_base_manifest.json", "llama32_3b"),
                                 ("zod03_manifest.json", "granite4_h_micro")):
        if model not in models or model in bases:
            continue
        manifests = list(input_root.rglob(manifest_name))
        if not manifests:
            if model == "granite4_h_micro":
                bases[model] = _find_model_input(input_root, ["granite-4.0-h-micro-base"])
                continue
            raise FileNotFoundError(f"required base manifest {manifest_name} missing")
        manifest = manifests[0]
        spec = json.loads(manifest.read_text())
        if manifest_name == "llama_base_manifest.json":
            files = spec["files"]
        else:
            row = next((r for r in spec["candidates"].values()
                        if r.get("model") == model and r.get("base_files")), None)
            if row is None:
                raise FileNotFoundError("Granite H-Micro base files absent from zod03 bundle")
            files = row["base_files"]
        dest = work / "bases" / model
        for item in files:
            source = manifest.parent / item["flat_filename"]
            target = dest / item["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source)
        bases[model] = dest

    # The Kaggle SmolLM3 source uses an older custom config name. Match the
    # compatibility repair used by the existing ZOD submission notebooks.
    smol = bases.get("smollm3_3b")
    if smol is not None:
        config_path = smol / "config.json"
        config = json.loads(config_path.read_text())
        if config.get("model_type") == "smollm3_text_base":
            config["model_type"] = "smollm3"
            config_path.write_text(json.dumps(config))
        if not (smol / "model.safetensors.index.json").exists():
            from safetensors import safe_open
            weight_map = {}
            for shard in sorted(smol.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt", device="cpu") as sf:
                    weight_map.update({key: shard.name for key in sf.keys()})
            if weight_map:
                (smol / "model.safetensors.index.json").write_text(
                    json.dumps({"metadata": {}, "weight_map": weight_map}))
    return bases


def run(strategy: str, assets: Path, output_path: Path) -> dict:
    import hashlib
    import numpy as np
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required for ORACLE04 inference")
    assets = Path(assets)
    input_root = Path("/kaggle/input")
    bundle = json.loads((assets / "oracle04_assets.json").read_text())
    if bundle.get("trust_list_version") != "trust-list-2-v025" or len(bundle.get("checkpoint_ids", [])) != 78:
        raise ValueError("runtime bundle is not the frozen ORACLE04 v025 roster")
    archive_path = assets / bundle.get("archive", "oracle04-assets.tar")
    if not archive_path.is_file() and not (assets / "oracle04_assets").is_dir():
        raise FileNotFoundError(f"missing ORACLE04 asset archive: {archive_path}")
    # Keep unpacked model bases/checkpoints in Kaggle's ephemeral scratch area;
    # only the finished submission should be retained in /kaggle/working.
    work = Path("/kaggle/temp/oracle04_runtime")
    shutil.rmtree(work, ignore_errors=True)
    src = work / "src"
    deploy = work / "deployment"
    src.mkdir(parents=True, exist_ok=True)
    deploy.mkdir(parents=True, exist_ok=True)
    store = AssetStore(assets, bundle)
    try:
        # Materialize only lightweight code and deployment models up front.
        for name in store.names():
            if name.startswith("source__") and name.endswith(".py"):
                store.extract(name, src / name.removeprefix("source__"))
            elif name.startswith("deployment__"):
                store.extract(name, deploy / name.removeprefix("deployment__"))
        bases = _base_paths(input_root, store, bundle, work)
        sys.path.insert(0, str(work))

        # Kaggle's preinstalled TorchAO may be incompatible with PEFT's optional
        # integration and fail during adapter loading. Existing NLM Kaggle
        # runtimes remove the known incompatible version before inference.
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
        from src.oracle04_deploy import predict_strategies
        from src.v2_checkpoint_trust import LABELS
        from src.ensemble_precision_recall_overlap import EQUAL9
        import joblib

        test_path = next((p for p in (
            input_root / "competitions/name-that-llm/test.csv",
            input_root / "name-that-llm/test.csv") if p.is_file()), None)
        if test_path is None:
            candidates = list(input_root.rglob("test.csv"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"cannot uniquely resolve competition test.csv: {candidates}")
            test_path = candidates[0]
        test = pd.read_csv(test_path, dtype={"id": "string"})
        if not {"id", "text"}.issubset(test.columns):
            raise ValueError(f"unexpected test schema: {test.columns.tolist()}")
        if test.id.isna().any() or test.id.duplicated().any():
            raise ValueError("competition test IDs must be unique and non-null")
        test["text"] = test.text.fillna("").astype(str)
        classes = list(bundle["classes"])
        ids, texts = list(bundle["checkpoint_ids"]), test.text.tolist()
        if classes != list(LABELS):
            raise ValueError("canonical class order differs from ORACLE04 deployment")
        matrices: dict[str, np.ndarray] = {}
        started = time.monotonic()

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
                        raise FileNotFoundError(f"base model not resolved for {model}")
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
                index = {str(label): i for i, label in enumerate(returned)}
                p = np.asarray(probabilities, dtype=np.float64)[:, [index[label] for label in classes]]
                del probabilities
                gc.collect()
                torch.cuda.empty_cache()
                if p.shape != (len(test), len(classes)) or not np.isfinite(p).all():
                    raise ValueError(f"invalid probability output for {sid}: {p.shape}")
                if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
                    raise ValueError(f"unnormalized probabilities for {sid}")
                return p
            finally:
                shutil.rmtree(folder, ignore_errors=True)

        for number, sid in enumerate(ids, start=1):
            matrices[sid] = infer_one(sid)
            print({"checkpoint": sid, "model": bundle["candidates"][sid]["model"],
                   "progress": f"{number}/78", "test_rows": len(test)}, flush=True)

        equal9 = np.mean(np.stack([matrices[sid] for sid in EQUAL9]), axis=0)
        equal9 /= equal9.sum(axis=1, keepdims=True)
        strategies = predict_strategies(deploy, ids, matrices, equal9)
        if strategy not in strategies:
            raise KeyError(f"unknown ORACLE04 strategy: {strategy}")
        output = np.asarray(strategies[strategy], dtype=np.float64)
        if output.shape != (len(test), len(classes)) or not np.isfinite(output).all():
            raise ValueError(f"invalid ORACLE04 probabilities for {strategy}")
        submission = pd.DataFrame({"id": test.id.astype(str),
                                   "model": np.asarray(classes)[output.argmax(axis=1)]})
        if submission.columns.tolist() != ["id", "model"]:
            raise ValueError("submission columns must be exactly [id, model]")
        if submission.id.tolist() != test.id.astype(str).tolist():
            raise ValueError("submission rows must preserve runtime test order")
        if submission.id.isna().any() or submission.id.duplicated().any() or submission.model.isna().any():
            raise ValueError("submission contains nulls or duplicate IDs")
        if not submission.model.isin(classes).all():
            raise ValueError("submission contains labels outside the competition label set")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_parquet(output_path, index=False)
        result = {"strategy": strategy, "rows": len(submission), "submission": str(output_path),
                  "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                  "runtime_seconds": round(time.monotonic() - started, 2)}
        print(json.dumps(result), flush=True)
        return result
    finally:
        store.close()
