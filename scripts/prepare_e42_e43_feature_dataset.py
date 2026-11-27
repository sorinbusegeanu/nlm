"""Package the E38 and E39 inference assets as a shared Kaggle dataset."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/kaggle/e42-e43-feature-members-v001"
DATASET_ID = "zodrak/e38-e39-feature-members-v001"
CLASSES = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    sources = {
        "e38_svc_bundle.joblib": ROOT / "artifacts/analysis/e38/enhanced/svc_bundle.joblib",
        "e39_model.cbm": ROOT / "artifacts/analysis/e39/model/model.cbm",
        "e39_stylometry.json": ROOT / "artifacts/analysis/e39/model/stylometry.json",
        "runtime_feature_factory.py": ROOT / "src/feature_factory.py",
        "e38_e39_inference.py": ROOT / "src/e38_e39_inference.py",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    tokenizer_files = {}
    for tokenizer in ("qwen", "gpt2"):
        tokenizer_files[tokenizer] = {}
        source_dir = ROOT / "artifacts/analysis/e39/model/tokenizers" / tokenizer
        for source in sorted(source_dir.iterdir()):
            if not source.is_file():
                continue
            target_name = f"e39_{tokenizer}_{source.name}"
            shutil.copy2(source, OUT / target_name)
            tokenizer_files[tokenizer][source.name] = target_name
    (OUT / "runtime_src_init.py").write_text('"""E38/E39 feature runtime."""\n')
    assets = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        assets.append({"path": path.relative_to(OUT).as_posix(),
                       "size": path.stat().st_size, "sha256": sha(path)})
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "experiments": ["E42", "E43"],
        "components": {
            "E38": {"type": "sklearn calibrated SVC bundle",
                    "bundle": "e38_svc_bundle.joblib",
                    "training_dataset": "v2.14", "classes": CLASSES},
            "E39": {"type": "CatBoost tokenizer-fingerprint classifier",
                    "model": "e39_model.cbm", "training_dataset": "v2.14",
                    "feature_metadata": "e39_stylometry.json",
                    "tokenizer_files": tokenizer_files,
                    "classes": CLASSES},
        },
        "runtime": {
            "helper": "e38_e39_inference.py",
            "feature_factory": "runtime_feature_factory.py",
            "feature_factory_class_module": "src.feature_factory",
            "inference": "E38 fitted FoldFeatures.transform + calibrated probabilities; E39 stylometry recipe + CatBoost raw logits + canonical softmax",
        },
        "combination": {
            "E41": "0.5 * E38 probabilities + 0.5 * E39 probabilities",
            "E42_E43": "0.5 * row-normalized host score vector + 0.5 * E41 probabilities",
            "host_score_normalization": "divide each nonnegative 18-label score row by its row sum",
        },
        "classes": CLASSES,
        "sklearn_training_version": "1.9.1",
        "assets": [
            {"dataset_ref": DATASET_ID, "flat_filename": Path(x["path"]).name,
             "size": x["size"], "sha256": x["sha256"]} for x in assets
        ],
    }
    (OUT / "e38_e39_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {
        "title": "E38 E39 Probability Members for E42 E43",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "subtitle": "Enhanced SVC and tokenizer CatBoost inference bundles",
        "description": "E38/E39 model assets, tokenizer files and their exact feature runtime for the E42 and E43 Kaggle ensemble notebooks.",
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"directory": str(OUT), "dataset_id": DATASET_ID,
            "asset_count": len(assets), "total_bytes": sum(x["size"] for x in assets),
            "manifest": str(OUT / "e38_e39_manifest.json")}


if __name__ == "__main__":
    print(json.dumps(package(), indent=2))
