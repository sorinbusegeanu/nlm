"""Add dynamic ORACLE05 inference to the all-checkpoint Kaggle runtime package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts/kaggle/oracle04-dynamic-assets"
KERNELS = ROOT / "artifacts/kaggle/oracle05-dynamic-kernels"
RULES = ROOT / "artifacts/analysis/oracle05/final_rules_full_v0.json"
RUNTIME = ROOT / "scripts/oracle05_kaggle_runtime.py"
TAR = ASSETS / "oracle04-assets.tar"
DATASET = "zodrak/oracle04-dynamic-assets"
MODEL_SOURCES = [
    "google/gemma-4/transformers/gemma-4-e2b/1",
    "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
    "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
    "qwen-lm/qwen-3/transformers/1.7b-base/1",
]
STRATEGIES = {
    "A_pair": ("oracle05-a-pair", "ORACLE05 A Pair"),
    "B_margin": ("oracle05-b-margin", "ORACLE05 B Margin"),
    "C_checkpoint": ("oracle05-c-checkpoint", "ORACLE05 C Checkpoint"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_assets() -> dict:
    if not ASSETS.is_dir() or not TAR.is_file():
        raise FileNotFoundError(f"ORACLE04 dynamic checkpoint package is missing: {TAR}")
    if not RULES.is_file() or not RUNTIME.is_file():
        raise FileNotFoundError("ORACLE05 rules or Kaggle inference runtime is missing")
    import tarfile
    with tarfile.open(TAR, "r:") as archive:
        names = {member.name for member in archive.getmembers()}
    entries = {
        "oracle04_assets/source__oracle05_kaggle_runtime.py": RUNTIME,
        "oracle04_assets/oracle05_full_v0_rules.json": RULES,
    }
    if any(name in names for name in entries):
        raise ValueError("ORACLE05 files are already present in the shared runtime archive")
    with tarfile.open(TAR, "a:") as archive:
        for name, path in entries.items():
            archive.add(path, arcname=name, recursive=False)
    metadata_path = ASSETS / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["subtitle"] = "Dynamic test inference assets for ORACLE04 and ORACLE05"
    metadata["description"] = (
        "Frozen Trust List 2 v025 checkpoint assets and offline inference code. "
        "Includes deployed ORACLE02/ORACLE04 models and ORACLE05 top-2 switching rules. "
        "Kaggle notebooks infer on the competition runtime test set."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"asset_dataset": DATASET, "archive": str(TAR.relative_to(ROOT)),
            "archive_bytes": TAR.stat().st_size,
            "runtime_sha256": _sha(RUNTIME), "rules_sha256": _sha(RULES)}


def build_kernels() -> list[Path]:
    KERNELS.mkdir(parents=True, exist_ok=True)
    folders = []
    for strategy, (slug, title) in STRATEGIES.items():
        folder = KERNELS / strategy
        folder.mkdir(parents=True, exist_ok=True)
        code = f'''from pathlib import Path
import json, os, sys, tarfile
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
assert torch.cuda.is_available(), "Kaggle GPU is required"
manifests = list(Path("/kaggle/input").rglob("oracle04_assets.json"))
assert len(manifests) == 1, f"expected one v025 runtime asset manifest; found {{manifests}}"
ASSETS = manifests[0].parent
runtime_path = Path("/kaggle/working/oracle05_kaggle_runtime.py")
with tarfile.open(ASSETS / "oracle04-assets.tar", "r:") as archive:
    member = archive.extractfile("oracle04_assets/source__oracle04_kaggle_runtime.py")
    assert member is not None
    Path("/kaggle/working/oracle04_kaggle_runtime.py").write_bytes(member.read())
    member = archive.extractfile("oracle04_assets/source__oracle05_kaggle_runtime.py")
    assert member is not None
    runtime_path.write_bytes(member.read())
sys.path.insert(0, "/kaggle/working")
from oracle05_kaggle_runtime import run
result = run({strategy!r}, ASSETS, Path("/kaggle/working/submission.parquet"))
print(json.dumps(result, indent=2))
'''
        nb = nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(
                f"# {title}\n\nRuns the frozen ORACLE02 + ORACLE05 strategy over every row in Kaggle's runtime test set. It performs checkpoint inference at submission time and does not reuse the three public-test predictions."
            ),
            nbf.v4.new_code_cell(code),
        ])
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        nbf.write(nb, folder / "submission.ipynb")
        metadata = {
            "id": f"zodrak/{slug}", "title": title, "code_file": "submission.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": ["oracle05", "dynamic-inference", "trust-list-2-v025"],
            "dataset_sources": [DATASET, "zodrak/zod03-llama-base", "pingjiale/qwen3-0-6b"],
            "kernel_sources": [], "competition_sources": ["name-that-llm"],
            "model_sources": MODEL_SOURCES, "machine_shape": "NvidiaTeslaT4",
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        folders.append(folder)
    return folders


if __name__ == "__main__":
    print(json.dumps({"assets": package_assets(),
                      "kernels": [str(p.relative_to(ROOT)) for p in build_kernels()]}, indent=2))
