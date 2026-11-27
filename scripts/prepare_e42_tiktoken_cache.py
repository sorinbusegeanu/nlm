"""Package the offline cl100k_base/o200k_base cache for Kaggle inference."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/kaggle/e42-e43-tiktoken-cache-v001"
DATASET_ID = "zodrak/e42-e43-tiktoken-cache-v001"
CACHE_KEYS = {
    "fb374d419588a4632f3f557e76b4b70aebbca790": "cl100k_base",
    "9b5ad71b2ce5302211f9c61530b329a4922fc6a4": "o200k_base",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    assets = []
    for cache_key, encoding in CACHE_KEYS.items():
        source = Path("/tmp/data-gym-cache") / cache_key
        if not source.is_file():
            raise FileNotFoundError(f"offline {encoding} cache is missing: {source}")
        target = OUT / cache_key
        shutil.copy2(source, target)
        assets.append({"dataset_ref": DATASET_ID, "flat_filename": cache_key,
                       "encoding": encoding, "size": target.stat().st_size,
                       "sha256": sha(target)})
    (OUT / "tiktoken_cache_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "dataset_id": DATASET_ID, "assets": assets}, indent=2) + "\n")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "Offline tiktoken Encoding Cache for E42 E43",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "subtitle": "cl100k_base and o200k_base offline rank files",
        "description": "Exact tiktoken cache data files used by E39 tokenizer fingerprints in E42/E43.",
    }, indent=2) + "\n")
    return {"directory": str(OUT), "dataset_id": DATASET_ID,
            "files": [{"filename": item["flat_filename"], "bytes": item["size"],
                       "encoding": item["encoding"]} for item in assets]}


if __name__ == "__main__":
    print(json.dumps(package(), indent=2))
