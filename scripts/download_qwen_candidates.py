"""Download the pinned Qwen17/Qwen4 checkpoints concurrently into the HF cache."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen17", "qwen4")


def download(model):
    config_path = ROOT / "configs" / "models" / f"{model}.yaml"
    config = yaml.safe_load(config_path.read_text())
    path = snapshot_download(repo_id=config["checkpoint"], revision=config["revision"])
    return model, config["checkpoint"], config["revision"], path


def main():
    failures = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {pool.submit(download, model): model for model in MODELS}
        for future in as_completed(futures):
            model = futures[future]
            try:
                name, repo, revision, path = future.result()
                print(f"{name}: {repo}@{revision} downloaded to {path}", flush=True)
            except Exception as exc:
                failures.append((model, exc))
                print(f"{model}: download failed: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} Qwen checkpoint download(s) failed")


if __name__ == "__main__":
    main()
