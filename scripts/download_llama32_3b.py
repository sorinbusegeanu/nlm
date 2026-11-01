"""Download the pinned, gated Llama 3.2 3B base model after HF access is granted."""
from pathlib import Path

import yaml
from huggingface_hub import get_token, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/models/llama32_3b.yaml"


def main():
    config = yaml.safe_load(CONFIG.read_text())
    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token is configured. meta-llama/Llama-3.2-3B is gated: "
            "accept Meta's model terms on Hugging Face, run `hf auth login`, then rerun "
            "`python scripts/download_llama32_3b.py`."
        )
    path = snapshot_download(repo_id=config["checkpoint"], revision=config["revision"], token=token)
    print(f"{config['checkpoint']}@{config['revision']} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
