"""Download the pinned Microsoft Phi-3.5 Mini checkpoint for offline training."""
from huggingface_hub import snapshot_download

from _hf_auth import hf_token

REPO_ID = "microsoft/Phi-3.5-mini-instruct"
REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, token=hf_token())
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
