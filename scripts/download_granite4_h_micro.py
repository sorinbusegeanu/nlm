"""Download the pinned Granite 4.0 H Micro Base checkpoint for offline training."""
from huggingface_hub import snapshot_download

from _hf_auth import hf_token


REPO_ID = "ibm-granite/granite-4.0-h-micro-base"
REVISION = "372ede0bc484284e08ce574e773c96cd2b89b367"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, token=hf_token())
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
