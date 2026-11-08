"""Download the pinned SmolLM3 3B Base checkpoint for offline training."""
from huggingface_hub import snapshot_download

from _hf_auth import hf_token


REPO_ID = "HuggingFaceTB/SmolLM3-3B-Base"
REVISION = "d78a42f79198603e614095753484a04c10c2b940"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, token=hf_token())
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
