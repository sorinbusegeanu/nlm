"""Download Google's smallest Gemma 4 model at a pinned Hub revision."""
from huggingface_hub import snapshot_download

from _hf_auth import hf_token


REPO_ID = "google/gemma-4-E2B"
REVISION = "433813a669639fd7eaa83155828e9000ed357d53"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, token=hf_token())
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
