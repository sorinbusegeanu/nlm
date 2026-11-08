"""Download Ministral 3 3B Base at a pinned Hugging Face revision."""
from huggingface_hub import snapshot_download

from _hf_auth import hf_token


REPO_ID = "mistralai/Ministral-3-3B-Base-2512"
REVISION = "6f9c4b12a95b139af68670a6713616b757923735"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, token=hf_token())
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
