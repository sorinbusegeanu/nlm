"""Download the pinned Granite 4.0 350M Base checkpoint for offline training."""
from huggingface_hub import snapshot_download

from _hf_auth import hf_token


REPO_ID = "ibm-granite/granite-4.0-350m-base"
REVISION = "a50b46cef21c8a86b15f0496cb794487a78a910b"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, token=hf_token())
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
