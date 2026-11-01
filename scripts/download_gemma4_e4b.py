"""Download Google's Gemma 4 E4B model at a pinned Hub revision."""
from huggingface_hub import snapshot_download


REPO_ID = "google/gemma-4-E4B"
REVISION = "411aa17b749aa952df1359d2dcea73917a544d9a"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION)
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
