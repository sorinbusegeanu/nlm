"""Download Google's EmbeddingGemma 2 model at a pinned Hub revision."""
from huggingface_hub import snapshot_download


REPO_ID = "google/embeddinggemma-2"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"


def main():
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION)
    print(f"{REPO_ID}@{REVISION} downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
