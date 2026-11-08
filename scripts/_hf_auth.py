"""Read an optional HF credential without ever printing it."""
from pathlib import Path

from huggingface_hub import get_token


def hf_token():
    token = get_token()
    if token:
        return token
    token_file = Path("/home/zodrak/hf.token")
    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        return token or None
    return None
