"""Create tiny Kaggle kernels that promote saved strategy parquet outputs to submission.parquet."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/kaggle/three-strategy-submit-kernels"
SOURCE_KERNEL = "zodrak/trust-v025-c3-equal-selected-c1"
STRATEGIES = {
    "equal-selected": ("equal-selected-trust-list-2-v025", "Equal Selected Trust List 2 v025", "submission_equal_selected.parquet"),
    "c1": ("c1-precision-trust-list-2-v025", "C1 Precision Trust List 2 v025", "submission_c1.parquet"),
}


def build() -> list[Path]:
    import nbformat as nbf
    paths = []
    for key, (slug, title, source_file) in STRATEGIES.items():
        folder = BASE / key
        folder.mkdir(parents=True, exist_ok=True)
        notebook = nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(f"# {title}\n\nExpose the checked inference output from `{SOURCE_KERNEL}` as this kernel's competition submission file."),
            nbf.v4.new_code_cell(f'''from pathlib import Path
import json
import pandas as pd
INPUT=Path('/kaggle/input')
found=list(INPUT.rglob({source_file!r}))
assert len(found)==1, f"expected one {source_file}; found {{found}}"
frame=pd.read_parquet(found[0])
assert frame.columns.tolist()==['id','model'] and len(frame)==3
assert frame.id.notna().all() and frame.id.is_unique and frame.model.notna().all()
frame.to_parquet('/kaggle/working/submission.parquet',index=False)
check=pd.read_parquet('/kaggle/working/submission.parquet')
assert check.equals(frame)
print(json.dumps({{'strategy':{key!r},'rows':len(frame),'ids':frame.id.astype(str).tolist(),'predictions':frame.model.astype(str).tolist()}},indent=2))
''')])
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        notebook_path = folder / f"{key.replace('-', '_')}_submission.ipynb"
        nbf.write(notebook, notebook_path)
        metadata = {
            "id": f"zodrak/{slug}", "title": title, "code_file": notebook_path.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": ["trust-list-2-v025", "submission"], "dataset_sources": [],
            "kernel_sources": [SOURCE_KERNEL], "competition_sources": ["name-that-llm"],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        paths.append(folder)
    return paths


if __name__ == "__main__":
    print("\n".join(map(str, build())))
