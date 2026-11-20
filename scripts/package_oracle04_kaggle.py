"""Build validated Oracle04 Kaggle submissions from the frozen 78-model test cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "artifacts/deployment/oracle04"
ORACLE05_CACHE = ROOT / "artifacts/analysis/oracle05/kaggle_submissions"
TRUST = ROOT / "docs/trust-list-2-v025.csv"
TEST = ROOT / "data/raw/test.csv"
OUT = ROOT / "artifacts/kaggle/oracle04-submissions-dataset"
KERNELS = ROOT / "artifacts/kaggle/oracle04-submission-kernels"
DATASET_ID = "zodrak/oracle04-strategy-submissions"
DATASET_SLUG = DATASET_ID.split("/", 1)[1]
STRATEGIES = {
    "combined_logistic_core": (
        "oracle04_combined_logistic_core",
        "ORACLE04 Combined Logistic Core",
        "ORACLE04 combined logistic core; Trust List 2 v025",
    ),
    "global_logistic_core": (
        "oracle04_global_logistic_core",
        "ORACLE04 Global Logistic Core",
        "ORACLE04 global logistic core; Trust List 2 v025",
    ),
    "global_catboost_core": (
        "oracle04_global_catboost_core",
        "ORACLE04 Global CatBoost Core",
        "ORACLE04 global CatBoost core; Trust List 2 v025",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    from src.ensemble_precision_recall_overlap import EQUAL9
    from src.oracle04_deploy import predict_strategies
    from src.v2_checkpoint_trust import LABELS

    output_manifest = json.loads((ORACLE05_CACHE / "submission_manifest.json").read_text())
    deploy_manifest = json.loads((DEPLOY / "deployment_manifest.json").read_text())
    oracle04_metrics = json.loads((ROOT / "artifacts/analysis/oracle04/metrics.json").read_text())
    trust = pd.read_csv(TRUST, dtype=str, keep_default_na=False)
    test = pd.read_csv(TEST)
    if list(test.columns) != ["id", "text"] or test.id.astype(str).duplicated().any():
        raise ValueError("Expected unique competition test IDs in columns [id, text]")
    ids = list(deploy_manifest["checkpoint_ids"])
    if len(ids) != 78 or len(set(ids)) != 78:
        raise ValueError(f"ORACLE04 expects 78 ordered checkpoints, got {len(ids)}")
    if ids != output_manifest.get("checkpoint_ids"):
        raise ValueError("ORACLE05 test cache checkpoint order differs from ORACLE04 manifest")
    if output_manifest.get("trust_list_version") != "trust-list-2-v025":
        raise ValueError("Test probability cache is not from Trust List 2 v025")
    if output_manifest.get("test_sha256") != sha256(TEST):
        raise ValueError("Test CSV hash differs from the cached prediction manifest")

    by_id = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    if not set(ids).issubset(by_id.index):
        raise ValueError("ORACLE04 checkpoint roster is not contained in frozen Trust List 2 v025")
    probability_dir = ORACLE05_CACHE / "checkpoint_probabilities"
    matrices: dict[str, np.ndarray] = {}
    dataframe_hash = hashlib.sha256(
        pd.util.hash_pandas_object(test, index=False).values.tobytes()
    ).hexdigest()
    for sid in ids:
        arr_path = probability_dir / f"{sid}.npy"
        meta_path = probability_dir / f"{sid}.json"
        if not arr_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing test probabilities for {sid}")
        meta = json.loads(meta_path.read_text())
        if meta.get("sequence_id") != sid or meta.get("experiment_id") != by_id.loc[sid, "experiment_id"]:
            raise ValueError(f"Checkpoint provenance mismatch for cached predictions {sid}")
        if meta.get("trust_list_version") != "trust-list-2-v025" or meta.get("test_sha256") != dataframe_hash:
            raise ValueError(f"Test or trust-list identity mismatch for cached predictions {sid}")
        if meta.get("probability_sha256") != sha256(arr_path):
            raise ValueError(f"Probability cache hash mismatch for {sid}")
        p = np.load(arr_path, allow_pickle=False).astype(np.float64)
        if p.shape != (len(test), len(LABELS)) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError(f"Invalid probability matrix for {sid}: {p.shape}")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Unnormalized probability matrix for {sid}")
        matrices[sid] = p

    equal9 = np.mean(np.stack([matrices[sid] for sid in EQUAL9]), axis=0)
    equal9 /= equal9.sum(axis=1, keepdims=True)
    result = predict_strategies(DEPLOY, ids, matrices, equal9)
    expected = pd.read_parquet(ORACLE05_CACHE / "sample_submission.parquet")
    if expected.id.astype(str).tolist() != test.id.astype(str).tolist():
        raise ValueError("Local test IDs/order do not match the Kaggle sample submission")

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    probability_outputs = {}
    for strategy, (slug, title, description) in STRATEGIES.items():
        if strategy not in result:
            raise ValueError(f"Deployment did not return {strategy}")
        probabilities = np.asarray(result[strategy], dtype=np.float64)
        if probabilities.shape != (len(test), len(LABELS)) or not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid Oracle04 probabilities for {strategy}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Oracle04 probabilities are not normalized for {strategy}")
        frame = pd.DataFrame({"id": test.id.astype(str), "model": np.asarray(LABELS)[probabilities.argmax(1)]})
        if frame.columns.tolist() != ["id", "model"] or frame.id.tolist() != expected.id.astype(str).tolist():
            raise ValueError(f"Submission schema or row order is invalid for {strategy}")
        if frame.id.isna().any() or frame.id.duplicated().any() or frame.model.isna().any():
            raise ValueError(f"Null or duplicate submission fields for {strategy}")
        if not set(frame.model).issubset(set(LABELS)):
            raise ValueError(f"Invalid label in {strategy}")
        file = OUT / f"submission_{slug}.parquet"
        frame.to_parquet(file, index=False)
        files[strategy] = file
        probability_outputs[strategy] = probabilities.astype(np.float32)

    np.savez_compressed(OUT / "oracle04_test_probabilities.npz", **{
        strategy: values for strategy, values in probability_outputs.items()
    })
    dataset_meta = {
        "title": "ORACLE04 Three Strategy Submissions",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "subtitle": "Validated Name That LLM outputs for ORACLE04 decision strategies",
        "description": "Three id,model Parquet outputs from ORACLE04 combined logistic, global logistic, and global CatBoost strategies. Test probabilities were computed from the frozen Trust List 2 v025 checkpoint cache; no labels were used.",
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2) + "\n")

    kernel_info = {}
    for strategy, file in files.items():
        slug, title, description = STRATEGIES[strategy]
        folder = KERNELS / strategy
        folder.mkdir(parents=True, exist_ok=True)
        filename = file.name
        notebook = nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(
                f"# {title}\n\nThis kernel exposes the locally validated ORACLE04 result for `{strategy}` as the competition-required `submission.parquet`."
            ),
            nbf.v4.new_code_cell(f'''from pathlib import Path
import hashlib, json
import pandas as pd

INPUT = Path("/kaggle/input")
test_files = list(INPUT.rglob("test.csv"))
assert len(test_files) == 1, f"expected one competition test.csv, found {{test_files}}"
test = pd.read_csv(test_files[0])
assert test.columns.tolist() == ["id", "text"] and test.id.notna().all() and test.id.is_unique
source_files = list(INPUT.rglob({filename!r}))
assert len(source_files) == 1, f"expected one prepared strategy file, found {{source_files}}"
frame = pd.read_parquet(source_files[0])
assert frame.columns.tolist() == ["id", "model"]
assert frame.id.astype(str).tolist() == test.id.astype(str).tolist(), "submission must match all test IDs in input order"
assert frame.id.notna().all() and frame.id.is_unique and frame.model.notna().all()
valid_labels = {list(LABELS)!r}
assert set(frame.model.astype(str)).issubset(set(valid_labels))
out = Path("/kaggle/working/submission.parquet")
frame.to_parquet(out, index=False)
check = pd.read_parquet(out)
assert check.equals(frame) and check.columns.tolist() == ["id", "model"]
digest = hashlib.sha256(out.read_bytes()).hexdigest()
print(json.dumps({{"strategy": {strategy!r}, "rows": len(check), "ids": check.id.astype(str).tolist(),
                  "predictions": check.model.astype(str).tolist(), "submission_sha256": digest}}, indent=2))
''')
        ])
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        notebook_path = folder / "submission.ipynb"
        nbf.write(notebook, notebook_path)
        metadata = {
            "id": f"zodrak/{slug}",
            "title": title,
            "code_file": notebook_path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["oracle04", "trust-list-2-v025", "submission"],
            "dataset_sources": [DATASET_ID],
            "kernel_sources": [],
            "competition_sources": ["name-that-llm"],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        kernel_info[strategy] = {
            "kernel": f"zodrak/{slug}",
            "title": title,
            "description": description,
            "dataset_file": filename,
            "local_submission_sha256": sha256(file),
            "local_predictions": pd.read_parquet(file).model.astype(str).tolist(),
        }

    manifest = {
        "experiment": "ORACLE04 three Kaggle strategies",
        "trust_list_version": "trust-list-2-v025",
        "test_csv_sha256": sha256(TEST),
        "test_rows": len(test),
        "test_ids": test.id.astype(str).tolist(),
        "checkpoint_count": len(ids),
        "checkpoint_ids": ids,
        "checkpoint_cache_source": "artifacts/analysis/oracle05/kaggle_submissions/checkpoint_probabilities",
        "checkpoint_probability_hashes": {
            sid: sha256(probability_dir / f"{sid}.npy") for sid in ids
        },
        "oracle04_deployment_manifest_sha256": sha256(DEPLOY / "deployment_manifest.json"),
        "oracle04_strategy_outputs": {
            strategy: {
                "file": file.name,
                "sha256": sha256(file),
                "predictions": pd.read_parquet(file).model.astype(str).tolist(),
            } for strategy, file in files.items()
        },
        "kernels": kernel_info,
        "v0_balanced_accuracy_crossfit": {
            name: float(oracle04_metrics["baseline_metrics"][name]["balanced_accuracy"])
            for name in STRATEGIES
        },
        "limitations": [
            "V0 was used in checkpoint, roster, and meta-model selection; development scores are not independent Kaggle estimates.",
            "The competition test set currently contains three rows; leaderboard scores may be pending or unreported.",
            "Kaggle kernels package the validated predictions after checking all competition test IDs and their order.",
        ],
    }
    (OUT / "oracle04_submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
