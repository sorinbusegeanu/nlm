#!/usr/bin/env python3
"""Package a completed Kaggle-only UltraBERT V0 fine-tune for offline Kaggle inference."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_package(manifest_path: Path, output_root: Path, *, model_slug: str, kernel_slug: str,
                  model_version: int = 1):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scope") != "kaggle_only_not_in_trust_list_2":
        raise ValueError("refusing to package a run outside the Kaggle-only scope")
    checkpoint = Path(manifest["best_checkpoint_path"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    package_dir = output_root / f"{model_slug}-model"
    kernel_dir = output_root / f"{kernel_slug}-kernel"
    for path in (package_dir, kernel_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    excluded = {"training_log.json", "training_progress.json", "batch_size_probe.json",
                "baseline_validation.json", "checkpoint_selection.json", "training_state",
                "epoch-1", "epoch-2", "epoch-3", "epoch-4", "epoch-5", "epoch-6",
                "epoch-7", "epoch-8", "epoch-9", "epoch-10"}
    for source in checkpoint.iterdir():
        if source.name in excluded or source.name.startswith("checkpoint-"):
            continue
        destination = package_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)
    selection = {
        "experiment_id": manifest["experiment_id"],
        "source_experiment": manifest["source_experiment"],
        "epoch": manifest["selected_epoch"],
        "accuracy": manifest["validation_metrics"]["accuracy"] if "accuracy" in manifest["validation_metrics"] else None,
        "balanced_accuracy": manifest["validation_metrics"]["balanced_accuracy"],
        "max_length": manifest["config"]["max_length"],
        "classes": manifest["classes"],
        "validation_rows": manifest["validation_rows"],
        "scope": manifest["scope"],
    }
    (package_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    metadata = {
        "ownerSlug": "zodrak", "modelSlug": "ultrabert380-v2",
        "instanceSlug": model_slug, "framework": "PyTorch",
        "overview": ("Kaggle-only UltraBERT checkpoint initialized from full V2.14 and fine-tuned "
                     "on the prompt-grouped V0 training partition. This run is excluded from Trust List 2."),
        "usage": ("Offline Transformers sequence classifier. Load from the model variation root with "
                  "AutoTokenizer and AutoModelForSequenceClassification, local_files_only=True and "
                  "trust_remote_code=True. Tokenizer max length is recorded in selection.json."),
        "licenseName": "Apache 2.0", "fineTunable": True,
        "trainingData": ["Name That LLM official V0 training partition (2,159 rows)"],
        "modelInstanceType": "ExternalVariant", "baseModelInstanceId": 0,
        "externalBaseModelUrl": "https://huggingface.co/dmis-lab/UltraBERT-380M",
    }
    (package_dir / "model-instance-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (package_dir / "README.md").write_text(
        f"# Kaggle-only UltraBERT V0 fine-tune\n\n"
        f"Experiment: `{manifest['experiment_id']}`\n\n"
        f"Source: `{manifest['source_experiment']}`\n\n"
        f"Selected epoch: {manifest['selected_epoch']}\n\n"
        f"Validation: {manifest['validation_rows']} frozen V0 prompt-group holdout rows.\n\n"
        f"Balanced accuracy: {manifest['validation_metrics']['balanced_accuracy']:.6f}.\n\n"
        "This checkpoint is Kaggle-only and is not registered in Trust List 2.\n")
    notebook = _notebook(manifest["experiment_id"], model_slug, model_version)
    (kernel_dir / f"{kernel_slug}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    kernel_meta = {
        "id": f"zodrak/{kernel_slug}",
        "title": "UltraBERT V0 Fine Tune Name That LLM",
        "code_file": f"{kernel_slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
        "competition_sources": ["name-that-llm"],
        "model_sources": [f"zodrak/ultrabert380-v2/PyTorch/{model_slug}/{model_version}"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2) + "\n")
    return package_dir, kernel_dir


def _notebook(experiment_id: str, model_slug: str, model_version: int):
    source = f'''from pathlib import Path
import json, os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXPECTED_EXPERIMENT_ID = {experiment_id!r}
INPUT_ROOT = Path('/kaggle/input')
MODEL_ROOT_CANDIDATES = [
    INPUT_ROOT / 'models' / 'zodrak' / 'ultrabert380-v2' / 'pytorch' / {model_slug!r} / str({model_version}),
    INPUT_ROOT / 'models' / 'zodrak' / 'ultrabert380-v2' / 'PyTorch' / {model_slug!r} / str({model_version}),
]
selection_paths = []
for root in MODEL_ROOT_CANDIDATES:
    candidate = root / 'selection.json'
    if candidate.is_file():
        selection_paths.append(candidate)
if not selection_paths:
    visited = set()
    for root, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        try:
            stat = os.stat(root)
            identity = (stat.st_dev, stat.st_ino)
        except OSError:
            dirs[:] = []
            continue
        if identity in visited:
            dirs[:] = []
            continue
        visited.add(identity)
        if 'selection.json' in files:
            selection_paths.append(Path(root) / 'selection.json')
matching = []
for path in selection_paths:
    try:
        if json.loads(path.read_text()).get('experiment_id') == EXPECTED_EXPERIMENT_ID:
            matching.append(path)
    except (OSError, json.JSONDecodeError):
        pass
assert len(matching) == 1, (f'Expected one selected checkpoint; found {{[str(p) for p in matching]}}; '
                            f'input roots={{[str(p) for p in INPUT_ROOT.iterdir()]}}')
MODEL_DIR = matching[0].parent
selection = json.loads(matching[0].read_text())
assert selection['scope'] == 'kaggle_only_not_in_trust_list_2'
MAX_LENGTH = int(selection['max_length'])
TEST_CANDIDATES = [Path('/kaggle/input/competitions/name-that-llm/test.csv'), Path('/kaggle/input/name-that-llm/test.csv')]
TEST_CSV = next((p for p in TEST_CANDIDATES if p.is_file()), None)
assert TEST_CSV is not None, f'Competition test.csv not found: {{TEST_CANDIDATES}}'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
assert torch.cuda.is_available(), 'Kaggle GPU is unavailable'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True).to('cuda').eval()
label_map = model.config.id2label
classes = [label_map[str(i)] if str(i) in label_map else label_map[i] for i in range(model.config.num_labels)]
assert classes == selection['classes'] and len(classes) == 18
test = pd.read_csv(TEST_CSV, dtype={{'id':'string'}})
assert 'id' in test and ('text' in test or 'response' in test)
text_column = 'text' if 'text' in test else 'response'
assert test.id.notna().all() and not test.id.duplicated().any()
texts = test[text_column].fillna('').astype(str).tolist()
predicted_ids = []; batch_size = 8; start = 0
with torch.inference_mode():
    while start < len(texts):
        current = min(batch_size, len(texts) - start)
        try:
            batch = tokenizer(texts[start:start+current], padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors='pt')
            batch = {{key:value.to('cuda') for key,value in batch.items()}}
            logits = model(**batch).logits
            predicted_ids.extend(logits.argmax(dim=-1).cpu().tolist())
            start += current
        except torch.cuda.OutOfMemoryError:
            if current <= 1: raise
            batch = None; torch.cuda.empty_cache(); batch_size = max(1, current // 2)
assert len(predicted_ids) == len(test)
submission = pd.DataFrame({{'id':test.id, 'model':[classes[i] for i in predicted_ids]}})
assert submission.columns.tolist() == ['id','model']
assert submission.id.astype(str).tolist() == test.id.astype(str).tolist()
assert submission.id.is_unique and submission.model.notna().all() and submission.model.isin(classes).all()
output = Path('/kaggle/working/submission.parquet'); submission.to_parquet(output, index=False)
print({{'experiment_id':EXPECTED_EXPERIMENT_ID,'selected_epoch':selection['epoch'],'rows':len(submission),
       'columns':submission.columns.tolist(),'output':str(output),'validation_balanced_accuracy':selection['balanced_accuracy']}})
display(submission.head())
'''
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Name That LLM — Kaggle-only V0 UltraBERT fine-tune\n",
                "Offline GPU inference from the V0 fine-tuned checkpoint. This run is excluded from Trust List 2.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": source.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/kaggle"))
    parser.add_argument("--model-slug", default="ultrabert380-v0-finetuned")
    parser.add_argument("--kernel-slug", default="ultrabert-v0-fine-tune-name-that-llm")
    parser.add_argument("--model-version", type=int, default=1)
    args = parser.parse_args()
    package, kernel = build_package(args.manifest, args.output_root,
        model_slug=args.model_slug, kernel_slug=args.kernel_slug,
        model_version=args.model_version)
    print(json.dumps({"model_package": str(package), "kernel_package": str(kernel)}, indent=2))


if __name__ == "__main__":
    main()
