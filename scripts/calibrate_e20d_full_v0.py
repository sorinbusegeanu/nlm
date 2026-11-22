"""Calibrate and time every E20-D checkpoint on full V0 locally.

For each neural checkpoint, start with one batch containing all V0 rows and let
the production inference path halve on CUDA OOM. Rows are sorted by tokenizer
length before inference so the first successful batch includes the longest
V0 inputs. Results are resumable and written after each checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from scripts.publish_kaggle_checkpoint_store import ROOT, resolve_checkpoint
from src.inference import predict_candidate
from src.v2_checkpoint_trust import LABELS


MANIFEST = ROOT / "artifacts/kaggle/e20d-checkpoints-v001/e20d_manifest.json"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
OUT_JSON = ROOT / "artifacts/analysis/e20d_full_v0_batch_calibration.json"
OUT_CSV = ROOT / "artifacts/analysis/e20d_full_v0_batch_calibration.csv"
MAX_BATCH = 2699
CPU_MODELS = {"svc", "catboost_stylometry"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_texts(tokenizer, texts, model, config):
    output = list(map(str, texts))
    if model == "embeddinggemma2":
        output = ["task: classification | query: " + text for text in output]
    if config.get("use_chat_template", False):
        user_prompt = config.get("chat_template_user_prompt",
                                 "Identify the source model for the assistant response.")
        output = [tokenizer.apply_chat_template(
            [{"role": "user", "content": str(user_prompt)},
             {"role": "assistant", "content": text}],
            tokenize=False, add_generation_prompt=False) for text in output]
    return output


def write_results(doc):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(doc["results"]).to_csv(OUT_CSV, index=False)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Full V0 batch calibration requires the local CUDA GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(MANIFEST.read_text())
    v0 = pd.read_parquet(V0_PATH)
    required = {"id", "text"}
    if len(v0) != MAX_BATCH or not required.issubset(v0.columns):
        raise ValueError(f"Expected {MAX_BATCH} V0 rows with id and text columns")
    row_ids = v0.id.astype(str).tolist()
    texts = v0.text.fillna("").astype(str).tolist()
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("V0 row IDs are not unique")
    data_hash = sha256_file(V0_PATH)
    old_results = {}
    if OUT_JSON.is_file():
        try:
            old = json.loads(OUT_JSON.read_text())
            if old.get("v0_sha256") == data_hash:
                old_results = {x["checkpoint_id"]: x for x in old.get("results", [])
                               if x.get("status") == "ok"}
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    device = torch.cuda.get_device_name(0)
    results = []
    for checkpoint_id in manifest["checkpoint_ids"]:
        item = manifest["checkpoint_members"][checkpoint_id]
        if checkpoint_id in old_results:
            prior = old_results[checkpoint_id]
            results.append(prior)
            print(f"Reusing {checkpoint_id}: batch={prior['max_batch_size']} elapsed={prior['inference_seconds']:.1f}s", flush=True)
            continue

        model = item["model"]
        config = dict(item.get("config", {}))
        experiment_id = item["experiment_id"]
        _, checkpoint = resolve_checkpoint(experiment_id)
        config["inference_batch_size"] = MAX_BATCH
        spec = {
            "candidate_id": checkpoint_id,
            "model": model,
            "checkpoint": str(checkpoint),
            "base_model": None,
            "config": config,
            "model_revision": item.get("model_revision"),
            "calibration_temperature": float(item.get("calibration_temperature") or 1.0),
        }
        started = time.perf_counter()
        peak_mib = None
        longest = None
        sequence_order = np.arange(len(texts), dtype=np.int64)
        try:
            if model in CPU_MODELS:
                inference_texts = texts
                effective_batch = None
                batch_note = "CPU model; inference is not GPU-batched"
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, local_files_only=True, trust_remote_code=True)
                prepared = prepare_texts(tokenizer, texts, model, config)
                lengths = np.asarray(tokenizer(
                    prepared, add_special_tokens=True, truncation=False,
                    padding=False, return_attention_mask=False)["input_ids"], dtype=object)
                length_values = np.asarray([len(x) for x in lengths], dtype=np.int32)
                sequence_order = np.argsort(-length_values, kind="stable")
                longest = int(length_values[sequence_order[0]])
                inference_texts = [texts[int(i)] for i in sequence_order]
                del tokenizer, prepared, lengths, length_values
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                effective_batch = None
                batch_note = ""

            probabilities, metadata = predict_candidate(
                spec, inference_texts, LABELS, Path("/"), return_batch_metadata=True)
            probabilities = np.asarray(probabilities)
            if probabilities.shape != (len(v0), len(LABELS)):
                raise ValueError(f"probability shape mismatch: {probabilities.shape}")
            if not np.isfinite(probabilities).all():
                raise ValueError("non-finite probabilities")
            if model not in CPU_MODELS:
                effective_batch = int(metadata["effective_batch_size"])
                peak_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                # Reorder output to canonical V0 row order before hashing.
                restored = np.empty_like(probabilities)
                restored[sequence_order] = probabilities
                probabilities = restored
                batch_note = "Longest-token V0 rows first; adaptive production CUDA OOM halving"
            # Record a stable proof that all outputs were computed in V0 row order.
            pred_hash = hashlib.sha256(np.asarray(probabilities, dtype=np.float32).tobytes()).hexdigest()
            result = {
                "checkpoint_id": checkpoint_id,
                "experiment_id": experiment_id,
                "model": model,
                "max_length": config.get("max_length"),
                "v0_rows": len(v0),
                "v0_row_id_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
                "probability_sha256_float32": pred_hash,
                "max_batch_size": effective_batch if model not in CPU_MODELS else "not_applicable_cpu",
                "configured_probe_batch_size": MAX_BATCH,
                "longest_token_length": longest,
                "inference_seconds": round(time.perf_counter() - started, 3),
                "responses_per_second": round(len(v0) / max(time.perf_counter() - started, 1e-9), 3),
                "peak_allocated_mib": peak_mib,
                "batch_note": batch_note,
                "status": "ok",
            }
            del probabilities
        except Exception as exc:
            result = {
                "checkpoint_id": checkpoint_id,
                "experiment_id": experiment_id,
                "model": model,
                "v0_rows": len(v0),
                "configured_probe_batch_size": MAX_BATCH,
                "inference_seconds": round(time.perf_counter() - started, 3),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
        write_results({
            "experiment": "E20-D full V0 per-checkpoint batch calibration",
            "checkpoint_manifest": str(MANIFEST.relative_to(ROOT)),
            "checkpoint_manifest_sha256": sha256_file(MANIFEST),
            "v0_path": str(V0_PATH.relative_to(ROOT)),
            "v0_sha256": data_hash,
            "v0_rows": len(v0),
            "canonical_labels": LABELS,
            "device": device,
            "device_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1),
            "probe_method": "Run full V0 beginning at batch 2699; production inference halves batch on CUDA OOM; rows are sorted by untruncated tokenizer length so initial chunks contain the longest inputs; outputs restored to original row order.",
            "results": results,
        })
        del spec
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Wrote {OUT_JSON} and {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
