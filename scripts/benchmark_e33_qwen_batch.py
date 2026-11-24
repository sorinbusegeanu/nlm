"""Calibrate E19-D's exact Qwen3.5-4B generation batch on longest V0 texts."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src import e19b_reconstruct, e19d_reconstruct, ensemble_catalog

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/home/zodrak/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
OUTPUT = ROOT / "artifacts/analysis/e33/e33_qwen4b_batch_calibration.json"
SAMPLE_ROWS = 64
START_BATCH = 64
MAX_NEW_TOKENS = 1024


def calibrate() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("E33 Qwen calibration requires CUDA")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Pinned Qwen3.5-4B snapshot is missing: {MODEL_DIR}")
    v0 = ensemble_catalog.cascades._v0(ROOT)
    texts = v0.text.fillna("").astype(str).tolist()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    # Select the 64 longest response inputs under the same tokenizer used by
    # the reconstruction generation path; fixed instruction overhead is equal.
    tokens = processor.tokenizer(texts, add_special_tokens=True,
        truncation=False, padding=False)["input_ids"]
    lengths = np.asarray([len(x) for x in tokens], dtype=np.int64)
    order = np.argsort(-lengths, kind="stable")[:SAMPLE_ROWS]
    sample = [texts[int(i)] for i in order]
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR, device_map="auto", dtype="auto", local_files_only=True).eval()
    device = next(model.parameters()).device
    free_before, total_bytes = torch.cuda.mem_get_info()
    size = START_BATCH
    start = 0
    attempts = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    while start < len(sample):
        current = min(size, len(sample) - start)
        attempt_start = time.perf_counter()
        try:
            inputs = e19b_reconstruct._generation_inputs(processor, sample[start:start + current])
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            prefix = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id)
            new_tokens = int((outputs[:, prefix:] != processor.tokenizer.pad_token_id).sum().item())
            attempts.append({"batch_size": current, "status": "ok",
                "seconds": round(time.perf_counter() - attempt_start, 3),
                "generated_tokens": new_tokens})
            start += current
            del inputs, outputs
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": current, "status": "oom",
                "seconds": round(time.perf_counter() - attempt_start, 3)})
            if "inputs" in locals(): del inputs
            if "outputs" in locals(): del outputs
            import gc
            gc.collect(); torch.cuda.empty_cache()
            if current <= 1:
                raise
            size = max(1, current // 2)
    elapsed = time.perf_counter() - started
    success_batches = [r["batch_size"] for r in attempts if r["status"] == "ok"]
    result = {
        "experiment": "E33 E19-D Qwen3.5-4B full-V0 longest-input batch calibration",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": e19d_reconstruct.MODEL_REVISION,
        "v0_rows": len(v0), "sample_rows": len(sample),
        "sample_policy": "64 longest raw V0 responses by Qwen3.5-4B tokenizer length; generation uses the E19-D instruction template",
        "longest_response_tokens": int(lengths[order[0]]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_input_tokens": e19b_reconstruct.MAX_INPUT_TOKENS,
        "start_batch": START_BATCH,
        "minimum_successful_batch": min(success_batches),
        "maximum_successful_batch": max(success_batches),
        "effective_oom_safe_batch": int(size),
        "elapsed_seconds": round(elapsed, 3),
        "responses_per_second": round(len(sample) / elapsed, 4),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "free_vram_before_probe_mib": round(free_before / 2**20, 1),
        "total_vram_mib": round(total_bytes / 2**20, 1),
        "attempts": attempts,
        "timing_scope": "generation only; model load and prompt embedding excluded",
        "device": torch.cuda.get_device_name(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    del model, processor
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
