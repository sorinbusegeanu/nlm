"""Classical and optional neural model adapters with aligned outputs."""
from __future__ import annotations

import time
import json
import math
import datetime
import os
import inspect
import hashlib
import gc
import shutil
from collections import deque
from functools import wraps
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.svm import LinearSVC

from .feature_factory import (DEFAULT_STYLOMETRY_RECIPE, STYLOMETRY_VERSION,
                              FoldFeatures, load_or_fit, stylometry)


def _aligned(values, fitted_classes, classes):
    values = np.asarray(values)
    fitted_classes = list(fitted_classes)
    if values.ndim == 1:
        if len(fitted_classes) != 2:
            raise ValueError("one-dimensional decision scores require exactly two fitted classes")
        # LinearSVC's positive binary margin belongs to classes_[1].
        values = np.column_stack([-values, values])
    if values.ndim != 2 or values.shape[1] != len(fitted_classes):
        raise ValueError("decision score columns do not match fitted classes")
    result = np.zeros((len(values), len(classes)), dtype=float)
    indices = {label: i for i, label in enumerate(classes)}
    for j, label in enumerate(fitted_classes):
        result[:, indices[label]] = values[:, j]
    return result


def _length_order(lengths):
    """Stable short-to-long order for dynamic-padding inference batches."""
    return np.argsort(np.asarray(lengths, dtype=np.int64), kind="stable")


def _restore_prediction_order(sorted_predictions, sorted_indices):
    """Map predictions generated in a sorted batch order back to source rows."""
    predictions = np.asarray(sorted_predictions)
    order = np.asarray(sorted_indices, dtype=np.int64)
    if len(predictions) != len(order):
        raise ValueError("sorted predictions and source indices must have equal lengths")
    return predictions[np.argsort(order, kind="stable")]


def _selected_epoch_scores(epoch_predictions, selected_epoch, expected_shape):
    """Return aligned logits only when they belong to the selected checkpoint epoch."""
    if selected_epoch is None:
        return None
    for item in epoch_predictions or []:
        if int(item.get("epoch", -1)) != int(selected_epoch):
            continue
        scores = np.asarray(item.get("scores"))
        if scores.shape == tuple(expected_shape) and np.isfinite(scores).all():
            return scores
        return None
    return None


def _snapshot_state_dict_cpu(model):
    """Copy a selected neural checkpoint to CPU once, without retaining GPU storage."""
    return {key: value.detach().to(device="cpu", copy=True)
            for key, value in model.state_dict().items()}


def _restore_model_state(model, state):
    """Restore a previously captured full model state exactly."""
    model.load_state_dict(state)


def _should_restore_selected_epoch(best_epoch, last_evaluated_epoch):
    """Current weights are selected if the best epoch is the last validation."""
    return best_epoch != last_evaluated_epoch


def _largest_safe_batch_size(maximum, fits):
    """Find the greatest safe integer, then verify the boundary by stepping down by one."""
    maximum = max(1, int(maximum))
    successful, failed, candidate = 0, None, 1
    while candidate <= maximum:
        if not fits(candidate):
            failed = candidate
            break
        successful = candidate
        if candidate == maximum:
            break
        candidate = min(candidate * 2, maximum)
    if successful == 0:
        raise MemoryError("GPU batch-size probe could not fit batch size 1; reduce max_length or use a smaller model")
    if failed is not None:
        low, high = successful + 1, failed - 1
        while low <= high:
            middle = (low + high) // 2
            if fits(middle):
                successful = middle
                low = middle + 1
            else:
                high = middle - 1
        # Verify the boundary in the same one-at-a-time way used by OOM
        # recovery: after a failure at N, explicitly retry N-1, then keep
        # stepping down if memory pressure makes that attempt fail too.
        candidate = min(maximum, successful + 1)
        while candidate <= maximum:
            if fits(candidate):
                successful = candidate
                candidate += 1
                continue
            candidate -= 1
            verified = None
            while candidate > 0:
                if fits(candidate):
                    verified = candidate
                    break
                candidate -= 1
            if verified is None:
                raise MemoryError("GPU batch-size probe could not re-verify a safe batch after OOM")
            successful = verified
            break
    return successful


def _batch_probe_memory_limit(base_reserved_bytes, free_bytes, usable_fraction,
                              explicit_reserve_bytes=0):
    """Return a memory ceiling while retaining the configured fraction of free VRAM."""
    usable_fraction = float(usable_fraction)
    if not 0 < usable_fraction <= 1:
        raise ValueError("batch_probe_safety_fraction must be in (0, 1]")
    free_bytes = max(0, int(free_bytes))
    reserve = max(max(0, int(explicit_reserve_bytes)),
                  int(round(free_bytes * (1.0 - usable_fraction))))
    return int(base_reserved_bytes) + max(0, free_bytes - reserve), reserve


def _select_probed_batch_size(maximum_safe, benchmark_results, attempts):
    """Always select the largest verified safe batch; throughput is diagnostic only."""
    safe_maximum = next((item for item in reversed(benchmark_results)
                         if int(item.get("batch_size", 0)) == int(maximum_safe)
                         and item.get("fits")), None)
    if safe_maximum is None:
        safe_maximum = next((item for item in reversed(attempts)
                             if int(item.get("batch_size", 0)) == int(maximum_safe)
                             and item.get("fits")), None)
    if safe_maximum is None:
        raise MemoryError("batch-size probe has no successful measurement for its safe maximum")
    return int(maximum_safe), safe_maximum, False


def _resolve_batch_accumulation(physical_batch_size, configured_accumulation=1,
                                target_effective_batch_size=None):
    """Honor explicit effective-batch targets with the largest exact-fit microbatch."""
    physical = max(1, int(physical_batch_size))
    accumulation = max(1, int(configured_accumulation))
    if target_effective_batch_size is None:
        return physical, accumulation
    target = int(target_effective_batch_size)
    if target < 1:
        raise ValueError("target_effective_batch_size must be positive")
    physical = min(physical, target)
    while target % physical:
        physical -= 1
    return physical, target // physical


def _optimizer_steps_per_epoch(example_count, physical_batch_size, gradient_accumulation_steps):
    """Count optimizer updates for padded/dynamic batches and configured accumulation."""
    denominator = max(1, int(physical_batch_size)) * max(1, int(gradient_accumulation_steps))
    return max(1, math.ceil(max(0, int(example_count)) / denominator))


def select_epoch_view_indices(view_counts, *, seed=2026, epoch=0, view_ids=None):
    """Choose one reproducible view per original response for a global epoch."""
    choices, counts = [], {}
    for row_index, count in enumerate(view_counts):
        count = int(count)
        if count < 1:
            raise ValueError(f"original response {row_index} has no training views")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), row_index]))
        choice = int(rng.integers(0, count))
        choices.append(choice)
        names = (view_ids[row_index] if view_ids is not None else
                 ["full", *[f"chunk-{i:02d}" for i in range(1, count)]])
        name = str(names[choice])
        counts[name] = counts.get(name, 0) + 1
    return choices, counts


def _is_oom_error(exc):
    message = str(exc).lower()
    return any(marker in message for marker in (
        "out of memory", "cuda oom", "cublas_status_alloc_failed", "mps backend out of memory"))


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _release_cuda_after_oom():
    """Drop Python references and release cached CUDA allocations after a failed attempt."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError, AttributeError):
        pass


def _is_complete_trainer_checkpoint(path):
    """Whether a Trainer checkpoint is complete enough to restore exact state."""
    path = Path(path)
    if not path.is_dir():
        return False
    required_state = ("trainer_state.json", "scheduler.pt")
    try:
        files = {item.name for item in path.iterdir() if item.is_file()}
    except OSError:
        return False
    has_adapter = any(name in files for name in ("adapter_model.safetensors", "adapter_model.bin"))
    has_model = any(name in files for name in (
        "model.safetensors", "pytorch_model.bin", "adapter_model.safetensors",
        "adapter_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
    has_rng = any(name == "rng_state.pth" or name.startswith("rng_state_") for name in files)
    has_optimizer = "optimizer.pt" in files or "optimizer.bin" in files
    has_head = not has_adapter or "classifier.pt" in files
    return has_model and has_head and has_rng and has_optimizer and all(name in files for name in required_state)


def _latest_complete_trainer_checkpoint(directory):
    """Return the newest HF Trainer checkpoint with model and optimizer state."""
    directory = Path(directory)
    candidates = []
    for path in directory.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if _is_complete_trainer_checkpoint(path):
            candidates.append((step, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _copy_training_recovery_state(resume_checkpoint, output_dir):
    """Carry validation-selection sidecars forward with a resumed Trainer state."""
    if not resume_checkpoint:
        return
    source = Path(resume_checkpoint).parent / "training_state"
    if source.is_dir():
        destination = Path(output_dir) / "training_state"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    previous_output = Path(resume_checkpoint).parent
    for epoch_checkpoint in previous_output.glob("epoch-*"):
        if epoch_checkpoint.is_dir():
            target = Path(output_dir) / epoch_checkpoint.name
            if not target.exists():
                shutil.copytree(epoch_checkpoint, target)


def _atomic_torch_save(torch, value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _remove_trainer_recovery_files(output_dir):
    """Remove transient optimizer checkpoints and selection sidecars on success."""
    output_dir = Path(output_dir)
    for checkpoint in output_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint, ignore_errors=True)
    shutil.rmtree(output_dir / "training_state", ignore_errors=True)


def _with_cuda_oom_recovery(function):
    """Retry neural training at a smaller microbatch from its latest safe state.

    The wrapped Trainer keeps a rolling optimizer-step checkpoint. On OOM, the
    latest complete checkpoint restores model, optimizer, scheduler, RNG and
    Trainer progress; without one, the attempt restarts from its configured
    base/continuation. Trainer rebuilds the remaining-step schedule for the
    recovered batch. Validation and prediction OOMs use their own fallbacks.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        output_dir = Path(bound.arguments["output_dir"])
        configured_batch = max(1, int(bound.arguments.get("batch_size", 1)))
        configured_accumulation = max(1, int(bound.arguments.get("gradient_accumulation_steps", 1)))
        original_effective_batch = configured_batch * configured_accumulation
        events = []
        attempt_number = 0
        failed_attempt_seconds = 0.0
        fixed_batch_after_recovery = False
        recovery_file = output_dir.parent / f"oom_recovery_{output_dir.name}.json"
        archive_root = output_dir.parent / f"oom_attempts_{output_dir.name}"

        def save_recovery(status="retrying"):
            _write_json_atomic(recovery_file, {
                "status": status,
                "training_entry_point": function.__name__,
                "configured_microbatch": configured_batch,
                "configured_gradient_accumulation_steps": configured_accumulation,
                "configured_effective_batch_size": original_effective_batch,
                "events": events,
            })

        while True:
            attempt_number += 1
            attempt_output = output_dir
            if events:
                attempt_output = output_dir.parent / f".{output_dir.name}.oom_try_{attempt_number:02d}"
                if attempt_output.exists():
                    raise FileExistsError(f"stale CUDA OOM attempt directory exists: {attempt_output}")
            bound.arguments["output_dir"] = attempt_output
            context = {"phase": "initialization"}
            bound.arguments["_oom_recovery_context"] = context
            attempt_started = time.perf_counter()
            try:
                result = function(*bound.args, **bound.kwargs)
            except Exception as exc:
                recoverable_phase = context.get("phase") in {"initialization", "training"}
                if not _is_oom_error(exc) or not recoverable_phase:
                    raise
                try:
                    probe_path = attempt_output / "batch_size_probe.json"
                    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
                except (OSError, ValueError):
                    probe = {}
                failed_batch = max(1, int(probe.get("effective_batch_size", bound.arguments.get("batch_size", 1))))
                accumulation = max(1, int(bound.arguments.get("gradient_accumulation_steps", 1)))
                old_effective = failed_batch * accumulation
                event = {
                    "attempt": attempt_number,
                    "attempt_elapsed_seconds": time.perf_counter() - attempt_started,
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "phase": context.get("phase"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "original_microbatch": failed_batch,
                    "original_gradient_accumulation_steps": accumulation,
                    "original_effective_batch_size": old_effective,
                    "batch_probe": probe or None,
                    "scheduler_action": "rebuild_schedule_for_recovered_batch_then_restore_if_available",
                }
                if failed_batch <= 1:
                    event.update({"recovered_microbatch": None,
                                  "recovered_gradient_accumulation_steps": None,
                                  "action": "propagated_cuda_oom_at_microbatch_1"})
                    events.append(event)
                    if attempt_output.exists():
                        archive_root.mkdir(parents=True, exist_ok=True)
                        archive_path = archive_root / f"attempt_{attempt_number:02d}"
                        shutil.move(str(attempt_output), str(archive_path))
                        event["archived_attempt"] = str(archive_path)
                    save_recovery("failed_at_minimum_batch")
                    _release_cuda_after_oom()
                    raise

                recovered_batch = max(1, failed_batch // 2)
                recovered_batch, recovered_accumulation = _resolve_batch_accumulation(
                    recovered_batch, 1, old_effective)
                event.update({
                    "recovered_microbatch": recovered_batch,
                    "recovered_gradient_accumulation_steps": recovered_accumulation,
                    "recovered_effective_batch_size": recovered_batch * recovered_accumulation,
                    "effective_batch_size_preserved_exactly": recovered_batch * recovered_accumulation == old_effective,
                    "action": "reduce_batch_and_recover_latest_consistent_state",
                })
                events.append(event)
                failed_attempt_seconds += float(event["attempt_elapsed_seconds"])
                archived_attempt = None
                if attempt_output.exists():
                    archive_root.mkdir(parents=True, exist_ok=True)
                    archive_path = archive_root / f"attempt_{attempt_number:02d}"
                    shutil.move(str(attempt_output), str(archive_path))
                    event["archived_attempt"] = str(archive_path)
                    archived_attempt = archive_path
                recovery_checkpoint = (_latest_complete_trainer_checkpoint(archived_attempt)
                                       if archived_attempt is not None else None)
                training_rows = len(bound.arguments.get("train_text", []))
                steps_per_epoch = _optimizer_steps_per_epoch(
                    training_rows, recovered_batch, recovered_accumulation)
                epochs = int(bound.arguments.get("epochs", 1))
                warmup_epochs = bound.arguments.get("warmup_epochs")
                warmup_basis_epochs = min(epochs, int(warmup_epochs)) if warmup_epochs is not None else epochs
                warmup_steps = math.ceil(float(bound.arguments.get("warmup_ratio", 0.0))
                                         * warmup_basis_epochs * steps_per_epoch)
                event["recovered_schedule"] = {
                    "steps_per_epoch": steps_per_epoch,
                    "planned_optimizer_steps": steps_per_epoch * epochs,
                    "warmup_steps": warmup_steps,
                    "remaining_optimizer_steps": None,
                }
                if recovery_checkpoint is not None:
                    bound.arguments["_trainer_resume_checkpoint"] = str(recovery_checkpoint)
                    event["recovery_checkpoint"] = str(recovery_checkpoint)
                    event["recovery_global_step"] = int(recovery_checkpoint.name.rsplit("-", 1)[1])
                    event["resume_mode"] = "resume_model_optimizer_scheduler_rng_and_trainer_state"
                    try:
                        trainer_state = json.loads((recovery_checkpoint / "trainer_state.json").read_text())
                        event["recovery_epoch"] = trainer_state.get("epoch")
                        event["recovered_schedule"]["remaining_optimizer_steps"] = max(
                            0, event["recovered_schedule"]["planned_optimizer_steps"]
                            - event["recovery_global_step"])
                        event["previous_planned_optimizer_steps"] = trainer_state.get("max_steps")
                    except (OSError, ValueError, TypeError):
                        pass
                else:
                    configured_resume = bound.arguments.get("trainer_resume_checkpoint")
                    if configured_resume and _is_complete_trainer_checkpoint(configured_resume):
                        bound.arguments["_trainer_resume_checkpoint"] = str(configured_resume)
                        event["action"] = "reduce_batch_and_resume_configured_trainer_checkpoint"
                        event["recovery_checkpoint"] = str(configured_resume)
                        try:
                            saved_state = json.loads((Path(configured_resume) / "trainer_state.json").read_text())
                            event["recovery_global_step"] = int(saved_state.get("global_step", 0))
                            event["recovery_epoch"] = saved_state.get("epoch")
                        except (OSError, ValueError, TypeError):
                            event["recovery_global_step"] = 0
                        event["resume_mode"] = "resume_configured_model_optimizer_scheduler_rng_and_trainer_state"
                    else:
                        event["action"] = "reduce_batch_and_restart_cleanly_no_recoverable_state"
                        bound.arguments["_trainer_resume_checkpoint"] = None
                        event["recovery_checkpoint"] = None
                        event["recovery_global_step"] = 0
                        event["resume_mode"] = "clean_restart_no_complete_trainer_checkpoint"
                bound.arguments["batch_size"] = recovered_batch
                bound.arguments["gradient_accumulation_steps"] = recovered_accumulation
                if "auto_batch_size" in bound.arguments:
                    # The original probe is recorded above. Fix the recovered
                    # batch to ensure the retry cannot choose the same failure.
                    bound.arguments["auto_batch_size"] = False
                fixed_batch_after_recovery = True
                save_recovery()
                # The exception traceback otherwise keeps the failed Trainer,
                # model and optimizer frames alive during cache cleanup.
                exc.__traceback__ = None
                exc.__cause__ = None
                exc.__context__ = None
                del exc
                _release_cuda_after_oom()
                continue

            if attempt_output != output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                for child in attempt_output.iterdir():
                    target = output_dir / child.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(child), str(target))
                attempt_output.rmdir()
            if events:
                final_batch = max(1, int(result.get("effective_batch_size", bound.arguments.get("batch_size", 1))))
                final_accumulation = int(bound.arguments.get("gradient_accumulation_steps", 1))
                result["requested_batch_size"] = configured_batch
                result["gradient_accumulation_steps"] = final_accumulation
                result["oom_recovery"] = {
                    "recovered": True,
                    "configured_microbatch": configured_batch,
                    "configured_gradient_accumulation_steps": configured_accumulation,
                    "configured_effective_batch_size": original_effective_batch,
                    "effective_batch_size_preserved_exactly": final_batch * final_accumulation == original_effective_batch,
                    "events": events,
                    "recovery_artifact": str(recovery_file),
                    "attempt_archive": str(archive_root),
                    "batch_size_probe_disabled_after_recovery": fixed_batch_after_recovery,
                }
                result["fit_seconds"] = float(result.get("fit_seconds", 0.0)) + failed_attempt_seconds
                if isinstance(result.get("runtime_breakdown"), dict):
                    result["runtime_breakdown"]["oom_recovery_retry_seconds"] = failed_attempt_seconds
                if isinstance(result.get("batch_size_probe"), dict):
                    result["batch_size_probe"]["oom_recovery"] = result["oom_recovery"]
                for key in ("best_checkpoint_path", "selected_checkpoint_path"):
                    path = result.get(key)
                    if path and str(path).startswith(str(attempt_output)):
                        result[key] = str(output_dir) + str(path)[len(str(attempt_output)):]
                result["epoch_checkpoints"] = [
                    str(output_dir) + str(path)[len(str(attempt_output)):]
                    if str(path).startswith(str(attempt_output) + os.sep) else path
                    for path in result.get("epoch_checkpoints", [])]
                save_recovery("completed_after_retry")
            return result

    return wrapped


def _predict_cached_dataset(model, dataset, *, batch_size, data_collator, device,
                            amp_dtype=None, num_workers=0, return_info=False):
    """Infer length-grouped cached rows, retrying CUDA OOMs at smaller batches."""
    import torch
    from torch.utils.data import DataLoader, Subset

    if hasattr(dataset, "lengths"):
        lengths = dataset.lengths
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, "lengths"):
        lengths = [dataset.dataset.lengths[int(index)] for index in dataset.indices]
    else:
        raise ValueError("cached inference dataset must expose token lengths")
    order = _length_order(lengths)
    ordered = Subset(dataset, order.tolist())
    effective_batch_size = max(1, int(batch_size))
    fallback_count = 0
    started = time.perf_counter()
    model.eval()
    chunks = []
    offset = 0
    while offset < len(ordered):
        remaining = Subset(ordered, range(offset, len(ordered)))
        loader = DataLoader(remaining, batch_size=effective_batch_size, shuffle=False,
                            collate_fn=data_collator, num_workers=int(num_workers))
        try:
            with torch.inference_mode():
                for batch in loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    amp = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                           if getattr(device, "type", None) == "cuda" and amp_dtype is not None
                           else nullcontext())
                    with amp:
                        output = model(**batch)
                    logits = output.logits if hasattr(output, "logits") else output
                    values = logits.float().cpu().numpy()
                    chunks.append(values)
                    offset += len(values)
            continue
        except (RuntimeError, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)) as exc:
            if not _is_oom_error(exc) or effective_batch_size <= 1:
                raise
            effective_batch_size = max(1, effective_batch_size // 2)
            fallback_count += 1
            output = logits = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    sorted_predictions = (np.concatenate(chunks, axis=0) if chunks
                          else np.empty((0, 0), dtype=np.float32))
    predictions = _restore_prediction_order(sorted_predictions, order)
    elapsed = time.perf_counter() - started
    info = {"effective_inference_batch_size": effective_batch_size,
            "inference_oom_fallbacks": fallback_count,
            "inference_rows_per_second": (len(predictions) / elapsed if elapsed else None),
            "inference_seconds": elapsed}
    return (predictions, info) if return_info else predictions


def _evaluate_with_oom_batch_fallback(evaluate, *, get_batch_size, set_batch_size,
                                      empty_cache=None):
    """Retry a failed evaluation at a smaller eval batch without touching train settings."""
    size = max(1, int(get_batch_size()))
    fallbacks = 0
    while True:
        try:
            return evaluate(), size, fallbacks
        except RuntimeError as exc:
            if not _is_oom_error(exc) or size <= 1:
                raise
            size = max(1, size // 2)
            set_batch_size(size)
            fallbacks += 1
            if empty_cache is not None:
                empty_cache()


def _set_gradient_checkpointing(model, enabled):
    """Set Hugging Face model activation checkpointing when the architecture supports it."""
    enabled = bool(enabled)
    currently_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
    if enabled == currently_enabled:
        return
    if enabled:
        method = getattr(model, "gradient_checkpointing_enable", None)
    else:
        method = getattr(model, "gradient_checkpointing_disable", None)
    if method is None:
        if enabled:
            raise RuntimeError(f"{type(model).__name__} does not support gradient checkpointing")
        return
    method()
    if enabled and hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False


def _gradient_checkpointing_for_batch(requested, _batch_size):
    """Use the configured checkpointing mode consistently for every batch size."""
    return bool(requested)


def _gradient_checkpointing_target(model):
    """Return the backbone that owns activation checkpointing for wrapped models."""
    return getattr(model, "backbone", model)


def _validation_loss_raw(y_true, raw_scores, classes):
    """Calculate validation log loss from unscaled model scores."""
    from sklearn.metrics import log_loss
    raw_probabilities = softmax(np.asarray(raw_scores, dtype=float), axis=1)
    return float(log_loss(y_true, raw_probabilities, labels=list(classes)))


def _write_ragged_token_cache(path, cache_key, rows):
    """Persist processor tokens as numeric ragged arrays without pickle."""
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    payload = {"cache_key": np.asarray(cache_key), "rows": np.asarray(len(rows)),
               "keys": np.asarray(json.dumps(keys, separators=(",", ":")))}
    for key in keys:
        offsets = [0]
        values = []
        for row in rows:
            value = np.asarray(row.get(key, []), dtype=np.int64).reshape(-1)
            values.extend(value.tolist())
            offsets.append(len(values))
        payload[f"{key}_values"] = np.asarray(values, dtype=np.int32)
        payload[f"{key}_offsets"] = np.asarray(offsets, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _read_ragged_token_cache(path, cache_key, row_count):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["cache_key"].item()) != cache_key or int(cache["rows"].item()) != row_count:
                return None
            result = {}
            for key in json.loads(str(cache["keys"].item())):
                values = cache[f"{key}_values"]
                offsets = cache[f"{key}_offsets"]
                if len(offsets) != row_count + 1 or offsets[-1] != len(values):
                    return None
                result[key] = [values[offsets[index]:offsets[index + 1]].astype(np.int64, copy=False)
                               for index in range(row_count)]
            return result
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _ragged_token_cache_valid(path, cache_key, row_count):
    try:
        with np.load(path, allow_pickle=False) as cache:
            return (str(cache["cache_key"].item()) == cache_key
                    and int(cache["rows"].item()) == row_count)
    except (OSError, ValueError, KeyError):
        return False


def train_svc(train_text, y_train, validation_text, classes, *, c=.5, seed=2026, feature_config=None, cache_dir=None, fold_token="", validation_labels=None, calibration=None, calibration_cv=3):
    """Fit TF-IDF and LinearSVC; optionally calibrate its margins using train-only CV."""
    start = time.perf_counter()
    if cache_dir:
        x_train, x_val, features, cache_key = load_or_fit(train_text, validation_text,
            config=feature_config or {"views": ["char", "word"]}, cache_dir=cache_dir, fold_token=fold_token)
    else:
        features = FoldFeatures(feature_config)
        x_train = features.fit_transform(train_text)
        x_val = features.transform(validation_text)
        cache_key = None
    base = LinearSVC(C=c, class_weight="balanced", random_state=seed)
    if calibration is None:
        fitted = base.fit(x_train, y_train)
    elif calibration == "sigmoid":
        from sklearn.calibration import CalibratedClassifierCV
        fitted = CalibratedClassifierCV(estimator=base, method="sigmoid",
                                        cv=int(calibration_cv), ensemble="auto")
        fitted.fit(x_train, y_train)
    else:
        raise ValueError("SVC calibration must be unset or 'sigmoid'")
    pred_start = time.perf_counter()
    if calibration == "sigmoid":
        prob = _aligned(np.asarray(fitted.predict_proba(x_val)), fitted.classes_, classes)
        raw = np.log(np.maximum(prob, np.finfo(np.float64).tiny))
    else:
        raw = _aligned(np.asarray(fitted.decision_function(x_val)), fitted.classes_, classes)
        prob = softmax(raw, axis=1)
    validation_loss = (_validation_loss_raw(validation_labels, raw, classes)
                       if validation_labels is not None else None)
    inference_seconds = time.perf_counter() - pred_start
    elapsed = time.perf_counter() - start
    return {"scores": raw, "probabilities": prob, "model": fitted, "raw_model": fitted,
            "features": features, "cache_key": cache_key, "fit_seconds": elapsed, "classes": list(classes),
            "inference_seconds": inference_seconds, "peak_gpu_memory_bytes": None,
            "checkpoint_scoring_method": "in_memory_fitted_model_validation_predictions",
            "final_validation_reused_existing_predictions": False,
            "runtime_breakdown": {"training_seconds": float(pred_start - start),
                "final_validation_inference_seconds": float(inference_seconds),
                "best_checkpoint_restoration_seconds": 0.0},
            "validation_loss_raw": validation_loss,
            "calibration_method": (f"sigmoid train-only CV ({int(calibration_cv)} folds)"
                                   if calibration == "sigmoid" else "none")}


class _BalancedAccuracyMetric:
    """CatBoost eval metric matching sklearn's unweighted macro recall."""

    def __init__(self, classes):
        self.class_to_index = {str(label): index for index, label in enumerate(classes)}
        self.class_count = len(classes)

    def is_max_optimal(self):
        return True

    def get_final_error(self, error, weight):
        return float(error) / max(float(weight), 1.0)

    def evaluate(self, approxes, target, weight):
        predicted = np.asarray(approxes, dtype=float).argmax(axis=0)
        actual = []
        for value in target:
            label = str(value)
            if label in self.class_to_index:
                actual.append(self.class_to_index[label])
                continue
            numeric = int(float(value))
            if not 0 <= numeric < self.class_count:
                raise ValueError(f"CatBoost validation target is not a known class: {value!r}")
            actual.append(numeric)
        actual = np.asarray(actual, dtype=int)
        recalls = [float(np.mean(predicted[actual == index] == index))
                   for index in range(self.class_count) if np.any(actual == index)]
        return float(sum(recalls)), float(len(recalls))


def train_catboost_stylometry(train_text, y_train, validation_text, classes, *, output_dir,
                              iterations=800, depth=7, learning_rate=.05, l2_leaf_reg=5.0,
                              thread_count=8, seed=2026,
                              validation_labels=None, early_stopping_rounds=100,
                              feature_recipe=None, feature_matrices=None, init_model=None,
                              checkpoint_metadata=None, checkpoint_selection_metric="balanced_accuracy"):
    """Fit CatBoost on deterministic row-local style statistics; no validation fitting."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost model requires catboost; install the project's requirements.txt") from exc
    start = time.perf_counter()
    if checkpoint_selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be accuracy or balanced_accuracy")
    feature_recipe = {"baseline": list(DEFAULT_STYLOMETRY_RECIPE["baseline"]),
                      "groups": [], **(feature_recipe or {})}
    if feature_matrices is None:
        x_train, feature_names = stylometry(train_text, feature_recipe, return_feature_names=True)
        x_validation = stylometry(validation_text, feature_recipe)
    else:
        x_train, x_validation, feature_names = feature_matrices
        x_train = np.asarray(x_train, dtype=np.float64)
        x_validation = np.asarray(x_validation, dtype=np.float64)
        feature_names = list(feature_names)
    metric = _BalancedAccuracyMetric(classes) if checkpoint_selection_metric == "balanced_accuracy" else "Accuracy"
    model = CatBoostClassifier(iterations=int(iterations), depth=int(depth),
        learning_rate=float(learning_rate), l2_leaf_reg=float(l2_leaf_reg),
        loss_function="MultiClass", eval_metric=metric, auto_class_weights="Balanced",
        random_seed=int(seed), thread_count=int(thread_count), verbose=False,
        allow_writing_files=False)
    fit_kwargs = {}
    if init_model is not None:
        fit_kwargs["init_model"] = str(init_model)
    if validation_labels is not None:
        fit_kwargs.update({"eval_set": (x_validation, list(validation_labels)),
                           "use_best_model": True,
                           "early_stopping_rounds": int(early_stopping_rounds)})
    startup_seconds = time.perf_counter() - start
    training_started = time.perf_counter()
    model.fit(x_train, list(y_train), **fit_kwargs)
    post_training_started = time.perf_counter()
    training_seconds = time.perf_counter() - training_started
    iteration_metrics = []
    selected_iteration = (int(model.get_best_iteration()) if validation_labels is not None
                          else int(model.tree_count_) if model.tree_count_ else None)
    if validation_labels is not None and selected_iteration < 0:
        raise RuntimeError("CatBoost did not identify a best validation iteration")
    selected_validation_accuracy = None
    selected_validation_balanced_accuracy = None
    if validation_labels is not None:
        history = model.get_evals_result().get("validation", {})
        metric_prefix = "Balanced" if checkpoint_selection_metric == "balanced_accuracy" else "Accuracy"
        metric_values = next((values for name, values in history.items()
                              if metric_prefix.casefold() in name.casefold()), [])
        iteration_metrics = [{"iteration": index + 1,
                              f"validation_{checkpoint_selection_metric}": float(value)}
                             for index, value in enumerate(metric_values)]
    pred_start = time.perf_counter()
    validation_raw = np.asarray(model.predict(x_validation, prediction_type="RawFormulaVal"))
    inference_seconds = time.perf_counter() - pred_start
    model_classes = list(model.classes_)
    if validation_raw.ndim == 1:
        validation_raw = np.column_stack([-validation_raw, validation_raw])
    raw = _aligned(validation_raw, model_classes, classes)
    probabilities = softmax(raw, axis=1)
    validation_loss = (_validation_loss_raw(validation_labels, raw, classes)
                       if validation_labels is not None else None)
    from sklearn.metrics import accuracy_score
    validation_accuracy = (float(accuracy_score(validation_labels,
        np.asarray(classes)[raw.argmax(axis=1)])) if validation_labels is not None else None)
    if validation_labels is not None:
        selected_validation_accuracy = validation_accuracy
        from sklearn.metrics import balanced_accuracy_score
        selected_validation_balanced_accuracy = float(balanced_accuracy_score(
            validation_labels, np.asarray(classes)[raw.argmax(axis=1)]))
    best_iteration = selected_iteration if validation_labels is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_started = time.perf_counter()
    model.save_model(str(output_dir / "model.cbm"))
    metadata = {
        "feature_version": STYLOMETRY_VERSION,
        "feature_names": feature_names,
        "feature_recipe": feature_recipe,
        "feature_count": int(x_train.shape[1]),
        "validation_loss_raw": validation_loss,
        "selected_iteration": best_iteration,
        "selected_validation_accuracy": selected_validation_accuracy,
        "selected_validation_balanced_accuracy": selected_validation_balanced_accuracy,
        "selection_metric": f"validation_{checkpoint_selection_metric}" if validation_labels is not None else None,
        "classes": list(classes), "recipe": {"iterations": int(iterations), "depth": int(depth),
            "learning_rate": float(learning_rate), "l2_leaf_reg": float(l2_leaf_reg),
            "thread_count": int(thread_count), "seed": int(seed),
            "feature_recipe": feature_recipe}
    }
    metadata.update({"tree_count": int(model.tree_count_),
                     "added_iterations": int(iterations),
                     "parent_checkpoint": str(init_model) if init_model is not None else None})
    if checkpoint_metadata:
        metadata["experiment"] = dict(checkpoint_metadata)
    (output_dir / "stylometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checkpoint_save_seconds = time.perf_counter() - checkpoint_started
    return {"scores": raw, "probabilities": probabilities, "model": model,
        "fit_seconds": time.perf_counter() - start, "inference_seconds": inference_seconds,
        "runtime_breakdown": {"startup_seconds": startup_seconds,
            "training_including_validation_seconds": training_seconds,
            "validation_seconds": None,
            "final_validation_seconds": inference_seconds,
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "effective_inference_batch_size": None,
            "checkpoint_serialization_seconds": float(checkpoint_save_seconds),
            "model_post_training_seconds": float(time.perf_counter() - post_training_started)},
        "classes": list(classes),
        "selected_iteration": best_iteration, "selected_validation_accuracy": selected_validation_accuracy,
        "selected_validation_balanced_accuracy": selected_validation_balanced_accuracy,
        "validation_metrics_by_iteration": iteration_metrics,
        "validation_loss_raw": validation_loss,
        "feature_count": int(x_train.shape[1]), "feature_names": feature_names,
        "feature_recipe": feature_recipe,
        "checkpoint_selection_metric": f"validation_{checkpoint_selection_metric}" if validation_labels is not None else None,
        "checkpoint": str(output_dir), "tree_count": int(model.tree_count_),
        "checkpoint_scoring_method": "in_memory_selected_iteration_validation_predictions",
        "final_validation_reused_existing_predictions": False,
        "added_iterations": int(iterations), "parent_checkpoint": str(init_model) if init_model is not None else None,
        "peak_gpu_memory_bytes": None}


FROZEN_BACKBONE_MODELS = {"embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"}
EMBEDDINGGEMMA2_TASK_PREFIX = "task: classification | query: "


def calculate_max_token_length(model_name, texts, *, checkpoint, revision=None,
                               task_prefix="", use_processor=False,
                               trust_remote_code=False, fix_mistral_regex=False,
                               local_files_only=True, batch_size=256):
    """Return the uncapped maximum encoded length and reject context overflow."""
    _force_huggingface_offline()
    _, tr = _require_neural()
    load_kwargs = {"local_files_only": local_files_only}
    if revision is not None:
        load_kwargs["revision"] = revision
    if not use_processor and trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if use_processor and model_name == "ministral3_3b" and fix_mistral_regex:
        load_kwargs["fix_mistral_regex"] = True
    tokenizer = (tr.AutoProcessor.from_pretrained(checkpoint, **load_kwargs) if use_processor
                 else tr.AutoTokenizer.from_pretrained(checkpoint, **load_kwargs))
    config = tr.AutoConfig.from_pretrained(checkpoint, **{
        key: value for key, value in load_kwargs.items() if key != "fix_mistral_regex"})

    context_fields = ("max_position_embeddings", "n_positions", "max_sequence_length",
                      "max_seq_len", "max_seq_length", "seq_length")
    context_limits = []
    configs = [config]
    for field in ("text_config", "language_config"):
        nested = getattr(config, field, None)
        if nested is not None:
            configs.append(nested)
    for item in configs:
        for field in context_fields:
            value = getattr(item, field, None)
            if isinstance(value, int) and 0 < value < 10_000_000:
                context_limits.append(value)
    tokenizer_obj = getattr(tokenizer, "tokenizer", tokenizer)
    tokenizer_limit = getattr(tokenizer_obj, "model_max_length", None)
    if not context_limits and isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        context_limits.append(tokenizer_limit)
    if not context_limits:
        raise ValueError(f"could not determine supported context window for {model_name} ({checkpoint})")
    context_window = min(context_limits)

    prefix = (EMBEDDINGGEMMA2_TASK_PREFIX if use_processor and model_name == "embeddinggemma2"
              else "" if use_processor else str(task_prefix or ""))
    all_texts = [prefix + str(text) for text in texts]
    cache_payload = json.dumps({"model": model_name, "checkpoint": str(checkpoint),
        "revision": revision, "prefix": prefix, "processor": bool(use_processor),
        "trust_remote_code": bool(trust_remote_code), "fix_mistral_regex": bool(fix_mistral_regex),
        "texts_sha256": hashlib.sha256("\0".join(all_texts).encode("utf-8")).hexdigest(),
        "rows": len(all_texts)}, sort_keys=True)
    cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()
    cache_root = Path(os.environ.get("NLM_TOKEN_LENGTH_CACHE", "artifacts/cache/token_lengths"))
    cache_path = cache_root / f"{cache_key}.json"
    tokenized_cache_path = cache_root / f"tokenized_{cache_key}.npz"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            maximum = int(cached["maximum_tokens"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # Ignore malformed/stale cache files and recompute from the tokenizer.
            cached, maximum = None, None
        token_cache_valid = (not use_processor or
            _ragged_token_cache_valid(tokenized_cache_path, cache_key, len(all_texts)))
        if (cached is not None and cached.get("cache_key") == cache_key and token_cache_valid
                and int(cached.get("rows", -1)) == len(all_texts)):
            if maximum > context_window:
                raise ValueError(f"{model_name} dataset requires {maximum} tokens (including special tokens and task prefix), "
                                 f"exceeding the model context window of {context_window}")
            return maximum
    maximum = 0
    token_rows = []
    for start in range(0, len(all_texts), max(1, int(batch_size))):
        batch_texts = all_texts[start:start + max(1, int(batch_size))]
        encoded = (tokenizer(text=batch_texts, truncation=False, padding=False,
                             add_special_tokens=True)
                   if use_processor else tokenizer(batch_texts, truncation=False, padding=False,
                                                   add_special_tokens=True))
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if lengths:
            maximum = max(maximum, max(lengths))
        if use_processor:
            for index in range(len(batch_texts)):
                token_rows.append({key: encoded[key][index]
                    for key in ("input_ids", "attention_mask", "mm_token_type_ids")
                    if key in encoded})
    if maximum > context_window:
        raise ValueError(f"{model_name} dataset requires {maximum} tokens (including special tokens and task prefix), "
                         f"exceeding the model context window of {context_window}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"cache_key": cache_key, "maximum_tokens": maximum,
        "rows": len(all_texts), "context_window": context_window}, indent=2) + "\n", encoding="utf-8")
    if use_processor:
        _write_ragged_token_cache(tokenized_cache_path, cache_key, token_rows)
    return maximum


def _load_frozen_backbone(model_name, checkpoint, revision, *, fix_mistral_regex=False):
    """Load the text feature path for one of the additional multimodal backbones."""
    _force_huggingface_offline()
    torch, tr = _require_neural()
    processor_kwargs = {"revision": revision, "local_files_only": True}
    if model_name == "ministral3_3b" and fix_mistral_regex:
        processor_kwargs["fix_mistral_regex"] = True
    processor = tr.AutoProcessor.from_pretrained(checkpoint, **processor_kwargs)
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16 if torch.cuda.is_available() else torch.float32)
    if model_name == "embeddinggemma2":
        model = tr.AutoModel.from_pretrained(checkpoint, revision=revision, local_files_only=True,
                                             torch_dtype=dtype)
        text_model = model
    elif model_name in {"gemma4_e2b", "gemma4_e4b"}:
        if not hasattr(tr, "AutoModelForMultimodalLM"):
            raise RuntimeError("Gemma 4 requires a Transformers release with AutoModelForMultimodalLM")
        model = tr.AutoModelForMultimodalLM.from_pretrained(checkpoint, revision=revision,
                                                            local_files_only=True, torch_dtype=dtype)
        text_model = model.model
    elif model_name == "ministral3_3b":
        model = tr.Mistral3ForConditionalGeneration.from_pretrained(
            checkpoint, revision=revision, local_files_only=True, torch_dtype=dtype)
        text_model = model.model
    else:
        raise ValueError(f"unsupported frozen backbone: {model_name}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, text_model, processor


def _read_embedding_cache(cache_path, metadata_path, cache_key, row_ids):
    """Load cached features only when their exact ordered row alignment matches."""
    cache_path, metadata_path = Path(cache_path), Path(metadata_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") != cache_key or metadata.get("row_ids") != list(row_ids):
            return None
        with np.load(cache_path) as archive:
            features = archive["features"]
        if features.shape[0] != len(row_ids) or not np.isfinite(features).all():
            return None
        return features
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _restore_qwen_selected_state(backbone, classifier, adapter_state, classifier_state, *, setter=None):
    """Restore the selected LoRA adapter and its matching classifier readout."""
    if setter is None:
        from peft.utils.save_and_load import set_peft_model_state_dict
        setter = set_peft_model_state_dict
    setter(backbone, adapter_state, adapter_name="default")
    classifier.load_state_dict(classifier_state)


def extract_frozen_backbone_features(model_name, checkpoint, revision, texts, *, max_length=384,
                                     batch_size=8, model_dir=None, fix_mistral_regex=False,
                                     row_ids=None, cache_dir="artifacts/cache/frozen_embeddings",
                                     inference_stats=None, trust_remote_code=False):
    """Extract deterministic, text-only mean-pooled features without gradient updates."""
    import torch
    texts = [str(text) for text in texts]
    aligned_ids = ([str(value) for value in row_ids] if row_ids is not None else
                   [f"{index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
                    for index, text in enumerate(texts)])
    if len(aligned_ids) != len(texts) or len(set(aligned_ids)) != len(aligned_ids):
        raise ValueError("frozen embedding row IDs must be unique and aligned to texts")
    prefix = EMBEDDINGGEMMA2_TASK_PREFIX if model_name == "embeddinggemma2" else ""
    prepared = [prefix + text for text in texts]
    token_payload = json.dumps({"model": model_name, "checkpoint": str(checkpoint),
        "revision": revision, "prefix": prefix, "processor": True,
        "trust_remote_code": bool(trust_remote_code),
        "fix_mistral_regex": bool(fix_mistral_regex),
        "texts_sha256": hashlib.sha256("\0".join(prepared).encode("utf-8")).hexdigest(),
        "rows": len(prepared)}, sort_keys=True)
    token_key = hashlib.sha256(token_payload.encode()).hexdigest()
    token_root = Path(os.environ.get("NLM_TOKEN_LENGTH_CACHE", "artifacts/cache/token_lengths"))
    tokenized_rows = _read_ragged_token_cache(
        token_root / f"tokenized_{token_key}.npz", token_key, len(prepared))
    cache_config = {"model": model_name, "checkpoint": str(checkpoint), "revision": revision,
        "tokenizer_config": {"processor": True, "fix_mistral_regex": bool(fix_mistral_regex)},
        "max_length": int(max_length), "pooling": "attention-mask mean",
        "texts_sha256": hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest(),
        "row_ids_sha256": hashlib.sha256("\0".join(aligned_ids).encode("utf-8")).hexdigest()}
    cache_key = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.npz"
    metadata_path = Path(cache_dir) / f"{cache_key}.json"
    cached_features = _read_embedding_cache(cache_path, metadata_path, cache_key, aligned_ids)
    if cached_features is not None:
        return cached_features, 0.0
    model, text_model, processor = _load_frozen_backbone(
        model_name, model_dir or checkpoint, revision,
        fix_mistral_regex=fix_mistral_regex)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []
    started = time.perf_counter()
    inference_batch_size = max(1, int(batch_size))
    tokenization_chunk_size = inference_batch_size * 8
    with torch.inference_mode():
        for start in range(0, len(prepared), tokenization_chunk_size):
            stop = min(len(prepared), start + tokenization_chunk_size)
            encoded = ({key: values[start:stop] for key, values in tokenized_rows.items()}
                       if tokenized_rows is not None else
                       processor(text=prepared[start:stop], truncation=True,
                                 max_length=max_length, padding=False))
            # These three text models need token IDs and masks; Gemma 4 also
            # consumes its text modality markers. Ignore processor-only extras.
            keys = {"input_ids", "attention_mask"}
            if model_name == "gemma4_e4b":
                keys.add("mm_token_type_ids")
            row_count = stop - start
            lengths = [len(encoded["input_ids"][index]) for index in range(row_count)]
            order = _length_order(lengths)
            tokenizer_obj = getattr(processor, "tokenizer", processor)
            pad_id = getattr(tokenizer_obj, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer_obj, "eos_token_id", None)
            if pad_id is None:
                pad_id = 0
            chunk_outputs = []
            offset = 0
            while offset < row_count:
                indexes = order[offset:offset + inference_batch_size].tolist()
                try:
                    padded = {}
                    for key in keys:
                        if key not in encoded:
                            if key == "attention_mask":
                                seqs = [torch.ones(len(encoded["input_ids"][index]), dtype=torch.long)
                                        for index in indexes]
                            else:
                                continue
                        else:
                            seqs = [torch.as_tensor(encoded[key][index], dtype=torch.long)
                                    for index in indexes]
                        padding_value = int(pad_id) if key == "input_ids" else 0
                        padded[key] = torch.nn.utils.rnn.pad_sequence(
                            seqs, batch_first=True, padding_value=padding_value).to(device)
                    result = text_model(**padded, use_cache=False, return_dict=True)
                    hidden = result.last_hidden_state.float()
                    mask = padded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    if model_name == "embeddinggemma2":
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    values = pooled.cpu().numpy()
                    chunk_outputs.append(values)
                    offset += len(values)
                except RuntimeError as exc:
                    if not _is_oom_error(exc) or inference_batch_size <= 1:
                        raise
                    inference_batch_size = max(1, inference_batch_size // 2)
                    if inference_stats is not None:
                        inference_stats["oom_fallbacks"] = inference_stats.get("oom_fallbacks", 0) + 1
                    del padded
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            outputs.append(_restore_prediction_order(np.concatenate(chunk_outputs, axis=0), order))
    elapsed = time.perf_counter() - started
    if inference_stats is not None:
        inference_stats["effective_inference_batch_size"] = inference_batch_size
        inference_stats["inference_rows_per_second"] = len(texts) / elapsed if elapsed else None
        inference_stats["inference_seconds"] = elapsed
    features = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
    del model, text_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not np.isfinite(features).all():
        raise ValueError(f"{model_name} produced non-finite text features")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    metadata_path.write_text(json.dumps({"cache_key": cache_key, "config": cache_config,
        "row_ids": aligned_ids, "rows": len(texts), "shape": list(features.shape)},
        indent=2) + "\n", encoding="utf-8")
    return features, elapsed


def train_frozen_backbone_fold(model_name, train_text, y_train, validation_text, classes, *,
                               checkpoint, revision, output_dir, max_length=384, batch_size=8,
                               inference_batch_size=None,
                               c=.5, seed=2026, fix_mistral_regex=False,
                               validation_labels=None, train_row_ids=None, validation_row_ids=None,
                               embedding_cache_dir="artifacts/cache/frozen_embeddings",
                               trust_remote_code=False):
    """Fit a train-only LinearSVC readout over frozen model-specific text features."""
    if model_name not in FROZEN_BACKBONE_MODELS:
        raise ValueError(f"unsupported frozen backbone: {model_name}")
    start = time.perf_counter()
    same_rows = list(train_text) == list(validation_text)
    all_text = list(train_text) if same_rows else list(train_text) + list(validation_text)
    all_ids = (list(train_row_ids) if train_row_ids is not None else
               [f"train:{i}:{hashlib.sha256(str(text).encode()).hexdigest()}" for i, text in enumerate(train_text)])
    if not same_rows:
        all_ids += (list(validation_row_ids) if validation_row_ids is not None else
                    [f"validation:{i}:{hashlib.sha256(str(text).encode()).hexdigest()}" for i, text in enumerate(validation_text)])
    extraction_stats = {}
    all_features, extraction_seconds = extract_frozen_backbone_features(
        model_name, checkpoint, revision, all_text, max_length=max_length,
        batch_size=inference_batch_size or batch_size, fix_mistral_regex=fix_mistral_regex, row_ids=all_ids,
        cache_dir=embedding_cache_dir, inference_stats=extraction_stats,
        trust_remote_code=trust_remote_code)
    train_features = all_features[:len(train_text)]
    validation_features = all_features if same_rows else all_features[len(train_text):]
    base = LinearSVC(C=float(c), class_weight="balanced", random_state=seed)
    readout_fit_started = time.perf_counter()
    base.fit(train_features, y_train)
    readout_fit_seconds = time.perf_counter() - readout_fit_started
    inference_started = time.perf_counter()
    raw = _aligned(np.asarray(base.decision_function(validation_features)), base.classes_, classes)
    probabilities = softmax(raw, axis=1)
    validation_loss = (_validation_loss_raw(validation_labels, raw, classes)
                       if validation_labels is not None else None)
    validation_fraction = 1.0 if same_rows else len(validation_text) / max(1, len(train_text) + len(validation_text))
    validation_inference_seconds = time.perf_counter() - inference_started
    inference_seconds = extraction_seconds * validation_fraction + validation_inference_seconds
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = output_dir / "readout.joblib"
    import joblib
    serialization_started = time.perf_counter()
    joblib.dump({"model": base, "raw_model": base, "classes": list(classes),
                 "model_name": model_name, "checkpoint": checkpoint, "revision": revision,
                 "max_length": int(max_length), "feature_pooling": "attention-mask mean",
                 "task_prefix": "task: classification | query: " if model_name == "embeddinggemma2" else "",
                 "fix_mistral_regex": bool(fix_mistral_regex)}, joblib_path)
    checkpoint_serialization_seconds = time.perf_counter() - serialization_started
    return {"scores": raw, "probabilities": probabilities, "model": base,
            "raw_model": base, "fit_seconds": time.perf_counter() - start,
            "inference_seconds": inference_seconds, "classes": list(classes),
            "checkpoint_scoring_method": "in_memory_fitted_readout_validation_predictions",
            "final_validation_reused_existing_predictions": True,
            "checkpoint": str(output_dir),
            "readout_path": str(joblib_path), "peak_gpu_memory_bytes": None,
            "runtime_breakdown": {"frozen_feature_extraction_seconds": float(extraction_seconds),
                "readout_fit_seconds": float(readout_fit_seconds),
                "final_validation_inference_seconds": float(validation_inference_seconds),
                "checkpoint_serialization_seconds": float(checkpoint_serialization_seconds),
                "effective_inference_batch_size": extraction_stats.get("effective_inference_batch_size",
                    inference_batch_size or batch_size),
                "inference_oom_fallbacks": extraction_stats.get("oom_fallbacks", 0),
                "frozen_feature_extraction_rows_per_second": extraction_stats.get("inference_rows_per_second"),
                "best_checkpoint_restoration_seconds": 0.0},
            "validation_loss_raw": validation_loss}


def _require_neural():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Neural training needs the optional neural dependencies. Install this project's [neural] extra on the training computer.") from exc
    return torch, transformers


def _length_grouping_args(transformers_module):
    """Enable length grouping across supported Transformers APIs."""
    parameters = inspect.signature(transformers_module.TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in parameters:
        return {"train_sampling_strategy": "group_by_length"}
    if "group_by_length" in parameters:
        return {"group_by_length": True}
    return {}


def _force_huggingface_offline():
    """Prevent model training processes from making any Hugging Face Hub requests."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _live_training_progress_callback(TrainerCallback, output_dir, *, steps_per_epoch=None, epochs=None,
                                    physical_batch_size=None, gradient_accumulation_steps=1,
                                    mean_tokens_per_example=None, torch_module=None):
    """Write epoch metrics and lightweight live step/ETA state for the dashboard."""
    progress_path = Path(output_dir) / "training_progress.json"
    class LiveTrainingProgressCallback(TrainerCallback):
        def __init__(self):
            self.epochs = {}
            self.baseline = None
            self.live = None
            self.last_step_at = None
            self.step_durations = deque(maxlen=20)
            self.last_live_write = 0.0

        def _write_progress(self):
            payload = {"pid": os.getpid(), "updated_at": time.time(),
                       "epochs": [self.epochs[key] for key in sorted(self.epochs)]}
            if self.baseline is not None:
                payload["baseline"] = self.baseline
            if self.live is not None:
                payload["live"] = self.live
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, progress_path)

        def set_baseline(self, metrics):
            self.baseline = dict(metrics)
            self._write_progress()

        def on_train_begin(self, args, state, control, **kwargs):
            # HF checkpoints retain log_history; reconstruct the dashboard view
            # before resumed training emits its next event.
            self.epochs = {}
            for logs in state.log_history or []:
                raw_epoch = logs.get("epoch")
                if raw_epoch is None:
                    continue
                epoch = int(round(float(raw_epoch)))
                if epoch < 1:
                    continue
                row = self.epochs.setdefault(epoch, {"epoch": epoch})
                for source, target in (("learning_rate", "learning_rate"), ("loss", "train_loss"),
                                       ("eval_loss", "validation_loss_raw"), ("eval_accuracy", "accuracy"),
                                       ("eval_balanced_accuracy", "balanced_accuracy")):
                    if logs.get(source) is not None:
                        row[target] = float(logs[source])
                if logs.get("eval_loss") is not None:
                    row["validation_loss"] = float(logs["eval_loss"])
            per_epoch = int(steps_per_epoch or max(
                1, math.ceil(state.max_steps / max(1, int(epochs or args.num_train_epochs or 1)))))
            global_step = int(state.global_step)
            self.live = {"global_step": global_step,
                         "total_steps": int(state.max_steps),
                         "epoch": (min((global_step - 1) // per_epoch + 1,
                                       max(1, int(epochs or args.num_train_epochs or 1)))
                                   if global_step else 1),
                         "epoch_step": (((global_step - 1) % per_epoch) + 1
                                        if global_step else 0),
                         "steps_per_epoch": per_epoch,
                         "epoch_eta_seconds": None,
                         "updated_at": time.time()}
            if torch_module is not None and torch_module.cuda.is_available():
                try:
                    device = torch_module.cuda.current_device()
                    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
                    self.live["training_peak_allocated_bytes"] = int(
                        torch_module.cuda.max_memory_allocated(device))
                    self.live["training_peak_reserved_bytes"] = int(
                        torch_module.cuda.max_memory_reserved(device))
                    self.live["gpu_vram_utilization"] = 1.0 - free_bytes / max(1, total_bytes)
                except (RuntimeError, AttributeError):
                    pass
            self.last_step_at = time.monotonic()
            self._write_progress()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            now = time.monotonic()
            if self.last_step_at is not None:
                duration = now - self.last_step_at
                if duration > 0:
                    self.step_durations.append(duration)
            self.last_step_at = now
            total_epochs = max(1, int(epochs or args.num_train_epochs or 1))
            per_epoch = int(steps_per_epoch or max(1, math.ceil(state.max_steps / total_epochs)))
            global_step = int(state.global_step)
            epoch_step = ((global_step - 1) % per_epoch) + 1 if global_step else 0
            current_epoch = min((global_step - 1) // per_epoch + 1, total_epochs) if global_step else 1
            step_seconds = (sum(self.step_durations) / len(self.step_durations)
                            if self.step_durations else None)
            self.live = {"global_step": global_step,
                         "total_steps": int(state.max_steps),
                         "epoch": int(current_epoch),
                         "epoch_step": int(epoch_step),
                         "steps_per_epoch": per_epoch,
                         "epoch_eta_seconds": (max(0, per_epoch - epoch_step) * step_seconds
                                               if step_seconds is not None else None),
                         "updated_at": time.time()}
            if step_seconds:
                examples_per_step = (max(1, int(physical_batch_size or args.per_device_train_batch_size))
                                     * max(1, int(gradient_accumulation_steps)))
                self.live["training_examples_per_second"] = examples_per_step / step_seconds
                if mean_tokens_per_example is not None:
                    self.live["training_tokens_per_second"] = (
                        examples_per_step * float(mean_tokens_per_example) / step_seconds)
                self.live["training_step_seconds"] = step_seconds
            should_write = now - self.last_live_write >= 5 or epoch_step >= per_epoch
            if should_write and torch_module is not None and torch_module.cuda.is_available():
                try:
                    device = torch_module.cuda.current_device()
                    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
                    self.live["training_peak_allocated_bytes"] = int(
                        torch_module.cuda.max_memory_allocated(device))
                    self.live["training_peak_reserved_bytes"] = int(
                        torch_module.cuda.max_memory_reserved(device))
                    self.live["gpu_vram_utilization"] = 1.0 - free_bytes / max(1, total_bytes)
                except (RuntimeError, AttributeError):
                    pass
            # The dashboard refreshes every ten seconds; avoid a disk replace
            # on every optimizer update while keeping the displayed ETA current.
            if should_write:
                self._write_progress()
                self.last_live_write = now
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            # Exclude validation/checkpoint time between epochs from the next
            # epoch's step-time estimate.
            self.step_durations.clear()
            self.last_step_at = time.monotonic()
            if self.live is not None:
                self.live["updated_at"] = time.time()
                self._write_progress()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            raw_epoch = logs.get("epoch", state.epoch)
            if raw_epoch is None:
                return control
            epoch = int(round(float(raw_epoch)))
            if epoch < 1:
                return control
            row = self.epochs.setdefault(epoch, {"epoch": epoch})
            for source, target in (("learning_rate", "learning_rate"), ("loss", "train_loss"),
                                   ("eval_loss", "validation_loss_raw"), ("eval_accuracy", "accuracy"),
                                   ("eval_balanced_accuracy", "balanced_accuracy")):
                if logs.get(source) is not None:
                    row[target] = float(logs[source])
            if logs.get("eval_loss") is not None:
                row["validation_loss"] = float(logs["eval_loss"])
            self._write_progress()
            return control
    return LiveTrainingProgressCallback()


def _accuracy_early_stopping_callback(TrainerCallback, patience, metric="accuracy"):
    """Stop after `patience` consecutive validations without a better metric."""
    patience = int(patience)
    if patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")

    class AccuracyEarlyStoppingCallback(TrainerCallback):
        def __init__(self):
            self.patience = patience
            self.best_metric_value = float("-inf")
            self.best_accuracy = None
            self.best_balanced_accuracy = None
            self.best_epoch = None
            self.bad_epochs = 0
            self.stopped_epoch = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            value = (metrics or {}).get(f"eval_{metric}")
            if value is None:
                return control
            epoch = int(round(float(state.epoch or 0)))
            value = float(value)
            if value > self.best_metric_value:
                self.best_metric_value = value
                self.best_accuracy = (metrics or {}).get("eval_accuracy")
                self.best_balanced_accuracy = (metrics or {}).get("eval_balanced_accuracy")
                self.best_epoch = epoch
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= patience:
                    self.stopped_epoch = epoch
                    control.should_training_stop = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            self.best_metric_value = float("-inf")
            self.best_accuracy = None
            self.best_balanced_accuracy = None
            self.best_epoch = None
            self.bad_epochs = 0
            self.stopped_epoch = None
            for record in state.log_history or []:
                value = record.get(f"eval_{metric}")
                if value is not None:
                    self._record(float(value), record, control, restoring=True)
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True
            return control

        def _record(self, value, record, control, *, restoring=False):
            epoch = int(round(float(record.get("epoch", 0))))
            if value > self.best_metric_value:
                self.best_metric_value = value
                self.best_accuracy = record.get("eval_accuracy")
                self.best_balanced_accuracy = record.get("eval_balanced_accuracy")
                self.best_epoch = epoch
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    self.stopped_epoch = epoch
                    if not restoring:
                        control.should_training_stop = True

    return AccuracyEarlyStoppingCallback()


def _early_stopping_summary(callback, patience, max_epochs, metric="accuracy"):
    return {
        "metric": f"validation_{metric}",
        "patience_epochs": int(patience),
        "best_epoch": callback.best_epoch,
        "best_validation_accuracy": callback.best_accuracy,
        "best_validation_balanced_accuracy": callback.best_balanced_accuracy,
        "best_validation_metric_value": (callback.best_metric_value
                                         if callback.best_epoch is not None else None),
        "stopped_early": callback.stopped_epoch is not None and callback.stopped_epoch < max_epochs,
        "stopped_epoch": callback.stopped_epoch,
    }


def _epoch_prediction_capture_callback(TrainerCallback, latest_prediction, state_dir=None):
    """Capture the probability-model inputs emitted by each validation epoch."""
    class EpochPredictionCaptureCallback(TrainerCallback):
        def __init__(self):
            self.state_dir = Path(state_dir) if state_dir is not None else None
            self.predictions = []
            if self.state_dir is not None:
                for path in sorted(self.state_dir.glob("validation_epoch_*.npy")):
                    try:
                        epoch = int(path.stem.rsplit("_", 1)[1])
                        self.predictions.append({"epoch": epoch,
                                                 "scores": np.load(path, allow_pickle=False)})
                    except (OSError, ValueError):
                        continue

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            logits = latest_prediction.pop("logits", None)
            if logits is not None:
                epoch = int(round(float(state.epoch or len(self.predictions) + 1)))
                self.predictions = [item for item in self.predictions if int(item["epoch"]) != epoch]
                self.predictions.append({"epoch": epoch, "scores": logits})
                if self.state_dir is not None:
                    self.state_dir.mkdir(parents=True, exist_ok=True)
                    path = self.state_dir / f"validation_epoch_{epoch:04d}.npy"
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, logits, allow_pickle=False)
                    temporary.replace(path)
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            evaluated_epochs = {int(round(float(item.get("epoch", 0))))
                                for item in state.log_history or []
                                if any(key.startswith("eval_") for key in item)}
            self.predictions = [item for item in self.predictions
                                if int(item["epoch"]) in evaluated_epochs]
            return control

    return EpochPredictionCaptureCallback()


def _epoch_decay_lambda(*, decay, steps_per_epoch, warmup_steps=0):
    """Combine linear warmup with a multiplicative decay at each epoch boundary."""
    steps_per_epoch = max(1, int(steps_per_epoch))
    warmup_steps = max(0, int(warmup_steps))
    def scale(step):
        warmup = min(1.0, max(0.0, float(step) / warmup_steps)) if warmup_steps else 1.0
        epoch = max(0, int(step) // steps_per_epoch)
        return warmup * float(decay) ** epoch
    return scale


@_with_cuda_oom_recovery
def train_encoder_fold(train_text, y_train, validation_text, classes, *, checkpoint, revision, output_dir,
                       epochs=20, batch_size=16, inference_batch_size=None, max_length=384, learning_rate=2e-5, seed=2026,
                       gradient_accumulation_steps=1, target_effective_batch_size=None,
                       local_files_only=True, validation_labels=None,
                       resume_checkpoint=None, learning_rate_decay=None,
                       trainer_resume_checkpoint=None,
                       select_best_checkpoint=False,
                       optimizer="adamw_torch", weight_decay=0.0, lr_scheduler_type=None,
                       warmup_ratio=0.0, warmup_epochs=None, early_stopping_patience=2,
                       auto_batch_size=False, max_batch_size=1024,
                       batch_probe_safety_fraction=0.95,
                       checkpoint_selection_metric="balanced_accuracy", save_epoch_checkpoints=False,
                       gradient_checkpointing=False, training_views=None, training_view_ids=None,
                       _oom_recovery_context=None, _trainer_resume_checkpoint=None,
                       copy_resume_state=True):
    """Fine-tune or continue an encoder after recording its validation baseline."""
    _force_huggingface_offline()
    recovery_context = _oom_recovery_context if _oom_recovery_context is not None else {"phase": "initialization"}
    recovery_context["phase"] = "initialization"
    if _trainer_resume_checkpoint is None:
        _trainer_resume_checkpoint = trainer_resume_checkpoint
    local_files_only = True
    torch, tr = _require_neural()
    start = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    label_map = {label: i for i, label in enumerate(classes)}
    y_indices = np.asarray([label_map[y] for y in y_train], dtype=np.int64)
    model_source = str(_trainer_resume_checkpoint or resume_checkpoint or checkpoint)
    model_revision = None if (_trainer_resume_checkpoint or resume_checkpoint) else revision
    tokenizer = tr.AutoTokenizer.from_pretrained(model_source, revision=model_revision, trust_remote_code=True,
                                                  local_files_only=local_files_only)
    model = tr.AutoModelForSequenceClassification.from_pretrained(
        model_source, revision=model_revision, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id=label_map, trust_remote_code=True,
        local_files_only=local_files_only)
    if gradient_checkpointing:
        _set_gradient_checkpointing(model, True)
    train = _TextDataset(train_text, y_indices, tokenizer, max_length,
                         training_views=training_views, training_view_ids=training_view_ids, seed=seed)
    val_indices = ([label_map[label] for label in validation_labels]
                   if validation_labels is not None else None)
    val = _TextDataset(validation_text, val_indices, tokenizer, max_length)
    data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train.collate_fn = data_collator
    val.collate_fn = data_collator
    requested_batch_size = int(batch_size)
    requested_inference_batch_size = max(1, int(inference_batch_size or batch_size))
    effective_batch_size = requested_batch_size
    batch_probe = {"enabled": False, "selected_batch_size": effective_batch_size}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    if auto_batch_size:
        if training_views is not None:
            train.set_probe_views()
        recovery_context["phase"] = "batch_probe"
        effective_batch_size, batch_probe = _probe_max_batch_size(
            model, train, requested_batch_size=requested_batch_size,
            max_batch_size=(min(int(max_batch_size), int(target_effective_batch_size))
                            if target_effective_batch_size is not None else max_batch_size),
            safety_fraction=batch_probe_safety_fraction,
            use_bf16=use_bf16, use_fp16=use_fp16, optimizer_name=optimizer,
            weight_decay=weight_decay)
        recovery_context["phase"] = "initialization"
        if training_views is not None:
            train.restore_epoch_views(0)
    effective_batch_size, gradient_accumulation_steps = _resolve_batch_accumulation(
        effective_batch_size, gradient_accumulation_steps, target_effective_batch_size)
    checkpointing_enabled = _gradient_checkpointing_for_batch(
        gradient_checkpointing, effective_batch_size)
    if gradient_checkpointing:
        _set_gradient_checkpointing(model, checkpointing_enabled)
    batch_probe.update({"requested_batch_size": requested_batch_size,
                        "effective_batch_size": effective_batch_size,
                        "selected_physical_batch_size": effective_batch_size,
                        "target_effective_batch_size": target_effective_batch_size,
                        "effective_optimizer_batch_size": effective_batch_size * gradient_accumulation_steps,
                        "gradient_accumulation_steps": int(gradient_accumulation_steps),
                        "gradient_checkpointing_enabled": checkpointing_enabled,
                        "gradient_checkpointing_disabled_for_batch_threshold": False,
                        "gradient_checkpointing_batch_policy": "configured_for_all_batch_sizes",
                        "padding_strategy": "dynamic", "pad_to_multiple_of": 8})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "batch_size_probe.json").write_text(json.dumps(batch_probe, indent=2) + "\n")
    print(f"Encoder batch-size selection: {json.dumps(batch_probe, sort_keys=True)}", flush=True)
    captured_validation_prediction = {}
    validation_order = _length_order(val.lengths)
    def compute_metrics(eval_prediction):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        logits = _restore_prediction_order(logits, validation_order)
        labels = _restore_prediction_order(eval_prediction.label_ids, validation_order)
        captured_validation_prediction["logits"] = np.asarray(logits, dtype=np.float32).copy()
        predictions = np.asarray(logits).argmax(axis=1)
        return {"accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions))}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    training_state_dir = out / "training_state"
    if copy_resume_state:
        _copy_training_recovery_state(_trainer_resume_checkpoint or resume_checkpoint, out)
    from transformers import TrainerCallback
    from torch.utils.data import Sampler

    class LengthOrderedSampler(Sampler):
        def __init__(self, dataset):
            self.order = _length_order(dataset.lengths).tolist()
        def __iter__(self): return iter(self.order)
        def __len__(self): return len(self.order)

    class OrderedValidationTrainer(tr.Trainer):
        validation_seconds = 0.0
        eval_oom_fallbacks = 0
        def _get_eval_sampler(self, eval_dataset):
            # Keep each epoch's saved validation outputs aligned to input rows.
            # Training still uses its configured length-grouped sampler.
            if hasattr(eval_dataset, "lengths"):
                return LengthOrderedSampler(eval_dataset)
            return LengthOrderedSampler(val)
        def evaluate(self, *args, **kwargs):
            started = time.perf_counter()
            previous_phase = recovery_context.get("phase", "training")
            recovery_context["phase"] = "validation"
            try:
                metrics, effective, fallbacks = _evaluate_with_oom_batch_fallback(
                    lambda: super(OrderedValidationTrainer, self).evaluate(*args, **kwargs),
                    get_batch_size=lambda: self.args.per_device_eval_batch_size,
                    set_batch_size=lambda value: setattr(self.args, "per_device_eval_batch_size", value),
                    empty_cache=torch.cuda.empty_cache if torch.cuda.is_available() else None)
                self.effective_eval_batch_size = effective
                self.eval_oom_fallbacks += fallbacks
                return metrics
            except Exception as exc:
                if _is_oom_error(exc):
                    recovery_context["phase"] = "validation_oom"
                raise
            finally:
                self.validation_seconds += time.perf_counter() - started
                if recovery_context.get("phase") == "validation":
                    recovery_context["phase"] = previous_phase

    epoch_prediction_callback = _epoch_prediction_capture_callback(
        TrainerCallback, captured_validation_prediction, training_state_dir)
    if learning_rate_decay is not None and not 0 < float(learning_rate_decay) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1]")
    if validation_labels is None:
        raise ValueError("encoder training requires validation_labels for baseline and epoch validation")
    if checkpoint_selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be accuracy or balanced_accuracy")
    best_state = {"metric": -1.0, "accuracy": None, "balanced_accuracy": None,
                  "epoch": None, "model_state": None, "snapshot_seconds": 0.0,
                  "last_evaluated_epoch": None}

    class BestCheckpointCallback(TrainerCallback):
        def __init__(self):
            self.save_seconds = 0.0
            self.epoch_checkpoints = [str(path) for path in sorted(out.glob("epoch-*")) if path.is_dir()]

        def on_train_begin(self, args, state, control, **kwargs):
            best_state["last_evaluated_epoch"] = max((
                int(round(float(item.get("epoch", 0)))) for item in state.log_history or []
                if item.get(f"eval_{checkpoint_selection_metric}") is not None), default=None)
            state_path = training_state_dir / "best_model_state.pt"
            metadata_path = training_state_dir / "best_model_state.json"
            if select_best_checkpoint and state_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                evaluated_epochs = {int(round(float(item.get("epoch", 0))))
                                    for item in state.log_history or []
                                    if item.get(f"eval_{checkpoint_selection_metric}") is not None}
                if int(metadata["epoch"]) in evaluated_epochs:
                    best_state.update({
                        "metric": float(metadata["metric"]),
                        "accuracy": metadata.get("accuracy"),
                        "balanced_accuracy": metadata.get("balanced_accuracy"),
                        "epoch": int(metadata["epoch"]),
                        "model_state": torch.load(state_path, map_location="cpu", weights_only=True),
                    })
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            best_state["last_evaluated_epoch"] = int(round(float(state.epoch or 0)))
            metric_value = (metrics or {}).get(f"eval_{checkpoint_selection_metric}")
            if select_best_checkpoint and metric_value is not None and float(metric_value) > best_state["metric"]:
                best_state["metric"] = float(metric_value)
                best_state["accuracy"] = (metrics or {}).get("eval_accuracy")
                best_state["balanced_accuracy"] = (metrics or {}).get("eval_balanced_accuracy")
                best_state["epoch"] = int(round(float(state.epoch or 0)))
                snapshot_started = time.perf_counter()
                best_state["model_state"] = _snapshot_state_dict_cpu(model)
                best_state["snapshot_seconds"] += time.perf_counter() - snapshot_started
                _atomic_torch_save(torch, best_state["model_state"],
                                   training_state_dir / "best_model_state.pt")
                _write_json_atomic(training_state_dir / "best_model_state.json", {
                    "metric": best_state["metric"], "accuracy": best_state["accuracy"],
                    "balanced_accuracy": best_state["balanced_accuracy"],
                    "epoch": best_state["epoch"],
                })
            if save_epoch_checkpoints:
                epoch = int(round(float(state.epoch or 0)))
                epoch_dir = out / f"epoch-{epoch:02d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                save_started = time.perf_counter()
                model.save_pretrained(epoch_dir, safe_serialization=True)
                tokenizer.save_pretrained(epoch_dir)
                (epoch_dir / "validation_metrics.json").write_text(
                    json.dumps(metrics or {}, indent=2) + "\n")
                self.epoch_checkpoints.append(str(epoch_dir))
                self.save_seconds += time.perf_counter() - save_started
            return control

    encoder_steps_per_epoch = _optimizer_steps_per_epoch(
        len(train), effective_batch_size, gradient_accumulation_steps)
    live_callback = _live_training_progress_callback(
        TrainerCallback, out, steps_per_epoch=encoder_steps_per_epoch, epochs=epochs,
        physical_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mean_tokens_per_example=(float(np.mean(train.lengths)) if train.lengths else None),
        torch_module=torch)
    warmup_basis_epochs = min(int(epochs), int(warmup_epochs)) if warmup_epochs is not None else int(epochs)
    encoder_warmup_steps = math.ceil(float(warmup_ratio) * warmup_basis_epochs * encoder_steps_per_epoch)
    args = tr.TrainingArguments(output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch_size, per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate,
        optim=optimizer, weight_decay=float(weight_decay), warmup_steps=encoder_warmup_steps,
        lr_scheduler_type=("constant" if learning_rate_decay is not None else (lr_scheduler_type or "linear")),
        seed=seed, data_seed=seed, save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch" if validation_labels is not None else "no", report_to=[],
        logging_strategy="epoch", fp16=use_fp16, bf16=use_bf16,
        dataloader_num_workers=0, **_length_grouping_args(tr))

    trainer_type = OrderedValidationTrainer
    if learning_rate_decay is not None:
        decay = float(learning_rate_decay)

        class EpochDecayTrainer(OrderedValidationTrainer):
            def create_scheduler(self, num_training_steps, optimizer=None):
                if self.lr_scheduler is None:
                    optimizer = optimizer or self.optimizer
                    steps_per_epoch = encoder_steps_per_epoch
                    warmup_steps = self.args.get_warmup_steps(num_training_steps)
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=_epoch_decay_lambda(
                            decay=decay, steps_per_epoch=steps_per_epoch, warmup_steps=warmup_steps))
                return self.lr_scheduler

        trainer_type = EpochDecayTrainer

    callbacks = [live_callback, epoch_prediction_callback]
    if training_views is not None:
        class EpochTrainingViewCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                epoch = int(round(float(state.epoch or 0)))
                train.set_epoch(epoch)
                return control
        callbacks.append(EpochTrainingViewCallback())
    early_stopping_callback = None
    best_checkpoint_callback = BestCheckpointCallback()
    if validation_labels is not None:
        early_stopping_callback = _accuracy_early_stopping_callback(
            TrainerCallback, early_stopping_patience, metric=checkpoint_selection_metric)
        callbacks.append(early_stopping_callback)
        callbacks.append(best_checkpoint_callback)
    trainer = trainer_type(model=model, args=args, train_dataset=train, data_collator=data_collator,
        eval_dataset=val if validation_labels is not None else None,
        compute_metrics=compute_metrics if validation_labels is not None else None,
        processing_class=tokenizer, callbacks=callbacks)
    # Trainer's length-grouped prediction sampler can return predictions in a
    # different row order from the supplied text list. Use a length-grouped loader
    # for baseline and final validation scores so outputs remain
    # aligned with their original ids and labels.
    prediction_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(prediction_device)
    prediction_dtype = (torch.bfloat16 if use_bf16 else
                       torch.float16 if use_fp16 else None)

    inference_stats = []
    def predict_cached(dataset):
        scores, stats = _predict_cached_dataset(model, dataset,
            batch_size=requested_inference_batch_size, data_collator=data_collator,
            device=prediction_device, amp_dtype=prediction_dtype, return_info=True)
        inference_stats.append(stats)
        return scores, stats

    recovery_context["phase"] = "inference"
    baseline_validation_started = time.perf_counter()
    baseline_logits, baseline_inference_stats = predict_cached(val)
    baseline_validation_seconds = time.perf_counter() - baseline_validation_started
    baseline_predictions = np.asarray(classes)[baseline_logits.argmax(axis=1)]
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
    baseline_metrics = {
        "accuracy": float(accuracy_score(validation_labels, baseline_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(validation_labels, baseline_predictions)),
        "validation_loss": float(log_loss(validation_labels, softmax(baseline_logits, axis=1), labels=classes)),
        "checkpoint": str(resume_checkpoint or checkpoint),
    }
    (out / "baseline_validation.json").write_text(json.dumps(baseline_metrics, indent=2) + "\n")
    live_callback.set_baseline(baseline_metrics)
    startup_seconds = max(0.0, time.perf_counter() - start - baseline_validation_seconds)
    recovery_context["phase"] = "training"
    training_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=_trainer_resume_checkpoint)
    recovery_context["phase"] = "inference"
    training_seconds = time.perf_counter() - training_started
    post_training_started = time.perf_counter()
    _remove_trainer_recovery_files(out)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir)/"training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    best_checkpoint_restoration_seconds = 0.0
    if select_best_checkpoint:
        if best_state["epoch"] is None:
            raise RuntimeError("best checkpoint selection requested but no epoch validation metric was recorded")
        if _should_restore_selected_epoch(best_state["epoch"], best_state["last_evaluated_epoch"]):
            restore_started = time.perf_counter()
            _restore_model_state(model, best_state["model_state"])
            best_checkpoint_restoration_seconds = time.perf_counter() - restore_started
        (out / "checkpoint_selection.json").write_text(json.dumps({
            "metric": f"validation_{checkpoint_selection_metric}", "selected_epoch": best_state["epoch"],
            "selected_validation_accuracy": best_state["accuracy"],
            "selected_validation_balanced_accuracy": best_state["balanced_accuracy"]}, indent=2) + "\n")
    selected_prediction_epoch = (best_state["epoch"] if select_best_checkpoint else
        max((int(item["epoch"]) for item in epoch_prediction_callback.predictions), default=None))
    selected_prediction = _selected_epoch_scores(epoch_prediction_callback.predictions,
        selected_prediction_epoch, (len(val), len(classes)))
    final_validation_reused_epoch_predictions = selected_prediction is not None
    pred_start = time.perf_counter()
    if final_validation_reused_epoch_predictions:
        logits = selected_prediction
        final_inference_stats = None
    else:
        logits, final_inference_stats = predict_cached(val)
    inference_seconds = (0.0 if final_validation_reused_epoch_predictions
                         else time.perf_counter() - pred_start)
    checkpoint_started = time.perf_counter()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    checkpoint_save_seconds = time.perf_counter() - checkpoint_started
    if not np.isfinite(logits).all(): raise ValueError("encoder produced non-finite scores")
    probs = softmax(logits, axis=1)
    epoch_history = [dict(entry) for entry in trainer.state.log_history if "eval_accuracy" in entry]
    for entry in epoch_history:
        if entry.get("eval_loss") is not None:
            entry["validation_loss_raw"] = float(entry["eval_loss"])
    result_epoch = (best_state["epoch"] if select_best_checkpoint else
                    max((int(round(float(item.get("epoch", 0)))) for item in epoch_history), default=None))
    selected_epoch_metrics = next((item for item in epoch_history
        if int(round(float(item.get("epoch", 0)))) == result_epoch), {})
    validation_loss_raw = selected_epoch_metrics.get("eval_loss")
    if validation_loss_raw is None:
        validation_loss_raw = log_loss(validation_labels, softmax(logits, axis=1), labels=classes)
    if learning_rate_decay is not None:
        for entry in epoch_history:
            epoch = int(round(float(entry.get("epoch", 1))))
            entry["learning_rate_used"] = float(learning_rate) * float(learning_rate_decay) ** (epoch - 1)
    effective_post_inference_batch_size = min(
        (item["effective_inference_batch_size"] for item in inference_stats),
        default=requested_inference_batch_size)
    return {"scores": logits, "probabilities": probs, "model": model, "tokenizer": tokenizer,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "selected_physical_batch_size": effective_batch_size,
            "effective_optimizer_batch_size": effective_batch_size * int(gradient_accumulation_steps),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "gradient_checkpointing_enabled": checkpointing_enabled,
            "requested_inference_batch_size": requested_inference_batch_size,
            "effective_inference_batch_size": effective_post_inference_batch_size,
            "batch_size_probe": batch_probe,
            "fit_seconds": time.perf_counter() - start, "inference_seconds": inference_seconds,
            "classes": list(classes),
            "baseline_validation_metrics": baseline_metrics,
            "validation_metrics_by_epoch": epoch_history,
            "validation_predictions_by_epoch": epoch_prediction_callback.predictions,
            "validation_loss_raw": float(validation_loss_raw),
            "best_epoch": best_state["epoch"],
            "best_validation_accuracy": best_state["accuracy"] if best_state["epoch"] is not None else None,
            "best_validation_balanced_accuracy": best_state["balanced_accuracy"] if best_state["epoch"] is not None else None,
            "checkpoint_scoring_method": "best_epoch_validation_predictions" if final_validation_reused_epoch_predictions else "length_grouped_restored_model_forward",
            "final_validation_reused_epoch_predictions": bool(final_validation_reused_epoch_predictions),
            "runtime_breakdown": {"startup_seconds": startup_seconds,
                "baseline_validation_seconds": baseline_validation_seconds,
                "training_including_validation_seconds": training_seconds,
                "validation_seconds": float(getattr(trainer, "validation_seconds", 0.0)),
                "epoch_validation_oom_fallbacks": int(getattr(trainer, "eval_oom_fallbacks", 0)),
                "effective_epoch_eval_batch_size": int(getattr(trainer, "effective_eval_batch_size",
                    trainer.args.per_device_eval_batch_size)),
                "best_checkpoint_restoration_seconds": float(best_checkpoint_restoration_seconds),
                "final_validation_inference_seconds": float(inference_seconds),
                "final_validation_rows_per_second": (final_inference_stats or {}).get("inference_rows_per_second"),
                "effective_inference_batch_size": effective_post_inference_batch_size,
                "inference_oom_fallbacks": sum(x.get("inference_oom_fallbacks", 0) for x in inference_stats),
                "checkpoint_serialization_seconds": float(checkpoint_save_seconds + best_checkpoint_callback.save_seconds),
                "best_checkpoint_state_snapshot_seconds": float(best_state["snapshot_seconds"]),
                "training_excluding_validation_seconds": max(0.0, training_seconds -
                    float(getattr(trainer, "validation_seconds", 0.0))),
                "model_post_training_seconds": float(time.perf_counter() - post_training_started)},
            "early_stopping": (_early_stopping_summary(early_stopping_callback, early_stopping_patience, epochs,
                                                        metric=checkpoint_selection_metric)
                               if early_stopping_callback is not None else None),
            "checkpoint_selection_metric": f"validation_{checkpoint_selection_metric}" if select_best_checkpoint else None,
            "best_checkpoint_path": str(out) if best_state["epoch"] is not None else None,
            "selected_checkpoint_path": str(out),
            "training_view_sampling": (train.sampling_summary() if training_views is not None else None),
            "epoch_checkpoints": list(best_checkpoint_callback.epoch_checkpoints),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None}


class _TextDataset:
    def __new__(cls, texts, labels, tokenizer, max_length, *, training_views=None,
                training_view_ids=None, seed=2026):
        torch, _ = _require_neural()
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):
                self.texts = list(texts)
                self.views = ([list(map(str, row)) for row in training_views]
                              if training_views is not None else [[str(text)] for text in texts])
                if len(self.views) != len(self.texts) or any(not row for row in self.views):
                    raise ValueError("training views must contain at least one view per original row")
                self.view_ids = ([list(map(str, row)) for row in training_view_ids]
                                 if training_view_ids is not None else
                                 [["full", *[f"chunk-{i:02d}" for i in range(1, len(row))]]
                                  for row in self.views])
                if any(len(views) != len(ids) for views, ids in zip(self.views, self.view_ids)):
                    raise ValueError("training view IDs must align with training views")
                self.seed = int(seed)
                self.enc = None
                self.lengths = []
                self.labels = labels
                self.epoch_records = []
                self.set_epoch(0)
            def set_epoch(self, epoch, force=False):
                epoch = int(epoch)
                self.epoch = getattr(self, "epoch", None)
                if not force and self.epoch == epoch:
                    return
                selected, counts = select_epoch_view_indices(
                    [len(views) for views in self.views], seed=self.seed, epoch=epoch,
                    view_ids=self.view_ids)
                selected_texts = [views[index] for views, index in zip(self.views, selected)]
                self.enc = tokenizer(selected_texts, truncation=True, max_length=max_length, padding=False)
                self.lengths = [len(ids) for ids in self.enc["input_ids"]]
                self.epoch = epoch
                record = {"epoch": epoch + 1, "original_responses": len(selected),
                          "selected_view_counts": counts}
                self.epoch_records = [item for item in self.epoch_records if item["epoch"] != epoch + 1]
                self.epoch_records.append(record)
                self.epoch_records.sort(key=lambda item: item["epoch"])
            def set_probe_views(self):
                # Probe using full responses, the longest raw-text view; the
                # configured max_length is already computed across every view.
                selected_texts = [views[0] for views in self.views]
                self.enc = tokenizer(selected_texts, truncation=True, max_length=max_length, padding=False)
                self.lengths = [len(ids) for ids in self.enc["input_ids"]]
            def restore_epoch_views(self, epoch=0):
                self.set_epoch(epoch, force=True)
            def sampling_summary(self):
                return {"policy": "uniform one-view-per-original-response per epoch",
                        "seed": self.seed, "choices": "complete response plus available chunks",
                        "epoch_records": list(self.epoch_records)}
            def __len__(self): return len(self.texts)
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.enc.items()}
                if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        return Dataset()


def _probe_max_batch_size(model, dataset, *, requested_batch_size, max_batch_size,
                         safety_fraction=0.92, use_bf16=False, use_fp16=False,
                         optimizer_name="adamw_torch", weight_decay=0.0,
                         runtime_overhead_reserve_bytes=0):
    """Find and benchmark the largest safe integer batch under the configured mode."""
    torch, _ = _require_neural()
    requested = int(requested_batch_size)
    maximum = max(1, min(int(max_batch_size or requested), len(dataset)))
    report = {"enabled": True, "max_batch_size": maximum, "attempts": []}
    if not torch.cuda.is_available():
        report.update({"status": "gpu_unavailable", "selected_batch_size": requested})
        return requested, report
    device = torch.device("cuda", torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    base_reserved = int(torch.cuda.memory_reserved(device))
    runtime_overhead_reserve_bytes = max(0, int(runtime_overhead_reserve_bytes or 0))
    max_allowed, reserve_bytes = _batch_probe_memory_limit(
        base_reserved, free_bytes, safety_fraction, runtime_overhead_reserve_bytes)
    report.update({"status": "probing", "device": str(device),
                   "free_memory_bytes_before_probe": int(free_bytes),
                   "total_memory_bytes": int(total_bytes),
                   "usable_free_vram_fraction": float(safety_fraction),
                   "reserved_free_vram_fraction": 1.0 - float(safety_fraction),
                   "memory_reserve_bytes": reserve_bytes,
                   "runtime_overhead_reserve_bytes": runtime_overhead_reserve_bytes,
                   "memory_limit_reserved_bytes": max_allowed,
                   "baseline_reserved_bytes": base_reserved})
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    dataset_lengths = getattr(dataset, "lengths", None)
    probe_order = (sorted(range(len(dataset)), key=lambda idx: dataset_lengths[idx], reverse=True)
                   if dataset_lengths is not None else list(range(len(dataset))))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    checkpointing_target = _gradient_checkpointing_target(model)
    checkpointing_requested = bool(getattr(checkpointing_target, "is_gradient_checkpointing", False))
    # Keep recovery copies on CPU so the probe does not consume VRAM with a
    # second copy of every trainable parameter before measuring batch capacity.
    saved_trainable = [parameter.detach().to(device="cpu", copy=True) for parameter in trainable]
    # Trainer initializes AdamW moments lazily on its first update. Simulate a
    # full-precision AdamW update during probing so the selected batch includes
    # that persistent memory. This is conservative for lower-memory optimizers.
    optimizer_probe = str(optimizer_name)

    def make_probe_optimizer():
        name = str(optimizer_name).lower()
        if name in {"paged_adamw_8bit", "adamw_bnb_8bit"}:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise RuntimeError(f"configured optimizer {name} requires bitsandbytes") from exc
            return bnb.optim.PagedAdamW8bit(trainable, lr=1e-4, weight_decay=float(weight_decay))
        if name in {"adamw_torch", "adamw", "adamw_torch_fused"}:
            cls = torch.optim.AdamW
            extra = {"fused": True} if name == "adamw_torch_fused" else {}
            return cls(trainable, lr=1e-4, weight_decay=float(weight_decay), **extra)
        if name == "adam":
            return torch.optim.Adam(trainable, lr=1e-4, weight_decay=float(weight_decay))
        if name == "sgd":
            return torch.optim.SGD(trainable, lr=1e-4, weight_decay=float(weight_decay))
        raise ValueError(f"unsupported optimizer for batch-size probe: {optimizer_name}")

    def attempt(size, *, run_steps=1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        batch = None
        probe_optimizer = None
        checkpointing_enabled = _gradient_checkpointing_for_batch(
            checkpointing_requested, size)
        if checkpointing_requested:
            _set_gradient_checkpointing(checkpointing_target, checkpointing_enabled)
        try:
            # Probe the longest examples first so the chosen batch also fits
            # length-grouped batches at the truncation limit.
            examples = [dataset[i] for i in probe_order[:size]]
            collate_fn = getattr(dataset, "collate_fn", None)
            if collate_fn is not None:
                batch = {key: value.to(device) for key, value in collate_fn(examples).items()}
            else:
                batch = {key: torch.stack([example[key] for example in examples]).to(device)
                         for key in examples[0]}
            probe_optimizer = make_probe_optimizer()
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            durations = []
            actual_tokens = int(batch.get("attention_mask", batch["input_ids"]).sum().item())
            padded_tokens = int(batch["input_ids"].numel())
            for _ in range(max(1, int(run_steps))):
                torch.cuda.synchronize(device)
                step_started = time.perf_counter()
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16 or use_fp16):
                    loss = model(**batch).loss
                if loss is None:
                    raise RuntimeError("batch size probe needs labeled training examples")
                loss.backward()
                probe_optimizer.step()
                torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - step_started)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            fits_headroom = peak_reserved <= max_allowed
            measured_durations = durations[1:] if len(durations) > 1 else durations
            measured_seconds = max(sum(measured_durations), 1e-9)
            measured_steps = len(measured_durations)
            baseline_used = int(total_bytes) - int(free_bytes)
            estimated_peak_used = baseline_used + max(0, peak_reserved - base_reserved)
            record = {"batch_size": size, "fits": fits_headroom,
                      "gradient_checkpointing": checkpointing_enabled,
                      "peak_allocated_bytes": peak_allocated,
                      "peak_reserved_bytes": peak_reserved,
                      "estimated_peak_gpu_used_bytes": estimated_peak_used,
                      "peak_vram_utilization": min(1.0, estimated_peak_used / max(1, int(total_bytes))),
                      "padded_tokens_per_batch": padded_tokens,
                      "tokens_per_batch": actual_tokens,
                      "benchmark_steps": measured_steps,
                      "gpu_processing_seconds": measured_seconds,
                      "examples_per_second": size * measured_steps / measured_seconds,
                      "tokens_per_second": actual_tokens * measured_steps / measured_seconds,
                      "reason": None if fits_headroom else "memory_reserve"}
            report["attempts"].append(record)
            return fits_headroom
        except RuntimeError as exc:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            is_oom = (oom_type is not None and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            try:
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            except RuntimeError:
                peak_allocated = peak_reserved = None
            report["attempts"].append({"batch_size": size, "fits": False,
                "gradient_checkpointing": checkpointing_enabled,
                "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                "reason": "cuda_oom"})
            return False
        finally:
            model.zero_grad(set_to_none=True)
            if probe_optimizer is not None:
                del probe_optimizer
            with torch.no_grad():
                for parameter, saved in zip(trainable, saved_trainable):
                    parameter.copy_(saved)
            del batch
            torch.cuda.empty_cache()

    try:
        # BitsAndBytes models are dispatched at load time and cannot be moved
        # later with nn.Module.to().
        if not getattr(model, "_qlora_quantized", False):
            model.to(device)
        model.train()
        config = getattr(model, "config", None)
        identity = (getattr(model, "_batch_probe_model_id", None) or
                    getattr(config, "_name_or_path", None) or model.__class__.__qualname__)
        cache_payload = {"model": identity, "model_class": model.__class__.__qualname__,
            "gpu": torch.cuda.get_device_name(device), "device": str(device),
            "total_memory_bytes": int(total_bytes),
            "free_memory_bucket_bytes": int(free_bytes // (256 * 1024**2) * (256 * 1024**2)),
            "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
            "sequence_length": max(dataset_lengths) if dataset_lengths else None,
            "optimizer": str(optimizer_name), "gradient_checkpointing": bool(
                getattr(getattr(model, "backbone", model), "is_gradient_checkpointing", False)),
            "weight_decay": float(weight_decay),
            "usable_free_vram_fraction": float(safety_fraction),
            "runtime_overhead_reserve_bytes": runtime_overhead_reserve_bytes,
            "throughput_benchmark_steps": 3,
            "max_batch_size": maximum,
            "batch_probe_algorithm_version": 2,
            "checkpointing_policy": "configured_for_all_batch_sizes"}
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
        cache_path = Path(os.environ.get("NLM_BATCH_SIZE_CACHE", "artifacts/cache/batch_sizes")) / f"{cache_key}.json"
        cached_size = None
        cached_max_safe = None
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                candidate_size = int(cached.get("selected_batch_size", 0))
                if cached.get("cache_key") == cache_key and 0 < candidate_size <= maximum:
                    cached_size = candidate_size
                    candidate_max = int(cached.get("max_safe_batch_size", 0))
                    if candidate_size <= candidate_max <= maximum:
                        cached_max_safe = candidate_max
            except (OSError, ValueError, json.JSONDecodeError):
                cached_size = None
        cache_validated = (cached_size is not None and cached_max_safe is not None
                           and attempt(cached_max_safe))
        if cache_validated:
            maximum_safe = cached_max_safe
            report["cache_validation_passes"] = 1
        else:
            maximum_safe = None
        if cached_size is not None and maximum_safe is None:
            cache_path.unlink(missing_ok=True)
        if maximum_safe is None:
            maximum_safe = _largest_safe_batch_size(maximum, attempt)
        one_step_maximum_safe = maximum_safe
        benchmark_results = []
        selected = maximum_safe
        while selected > 0:
            before = len(report["attempts"])
            if attempt(selected, run_steps=3):
                benchmark_results.append(report["attempts"][-1])
                break
            if len(report["attempts"]) == before:
                break
            selected -= 1
        if not benchmark_results:
            raise MemoryError("no batch size passed the repeated-step safety check")
        # Throughput at smaller sizes is diagnostic only. Never discard the
        # largest safe microbatch because a short benchmark was noisy.
        maximum_safe = selected
        selected, selected_metrics, throughput_fallback = _select_probed_batch_size(
            maximum_safe, benchmark_results, report["attempts"])
        report.update({"status": ("safe_maximum_fallback" if throughput_fallback else
                                  "cached_validated" if cache_validated else "selected"),
                       "selection_policy": "largest_verified_safe_batch",
                       "boundary_backoff": "one_at_a_time",
                       "throughput_benchmark_fallback": throughput_fallback,
                       "max_safe_batch_size": one_step_maximum_safe,
                       "repeated_step_safe_batch_size": maximum_safe,
                       "selected_batch_size": selected,
                       "selected_physical_batch_size": selected,
                       "effective_optimizer_batch_size": None,
                       "throughput_candidates": benchmark_results,
                       "peak_probe_allocated_bytes": max(
                           (x.get("peak_allocated_bytes") or 0 for x in report["attempts"]), default=0),
                       "peak_probe_reserved_bytes": max(
                           (x.get("peak_reserved_bytes") or 0 for x in report["attempts"]), default=0),
                       "selected_examples_per_second": selected_metrics["examples_per_second"],
                       "selected_tokens_per_second": selected_metrics["tokens_per_second"],
                       "selected_padded_tokens_per_batch": selected_metrics["padded_tokens_per_batch"],
                       "selected_peak_vram_utilization": selected_metrics["peak_vram_utilization"],
                       "cache_key": cache_key,
                       "configured_optimizer": str(optimizer_name),
                       "probe_optimizer": optimizer_probe,
                       "optimizer_state_included": True})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"cache_key": cache_key,
            "selected_batch_size": selected, "max_safe_batch_size": maximum_safe,
            "throughput_candidates": benchmark_results, "config": cache_payload}, indent=2) + "\n")
        return selected, report
    finally:
        model.zero_grad(set_to_none=True)
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.empty_cache()


def _probe_lora_batch_size(model, dataset, *, requested_batch_size, max_batch_size,
                           safety_fraction, use_bf16, use_fp16, optimizer_name,
                           weight_decay, gradient_checkpointing_enabled):
    """Probe once in the already configured LoRA checkpointing/optimizer mode."""
    selected, report = _probe_max_batch_size(model, dataset,
        requested_batch_size=requested_batch_size, max_batch_size=max_batch_size,
        safety_fraction=safety_fraction, use_bf16=use_bf16, use_fp16=use_fp16,
        optimizer_name=optimizer_name, weight_decay=weight_decay)
    report["gradient_checkpointing_probe_mode"] = bool(gradient_checkpointing_enabled)
    report["selected_batch_size"] = int(selected)
    return int(selected), report


def load_text_only_backbone(transformers_module, checkpoint, revision, *, torch_dtype,
                            local_files_only=True, trust_remote_code=False, **model_kwargs):
    """Load a multimodal checkpoint and retain its text tower only."""
    composite = transformers_module.AutoModel.from_pretrained(
        checkpoint, revision=revision, torch_dtype=torch_dtype, local_files_only=local_files_only,
        trust_remote_code=trust_remote_code, **model_kwargs)
    text_backbone = getattr(composite, "language_model", None)
    if text_backbone is None:
        raise ValueError(f"{checkpoint} has no language_model text backbone")
    del composite
    return text_backbone


def _require_flash_attention_2_runtime(torch):
    """Fail closed unless the installed FlashAttention-2 kernel runs on this GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("flash_attention_2 requires CUDA")
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError("flash_attention_2 was requested but flash-attn is not installed") from exc
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"flash_attention_2 needs SM80 or newer; this GPU is SM{major}{minor}")
    try:
        q = torch.zeros((1, 8, 1, 64), device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q, causal=True)
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"installed flash-attn has no working kernel for this GPU/CUDA stack: {exc}"
        ) from exc


@_with_cuda_oom_recovery
def train_qwen_lora_fold(train_text, y_train, validation_text, classes, *, checkpoint="Qwen/Qwen3-0.6B",
                         revision, output_dir="artifacts/checkpoints/qwen", epochs=20, batch_size=4,
                         inference_batch_size=None,
                         max_length=384, learning_rate=1e-4, seed=2026, gradient_accumulation_steps=1,
                         target_effective_batch_size=None,
                         lora_rank=16, lora_alpha=32,
                         lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                         resume_checkpoint=None, validation_labels=None, gradient_checkpointing=False,
                         local_files_only=False, learning_rate_decay=1.0, auto_batch_size=False,
                         max_batch_size=1024, batch_probe_safety_fraction=0.95,
                         readout_hidden_size=None, task_prefix="", fix_mistral_regex=False,
                         lora_exclude_modules=None, text_only_backbone=False,
                         lr_scheduler_type=None, warmup_ratio=0.0,
                         warmup_epochs=None, optimizer="adamw_torch", weight_decay=0.0,
                         early_stopping_patience=2, trust_remote_code=False,
                         quantization="none", attention_backend=None,
                         checkpoint_selection_metric="balanced_accuracy", save_epoch_checkpoints=False,
                         training_views=None, training_view_ids=None,
                         _oom_recovery_context=None, _trainer_resume_checkpoint=None):
    """LoRA fine-tune causal hidden states with masked-mean classifier readout."""
    _force_huggingface_offline()
    recovery_context = _oom_recovery_context if _oom_recovery_context is not None else {"phase": "initialization"}
    recovery_context["phase"] = "initialization"
    torch, tr = _require_neural()
    try:
        from peft import LoraConfig, TaskType, PeftModel, get_peft_model
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError("Qwen LoRA training needs the project's optional neural dependencies") from exc
    # Training is deliberately cache-only, even if a caller supplies a false config value.
    local_files_only = True
    from torch import nn
    from torch.utils.data import Dataset, Sampler
    from transformers.modeling_outputs import SequenceClassifierOutput
    start = time.perf_counter(); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    tokenizer_kwargs = {"revision": revision, "local_files_only": local_files_only,
                        "trust_remote_code": trust_remote_code}
    if fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = tr.AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    quantization = str(quantization or "none").lower()
    if quantization not in {"none", "qlora4"}:
        raise ValueError("quantization must be 'none' or 'qlora4'")
    if attention_backend not in {None, "sdpa", "flash_attention_2"}:
        raise ValueError("attention_backend must be 'sdpa', 'flash_attention_2', or None")
    checkpointing_requested = bool(gradient_checkpointing)
    # Probe in the configured mode. A batch size measured with checkpointing
    # disabled is not valid for the configured training recipe.
    checkpointing_disabled_for_batch = False
    load_kwargs = {"attn_implementation": attention_backend} if attention_backend else {}
    quantization_config = None
    if attention_backend == "flash_attention_2":
        _require_flash_attention_2_runtime(torch)
    if quantization == "qlora4":
        if not torch.cuda.is_available():
            raise RuntimeError("qlora4 requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("qlora4 requires BF16-capable CUDA hardware")
        from transformers import BitsAndBytesConfig
        quantization_config = {"load_in_4bit": True,
                               "bnb_4bit_quant_type": "nf4",
                               "bnb_4bit_use_double_quant": True,
                               "bnb_4bit_compute_dtype": "bfloat16"}
        load_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": {"": torch.cuda.current_device()},
        })
    if text_only_backbone:
        backbone = load_text_only_backbone(tr, checkpoint, revision, torch_dtype=model_dtype,
                                           local_files_only=local_files_only,
                                           trust_remote_code=trust_remote_code, **load_kwargs)
    else:
        backbone = tr.AutoModel.from_pretrained(checkpoint, revision=revision, torch_dtype=model_dtype,
                                                 local_files_only=local_files_only,
                                                 trust_remote_code=trust_remote_code, **load_kwargs)
    if quantization == "qlora4":
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("QLoRA requires PEFT prepare_model_for_kbit_training") from exc
        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=bool(gradient_checkpointing))
    if resume_checkpoint:
        backbone = PeftModel.from_pretrained(backbone, resume_checkpoint, is_trainable=True, local_files_only=True)
    else:
        lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_rank, lora_alpha=lora_alpha,
                          lora_dropout=.05, target_modules=list(lora_target_modules),
                          exclude_modules=lora_exclude_modules)
        backbone = get_peft_model(backbone, lora)
    backbone.config.use_cache = False
    if gradient_checkpointing and quantization != "qlora4":
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    label_map = {label: i for i, label in enumerate(classes)}
    config = backbone.config
    text_config = getattr(config, "text_config", None)
    hidden_size = (int(readout_hidden_size) if readout_hidden_size is not None else
                   getattr(config, "hidden_size", None) or
                   getattr(text_config, "hidden_size", None))
    if hidden_size is None:
        raise ValueError(f"{checkpoint} needs a configured readout_hidden_size")
    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone
            self.classifier=nn.Linear(hidden_size,len(classes))
            if quantization == "qlora4":
                backbone_device = next(backbone.parameters()).device
                self.classifier.to(backbone_device)
                self.hf_device_map = getattr(backbone, "hf_device_map", {"": backbone_device.index or 0})
                self._qlora_quantized = True
        def forward(self,input_ids,attention_mask,labels=None):
            hidden=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            logits=self.classifier(pooled.float())
            if labels is not None:
                labels = labels.to(logits.device)
            loss=nn.functional.cross_entropy(logits,labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss,logits=logits)
    class Rows(Dataset):
        def __init__(self,texts,ys=None,view_texts=None,view_ids=None):
            self.texts=list(map(str,texts));self.ys=ys
            self.views=([list(map(str,row)) for row in view_texts]
                        if view_texts is not None else [[text] for text in self.texts])
            if len(self.views)!=len(self.texts) or any(not row for row in self.views):
                raise ValueError("training views must contain at least one view per original row")
            self.view_ids=([list(map(str,row)) for row in view_ids]
                           if view_ids is not None else
                           [["full",*[f"chunk-{i:02d}" for i in range(1,len(row))]] for row in self.views])
            if any(len(views)!=len(ids) for views,ids in zip(self.views,self.view_ids)):
                raise ValueError("training view IDs must align with training views")
            self.epoch=None;self.epoch_records=[];self.enc={};self.lengths=[]
            self.set_epoch(0)
        def set_epoch(self,epoch,force=False):
            epoch=int(epoch)
            if not force and self.epoch==epoch:return
            selected,counts=select_epoch_view_indices([len(row) for row in self.views],
                seed=seed,epoch=epoch,view_ids=self.view_ids)
            prepared=[str(task_prefix)+views[choice] for views,choice in zip(self.views,selected)]
            encoded=tokenizer(prepared,truncation=True,max_length=max_length,padding=False)
            self.enc={k:v for k,v in encoded.items() if k in {"input_ids","attention_mask"}}
            self.lengths=[len(ids) for ids in self.enc["input_ids"]]
            record={"epoch":epoch+1,"original_responses":len(selected),"selected_view_counts":counts}
            self.epoch_records=[item for item in self.epoch_records if item["epoch"]!=epoch+1]
            self.epoch_records.append(record);self.epoch_records.sort(key=lambda item:item["epoch"])
            self.epoch=epoch
        def set_probe_views(self):
            prepared=[str(task_prefix)+views[0] for views in self.views]
            encoded=tokenizer(prepared,truncation=True,max_length=max_length,padding=False)
            self.enc={k:v for k,v in encoded.items() if k in {"input_ids","attention_mask"}}
            self.lengths=[len(ids) for ids in self.enc["input_ids"]]
            self.epoch=None
        def sampling_summary(self):
            return {"policy":"uniform one-view-per-original-response per epoch","seed":int(seed),
                    "choices":"complete response plus available chunks","epoch_records":list(self.epoch_records)}
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self,i):
            item={k:torch.tensor(v[i],dtype=torch.long) for k,v in self.enc.items()}
            if self.ys is not None: item["labels"]=torch.tensor(label_map[self.ys[i]],dtype=torch.long)
            return item
    model=Head()
    model._batch_probe_model_id = f"{checkpoint}@{revision}:{quantization}"
    if resume_checkpoint:
        classifier_path = Path(resume_checkpoint)/"classifier.pt"
        if not classifier_path.is_file(): raise FileNotFoundError(f"continuation checkpoint has no classifier head: {classifier_path}")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=True))
    captured_validation_prediction = {}
    def compute_metrics(eval_prediction):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        logits = _restore_prediction_order(logits, validation_order)
        captured_validation_prediction["logits"] = np.asarray(logits, dtype=np.float32).copy()
        labels = _restore_prediction_order(eval_prediction.label_ids, validation_order)
        predicted = np.asarray(logits).argmax(axis=1)
        return {"accuracy": float(accuracy_score(labels, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted))}
    out=Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    training_state_dir = out / "training_state"
    _copy_training_recovery_state(_trainer_resume_checkpoint, out)
    checkpoint_metadata = {
        "checkpoint": checkpoint, "revision": revision,
        "max_length": int(max_length), "task_prefix": str(task_prefix),
        "classes": list(classes), "pooling": "attention-mask mean",
        "quantization": quantization, "quantization_config": quantization_config,
        "attention_backend": attention_backend,
        "lora_rank": int(lora_rank), "lora_alpha": int(lora_alpha),
        "lora_target_modules": list(lora_target_modules),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "training_view_sampling_policy": ("uniform one-view-per-original-response per epoch"
                                           if training_views is not None else None),
    }
    class EpochSnapshotCallback(TrainerCallback):
        def __init__(self):
            self.checkpoints=[str(path) for path in sorted(out.glob("epoch-*")) if path.is_dir()]
            self.validation_predictions=[]; self.best_state=None
            self.save_seconds=0.0; self.snapshot_seconds=0.0
            for path in sorted(training_state_dir.glob("validation_epoch_*.npy")):
                try:
                    epoch = int(path.stem.rsplit("_", 1)[1])
                    self.validation_predictions.append({"epoch": epoch,
                                                        "scores": np.load(path, allow_pickle=False)})
                except (OSError, ValueError):
                    continue

        def on_train_begin(self,args,state,control,**kwargs):
            evaluated_epochs = {int(round(float(item.get("epoch", 0))))
                                for item in state.log_history or []
                                if item.get(f"eval_{checkpoint_selection_metric}") is not None}
            self.validation_predictions = [item for item in self.validation_predictions
                                           if int(item["epoch"]) in evaluated_epochs]
            state_path = training_state_dir / "best_qwen_state.pt"
            metadata_path = training_state_dir / "best_qwen_state.json"
            if state_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                if int(metadata["epoch"]) in evaluated_epochs:
                    saved = torch.load(state_path, map_location="cpu", weights_only=True)
                    self.best_state = {"adapter": saved["adapter"],
                        "classifier": saved["classifier"], "metrics": metadata["metrics"],
                        "epoch": int(metadata["epoch"])}
            return control

        def on_evaluate(self,args,state,control,metrics=None,**kwargs):
            epoch=int(round(state.epoch or len(self.checkpoints)+1))
            if training_views is not None:
                checkpoint_metadata["training_view_sampling"] = train_dataset.sampling_summary()
            logits=captured_validation_prediction.pop("logits",None)
            if logits is not None:
                self.validation_predictions = [item for item in self.validation_predictions
                                               if int(item["epoch"]) != epoch]
                self.validation_predictions.append({"epoch":epoch,"scores":logits})
                training_state_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = training_state_dir / f"validation_epoch_{epoch:04d}.npy"
                temporary_prediction_path = prediction_path.with_name(prediction_path.name + ".tmp")
                with temporary_prediction_path.open("wb") as stream:
                    np.save(stream, logits, allow_pickle=False)
                temporary_prediction_path.replace(prediction_path)
            metric_value = (metrics or {}).get(f"eval_{checkpoint_selection_metric}")
            if metric_value is not None and (self.best_state is None or
                    float(metric_value) > float(self.best_state["metrics"].get(
                        f"eval_{checkpoint_selection_metric}", -float("inf")))):
                snapshot_started = time.perf_counter()
                from peft import get_peft_model_state_dict
                self.best_state = {"adapter": {k:v.detach().cpu().clone() for k,v in
                    get_peft_model_state_dict(backbone).items()},
                    "classifier": {k:v.detach().cpu().clone() for k,v in model.classifier.state_dict().items()},
                    "metrics": dict(metrics or {}), "epoch": epoch}
                self.snapshot_seconds += time.perf_counter() - snapshot_started
                _atomic_torch_save(torch, {
                    "adapter": self.best_state["adapter"],
                    "classifier": self.best_state["classifier"],
                }, training_state_dir / "best_qwen_state.pt")
                _write_json_atomic(training_state_dir / "best_qwen_state.json", {
                    "metrics": self.best_state["metrics"], "epoch": self.best_state["epoch"]})
            if save_epoch_checkpoints:
                save_started = time.perf_counter()
                epoch_dir=out/f"epoch-{epoch:02d}"
                epoch_dir.mkdir(parents=True,exist_ok=True)
                model.backbone.save_pretrained(epoch_dir,safe_serialization=True)
                tokenizer.save_pretrained(epoch_dir)
                torch.save({k:v.detach().cpu() for k,v in model.classifier.state_dict().items()},
                           epoch_dir/"classifier.pt")
                (epoch_dir/"checkpoint_manifest.json").write_text(json.dumps(checkpoint_metadata,indent=2)+"\n")
                (epoch_dir/"validation_metrics.json").write_text(json.dumps(metrics or {},indent=2)+"\n")
                self.checkpoints.append(str(epoch_dir))
                self.save_seconds += time.perf_counter() - save_started
            return control
    snapshot_callback=EpochSnapshotCallback()
    if checkpoint_selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be accuracy or balanced_accuracy")
    eval_each_epoch = validation_labels is not None
    early_stopping_callback = (_accuracy_early_stopping_callback(
        TrainerCallback, early_stopping_patience, metric=checkpoint_selection_metric)
                               if eval_each_epoch else None)
    if not 0 < float(learning_rate_decay) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1]")
    if not 0 <= float(warmup_ratio) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    # Preserve the historical epoch-wise multiplicative schedule when callers
    # explicitly provide its decay. New recipes can select a standard HF schedule.
    scheduler = lr_scheduler_type or "epoch_decay"
    if scheduler not in {"linear", "cosine", "constant", "epoch_decay"}:
        raise ValueError("lr_scheduler_type must be linear, cosine, constant, or epoch_decay")
    train_dataset=Rows(train_text,y_train,training_views,training_view_ids)
    validation_dataset_rows=Rows(validation_text,validation_labels)
    validation_order = _length_order(validation_dataset_rows.lengths)

    class LengthOrderedSampler(Sampler):
        def __init__(self, dataset):
            self.order = _length_order(dataset.lengths).tolist()
        def __iter__(self): return iter(self.order)
        def __len__(self): return len(self.order)
    data_collator=tr.DataCollatorWithPadding(tokenizer=tokenizer,pad_to_multiple_of=8)
    train_dataset.collate_fn=data_collator
    use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16=torch.cuda.is_available() and not use_bf16
    effective_batch_size=int(batch_size)
    requested_inference_batch_size=max(1,int(inference_batch_size or batch_size))
    batch_probe={"enabled": False, "selected_batch_size": effective_batch_size}
    if auto_batch_size:
        if training_views is not None:
            train_dataset.set_probe_views()
        recovery_context["phase"] = "batch_probe"
        effective_batch_size, batch_probe = _probe_lora_batch_size(
            model, train_dataset, requested_batch_size=batch_size,
            max_batch_size=min(int(max_batch_size or batch_size), int(target_effective_batch_size))
                if target_effective_batch_size is not None else max_batch_size or batch_size,
            safety_fraction=batch_probe_safety_fraction, use_bf16=use_bf16, use_fp16=use_fp16,
            optimizer_name=optimizer, weight_decay=weight_decay,
            gradient_checkpointing_enabled=gradient_checkpointing)
        recovery_context["phase"] = "initialization"
        if training_views is not None:
            train_dataset.set_epoch(0, force=True)
    effective_batch_size, gradient_accumulation_steps = _resolve_batch_accumulation(
        effective_batch_size, gradient_accumulation_steps, target_effective_batch_size)
    checkpointing_probe_batch_size = int(effective_batch_size)
    out.mkdir(parents=True,exist_ok=True)
    batch_probe.update({"requested_batch_size":int(batch_size),
                        "effective_batch_size":effective_batch_size,
                        "selected_physical_batch_size":effective_batch_size,
                        "target_effective_batch_size":target_effective_batch_size,
                        "effective_optimizer_batch_size":effective_batch_size*gradient_accumulation_steps,
                        "gradient_accumulation_steps":int(gradient_accumulation_steps),
                        "gradient_checkpointing_enabled":bool(gradient_checkpointing),
                        "gradient_checkpointing_probe_mode":bool(checkpointing_requested),
                        "gradient_checkpointing_disabled_for_batch_threshold":checkpointing_disabled_for_batch,
                        "checkpointing_probe_batch_size":checkpointing_probe_batch_size,
                        "padding_strategy":"dynamic",
                        "pad_to_multiple_of":8})
    (out/"batch_size_probe.json").write_text(json.dumps(batch_probe,indent=2)+"\n")
    print(f"LoRA batch-size selection: {json.dumps(batch_probe,sort_keys=True)}",flush=True)
    steps_per_epoch=_optimizer_steps_per_epoch(
        len(train_dataset),effective_batch_size,gradient_accumulation_steps)
    progress_callback=_live_training_progress_callback(
        TrainerCallback, out, steps_per_epoch=steps_per_epoch, epochs=epochs,
        physical_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mean_tokens_per_example=(float(np.mean(train_dataset.lengths))
                                 if train_dataset.lengths else None),
        torch_module=torch)
    warmup_basis_epochs=min(int(epochs),int(warmup_epochs)) if warmup_epochs is not None else int(epochs)
    warmup_steps=math.ceil(float(warmup_ratio)*steps_per_epoch*warmup_basis_epochs)
    args=tr.TrainingArguments(output_dir=str(output_dir),num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch_size,per_device_eval_batch_size=requested_inference_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,learning_rate=learning_rate,
        lr_scheduler_type="constant" if scheduler == "epoch_decay" else scheduler,
        warmup_steps=warmup_steps, optim=optimizer, weight_decay=float(weight_decay),
        seed=seed,data_seed=seed,save_strategy="epoch",save_total_limit=2,
        eval_strategy="epoch" if eval_each_epoch else "no",report_to=[],
        logging_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,dataloader_num_workers=0,
        **_length_grouping_args(tr))
    class OrderedValidationTrainer(tr.Trainer):
        validation_seconds = 0.0
        eval_oom_fallbacks = 0

        def _save(self, output_dir=None, state_dict=None):
            # Save only trainable PEFT weights and the classifier head in
            # recovery checkpoints. Serializing the frozen QLoRA backbone at
            # every optimizer step would make reliable recovery impractical.
            output = Path(output_dir or self.args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            if self.args.should_save:
                self.model.backbone.save_pretrained(output, safe_serialization=True)
                torch.save({key: value.detach().cpu() for key, value in
                            self.model.classifier.state_dict().items()}, output / "classifier.pt")
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output)
                torch.save(self.args, output / "training_args.bin")
                (output / "checkpoint_manifest.json").write_text(
                    json.dumps(checkpoint_metadata, indent=2) + "\n")

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            model = model or self.model
            safe_adapter = Path(resume_from_checkpoint) / "adapter_model.safetensors"
            binary_adapter = Path(resume_from_checkpoint) / "adapter_model.bin"
            classifier_path = Path(resume_from_checkpoint) / "classifier.pt"
            if not safe_adapter.is_file() and not binary_adapter.is_file():
                return super()._load_from_checkpoint(resume_from_checkpoint, model)
            if safe_adapter.is_file():
                from safetensors.torch import load_file
                adapter_state = load_file(str(safe_adapter), device="cpu")
            else:
                adapter_state = torch.load(binary_adapter, map_location="cpu", weights_only=True)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model.backbone, adapter_state)
            if classifier_path.is_file():
                model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location="cpu", weights_only=True))
            else:
                raise FileNotFoundError(f"recovery checkpoint has no classifier state: {classifier_path}")

        def _get_eval_sampler(self, eval_dataset):
            if hasattr(eval_dataset, "lengths"):
                return LengthOrderedSampler(eval_dataset)
            return LengthOrderedSampler(validation_dataset_rows)
        def evaluate(self, *args, **kwargs):
            started = time.perf_counter()
            previous_phase = recovery_context.get("phase", "training")
            recovery_context["phase"] = "validation"
            try:
                metrics, effective, fallbacks = _evaluate_with_oom_batch_fallback(
                    lambda: super(OrderedValidationTrainer, self).evaluate(*args, **kwargs),
                    get_batch_size=lambda: self.args.per_device_eval_batch_size,
                    set_batch_size=lambda value: setattr(self.args, "per_device_eval_batch_size", value),
                    empty_cache=torch.cuda.empty_cache if torch.cuda.is_available() else None)
                self.effective_eval_batch_size = effective
                self.eval_oom_fallbacks += fallbacks
                return metrics
            except Exception as exc:
                if _is_oom_error(exc):
                    recovery_context["phase"] = "validation_oom"
                raise
            finally:
                self.validation_seconds += time.perf_counter() - started
                if recovery_context.get("phase") == "validation":
                    recovery_context["phase"] = previous_phase

    trainer_type=OrderedValidationTrainer
    if scheduler == "epoch_decay":
        class EpochDecayTrainer(OrderedValidationTrainer):
            def create_scheduler(self,num_training_steps,optimizer=None):
                if self.lr_scheduler is None:
                    optimizer=optimizer or self.optimizer
                    decay=float(learning_rate_decay)
                    warmup_steps=self.args.get_warmup_steps(num_training_steps)
                    self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                        optimizer,lr_lambda=_epoch_decay_lambda(
                            decay=decay,steps_per_epoch=steps_per_epoch,warmup_steps=warmup_steps))
                return self.lr_scheduler
        trainer_type=EpochDecayTrainer
    trainer=trainer_type(model=model,args=args,train_dataset=train_dataset,data_collator=data_collator,
        eval_dataset=validation_dataset_rows if eval_each_epoch else None,
        compute_metrics=compute_metrics if eval_each_epoch else None,
        callbacks=([snapshot_callback,progress_callback,early_stopping_callback]
                   if eval_each_epoch else [progress_callback]))
    if training_views is not None:
        class EpochTrainingViewCallback(TrainerCallback):
            def on_epoch_begin(self,args,state,control,**kwargs):
                train_dataset.set_epoch(int(round(float(state.epoch or 0))))
                return control
        trainer.add_callback(EpochTrainingViewCallback())
    startup_seconds = time.perf_counter() - start
    recovery_context["phase"] = "training"
    training_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=_trainer_resume_checkpoint)
    recovery_context["phase"] = "inference"
    training_seconds = time.perf_counter() - training_started
    post_training_started = time.perf_counter()
    _remove_trainer_recovery_files(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    eval_history = [entry for entry in trainer.state.log_history if "eval_balanced_accuracy" in entry]
    train_history = {int(round(float(entry["epoch"]))): entry for entry in trainer.state.log_history
                     if "loss" in entry and "epoch" in entry}
    validation_metrics_by_epoch = []
    for epoch, entry in enumerate(eval_history, start=1):
        epoch_number=int(round(float(entry.get("epoch", epoch))))
        train_entry=train_history.get(epoch_number,{})
        validation_metrics_by_epoch.append({**entry,
            "train_loss": train_entry.get("loss"),
            "learning_rate_used": train_entry.get("learning_rate",
                float(learning_rate) * float(learning_rate_decay) ** (epoch_number-1)
                if scheduler == "epoch_decay" else None)})
    best_epoch = None
    best_validation_accuracy = None
    best_validation_balanced_accuracy = None
    best_checkpoint_restoration_seconds = 0.0
    if eval_each_epoch:
        metric_key = f"eval_{checkpoint_selection_metric}"
        candidates = [item for item in validation_metrics_by_epoch if item.get(metric_key) is not None]
        if not candidates:
            raise RuntimeError(f"epoch validation ran without recording {checkpoint_selection_metric}")
        selected = max(candidates, key=lambda item: (float(item[metric_key]),
                                                     -int(round(float(item.get("epoch", 0))))))
        best_epoch = int(round(float(selected["epoch"])))
        best_validation_accuracy = float(selected["eval_accuracy"])
        best_validation_balanced_accuracy = float(selected["eval_balanced_accuracy"])
        selected_state = snapshot_callback.best_state
        if selected_state is None or selected_state["epoch"] != best_epoch:
            raise RuntimeError("selected Qwen epoch has no captured adapter and classifier state")
        last_evaluated_epoch = max((int(item["epoch"]) for item in snapshot_callback.validation_predictions),
                                   default=None)
        if best_epoch != last_evaluated_epoch:
            restore_started = time.perf_counter()
            _restore_qwen_selected_state(backbone, model.classifier,
                selected_state["adapter"], selected_state["classifier"])
            best_checkpoint_restoration_seconds = time.perf_counter() - restore_started
    selection = {"metric": f"validation_{checkpoint_selection_metric}" if eval_each_epoch else None,
                 "best_epoch": best_epoch,
                 "best_validation_accuracy": best_validation_accuracy,
                 "best_validation_balanced_accuracy": best_validation_balanced_accuracy,
                 "tie_break": "earliest epoch" if eval_each_epoch else None}
    (out / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Score the exact restored model object. Trainer.predict uses its cached
    # accelerator wrapper, which can retain pre-restore weights after the
    # selected epoch adapter/head have been loaded above.
    prediction_device = next(model.classifier.parameters()).device
    amp_dtype = (torch.bfloat16 if use_bf16 else
                 torch.float16 if use_fp16 else None)

    inference_stats=[]
    def predict_cached(dataset):
        scores, stats = _predict_cached_dataset(model, dataset,
            batch_size=requested_inference_batch_size, data_collator=data_collator,
            device=prediction_device, amp_dtype=amp_dtype, return_info=True)
        inference_stats.append(stats)
        return scores, stats

    selected_prediction = _selected_epoch_scores(snapshot_callback.validation_predictions,
        best_epoch, (len(validation_dataset_rows), len(classes)))
    final_validation_reused_epoch_predictions = selected_prediction is not None
    pred_start = time.perf_counter()
    if final_validation_reused_epoch_predictions:
        logits=selected_prediction;final_inference_stats=None
    else:
        logits,final_inference_stats=predict_cached(validation_dataset_rows)
    inference_seconds = (0.0 if final_validation_reused_epoch_predictions
                         else time.perf_counter() - pred_start)
    checkpoint_started = time.perf_counter()
    if training_views is not None:
        checkpoint_metadata["training_view_sampling"] = train_dataset.sampling_summary()
    backbone.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
    torch.save(model.classifier.state_dict(),out/"classifier.pt")
    (out/"checkpoint_manifest.json").write_text(json.dumps(checkpoint_metadata,indent=2)+"\n")
    checkpoint_save_seconds = time.perf_counter() - checkpoint_started
    if not np.isfinite(logits).all(): raise ValueError("Qwen adapter produced non-finite scores")
    probs=softmax(logits,axis=1)
    validation_loss_raw = (selected.get("eval_loss") if eval_each_epoch else None)
    if validation_loss_raw is None and validation_labels is not None:
        from sklearn.metrics import log_loss
        validation_loss_raw = float(log_loss(
            validation_labels, softmax(logits, axis=1), labels=list(classes)))
    if eval_each_epoch:
        selected["validation_loss_raw"] = float(validation_loss_raw)
    for entry in validation_metrics_by_epoch:
        if entry.get("eval_loss") is not None:
            entry["validation_loss_raw"] = float(entry["eval_loss"])
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_post_inference_batch_size = min(
        (item["effective_inference_batch_size"] for item in inference_stats),
        default=requested_inference_batch_size)
    return {"scores":logits,"probabilities":probs,"model":model,"tokenizer":tokenizer,
        "fit_seconds":time.perf_counter()-start,"inference_seconds":inference_seconds,
        "classes":list(classes),"trainable_parameters":trainable,
        "validation_metrics_by_epoch":validation_metrics_by_epoch,
        "validation_loss_raw": (float(validation_loss_raw)
                                if validation_loss_raw is not None else None),
        "validation_predictions_by_epoch":snapshot_callback.validation_predictions,
        "early_stopping": (_early_stopping_summary(early_stopping_callback, early_stopping_patience, epochs,
                                                    metric=checkpoint_selection_metric)
                           if early_stopping_callback is not None else None),
        "epoch_checkpoints":snapshot_callback.checkpoints,
        "selected_checkpoint_path":str(out),
        "best_checkpoint_path":str(out),
        "best_epoch":best_epoch,"best_validation_accuracy":best_validation_accuracy,
        "best_validation_balanced_accuracy":best_validation_balanced_accuracy,
        "checkpoint_selection_metric":f"validation_{checkpoint_selection_metric}" if eval_each_epoch else None,
        "checkpoint_scoring_method":"best_epoch_validation_predictions" if final_validation_reused_epoch_predictions else "length_grouped_restored_model_forward",
        "final_validation_reused_epoch_predictions":bool(final_validation_reused_epoch_predictions),
        "training_view_sampling":(train_dataset.sampling_summary() if training_views is not None else None),
        "runtime_breakdown":{"startup_seconds":startup_seconds,
            "training_including_validation_seconds":training_seconds,
            "validation_seconds":float(getattr(trainer,"validation_seconds",0.0)),
            "epoch_validation_oom_fallbacks":int(getattr(trainer,"eval_oom_fallbacks",0)),
            "effective_epoch_eval_batch_size":int(getattr(trainer,"effective_eval_batch_size",
                trainer.args.per_device_eval_batch_size)),
            "best_checkpoint_restoration_seconds":float(best_checkpoint_restoration_seconds),
            "final_validation_inference_seconds":float(inference_seconds),
            "final_validation_rows_per_second":(final_inference_stats or {}).get("inference_rows_per_second"),
            "effective_inference_batch_size":effective_post_inference_batch_size,
            "inference_oom_fallbacks":sum(x.get("inference_oom_fallbacks",0) for x in inference_stats),
            "checkpoint_serialization_seconds":float(checkpoint_save_seconds+snapshot_callback.save_seconds),
            "checkpoint_snapshot_seconds":snapshot_callback.snapshot_seconds,
            "training_excluding_validation_seconds":max(0.0,training_seconds-
                float(getattr(trainer,"validation_seconds",0.0))),
            "model_post_training_seconds":float(time.perf_counter()-post_training_started)},
        "requested_batch_size":int(batch_size),"effective_batch_size":effective_batch_size,
        "selected_physical_batch_size":effective_batch_size,
        "effective_optimizer_batch_size":effective_batch_size*int(gradient_accumulation_steps),
        "gradient_accumulation_steps":int(gradient_accumulation_steps),
        "requested_inference_batch_size":requested_inference_batch_size,
        "lr_scheduler_type":scheduler,"warmup_ratio":float(warmup_ratio),
        "batch_size_probe":batch_probe,
        "quantization":quantization,"quantization_config":quantization_config,
        "attention_backend":attention_backend,
        "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_gpu_memory_reserved_bytes":int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None}
