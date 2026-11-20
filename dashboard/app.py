"""Streamlit report for the local experiment registry and stored predictions."""
from __future__ import annotations

import hashlib
import json
import os
import ast
import math
import re
import time
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.registry import Registry
from dashboard.training_lineage import training_lineage, trainer_completion
from dashboard.training_config import live_training_config
from src.dataset_catalog import load_dataset_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Name That LLM experiments", layout="wide")
st.title("Name That LLM — experiment results")
EXPERIMENT_REPORTS = {"zod_04": Path(__file__).resolve().parents[1] / "docs/zod_04.md"}
requested_report = st.query_params.get("experiment_report")
if requested_report:
    report_path = EXPERIMENT_REPORTS.get(str(requested_report))
    if report_path is not None and report_path.is_file():
        st.markdown("[← Back to dashboard](./)")
        st.markdown(report_path.read_text())
        st.stop()
    st.caption(f"No report is registered for experiment {requested_report}.")

registry_path = Path("experiments/registry.sqlite")
records = Registry(registry_path).rows() if registry_path.exists() else []
records = sorted(records, key=lambda row: int(row.get("sequence_number") or 0), reverse=True)
if not records:
    st.info("No completed experiments are recorded yet. Active training appears below while it runs.")

profile_path = PROJECT_ROOT / "data/processed/dataset_profile.json"
profile_data = json.loads(profile_path.read_text()) if profile_path.exists() else {}
profile_validation_rows = profile_data.get("validation_rows")

def metrics(row): return json.loads(row["metrics_json"])

MODEL_ALIASES = {"qwen06_best_epoch03": "qwen06"}

def canonical_model(name):
    return MODEL_ALIASES.get(name, name)

def model_display(name):
    name = canonical_model(name)
    return {"qwen06": "qwen3-0.6B", "qwen17": "qwen3-1.7B", "qwen35_08b": "qwen3.5-0.8B", "qwen4": "qwen3-4.0B",
            "ultrabert380": "UltraBERT-380M", "phi35_mini": "Phi-3.5 Mini",
            "embeddinggemma2": "EmbeddingGemma 2",
            "granite4_350m": "Granite 4.0 350M Base",
            "granite41_3b": "Granite 4.1 3B Base",
            "modernbert_large": "ModernBERT-large",
            "deberta_v3_large": "DeBERTa-v3-large",
            "electra_large": "ELECTRA-large-discriminator",
            "gte_large": "GTE-large-en-v1.5",
            "mpnet_base": "MPNet-base",
            "deberta_gemini": "DeBERTa-v3-base Gemini specialist",
            "deberta_glm": "DeBERTa-v3-base GLM specialist",
            "minicpm5_2b": "MiniCPM5-2B-Base",
            "hunyuan_1_8b": "Hunyuan-1.8B-Pretrain",
            "catboost_stylometry": "CatBoost + Stylometry"}.get(name, name)

def manifest_for(row):
    path = row.get("artifact_path")
    if path and Path(path).name.endswith("_validation.parquet"):
        candidates = [Path(path).with_name(f"{row['experiment_id']}_manifest.json")]
        for candidate in candidates:
            if candidate.exists(): return json.loads(candidate.read_text())
    return {}

def active_training(root):
    """Return a live pipeline training process for this checkout, if one exists."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if Path(os.readlink(entry / "cwd")).resolve() != root.resolve():
                continue
            executable = Path(os.readlink(entry / "exe")).name.lower()
            if not executable.startswith(("python", "pypy")):
                continue
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            args = [arg for arg in args if arg]
        except (OSError, PermissionError):
            continue
        if "-m" not in args or "src.pipeline" not in args:
            continue
        try:
            module_index = args.index("src.pipeline")
            command = args[module_index + 1:]
            if len(command) < 2 or command[0] != "models" or command[1] not in {
                "train", "continue-qwen", "continue-lora", "continue-encoder"
            }:
                continue
            action = command[1]
            options = {}
            index = 2
            while index < len(command):
                token = command[index]
                if token.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
                    options[token] = command[index + 1]
                    index += 2
                else:
                    index += 1
            model = options.get("--model", "qwen06" if action == "continue-qwen" else "svc")
            training_dataset = options.get("--dataset", "v1")
            config = live_training_config(root, action, model, options)
            try:
                log_path = Path(os.readlink(entry / "fd/1"))
                if not log_path.is_absolute():
                    log_path = root / log_path
            except OSError:
                log_path = None
            progress_path = None
            for candidate in (root / "artifacts/runs").rglob("training_progress.json"):
                try:
                    progress = json.loads(candidate.read_text())
                    if int(progress.get("pid", -1)) == int(entry.name):
                        progress_path = candidate
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            console_log_path = None
            if progress_path is not None:
                candidate = progress_path.parents[2] / "training_console.log"
                if candidate.is_file():
                    console_log_path = candidate
            if console_log_path is None:
                # The console log is created before model loading and baseline
                # inference, so it can identify the active run before the
                # trainer has emitted training_progress.json.
                for candidate in (root / "artifacts/runs").rglob("training_console.log"):
                    try:
                        first_line = candidate.open(encoding="utf-8", errors="replace").readline()
                        if not first_line.startswith("TRAINING_RUN "):
                            continue
                        if int(json.loads(first_line[len("TRAINING_RUN "):]).get("pid", -1)) == int(entry.name):
                            console_log_path = candidate
                            break
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
            if console_log_path is not None:
                log_path = console_log_path
            return {"pid": int(entry.name), "model": model, "config": config,
                    "training_dataset": training_dataset,
                    "action": action, "log_path": log_path, "progress_path": progress_path,
                    "console_log_path": console_log_path}
        except (OSError, ValueError, IndexError):
            continue
    return None

def latest_training_step(log_path):
    """Return the most recent progress counter from a trainer's stdout log."""
    if not log_path or not log_path.is_file():
        return None
    try:
        content = log_path.read_bytes()[-65536:].decode(errors="replace")
    except OSError:
        return None
    matches = re.findall(r"(?<!\d)(\d{1,9})/(\d{1,9})(?!\d)", content)
    return tuple(map(int, matches[-1])) if matches else None

def _format_epoch_eta(seconds):
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"

def live_progress_details(progress_path, log_path, epoch_limit):
    """Return epoch-local step progress and ETA, without exposing process IDs."""
    live_state = {}
    if progress_path and progress_path.is_file():
        try:
            live_state = json.loads(progress_path.read_text()).get("live", {})
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    current = live_state.get("epoch_step")
    total = live_state.get("steps_per_epoch")
    eta = live_state.get("epoch_eta_seconds")
    live_epoch = live_state.get("epoch")
    global_step = live_state.get("global_step")
    progress_age = None
    if live_state.get("updated_at") is not None:
        try:
            progress_age = max(0.0, time.time() - float(live_state["updated_at"]))
        except (TypeError, ValueError):
            progress_age = None
    if total and current is None and global_step is not None:
        current = ((int(global_step) - 1) % int(total)) + 1 if int(global_step) else 0
        if live_epoch is None:
            live_epoch = min((int(global_step) - 1) // int(total) + 1,
                             max(1, int(epoch_limit or 1))) if int(global_step) else 1
    if current is None or total is None:
        step = latest_training_step(log_path)
        if step and step[1]:
            # Older runs may expose only Trainer's whole-run tqdm counter.
            # Convert it to an epoch-local counter when the epoch count is known.
            total = max(1, math.ceil(step[1] / max(1, int(epoch_limit or 1))))
            current = ((step[0] - 1) % total) + 1 if step[0] else 0
    step_text = f"{int(current):,}/{int(total):,}" if current is not None and total is not None else "—/—"
    eta_text = _format_epoch_eta(eta) if eta is not None else "estimating"
    if progress_age is not None and progress_age > 45:
        step_text += " (stale)"
        eta_text = "— (stale)"
    return step_text, eta_text, live_epoch


def effective_epoch_limit(config, progress_path):
    """Resolve the absolute epoch target, including continuation progress.

    Continuation configs store the number of additional epochs, while the
    trainer's progress file reports absolute epochs from the source run. Prefer
    the trainer's total-step metadata when available and never hide epochs
    already reported by the trainer.
    """
    try:
        limit = max(0, int(config.get("epochs", 0) or 0))
    except (TypeError, ValueError):
        limit = 0
    if not progress_path or not Path(progress_path).is_file():
        return limit
    try:
        progress = json.loads(Path(progress_path).read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return limit

    epochs = progress.get("epochs", [])
    completed = []
    for item in epochs:
        if not isinstance(item, dict) or item.get("epoch") is None:
            continue
        try:
            completed.append(int(item["epoch"]))
        except (TypeError, ValueError):
            continue
    live = progress.get("live", {})
    if live.get("epoch") is not None:
        try:
            completed.append(int(live["epoch"]))
        except (TypeError, ValueError):
            pass
    if live.get("total_steps") and live.get("steps_per_epoch"):
        try:
            total_epochs = math.ceil(int(live["total_steps"]) / int(live["steps_per_epoch"]))
            limit = max(limit, total_epochs)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return max([limit, *completed], default=limit)

def training_log_tail(log_path, *, max_lines=100, max_bytes=256_000):
    """Read only the newest bounded section of a live training console log."""
    if not log_path or not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            content = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return "\n".join(content.replace("\r", "\n").splitlines()[-max_lines:])

@st.fragment(run_every="10s")
def live_training_banner():
    live = active_training(Path(__file__).resolve().parents[1])
    if live is None:
        st.caption("No model training process is currently running.")
        return
    config = live["config"]
    title = config.get("experiment_name") or live["model"]
    trainer_state = trainer_completion(live.get("progress_path"))
    if trainer_state["finished"]:
        epoch = trainer_state.get("epoch")
        epoch_text = f" after epoch {epoch}" if epoch is not None else ""
        st.info(f"Training finished{epoch_text}; post-training pipeline work is still running for {title} · {model_display(live['model'])}.")
        return
    epoch_limit = effective_epoch_limit(config, live.get("progress_path"))
    completed_epochs = 0
    progress_path = live.get("progress_path")
    if progress_path and progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text())
            completed_epochs = max(
                (int(item.get("epoch", 0)) for item in progress.get("epochs", [])
                 if item.get("epoch") is not None),
                default=0,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            completed_epochs = 0
    # Trainer callbacks publish epoch metrics at the end of each epoch. Before
    # that first update, the active training process is working on epoch one.
    step_text, eta_text, live_epoch = live_progress_details(
        live.get("progress_path"), live.get("log_path"), epoch_limit)
    current_epoch = (int(live_epoch) if live_epoch is not None else
                     min(completed_epochs + 1, epoch_limit) if epoch_limit else None)
    epoch_label = (f"epoch {current_epoch}/{epoch_limit} max" if current_epoch is not None
                   else "epoch unknown")
    st.info(f"Training in progress: {title} · {model_display(live['model'])} · {epoch_label} · steps {step_text} · epoch ETA {eta_text}")

st.subheader("Live training")
live_training_banner()

def live_epoch_metrics(log_path):
    """Load epoch metrics emitted by the active training process."""
    by_epoch = {}
    if not log_path or not log_path.is_file():
        return by_epoch
    try:
        progress = json.loads(log_path.read_text())
        for item in progress.get("epochs", []):
            if item.get("epoch") is not None:
                by_epoch[int(item["epoch"])] = item
        if by_epoch:
            return by_epoch
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    # Older trainer versions may print metric dictionaries directly to stdout.
    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return by_epoch
    content = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", content)
    for line in re.split(r"[\r\n]+", content):
        match = re.search(r"\{[^{}]*['\"](?:epoch|eval_loss|eval_accuracy|loss)['\"][^{}]*\}", line)
        if not match:
            continue
        try:
            item = ast.literal_eval(match.group(0))
            epoch_value = item.get("epoch")
            if epoch_value is None:
                continue
            epoch = int(round(float(epoch_value)))
            if epoch < 1:
                continue
            by_epoch.setdefault(epoch, {}).update(item)
        except (ValueError, SyntaxError, TypeError):
            continue
    return by_epoch

comparison = []
for row in records:
    value = metrics(row)
    run_manifest = manifest_for(row)
    weighted_accuracy = None
    if value.get("ensembles"):
        weighted = next((item for name, item in value["ensembles"].items()
                         if name.startswith("train_weighted:")), None)
        if weighted:
            weighted_accuracy = weighted.get("metrics", {}).get("balanced_accuracy")
    row_validation_rows = value.get("validation_rows", run_manifest.get("validation_rows"))
    if row_validation_rows is None or pd.isna(row_validation_rows):
        row_validation_rows = profile_validation_rows
    comparison.append({"Experiment": f"{int(row['sequence_number']):04d}", "Description": row.get("description", ""),
        "Model": model_display(row["model"]), "Split": row["split"],
        "acc_raw": run_manifest.get("selected_validation_accuracy", value.get("balanced_accuracy")),
        "acc_weighted": weighted_accuracy, "Macro F1": value.get("macro_f1"),
        "Log loss": value.get("log_loss"), "Runtime (s)": row.get("runtime_seconds"),
        "Validation rows": row_validation_rows,
        "Seed": row.get("seed"), "Created": row.get("created_at")})
frame = pd.DataFrame(comparison, columns=["Experiment", "Description", "Model", "Split", "acc_raw",
    "acc_weighted", "Macro F1", "Log loss", "Runtime (s)", "Validation rows", "Seed", "Created"])
validation = frame[frame.Split.eq("validation") & frame["acc_raw"].notna()]

# The headline validation metrics refer to V2 training runs, not the historical
# 540-row profile or unrelated older experiments.
v2_validation_runs = []
for row in records:
    if row.get("split") != "validation":
        continue
    run_manifest = manifest_for(row)
    dataset_name = str(run_manifest.get("dataset") or run_manifest.get("training_dataset") or
                       run_manifest.get("dataset_version") or "")
    if not dataset_name.lower().startswith("v2"):
        continue
    accuracy = run_manifest.get("selected_validation_accuracy")
    if accuracy is None:
        try:
            accuracy = metrics(row).get("accuracy")
        except (KeyError, TypeError, json.JSONDecodeError):
            accuracy = None
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        continue
    validation_rows = run_manifest.get("validation_rows")
    if validation_rows is None:
        try:
            validation_rows = metrics(row).get("validation_rows")
        except (KeyError, TypeError, json.JSONDecodeError):
            validation_rows = None
    v2_validation_runs.append({"accuracy": accuracy, "validation_rows": validation_rows,
                               "dataset": dataset_name, "experiment_id": row.get("experiment_id")})
v2_best_validation = max(v2_validation_runs, key=lambda item: item["accuracy"]) if v2_validation_runs else None

@st.fragment(run_every="10s")
def render_current_training():
    st.subheader("Current training")
    live_training = active_training(Path(__file__).resolve().parents[1])
    if live_training is None:
        st.caption("No model is currently training.")
        return
    config = live_training["config"]
    trainer_state = trainer_completion(live_training.get("progress_path"))
    if trainer_state["finished"]:
        epoch = trainer_state.get("epoch")
        epoch_text = f" at epoch {epoch}" if epoch is not None else ""
        st.caption(f"Trainer has finished{epoch_text}; the pipeline process is completing post-training work.")
    st.markdown(f"**Model:** {model_display(live_training['model'])}")
    dataset_id = live_training.get("training_dataset") or config.get("training_data") or config.get("dataset", "v1")
    dataset_labels = {"v0": "V0", "v1": "V1", "v1.1": "V1.1", "v1.2_bm": "V1.2-BM",
                      "v1_all": "V1-all", "v1.3": "V1.3", "v1.5": "V1.5",
                      "v2.0": "V2.0", "v2.1": "V2.1"}
    st.markdown(f"**Training dataset:** {dataset_labels.get(str(dataset_id).lower(), dataset_id)} (`{dataset_id}`)")
    configured_epochs = config.get("epochs", "—")
    total_epochs = effective_epoch_limit(config, live_training.get("progress_path"))
    try:
        configured_epoch_count = int(configured_epochs or 0)
    except (TypeError, ValueError):
        configured_epoch_count = 0
    if str(live_training.get("action", "")).startswith("continue-") and total_epochs > configured_epoch_count:
        st.markdown(f"**Epochs:** {configured_epochs} additional · {total_epochs} total including source run")
    else:
        st.markdown(f"**Epochs:** {total_epochs or configured_epochs}")
    recent_log = training_log_tail(live_training.get("console_log_path") or live_training.get("log_path"))
    with st.expander("Recent training log", expanded=False):
        if recent_log:
            st.code(recent_log, language="text")
        else:
            st.caption("No per-run console log is available for this training process.")
    st.markdown("**Training parameters**")
    if config:
        batch_probe = {}
        live_metrics = {}
        progress_path = live_training.get("progress_path")
        if progress_path:
            probe_path = Path(progress_path).with_name("batch_size_probe.json")
            try:
                if probe_path.is_file():
                    batch_probe = json.loads(probe_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                batch_probe = {}
            try:
                live_metrics = json.loads(Path(progress_path).read_text()).get("live", {})
            except (OSError, ValueError, json.JSONDecodeError):
                live_metrics = {}
        requested_batch = batch_probe.get("requested_batch_size", config.get("batch_size"))
        selected_physical_batch = batch_probe.get("selected_physical_batch_size",
            batch_probe.get("selected_batch_size", batch_probe.get("effective_batch_size", requested_batch)))
        if selected_physical_batch is None:
            selected_physical_batch = requested_batch
        accumulation = batch_probe.get("gradient_accumulation_steps",
                                       config.get("gradient_accumulation_steps", 1))
        if accumulation is None:
            accumulation = config.get("gradient_accumulation_steps", 1)
        effective_optimizer_batch = batch_probe.get("effective_optimizer_batch_size")
        if effective_optimizer_batch is None:
            effective_optimizer_batch = config.get("target_effective_batch_size")
        if effective_optimizer_batch is None:
            try:
                effective_optimizer_batch = int(selected_physical_batch) * int(accumulation)
            except (TypeError, ValueError):
                effective_optimizer_batch = "—"
        params = [{"Parameter": key, "Value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                  for key, value in config.items() if key not in {"model", "epochs", "batch_size",
                      "gradient_accumulation_steps", "target_effective_batch_size"}]
        params.extend([
            {"Parameter": "Requested physical batch (per GPU)", "Value": requested_batch},
            {"Parameter": "Selected physical batch (per GPU)", "Value": selected_physical_batch},
            {"Parameter": "Gradient accumulation steps", "Value": accumulation},
            {"Parameter": "Effective batch per optimizer step", "Value": effective_optimizer_batch},
        ])
        allocated = live_metrics.get("training_peak_allocated_bytes",
                                     batch_probe.get("peak_probe_allocated_bytes"))
        reserved = live_metrics.get("training_peak_reserved_bytes",
                                    batch_probe.get("peak_probe_reserved_bytes"))
        utilization = live_metrics.get("gpu_vram_utilization",
                                       batch_probe.get("selected_peak_vram_utilization"))
        throughput_examples = batch_probe.get("selected_examples_per_second")
        throughput_tokens = batch_probe.get("selected_tokens_per_second")
        padded_tokens = batch_probe.get("selected_padded_tokens_per_batch")
        training_examples = live_metrics.get("training_examples_per_second")
        training_tokens = live_metrics.get("training_tokens_per_second")
        params.extend([
            {"Parameter": "Peak allocated VRAM (GiB)",
             "Value": round(float(allocated) / 1024**3, 2) if allocated is not None else "—"},
            {"Parameter": "Peak reserved VRAM (GiB)",
             "Value": round(float(reserved) / 1024**3, 2) if reserved is not None else "—"},
            {"Parameter": "VRAM utilization", "Value":
             f"{float(utilization):.1%}" if utilization is not None else "—"},
            {"Parameter": "Probe throughput (examples/sec)", "Value":
             round(float(throughput_examples), 2) if throughput_examples is not None else "—"},
            {"Parameter": "Probe throughput (tokens/sec)", "Value":
             round(float(throughput_tokens), 2) if throughput_tokens is not None else "—"},
            {"Parameter": "Training examples/sec (live estimate)", "Value":
             round(float(training_examples), 2) if training_examples is not None else
             (round(float(throughput_examples), 2) if throughput_examples is not None else "—")},
            {"Parameter": "Training tokens/sec (live estimate)", "Value":
             round(float(training_tokens), 2) if training_tokens is not None else
             (round(float(throughput_tokens), 2) if throughput_tokens is not None else "—")},
            {"Parameter": "Padded tokens per selected batch", "Value":
             int(padded_tokens) if padded_tokens is not None else "—"},
        ])
        params_frame = pd.DataFrame(params)
        # The settings table intentionally combines values of different types.
        # Normalize for Streamlit's Arrow serializer instead of triggering a
        # failed conversion and fallback on each dashboard rerun.
        params_frame["Value"] = params_frame["Value"].map(
            lambda value: "—" if value is None else str(value))
        st.dataframe(params_frame, use_container_width=True, hide_index=True)
    else:
        st.caption("No training parameters are available.")
    epoch_count = total_epochs
    learning_rate = config.get("learning_rate")
    decay = float(config.get("learning_rate_decay", 1.0) or 1.0)
    progress_path = live_training.get("progress_path")
    history_by_epoch = live_epoch_metrics(progress_path or live_training.get("log_path"))
    epoch_rows = []
    baseline = {}
    if progress_path and Path(progress_path).is_file():
        try:
            baseline = json.loads(Path(progress_path).read_text()).get("baseline", {})
        except (OSError, ValueError, json.JSONDecodeError):
            baseline = {}
    if baseline:
        epoch_rows.append({"Epoch": "Baseline", "Learning rate": None,
                           "Accuracy": baseline.get("accuracy"),
                           "Accuracy change vs baseline (pp)": None, "Train error": None,
                           "Validation loss (raw)": baseline.get("validation_loss_raw",
                                                                  baseline.get("validation_loss"))})
    for epoch in range(1, epoch_count + 1):
        item = history_by_epoch.get(epoch, {})
        accuracy = item.get("accuracy", item.get("eval_accuracy"))
        baseline_accuracy = baseline.get("accuracy")
        try:
            accuracy_change = (100.0 * (float(accuracy) - float(baseline_accuracy))
                               if accuracy is not None and baseline_accuracy is not None else None)
        except (TypeError, ValueError):
            accuracy_change = None
        epoch_rows.append({
            "Epoch": epoch,
            "Learning rate": item.get("learning_rate", float(learning_rate) * decay ** (epoch - 1)
                                       if learning_rate is not None else None),
            "Accuracy": accuracy,
            "Accuracy change vs baseline (pp)": accuracy_change,
            "Train error": item.get("train_loss", item.get("loss")),
            "Validation loss (raw)": item.get("validation_loss_raw",
                                               item.get("validation_loss", item.get("eval_loss"))),
        })
    st.markdown("**Epochs**")
    st.dataframe(pd.DataFrame(epoch_rows, columns=["Epoch", "Learning rate", "Accuracy",
                 "Accuracy change vs baseline (pp)", "Train error",
                 "Validation loss (raw)"]),
                 use_container_width=True, hide_index=True)
    if baseline:
        st.caption("Baseline accuracy is measured before the first optimizer step. Accuracy can stay unchanged while validation loss improves if no example changes its top-1 label.")

def render_trust_list_table(trust_path, metadata, *, empty_message, caption):
    trust = pd.read_csv(trust_path)
    if trust.empty:
        st.info(empty_message)
        return
    trust["SeqID"] = pd.to_numeric(trust["SeqID"], errors="raise").astype(int)
    trust["prediction_count"] = pd.to_numeric(trust["prediction_count"], errors="coerce")
    trust["checkpoint_model_display"] = trust["model"].map(model_display)
    trust["checkpoint_status"] = trust["checkpoint_available"].map(
        lambda value: "Available" if str(value).lower() in {"true", "1"} else "Missing")
    trust["rule_eligible_display"] = trust["rule_eligible"].map(
        lambda value: "Yes" if str(value).lower() in {"true", "1"} else "No")
    trust = trust.rename(columns={
        "sequence_id": "ID", "target_label": "Model label",
        "checkpoint_model_display": "Checkpoint model", "checkpoint": "Checkpoint",
        "dataset_lineage": "Training dataset", "checkpoint_date": "Date (MM.DD)",
        "precision": "Label accuracy (precision)", "recall": "Recall",
        "prediction_count": "Predicted rows", "rule_eligible_display": "Rule eligible",
        "checkpoint_status": "Checkpoint status", "evaluation_note": "Validation note",
    })
    columns = ["SeqID", "ID", "Model label", "Checkpoint model", "Checkpoint",
               "Training dataset", "Date (MM.DD)", "Label accuracy (precision)",
               "Recall", "Predicted rows", "Rule eligible", "Checkpoint status",
               "Validation note"]
    table = trust[columns].style.format({
        "SeqID": "{:.0f}", "Label accuracy (precision)": "{:.1%}",
        "Recall": "{:.1%}", "Predicted rows": "{:.0f}",
    }, na_rep="—")
    version = metadata.get("version", "unknown version")
    generated = metadata.get("generated_at_utc", "unknown generation time")
    st.caption(f"{version}, generated {generated}. {caption}")
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_trust_list_two(root):
    st.subheader("Trust List 2")
    try:
        from src.v2_checkpoint_trust import latest_trust_paths
        latest = latest_trust_paths(root)
        if latest is None:
            st.info("Trust List 2 has not been generated yet.")
            return
        trust_path, _, metadata = latest
        render_trust_list_table(
            trust_path, metadata, empty_message="Trust List 2 is empty.",
            caption="Scores use the full 2,699-row V0 validation set and are development diagnostics.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not read Trust List 2: {exc}")


def render_trust_list_two_two(root):
    st.subheader("Trust List 2.2")
    try:
        from src.trust_list2_2 import latest_trust_list_2_2_paths
        latest = latest_trust_list_2_2_paths(root)
        if latest is None:
            st.info("Trust List 2.2 has not been generated yet.")
            return
        trust_path, _, metadata = latest
        render_trust_list_table(
            trust_path, metadata, empty_message="Trust List 2.2 is empty.",
            caption="Includes only checkpoints trained on v2.x datasets. Scores use the full 2,699-row V0 validation set and are development diagnostics.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not read Trust List 2.2: {exc}")


def render_prediction_factory():
    root = Path(__file__).resolve().parents[1]
    render_trust_list_two(root)
    render_trust_list_two_two(root)


def render_ensemble_catalog(root):
    from src.ensemble_catalog import CATALOG

    try:
        catalog_bytes = CATALOG.read_bytes()
        catalog = json.loads(catalog_bytes)
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the ensemble catalog: {exc}")
        return

    entries = {entry["id"]: entry for entry in catalog["strategies"]}
    catalog_hash = hashlib.sha256(catalog_bytes).hexdigest()
    latest_scores = {}
    results_root = root / "artifacts/analysis/ensemble_catalog"
    manifest_paths = list(results_root.glob("*/results.json"))
    manifest_paths.extend((root / "artifacts/analysis/e13").glob("*/catalog_results.json"))
    for manifest in sorted(manifest_paths, key=lambda path: path.parent.name, reverse=True):
        try:
            run = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if run.get("catalog_sha256") != catalog_hash:
            continue
        for result in run.get("results", []):
            code = result.get("ensemble")
            if code in entries:
                latest_scores.setdefault(code, (result, run["trust_list"]["version"], manifest.parent.name))
        if len(latest_scores) == len(entries):
            break

    rows = []
    for code, entry in entries.items():
        result, trust_version, _ = latest_scores.get(code, ({}, None, None))
        deployment = result.get("deployment")
        rows.append({
            "Code": code, "Ensemble": entry["name"],
            "Accuracy": result.get("accuracy"), "Balanced accuracy": result.get("balanced_accuracy"),
            "Trust List 2": trust_version,
            "Checkpoints used": (len(deployment["checkpoint_ids"]) if deployment else
                                 len(result.get("checkpoint_ids", [])) or None),
            "Status": result.get("status", "Awaiting score").replace("_", " ").capitalize(),
        })

    st.subheader("Ensemble catalog")
    st.caption(f"Version {catalog['version']}. Saved scores use held-out V0 prompt groups.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={
                     "Accuracy": st.column_config.NumberColumn(format="percent"),
                     "Balanced accuracy": st.column_config.NumberColumn(format="percent"),
                     "Checkpoints used": st.column_config.NumberColumn(),
                 })
    selected = st.selectbox("Ensemble", list(entries),
                            format_func=lambda code: f"{code} — {entries[code]['name']}",
                            key="ensemble_catalog_strategy")
    result, trust_version, timestamp = latest_scores.get(selected, ({}, None, None))
    if timestamp:
        st.caption(f"Scored with {trust_version} · Run {timestamp}")
    if result.get("reason"):
        st.warning(result["reason"])
    if deployment := result.get("deployment"):
        st.markdown("**Deployment roster**")
        st.write(", ".join(deployment["checkpoint_ids"]))
    elif checkpoint_ids := result.get("checkpoint_ids"):
        st.markdown("**Checkpoints selected across held-out folds**")
        st.write(", ".join(checkpoint_ids))
    if "oracle_accuracy" in result:
        st.metric("Held-out top-1 oracle coverage", f"{result['oracle_accuracy']:.2%}")
        st.caption("Share of held-out rows where at least one selected checkpoint predicted the true label.")
    if "candidate_coverage" in result:
        coverage, arbitration = st.columns(2)
        coverage.metric("Candidate coverage", f"{result['candidate_coverage']:.2%}")
        if result.get("arbitration_accuracy_given_covered") is not None:
            arbitration.metric("Accuracy when candidate is covered", f"{result['arbitration_accuracy_given_covered']:.2%}")
    if comparison := result.get("comparison_vs_E08"):
        corrected, lost = st.columns(2)
        corrected.metric("Corrected vs E08", comparison["corrected"])
        lost.metric("Lost vs E08", comparison["lost"])
    if per_label := result.get("per_label"):
        with st.expander("Per-label held-out results"):
            st.dataframe(pd.DataFrame.from_dict(per_label, orient="index"), width="stretch",
                         column_config={
                             "precision": st.column_config.NumberColumn(format="percent"),
                             "recall": st.column_config.NumberColumn(format="percent"),
                         })
    with st.expander("Selection configuration"):
        st.json({key: value for key, value in entries[selected].items() if key not in {"id", "name"}})

columns = st.columns(4)
columns[0].metric("Experiments", len(records))
columns[1].metric("Models", int(frame.Model.nunique()))
columns[2].metric("Best validation accuracy", f"{v2_best_validation['accuracy']:.4f}" if v2_best_validation else "—",
                  help=(f"Best full-V0 result among V2 training runs: {v2_best_validation['dataset']} · "
                        f"{v2_best_validation['experiment_id']}" if v2_best_validation else "No V2 validation runs found."))
rows_value = v2_best_validation.get("validation_rows") if v2_best_validation else None
columns[3].metric("Validation rows", int(rows_value) if rows_value is not None and not pd.isna(rows_value) else "—",
                  help="Validation row count from the selected best V2 training run.")

tabs = st.tabs(["Overview", "Experiments", "Prediction factory", "Current training", "Models", "Errors", "Compute", "Data", "Ensemble catalog"])
with tabs[0]:
    st.subheader("Experiment comparison")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if not validation.empty:
        st.plotly_chart(px.bar(validation, x="Experiment", y="acc_raw", color="Model", hover_data=["Split", "Seed"]), use_container_width=True)
    st.download_button("Download experiment table (CSV)", frame.to_csv(index=False), "experiments.csv", "text/csv")
    st.download_button("Download experiment table (JSON)", frame.to_json(orient="records", indent=2), "experiments.json", "application/json")
with tabs[1]:
    experiments = pd.DataFrame([{
        "ID": "Analysis 01",
        "Name": "zod_04",
        "Description": (
            "Existing V1-all → V0 best UltraBERT checkpoint; extracts and caches all 24 "
            "attention-mask-pooled layer representations with aligned V0 IDs and labels, "
            "ranks layers using five-fold prompt-grouped CV, and fits 28 final classifiers: "
            "24 individual layers and four multi-layer combinations."
        ),
        "Training dataset": "—", "Validation": "—", "Accuracy": "—",
        "Balanced accuracy": "—", "Baseline balanced accuracy": "—", "Change": "—",
        "Report": "?experiment_report=zod_04",
    }])
    st.dataframe(experiments, use_container_width=True, hide_index=True,
        column_config={"Report": st.column_config.LinkColumn(
            "Report", display_text="docs/zod_04.md")})

    hn_experiments = []
    for row in records:
        manifest = manifest_for(row)
        dataset_name = str(manifest.get("training_dataset") or manifest.get("dataset") or "")
        if dataset_name.lower() != "v2.14hn" or row.get("split") != "validation":
            continue
        values = metrics(row)
        baseline = manifest.get("baseline_validation_metrics") or {}
        accuracy = manifest.get("selected_validation_accuracy", values.get("accuracy"))
        balanced_accuracy = manifest.get("selected_validation_balanced_accuracy")
        if balanced_accuracy is None:
            balanced_accuracy = values.get("balanced_accuracy")
        baseline_balanced = baseline.get("balanced_accuracy")
        change = (100.0 * (float(balanced_accuracy) - float(baseline_balanced))
                  if balanced_accuracy is not None and baseline_balanced is not None else None)
        hn_experiments.append({
            "ID": f"{int(row.get('sequence_number') or 0):04d}",
            "Name": row.get("description") or row.get("experiment_id"),
            "Model": model_display(row.get("model", "")),
            "Training dataset": dataset_name,
            "Validation": f"{manifest.get('validation_dataset', 'v0')} ({int(manifest.get('validation_rows') or values.get('validation_rows') or 0):,} rows)",
            "Epochs": int(manifest.get("selected_epoch") or manifest.get("best_epoch") or 0),
            "Accuracy": f"{100.0 * float(accuracy):.2f}%" if accuracy is not None else "—",
            "Balanced accuracy": f"{100.0 * float(balanced_accuracy):.2f}%" if balanced_accuracy is not None else "—",
            "Baseline balanced accuracy": f"{100.0 * float(baseline_balanced):.2f}%" if baseline_balanced is not None else "—",
            "Change": f"{change:+.2f} pp" if change is not None else "—",
            "Checkpoint": Path(str(manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path") or "—")).name,
        })
    st.subheader("V2.14HN training experiments")
    if hn_experiments:
        st.dataframe(pd.DataFrame(hn_experiments), use_container_width=True, hide_index=True)
    else:
        st.caption("No V2.14HN training experiments are registered yet.")

with tabs[2]:
    render_prediction_factory()

with tabs[3]:
    render_current_training()

with tabs[4]:
    st.subheader("Model summary")
    best_runs = {}
    for row in records:
        model = canonical_model(row.get("model", ""))
        if row.get("split") != "validation" or "ensemble" in model.lower() or "+" in model:
            continue
        run_manifest = manifest_for(row)
        # This column and checkpoint comparison are balanced-accuracy based.
        # Prefer the selected balanced score, then balanced epoch history, then
        # historical registry metrics; use ordinary accuracy only for legacy
        # artifacts that recorded no balanced metric.
        score = run_manifest.get("selected_validation_balanced_accuracy")
        if score is None:
            history = run_manifest.get("validation_metrics_by_epoch", [])
            history_scores = [item.get("eval_balanced_accuracy") for item in history]
            history_scores = [float(value) for value in history_scores if value is not None]
            score = max(history_scores) if history_scores else metrics(row).get("balanced_accuracy")
        if score is None:
            score = run_manifest.get("selected_validation_accuracy", metrics(row).get("accuracy"))
        if score is None or pd.isna(score):
            continue
        # records are newest first, so ties keep the most recent run.
        if model not in best_runs or float(score) > best_runs[model]["accuracy"]:
            best_runs[model] = {"row": row, "accuracy": float(score)}

    model_rows = []
    for model, best in best_runs.items():
        row = best["row"]
        manifest = manifest_for(row)
        validation_metrics = manifest.get("validation_metrics") or {}
        history = manifest.get("validation_metrics_by_epoch", [])
        selected_history = None
        if history:
            def epoch_accuracy(item):
                value = item.get("eval_balanced_accuracy")
                if value is None:
                    value = item.get("eval_accuracy")
                return float(value) if value is not None else float("-inf")

            selected_history = max(history, key=epoch_accuracy)
        model_rows.append({
            "Model": model_display(model),
            "Best balanced accuracy": best["accuracy"],
            "Train loss": selected_history.get("train_loss") if selected_history else None,
            "Validation loss (raw)": manifest.get("validation_loss_raw",
                validation_metrics.get("validation_loss_raw",
                    selected_history.get("eval_loss") if selected_history else None)),
            "Best run": row.get("description") or row.get("experiment_id"),
            "Training": training_lineage(
                manifest, Path(row.get("artifact_path") or "artifacts/predictions").parent),
            "Recipe": (f"{manifest.get('feature_count')} features; "
                       f"baseline + {','.join(manifest.get('feature_recipe', {}).get('groups', []))}"
                       if manifest.get("feature_recipe") else None),
            "Trees": manifest.get("tree_count"),
            "Checkpoint": (manifest.get("checkpoint_filename") or
                           ("model.cbm" if manifest.get("checkpoint_path") and
                            (Path(manifest["checkpoint_path"]) / "model.cbm").is_file() else
                            Path(manifest["checkpoint_path"]).name if manifest.get("checkpoint_path") else None)),
        })

    if model_rows:
        models_frame = pd.DataFrame(model_rows).sort_values("Model", kind="stable")
        st.dataframe(models_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No validation model results are recorded yet.")

with tabs[5]:
    options = [r for r in records if metrics(r).get("confusion_matrix")]
    if not options: st.info("No confusion matrices are recorded yet.")
    else:
        chosen = st.selectbox("Experiment", options, format_func=lambda r: f"{int(r['sequence_number']):04d} · {r.get('description', '')}", key="error_exp")
        data = metrics(chosen); classes = data.get("classes", [])
        cm=data.get("confusion_matrix",[])
        st.caption(f"Experiment {int(chosen['sequence_number']):04d} · {chosen.get('description', '')} · Internal ID: {chosen['experiment_id']}")
        st.plotly_chart(px.imshow(cm, x=classes, y=classes, labels={"x":"Predicted class", "y":"Actual class", "color":"Rows"}, aspect="auto"), use_container_width=True)
        recalls = data.get("per_class_recall", {})
        if recalls: st.plotly_chart(px.bar(pd.DataFrame({"Class": list(recalls), "Recall": list(recalls.values())}), x="Class", y="Recall"), use_container_width=True)
        reports = sorted(Path("artifacts/predictions").glob("ensemble_v0_*_metrics.json"))
        if reports:
            report_path = reports[-1]
            ensemble = json.loads(report_path.read_text())
            st.subheader("Model disagreement and unique corrections")
            rows = [{"Pair": pair, **values} for pair, values in ensemble.get("complementarity_all", {}).items()]
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Training and inference cost")
    cost = frame[frame["Runtime (s)"].notna()]
    if cost.empty: st.info("Runtime measurements are not available yet.")
    else:
        st.plotly_chart(px.bar(cost, x="Experiment", y="Runtime (s)", color="Model", hover_data=["Split"]), use_container_width=True)
        manifests = []
        for row in records:
            manifest = manifest_for(row)
            if manifest: manifests.append({"Experiment": f"{int(row['sequence_number']):04d}", "Description": row.get("description", ""), "Fit seconds": manifest.get("fit_seconds"),
                "Inference seconds": manifest.get("inference_seconds"), "Checkpoint": manifest.get("checkpoint_path")})
        if manifests: st.json(manifests)
        timing_fields = [
            ("Best checkpoint restore", "best_checkpoint_restoration_seconds"),
            ("Final validation inference", "final_validation_inference_seconds"),
            ("Final validation rows/sec", "final_validation_rows_per_second"),
            ("Effective inference batch", "effective_inference_batch_size"),
            ("OOM batch fallbacks", "inference_oom_fallbacks"),
            ("Best checkpoint snapshot", "best_checkpoint_state_snapshot_seconds"),
            ("Checkpoint serialization", "checkpoint_serialization_seconds"),
            ("Prediction export", "prediction_export_seconds"),
            ("Manifest export", "manifest_export_seconds"),
            ("Manifest and prediction export", "manifest_prediction_export_seconds"),
            ("Ensemble/catalog refresh", "downstream_refresh_seconds"),
            ("Model post-training stages", "model_post_training_seconds"),
            ("Total post-training", "total_post_training_seconds"),
        ]
        timing_rows = []
        for row in records:
            manifest = manifest_for(row)
            if not manifest:
                continue
            runtime = manifest.get("runtime_breakdown") or {}
            timing_rows.append({"Experiment": f"{int(row['sequence_number']):04d}",
                "Model": row.get("model"),
                **{label: (runtime[key] if key in runtime and runtime[key] is not None else "not recorded")
                   for label, key in timing_fields}})
        if timing_rows:
            st.subheader("Post-training stages (seconds)")
            timing_frame = pd.DataFrame(timing_rows)
            for label, _ in timing_fields:
                timing_frame[label] = pd.to_numeric(timing_frame[label], errors="coerce")
            st.dataframe(timing_frame, use_container_width=True, hide_index=True)
        epoch_runs = [row for row in records if manifest_for(row).get("validation_metrics_by_epoch")]
        if epoch_runs:
            chosen_run = st.selectbox("Validation accuracy by epoch", epoch_runs,
                format_func=lambda row: f"{int(row['sequence_number']):04d} · {row['model']}", key="epoch_validation_run")
            history = manifest_for(chosen_run).get("validation_metrics_by_epoch", [])
            selected_epoch = manifest_for(chosen_run).get("selected_epoch")
            selected_iteration = manifest_for(chosen_run).get("selected_iteration")
            selected_balanced_accuracy = manifest_for(chosen_run).get("selected_validation_balanced_accuracy")
            if selected_balanced_accuracy is None:
                selected_balanced_accuracy = manifest_for(chosen_run).get("validation_metrics", {}).get("balanced_accuracy")
            if selected_balanced_accuracy is not None:
                picked = (f"epoch {selected_epoch}" if selected_epoch is not None else
                          f"boosting iteration {selected_iteration}" if selected_iteration is not None else
                          "selected checkpoint")
                st.caption(f"Selected by highest validation balanced accuracy: {picked} · balanced accuracy {float(selected_balanced_accuracy):.4f}")
            run_manifest = manifest_for(chosen_run)
            baseline = run_manifest.get("baseline_validation_metrics") or {}
            history_rows = []
            if baseline:
                history_rows.append({"Epoch": "Baseline (pre-training)",
                    "Accuracy": baseline.get("accuracy"),
                    "Balanced accuracy": baseline.get("balanced_accuracy"),
                    "Accuracy change vs baseline (pp)": None,
                    "Validation loss (raw)": baseline.get("validation_loss_raw",
                                                           baseline.get("validation_loss"))})
            for item in history:
                accuracy = item.get("eval_accuracy")
                try:
                    accuracy_change = (100.0 * (float(accuracy) - float(baseline["accuracy"]))
                                       if accuracy is not None and baseline.get("accuracy") is not None else None)
                except (TypeError, ValueError):
                    accuracy_change = None
                history_rows.append({"Epoch": item.get("epoch"),
                    "Accuracy": accuracy, "Balanced accuracy": item.get("eval_balanced_accuracy"),
                    "Accuracy change vs baseline (pp)": accuracy_change,
                    "Validation loss (raw)": item.get("validation_loss_raw", item.get("eval_loss"))})
            history_frame = pd.DataFrame(history_rows)
            if not history_frame.empty:
                for column in ("Accuracy", "Balanced accuracy", "Accuracy change vs baseline (pp)",
                               "Validation loss (raw)"):
                    history_frame[column] = pd.to_numeric(history_frame[column], errors="coerce")
                epoch_numbers = pd.to_numeric(history_frame["Epoch"], errors="coerce")
                if selected_epoch is not None:
                    history_frame["Selected"] = epoch_numbers.round().eq(int(selected_epoch))
                st.dataframe(history_frame, use_container_width=True, hide_index=True)
                st.plotly_chart(px.line(history_frame.loc[epoch_numbers.notna()],
                                        x="Epoch", y="Accuracy", markers=True), use_container_width=True)

with tabs[7]:
    try:
        dataset_catalog = load_dataset_catalog(PROJECT_ROOT / "configs/datasets.yaml")
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the dataset catalog: {exc}")
    else:
        dataset_entries = dataset_catalog["datasets"]
        if not dataset_entries:
            st.info("Add dataset entries to configs/datasets.yaml.")
        else:
            entries_by_id = {entry["id"]: entry for entry in dataset_entries}
            selected_id = st.selectbox(
                "Dataset",
                list(entries_by_id),
                format_func=lambda dataset_id: entries_by_id[dataset_id]["name"],
                key="data_tab_dataset_id",
            )
            entry = entries_by_id[selected_id]
            artifact_path = PROJECT_ROOT / entry["artifact"]
            selected_profile_path = PROJECT_ROOT / entry["profile"]
            st.caption(f"Dataset {selected_id} · {entry['artifact']}")
            st.markdown(entry["description"])
            st.caption(f"Validation: {entry['validation']}")
            if not artifact_path.is_file():
                st.warning(f"Dataset artifact not found: {entry['artifact']}")
            if not selected_profile_path.is_file():
                st.warning(f"Dataset profile not found: {entry['profile']}")
                data = {}
            else:
                try:
                    data = json.loads(selected_profile_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    st.warning(f"Could not read dataset profile {entry['profile']}: {exc}")
                    data = {}
            if data:
                st.caption(f"{data.get('rows', '—')} rows · SHA-256 {data.get('dataset_sha256', '—')}")
                st.json(data)
                counts = data.get("class_counts", {})
                if counts:
                    st.plotly_chart(px.bar(pd.DataFrame({"Class": list(counts), "Rows": list(counts.values())}),
                                           x="Class", y="Rows"), use_container_width=True)
                lengths = data.get("text_length_chars", {})
                if lengths:
                    st.dataframe(pd.DataFrame([lengths], index=["Character length"]), use_container_width=True)
            reports = entry.get("reports", [])
            if reports:
                report_links = [f"[{Path(report).name}](../{report})" for report in reports]
                st.markdown("Reports: " + " · ".join(report_links))


with tabs[8]:
    render_ensemble_catalog(PROJECT_ROOT)
