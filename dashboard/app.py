"""Streamlit report for the local experiment registry and stored predictions."""
from __future__ import annotations

import json
import os
import ast
import re
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.registry import Registry
from dashboard.training_lineage import training_lineage

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

profile_path = Path("data/processed/dataset_profile.json")
profile_data = json.loads(profile_path.read_text()) if profile_path.exists() else {}
profile_validation_rows = profile_data.get("validation_rows")
DATASET_PROFILES = {
    "V0 — Kaggle baseline": ("data/processed/dataset_profile.json", "data/processed/dataset_v0.parquet"),
    "V1 — Balanced Open Pool": ("data/processed/dataset_profile_v1.json", "data/processed/dataset_v1.parquet"),
    "V1-all": ("data/processed/dataset_profile_v1_all.json", "data/processed/dataset_v1_all.parquet"),
    "V1.2-BM": ("data/processed/dataset_profile_v1.2_bm.json", "data/processed/dataset_v1.2_bm.parquet"),
    "V1.3": ("data/processed/dataset_profile_v1.3.json", "data/processed/dataset_v1.3.parquet"),
}

def metrics(row): return json.loads(row["metrics_json"])

MODEL_ALIASES = {"qwen06_best_epoch03": "qwen06"}

def canonical_model(name):
    return MODEL_ALIASES.get(name, name)

def model_display(name):
    name = canonical_model(name)
    return {"qwen06": "qwen3-0.6B", "qwen17": "qwen3-1.7B", "qwen4": "qwen3-4.0B",
            "ultrabert380": "UltraBERT-380M", "phi35_mini": "Phi-3.5 Mini",
            "embeddinggemma2": "EmbeddingGemma 2",
            "granite4_350m": "Granite 4.0 350M Base",
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
            if len(command) < 2 or command[0] != "models" or command[1] not in {"train", "continue-qwen", "continue-encoder"}:
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
            model = "qwen06" if action == "continue-qwen" else options.get("--model", "svc")
            if action == "continue-encoder":
                config = {
                    "model": model,
                    "epochs": int(options.get("--epochs", 10)),
                    "learning_rate": float(options.get("--learning-rate", 5e-5)),
                    "learning_rate_decay": float(options.get("--learning-rate-decay", 0.9)),
                    "training_data": options.get("--dataset", "v0"),
                    "validation_data": options.get("--dataset", "v0"),
                }
            else:
                config_path = Path(options.get("--config", f"configs/models/{model}.yaml"))
                if not config_path.is_absolute():
                    config_path = root / config_path
                import yaml
                config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {"model": model}
                if action == "continue-qwen":
                    config = {**config, "epochs": int(options.get("--epochs", 4))}
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
            return {"pid": int(entry.name), "model": model, "config": config,
                    "action": action, "log_path": log_path, "progress_path": progress_path}
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

@st.fragment(run_every="10s")
def live_training_banner():
    live = active_training(Path(__file__).resolve().parents[1])
    if live is None:
        st.caption("No model training process is currently running.")
        return
    config = live["config"]
    title = config.get("experiment_name") or live["model"]
    epoch_limit = int(config.get("epochs", 0) or 0)
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
    current_epoch = min(completed_epochs + 1, epoch_limit) if epoch_limit else None
    step = latest_training_step(live.get("log_path"))
    epoch_label = (f"epoch {current_epoch}/{epoch_limit} max" if current_epoch is not None
                   else "epoch unknown")
    if step and step[1]:
        st.progress(min(step[0] / step[1], 1.0),
                    text=f"Training: {title} · {model_display(live['model'])} · {epoch_label} · step {step[0]:,}/{step[1]:,}")
    else:
        st.info(f"Training in progress: {title} · {model_display(live['model'])} · {epoch_label} · PID {live['pid']}")

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

@st.fragment(run_every="10s")
def render_current_training():
    st.subheader("Current training")
    live_training = active_training(Path(__file__).resolve().parents[1])
    if live_training is None:
        st.caption("No model is currently training.")
        return
    config = live_training["config"]
    st.markdown(f"**Model:** {model_display(live_training['model'])}")
    st.markdown(f"**Epochs:** {config.get('epochs', '—')}")
    st.markdown("**Training parameters**")
    if config:
        batch_probe = {}
        progress_path = live_training.get("progress_path")
        if progress_path:
            probe_path = Path(progress_path).with_name("batch_size_probe.json")
            try:
                if probe_path.is_file():
                    batch_probe = json.loads(probe_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                batch_probe = {}
        requested_batch = batch_probe.get("requested_batch_size", config.get("batch_size"))
        effective_batch = batch_probe.get("effective_batch_size",
                                          batch_probe.get("selected_batch_size", requested_batch))
        accumulation = batch_probe.get("gradient_accumulation_steps",
                                       config.get("gradient_accumulation_steps", 1))
        params = [{"Parameter": key, "Value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                  for key, value in config.items() if key not in {"model", "epochs", "batch_size"}]
        params.extend([
            {"Parameter": "Batch size (requested per GPU)", "Value": requested_batch},
            {"Parameter": "Batch size (effective per GPU)", "Value": effective_batch},
            {"Parameter": "Gradient accumulation steps", "Value": accumulation},
            {"Parameter": "Effective examples per optimizer step",
             "Value": int(effective_batch) * int(accumulation) if effective_batch is not None else None},
        ])
        st.dataframe(params, use_container_width=True, hide_index=True)
    else:
        st.caption("No training parameters are available.")
    epoch_count = int(config.get("epochs", 0) or 0)
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
                           "Accuracy": baseline.get("accuracy"), "Train error": None,
                           "Validation error": baseline.get("validation_loss")})
    for epoch in range(1, epoch_count + 1):
        item = history_by_epoch.get(epoch, {})
        epoch_rows.append({
            "Epoch": epoch,
            "Learning rate": item.get("learning_rate", float(learning_rate) * decay ** (epoch - 1)
                                       if learning_rate is not None else None),
            "Accuracy": item.get("accuracy", item.get("eval_accuracy")),
            "Train error": item.get("train_loss", item.get("loss")),
            "Validation error": item.get("validation_loss", item.get("eval_loss")),
        })
    st.markdown("**Epochs**")
    st.caption("Accuracy and validation error appear after each epoch evaluation; train error appears when that epoch's training log is written.")
    st.dataframe(pd.DataFrame(epoch_rows, columns=["Epoch", "Learning rate", "Accuracy", "Train error", "Validation error"]),
                 use_container_width=True, hide_index=True)

def render_prediction_factory():
    st.subheader("Checkpoint accuracy by model label")
    root = Path(__file__).resolve().parents[1]
    trust_list_path = root / "docs" / "checkpoint_label_trust_list.csv"
    trust_metadata_path = root / "docs" / "checkpoint_label_trust_list.json"
    try:
        if not trust_list_path.is_file():
            st.info("The versioned checkpoint-label trust list has not been generated yet.")
            return
        catalog = pd.read_csv(trust_list_path)
        if catalog.empty:
            st.info("The checkpoint-label trust list is empty.")
            return
        metadata = json.loads(trust_metadata_path.read_text()) if trust_metadata_path.is_file() else {}
        if "SeqID" not in catalog or catalog["SeqID"].isna().any():
            raise ValueError("versioned trust list must assign a SeqID to every checkpoint-label row")
        catalog["SeqID"] = pd.to_numeric(catalog["SeqID"], errors="raise").astype(int)
        catalog = catalog.sort_values("SeqID", kind="stable").reset_index(drop=True)
        catalog["prediction_count"] = pd.to_numeric(catalog["prediction_count"], errors="coerce")
        catalog["validation_training_overlap"] = pd.to_numeric(
            catalog["validation_training_overlap"], errors="coerce").fillna(0)
        catalog["checkpoint_model_display"] = catalog["model"].map(model_display)
        catalog["validation_note"] = catalog.apply(
            lambda row: (f"V0 training overlap: {int(row['validation_training_overlap'])} rows"
                         if row["validation_training_overlap"] else
                         (str(row.get("evaluation_note") or "Not eligible for ranking")
                          if str(row.get("evaluation_eligible", "True")).lower() not in {"true", "1"} else "")),
            axis=1)
        catalog["checkpoint_status"] = catalog["checkpoint_available"].map(
            lambda value: "Available" if str(value).lower() in {"true", "1"} else "Missing")
        catalog["rule_eligible_display"] = catalog["rule_eligible"].map(
            lambda value: "Yes" if str(value).lower() in {"true", "1"} else "No")
        catalog = catalog.rename(columns={
            "sequence_id": "ID", "target_label": "Model label",
            "checkpoint_model_display": "Checkpoint model", "checkpoint": "Checkpoint",
            "dataset_lineage": "Training dataset", "checkpoint_date": "Date (MM.DD)",
            "precision": "Label accuracy (precision)", "recall": "Recall",
            "prediction_count": "Predicted rows", "validation_note": "Validation note",
            "checkpoint_status": "Checkpoint status", "rule_eligible_display": "Rule eligible",
        })
        columns = ["SeqID", "ID", "Model label", "Checkpoint model", "Checkpoint", "Training dataset",
                   "Date (MM.DD)", "Label accuracy (precision)", "Recall", "Predicted rows",
                   "Rule eligible", "Checkpoint status", "Validation note"]
        table = catalog[columns].style.format({
            "SeqID": "{:.0f}",
            "Label accuracy (precision)": "{:.1%}", "Recall": "{:.1%}",
            "Predicted rows": "{:.0f}",
        }, na_rep="—")
        generated = metadata.get("generated_at_utc", "unknown generation time")
        version = metadata.get("version", catalog.trust_list_version.iloc[0])
        st.caption(f"Trust-list reference: {version}, generated {generated}. SeqID is the row's order in this complete ranked list; ID identifies the checkpoint. ZOD03 skips rows that fail support eligibility or its repeated-label rule. Metrics use frozen V0 validation.")
        st.dataframe(table, use_container_width=True, hide_index=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not read checkpoint-label trust list: {exc}")

columns = st.columns(4)
columns[0].metric("Experiments", len(records))
columns[1].metric("Models", int(frame.Model.nunique()))
columns[2].metric("Best validation accuracy", f"{validation['acc_raw'].max():.4f}" if not validation.empty else "—")
columns[3].metric("Validation rows", int(profile_validation_rows) if profile_validation_rows is not None else "—")

tabs = st.tabs(["Overview", "Experiments", "Prediction factory", "Current training", "Models", "Errors", "Compute", "Data"])
with tabs[0]:
    st.subheader("Experiment comparison")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if not validation.empty:
        st.plotly_chart(px.bar(validation, x="Experiment", y="acc_raw", color="Model", hover_data=["Split", "Seed"]), use_container_width=True)
    st.download_button("Download experiment table (CSV)", frame.to_csv(index=False), "experiments.csv", "text/csv")
    st.download_button("Download experiment table (JSON)", frame.to_json(orient="records", indent=2), "experiments.json", "application/json")
with tabs[1]:
    experiments = pd.DataFrame([{
        "ID": 1,
        "Name": "zod_04",
        "Description": (
            "Existing V1-all → V0 best UltraBERT checkpoint; extracts and caches all 24 "
            "attention-mask-pooled layer representations with aligned V0 IDs and labels, "
            "ranks layers using five-fold prompt-grouped CV, and fits 28 final classifiers: "
            "24 individual layers and four multi-layer combinations."
        ),
        "Report": "?experiment_report=zod_04",
    }])
    st.dataframe(experiments, use_container_width=True, hide_index=True,
        column_config={"Report": st.column_config.LinkColumn(
            "Report", display_text="docs/zod_04.md")})

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
        # For checkpoint-selected runs, the manifest records the best epoch's
        # validation score. Prefer it when choosing the displayed model maximum;
        # stale registry scores from older prediction-order bugs can otherwise
        # hide the selected checkpoint's actual validation result.
        accuracy = run_manifest.get("selected_validation_accuracy")
        if accuracy is None:
            history = run_manifest.get("validation_metrics_by_epoch", [])
            history_scores = [item.get("eval_balanced_accuracy", item.get("eval_accuracy"))
                              for item in history]
            history_scores = [float(value) for value in history_scores if value is not None]
            accuracy = max(history_scores) if history_scores else metrics(row).get("balanced_accuracy")
        if accuracy is None or pd.isna(accuracy):
            continue
        # records are newest first, so ties keep the most recent run.
        if model not in best_runs or float(accuracy) > best_runs[model]["accuracy"]:
            best_runs[model] = {"row": row, "accuracy": float(accuracy)}

    model_rows = []
    for model, best in best_runs.items():
        row = best["row"]
        manifest = manifest_for(row)
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
            "Best accuracy (balanced)": best["accuracy"],
            "Train loss": selected_history.get("train_loss") if selected_history else None,
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
        epoch_runs = [row for row in records if manifest_for(row).get("validation_metrics_by_epoch")]
        if epoch_runs:
            chosen_run = st.selectbox("Validation accuracy by epoch", epoch_runs,
                format_func=lambda row: f"{int(row['sequence_number']):04d} · {row['model']}", key="epoch_validation_run")
            history = manifest_for(chosen_run).get("validation_metrics_by_epoch", [])
            selected_epoch = manifest_for(chosen_run).get("selected_epoch")
            selected_iteration = manifest_for(chosen_run).get("selected_iteration")
            selected_accuracy = manifest_for(chosen_run).get("selected_validation_accuracy")
            if selected_accuracy is not None:
                picked = (f"epoch {selected_epoch}" if selected_epoch is not None else
                          f"boosting iteration {selected_iteration}" if selected_iteration is not None else
                          "selected checkpoint")
                st.caption(f"Selected by highest validation accuracy: {picked} · accuracy {float(selected_accuracy):.4f}")
            history_frame = pd.DataFrame([{"Epoch": item.get("epoch"),
                "Accuracy": item.get("eval_accuracy"), "Balanced accuracy": item.get("eval_balanced_accuracy"),
                "Validation loss": item.get("eval_loss")} for item in history])
            if not history_frame.empty:
                if selected_epoch is not None:
                    history_frame["Selected"] = history_frame["Epoch"].round().eq(int(selected_epoch))
                st.dataframe(history_frame, use_container_width=True, hide_index=True)
                st.plotly_chart(px.line(history_frame, x="Epoch", y="Accuracy", markers=True), use_container_width=True)

with tabs[7]:
    available_datasets = [name for name, (profile, _) in DATASET_PROFILES.items() if Path(profile).exists()]
    if not available_datasets:
        st.info("Run dataset preparation to create dataset profiles.")
    else:
        selected_dataset = st.selectbox("Dataset", available_datasets, key="data_tab_dataset")
        profile_file, dataset_file = DATASET_PROFILES[selected_dataset]
        data = json.loads(Path(profile_file).read_text())
        st.caption(f"Dataset {data.get('dataset_version')} · {dataset_file} · SHA-256 {data.get('dataset_sha256')}")
        if not Path(dataset_file).exists(): st.warning(f"Dataset artifact not found: {dataset_file}")
        st.json(data)
        counts = data.get("class_counts", {})
        if counts: st.plotly_chart(px.bar(pd.DataFrame({"Class": list(counts), "Rows": list(counts.values())}), x="Class", y="Rows"), use_container_width=True)
        lengths = data.get("text_length_chars", {})
        if lengths: st.dataframe(pd.DataFrame([lengths], index=["Character length"]), use_container_width=True)
