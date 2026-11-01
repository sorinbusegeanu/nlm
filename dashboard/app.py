"""Streamlit report for the local experiment registry and stored predictions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.registry import Registry

st.set_page_config(page_title="Name That LLM experiments", layout="wide")
st.title("Name That LLM — experiment results")
registry_path = Path("experiments/registry.sqlite")
records = Registry(registry_path).rows() if registry_path.exists() else []
records = sorted(records, key=lambda row: int(row.get("sequence_number") or 0), reverse=True)
if not records:
    st.info("No experiments recorded. Prepare V0 and train at least one model to populate this dashboard.")
    st.stop()

profile_path = Path("data/processed/dataset_profile.json")
profile_data = json.loads(profile_path.read_text()) if profile_path.exists() else {}
profile_validation_rows = profile_data.get("validation_rows")

def metrics(row): return json.loads(row["metrics_json"])

def model_display(name):
    return {"qwen06_best_epoch03": "Qwen3-0.6B best checkpoint (epoch 3)"}.get(name, name)

def blend_label(name):
    if not name: return ""
    train_weighted = name.startswith("train_weighted:")
    if train_weighted: name = name.removeprefix("train_weighted:")
    labels = {"svc": "SVC", "ultrabert380": "UltraBERT-380M", "qwen06": "Qwen3-0.6B",
              "qwen06_best_epoch03": "Qwen3-0.6B best checkpoint (epoch 3)",
              "qwen17": "Qwen3-1.7B", "qwen4": "Qwen3-4B", "llama32_3b": "Llama-3.2-3B"}
    model_ids = sorted(labels, key=len, reverse=True)
    members = []
    for part in name.split("+"):
        model_id = next((item for item in model_ids if part == item or part.startswith(item + "_")), None)
        members.append(labels.get(model_id, part.split("_", 1)[0]))
    kind = "Train-fitted weighted blend" if train_weighted else "Equal-weight blend"
    return f"{kind}: {' + '.join(members)}"

def manifest_for(row):
    path = row.get("artifact_path")
    if path and Path(path).name.endswith("_validation.parquet"):
        candidates = [Path(path).with_name(f"{row['experiment_id']}_manifest.json")]
        for candidate in candidates:
            if candidate.exists(): return json.loads(candidate.read_text())
    return {}

comparison = []
for row in records:
    value = metrics(row)
    weighted_accuracy = None
    if value.get("ensembles"):
        weighted = next((item for name, item in value["ensembles"].items()
                         if name.startswith("train_weighted:")), None)
        if weighted:
            weighted_accuracy = weighted.get("metrics", {}).get("balanced_accuracy")
    row_validation_rows = value.get("validation_rows", manifest_for(row).get("validation_rows"))
    if row_validation_rows is None or pd.isna(row_validation_rows):
        row_validation_rows = profile_validation_rows
    comparison.append({"Experiment": f"{int(row['sequence_number']):04d}", "Description": row.get("description", ""),
        "Model": model_display(row["model"]), "Split": row["split"],
        "acc_raw": value.get("balanced_accuracy"), "acc_weighted": weighted_accuracy, "Macro F1": value.get("macro_f1"),
        "Log loss": value.get("log_loss"), "Runtime (s)": row.get("runtime_seconds"),
        "Validation rows": row_validation_rows,
        "Seed": row.get("seed"), "Created": row.get("created_at")})
frame = pd.DataFrame(comparison)
validation = frame[frame.Split.eq("validation") & frame["acc_raw"].notna()]
columns = st.columns(4)
columns[0].metric("Experiments", len(records))
columns[1].metric("Models", int(frame.Model.nunique()))
columns[2].metric("Best validation accuracy", f"{validation['acc_raw'].max():.4f}" if not validation.empty else "—")
columns[3].metric("Validation rows", int(profile_validation_rows) if profile_validation_rows is not None else "—")

tabs = st.tabs(["Overview", "Errors", "Ablations", "Compute", "Data"])
with tabs[0]:
    st.subheader("Experiment comparison")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if not validation.empty:
        st.plotly_chart(px.bar(validation, x="Experiment", y="acc_raw", color="Model", hover_data=["Split", "Seed"]), use_container_width=True)
    st.download_button("Download experiment table (CSV)", frame.to_csv(index=False), "experiments.csv", "text/csv")
    st.download_button("Download experiment table (JSON)", frame.to_json(orient="records", indent=2), "experiments.json", "application/json")

with tabs[1]:
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

with tabs[2]:
    st.subheader("Ablations and model additions")
    ablations = validation[~validation.Model.str.contains("ensemble", case=False)]
    ensemble_files = sorted(Path("artifacts/predictions").glob("ensemble_v0_*_metrics.json"))
    if ablations.empty: st.info("Train multiple feature/model configurations to compare ablations.")
    else:
        baseline = st.selectbox("Reference experiment", ablations.Experiment.tolist())
        baseline_score = float(ablations.loc[ablations.Experiment.eq(baseline), "acc_raw"].iloc[0])
        ablations = ablations.copy(); ablations["Delta vs reference"] = ablations["acc_raw"] - baseline_score
        st.caption(f"Validation · Reference: {baseline} · balanced accuracy {baseline_score:.4f}")
        st.dataframe(ablations, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(ablations, x="Experiment", y="Delta vs reference", color="Model"), use_container_width=True)
    if ensemble_files:
        ensemble_file = ensemble_files[-1]
        data = json.loads(ensemble_file.read_text())
        blend_rows = []
        for name, value in data.get("ensembles", {}).items():
            ci=value.get("paired_bootstrap_vs_best_single", {})
            metrics_without_members = {k: v for k, v in value["metrics"].items() if k != "members"}
            blend_rows.append({"Blend": blend_label(name), "Weights": value.get("weights"), **metrics_without_members,
                "95% delta lower": ci.get("lower"), "95% delta upper": ci.get("upper"),
                "marginal contribution": value.get("marginal_contribution")})
        if blend_rows: st.dataframe(pd.DataFrame(blend_rows), use_container_width=True, hide_index=True)

with tabs[3]:
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
            history_frame = pd.DataFrame([{"Epoch": item.get("epoch"),
                "Accuracy": item.get("eval_accuracy"), "Balanced accuracy": item.get("eval_balanced_accuracy"),
                "Validation loss": item.get("eval_loss")} for item in history])
            if not history_frame.empty:
                st.dataframe(history_frame, use_container_width=True, hide_index=True)
                st.plotly_chart(px.line(history_frame, x="Epoch", y="Balanced accuracy", markers=True), use_container_width=True)

with tabs[4]:
    if not profile_path.exists(): st.info("Run dataset preparation to create the V0 profile.")
    else:
        data = profile_data
        st.caption(f"Dataset {data.get('dataset_version')} · SHA-256 {data.get('dataset_sha256')}")
        st.json(data)
        counts = data.get("class_counts", {})
        if counts: st.plotly_chart(px.bar(pd.DataFrame({"Class": list(counts), "Rows": list(counts.values())}), x="Class", y="Rows"), use_container_width=True)
        lengths = data.get("text_length_chars", {})
        if lengths: st.dataframe(pd.DataFrame([lengths], index=["Character length"]), use_container_width=True)
