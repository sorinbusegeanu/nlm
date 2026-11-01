"""Streamlit report for the local experiment registry and stored predictions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from src.registry import Registry

st.set_page_config(page_title="Name That LLM experiments", layout="wide")
st.title("Name That LLM — experiment results")
registry_path = Path("experiments/registry.sqlite")
records = Registry(registry_path).rows() if registry_path.exists() else []
if not records:
    st.info("No experiments recorded. Prepare V0 and train at least one model to populate this dashboard.")
    st.stop()

def metrics(row): return json.loads(row["metrics_json"])

def manifest_for(row):
    path = row.get("artifact_path")
    if path and Path(path).name.endswith("_oof.parquet"):
        candidates = [Path(path).with_name(f"{row['experiment_id']}_manifest.json")]
        for candidate in candidates:
            if candidate.exists(): return json.loads(candidate.read_text())
    return {}

comparison = []
for row in records:
    value = metrics(row)
    comparison.append({"Experiment": row["experiment_id"], "Model": row["model"], "Split": row["split"],
        "Balanced accuracy (raw)": value.get("balanced_accuracy"), "Balanced accuracy (calibrated)": value.get("calibrated_balanced_accuracy"), "Macro F1": value.get("macro_f1"),
        "Log loss": value.get("log_loss"), "Runtime (s)": row.get("runtime_seconds"),
        "Seed": row.get("seed"), "Created": row.get("created_at")})
frame = pd.DataFrame(comparison)
cv = frame[frame.Split.eq("development_oof") & frame["Balanced accuracy"].notna()]
holdout = frame[frame.Split.eq("holdout") & frame["Balanced accuracy"].notna()]
columns = st.columns(4)
columns[0].metric("Experiments", len(records))
columns[1].metric("Models", int(frame.Model.nunique()))
columns[2].metric("Best development OOF", f"{cv['Balanced accuracy'].max():.4f}" if not cv.empty else "—")
columns[3].metric("Holdout (reported separately)", f"{holdout['Balanced accuracy'].max():.4f}" if not holdout.empty else "Not evaluated")

tabs = st.tabs(["Overview", "Errors", "Ablations", "Compute", "Data"])
with tabs[0]:
    st.subheader("Experiment comparison")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if not cv.empty:
        st.plotly_chart(px.bar(cv, x="Experiment", y="Balanced accuracy", color="Model", hover_data=["Split", "Seed"]), use_container_width=True)
    st.download_button("Download experiment table (CSV)", frame.to_csv(index=False), "experiments.csv", "text/csv")
    st.download_button("Download experiment table (JSON)", frame.to_json(orient="records", indent=2), "experiments.json", "application/json")

with tabs[1]:
    options = [r for r in records if metrics(r).get("confusion_matrix")]
    if not options: st.info("No confusion matrices are recorded yet.")
    else:
        chosen = st.selectbox("Experiment", options, format_func=lambda r: f"{r['experiment_id']} · {r['split']}", key="error_exp")
        data = metrics(chosen); classes = data.get("classes", [])
        metric_view=st.radio("Prediction view",["Raw scores","Calibrated probabilities"],horizontal=True)
        calibrated=metric_view=="Calibrated probabilities"
        cm=data.get("calibrated_confusion_matrix",[]) if calibrated else data.get("confusion_matrix",[])
        st.caption(f"Source split: {chosen['split']} · Experiment ID: {chosen['experiment_id']}")
        st.plotly_chart(px.imshow(cm, x=classes, y=classes, labels={"x":"Predicted class", "y":"Actual class", "color":"Rows"}, aspect="auto"), use_container_width=True)
        recalls = data.get("calibrated_per_class_recall", {}) if calibrated else data.get("per_class_recall", {})
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
    ablations = cv[cv.Model.ne("ensemble")]
    if ablations.empty: st.info("Train multiple feature/model configurations to compare ablations.")
    else:
        baseline = st.selectbox("Reference experiment", ablations.Experiment.tolist())
        baseline_score = float(ablations.loc[ablations.Experiment.eq(baseline), "Balanced accuracy"].iloc[0])
        ablations = ablations.copy(); ablations["Delta vs reference"] = ablations["Balanced accuracy"] - baseline_score
        st.caption(f"Development OOF · Reference: {baseline} · balanced accuracy {baseline_score:.4f}")
        st.dataframe(ablations, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(ablations, x="Experiment", y="Delta vs reference", color="Model"), use_container_width=True)
    ensemble_files = sorted(Path("artifacts/predictions").glob("ensemble_v0_*_metrics.json"))
    if ensemble_files:
        ensemble_file = ensemble_files[-1]
        data = json.loads(ensemble_file.read_text())
        blend_rows = []
        for name, value in data.get("ensembles", {}).items():
            ci=value.get("paired_bootstrap_vs_best_single", {})
            blend_rows.append({"Ensemble": name, **value["metrics"],
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
            if manifest: manifests.append({"Experiment": row["experiment_id"], "Fit seconds": manifest.get("fit_seconds"),
                "Inference seconds per fold": manifest.get("inference_seconds_by_fold"), "Checkpoints": manifest.get("checkpoint_paths")})
        if manifests: st.json(manifests)

with tabs[4]:
    profile = Path("data/processed/dataset_profile.json")
    if not profile.exists(): st.info("Run dataset preparation to create the V0 profile.")
    else:
        data = json.loads(profile.read_text())
        st.caption(f"Dataset {data.get('dataset_version')} · SHA-256 {data.get('dataset_sha256')}")
        st.json(data)
        counts = data.get("class_counts", {})
        if counts: st.plotly_chart(px.bar(pd.DataFrame({"Class": list(counts), "Rows": list(counts.values())}), x="Class", y="Rows"), use_container_width=True)
        lengths = data.get("text_length_chars", {})
        if lengths: st.dataframe(pd.DataFrame([lengths], index=["Character length"]), use_container_width=True)
