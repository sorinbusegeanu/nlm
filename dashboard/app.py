"""Local experiment dashboard. Requires optional dashboard dependencies."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from src.registry import Registry

st.set_page_config(page_title="Name That LLM experiments", layout="wide")
st.title("Name That LLM — experiments")
registry_path = Path("experiments/registry.sqlite")
rows = Registry(registry_path).rows() if registry_path.exists() else []
if not rows:
    st.info("No experiments recorded yet. Prepare the dataset and train a model to populate this dashboard.")
else:
    records=[]
    for row in rows:
        metrics=json.loads(row["metrics_json"])
        records.append({"Experiment":row["experiment_id"],"Model":row["model"],"Split":row["split"],"Balanced accuracy":metrics.get("balanced_accuracy"),"Macro F1":metrics.get("macro_f1"),"Log loss":metrics.get("log_loss"),"Created":row["created_at"]})
    frame=pd.DataFrame(records)
    st.subheader("Overview and experiment comparison")
    st.dataframe(frame,use_container_width=True,hide_index=True)
    usable=frame.dropna(subset=["Balanced accuracy"])
    if not usable.empty: st.plotly_chart(px.bar(usable,x="Experiment",y="Balanced accuracy",color="Split",hover_data=["Model"]),use_container_width=True)
    selected=st.selectbox("Experiment",[x["experiment_id"] for x in rows])
    row=next(x for x in rows if x["experiment_id"]==selected)
    metrics=json.loads(row["metrics_json"]); st.caption(f"Source split: {row['split']} · Experiment ID: {selected}")
    tabs=st.tabs(["Class recalls","Confusion matrix","Ablations & costs","Data profile"])
    with tabs[0]:
        recalls=metrics.get("per_class_recall",{})
        if recalls: st.plotly_chart(px.bar(pd.DataFrame({"Class":list(recalls),"Recall":list(recalls.values())}),x="Class",y="Recall"),use_container_width=True)
        else: st.info("Class recall data is not available for this experiment.")
    with tabs[1]:
        cm=metrics.get("confusion_matrix"); classes=metrics.get("classes")
        if cm and classes: st.plotly_chart(px.imshow(cm,x=classes,y=classes,labels={"x":"Predicted","y":"Actual","color":"Count"}),use_container_width=True)
        else: st.info("Confusion matrix is not available for this experiment.")
    with tabs[2]:
        st.json({"config":json.loads(row["config_json"]),"runtime_seconds":metrics.get("fit_seconds")})
        st.download_button("Download experiment metrics",json.dumps(metrics,indent=2),file_name=f"{selected}.json",mime="application/json")
    with tabs[3]:
        profile=Path("data/processed/dataset_profile.json")
        if profile.exists():
            data=json.loads(profile.read_text());st.json(data)
            counts=data.get("class_counts",{})
            if counts:st.plotly_chart(px.bar(pd.DataFrame({"Class":list(counts),"Rows":list(counts.values())}),x="Class",y="Rows"),use_container_width=True)
        else: st.info("Prepare dataset V0 to create the data profile.")
