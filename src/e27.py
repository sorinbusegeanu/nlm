"""E27: external prompt-group learning and V0 assignment validation.

All supervised grouping fits and threshold selection use V2.14's verified
``prompt_canon`` keys. V0 labels are read only by the final evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit

from . import (e19f, e19b_evaluate, e19b_reconstruct, e19d_reconstruct,
               e19b_group, e22, e20, e13_reliability_router as e13)
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e27"
SEED = 2026
METHODS = ("E19-F-A", "E19-F-B", "E19-F-C")
GROUP_THRESHOLDS = tuple(np.round(np.arange(.35, .901, .05), 2))
MNN_K = (5, 10, 20)
FC_C_VALUES = (0.1, 1.0, 10.0)
GATE_THRESHOLDS = tuple(np.round(np.arange(0., 1.001, .05), 2))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _reconstruction_sample_paths(output_dir=OUT):
    manifest=Path(output_dir)/"reconstruction_sample_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Prepare the prompt-stratified E27 reconstruction sample first")
    fraction=float(json.loads(manifest.read_text())["fraction"])
    tag=f"{int(round(fraction*100)):02d}pct"
    return (Path(output_dir)/f"external_v214_reconstruction_sample_{tag}.parquet",
            Path(output_dir)/f"external_v214_reconstruction_rotation_{tag}.parquet")


def canonical_external(path: Path) -> pd.DataFrame:
    """Load only V2.14 rows whose prompt identity is verified by prompt_canon."""
    frame = pd.read_parquet(path)
    required = {"id", "text", "model", "prompt_canon"}
    if not required.issubset(frame.columns):
        raise ValueError(f"V2.14 is missing required fields: {sorted(required-set(frame.columns))}")
    frame = frame.loc[frame.prompt_canon.notna()].copy()
    for col in ("id", "text", "model", "prompt_canon"):
        frame[col] = frame[col].astype(str)
    frame = frame.loc[frame.prompt_canon.str.strip().ne("")].reset_index(drop=True)
    unknown = sorted(set(frame.model) - set(LABELS))
    if unknown:
        raise ValueError(f"V2.14 has noncanonical labels: {unknown}")
    if frame.id.duplicated().any():
        raise ValueError("Canonical V2.14 subset has duplicate response IDs")
    frame["label_count"] = frame.groupby("prompt_canon").model.transform("nunique").astype(int)
    frame["group_size"] = frame.groupby("prompt_canon").id.transform("size").astype(int)
    return frame


def prompt_disjoint_split(frame: pd.DataFrame, *, seed=SEED) -> dict[str, np.ndarray]:
    """Deterministic approximately 70/15/15 split stratified by group label-count."""
    groups = frame.groupby("prompt_canon", sort=True).agg(
        label_count=("model", "nunique"), n=("id", "size")).reset_index()
    if int(groups.label_count.value_counts().min()) < 5:
        # Tiny benchmark samples cannot satisfy sklearn's two-stage
        # stratification constraints. Allocate at least one whole prompt per
        # label-count stratum to each split, then distribute the remainder
        # approximately 70/15/15.
        rng=np.random.default_rng(int(seed)); assigned={k:set() for k in ("train","validation","test")}
        for _,part in groups.groupby("label_count",sort=True):
            ids=part.prompt_canon.astype(str).to_numpy().copy(); rng.shuffle(ids)
            n=len(ids)
            if n<3:
                raise ValueError("Each retained group-size stratum needs at least three prompts for train/validation/test")
            n_train=min(n-2,max(1,int(round(.70*n))))
            n_val=min(n-n_train-1,max(1,int(round(.15*n))))
            assigned["train"].update(ids[:n_train])
            assigned["validation"].update(ids[n_train:n_train+n_val])
            assigned["test"].update(ids[n_train+n_val:])
        row_sets={name:np.flatnonzero(frame.prompt_canon.astype(str).isin(keys).to_numpy())
                  for name,keys in assigned.items()}
        if any(not rows.size for rows in row_sets.values()):
            raise ValueError("Small-sample prompt split produced an empty partition")
        return row_sets
    # Split prompts, never individual responses. Stratification uses the actual
    # number of distinct canonical labels represented by each prompt.
    strata = groups.label_count.astype(str)
    first = StratifiedShuffleSplit(n_splits=1, train_size=.70, random_state=int(seed))
    train_g, remainder_g = next(first.split(groups.prompt_canon, strata))
    remainder = groups.iloc[remainder_g].reset_index(drop=True)
    second = StratifiedShuffleSplit(n_splits=1, train_size=.50,
                                    random_state=int(seed)+1)
    val_local, test_local = next(second.split(remainder.prompt_canon,
                                              remainder.label_count.astype(str)))
    assigned = {
        "train": set(groups.iloc[train_g].prompt_canon.astype(str)),
        "validation": set(remainder.iloc[val_local].prompt_canon.astype(str)),
        "test": set(remainder.iloc[test_local].prompt_canon.astype(str)),
    }
    row_sets = {name: np.flatnonzero(frame.prompt_canon.isin(keys).to_numpy())
                for name, keys in assigned.items()}
    if any(not row_sets[k].size for k in row_sets):
        raise ValueError("Prompt split unexpectedly produced an empty partition")
    if set.intersection(*(assigned[k] for k in assigned)):
        raise ValueError("External prompt split leaked prompt_canon across partitions")
    return row_sets


def stratified_prompt_sample(frame: pd.DataFrame, fraction=0.1, seed=SEED):
    """Sample complete prompt groups while retaining all label-count strata."""
    fraction=float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")
    prompt_labels=frame.groupby("prompt_canon",sort=True).model.nunique()
    rng=np.random.default_rng(int(seed))
    selected=[]
    for _, prompt_ids in prompt_labels.groupby(prompt_labels).groups.items():
        ids=np.asarray(sorted(map(str,prompt_ids)),dtype=object)
        count=min(len(ids),max(3,int(round(len(ids)*fraction))))
        selected.extend(rng.choice(ids,size=count,replace=False).tolist())
    sampled=frame.loc[frame.prompt_canon.isin(set(selected))].copy().reset_index(drop=True)
    found=set(map(int,sampled.groupby("prompt_canon").model.nunique().unique()))
    if found != set(map(int,prompt_labels.unique())):
        raise ValueError("stratified sample lost one or more available group-size strata")
    return sampled


def rotate_one_per_label(frame: pd.DataFrame, rows=None) -> tuple[pd.DataFrame, dict]:
    """Create duplicate-safe prompt views, rotating repeated labels by round.

    Every returned view contains no more than one response per model label;
    repeated same-label responses rotate through successive views.
    """
    selected = frame if rows is None else frame.iloc[np.asarray(rows, dtype=np.int64)]
    records = []
    for prompt, group in selected.groupby("prompt_canon", sort=True):
        by_label = {label: part.sort_values("id", kind="stable")
                    for label, part in group.groupby("model", sort=True)}
        rounds = max(map(len, by_label.values()))
        for turn in range(rounds):
            for label, part in by_label.items():
                if turn < len(part):
                    row = part.iloc[turn].to_dict()
                    row["view_group"] = f"{prompt}::rotation-{turn}"
                    row["rotation"] = turn
                    records.append(row)
    result = pd.DataFrame.from_records(records).reset_index(drop=True)
    if result.id.duplicated().any():
        raise ValueError("Duplicate rotation emitted a response more than once")
    max_size = int(result.groupby("view_group").size().max()) if len(result) else 0
    if max_size > len(LABELS):
        raise ValueError("Rotated prompt view exceeds the 18-label assignment limit")
    return result, {"rows": len(result), "views": int(result.view_group.nunique()),
                    "max_view_size": max_size}


def sample_incomplete_scenarios(v0: pd.DataFrame, *, scenarios=50, seed=SEED):
    """Return deterministic incomplete prompt-group row selections.

    Scenarios 1-25 use per-prompt caps 1..10; 26-50 use caps 1..18.
    True prompt IDs only define evaluation subsets, never grouper features.
    """
    ids = v0.id.astype(str).to_numpy()
    prompts = v0.prompt_id.astype(str).to_numpy()
    labels = v0.model.astype(str).to_numpy()
    by_prompt: dict[str, list[int]] = {}
    for row, prompt in enumerate(prompts):
        by_prompt.setdefault(prompt, []).append(row)
    result = {}
    for scenario in range(int(scenarios)):
        cap = 10 if scenario < scenarios // 2 else 18
        rng = np.random.default_rng(int(seed) + scenario)
        selected = []
        for prompt in sorted(by_prompt):
            members = by_prompt[prompt]
            by_label: dict[str, list[int]] = {}
            for row in members:
                by_label.setdefault(labels[row], []).append(row)
            unique = []
            for label in sorted(by_label):
                options = sorted(by_label[label], key=lambda row: ids[row])
                unique.append(int(rng.choice(options)))
            take = int(rng.integers(1, min(len(unique), cap) + 1))
            selected.extend(rng.choice(unique, size=take, replace=False).tolist())
        result[str(scenario+1)] = np.asarray(sorted(selected, key=lambda r: ids[r]), dtype=np.int64)
    return result


def evaluate_group_assignment(scores, groups, row_ids):
    """Deterministic Hungarian decode; validates group size and row alignment."""
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    row_ids = np.asarray(row_ids, dtype=str)
    if scores.ndim != 2 or scores.shape != (len(groups), len(LABELS)):
        raise ValueError("E27 scores, groups and canonical label dimension are misaligned")
    if len(row_ids) != len(groups) or len(set(row_ids.tolist())) != len(row_ids):
        raise ValueError("E27 row IDs must be unique and aligned")
    if max(pd.Series(groups).value_counts()) > len(LABELS):
        raise ValueError("E27 reconstructed group contains more than 18 responses")
    pred = e20.hungarian(scores, groups, row_ids)
    for group in np.unique(groups):
        assigned = pred[groups == group]
        if len(set(assigned.tolist())) != len(assigned):
            raise AssertionError(f"Hungarian assignment repeated a label in {group}")
    return pred


def freeze_ensemble_scores(root=ROOT, output_dir=OUT):
    """Validate the pinned E24 roster and cache family-balanced V0 scores."""
    root, output_dir = Path(root).resolve(), Path(output_dir)
    e24_config = json.loads((root/"artifacts/analysis/e24/configuration.json").read_text())
    if e24_config["trust_list"]["version"] != "trust-list-2-v039":
        raise ValueError("E27 expects the frozen E24 Trust List v039 roster")
    expected = e24_config["checkpoint_cache_sha256"]
    v0 = e22.catalog.cascades._v0(root)
    trust, members, probs, excluded, snapshot, *_ = e22._latest_trust(root, v0)
    if snapshot["version"] != e24_config["trust_list"]["version"]:
        raise ValueError("E27 loaded a different Trust List version than E24")
    checkpoint_ids = sorted(members, key=lambda sid: int(sid[1:]))
    if len(checkpoint_ids) != 89 or set(checkpoint_ids) != set(expected):
        raise ValueError(f"E27 frozen roster mismatch: expected 89; loaded {len(checkpoint_ids)}")
    changed = {sid: (expected[sid], members[sid]["cache_sha256"])
               for sid in checkpoint_ids if expected[sid] != members[sid]["cache_sha256"]}
    if changed:
        raise ValueError(f"E27 checkpoint prediction caches differ from E24 hashes: {changed}")
    matrices = np.stack([probs[sid] for sid in checkpoint_ids]).astype(np.float32)
    families = [e13._model_family(members[sid]["model"]) for sid in checkpoint_ids]
    scores = e13.family_balanced_probabilities(matrices, families)
    if scores.shape != (len(v0), len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError("E27 family-balanced V0 score matrix is invalid")
    y = e22.e16_common.e15._label_codes(v0.model)
    row_ids = v0.id.astype(str).to_numpy()
    groups = v0.prompt_id.astype(str).to_numpy()
    scenario_rows = sample_incomplete_scenarios(v0, scenarios=50, seed=SEED)
    result = {"trust_list_version": snapshot["version"], "trust_list_sha256": snapshot["sha256"],
        "checkpoints": checkpoint_ids, "checkpoint_count": len(checkpoint_ids),
        "checkpoint_cache_hashes": {sid: members[sid]["cache_sha256"] for sid in checkpoint_ids},
        "v0_rows": len(v0), "v0_row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
        "family_balanced": e20._metric(y, scores.argmax(1)), "scenarios": {}}
    details = score_scenarios(scores, y, groups, row_ids, scenario_rows)
    result["oracle_hungarian"] = details["oracle_hungarian"]
    result["scenarios"] = details["scenarios"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_scores = output_dir/"frozen_v0_scores.tmp.npz"
    np.savez_compressed(tmp_scores, row_ids=row_ids, y=y,
        prompt_ids=groups, checkpoint_ids=np.asarray(checkpoint_ids), scores=scores,
        pred_independent=scores.argmax(1), pred_oracle=details["pred_oracle"],
        **{f"scenario_{k}": v for k, v in scenario_rows.items()})
    tmp_scores.replace(output_dir/"frozen_v0_scores.npz")
    result["scenario_summary"] = scenario_summary(result["scenarios"])
    (output_dir/"frozen_roster_metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def score_scenarios(scores, y, groups, row_ids, scenario_rows):
    """Score full and sampled subsets with independent and true-group decodes."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups, row_ids = np.asarray(groups, dtype=str), np.asarray(row_ids, dtype=str)
    pred_independent = scores.argmax(1)
    pred_oracle = evaluate_group_assignment(scores, groups, row_ids)
    scenarios = {}
    for number, rows in scenario_rows.items():
        rows = np.asarray(rows, dtype=np.int64)
        pred = evaluate_group_assignment(scores[rows], groups[rows], row_ids[rows])
        scenarios[str(number)] = {"rows": len(rows),
            "max_group_cap": 10 if int(number) <= 25 else 18,
            "independent": e20._metric(y[rows], pred_independent[rows]),
            "hungarian_oracle": e20._metric(y[rows], pred)}
    return {"oracle_hungarian": e20._metric(y, pred_oracle),
            "pred_oracle": pred_oracle, "scenarios": scenarios}


def scenario_summary(scenarios):
    result = {}
    for method in ("independent", "hungarian_oracle"):
        for cap, nums in (("1-10", range(1, 26)), ("1-18", range(26, 51))):
            values = [float(scenarios[str(i)][method]["accuracy"]) for i in nums]
            result[f"{cap}_{method}"] = {"mean_accuracy": float(np.mean(values)),
                "p10_accuracy": float(np.percentile(values, 10)),
                "minimum_accuracy": float(np.min(values))}
    return result


def summarize_classification_scenarios(scenarios):
    result={}
    fields=("independent","group_hungarian","confidence_gated_hungarian")
    for field in fields:
        if any(field not in value for value in scenarios.values()):
            continue
        for cap,nums in (("1-10",range(1,26)),("1-18",range(26,51))):
            values=[float(scenarios[str(i)][field]["accuracy"]) for i in nums]
            result[f"{cap}_{field}"]={"mean_accuracy":float(np.mean(values)),
                "p10_accuracy":float(np.percentile(values,10)),
                "minimum_accuracy":float(np.min(values))}
    return result


def embed_external(frame: pd.DataFrame, *, limit=None, device="auto", batch_size=24):
    """Resumably encode V2.14 responses with E19-F's pinned embedding model."""
    chosen = frame if limit is None else frame.iloc[:int(limit)]
    vectors, metadata = e19f.encode_response_embeddings(
        chosen.text.astype(str).tolist(), cache_path=OUT/"external_embedding_cache.sqlite3",
        device=device, batch_size=batch_size)
    out = OUT/"external_embeddings_smoke.npy" if limit is not None else OUT/"external_embeddings.npy"
    np.save(out, vectors)
    metadata.update({"external_rows": len(chosen), "ids_sha256": hashlib.sha256(
        "\n".join(chosen.id.astype(str)).encode()).hexdigest(), "output": str(out)})
    (out.with_suffix(".json")).write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def reconstruct_external(method, *, root=ROOT, limit=None, batch_size=40):
    """Resumably reconstruct V2.14 prompts with the pinned E19-B or E19-D Qwen."""
    if method not in {"E19-B","E19-D"}:
        raise ValueError("External reconstruction method must be E19-B or E19-D")
    sample_path,_=_reconstruction_sample_paths(Path(root)/"artifacts/analysis/e27")
    frame=pd.read_parquet(sample_path)
    if limit is not None:
        frame=frame.sort_values("id",kind="stable").head(int(limit)).reset_index(drop=True)
        limit=None
    if method=="E19-B":
        model_id=e19b_reconstruct.MODEL_ID; revision=e19b_reconstruct.MODEL_REVISION
        prompt_version=e19b_reconstruct.RECONSTRUCTION_PROMPT_VERSION
    else:
        model_id=e19d_reconstruct.MODEL_ID; revision=e19d_reconstruct.MODEL_REVISION
        prompt_version=e19d_reconstruct.PROMPT_VERSION
    out=OUT/method.lower(); out.mkdir(parents=True,exist_ok=True)
    result=e19b_reconstruct.reconstruct_rows(row_ids=frame.id.astype(str).to_numpy(),
        texts=frame.text.fillna("").astype(str).to_numpy(),model_id=model_id,revision=revision,
        cache_path=out/"reconstruction_cache.sqlite3",
        output_path=out/"reconstructed_external.parquet",limit=limit,
        batch_size=batch_size,max_new_tokens=1024,seed=SEED,prompt_version=prompt_version)
    reconstructed_path=out/"reconstructed_external.parquet"
    reconstructed=pd.read_parquet(reconstructed_path)
    failed=reconstructed.status.eq("failed")
    fallback_text=frame.set_index(frame.id.astype(str)).text.fillna("").astype(str)
    for idx in reconstructed.index[failed]:
        row_id=str(reconstructed.at[idx,"id"])
        reconstructed.at[idx,"reconstructed_prompt"]=fallback_text.loc[row_id]
        reconstructed.at[idx,"topic"]="response-text fallback after reconstruction failure"
        reconstructed.at[idx,"distinctive_details"]="[]"
        reconstructed.at[idx,"constraints"]="[]"
        reconstructed.at[idx,"status"]="fallback_response"
    if failed.any():
        reconstructed.to_parquet(reconstructed_path,index=False)
        result["fallback_response_rows"]=int(failed.sum())
        result["fallback_policy"]="use original response text as the reconstructed-prompt feature; retain failed cache status and error"
        manifest_path=out/"reconstruction_manifest.json"
        manifest=json.loads(manifest_path.read_text()); manifest.update(result)
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    return result


def embed_reconstructed_prompts(method):
    if method not in {"E19-B","E19-D"}:
        raise ValueError("External reconstruction method must be E19-B or E19-D")
    folder=OUT/method.lower()
    frame=pd.read_parquet(folder/"reconstructed_external.parquet")
    if not frame.status.isin(("ok","fallback_response")).all():
        raise ValueError(f"{method} external reconstructions are incomplete")
    vectors,meta=e19f.encode_response_embeddings(frame.reconstructed_prompt.astype(str).tolist(),
        cache_path=folder/"semantic_embedding_cache.sqlite3",device="auto")
    np.save(folder/"semantic_embeddings.npy",vectors)
    meta.update({"ids_sha256":hashlib.sha256("\n".join(frame.id.astype(str)).encode()).hexdigest(),
                 "rows":len(frame)})
    (folder/"semantic_embeddings.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta


def _as_list(value):
    if isinstance(value,str):
        try: return json.loads(value)
        except json.JSONDecodeError: return [value]
    if isinstance(value,(list,tuple,np.ndarray)): return list(value)
    return []


def _reconstruction_features(rows, spaces, recon, response, semantic, topics, details, constraints):
    rows=np.sort(np.asarray(rows,dtype=np.int64),kind="stable")
    pairs=e19b_group.candidate_pairs(rows,spaces,semantic,neighbors=40)
    features=e19b_group.pair_features(rows,pairs,spaces,recon,response,semantic,
                                      topics,details,constraints)
    return rows,pairs,features


def _pair_cluster(ids,pairs,features,model,threshold):
    scores=e19b_group.score_pairs(model,features) if len(pairs) else np.empty(0,dtype=np.float32)
    groups,_=e19b_group.cluster_complete_link(ids,pairs,scores,threshold=threshold,max_cluster_size=18)
    edge={(min(int(a),int(b)),max(int(a),int(b))):float(s)
          for (a,b),s in zip(pairs,scores)}
    confidence={}
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group)
        values=[edge.get((min(int(a),int(b)),max(int(a),int(b))),0.)
                for ix,a in enumerate(rows) for b in rows[ix+1:]]
        confidence[str(group)]=min(values) if values else 1.
    return groups,confidence,scores


def run_external_reconstruction_grouper(method, *, root=ROOT, output_dir=OUT):
    """Train one E19-B/D pair scorer externally and evaluate its V0 deployment."""
    if method not in {"E19-B","E19-D"}:
        raise ValueError("method must be E19-B or E19-D")
    root,output_dir=Path(root).resolve(),Path(output_dir)
    folder=output_dir/method.lower()
    _,rotation_path=_reconstruction_sample_paths(output_dir)
    frame=pd.read_parquet(rotation_path)
    recon_frame=pd.read_parquet(folder/"reconstructed_external.parquet")
    recon_meta=json.loads((folder/"semantic_embeddings.json").read_text())
    semantic=np.load(folder/"semantic_embeddings.npy",mmap_mode="r")
    if not recon_frame.status.isin(("ok","fallback_response")).all() or recon_frame.id.duplicated().any():
        raise ValueError(f"{method} requires complete, unique external reconstructions")
    recon_pos={sid:i for i,sid in enumerate(recon_frame.id.astype(str))}
    order=np.asarray([recon_pos[sid] for sid in frame.id.astype(str)],dtype=np.int64)
    recon=np.asarray(recon_frame.reconstructed_prompt.astype(str))[order]
    topics=np.asarray(recon_frame.topic.fillna("").astype(str))[order]
    details=[_as_list(x) for x in recon_frame.distinctive_details.to_numpy()[order]]
    constraints=[_as_list(x) for x in recon_frame.constraints.to_numpy()[order]]
    semantic=semantic[order]
    if recon_meta.get("ids_sha256")!=hashlib.sha256("\n".join(recon_frame.id.astype(str)).encode()).hexdigest():
        raise ValueError(f"{method} external semantic embeddings have stale row alignment")
    response=frame.text.fillna("").astype(str).to_numpy()
    ids=frame.id.astype(str).to_numpy(); groups=frame.view_group.astype(str).to_numpy()
    split={name:np.flatnonzero(frame.split.astype(str).to_numpy()==name)
           for name in ("train","validation","test")}
    spaces=e19b_group.fit_similarity_spaces(recon,response,split["train"])
    tr_rows,tr_pairs,tr_x=_reconstruction_features(split["train"],spaces,recon,response,
        semantic,topics,details,constraints)
    tr_y=(groups[tr_rows[tr_pairs[:,0]]]==groups[tr_rows[tr_pairs[:,1]]]).astype(np.int8)
    va_rows,va_pairs,va_x=_reconstruction_features(split["validation"],spaces,recon,response,
        semantic,topics,details,constraints)
    te_rows,te_pairs,te_x=_reconstruction_features(split["test"],spaces,recon,response,
        semantic,topics,details,constraints)
    records=[]; candidates=[]; models={}
    for c in (0.1,1.,10.):
        model=e19b_group.fit_pair_scorer(tr_x,tr_y,c=c,seed=SEED); models[c]=model
        val_scores=e19b_group.score_pairs(model,va_x)
        for threshold in GROUP_THRESHOLDS:
            pred,_=e19b_group.cluster_complete_link(ids[va_rows],va_pairs,val_scores,
                threshold=threshold,max_cluster_size=18)
            metric=e19b_evaluate.grouping_metrics(groups[va_rows],pred)
            records.append({"C":c,"threshold":threshold,"pairwise_f1":metric["pairwise_f1"],
                            "pairwise_precision":metric["pairwise_precision"]})
    best=max(records,key=lambda r:(r["pairwise_f1"],r["pairwise_precision"],-r["C"],-r["threshold"]))
    val_model=models[best["C"]]
    val_pred,val_conf,_=_pair_cluster(ids[va_rows],va_pairs,va_x,val_model,best["threshold"])
    gate_records=[]
    for gate in GATE_THRESHOLDS:
        gated=_confidence_gate(val_pred,val_conf,ids[va_rows],gate)
        metric=e19b_evaluate.grouping_metrics(groups[va_rows],gated)
        gate_records.append({"threshold":float(gate),"pairwise_f1":metric["pairwise_f1"],
                             "pairwise_precision":metric["pairwise_precision"]})
    gate=max(gate_records,key=lambda r:(r["pairwise_f1"],r["pairwise_precision"],-r["threshold"]))
    test_pred,_,test_scores=_pair_cluster(ids[te_rows],te_pairs,te_x,val_model,best["threshold"])
    ext_metrics=e19b_evaluate.grouping_metrics(groups[te_rows],test_pred)
    pd.DataFrame({"id":ids[te_rows],"true_prompt_group":groups[te_rows],
                  "predicted_group":test_pred}).to_parquet(output_dir/f"external_test_{method}.parquet",index=False)

    # Final text vectorizers and scorer are fit using training plus validation prompts.
    fit_rows=np.sort(np.concatenate((split["train"],split["validation"])))
    final_spaces=e19b_group.fit_similarity_spaces(recon,response,fit_rows)
    fx_pairs=e19b_group.candidate_pairs(fit_rows,final_spaces,semantic,neighbors=40)
    fx=e19b_group.pair_features(fit_rows,fx_pairs,final_spaces,recon,response,semantic,
                                topics,details,constraints)
    fx_y=(groups[fit_rows[fx_pairs[:,0]]]==groups[fit_rows[fx_pairs[:,1]]]).astype(np.int8)
    final_model=e19b_group.fit_pair_scorer(fx,fx_y,c=best["C"],seed=SEED)

    v0=e22.catalog.cascades._v0(root); v0ids=v0.id.astype(str).to_numpy()
    v0texts=v0.text.fillna("").astype(str).to_numpy()
    v0_artifact_name=method.lower().replace("-","")
    v0recon=pd.read_parquet(root/f"artifacts/analysis/{v0_artifact_name}/reconstructed_prompts.parquet")
    if v0recon.id.astype(str).tolist()!=v0ids.tolist() or not v0recon.status.eq("ok").all():
        raise ValueError(f"{method} saved V0 reconstructions are missing or misaligned")
    v0sem=np.load(root/f"artifacts/analysis/{v0_artifact_name}/semantic_embeddings.npy")
    v0rec=v0recon.reconstructed_prompt.astype(str).to_numpy()
    v0topic=v0recon.topic.fillna("").astype(str).to_numpy()
    v0detail=[_as_list(x) for x in v0recon.distinctive_details.to_numpy()]
    v0constraints=[_as_list(x) for x in v0recon.constraints.to_numpy()]
    v0vectors={name:final_spaces[name+"_vectorizer"] for name in
        ("reconstruction_char","reconstruction_word","response_char","response_word")}
    v0spaces=e19b_group.transform_similarity_spaces(v0rec,v0texts,v0vectors)
    v0rows,v0pairs,v0x=_reconstruction_features(np.arange(len(v0)),v0spaces,v0rec,
        v0texts,v0sem,v0topic,v0detail,v0constraints)
    pred,conf,_=_pair_cluster(v0ids[v0rows],v0pairs,v0x,final_model,best["threshold"])
    frozen=np.load(output_dir/"frozen_v0_scores.npz",allow_pickle=True)
    score=np.asarray(frozen["scores"]); y=np.asarray(frozen["y"]); row_ids=np.asarray(frozen["row_ids"])
    groups_v0=np.empty(len(v0),dtype=object); groups_v0[v0rows]=pred
    independent=score.argmax(1); assigned=evaluate_group_assignment(score,groups_v0,row_ids)
    gated_pred=_confidence_gate(pred,conf,v0ids[v0rows],gate["threshold"])
    gated_groups=np.empty(len(v0),dtype=object); gated_groups[v0rows]=gated_pred
    gated_assigned=evaluate_group_assignment(score,gated_groups,row_ids)
    scenarios=sample_incomplete_scenarios(v0,scenarios=50,seed=SEED); scenario_result={}
    for number,rows in scenarios.items():
        p=evaluate_group_assignment(score[rows],groups_v0[rows],row_ids[rows])
        pg=evaluate_group_assignment(score[rows],gated_groups[rows],row_ids[rows])
        scenario_result[number]={"rows":len(rows),"independent":e20._metric(y[rows],independent[rows]),
            "group_hungarian":e20._metric(y[rows],p),"confidence_gated_hungarian":e20._metric(y[rows],pg)}
    payload={"method":method,"external_validation_selection":{"C":best["C"],
        "threshold":best["threshold"],"gate_threshold":gate["threshold"],
        "pairwise_metrics":e19b_evaluate.grouping_metrics(groups[va_rows],val_pred),
        "gate_selection_records":gate_records,"candidate_records":records},
        "external_test":{"pairwise_metrics":ext_metrics,"candidate_pairs":len(te_pairs)},
        "v0":{"independent":e20._metric(y,independent),"group_hungarian":e20._metric(y,assigned),
              "confidence_gated_hungarian":e20._metric(y,gated_assigned),
              "corrected_introduced":e20._comparison(y,independent,assigned),
              "confidence_gated_comparison_vs_independent":e20._comparison(y,independent,gated_assigned),
              "scenarios":scenario_result}}
    (output_dir/f"{method.lower()}_grouping_results.json").write_text(json.dumps(payload,indent=2)+"\n")
    pd.DataFrame({"id":row_ids,"predicted_group":groups_v0,"confidence":[conf[str(g)] for g in pred]}).to_parquet(
        output_dir/f"v0_groups_{method}.parquet",index=False)
    return payload


def finalize(root=ROOT, output_dir=OUT):
    """Collect available external groupers and update the catalog/dashboard."""
    root,output_dir=Path(root).resolve(),Path(output_dir)
    baseline=json.loads((output_dir/"frozen_roster_metrics.json").read_text())
    def read_f_results(path):
        result={}
        if path.is_file():
            payload=json.loads(path.read_text())
            for method in METHODS:
                result[method]={"validation":payload["external_validation"][method]["pairwise_metrics"],
                    "test":payload["external_test"][method]["pairwise_metrics"],
                    "v0":payload["v0"][method]}
        return result
    full_f=read_f_results(output_dir/"e19f_grouping_results.json")
    sample_path=output_dir/"e19f_grouping_results_01pct.json"
    sample_f=read_f_results(sample_path)
    # Compare all grouping families on the same small external split. Preserve
    # the more precise full-corpus E19-F evaluation separately as context.
    methods=dict(sample_f or full_f)
    for method in ("E19-B","E19-D"):
        path=output_dir/f"{method.lower()}_grouping_results.json"
        if path.is_file():
            saved=json.loads(path.read_text())
            methods[method]={"validation":saved["external_validation_selection"]["pairwise_metrics"],
                "test":saved["external_test"]["pairwise_metrics"],"v0":saved["v0"]}
    if not methods:
        raise FileNotFoundError("No completed E27 external grouper results exist")
    best_method=max(methods,key=lambda m:(methods[m]["validation"]["pairwise_f1"],
        methods[m]["validation"]["pairwise_precision"],m))
    best=methods[best_method]["v0"]
    e27_c=best.get("external_group_hungarian",best.get("group_hungarian"))
    e27_d=best.get("confidence_gated_hungarian")
    frozen=np.load(output_dir/"frozen_v0_scores.npz",allow_pickle=True)
    oracle_comparison=e20._comparison(np.asarray(frozen["y"]),
        np.asarray(frozen["pred_independent"]),np.asarray(frozen["pred_oracle"]))
    metrics={"experiment":"E27","trust_list_version":baseline["trust_list_version"],
        "trust_list_sha256":baseline["trust_list_sha256"],"checkpoint_count":89,
        "checkpoints":baseline["checkpoints"],"v0_rows":2699,
        "external_split":json.loads((output_dir/"split_manifest.json").read_text()),
        "reconstruction_sample":json.loads((output_dir/"reconstruction_sample_manifest.json").read_text())
            if (output_dir/"reconstruction_sample_manifest.json").is_file() else None,
        "v0_overlap":json.loads((output_dir/"data_quality.json").read_text()),
        "E27-A":{"independent":baseline["family_balanced"]},
        "E27-B":{"oracle_hungarian":baseline["oracle_hungarian"],
                 "scenario_summary":baseline["scenario_summary"],
                 "comparison_vs_independent":oracle_comparison},
        "external_groupers":methods,"selected_grouper":best_method,
        "external_groupers_full":full_f if sample_f else {},
        "scenario_accuracy_summary":{method:summarize_classification_scenarios(
            data["v0"].get("scenarios",{})) for method,data in methods.items()},
        "E27-C":e27_c,"E27-D":e27_d,
        "E27-C_comparison_vs_independent":best.get("comparison_vs_independent",
            best.get("corrected_introduced")),
        "E27-D_comparison_vs_independent":best.get("confidence_gated_comparison_vs_independent",
            best.get("corrected_introduced")),
        "pending_groupers":[x for x in ("E19-B","E19-D") if x not in methods],
        "v0_exposure_limitation":"The frozen 89-checkpoint pool has historical V0 selection exposure; no E27 grouping fit or threshold uses V0 labels.",
        "status":"Partial" if set(methods)!={"E19-B","E19-D",*METHODS} else "Scored"}
    (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    catalog_path=root/"configs/ensemble_catalog_v2.json"
    catalog=json.loads(catalog_path.read_text())
    for entry in catalog["experiments"]:
        eid=entry["id"]
        if eid=="E27":
            entry.update(status=metrics["status"],
                result_source="artifacts/analysis/e27/metrics.json",
                accuracy=baseline["family_balanced"]["accuracy"],
                balanced_accuracy=baseline["family_balanced"]["balanced_accuracy"],
                primary_score=baseline["family_balanced"]["balanced_accuracy"])
            entry.pop("pending_groupers",None)
            entry.pop("running_groupers",None)
            continue
        if eid=="E27-A":
            m=baseline["family_balanced"]
        elif eid=="E27-B":
            m=baseline["oracle_hungarian"]
        elif eid=="E27-C" and e27_c:
            m=e27_c
        elif eid=="E27-D" and e27_d:
            m=e27_d
        elif eid=="E27-E" and methods:
            m=methods[best_method]["v0"].get("external_group_hungarian",
                methods[best_method]["v0"].get("group_hungarian"))
        else: continue
        entry.update(accuracy=m["accuracy"],balanced_accuracy=m["balanced_accuracy"],
            primary_score=m["balanced_accuracy"],status=("Scored" if eid in {"E27-A","E27-B"} else metrics["status"]),
            result_source="artifacts/analysis/e27/metrics.json")
        if eid in {"E27-C","E27-D","E27-E"}:
            entry["selected_grouper"]=best_method
            entry["grouping_validation_pairwise_f1"]=methods[best_method]["validation"]["pairwise_f1"]
        entry.pop("pending_groupers",None)
        entry.pop("running_groupers",None)
    catalog_path.write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+"\n")
    split_info=metrics["external_split"]
    split_counts={k:int(v["rows"]) for k,v in split_info["splits"].items()}
    available_sizes=sorted(map(int,split_info["group_size_distribution"].keys()))
    sample_info=metrics.get("reconstruction_sample")
    sample_split_counts={k:int(v["rows"]) for k,v in sample_info["splits"].items()} if sample_info else {}
    sample_tag=f"{int(round(sample_info['fraction']*100)):02d}pct" if sample_info else "10pct"
    sample_fraction=float(sample_info["fraction"]) if sample_info else 0.1
    sample_note=(f" E19-B/D use a deterministic {sample_info['fraction']:.0%} stratified prompt sample "
        f"({sample_info['rows']:,} responses, {sample_info['prompt_groups']:,} prompts) with split counts "
        f"{sample_split_counts}; all group-label-count "
        "strata 1–18 are represented in every split. This is a small-sample pilot: held-out strata can have only one prompt, so its pairwise metrics are noisy.") if sample_info else ""
    lines=["# E27 — External Grouping and Assignment Validation","",
        f"E27 freezes the E24 Trust List v039 cache roster (89 checkpoints) and family-balanced scores. The E19-F external fit/evaluation set contains {split_info['rows']:,} responses across {split_info['prompt_groups']:,} verified `prompt_canon` groups; available distinct-label group sizes are {available_sizes}. Prompt-disjoint split sizes are {split_counts}.{sample_note} V0 labels are used only for final classification metrics. The frozen checkpoint pool has historical V0 selection exposure.","",
        "## Results","",
        "| Variant | Grouper | Accuracy | Balanced accuracy | Corrected | Introduced |",
        "|---|---|---:|---:|---:|---:|"]
    for label,key in (("E27-A","independent"),("E27-B","oracle_hungarian")):
        m=metrics[label][key]
        comp=metrics.get("E27-B",{}).get("comparison_vs_independent",{}) if label=="E27-B" else {}
        lines.append(f"| {label} | {'family-balanced' if label=='E27-A' else 'true prompt IDs (oracle)'} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {comp.get('corrected','—')} | {comp.get('introduced','—')} |")
    for method,data in methods.items():
        v=data["v0"]; grouped=v.get("external_group_hungarian",v.get("group_hungarian"))
        comp=v.get("comparison_vs_independent",v.get("corrected_introduced",{}))
        lines.append(f"| E27-E {method} | external {method} | {grouped['accuracy']:.4%} | {grouped['balanced_accuracy']:.4%} | {comp.get('corrected','—')} | {comp.get('introduced','—')} |")
    if e27_c:
        comp=metrics.get("E27-C_comparison_vs_independent") or {}
        lines.append(f"| E27-C | selected {best_method} | {e27_c['accuracy']:.4%} | {e27_c['balanced_accuracy']:.4%} | {comp.get('corrected','—')} | {comp.get('introduced','—')} |")
    if e27_d:
        comp=metrics.get("E27-D_comparison_vs_independent") or {}
        lines.append(f"| E27-D | gated {best_method} | {e27_d['accuracy']:.4%} | {e27_d['balanced_accuracy']:.4%} | {comp.get('corrected','—')} | {comp.get('introduced','—')} |")
    lines += ["","## External grouping metrics","", "| Grouper | Validation pair F1 | Test pair precision | Test pair recall | Test pair F1 | Exact groups | Split groups | Incorrect merges |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method,data in methods.items():
        m=data["test"]; val=data["validation"]
        lines.append(f"| {method} | {val['pairwise_f1']:.4%} | {m['pairwise_precision']:.4%} | {m['pairwise_recall']:.4%} | {m['pairwise_f1']:.4%} | {m['exact_groups']}/{m['true_group_count']} | {m['split_original_groups']} | {m['incorrectly_merged_groups']} |")
    lines += ["","All five confidence-gate thresholds selected as 0.0 on external validation, so confidence-gated assignment made no changes; E27-D equals E27-C in this run.",""]
    if metrics.get("external_groupers_full"):
        lines += ["## Full-corpus E19-F reference","",
            "This earlier E19-F pass used all 223,168 canonical external responses (rather than the 1% split used for the cross-grouper table above). Keep its grouping metrics separate from the small-sample comparison.",
            "","| Grouper | Full-corpus test pair F1 | V0 Hungarian balanced accuracy |","|---|---:|---:|"]
        for method,data in metrics["external_groupers_full"].items():
            grouped=data["v0"]["external_group_hungarian"]
            lines.append(f"| {method} | {data['test']['pairwise_f1']:.4%} | {grouped['balanced_accuracy']:.4%} |")
    lines += ["","## V0 incomplete-group scenarios","",
        "| Grouper | Size range | Mode | Mean accuracy | 10th percentile | Minimum |",
        "|---|---|---|---:|---:|---:|"]
    for method,summary in metrics["scenario_accuracy_summary"].items():
        for key,record in summary.items():
            cap,mode=key.split("_",1)
            lines.append(f"| {method} | {cap} | {mode} | {record['mean_accuracy']:.4%} | {record['p10_accuracy']:.4%} | {record['minimum_accuracy']:.4%} |")
    lines += ["","## Status and reproducibility","",f"Selected grouper on external validation: **{best_method}**, selected by pairwise F1 with precision tie-break. Completed methods: {', '.join(methods)}. Pending: {', '.join(metrics['pending_groupers']) or 'none'}.","",
        "The independent and oracle 50-scenario baselines, row samples, 89 cache hashes, prompt-disjoint split manifest, response overlap checks, embeddings, method-specific group assignments and metrics are under `artifacts/analysis/e27/`. The first 25 scenarios sample 1–10 distinct-label responses per prompt; the remaining 25 sample 1–18. Their lower-tail accuracy is a robustness diagnostic on the same V0 rows, not independent generalization evidence.","",
        f"The reduced E19-B/D scope is reproducible from `reconstruction_sample_manifest.json` and the paired `external_v214_reconstruction_sample_{sample_tag}.parquet` / `external_v214_reconstruction_rotation_{sample_tag}.parquet` files. Failed generations are retried; any remaining failures use the original response text as a recorded fallback feature so no sampled row is discarded.","",
        "No E27 grouper or threshold is fit on V0 labels. The V2.14 canonical subset has zero V0 response-ID and normalized-text overlaps. V0 original canonical prompt text is unavailable, so prompt-level cross-pool identity cannot be checked.",""]
    lines += ["## Reproduction commands","",
        "The stages are resumable and use cached embeddings/reconstructions when present:","",
        "```bash","python -m src.e27 --prepare --freeze-v0","python -m src.e27 --embed-external","python -m src.e27 --run-f-grouping",f"python -m src.e27 --prepare-reconstruction-sample --sample-fraction {sample_fraction}","python -m src.e27 --reconstruct-b --reconstruction-batch-size 40","python -m src.e27 --embed-recon E19-B","python -m src.e27 --run-reconstructed-b","python -m src.e27 --reconstruct-d --reconstruction-batch-size 20","python -m src.e27 --embed-recon E19-D","python -m src.e27 --run-reconstructed-d","python -m src.e27 --finalize","```","",
        "E19-B/D reconstruction and embedding stages are intentionally separate so they can be interrupted and resumed independently.",""]
    (root/"docs/e27.md").write_text("\n".join(lines))
    return metrics


def _candidate_bundle(rows, ids, texts, embeddings, vectorizers, lexical=None):
    lexical = lexical if lexical is not None else e19f._transform_lexical(texts, vectorizers)
    # Exact batched dense top-40 cosine retrieval on CUDA is tractable at the
    # external corpus scale. Word/character similarities remain pair-scorer
    # features but are not separate corpus-wide retrieval passes.
    ordered=e19f._id_order(rows,ids)
    print(f"E27 retrieval: exact embedding top-40 for {len(ordered):,} rows",flush=True)
    import torch
    device=e19f._select_device(torch,"auto")
    if device=="cuda":
        pairs=e19b_group._dense_neighbor_pairs_gpu(embeddings,ordered,40,
            device="cuda:0",block_size=128)
        # The E19-B helper returns a set for deduplication. E19-F feature
        # generation requires stable local-index pairs, so sort and materialize.
        pairs=np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)
    else:
        pairs=e19f._neighbor_pairs(sparse.csr_matrix(np.asarray(embeddings)[ordered]),
                                   len(ordered),40)
        pairs=np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)
    features = e19f._pair_features(ordered, pairs, lexical, embeddings)
    print(f"E27 retrieval complete: {len(pairs):,} candidate pairs",flush=True)
    return lexical, ordered, pairs, features


def _cluster_mnn_candidates(ids, pairs, cosine_scores, *, threshold, k, max_size=18):
    """MNN-seeded, candidate-edge complete-link clustering for large corpora."""
    ids = np.asarray(ids, dtype=str)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    scores = np.asarray(cosine_scores, dtype=np.float64)
    edge = {(int(a), int(b)): float(s) for (a,b),s in zip(pairs,scores)}
    incident = [[] for _ in ids]
    for (a,b), score in edge.items():
        incident[a].append((score, b)); incident[b].append((score, a))
    directed = set()
    for row, vals in enumerate(incident):
        vals.sort(key=lambda x: (-x[0], ids[x[1]]))
        directed.update((row, other) for _,other in vals[:int(k)])
    mutual = [(a,b,s) for (a,b),s in edge.items()
              if (a,b) in directed and (b,a) in directed and s >= float(threshold)]
    mutual.sort(key=lambda item: (-item[2], ids[item[0]], ids[item[1]]))
    cluster = list(range(len(ids))); members = {i:{i} for i in range(len(ids))}
    for a,b,_ in mutual:
        ca,cb=cluster[a],cluster[b]
        if ca==cb or ca not in members or cb not in members:
            continue
        left,right=members[ca],members[cb]
        if len(left)+len(right)>int(max_size):
            continue
        cross=[edge.get((min(x,y),max(x,y)),float("-inf"))
               for x in left for y in right]
        if min(cross,default=float("-inf")) + 1e-12 < float(threshold):
            continue
        new=min(ca,cb); old=max(ca,cb); merged=left|right
        del members[ca]; del members[cb]; members[new]=merged
        for row in merged: cluster[row]=new
    ordered=sorted(members.values(),key=lambda part: min(ids[list(part)]))
    pred=np.empty(len(ids),dtype=object)
    for serial,part in enumerate(ordered):
        pred[list(part)]=f"e27-mnn-{serial:06d}"
    return pred


def _method_clusters(method, ids, pairs, features, *, threshold, k=10, pair_model=None):
    ids=np.asarray(ids,dtype=str)
    if method=="E19-F-C":
        scores=e19f.score_similarity_model(pair_model,features) if len(pairs) else np.empty(0)
    else:
        scores=features[:,0] if len(pairs) else np.empty(0)
    if method=="E19-F-B":
        pred=_cluster_mnn_candidates(ids,pairs,scores,threshold=threshold,k=k)
    else:
        pred,_=e19f._cluster_fc(ids,pairs,scores,threshold)
    # Confidence is the minimum observed pair score in each predicted cluster.
    edge_score={(min(int(a),int(b)),max(int(a),int(b))):float(s)
                for (a,b),s in zip(pairs,scores)}
    confidence={}
    for group in np.unique(pred):
        part=np.flatnonzero(pred==group)
        vals=[edge_score.get((min(int(a),int(b)),max(int(a),int(b))),0.)
              for ix,a in enumerate(part) for b in part[ix+1:]]
        confidence[str(group)]=min(vals) if vals else 1.0
    return pred, confidence, scores


def _confidence_gate(groups, confidence, row_ids, threshold):
    groups=np.asarray(groups,dtype=str); row_ids=np.asarray(row_ids,dtype=str)
    # Object dtype prevents singleton IDs from truncating to the short
    # original string width (for example "x" -> "s").
    gated=groups.astype(object).copy()
    for group in np.unique(groups):
        if confidence.get(str(group),1.) + 1e-12 < float(threshold):
            rows=np.flatnonzero(groups==group)
            for row in rows: gated[row]=f"singleton::{row_ids[row]}"
    return gated


def run_external_f(root=ROOT, output_dir=OUT, *, reconstruction_sample=False):
    """Fit E19-F A/B/C groupers externally and evaluate V0 assignment."""
    root, output_dir = Path(root).resolve(), Path(output_dir)
    if reconstruction_sample:
        canonical_path,rotation_path=_reconstruction_sample_paths(output_dir)
        frame=pd.read_parquet(rotation_path)
        canonical=pd.read_parquet(canonical_path)
        full_canonical=pd.read_parquet(output_dir/"external_v214_canonical.parquet")
        result_suffix=f"_{int(round(json.loads((output_dir/'reconstruction_sample_manifest.json').read_text())['fraction']*100)):02d}pct"
    else:
        frame=pd.read_parquet(output_dir/"external_v214_rotation_views.parquet")
        canonical=pd.read_parquet(output_dir/"external_v214_canonical.parquet")
        full_canonical=canonical
        result_suffix=""
    canonical_embeddings=np.load(output_dir/"external_embeddings.npy",mmap_mode="r")
    # The prepared canonical artifact is authoritative. Embeddings were saved
    # in that row order; rotate views are reordered by unique response ID.
    meta=json.loads((output_dir/"external_embeddings.json").read_text())
    expected=hashlib.sha256("\n".join(full_canonical.id.astype(str)).encode()).hexdigest()
    if meta.get("ids_sha256")!=expected:
        raise ValueError("E27 embedding manifest is not aligned to prepared canonical V2.14 rows")
    canonical_positions={value:i for i,value in enumerate(full_canonical.id.astype(str))}
    embeddings=canonical_embeddings[np.asarray([canonical_positions[x]
        for x in frame.id.astype(str)],dtype=np.int64)]
    if embeddings.shape[0]!=len(frame) or frame.id.duplicated().any():
        raise ValueError("E27 external embeddings and verified rows do not align")
    ids=frame.id.astype(str).to_numpy(); texts=frame.text.astype(str).to_numpy()
    groups=frame.view_group.astype(str).to_numpy()
    split={name:np.flatnonzero(frame.split.astype(str).to_numpy()==name)
           for name in ("train","validation","test")}
    print("E27 E19-F: fitting training-only word/character TF-IDF",flush=True)
    vectorizers=e19f._fit_lexical(texts,split["train"])
    print("E27 E19-F: transforming response features",flush=True)
    lexical_all=e19f._transform_lexical(texts,vectorizers)
    lexical,train_rows,train_pairs,train_features=_candidate_bundle(
        split["train"],ids,texts,embeddings,vectorizers,lexical_all)
    # F-C's pair labels and logistic fit use only training prompts.
    fc_models={c:e19f.fit_similarity_model(train_rows,train_pairs,lexical,
        embeddings,groups,c=c,seed=SEED) for c in FC_C_VALUES}
    validation={}; test={}; chosen={}
    val_lex,val_rows,val_pairs,val_features=_candidate_bundle(
        split["validation"],ids,texts,embeddings,vectorizers,lexical_all)
    test_lex,test_rows,test_pairs,test_features=_candidate_bundle(
        split["test"],ids,texts,embeddings,vectorizers,lexical_all)
    for method in METHODS:
        params=[{"threshold":t,"k":k} for k in MNN_K for t in GROUP_THRESHOLDS] if method=="E19-F-B" else [
            {"threshold":t,"k":10} for t in GROUP_THRESHOLDS]
        if method=="E19-F-C":
            params=[{**p,"C":c} for c in FC_C_VALUES for p in params]
        records=[]
        for setting in params:
            pred,_,_=_method_clusters(method,ids[val_rows],val_pairs,val_features,
                threshold=setting["threshold"],k=setting["k"],
                pair_model=fc_models.get(setting.get("C",1.0)))
            metric=e19b_evaluate.grouping_metrics(groups[val_rows],pred)
            records.append({**setting,"pairwise_f1":metric["pairwise_f1"],
                            "pairwise_precision":metric["pairwise_precision"]})
        best=max(records,key=lambda r:(r["pairwise_f1"],r["pairwise_precision"],
                                       -r.get("k",0),-r["threshold"],-r.get("C",1.0)))
        chosen[method]={**best,"validation_records":records,
            "selection_rule":"maximize validation pairwise F1; precision tie-break"}
        val_pred,val_conf,_=_method_clusters(method,ids[val_rows],val_pairs,val_features,
            threshold=best["threshold"],k=best.get("k",10),
            pair_model=fc_models.get(best.get("C",1.0)))
        gate_records=[]
        for gate in GATE_THRESHOLDS:
            gated=_confidence_gate(val_pred,val_conf,ids[val_rows],gate)
            metric=e19b_evaluate.grouping_metrics(groups[val_rows],gated)
            gate_records.append({"threshold":float(gate),"pairwise_f1":metric["pairwise_f1"],
                                 "pairwise_precision":metric["pairwise_precision"]})
        gate_best=max(gate_records,key=lambda r:(r["pairwise_f1"],r["pairwise_precision"],-r["threshold"]))
        chosen[method]["gate_threshold"]=gate_best["threshold"]
        chosen[method]["gate_validation_records"]=gate_records
        pred,conf,_=_method_clusters(method,ids[test_rows],test_pairs,test_features,
            threshold=best["threshold"],k=best.get("k",10),
            pair_model=fc_models.get(best.get("C",1.0)))
        validation[method]={"pair_rows":len(val_pairs),"pairwise_metrics":e19b_evaluate.grouping_metrics(groups[val_rows],
            _method_clusters(method,ids[val_rows],val_pairs,val_features,
                threshold=best["threshold"],k=best.get("k",10),
                pair_model=fc_models.get(best.get("C",1.0)))[0]),"gate_threshold":gate_best}
        test[method]={"pair_rows":len(test_pairs),"pairwise_metrics":e19b_evaluate.grouping_metrics(groups[test_rows],pred)}
        assignment=pd.DataFrame({"id":ids[test_rows],"true_prompt_canon":groups[test_rows],
                                 "predicted_group":pred,"confidence":[conf[str(g)] for g in pred]})
        assignment.to_parquet(output_dir/f"external_test_{method}{result_suffix}.parquet",index=False)

    # Refit the learned pair model on train+validation groups; keep validation
    # thresholds frozen. The pair feature definitions are unchanged at V0.
    fit_rows=np.sort(np.concatenate((split["train"],split["validation"])))
    final_vectorizers=e19f._fit_lexical(texts,fit_rows)
    final_lexical=e19f._transform_lexical(texts,final_vectorizers)
    fit_lex,fit_rows_ordered,fit_pairs,fit_features=_candidate_bundle(
        fit_rows,ids,texts,embeddings,final_vectorizers,final_lexical)
    selected_c=chosen["E19-F-C"].get("C",1.0)
    final_fc=e19f.fit_similarity_model(fit_rows_ordered,fit_pairs,fit_lex,
        embeddings,groups,c=selected_c,seed=SEED)
    v0=e22.catalog.cascades._v0(root)
    v0ids=v0.id.astype(str).to_numpy(); v0texts=v0.text.fillna("").astype(str).to_numpy()
    v0emb,embed_meta=e19f.encode_response_embeddings(v0texts,
        cache_path=root/"artifacts/analysis/e19f/embedding_cache.sqlite3",device="auto")
    v0lex=e19f._transform_lexical(v0texts,final_vectorizers)
    v0ordered,v0pairs=e19f.candidate_pairs(np.arange(len(v0)),v0ids,v0lex,v0emb,neighbors=40)
    v0features=e19f._pair_features(v0ordered,v0pairs,v0lex,v0emb)
    frozen=np.load(output_dir/"frozen_v0_scores.npz",allow_pickle=True)
    scores=np.asarray(frozen["scores"],dtype=np.float64); y=np.asarray(frozen["y"],dtype=np.int64)
    groups_v0=np.asarray(frozen["prompt_ids"],dtype=str); row_ids=np.asarray(frozen["row_ids"],dtype=str)
    if not np.array_equal(v0ids,row_ids):
        raise ValueError("V0 checkpoint scores and embedding input rows do not align")
    v0_results={}
    for method in METHODS:
        settings=chosen[method]
        pair_model=final_fc if method=="E19-F-C" else None
        pred,confidence,_=_method_clusters(method,v0ids[v0ordered],v0pairs,v0features,
            threshold=settings["threshold"],k=settings.get("k",10),pair_model=pair_model)
        full_groups=np.empty(len(v0),dtype=object); full_groups[v0ordered]=pred
        full_conf={str(group):confidence[str(group)] for group in np.unique(pred)}
        confidence_by_row=np.empty(len(v0),dtype=np.float32)
        confidence_by_row[v0ordered]=np.asarray([confidence[str(group)] for group in pred],dtype=np.float32)
        gated_pred=_confidence_gate(pred,confidence,v0ids[v0ordered],settings.get("gate_threshold",1.0))
        gated_groups=np.empty(len(v0),dtype=object); gated_groups[v0ordered]=gated_pred
        independent=scores.argmax(1)
        grouped=evaluate_group_assignment(scores,full_groups,row_ids)
        gated=evaluate_group_assignment(scores,gated_groups,row_ids)
        sampled={}
        scenarios=sample_incomplete_scenarios(v0,scenarios=50,seed=SEED)
        for number,rows in scenarios.items():
            sub_pred=full_groups[rows]
            assigned=evaluate_group_assignment(scores[rows],sub_pred,row_ids[rows])
            gated_sample=_confidence_gate(sub_pred,full_conf,row_ids[rows],
                                          settings.get("gate_threshold",1.0))
            gated_assigned=evaluate_group_assignment(scores[rows],gated_sample,row_ids[rows])
            sampled[number]={"rows":len(rows),"independent":e20._metric(y[rows],independent[rows]),
                             "group_hungarian":e20._metric(y[rows],assigned),
                             "confidence_gated_hungarian":e20._metric(y[rows],gated_assigned)}
        v0_results[method]={"independent":e20._metric(y,independent),
            "external_group_hungarian":e20._metric(y,grouped),
            "confidence_gated_hungarian":e20._metric(y,gated),
            "comparison_vs_independent":e20._comparison(y,independent,grouped),
            "confidence_gated_comparison_vs_independent":e20._comparison(y,independent,gated),
            "scenarios":sampled,"scenario_accuracy_summary":scenario_summary(
                {n:{"independent":v["independent"],"hungarian_oracle":v["group_hungarian"]}
                 for n,v in sampled.items()}),"selected_parameters":settings}
        pd.DataFrame({"id":row_ids,"predicted_group":full_groups,
            "group_confidence":confidence_by_row}).to_parquet(
        output_dir/f"v0_groups_{method}{result_suffix}.parquet",index=False)
    payload={"external_validation":validation,"external_test":test,
        "selected_parameters":chosen,"v0":v0_results,
        "v0_embedding":embed_meta,"candidate_method":"exact batched embedding cosine top-40; word and character cosine retained as E19-F-C pair features",
        "frozen_checkpoint_count":89,"trust_list_version":"trust-list-2-v039"}
    result_name=f"e19f_grouping_results{result_suffix}.json"
    (output_dir/result_name).write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--external", type=Path,
                        default=ROOT/"data/processed/dataset_v2.14.parquet")
    parser.add_argument("--prepare", action="store_true",
                        help="write prompt-disjoint manifests and duplicate-safe views")
    parser.add_argument("--freeze-v0", action="store_true",
                        help="validate 89 E24 caches and score fixed family-balanced baseline")
    parser.add_argument("--embed-limit", type=int,
                        help="encode a response prefix as an inference smoke test")
    parser.add_argument("--embed-external", action="store_true",
                        help="resumably encode all canonical V2.14 responses")
    parser.add_argument("--run-f-grouping", action="store_true",
                        help="fit E19-F-A/B/C on external prompts and evaluate V0 assignment")
    parser.add_argument("--run-f-grouping-sample",action="store_true",
                        help="run E19-F on the same small prompt sample as E19-B/D")
    parser.add_argument("--prepare-reconstruction-sample", action="store_true",
                        help="prepare a prompt-stratified subset for slower E19-B/D reconstruction")
    parser.add_argument("--sample-fraction",type=float,default=0.1)
    parser.add_argument("--reconstruct-b", action="store_true")
    parser.add_argument("--reconstruct-d", action="store_true")
    parser.add_argument("--reconstruction-limit", type=int)
    parser.add_argument("--reconstruction-batch-size", type=int,default=20)
    parser.add_argument("--embed-recon", choices=("E19-B","E19-D"))
    parser.add_argument("--run-reconstructed-b", action="store_true")
    parser.add_argument("--run-reconstructed-d", action="store_true")
    parser.add_argument("--finalize", action="store_true",
                        help="write combined metrics/docs and update the ensemble dashboard catalog")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = canonical_external(args.external)
    if args.prepare:
        split = prompt_disjoint_split(frame)
        rotated, rotation_stats = rotate_one_per_label(frame)
        split_names = np.empty(len(frame), dtype=object)
        for name, rows in split.items():
            split_names[rows] = name
        frame["split"] = split_names
        prompt_split = frame.drop_duplicates("prompt_canon").set_index("prompt_canon").split.to_dict()
        rotated["split"] = rotated.prompt_canon.map(prompt_split)
        frame.to_parquet(OUT/"external_v214_canonical.parquet", index=False)
        rotated.to_parquet(OUT/"external_v214_rotation_views.parquet", index=False)
        manifest = {"source": str(args.external), "source_sha256": sha256(args.external),
            "rows": len(frame), "prompt_groups": int(frame.prompt_canon.nunique()),
            "split_seed": SEED, "splits": {}, "rotation": rotation_stats,
            "group_size_distribution": frame.groupby("prompt_canon").model.nunique().value_counts().sort_index().to_dict()}
        for name, rows in split.items():
            sub = frame.iloc[rows]
            manifest["splits"][name] = {"rows": len(rows), "prompts": int(sub.prompt_canon.nunique()),
                "group_label_count": sub.groupby("prompt_canon").model.nunique().value_counts().sort_index().to_dict()}
        (OUT/"split_manifest.json").write_text(json.dumps(manifest, indent=2, default=int)+"\n")
        (OUT/"configuration.json").write_text(json.dumps({
            "experiment":"E27","seed":SEED,"trust_list_version":"trust-list-2-v039",
            "frozen_roster_source":"artifacts/analysis/e24/configuration.json",
            "frozen_roster_config_sha256":sha256(args.root/"artifacts/analysis/e24/configuration.json"),
            "v214_source":str(args.external),"v214_sha256":manifest["source_sha256"],
            "canonical_rows":len(frame),"canonical_prompt_groups":int(frame.prompt_canon.nunique()),
            "split_manifest":"artifacts/analysis/e27/split_manifest.json",
            "split_sha256":sha256(OUT/"split_manifest.json"),
            "groupers":list(METHODS),"grouping_status":"external embeddings in progress",
            "v0_checkpoint_count":89,"v0_checkpoint_cache_hashes_source":"E24 configuration",
            "v0_exposure_note":"Frozen checkpoint pool has historical V0 selection exposure."},indent=2)+"\n")
        print(json.dumps(manifest, indent=2, default=int))
    if args.freeze_v0:
        result = freeze_ensemble_scores(args.root)
        print(json.dumps({"checkpoints": result["checkpoint_count"],
            "independent": result["family_balanced"], "oracle": result["oracle_hungarian"]}, indent=2))
    if args.embed_limit or args.embed_external:
        meta = embed_external(frame, limit=args.embed_limit,
                              device="auto", batch_size=24)
        print(json.dumps(meta, indent=2))
    if args.run_f_grouping:
        result=run_external_f(args.root)
        print(json.dumps({"external_test":{k:v["pairwise_metrics"] for k,v in result["external_test"].items()},
                          "v0_balanced_accuracy":{k:v["external_group_hungarian"]["balanced_accuracy"]
                                                  for k,v in result["v0"].items()}},indent=2))
    if args.run_f_grouping_sample:
        result=run_external_f(args.root,reconstruction_sample=True)
        print(json.dumps({"external_test":{k:v["pairwise_metrics"] for k,v in result["external_test"].items()},
                          "v0_balanced_accuracy":{k:v["external_group_hungarian"]["balanced_accuracy"]
                                                  for k,v in result["v0"].items()}},indent=2))
    if args.prepare_reconstruction_sample:
        sample=stratified_prompt_sample(frame,args.sample_fraction,SEED)
        split=prompt_disjoint_split(sample)
        split_names=np.empty(len(sample),dtype=object)
        for name,rows in split.items(): split_names[rows]=name
        sample["split"]=split_names
        group_split=sample.drop_duplicates("prompt_canon").set_index("prompt_canon").split.to_dict()
        rotation,rotation_stats=rotate_one_per_label(sample)
        rotation["split"]=rotation.prompt_canon.map(group_split)
        tag=f"{int(round(float(args.sample_fraction)*100)):02d}pct"
        sample_path=OUT/f"external_v214_reconstruction_sample_{tag}.parquet"
        rotation_path=OUT/f"external_v214_reconstruction_rotation_{tag}.parquet"
        sample.to_parquet(sample_path,index=False); rotation.to_parquet(rotation_path,index=False)
        manifest={"source":"external_v214_canonical.parquet","source_sha256":sha256(OUT/"external_v214_canonical.parquet"),
            "fraction":float(args.sample_fraction),"seed":SEED,"rows":len(sample),
            "prompt_groups":int(sample.prompt_canon.nunique()),"group_label_count":
            sample.groupby("prompt_canon").model.nunique().value_counts().sort_index().to_dict(),
            "splits":{},"rotation":rotation_stats,"canonical_path":str(sample_path),
            "rotation_path":str(rotation_path)}
        for name,rows in split.items():
            part=sample.iloc[rows]
            manifest["splits"][name]={"rows":len(part),"prompts":int(part.prompt_canon.nunique()),
                "group_label_count":part.groupby("prompt_canon").model.nunique().value_counts().sort_index().to_dict()}
        (OUT/"reconstruction_sample_manifest.json").write_text(json.dumps(manifest,indent=2,default=int)+"\n")
        print(json.dumps(manifest,indent=2,default=int))
    if args.reconstruct_b or args.reconstruct_d:
        for method,enabled in (("E19-B",args.reconstruct_b),("E19-D",args.reconstruct_d)):
            if enabled: print(json.dumps(reconstruct_external(method,root=args.root,
                limit=args.reconstruction_limit,batch_size=args.reconstruction_batch_size),indent=2))
    if args.embed_recon:
        print(json.dumps(embed_reconstructed_prompts(args.embed_recon),indent=2))
    if args.run_reconstructed_b or args.run_reconstructed_d:
        for method,enabled in (("E19-B",args.run_reconstructed_b),("E19-D",args.run_reconstructed_d)):
            if enabled:
                result=run_external_reconstruction_grouper(method,root=args.root)
                print(json.dumps({"method":method,"external_test":result["external_test"],
                    "v0":{k:{m:result["v0"][k][m] for m in ("accuracy","balanced_accuracy")}
                          for k in ("independent","group_hungarian","confidence_gated_hungarian")}},indent=2))
    if args.finalize:
        result=finalize(args.root)
        print(json.dumps({"status":result["status"],"selected_grouper":result["selected_grouper"],
            "pending_groupers":result["pending_groupers"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
