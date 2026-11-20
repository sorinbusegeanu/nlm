"""Leakage-controlled evaluation for E19-B prompt reconstruction grouping."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold

from . import e16_common, e19b_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e19b"
EMBEDDING_MODEL = "google/embeddinggemma-2"
EMBEDDING_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
THRESHOLDS = tuple(np.round(np.arange(.35, .951, .05), 2))
C_VALUES = (0.1, 1.0, 10.0)
SEED = 2026


def encode_semantic(texts, *, model_id=EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
                    batch_size=32, max_length=512):
    """Transform reconstructed prompts with the pinned local Transformers model."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("E19-B semantic embedding inference requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
        local_files_only=True, padding_side="right")
    model = AutoModel.from_pretrained(model_id, revision=revision,
        local_files_only=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
    model.to("cuda").eval()
    vectors = []
    try:
        for start in range(0, len(texts), int(batch_size)):
            batch = ["task: clustering | query: " + str(x) for x in texts[start:start+batch_size]]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model(**inputs)
                hidden = output.last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            if (start // int(batch_size) + 1) % 20 == 0:
                print(f"E19-B semantic embeddings {min(start+batch_size,len(texts))}/{len(texts)}", flush=True)
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0)


def grouping_metrics(true_groups, predicted_groups):
    truth = np.asarray(true_groups, dtype=str)
    pred = np.asarray(predicted_groups, dtype=str)
    if truth.shape != pred.shape:
        raise ValueError("E19-B metric rows are not aligned")
    n = len(truth)
    contingency = {}
    true_sizes, pred_sizes = Counter(truth), Counter(pred)
    for t, p in zip(truth, pred):
        contingency[(t, p)] = contingency.get((t, p), 0) + 1
    tp = sum(v*(v-1)//2 for v in contingency.values())
    predicted_pairs = sum(v*(v-1)//2 for v in pred_sizes.values())
    true_pairs = sum(v*(v-1)//2 for v in true_sizes.values())
    precision = tp/predicted_pairs if predicted_pairs else 1.0
    recall = tp/true_pairs if true_pairs else 1.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    bcp = sum(v*v/ pred_sizes[p] for (t,p),v in contingency.items()) / max(1,n)
    bcr = sum(v*v/ true_sizes[t] for (t,p),v in contingency.items()) / max(1,n)
    true_members = {t:set() for t in true_sizes}; pred_members={p:set() for p in pred_sizes}
    for i,(t,p) in enumerate(zip(truth,pred)):
        true_members[t].add(i);pred_members[p].add(i)
    predicted_member_sets={frozenset(members) for members in pred_members.values()}
    exact_groups = sum(frozenset(members) in predicted_member_sets for members in true_members.values())
    exact_responses = sum(len(members) for members in true_members.values()
                          if frozenset(members) in predicted_member_sets)
    split = sum(len({pred[i] for i in members}) > 1 for members in true_members.values())
    merged = sum(len({truth[i] for i in members}) > 1 for members in pred_members.values())
    predicted_cluster_sizes = Counter(pred_sizes.values())
    true_cluster_sizes = Counter(true_sizes.values())
    size_analysis={}
    for size in sorted(set(true_sizes.values()) | set(pred_sizes.values())):
        true_at_size=[members for members in true_members.values() if len(members)==size]
        exact_at_size=sum(frozenset(members) in predicted_member_sets for members in true_at_size)
        size_analysis[str(size)]={"original_groups":len(true_at_size),
            "reconstructed_groups":int(predicted_cluster_sizes.get(size,0)),
            "exactly_recovered_original_groups":int(exact_at_size),
            "original_group_exact_recovery_rate":exact_at_size/max(1,len(true_at_size))}
    return {"rows": n, "exact_group_recovery": exact_groups/max(1,len(true_sizes)),
        "exact_groups": int(exact_groups), "true_group_count": len(true_sizes),
        "response_exact_grouping": exact_responses/max(1,n), "exact_responses": int(exact_responses),
        "pairwise_precision": precision, "pairwise_recall": recall, "pairwise_f1": f1,
        "cluster_purity": sum(max(Counter(truth[list(m)]).values()) for m in pred_members.values())/max(1,n),
        "bcubed_precision": bcp, "bcubed_recall": bcr,
        "bcubed_f1": 2*bcp*bcr/(bcp+bcr) if bcp+bcr else 0.0,
        "split_original_groups": int(split), "incorrectly_merged_groups": int(merged),
        "singleton_responses": int(sum(len(m)==1 for m in pred_members.values())),
        "predicted_group_count": len(pred_sizes),
        "predicted_size_distribution": dict(sorted(predicted_cluster_sizes.items())),
        "true_size_distribution": dict(sorted(true_cluster_sizes.items())),
        "group_size_analysis": size_analysis,
        "ari": float(adjusted_rand_score(truth,pred))}


def _pairs_and_features(rows, recon, responses, embeddings, topics, details, constraints, prompt_groups, *, fit_rows):
    rows=np.sort(np.asarray(rows,dtype=np.int64),kind="stable")
    fit_rows=np.sort(np.asarray(fit_rows,dtype=np.int64),kind="stable")
    spaces = e19b_group.fit_similarity_spaces(recon, responses, fit_rows)
    pairs = e19b_group.candidate_pairs(rows, spaces, embeddings)
    feats = e19b_group.pair_features(rows, pairs, spaces, recon, responses, embeddings,
                                     topics,
                                     details, constraints)
    target = None
    if prompt_groups is not None:
        target = (np.asarray(prompt_groups)[rows[pairs[:,0]]] ==
                  np.asarray(prompt_groups)[rows[pairs[:,1]]]).astype(np.int8)
    return spaces, pairs, feats, target


def _cluster(rows, ids, pairs, scores, threshold):
    return e19b_group.cluster_complete_link(ids[rows], pairs, scores, threshold=threshold)[0]


def tune_on_training(rows, recon, responses, embeddings, topics, details, constraints, groups,
                     *, threshold_grid=THRESHOLDS, c_values=C_VALUES):
    """Inner group cross-fit tuning; outer-held rows are never passed here."""
    if not threshold_grid or any(not np.isfinite(x) or not 0<=float(x)<=1 for x in threshold_grid):
        raise ValueError("E19-B threshold grid must contain values in [0, 1]")
    if not c_values or any(not np.isfinite(x) or float(x)<=0 for x in c_values):
        raise ValueError("E19-B C values must be positive")
    rows=np.sort(np.asarray(rows,dtype=np.int64),kind="stable")
    records = []
    splits = GroupKFold(n_splits=4).split(rows, groups=groups[rows])
    for inner, (fit_local, valid_local) in enumerate(splits, 1):
        fit_rows, valid_rows = np.sort(rows[fit_local]), np.sort(rows[valid_local])
        _, p_fit, x_fit, y_fit = _pairs_and_features(fit_rows, recon, responses, embeddings,
            topics, details, constraints, groups, fit_rows=fit_rows)
        _, p_valid, x_valid, _ = _pairs_and_features(valid_rows, recon, responses, embeddings,
            topics, details, constraints, None, fit_rows=fit_rows)
        for c in c_values:
            bundle = e19b_group.fit_pair_scorer(x_fit, y_fit, c=c, seed=SEED)
            scores = e19b_group.score_pairs(bundle, x_valid)
            for threshold in threshold_grid:
                pred = _cluster(valid_rows, np.asarray([str(x) for x in range(len(groups))]),
                                p_valid, scores, threshold)
                metric = grouping_metrics(groups[valid_rows], pred)
                records.append({"inner_fold": inner, "C": c, "threshold": float(threshold),
                    "pairwise_f1": metric["pairwise_f1"], "pairwise_precision": metric["pairwise_precision"]})
    summary = {}
    for c in c_values:
        for threshold in threshold_grid:
            subset = [x for x in records if x["C"] == c and x["threshold"] == float(threshold)]
            summary[(c,float(threshold))] = (np.mean([x["pairwise_f1"] for x in subset]),
                                             np.mean([x["pairwise_precision"] for x in subset]))
    best = max(summary, key=lambda key: (summary[key][0], summary[key][1], -key[0], key[1]))
    return {"C": best[0], "threshold": best[1], "inner_summary": records,
            "mean_pairwise_f1": float(summary[best][0]),
            "mean_pairwise_precision": float(summary[best][1])}


def _save_embeddings(texts, path):
    path = Path(path)
    meta = path.with_suffix(".json")
    if path.is_file() and meta.is_file():
        info = json.loads(meta.read_text())
        if (info.get("count") == len(texts) and info.get("model")==EMBEDDING_MODEL
            and info.get("revision")==EMBEDDING_REVISION
            and info.get("sha256") == hashlib.sha256("\0".join(texts).encode()).hexdigest()):
            return np.load(path)
    values = encode_semantic(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    meta.write_text(json.dumps({"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
        "count": len(texts), "sha256": __import__("hashlib").sha256("\0".join(texts).encode()).hexdigest()}, indent=2))
    return values


def predict_groups(row_ids, responses, reconstructions, semantic_embeddings, *,
                   model_path=OUT/"e19b_grouper.joblib"):
    """Group unlabeled response rows (for example Kaggle id/text rows).

    `reconstructions` are parsed JSON objects from `reconstruct_rows`; this
    function accepts no prompt IDs or labels.
    """
    bundle=joblib.load(model_path)
    ids=np.asarray(row_ids,dtype=str); responses=np.asarray(responses,dtype=str)
    reconstructions=list(reconstructions); embeddings=np.asarray(semantic_embeddings,dtype=np.float32)
    if len(set(ids.tolist()))!=len(ids) or len(ids)!=len(responses) or len(ids)!=len(reconstructions):
        raise ValueError("E19-B unlabeled IDs, texts, and reconstructions must be unique and aligned")
    if embeddings.ndim!=2 or embeddings.shape[0]!=len(ids):
        raise ValueError("E19-B semantic embeddings do not align to unlabeled rows")
    order=np.argsort(ids,kind="stable")
    ids=ids[order];responses=responses[order]
    reconstructions=[reconstructions[i] for i in order]
    embeddings=embeddings[order]
    recon=np.asarray([x["reconstructed_prompt"] for x in reconstructions],dtype=str)
    topics=np.asarray([x.get("topic","") for x in reconstructions],dtype=str)
    details=[x.get("distinctive_details",[]) for x in reconstructions]
    constraints=[x.get("constraints",[]) for x in reconstructions]
    spaces=e19b_group.transform_similarity_spaces(recon,responses,bundle["vectorizers"])
    rows=np.arange(len(ids))
    pairs=e19b_group.candidate_pairs(rows,spaces,embeddings)
    features=e19b_group.pair_features(rows,pairs,spaces,recon,responses,embeddings,
                                      topics,details,constraints)
    scores=e19b_group.score_pairs(bundle["pair_scorer"],features) if len(pairs) else np.empty(0)
    groups,_=e19b_group.cluster_complete_link(ids,pairs,scores,
        threshold=bundle["threshold"],max_cluster_size=18)
    aligned=np.empty(len(groups),dtype=object)
    aligned[order]=groups
    return aligned


def run(root=ROOT, *, reconstruct_path=None, embeddings_path=None, output_dir=OUT,
        threshold_grid=THRESHOLDS, c_values=C_VALUES):
    started = time.time()
    if not threshold_grid or any(not np.isfinite(x) or not 0<=float(x)<=1 for x in threshold_grid):
        raise ValueError("E19-B threshold grid must contain values in [0, 1]")
    ctx = e16_common.load_context(root)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = pd.read_parquet(reconstruct_path or output_dir/"reconstructed_prompts.parquet")
    if reconstructed.id.astype(str).tolist() != ctx["row_ids"].astype(str).tolist():
        raise ValueError("E19-B reconstruction rows do not exactly align to V0 IDs")
    if not reconstructed.status.eq("ok").all():
        raise ValueError("E19-B requires successful reconstructions for all V0 rows")
    recon = reconstructed.reconstructed_prompt.fillna("").astype(str).to_numpy()
    responses = ctx["texts"]
    details = reconstructed.distinctive_details.map(lambda x: json.loads(x) if isinstance(x,str) else x).tolist()
    constraints = reconstructed.constraints.map(lambda x: json.loads(x) if isinstance(x,str) else x).tolist()
    repair_mask=(reconstructed.generation_repaired.fillna(False).astype(bool).to_numpy()
                 if "generation_repaired" in reconstructed else np.zeros(len(reconstructed),dtype=bool))
    topics = reconstructed.topic.fillna("").astype(str).to_numpy()
    embeddings = _save_embeddings(recon, embeddings_path or output_dir/"semantic_embeddings.npy")
    n = len(recon); ids = ctx["row_ids"].astype(str)
    groups = ctx["groups"].astype(str)
    reconstruction_manifest_path=Path(reconstruct_path or output_dir/"reconstructed_prompts.parquet").parent/"reconstruction_manifest.json"
    reconstruction_manifest=(json.loads(reconstruction_manifest_path.read_text())
        if reconstruction_manifest_path.is_file() else {})
    configuration={"experiment":"E19-B","seed":SEED,"outer_folds":5,
        "inner_folds":4,"threshold_grid":[float(x) for x in threshold_grid],"C_grid":list(c_values),
        "neighbors":e19b_group.DEFAULT_NEIGHBORS,"max_cluster_size":18,
        "semantic_model":EMBEDDING_MODEL,"semantic_revision":EMBEDDING_REVISION,
        "reconstruction_model":str(reconstructed.model_id.iloc[0]),
        "reconstruction_revision":str(reconstructed.model_revision.iloc[0]),
        "prompt_version":str(reconstructed.prompt_version.iloc[0]),
        "reconstruction_settings":{k:reconstruction_manifest.get(k) for k in
            ("batch_size","max_input_tokens","max_new_tokens","do_sample","num_beams","thinking_mode")},
        "v0_row_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "v0_response_text_sha256":hashlib.sha256("\0".join(responses).encode()).hexdigest(),
        "grouping_uses_truth_for":"inner pair labels and threshold selection only; held outer rows excluded"}
    (output_dir/"configuration.json").write_text(json.dumps(configuration,indent=2))
    held_prediction = np.empty(n, dtype=object); seen = np.zeros(n,dtype=np.int8)
    folds=[]; diagnostics=[]
    for outer in ctx["outer_folds"]:
        fit_rows=np.sort(outer["fit_rows"],kind="stable")
        held_rows=np.sort(outer["held_rows"],kind="stable")
        selected = tune_on_training(fit_rows,recon,responses,embeddings,topics,details,constraints,
                                    groups,threshold_grid=threshold_grid,c_values=c_values)
        _, pfit,xfit,yfit = _pairs_and_features(fit_rows,recon,responses,embeddings,
            topics,details,constraints,groups,fit_rows=fit_rows)
        _, phold,xhold,_ = _pairs_and_features(held_rows,recon,responses,embeddings,
            topics,details,constraints,None,fit_rows=fit_rows)
        model=e19b_group.fit_pair_scorer(xfit,yfit,c=selected["C"],seed=SEED)
        scores=e19b_group.score_pairs(model,xhold)
        pred,cluster_info=e19b_group.cluster_complete_link(ids[held_rows],phold,scores,
            threshold=selected["threshold"],max_cluster_size=18)
        pred=np.asarray([f"fold-{outer['fold']}-"+str(value) for value in pred],dtype=object)
        held_prediction[held_rows]=pred; seen[held_rows]+=1
        fm=grouping_metrics(groups[held_rows],pred)
        folds.append({"fold":outer["fold"],"fit_rows":len(fit_rows),"held_rows":len(held_rows),
            "selected":{k:v for k,v in selected.items() if k!="inner_summary"},"metrics":fm,
            "coefficients":dict(zip(e19b_group.FEATURE_NAMES,model["coefficients"])),
            "clusters":cluster_info["clusters"]})
        for pi,(a,b) in enumerate(phold):
            diagnostics.append({"fold":outer["fold"],"left_id":ids[held_rows[a]],
                "right_id":ids[held_rows[b]],"left_true_prompt_id":groups[held_rows[a]],
                "right_true_prompt_id":groups[held_rows[b]],"same_true_group":bool(groups[held_rows[a]]==groups[held_rows[b]]),
                "similarity":float(scores[pi])})
        print(f"E19-B outer fold {outer['fold']}: pair-F1={fm['pairwise_f1']:.4f}, "
              f"precision={fm['pairwise_precision']:.4f}, threshold={selected['threshold']}",flush=True)
    if not np.all(seen==1): raise ValueError("E19-B held-out rows not covered exactly once")
    aggregate=grouping_metrics(groups,held_prediction)
    fold_mean={k:float(np.mean([x["metrics"][k] for x in folds])) for k in
        ("pairwise_precision","pairwise_recall","pairwise_f1","cluster_purity","bcubed_precision","bcubed_recall","bcubed_f1")}
    # Full-data clustering is explicitly diagnostic only and never part of CV metrics.
    all_rows=np.arange(n)
    fold_configs=[x["selected"] for x in folds]
    c_counts=Counter(x["C"] for x in fold_configs)
    full_c=min((c for c,count in c_counts.items() if count==max(c_counts.values())))
    full_threshold=float(np.median([x["threshold"] for x in fold_configs]))
    full_selection={"C":float(full_c),"threshold":full_threshold,
                    "source":"mode C and median threshold from outer-fold training-only selections"}
    full_spaces,pall,xall,yall=_pairs_and_features(all_rows,recon,responses,embeddings,topics,details,constraints,groups,fit_rows=all_rows)
    full_model=e19b_group.fit_pair_scorer(xall,yall,c=full_selection["C"],seed=SEED)
    full_scores=e19b_group.score_pairs(full_model,xall)
    full_pred,full_info=e19b_group.cluster_complete_link(ids,pall,full_scores,threshold=full_selection["threshold"])
    full_metrics=grouping_metrics(groups,full_pred)
    deploy_bundle={"pair_scorer":full_model,"vectorizers":{name:full_spaces[name+"_vectorizer"] for name in
        ("reconstruction_char","reconstruction_word","response_char","response_word")},
        "threshold":full_selection["threshold"],"max_cluster_size":18,
        "neighbors":e19b_group.DEFAULT_NEIGHBORS,"semantic_model":EMBEDDING_MODEL,
        "semantic_revision":EMBEDDING_REVISION,"configuration":full_selection}
    tmp_bundle=output_dir/"e19b_grouper.joblib.tmp"
    joblib.dump(deploy_bundle,tmp_bundle,compress=3)
    tmp_bundle.replace(output_dir/"e19b_grouper.joblib")
    pred_frame=pd.DataFrame({"id":ids,"true_prompt_id":groups,"predicted_group_id":held_prediction,
        "reconstructed_prompt":recon,"topic":topics,"distinctive_details":details,"constraints":constraints,
        "generation_repaired":repair_mask,
        "fold":np.zeros(n,dtype=int)})
    # Restore original row order for folds column, which is stored in context order.
    fold_col=np.empty(n,dtype=int)
    for x in ctx["outer_folds"]: fold_col[x["held_rows"]]=x["fold"]
    pred_frame["fold"]=fold_col
    pred_frame.to_parquet(output_dir/"oof_group_predictions.parquet",index=False)
    pd.DataFrame(diagnostics).to_parquet(output_dir/"oof_pair_diagnostics.parquet",index=False)
    pd.DataFrame({"id":ids,"true_prompt_id":groups,"predicted_group_id":full_pred,
        "reconstructed_prompt":recon}).to_parquet(output_dir/"all_v0_diagnostic_groups.parquet",index=False)
    metrics={"protocol":"five-fold saved E13 prompt-group outer folds; fold-local inner GroupKFold tuning",
        "rows":n,"true_prompt_groups":int(len(np.unique(groups))),"aggregate_oof":aggregate,
        "generation_repaired_rows":int(repair_mask.sum()),
        "mean_outer_fold_metrics":fold_mean,"outer_folds":folds,
        "all_v0_diagnostic_only":{"selected":{k:v for k,v in full_selection.items() if k!="inner_summary"},"metrics":full_metrics},
        "evaluation_seconds":time.time()-started,"semantic_model":{"model":EMBEDDING_MODEL,"revision":EMBEDDING_REVISION}}
    (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    examples=_diagnostic_examples(pred_frame,pd.DataFrame(diagnostics))
    (output_dir/"diagnostic_examples.json").write_text(json.dumps(examples,indent=2,ensure_ascii=False))
    report_path=output_dir/"e19b.md"
    _write_report(metrics, pred_frame, pd.DataFrame(diagnostics), output_dir/"diagnostics.json", report_path)
    (Path(root)/"docs/e19b.md").write_text(report_path.read_text())
    return metrics


def _write_report(metrics, predictions, pair_df, diagnostics_path, report_path):
    m=metrics["aggregate_oof"]
    correct_groups=m["exact_groups"]; total_groups=m["true_group_count"]
    correct_responses=m["exact_responses"]; rows=m["rows"]
    if m["pairwise_precision"]>=0.95:
        conclusion=(f"Proceed with E19-B Hungarian-assignment evaluation: held-out pairwise precision is "
            f"{100*m['pairwise_precision']:.2f}% (at the 95% conservative gate), with "
            f"{100*m['pairwise_recall']:.2f}% pairwise recall. Treat E19-A's +10.26pp as an oracle upper bound, not an expected gain.")
    else:
        conclusion=(f"Do not yet use these groups as hard Hungarian constraints: held-out pairwise precision is "
            f"{100*m['pairwise_precision']:.2f}%, below the 95% conservative gate. A 10.26pp gain was measured only with perfect groups; current merges could impose wrong unique-label constraints.")
    lines=[f"- Exact original-group recovery: {100*m['exact_group_recovery']:.2f}% ({correct_groups}/{total_groups})",
      f"- Response-level exact grouping: {100*m['response_exact_grouping']:.2f}% ({correct_responses}/{rows})",
      f"- Pairwise grouping precision: {100*m['pairwise_precision']:.2f}%",
      f"- Pairwise grouping recall: {100*m['pairwise_recall']:.2f}%",
      f"- Pairwise grouping F1: {100*m['pairwise_f1']:.2f}%",
      f"- Reconstructed group count: {m['predicted_group_count']}",
      f"- Incorrect merges: {m['incorrectly_merged_groups']}",
      f"- Split original groups: {m['split_original_groups']}","",
      "## Cross-validated metrics","", "```json",json.dumps(m,indent=2),"```","",
      "## Fold metrics","", "```json",json.dumps(metrics["outer_folds"],indent=2),"```","",
      "## Full-V0 diagnostic","",
      "This uses the complete V0 labels for fitting and is diagnostic only; it is not a held-out result.","",
      "```json",json.dumps(metrics["all_v0_diagnostic_only"],indent=2),"```","",
      "## Interpretation","",
      conclusion,"",
      "## Source availability","",
      "The local V0 source exposes response IDs, prompt_id, response text, and model label, but no original user prompt text. Therefore original_prompt_text is unavailable and is not reconstructed from hidden labels.",
      f"The reconstruction parser repaired {metrics.get('generation_repaired_rows',0)} truncated or malformed model outputs using only completed JSON fields and list items; raw outputs and attempt histories remain in the SQLite cache.","",
      "## Diagnostics","",
      "See `diagnostic_examples.json` for up to 20 correctly recovered, incorrectly merged, and split group examples. The OOF predictions had zero incorrectly merged groups, so there are no held-out incorrect-merge examples to show; the all-V0 diagnostic is separately labeled and must not be treated as held-out evidence. Pairwise edge scores are in `oof_pair_diagnostics.parquet`; OOF response-level details are in `oof_group_predictions.parquet`.","",
      "## Method and rerun commands","",
      "Qwen3.5-2B reconstructs each response independently with greedy deterministic generation and a resumable response-hash/model-revision/prompt-version cache. Grouping combines E16 word/character lexical similarities on reconstructed prompts and original responses with frozen EmbeddingGemma2 semantic similarity, topic overlap, distinctive-detail/entity/number overlap, and constraint overlap. A regularized logistic pair scorer supplies edge probabilities; complete-link agglomeration rejects weak transitive bridges and caps clusters at 18.","",
      "Each saved E13-B outer fold remains untouched for final scoring. Within its training partition, four prompt-group inner folds select the logistic regularization and threshold by mean pairwise F1, then precision. The full-V0 grouping is a separately marked diagnostic.","",
      "Run with `/home/zodrak/cuda13.x/bin/python -m src.e19b_reconstruct --limit 100`, then `... -m src.e19b_reconstruct`, then `/home/zodrak/cuda13.x/bin/python -m src.e19b_evaluate`. For new unlabeled rows use `reconstruct_rows` with separate cache/output paths and `predict_groups` with the saved `e19b_grouper.joblib`."]
    report_path.write_text("\n".join(lines)+"\n")


def _diagnostic_examples(frame, pair_df):
    """Build human-readable examples without source prompt text (not present in V0)."""
    true_groups = {key: part for key, part in frame.groupby("true_prompt_id", sort=True)}
    predicted = {key: part for key, part in frame.groupby("predicted_group_id", sort=True)}
    correctly_recovered, merged, split = [], [], []
    def example(part, kind):
        ids = set(part.id.astype(str))
        edges = pair_df[pair_df.left_id.astype(str).isin(ids) & pair_df.right_id.astype(str).isin(ids)]
        similarities = [{"left_id":str(r.left_id),"right_id":str(r.right_id),"score":float(r.similarity)}
                        for r in edges.nlargest(10,"similarity").itertuples()]
        return {"type":kind,"response_ids":sorted(ids),
            "true_prompt_ids":sorted(part.true_prompt_id.astype(str).unique().tolist()),
            "reconstructed_prompts":part[["id","reconstructed_prompt"]].to_dict("records"),
            "predicted_group_ids":sorted(part.predicted_group_id.astype(str).unique().tolist()),
            "similarity_scores":similarities,"original_prompt_text":None}
    for _, part in true_groups.items():
        if part.predicted_group_id.nunique()==1:
            cluster=predicted[str(part.predicted_group_id.iloc[0])]
            if len(cluster)==len(part) and set(cluster.id)==set(part.id):
                correctly_recovered.append(example(part,"correctly_recovered"))
        else:
            split.append(example(part,"split_original_group"))
    for _, part in predicted.items():
        if part.true_prompt_id.nunique()>1:
            merged.append(example(part,"incorrectly_merged"))
    # Prefer varied cases instead of selecting only largest groups.
    correctly_recovered.sort(key=lambda x:(len(x["response_ids"]),x["response_ids"][0]))
    merged.sort(key=lambda x:(-len(x["true_prompt_ids"]),x["response_ids"][0]))
    split.sort(key=lambda x:(-len(x["response_ids"]),x["response_ids"][0]))
    topic_collisions=0
    for _,part in predicted.items():
        if part.true_prompt_id.nunique()>1:
            topic_sets=[set(str(x).casefold().split()) for x in part.topic]
            if any(topic_sets[i] & topic_sets[j] for i in range(len(topic_sets))
                   for j in range(i+1,len(topic_sets))): topic_collisions+=1
    split_topic_divergence=sum(part.topic.nunique()>1 for part in
        (frame[frame.true_prompt_id==key] for key in frame.true_prompt_id.unique())
        if part.predicted_group_id.nunique()>1)
    return {"correctly_recovered":correctly_recovered[:20],
            "incorrectly_merged":merged[:20],"split_original_groups":split[:20],
            "failure_patterns":{
                "generic_reconstruction_examples":int(frame.reconstructed_prompt.str.split().str.len().lt(12).sum()),
                "low_detail_examples":int(frame.distinctive_details.map(lambda x:len(x) if isinstance(x,list) else 0).lt(2).sum()),
                "merged_groups_with_topic_token_overlap":int(topic_collisions),
                "split_original_groups_with_multiple_reconstructed_topics":int(split_topic_divergence),
                "hallucinated_details_assessable":False,
                "source_original_prompt_text_available":False}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--reconstructions",type=Path)
    parser.add_argument("--embeddings",type=Path)
    parser.add_argument("--thresholds",help="comma-separated pair-score thresholds; defaults to the built-in conservative grid")
    args=parser.parse_args()
    thresholds=THRESHOLDS if not args.thresholds else tuple(float(x) for x in args.thresholds.split(","))
    result=run(args.root,reconstruct_path=args.reconstructions,embeddings_path=args.embeddings,
               threshold_grid=thresholds)
    print(json.dumps(result["aggregate_oof"],indent=2))


if __name__=="__main__": main()
