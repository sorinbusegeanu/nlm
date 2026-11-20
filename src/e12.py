"""Nested prompt-group evaluation of E12 pairwise arbitration."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import ensemble_catalog as catalog
from .confusion_ensemble import error_correlations
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
CLASSES = np.asarray(LABELS, dtype=object)
CONFIGS = {"E12-1": (2, 15), "E12-2": (3, 15), "E12-3": (3, 20)}
C_GRID = (0.01, 0.1, 1.0, 10.0)


def select_roster(hard, truth, families, limit):
    support = np.bincount(truth, minlength=len(LABELS))
    if not support.all():
        raise ValueError("Fitting partition lacks canonical labels")
    hits = hard == truth
    ba = (hits / support[truth]).sum(axis=1) / len(LABELS)
    corr = error_correlations(hard, truth)
    order = sorted(range(len(hard)), key=lambda i: (-ba[i], -hits[i].sum(), i))
    selected, covered = [order[0]], hits[order[0]].copy()
    steps = [{"index": order[0], "newly_covered": int(covered.sum())}]
    while len(selected) < min(limit, len(hard)):
        gain = (hits & ~covered).sum(axis=1)
        present = {families[i] for i in selected}
        chosen = min((i for i in order if i not in selected),
                     key=lambda i: (-gain[i], corr[i, selected].mean(), families[i] in present, -ba[i], i))
        steps.append({"index": chosen, "newly_covered": int(gain[chosen]),
                      "mean_positive_error_correlation": float(corr[chosen, selected].mean()),
                      "new_family": families[chosen] not in present})
        selected.append(chosen)
        covered |= hits[chosen]
    return selected, steps


def reliability(hard, truth):
    k = len(LABELS)
    support = np.bincount(truth, minlength=k)
    counts = np.stack([np.bincount(truth.astype(np.int64)*k + row, minlength=k*k).reshape(k, k)
                       for row in hard])
    tp = np.diagonal(counts, axis1=1, axis2=2)
    return {"precision": (tp+1)/(counts.sum(axis=1)+2), "recall": (tp+1)/(support[None,:]+2),
            "directional": (counts+1)/(support[None,:,None]+k), "counts": counts, "support": support}


def pair_features(probs, stats, families, a, b):
    """Antisymmetric pair features, so swapping candidates negates the vector."""
    m, n, k = probs.shape
    a, b = np.broadcast_to(np.asarray(a, dtype=int), (n,)), np.broadcast_to(np.asarray(b, dtype=int), (n,))
    row = np.arange(n)
    base = probs.mean(axis=0)
    ranks = np.argsort(np.argsort(-probs, axis=2, kind="stable"), axis=2, kind="stable")
    base_ranks = np.argsort(np.argsort(-base, axis=1, kind="stable"), axis=1, kind="stable")
    top = probs.argmax(axis=2)
    parts = []
    for j in range(m):
        pa, pb = probs[j,row,a], probs[j,row,b]
        parts.extend([pa-pb, (pa-pb)*(pa+pb), (ranks[j,row,b]-ranks[j,row,a])/k,
                      (top[j]==a).astype(float)-(top[j]==b).astype(float),
                      stats["precision"][j,a]-stats["precision"][j,b],
                      stats["recall"][j,a]-stats["recall"][j,b],
                      stats["directional"][j,b,a]-stats["directional"][j,a,b]])
    fa, fb = np.zeros(n), np.zeros(n)
    for family in sorted(set(families)):
        group = np.flatnonzero(np.asarray(families)==family)
        fa += np.any(top[group]==a, axis=0)
        fb += np.any(top[group]==b, axis=0)
    parts.extend([fa-fb,base[row,a]-base[row,b],(base_ranks[row,b]-base_ranks[row,a])/k])
    out = np.column_stack(parts)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite pair features")
    return out


def prepare_examples(probs, stats, families, truth, count):
    candidates = np.argsort(-probs.mean(axis=0), axis=1, kind="stable")[:,:count]
    features, targets = [], []
    for left, right in itertools.combinations(range(count), 2):
        a, b = candidates[:,left], candidates[:,right]
        chosen = (truth==a) | (truth==b)
        if chosen.any():
            features.append(pair_features(probs,stats,families,a,b)[chosen])
            targets.append((truth[chosen]==a[chosen]).astype(int))
    if not features:
        raise ValueError("No training examples where the true label is a candidate")
    return np.concatenate(features), np.concatenate(targets)


def fit_classifier(X, target, C):
    X, target = np.concatenate([X,-X]), np.concatenate([target,1-target])
    scaler = StandardScaler(with_mean=False).fit(X)
    model = LogisticRegression(C=C,fit_intercept=False,solver="lbfgs",max_iter=1500,random_state=2026)
    model.fit(scaler.transform(X),target)
    if model.n_iter_[0] >= model.max_iter:
        raise RuntimeError("Logistic classifier failed to converge")
    return scaler, model


def arbitrate(probs, stats, families, count, scaler, model):
    candidates = np.argsort(-probs.mean(axis=0),axis=1,kind="stable")[:,:count]
    scores = np.zeros((probs.shape[1],count))
    for left,right in itertools.combinations(range(count),2):
        X = pair_features(probs,stats,families,candidates[:,left],candidates[:,right])
        p = model.predict_proba(scaler.transform(X))[:,1]
        scores[:,left] += p
        scores[:,right] += 1-p
    return candidates[np.arange(len(candidates)),scores.argmax(axis=1)], candidates, scores


def split_inputs(probs, hard, truth, families, fit, held, limit, count):
    roster, steps = select_roster(hard[:,fit],truth[fit],families,limit)
    selected, fam = probs[roster], [families[i] for i in roster]
    stats = reliability(hard[roster][:,fit],truth[fit])
    X, target = prepare_examples(selected[:,fit],stats,fam,truth[fit],count)
    return roster, steps, stats, X, target, selected[:,held], fam


def nested_fit(probs, hard, truth, groups, row_ids, ids, families, fit, held, limit, count,
               *, inner_folds=3, c_grid=C_GRID):
    inner_data, audit = [], []
    for a,b in GroupKFold(n_splits=inner_folds).split(fit,truth[fit],groups[fit]):
        tr, va = fit[a], fit[b]
        roster, _, stats, X, target, validation, fam = split_inputs(probs,hard,truth,families,tr,va,limit,count)
        Xval, yval = prepare_examples(validation,stats,fam,truth[va],count)
        inner_data.append((X,target,Xval,yval))
        audit.append({"fit_ids":row_ids[tr].tolist(),"held_ids":row_ids[va].tolist(),
                      "checkpoint_ids":[ids[i] for i in roster],"train_pairs":len(target),"validation_pairs":len(yval)})
    tuning = []
    for C in c_grid:
        estimates, targets = [], []
        for X,target,Xval,yval in inner_data:
            scaler, model = fit_classifier(X,target,C)
            estimates.append(model.predict_proba(scaler.transform(Xval))[:,1])
            targets.append(yval)
        tuning.append({"C":C,"pairwise_log_loss":float(log_loss(np.concatenate(targets),np.concatenate(estimates),labels=[0,1]))})
    chosen = min(tuning,key=lambda item:(item["pairwise_log_loss"],item["C"]))
    roster, steps, stats, X, target, validation, fam = split_inputs(probs,hard,truth,families,fit,held,limit,count)
    scaler, model = fit_classifier(X,target,chosen["C"])
    pred, candidates, scores = arbitrate(validation,stats,fam,count,scaler,model)
    record = {"fit_ids":row_ids[fit].tolist(),"held_ids":row_ids[held].tolist(),
              "checkpoint_ids":[ids[i] for i in roster],"families":fam,
              "selection_steps":[{**{k:v for k,v in step.items() if k!="index"},"checkpoint_id":ids[step["index"]]} for step in steps],
              "selected_C":chosen["C"],"tuning":tuning,"inner_folds":audit,"training_pair_count":len(target),
              "feature_scale":scaler.scale_.tolist(),"coefficients":model.coef_[0].tolist(),"intercept":0.0,
              "precision":stats["precision"].tolist(),"recall":stats["recall"].tolist(),
              "directional_confusion":stats["directional"].tolist(),"confusion_counts":stats["counts"].tolist(),
              "true_support":stats["support"].tolist()}
    return pred,candidates,scores,record


def metrics(truth, pred, top3, candidate_count, baseline):
    top2 = (top3[:,:2]==truth[:,None]).any(axis=1)
    top3_hit = (top3==truth[:,None]).any(axis=1)
    coverage = top2 if candidate_count==2 else top3_hit
    return {"accuracy":float(accuracy_score(truth,pred)),"balanced_accuracy":float(balanced_accuracy_score(truth,pred)),
            "candidate_coverage":float(coverage.mean()),"top2_coverage":float(top2.mean()),"top3_coverage":float(top3_hit.mean()),
            "arbitration_accuracy_given_covered":float((pred[coverage]==truth[coverage]).mean()) if coverage.any() else None,
            "correct":int((pred==truth).sum()),"corrected":int(((baseline!=truth)&(pred==truth)).sum()),
            "introduced":int(((baseline==truth)&(pred!=truth)).sum())}


def evaluate_catalog_strategy(strategy, folds, v0, probabilities, members):
    """Evaluate and fit a catalog E12 configuration from the supplied input snapshot."""
    ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    count, limit = strategy["candidates"], strategy["max_checkpoints"]
    if count not in (2, 3) or len(ids) < limit:
        raise ValueError(f"E12 requires {limit} eligible checkpoints and 2 or 3 candidates")
    probs = np.stack([probabilities[sid] for sid in ids])
    hard = probs.argmax(axis=2)
    truth = pd.Categorical(v0.model, categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any():
        raise ValueError("V0 contains labels outside the canonical order")
    row_ids = v0.id.astype(str).to_numpy()
    groups = v0.prompt_id.astype(str).to_numpy()
    families = [_model_family(members[sid]["model"]) for sid in ids]
    predicted = np.full(len(v0), -1, dtype=int)
    candidate_rows = np.full((len(v0), 3), -1, dtype=int)
    records = []
    with threadpool_limits(limits=1):
        for fit, held in GroupKFold(n_splits=folds).split(v0, truth, groups):
            pred, _, _, model = nested_fit(
                probs, hard, truth, groups, row_ids, ids, families, fit, held, limit, count,
                inner_folds=strategy["inner_folds"], c_grid=strategy["C_grid"])
            predicted[held] = pred
            roster = [ids.index(sid) for sid in model["checkpoint_ids"]]
            candidate_rows[held] = np.argsort(-probs[roster][:, held].mean(axis=0),
                                               axis=1, kind="stable")[:, :3]
            records.append(model)
        all_rows = np.arange(len(v0))
        deployment_pred, _, _, deployment = nested_fit(
            probs, hard, truth, groups, row_ids, ids, families, all_rows, all_rows, limit, count,
            inner_folds=strategy["inner_folds"], c_grid=strategy["C_grid"])
    if (predicted < 0).any() or (candidate_rows < 0).any():
        raise ValueError("E12 folds did not cover every validation row")
    deployment.pop("held_ids")
    deployment["fit_accuracy"] = float(accuracy_score(truth, deployment_pred))
    deployment["fit_balanced_accuracy"] = float(balanced_accuracy_score(truth, deployment_pred))
    top2 = (candidate_rows[:, :2] == truth[:, None]).any(axis=1)
    top3 = (candidate_rows == truth[:, None]).any(axis=1)
    covered = top2 if count == 2 else top3
    cm = confusion_matrix(truth, predicted, labels=np.arange(len(LABELS)))
    per_label = {}
    for index, label in enumerate(LABELS):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum()) - tp
        fn = int(cm[index].sum()) - tp
        per_label[label] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "support": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }
    return CLASSES[predicted], {
        "folds": records, "deployment": deployment, "per_label": per_label,
        "candidate_coverage": float(covered.mean()),
        "top2_coverage": float(top2.mean()), "top3_coverage": float(top3.mean()),
        "arbitration_accuracy_given_covered": float((predicted[covered] == truth[covered]).mean()) if covered.any() else None,
        "evaluation_checkpoint_ids": sorted(
            {sid for model in records for sid in model["checkpoint_ids"]}, key=lambda sid: int(sid[1:])),
    }


def run(root=ROOT):
    root = Path(root).resolve()
    v0 = catalog.cascades._v0(root)
    _, members, probabilities, excluded, snapshot = catalog._load_inputs(root,v0)
    ids = sorted(probabilities,key=lambda sid:int(sid[1:]))
    if len(ids)<20:
        raise ValueError("At least 20 eligible cached checkpoints are required")
    labels = v0.model.astype(str).to_numpy()
    truth = pd.Categorical(labels,categories=LABELS).codes.astype(np.int64)
    row_ids, groups = v0.id.astype(str).to_numpy(), v0.prompt_id.astype(str).to_numpy()
    probs = np.stack([probabilities[sid] for sid in ids])
    hard = probs.argmax(axis=2)
    families = [_model_family(members[sid]["model"]) for sid in ids]
    refpath = root/"artifacts/analysis/ensemble_hypotheses_nov15/predictions.parquet"
    oraclepath = root/"artifacts/analysis/oracle04/row_predictions.parquet"
    ref, oracle = [pd.read_parquet(path) for path in (refpath,oraclepath)]
    for frame in (ref,oracle):
        for name, expected in (("id",row_ids),("prompt_id",groups),("y_true",labels)):
            if frame[name].astype(str).tolist()!=expected.tolist():
                raise ValueError(f"Reference {name} differs from V0")
    h09 = pd.Categorical(ref["pred::H09-EQUAL"],categories=LABELS).codes.astype(int)
    oracle04 = pd.Categorical(oracle["pred::combined_logistic_core"],categories=LABELS).codes.astype(int)
    if (h09<0).any() or (oracle04<0).any() or abs(balanced_accuracy_score(truth,h09)-0.655004)>2e-6:
        raise ValueError("Historical reference is invalid")
    splits = list(GroupKFold(n_splits=5).split(v0,truth,groups))
    predictions = pd.DataFrame({"id":row_ids,"prompt_id":groups,"y_true":labels,
                                "pred::H09-EQUAL":CLASSES[h09],"pred::ORACLE04":CLASSES[oracle04]})
    results = {}
    with threadpool_limits(limits=1):
        for name,(count,limit) in CONFIGS.items():
            out = np.full(len(v0),-1,dtype=int)
            top3 = np.full((len(v0),3),-1,dtype=int)
            pair_scores = np.full((len(v0),count),np.nan)
            fold_id = np.zeros(len(v0),dtype=int)
            fold_reports, models = [], []
            for number,(fit,held) in enumerate(splits,1):
                assert not np.any(fold_id[held]) and set(groups[fit]).isdisjoint(groups[held])
                pred, candidates, scores, model = nested_fit(probs,hard,truth,groups,row_ids,ids,families,fit,held,limit,count)
                out[held],fold_id[held] = pred,number
                pair_scores[held] = scores
                roster = [ids.index(sid) for sid in model["checkpoint_ids"]]
                top3[held] = np.argsort(-probs[roster][:,held].mean(axis=0),axis=1,kind="stable")[:,:3]
                fold = metrics(truth[held],pred,top3[held],count,h09[held])
                fold.update(fold=number,rows=len(held),checkpoint_ids=model["checkpoint_ids"],selected_C=model["selected_C"])
                fold_reports.append(fold); models.append(model)
                print(f"{name} fold {number}/5 complete",flush=True)
            assert np.all(fold_id>0) and np.all(out>=0) and np.all(top3>=0)
            result = metrics(truth,out,top3,count,h09)
            cm = confusion_matrix(truth,out,labels=np.arange(len(LABELS)))
            result["per_label"] = {label:{"correct":int(cm[i,i]),"support":int(cm[i].sum()),
                                          "recall":float(cm[i,i]/cm[i].sum()) if cm[i].sum() else 0.0}
                                   for i,label in enumerate(LABELS)}
            result.update(config={"candidates":count,"checkpoints":limit},folds=fold_reports,models=models)
            results[name] = result
            predictions[f"fold::{name}"] = fold_id
            predictions[f"pred::{name}"] = CLASSES[out]
            for rank in range(3): predictions[f"candidate{rank+1}::{name}"] = CLASSES[top3[:,rank]]
            for rank in range(count): predictions[f"pairwise_score{rank+1}::{name}"] = pair_scores[:,rank]
    best = max(CONFIGS,key=lambda name:(results[name]["accuracy"],results[name]["balanced_accuracy"],name))
    outdir = root/"artifacts/analysis/e12"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    outdir.mkdir(parents=True)
    predictions.to_parquet(outdir/"oof_predictions.parquet",index=False)
    best_pred = pd.Categorical(predictions[f"pred::{best}"],categories=LABELS).codes
    pd.DataFrame(confusion_matrix(truth,best_pred,labels=np.arange(len(LABELS))),index=LABELS,columns=LABELS).to_csv(outdir/"best_confusion_matrix.csv",index_label="true_label")
    manifest = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"trust_list":snapshot,
                "v0_sha256":catalog.cascades._sha256(root/"data/processed/dataset_v0.parquet"),
                "h09_sha256":catalog.cascades._sha256(refpath),"oracle04_sha256":catalog.cascades._sha256(oraclepath),
                "source_sha256":catalog.cascades._sha256(Path(__file__)),"labels":list(LABELS),"rows":len(v0),
                "candidate_checkpoints":members,"excluded":excluded,"configs":CONFIGS,"C_grid":list(C_GRID),"best":best,
                "reference":{"H09-EQUAL":{"accuracy":float(accuracy_score(truth,h09)),"balanced_accuracy":float(balanced_accuracy_score(truth,h09))},
                             "ORACLE04-combined-logistic":{"accuracy":float(accuracy_score(truth,oracle04)),"balanced_accuracy":float(balanced_accuracy_score(truth,oracle04))}},"results":results}
    (outdir/"results.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    write_report(root/"docs/e12.md",manifest,outdir.relative_to(root))
    print(f"Artifacts: {outdir}",flush=True)
    return manifest


def write_report(path, manifest, output):
    results = manifest["results"]
    lines = ["# E12 Error-Aware Specialist Arbitration","",
             "Trust List 2 "+manifest["trust_list"]["version"]+"; five prompt-group outer folds and three nested prompt-group folds. All 2,699 V0 rows are scored once per experiment. H09 EQUAL is the historical fixed-roster reference. ORACLE04 uses Trust List 2 v025 and its previous nested stacking protocol. V0 was used in checkpoint selection and ensemble development.","",
             "| Experiment | Accuracy | Balanced accuracy | Candidate coverage | Corrected vs H09 | Introduced vs H09 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        lines.append(f"| {name} | {r['accuracy']:.4%} | {r['balanced_accuracy']:.4%} | {r['candidate_coverage']:.4%} | {r['corrected']} | {r['introduced']} |")
    lines.extend(["","| Reference | Accuracy | Balanced accuracy |","|---|---:|---:|"])
    for name,r in manifest["reference"].items(): lines.append(f"| {name} | {r['accuracy']:.4%} | {r['balanced_accuracy']:.4%} |")
    lines.extend(["","| Experiment | Top-2 coverage | Top-3 coverage | Accuracy when candidate includes true label |","|---|---:|---:|---:|"])
    for name,r in results.items(): lines.append(f"| {name} | {r['top2_coverage']:.4%} | {r['top3_coverage']:.4%} | {r['arbitration_accuracy_given_covered']:.4%} |")
    lines.extend(["","A single shared logistic classifier scores candidate pairs using antisymmetric per-checkpoint differences and distinct-family agreement. Swapping the candidates reverses the predicted probability. Each pair in the fitting candidate set contributes when its true label is one of the two candidates. Precision and recall use one positive and one negative pseudocount. Directional confusion uses one pseudocount for each of the 18 predicted labels. The model uses scale-only standardization, no intercept, and an inner three-fold search over C = 0.01, 0.1, 1 and 10 by pooled pairwise log loss.","",
                  "## Fold results","","| Experiment | Fold | Accuracy | Balanced accuracy | Top-2 coverage | Top-3 coverage | Accuracy given coverage | Corrected | Introduced | Checkpoints | C |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"])
    for name,r in results.items():
        for f in r['folds']:
            lines.append(f"| {name} | {f['fold']} | {f['accuracy']:.4%} | {f['balanced_accuracy']:.4%} | {f['top2_coverage']:.4%} | {f['top3_coverage']:.4%} | {f['arbitration_accuracy_given_covered']:.4%} | {f['corrected']} | {f['introduced']} | {', '.join(f['checkpoint_ids'])} | {f['selected_C']} |")
    lines.extend(["","## Per-label recall","","| Label | E12-1 | E12-2 | E12-3 |","|---|---:|---:|---:|"])
    for label in LABELS:
        lines.append("| "+label+" | "+" | ".join(f"{results[name]['per_label'][label]['recall']:.4%}" for name in CONFIGS)+" |")
    lines.extend(["",f"Best configuration by accuracy, then balanced accuracy: **{manifest['best']}**. The [confusion matrix](../{output}/best_confusion_matrix.csv), fold models, selected checkpoints, cache hashes and row-level predictions are in {output}.",""])
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    run()
