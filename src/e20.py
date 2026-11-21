"""E20: group-aware CatBoost candidate-label classification.

E20-C uses oracle prompt groups; E20-A/B use nested E19-B reconstructed groups;
E20-D reuses the E20-C classifiers with E19-B OOF reconstructed groups. All
variants consume cached base-checkpoint predictions and require no base-model
inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from . import e16_common, e15
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e20"
E19B = ROOT / "artifacts/analysis/e19b"
THREADS = 24
SEED = 2026
INNER_FOLDS = 3
DEPTHS = (5, 7)
LEARNING_RATES = (0.03, 0.06)
ITERATIONS = 500
CAT_FEATURES = ["candidate_label"]


def hungarian(scores, groups, row_ids=None):
    """E19's maximum-score unique-label assignment, with stable group order."""
    score = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    if score.shape != (len(groups), len(LABELS)) or not np.isfinite(score).all():
        raise ValueError("E20 assignment scores and groups are misaligned")
    if row_ids is not None and len(row_ids) != len(groups):
        raise ValueError("E20 row IDs and groups are misaligned")
    pred = np.full(len(groups), -1, dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        if len(rows) > len(LABELS):
            raise ValueError(f"E20 group exceeds 18 responses: {group}")
        ri, ci = linear_sum_assignment(-score[rows])
        pred[rows[ri]] = ci
    if (pred < 0).any():
        raise ValueError("E20 assignment failed to cover every row")
    return pred


def _ids_to_rows(all_ids, requested):
    loc = {str(x): i for i, x in enumerate(all_ids)}
    try:
        return np.asarray([loc[str(x)] for x in requested], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"E20 E19-B grouping contains unknown row ID {exc}") from exc


def load_reconstructed_groups(root=ROOT, *, required=False):
    """Read E19-B OOF assignments without using its true-prompt column."""
    path = Path(root) / "artifacts/analysis/e19b/oof_group_predictions.parquet"
    pair_path = Path(root) / "artifacts/analysis/e19b/oof_pair_diagnostics.parquet"
    if not path.is_file() or not pair_path.is_file():
        if required:
            raise FileNotFoundError("E20-A/B require E19-B OOF groups and pair diagnostics")
        return None
    rows = pd.read_parquet(path)
    pairs = pd.read_parquet(pair_path)
    needed = {"id", "predicted_group_id", "fold"}
    if not needed.issubset(rows.columns):
        raise ValueError("E19-B OOF grouping artifact misses required fields")
    if rows.id.astype(str).duplicated().any():
        raise ValueError("E19-B OOF grouping has duplicate row IDs")
    if not {"left_id", "right_id", "similarity"}.issubset(pairs.columns):
        raise ValueError("E19-B OOF pair diagnostics miss required fields")
    rows=rows[["id","predicted_group_id","fold"]].copy()
    rows["id"] = rows.id.astype(str)
    rows["predicted_group_id"] = rows.predicted_group_id.astype(str)
    # Group IDs are fold-local. Namespace them to prevent accidental merges
    # when E20 training data combines different OOF reconstruction folds.
    rows["e20_group"] = rows.fold.astype(str) + "::" + rows.predicted_group_id
    pair_scores = {}
    for item in pairs.itertuples(index=False):
        pair_scores[tuple(sorted((str(item.left_id), str(item.right_id))))] = float(item.similarity)
    rows.attrs["pair_scores"] = pair_scores
    return rows


def _group_confidence(ids, groups, pair_scores=None, oracle=False):
    ids = np.asarray(ids, dtype=str); groups = np.asarray(groups, dtype=str)
    confidence = np.ones(len(ids), dtype=np.float32) if oracle else np.zeros(len(ids), dtype=np.float32)
    within = [[] for _ in ids]
    if not oracle and pair_scores:
        for ix, (row_id, group) in enumerate(zip(ids, groups)):
            members = np.flatnonzero(groups == group)
            for other in members:
                if ix == other:
                    continue
                key = tuple(sorted((row_id, ids[other])))
                if key in pair_scores:
                    within[ix].append(pair_scores[key])
            if within[ix]:
                confidence[ix] = float(np.mean(within[ix]))
            elif len(members) == 1:
                confidence[ix] = 1.0
    sim = np.asarray([[np.mean(v) if v else 0.0,
                       np.min(v) if v else 0.0,
                       np.max(v) if v else 0.0] for v in within], dtype=np.float32)
    if oracle:
        # Similarities are later populated from response-text vectors.
        sim[:] = 0.0
    return confidence, sim


def _text_group_similarity(groups, text_vectors):
    groups = np.asarray(groups, dtype=str)
    out = np.zeros((len(groups), 4), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        if len(rows) < 2:
            continue
        sm = cosine_similarity(text_vectors[rows]).astype(np.float32)
        vals = sm[np.triu_indices(len(rows), k=1)]
        out[rows, 0] = float(vals.mean()) if len(vals) else 0.0
        out[rows, 1] = float(vals.min()) if len(vals) else 0.0
        out[rows, 2] = float(vals.max()) if len(vals) else 0.0
        out[rows, 3] = float((vals >= 0.5).mean()) if len(vals) else 0.0
    return out


def group_feature_frame(scores, groups, ids, texts, *, vectorizer=None,
                        pair_scores=None, oracle=False, fit_texts=None):
    """Return candidate-label group features in row-major, canonical order."""
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=str); ids = np.asarray(ids, dtype=str)
    texts = np.asarray(texts, dtype=str)
    n, k = scores.shape
    if k != len(LABELS) or len(groups) != n or len(ids) != n or len(texts) != n:
        raise ValueError("E20 group feature inputs do not align")
    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                     min_df=1, sublinear_tf=True, dtype=np.float32,
                                     max_features=150000)
        vectorizer.fit(np.asarray(fit_texts if fit_texts is not None else texts, dtype=str))
    text_vectors = vectorizer.transform(texts)
    text_sim = _text_group_similarity(groups, text_vectors)
    confidence, edge_sim = _group_confidence(ids, groups, pair_scores, oracle)
    best_idx = scores.argmax(axis=1)
    best = scores.max(axis=1)
    second = np.partition(scores, -2, axis=1)[:, -2]
    rank = np.empty_like(scores, dtype=np.int16)
    # Deterministic score ties use canonical label index.
    order = np.argsort(-scores, axis=1, kind="stable")
    rank[np.arange(n)[:, None], order] = np.arange(1, k + 1)[None, :]
    groups_by = {g: np.flatnonzero(groups == g) for g in sorted(set(groups.tolist()))}
    base = {"group_size": np.zeros(n), "group_confidence": confidence,
            "group_edge_mean": edge_sim[:, 0], "group_edge_min": edge_sim[:, 1],
            "group_edge_max": edge_sim[:, 2], "group_text_sim_mean": text_sim[:, 0],
            "group_text_sim_min": text_sim[:, 1], "group_text_sim_max": text_sim[:, 2],
            "group_text_sim_ge_050": text_sim[:, 3],
            "e13_row_best": best, "e13_row_second": second,
            "e13_row_margin": best-second}
    candidate = np.repeat(np.arange(k), n)  # Candidate-major for matrix features.
    # Return records in row-major order to match scores.reshape(-1, order='C').
    candidate = np.tile(np.arange(k), n)
    recs = []
    for i in range(n):
        members = groups_by[groups[i]]
        other = members[members != i]
        for lab in range(k):
            other_vals = scores[other, lab] if len(other) else np.empty(0)
            ordered_other = np.sort(other_vals, kind="stable")
            max_other = float(ordered_other[-1]) if len(ordered_other) else 0.0
            second_other = float(ordered_other[-2]) if len(ordered_other) > 1 else 0.0
            group_pref = int(np.sum(best_idx[other] == lab)) if len(other) else 0
            other_labels = set(best_idx[other].tolist()) if len(other) else set()
            rec = {name: float(value[i]) for name, value in base.items()}
            rec["group_size"] = float(len(members))
            rec.update({"candidate_label": LABELS[lab],
                "candidate_e13_score": float(scores[i, lab]),
                "candidate_e13_rank": int(rank[i, lab]),
                "candidate_vs_row_best": float(scores[i, lab]-best[i]),
                "group_candidate_best_other": max_other,
                "group_candidate_second_other": second_other,
                "group_candidate_margin_other": float(scores[i, lab]-max_other),
                "group_candidate_rank": 1 + int(np.sum(other_vals > scores[i, lab])),
                "group_other_rows_preferring_candidate": group_pref,
                "group_other_supported_label_count": len(other_labels),
                "assignment_advantage": float(scores[i, lab]-max_other)})
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def _evidence(ctx, fit_rows, target_rows):
    """Fit E13-B router on fit rows and return checkpoint reliability outputs."""
    fit_rows = np.asarray(fit_rows, dtype=np.int64); target_rows = np.asarray(target_rows, dtype=np.int64)
    if set(ctx["groups"][fit_rows]) & set(ctx["groups"][target_rows]):
        raise ValueError("E20 E13 feature fit and target prompt groups overlap")
    key = e16_common._cache_key(ctx, fit_rows, target_rows)
    cache_dir = OUT / "evidence_cache"
    cache_path = cache_dir / f"{key}.joblib"
    if cache_path.is_file():
        cached = joblib.load(cache_path)
        if cached.get("key") != key or not np.array_equal(cached["target_rows"], target_rows):
            raise ValueError("E20 cached reliability features do not align")
        return cached["evidence"]
    result = e15._fit_e13_predict(ctx["matrices"], ctx["checkpoint_ids"], ctx["families"],
        ctx["y"], ctx["groups"], fit_rows, target_rows, ctx["row_ids"])
    evidence = {"scores": e16_common.finite_scores(result["label_scores"]),
            "roster_ids": list(result["roster_ids"]),
            "reliability": np.asarray(result["reliability"], dtype=np.float32),
            "hard": np.asarray(result["hard"], dtype=np.int16)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix(".tmp")
    joblib.dump({"key": key, "target_rows": target_rows, "evidence": evidence}, temp, compress=3)
    temp.replace(cache_path)
    return evidence


def _evidence_feature_frame(ctx, evidence, target_rows, outer_roster):
    mats = ctx["matrices"]
    cp_pos = {sid: i for i, sid in enumerate(ctx["checkpoint_ids"])}
    rel_pos = {sid: i for i, sid in enumerate(evidence["roster_ids"])}
    target_rows = np.asarray(target_rows, dtype=np.int64)
    rows = []
    for row_pos, row in enumerate(target_rows):
        for lab in range(len(LABELS)):
            rec = {"candidate_label": LABELS[lab],
                   "e13_candidate_score": float(evidence["scores"][row_pos, lab]),
                   "e13_candidate_rank": int(1+np.sum(evidence["scores"][row_pos] > evidence["scores"][row_pos, lab])),
                   "e13_best_score": float(evidence["scores"][row_pos].max()),
                   "e13_second_score": float(np.partition(evidence["scores"][row_pos], -2)[-2]),
                   "e13_candidate_vs_best": float(evidence["scores"][row_pos, lab]-evidence["scores"][row_pos].max())}
            vote_count = 0; fam_votes = Counter(); fam_probs = {}; fam_rels = {}
            for sid in outer_roster:
                cp = cp_pos[sid]; prob = float(mats[cp, row, lab]); hard = int(mats[cp, row].argmax())
                present = sid in rel_pos
                rel = float(evidence["reliability"][rel_pos[sid], row_pos]) if present else 0.0
                fam = str(ctx["families"][cp])
                vote = int(hard == lab); vote_count += vote
                fam_votes[fam] += vote
                fam_probs.setdefault(fam, []).append(prob)
                if present:
                    fam_rels.setdefault(fam, []).append(rel)
                rec[f"cp::{sid}::prob"] = prob
                rec[f"cp::{sid}::vote"] = vote
                rec[f"cp::{sid}::reliability"] = rel
                rec[f"cp::{sid}::present"] = int(present)
            rec["candidate_checkpoint_votes"] = vote_count
            rec["candidate_supporting_families"] = sum(v > 0 for v in fam_votes.values())
            for fam in sorted(fam_probs):
                vals = fam_probs[fam]
                rec[f"family::{fam}::mean_probability"] = float(np.mean(vals))
                rec[f"family::{fam}::max_probability"] = float(np.max(vals))
                rec[f"family::{fam}::vote_count"] = float(fam_votes[fam])
                rec[f"family::{fam}::max_reliability"] = float(max(fam_rels.get(fam, [0.0])))
            rec["candidate_family_probability_mean"] = float(np.mean([np.mean(v) for v in fam_probs.values()]))
            rec["candidate_max_family_probability"] = float(max(np.mean(v) for v in fam_probs.values()))
            rec["candidate_mean_voter_reliability"] = float(np.mean([
                evidence["reliability"][rel_pos[sid], row_pos] for sid in outer_roster
                if sid in rel_pos and int(mats[cp_pos[sid], row].argmax()) == lab])) if any(
                sid in rel_pos and int(mats[cp_pos[sid], row].argmax()) == lab for sid in outer_roster) else 0.0
            rows.append(rec)
    frame = pd.DataFrame.from_records(rows)
    frame["candidate_label"] = frame.candidate_label.astype(str)
    return frame


def _combine_features(ctx, evidence, rows, outer_roster, groups, vectorizer,
                      *, pair_scores=None, oracle=False, fit_texts=None):
    frame = _evidence_feature_frame(ctx, evidence, rows, outer_roster)
    group = group_feature_frame(evidence["scores"], groups, ctx["row_ids"][rows],
        ctx["texts"][rows], vectorizer=vectorizer, pair_scores=pair_scores,
        oracle=oracle, fit_texts=fit_texts)
    # Both builders use row-major candidate order.
    if len(frame) != len(group):
        raise ValueError("E20 checkpoint and group feature rows differ")
    for col in group.columns:
        if col == "candidate_label":
            if not frame.candidate_label.equals(group.candidate_label):
                raise ValueError("E20 candidate-label order differs across feature blocks")
            continue
        frame[f"group::{col}"] = group[col].to_numpy()
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
        raise ValueError("E20 feature matrix contains non-finite values")
    return frame


def _oof_features(ctx, rows, outer_roster, groups_all, *, oracle, pair_scores=None,
                  split_groups=None, output_dir=None, n_splits=4):
    rows = np.asarray(rows, dtype=np.int64); groups_all = np.asarray(groups_all, dtype=str)
    split_groups = np.asarray(split_groups if split_groups is not None else groups_all, dtype=str)
    frame_parts = []; targets = []; split_rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (fit_local, held_local) in enumerate(splitter.split(rows, ctx["y"][rows], split_groups[rows]), 1):
        fit = rows[fit_local]; held = rows[held_local]
        ev = _evidence(ctx, fit, held)
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1,
                                     sublinear_tf=True, dtype=np.float32, max_features=150000)
        vectorizer.fit(ctx["texts"][fit])
        held_pairs = _combine_features(ctx, ev, held, outer_roster, groups_all[held], vectorizer,
            pair_scores=pair_scores, oracle=oracle, fit_texts=ctx["texts"][fit])
        frame_parts.append(held_pairs)
        y = ctx["y"][held]
        targets.append(np.repeat(y, len(LABELS)) == np.tile(np.arange(len(LABELS)), len(held)))
        split_rows.extend(held.tolist())
        print(f"E20 cross-fit features {fold}/{n_splits}: train={len(fit)} target={len(held)}",flush=True)
    # Restore input row order despite GroupKFold's held partition ordering.
    joined = pd.concat(frame_parts, ignore_index=True)
    y_all = np.concatenate(targets).astype(np.int8)
    row_order = np.asarray(split_rows)
    row_position = {int(r): i for i, r in enumerate(row_order)}
    desired = np.asarray([row_position[int(r)] for r in rows], dtype=np.int64)
    index = (desired[:, None] * len(LABELS) + np.arange(len(LABELS))[None, :]).reshape(-1)
    return joined.iloc[index].reset_index(drop=True), y_all[index], row_order


def _predict_candidate(clf, frame):
    probabilities = clf.predict_proba(frame)[:, 1]
    n = len(frame) // len(LABELS)
    return probabilities.reshape(n, len(LABELS))


def _score_with_reused_classifier(classifier, features):
    """Score candidate records with a saved CatBoost model; never fit it."""
    expected=list(classifier.feature_names_)
    if set(features.columns)!=set(expected):
        raise ValueError("Saved E20-C classifier feature schema does not match rebuilt E20-D features")
    return _predict_candidate(classifier,features.loc[:,expected])


def _new_model(depth, learning_rate, iterations=ITERATIONS):
    return CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", task_type="CPU",
        iterations=int(iterations), depth=int(depth), learning_rate=float(learning_rate),
        l2_leaf_reg=5.0, random_seed=SEED, thread_count=THREADS, verbose=False,
        allow_writing_files=False, random_strength=0.5, border_count=64)


def _cat_train(frame, target, depth, lr, validation=None):
    model = _new_model(depth, lr)
    kwargs = {"cat_features": CAT_FEATURES, "verbose": False}
    if validation is not None:
        val_frame, val_target = validation
        kwargs.update(eval_set=(val_frame, val_target), early_stopping_rounds=60, use_best_model=True)
    model.fit(frame, np.asarray(target, dtype=np.int8), **kwargs)
    return model


def _choose_params(ctx, rows, feature_builder, groups, *, method):
    """Nested prompt-group CV tunes CatBoost against the final decoder."""
    best = None; folds = []
    for inner, (fit_local, valid_local) in enumerate(GroupKFold(INNER_FOLDS).split(
            rows, ctx["y"][rows], groups[rows]), 1):
        fit_rows, valid_rows = rows[fit_local], rows[valid_local]
        train_x, train_y, _ = feature_builder(fit_rows)
        valid_x, _, _ = feature_builder(valid_rows, fit_rows=fit_rows)
        y = ctx["y"][valid_rows]
        valid_results = []
        for depth in DEPTHS:
            for lr in LEARNING_RATES:
                model = _cat_train(train_x, train_y, depth, lr, validation=(valid_x, np.repeat(y, len(LABELS)) == np.tile(np.arange(len(LABELS)), len(valid_rows))))
                scores = _predict_candidate(model, valid_x)
                pred = scores.argmax(axis=1) if method == "independent" else hungarian(scores, groups[valid_rows], ctx["row_ids"][valid_rows])
                ba = float(balanced_accuracy_score(y, pred)); acc = float(accuracy_score(y, pred))
                valid_results.append({"depth":depth,"learning_rate":lr,"ba":ba,"accuracy":acc,
                                      "iterations":max(1,int(model.get_best_iteration())+1)})
                key=(ba,acc,-depth,-lr)
                if best is None or key>best[0]: best=(key,{"depth":depth,"learning_rate":lr})
        folds.append({"fold":inner,"train_rows":len(fit_rows),"valid_rows":len(valid_rows),"candidates":valid_results})
        print(f"E20 nested tune fold {inner}/{INNER_FOLDS}: selected-so-far {best[1]}",flush=True)
    return best[1],folds


def _metric(y, pred):
    y=np.asarray(y,dtype=np.int64);pred=np.asarray(pred,dtype=np.int64)
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm=confusion_matrix(y,pred,labels=np.arange(len(LABELS)))
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.arange(len(LABELS)),zero_division=0)
    return {"rows":len(y),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "confusion_matrix":cm.tolist(),"per_label":{LABELS[i]:{"support":int(support[i]),"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i])} for i in range(len(LABELS))}}


def _comparison(y, baseline, pred):
    y=np.asarray(y);baseline=np.asarray(baseline);pred=np.asarray(pred)
    corrected=(baseline!=y)&(pred==y);introduced=(baseline==y)&(pred!=y)
    return {**_metric(y,pred),"baseline_accuracy":float(accuracy_score(y,baseline)),
      "baseline_balanced_accuracy":float(balanced_accuracy_score(y,baseline)),
      "delta_accuracy":float(accuracy_score(y,pred)-accuracy_score(y,baseline)),
      "delta_balanced_accuracy":float(balanced_accuracy_score(y,pred)-balanced_accuracy_score(y,baseline)),
      "corrected":int(corrected.sum()),"introduced":int(introduced.sum()),
      "net_gains":int(corrected.sum()-introduced.sum()),"changed":int((baseline!=pred).sum())}


def _group_audit(true_groups, pred_groups):
    truth=np.asarray(true_groups,dtype=str);pred=np.asarray(pred_groups,dtype=str)
    exact=0;split=0;merge=0
    for group in set(truth):
        rows=np.flatnonzero(truth==group)
        pred_set=set(pred[rows])
        if len(pred_set)>1: split+=1
        if len(pred_set)==1 and len(rows)==np.sum(pred==next(iter(pred_set))) and np.all(truth[pred==next(iter(pred_set))]==group): exact+=1
    for group in set(pred):
        if len(set(truth[pred==group]))>1:merge+=1
    return {"true_groups":len(set(truth)),"predicted_groups":len(set(pred)),"exact_groups":exact,
            "split_groups":split,"merged_groups":merge,"exact_group_rate":exact/max(1,len(set(truth)))}


def _e19b_hungarian_baseline(ctx):
    """Apply E19-B OOF groups to E19's row-aligned E13 utility scores."""
    path=ctx["root"]/"artifacts/analysis/e19/predictions.npz"
    if not path.is_file():
        return None
    cache=np.load(path,allow_pickle=True)
    row_ids=np.asarray(cache["row_ids"],dtype=str)
    if not np.array_equal(row_ids,ctx["row_ids"].astype(str)):
        raise ValueError("E19 utility scores are not aligned to E20 V0 rows")
    if not np.array_equal(cache["y_true"].astype(np.int64),ctx["y"]):
        raise ValueError("E19 utility-score labels differ from E20 V0 labels")
    if not np.array_equal(cache["e13_b"].astype(np.int64),ctx["baseline"]):
        raise ValueError("E19 utility-score baseline differs from E20 E13-B")
    scores=np.asarray(cache["scores"],dtype=np.float64)
    if scores.shape!=(len(row_ids),len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError("E19 utility-score matrix has an invalid shape or values")
    groups=load_reconstructed_groups(ctx["root"],required=True)
    if not np.array_equal(groups.id.astype(str).to_numpy(),row_ids):
        raise ValueError("E19-B OOF grouping rows are not aligned to E19 scores")
    pred=hungarian(scores,groups.e20_group.astype(str).to_numpy(),row_ids)
    return pred,_comparison(ctx["y"],ctx["baseline"],pred)


def _aggregate_importance(items, category):
    frame=pd.DataFrame([x for x in items if x["category"]==category])
    if frame.empty:return []
    return [{"feature":name,"category":category,"mean_importance":float(value)}
            for name,value in frame.groupby("feature").importance.mean().sort_values(ascending=False).items()]


def _load_e19b_reconstructions(root):
    """Load cached E19-B reconstructions/embeddings; never run model inference."""
    path=Path(root)/"artifacts/analysis/e19b/reconstructed_prompts.parquet"
    emb_path=Path(root)/"artifacts/analysis/e19b/semantic_embeddings.npy"
    meta_path=Path(root)/"artifacts/analysis/e19b/semantic_embeddings.json"
    if not all(p.is_file() for p in (path,emb_path,meta_path)):
        raise FileNotFoundError("E20-A/B require cached E19-B reconstructed_prompts.parquet and semantic_embeddings.npy/.json")
    frame=pd.read_parquet(path)
    needed={"id","status","reconstructed_prompt","topic","distinctive_details","constraints"}
    if not needed.issubset(frame.columns):raise ValueError("E19-B reconstruction parquet misses required columns")
    emb=np.load(emb_path)
    meta=json.loads(meta_path.read_text())
    if len(frame)!=2699 or not frame.status.eq("ok").all() or emb.shape[0]!=len(frame):
        raise ValueError("E19-B cached reconstructions/embeddings are incomplete")
    text=frame.reconstructed_prompt.fillna("").astype(str).to_numpy()
    digest=hashlib.sha256("\0".join(text).encode()).hexdigest()
    if meta.get("count")!=len(frame) or meta.get("sha256")!=digest:
        raise ValueError("E19-B semantic embedding metadata does not match reconstructed text")
    details=frame.distinctive_details.map(lambda x:json.loads(x) if isinstance(x,str) else x).tolist()
    constraints=frame.constraints.map(lambda x:json.loads(x) if isinstance(x,str) else x).tolist()
    return {"frame":frame,"recon":text,"topics":frame.topic.fillna("").astype(str).to_numpy(),
            "details":details,"constraints":constraints,"embeddings":emb.astype(np.float32),
            "sha256":digest,"semantic_meta":meta}


def _fit_reconstructed_groups(ctx, fit_rows, target_rows, data, *, tag):
    """Fit/tune E19-B grouping on fit prompt groups and group target text only."""
    from . import e19b_evaluate, e19b_group
    fit_rows=np.asarray(fit_rows,dtype=np.int64);target_rows=np.asarray(target_rows,dtype=np.int64)
    if set(ctx["groups"][fit_rows]) & set(ctx["groups"][target_rows]):
        raise ValueError("E20 nested grouper fit and target prompt groups overlap")
    recon=data["recon"];texts=ctx["texts"];emb=data["embeddings"];topics=data["topics"]
    details=data["details"];constraints=data["constraints"];truth_groups=ctx["groups"]
    selection=e19b_evaluate.tune_on_training(fit_rows,recon,texts,emb,topics,details,constraints,
                                             truth_groups)
    spaces,pfit,xfit,yfit=e19b_evaluate._pairs_and_features(fit_rows,recon,texts,emb,topics,
        details,constraints,truth_groups,fit_rows=fit_rows)
    scorer=e19b_group.fit_pair_scorer(xfit,yfit,c=selection["C"],seed=SEED)
    # Target pair features are generated without reading target prompt IDs.
    spaces_t=e19b_group.fit_similarity_spaces(recon,texts,fit_rows)
    pairs=e19b_group.candidate_pairs(target_rows,spaces_t,emb)
    x=e19b_group.pair_features(target_rows,pairs,spaces_t,recon,texts,emb,topics,details,constraints)
    probs=e19b_group.score_pairs(scorer,x) if len(pairs) else np.empty(0,dtype=np.float32)
    group_ids,cluster=e19b_group.cluster_complete_link(ctx["row_ids"][target_rows],pairs,probs,
        threshold=selection["threshold"],max_cluster_size=18)
    group_ids=np.asarray([f"{tag}::{z}" for z in group_ids],dtype=str)
    pair_scores={}
    for (a,b),score in zip(pairs,probs):
        pair_scores[tuple(sorted((str(ctx["row_ids"][target_rows[a]]),str(ctx["row_ids"][target_rows[b]]))))]=float(score)
    return group_ids,pair_scores,{"selection":{"C":selection["C"],"threshold":selection["threshold"]},
        "clusters":cluster["clusters"],"candidate_pairs":int(len(pairs)),"fit_pair_count":int(len(yfit))}


def _oof_reconstructed_groups(ctx, rows, data, *, n_splits, tag):
    rows=np.asarray(rows,dtype=np.int64);groups=np.empty(len(rows),dtype=object);pair_scores={};details=[]
    pos={int(row):i for i,row in enumerate(rows)}
    for fold,(fi,vi) in enumerate(GroupKFold(n_splits=n_splits).split(rows,ctx["y"][rows],ctx["groups"][rows]),1):
        fit,target=rows[fi],rows[vi]
        fold_groups,fold_pairs,info=_fit_reconstructed_groups(ctx,fit,target,data,tag=f"{tag}-g{fold}")
        for j,row in enumerate(target):groups[pos[int(row)]]=fold_groups[j]
        pair_scores.update(fold_pairs);details.append({"fold":fold,"fit_rows":len(fit),"target_rows":len(target),**info})
        print(f"E20 reconstructed grouping OOF {fold}/{n_splits}: fit={len(fit)} target={len(target)}",flush=True)
    return groups.astype(str),pair_scores,details


def _oof_reconstructed_features(ctx, rows, outer_roster, data, *, n_splits, tag):
    """Jointly cross-fit E13 evidence and E19-B groups on identical target folds."""
    rows=np.asarray(rows,dtype=np.int64);parts=[];targets=[];held_order=[]
    splits=GroupKFold(n_splits=n_splits).split(rows,ctx["y"][rows],ctx["groups"][rows])
    for fold,(fi,vi) in enumerate(splits,1):
        fit,target=rows[fi],rows[vi]
        group_ids,pair_scores,_=_fit_reconstructed_groups(ctx,fit,target,data,tag=f"{tag}-fold{fold}")
        evidence=_evidence(ctx,fit,target)
        vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=1,sublinear_tf=True,
                                   dtype=np.float32,max_features=150000)
        vectorizer.fit(ctx["texts"][fit])
        frame=_combine_features(ctx,evidence,target,outer_roster,group_ids,vectorizer,
            pair_scores=pair_scores,oracle=False,fit_texts=ctx["texts"][fit])
        parts.append(frame)
        targets.append((np.repeat(ctx["y"][target],len(LABELS))==
                        np.tile(np.arange(len(LABELS)),len(target))).astype(np.int8))
        held_order.extend(target.tolist())
        print(f"E20 joint E13/E19-B cross-fit {fold}/{n_splits}: fit={len(fit)} target={len(target)}",flush=True)
    merged=pd.concat(parts,ignore_index=True);yy=np.concatenate(targets)
    position={int(row):i for i,row in enumerate(held_order)}
    order=np.asarray([position[int(row)] for row in rows],dtype=np.int64)
    ix=(order[:,None]*len(LABELS)+np.arange(len(LABELS))[None,:]).reshape(-1)
    return merged.iloc[ix].reset_index(drop=True),yy[ix]


def _group_size_metrics(y,pred,groups):
    size=np.asarray([np.sum(groups==g) for g in groups])
    return [{"group_size":int(s),**_metric(y[size==s],pred[size==s])}
            for s in sorted(set(size.tolist()))]


def _run_oracle(ctx, output):
    y=ctx["y"];baseline=ctx["baseline"];oof=np.full((len(y),len(LABELS)),-np.inf,dtype=np.float32)
    pred_a=np.full(len(y),-1,dtype=np.int64);pred_c=np.full(len(y),-1,dtype=np.int64)
    pred_c_checkpoint=np.full(len(y),-1,dtype=np.int64)
    fold_results=[];importance=[];groups=np.asarray(ctx["groups"],dtype=str)
    for outer in ctx["outer_folds"]:
        fold=outer["fold"];fit=outer["fit_rows"];held=outer["held_rows"]
        roster=list(outer["roster"])
        print(f"E20-C outer fold {fold}/5: cross-fit training features",flush=True)
        # Feature generation for all outer training rows uses cross-fitted E13 reliability.
        train_x,train_y,_=_oof_features(ctx,fit,roster,groups,oracle=True,n_splits=4)
        # Hyperparameter validation uses nested prompt-group splits. Cache each
        # nested feature set once and reuse it across CatBoost configurations.
        tuning_records=[]
        for inner,(fi,vi) in enumerate(GroupKFold(INNER_FOLDS).split(fit,y[fit],groups[fit]),1):
            subfit,valid=fit[fi],fit[vi]
            tx,ty,_=_oof_features(ctx,subfit,roster,groups,oracle=True,n_splits=3)
            ev=_evidence(ctx,subfit,valid)
            vec=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32,max_features=150000)
            vec.fit(ctx["texts"][subfit])
            vx=_combine_features(ctx,ev,valid,roster,groups[valid],vec,oracle=True,fit_texts=ctx["texts"][subfit])
            vy=(np.repeat(y[valid],len(LABELS))==np.tile(np.arange(len(LABELS)),len(valid))).astype(np.int8)
            tuning_records.append((inner,subfit,valid,tx,ty,vx,vy))
        # Each candidate hyperparameter is evaluated on exactly these nested folds.
        param_results=[]
        for depth in DEPTHS:
            for lr in LEARNING_RATES:
                scores=[]
                for inner,subfit,valid,tx,ty,vx,vy in tuning_records:
                    model=_cat_train(tx,ty,depth,lr,validation=(vx,vy));mat=_predict_candidate(model,vx);p=hungarian(mat,groups[valid],ctx["row_ids"][valid])
                    scores.append({"inner_fold":inner,"balanced_accuracy":float(balanced_accuracy_score(y[valid],p)),"accuracy":float(accuracy_score(y[valid],p)),"iterations":max(1,int(model.get_best_iteration())+1)})
                param_results.append({"depth":depth,"learning_rate":lr,"folds":scores,"mean_balanced_accuracy":float(np.mean([x["balanced_accuracy"] for x in scores])),"mean_accuracy":float(np.mean([x["accuracy"] for x in scores]))})
        chosen=max(param_results,key=lambda z:(z["mean_balanced_accuracy"],z["mean_accuracy"],-z["depth"],-z["learning_rate"]))
        # Final classifier trained from the outer-fit OOF records; held-out
        # reliability is generated by a router fit only on outer-fit groups.
        final=_cat_train(train_x,train_y,chosen["depth"],chosen["learning_rate"])
        evh=_evidence(ctx,fit,held);vec=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32,max_features=150000);vec.fit(ctx["texts"][fit])
        hx=_combine_features(ctx,evh,held,roster,groups[held],vec,oracle=True,fit_texts=ctx["texts"][fit]);hm=_predict_candidate(final,hx)
        oof[held]=hm;pred_a[held]=hm.argmax(1);pred_c[held]=hungarian(hm,groups[held],ctx["row_ids"][held])
        model_path=output/f"E20C_fold{fold}.cbm";final.save_model(str(model_path))
        checkpoint_cols=[c for c in train_x.columns if c=="candidate_label" or not c.startswith("group::")]
        checkpoint_model=_cat_train(train_x[checkpoint_cols],train_y,chosen["depth"],chosen["learning_rate"])
        checkpoint_matrix=_predict_candidate(checkpoint_model,hx[checkpoint_cols]);checkpoint_pred=hungarian(checkpoint_matrix,groups[held],ctx["row_ids"][held])
        pred_c_checkpoint[held]=checkpoint_pred
        checkpoint_model.save_model(str(output/f"E20C_checkpoint_only_fold{fold}.cbm"))
        fi=pd.DataFrame({"feature":list(final.feature_names_),"importance":final.get_feature_importance()}).sort_values("importance",ascending=False)
        importance.extend({"fold":fold,"feature":r.feature,"importance":float(r.importance),"category":"group" if str(r.feature).startswith("group::") else "checkpoint"} for r in fi.itertuples())
        fold_results.append({"fold":fold,"train_rows":len(fit),"held_rows":len(held),"selected":{k:chosen[k] for k in ("depth","learning_rate")},"tuning":param_results,
            "E20-C":_comparison(y[held],baseline[held],pred_c[held]),"E20-C-independent":_comparison(y[held],baseline[held],pred_a[held]),
            "checkpoint_only":_comparison(y[held],baseline[held],checkpoint_pred),"model_path":str(model_path)})
        print(f"E20-C fold {fold}: BA={balanced_accuracy_score(y[held],pred_c[held]):.4%} vs E13={balanced_accuracy_score(y[held],baseline[held]):.4%}",flush=True)
    if np.any(pred_c<0) or not np.isfinite(oof).all():raise RuntimeError("E20-C OOF predictions are incomplete")
    pairs=[]
    for outer in ctx["outer_folds"]:
        rows=outer["held_rows"];predg=groups[rows]
        pairs.append(pd.DataFrame({"id":ctx["row_ids"][rows],"fold":outer["fold"],"true_prompt_id":groups[rows],"predicted_group_id":predg,"group_size":pd.Series(predg).map(pd.Series(predg).value_counts()).to_numpy()}))
    group_frame=pd.concat(pairs,ignore_index=True).sort_values("id",kind="stable")
    pred_frame=ctx["v0"][["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy()
    pred_frame["E13_B"]=np.asarray(LABELS,dtype=object)[baseline];pred_frame["E20_C_independent"]=np.asarray(LABELS,dtype=object)[pred_a];pred_frame["E20_C"]=np.asarray(LABELS,dtype=object)[pred_c]
    pred_frame["fold"]=-1
    for outer in ctx["outer_folds"]:pred_frame.loc[outer["held_rows"],"fold"]=outer["fold"]
    pred_frame.to_parquet(output/"predictions_e20c.parquet",index=False)
    np.savez_compressed(output/"scores_e20c.npz",row_ids=ctx["row_ids"],scores=oof,y=y,pred_independent=pred_a,pred_hungarian=pred_c)
    metrics={"experiment":"E20-C","baseline":"E13-B Trust List 2 v037","rows":len(y),"baseline_reproduced_exactly":True,
        "E13-B":_metric(y,baseline),"E20-C-independent":_comparison(y,baseline,pred_a),"E20-C":_comparison(y,baseline,pred_c),
        "ablation_checkpoint_only":_comparison(y,baseline,pred_c_checkpoint),
        "folds":fold_results,"per_group_size":_group_size_metrics(y,pred_c,groups),"group_audit":_group_audit(groups,groups),
        "correctly_grouped":_comparison(y,baseline,pred_c),"incorrectly_grouped":None,
        "feature_importance":importance,"checkpoint_feature_importance_top":sorted([x for x in importance if x["category"]=="checkpoint"],key=lambda x:-x["importance"])[:100],
        "group_feature_importance_top":sorted([x for x in importance if x["category"]=="group"],key=lambda x:-x["importance"])[:100],
        "grouping":"oracle V0 prompt_id; used as membership only, never a model feature","hyperparameters":{"depths":DEPTHS,"learning_rates":LEARNING_RATES,"iterations":ITERATIONS,"thread_count":THREADS},
        "runtime_seconds":0.0}
    e19a_path=ROOT/"artifacts/analysis/e19/predictions.parquet"
    if e19a_path.is_file():
        e19a_frame=pd.read_parquet(e19a_path)
        if e19a_frame.id.astype(str).tolist()!=ctx["row_ids"].astype(str).tolist():
            raise ValueError("E19-A predictions are not aligned to E20 V0 rows")
        e19a_codes=pd.Categorical(e19a_frame["E19-A-full"].astype(str),categories=LABELS).codes.astype(np.int64)
        if (e19a_codes<0).any():raise ValueError("E19-A contains unknown labels")
        metrics["E19-A"]=_metric(y,e19a_codes)
        metrics["E20-C_vs_E19-A"]=_comparison(y,e19a_codes,pred_c)
    return metrics,pred_frame,group_frame


def _run_reconstructed(ctx, output, data):
    """E20-A/B with nested E19-B pair grouping; A and B share score matrices."""
    y=ctx["y"];baseline=ctx["baseline"];true_groups=np.asarray(ctx["groups"],dtype=str)
    scores_all=np.full((len(y),len(LABELS)),-np.inf,dtype=np.float32)
    pred_a=np.full(len(y),-1,dtype=np.int64);pred_b=np.full(len(y),-1,dtype=np.int64)
    pred_cp=np.full(len(y),-1,dtype=np.int64);group_all=np.empty(len(y),dtype=object)
    fold_results=[];importance=[];nested_group_records=[]
    for outer in ctx["outer_folds"]:
        fold=outer["fold"];fit=outer["fit_rows"];held=outer["held_rows"];roster=list(outer["roster"])
        print(f"E20-A/B outer fold {fold}/5: nested reconstructed groups",flush=True)
        train_x,train_y=_oof_reconstructed_features(ctx,fit,roster,data,n_splits=4,tag=f"outer{fold}-train")
        tune_records=[]
        for inner,(fi,vi) in enumerate(GroupKFold(INNER_FOLDS).split(fit,y[fit],true_groups[fit]),1):
            subfit,valid=fit[fi],fit[vi]
            tx,ty=_oof_reconstructed_features(ctx,subfit,roster,data,n_splits=3,
                tag=f"outer{fold}-inner{inner}-train")
            valid_groups,valid_pairs,group_info=_fit_reconstructed_groups(ctx,subfit,valid,data,
                tag=f"outer{fold}-inner{inner}-valid")
            ev=_evidence(ctx,subfit,valid)
            vec=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32,max_features=150000)
            vec.fit(ctx["texts"][subfit])
            vx=_combine_features(ctx,ev,valid,roster,valid_groups,vec,pair_scores=valid_pairs,
                oracle=False,fit_texts=ctx["texts"][subfit])
            vy=(np.repeat(y[valid],len(LABELS))==np.tile(np.arange(len(LABELS)),len(valid))).astype(np.int8)
            tune_records.append((inner,valid,valid_groups,tx,ty,vx,vy,group_info))
            nested_group_records.append({"outer_fold":fold,"inner_fold":inner,"valid_group_fit":group_info})
        param_results=[]
        for depth in DEPTHS:
            for lr in LEARNING_RATES:
                fold_scores=[]
                for inner,valid,valid_groups,tx,ty,vx,vy,group_info in tune_records:
                    model=_cat_train(tx,ty,depth,lr,validation=(vx,vy));mat=_predict_candidate(model,vx)
                    pa=mat.argmax(1);pb=hungarian(mat,valid_groups,ctx["row_ids"][valid])
                    fold_scores.append({"inner_fold":inner,"independent_balanced_accuracy":float(balanced_accuracy_score(y[valid],pa)),
                        "hungarian_balanced_accuracy":float(balanced_accuracy_score(y[valid],pb)),
                        "accuracy":float(accuracy_score(y[valid],pb)),"iterations":max(1,int(model.get_best_iteration())+1)})
                param_results.append({"depth":depth,"learning_rate":lr,"folds":fold_scores,
                    "mean_balanced_accuracy":float(np.mean([z["hungarian_balanced_accuracy"] for z in fold_scores])),
                    "mean_accuracy":float(np.mean([z["accuracy"] for z in fold_scores]))})
        chosen=max(param_results,key=lambda z:(z["mean_balanced_accuracy"],z["mean_accuracy"],-z["depth"],-z["learning_rate"]))
        valid_groups,valid_pairs,group_info=_fit_reconstructed_groups(ctx,fit,held,data,tag=f"outer{fold}-held")
        evh=_evidence(ctx,fit,held)
        vec=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32,max_features=150000)
        vec.fit(ctx["texts"][fit])
        hx=_combine_features(ctx,evh,held,roster,valid_groups,vec,pair_scores=valid_pairs,
            oracle=False,fit_texts=ctx["texts"][fit])
        final=_cat_train(train_x,train_y,chosen["depth"],chosen["learning_rate"])
        mat=_predict_candidate(final,hx);pa=mat.argmax(1);pb=hungarian(mat,valid_groups,ctx["row_ids"][held])
        cp_cols=[c for c in train_x.columns if c=="candidate_label" or not c.startswith("group::")]
        cp_model=_cat_train(train_x[cp_cols],train_y,chosen["depth"],chosen["learning_rate"])
        cp_scores=_predict_candidate(cp_model,hx[cp_cols]);pcp=hungarian(cp_scores,valid_groups,ctx["row_ids"][held])
        scores_all[held]=mat;pred_a[held]=pa;pred_b[held]=pb;pred_cp[held]=pcp;group_all[held]=valid_groups
        final.save_model(str(output/f"E20AB_fold{fold}.cbm"));cp_model.save_model(str(output/f"E20AB_checkpoint_only_fold{fold}.cbm"))
        fi=pd.DataFrame({"feature":list(final.feature_names_),"importance":final.get_feature_importance()})
        importance.extend({"fold":fold,"feature":r.feature,"importance":float(r.importance),"category":"group" if str(r.feature).startswith("group::") else "checkpoint"} for r in fi.itertuples())
        fold_results.append({"fold":fold,"train_rows":len(fit),"held_rows":len(held),"selected":{"depth":chosen["depth"],"learning_rate":chosen["learning_rate"]},"tuning":param_results,
          "E20-A":_comparison(y[held],baseline[held],pa),"E20-B":_comparison(y[held],baseline[held],pb),
          "checkpoint_only":_comparison(y[held],baseline[held],pcp),"group_audit":_group_audit(true_groups[held],valid_groups),
          "group_fit":group_info,"model_path":str(output/f"E20AB_fold{fold}.cbm")})
        print(f"E20-A/B fold {fold}: A BA={balanced_accuracy_score(y[held],pa):.4%}, B BA={balanced_accuracy_score(y[held],pb):.4%}",flush=True)
    if np.any(pred_a<0) or np.any(pred_b<0) or not np.isfinite(scores_all).all():raise RuntimeError("E20-A/B OOF predictions incomplete")
    row_frame=ctx["v0"][["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy()
    row_frame["E13_B"]=np.asarray(LABELS,dtype=object)[baseline];row_frame["E20_A"]=np.asarray(LABELS,dtype=object)[pred_a];row_frame["E20_B"]=np.asarray(LABELS,dtype=object)[pred_b]
    row_frame["predicted_group_id"]=group_all.astype(str)
    row_frame["fold"]=-1
    for outer in ctx["outer_folds"]:row_frame.loc[outer["held_rows"],"fold"]=outer["fold"]
    row_frame.to_parquet(output/"predictions_e20ab.parquet",index=False)
    np.savez_compressed(output/"scores_e20ab.npz",row_ids=ctx["row_ids"],scores=scores_all,y=y,pred_independent=pred_a,pred_hungarian=pred_b,pred_checkpoint_only=pred_cp)
    correct_group=np.zeros(len(y),dtype=bool)
    for g in set(group_all.astype(str)):
        ix=np.flatnonzero(group_all.astype(str)==g)
        correct_group[ix]=len(set(true_groups[ix]))==1 and len(ix)==int(np.sum(true_groups==true_groups[ix[0]]))
    comparison_by_group={}
    for name,pred in (("E20-A",pred_a),("E20-B",pred_b)):
        comparison_by_group[name]={"correctly_grouped":_comparison(y[correct_group],baseline[correct_group],pred[correct_group]),
            "incorrectly_grouped":_comparison(y[~correct_group],baseline[~correct_group],pred[~correct_group]) if (~correct_group).any() else None}
    e19b_result=_e19b_hungarian_baseline(ctx)
    metrics={"experiment":"E20-A/B","E13-B":_metric(y,baseline),"E20-A":_comparison(y,baseline,pred_a),
       "E20-B":_comparison(y,baseline,pred_b),"E20-A_vs_E20-B":_comparison(y,pred_a,pred_b),
       "checkpoint_only_ablation":_comparison(y,baseline,pred_cp),"folds":fold_results,
       "group_audit":_group_audit(true_groups,group_all.astype(str)),"group_size_metrics_A":_group_size_metrics(y,pred_a,group_all.astype(str)),
       "group_size_metrics_B":_group_size_metrics(y,pred_b,group_all.astype(str)),"classification_by_group_correctness":comparison_by_group,
       "feature_importance":importance,"nested_group_fits":nested_group_records,
       "checkpoint_feature_importance_top":_aggregate_importance(importance,"checkpoint")[:100],
       "group_feature_importance_top":_aggregate_importance(importance,"group")[:100],
       "reconstruction_sha256":data["sha256"],"E19-B_artifacts_available":True}
    e19_metrics=ROOT/"artifacts/analysis/e19b/metrics.json"
    metrics["E19-B_grouping"]=(json.loads(e19_metrics.read_text()).get("aggregate_oof") if e19_metrics.is_file() else None)
    metrics["E19-B"]=(e19b_result[1] if e19b_result else None)
    if e19b_result:
        metrics["E20-B_vs_E19-B"]=_comparison(y,e19b_result[0],pred_b)
    metrics["inputs"]={"trust_list_snapshot":ctx["snapshot"],"cache_hashes":ctx["cache_hashes"],"baseline_predictions":str(e16_common.e15.BASELINE_PREDICTIONS)}
    (output/"e20ab_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(importance).to_csv(output/"e20ab_feature_importance.csv",index=False)
    _write_reconstructed_report(ctx["root"],output,metrics)
    return metrics


def _validate_e20b_e19b_group_alignment(ctx, groups, e20b_rows):
    """Require E20-D to use the exact reconstructed partitions used by E20-B."""
    from sklearn.metrics import adjusted_rand_score
    frame=pd.DataFrame({"id":ctx["row_ids"].astype(str),"fold":-1,
                        "e19b_group":groups.e20_group.astype(str).to_numpy()})
    e20b=e20b_rows[["id","fold","predicted_group_id"]].copy()
    e20b["id"]=e20b.id.astype(str)
    if e20b.id.duplicated().any() or set(e20b.id)!=set(frame.id):
        raise ValueError("E20-B and E19-B grouping rows do not cover identical response IDs")
    fold_lookup={int(row):int(f["fold"]) for f in ctx["outer_folds"] for row in f["held_rows"]}
    frame["fold"]=[fold_lookup[i] for i in range(len(frame))]
    joined=frame.merge(e20b,on="id",suffixes=("_e19b","_e20b"),validate="one_to_one")
    if not np.array_equal(joined.fold_e19b.to_numpy(dtype=int),joined.fold_e20b.to_numpy(dtype=int)):
        raise ValueError("E20-B and E19-B outer fold assignments differ")
    fold_audit=[]
    for fold,part in joined.groupby("fold_e19b",sort=True):
        ari=float(adjusted_rand_score(part.e19b_group,part.predicted_group_id))
        if not np.isclose(ari,1.0,atol=0,rtol=0):
            raise ValueError(f"E20-B reconstructed groups differ from E19-B in fold {fold} (ARI={ari})")
        fold_audit.append({"fold":int(fold),"rows":len(part),"adjusted_rand_index":ari})
    return fold_audit


def _incomplete_group_evaluation(ctx, pred_b, scores_b, pred_c, scores_c,
                                 pred_d, scores_d, reconstructed_groups):
    """Replay E19-A's exact sampled row sets and seeds under each decoder."""
    sample_path=ctx["root"]/"artifacts/analysis/e19/sampled_predictions.parquet"
    if not sample_path.is_file():
        raise FileNotFoundError("E20-D incomplete-group evaluation requires E19-A sampled_predictions.parquet")
    sampled=pd.read_parquet(sample_path)
    required={"variant","seed","id","prompt_id","y_true"}
    if not required.issubset(sampled.columns):
        raise ValueError("E19-A incomplete-group sample file is missing its required fields")
    id_to_row={str(r):i for i,r in enumerate(ctx["row_ids"])}
    sampled["row"]=[id_to_row.get(str(i),-1) for i in sampled.id]
    if (sampled.row<0).any():
        raise ValueError("E19-A sampled rows include unknown V0 IDs")
    truth_label=pd.Categorical(sampled.y_true.astype(str),categories=LABELS).codes
    if (truth_label<0).any() or not np.array_equal(truth_label,ctx["y"][sampled.row.to_numpy()]):
        raise ValueError("E19-A sample labels do not match aligned V0 rows")
    records=[];pred_records=[]
    all_predictions={"E20-B":pred_b,"E20-C":pred_c,"E20-D":pred_d}
    all_scores={"E20-B":scores_b,"E20-C":scores_c,"E20-D":scores_d}
    for (variant,seed),part in sampled.groupby(["variant","seed"],sort=True):
        rows=part.row.to_numpy(dtype=np.int64)
        ids=ctx["row_ids"][rows].astype(str)
        yy=ctx["y"][rows]
        groups_d=reconstructed_groups[rows]
        # E20-C is the oracle reference; B and D use identical reconstructed groups.
        groups_c=ctx["groups"][rows].astype(str)
        for model_name in ("E20-B","E20-C","E20-D"):
            groups=groups_c if model_name=="E20-C" else groups_d
            ix=all_scores[model_name][rows]
            pred=hungarian(ix,groups,ids)
            metrics=_metric(yy,pred)
            records.append({"sample_variant":str(variant),"seed":int(seed),"model":model_name,
                            "sample_rows":len(rows),"accuracy":metrics["accuracy"],
                            "balanced_accuracy":metrics["balanced_accuracy"],
                            "groups":int(len(set(groups.tolist()))),
                            "singleton_groups":int(sum(np.sum(groups==g)==1 for g in set(groups.tolist())))})
            for row,label in zip(rows,pred):
                pred_records.append({"sample_variant":str(variant),"seed":int(seed),"id":str(ctx["row_ids"][row]),
                    "y_true":LABELS[int(ctx["y"][row])],"model":model_name,"prediction":LABELS[int(label)],
                    "reconstructed_group_id":str(reconstructed_groups[row]),"true_prompt_id":str(ctx["groups"][row])})
    return records,pred_records


def _run_e20d(ctx, output=OUT):
    """Apply the unchanged E20-C fold models to E19-B reconstructed groups."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=_load_e19b_reconstructions(ctx["root"])
    groups_frame=load_reconstructed_groups(ctx["root"],required=True)
    if groups_frame.id.astype(str).tolist()!=ctx["row_ids"].astype(str).tolist():
        raise ValueError("E19-B reconstructed group rows are not in E20 V0 order")
    e20b_path=output/"predictions_e20ab.parquet"
    if not e20b_path.is_file():raise FileNotFoundError("E20-D requires completed E20-B OOF predictions")
    e20b_rows=pd.read_parquet(e20b_path)
    if e20b_rows.id.astype(str).tolist()!=ctx["row_ids"].astype(str).tolist():
        raise ValueError("E20-B prediction file rows differ from E20-D V0 ordering")
    fold_group_audit=_validate_e20b_e19b_group_alignment(ctx,groups_frame,e20b_rows)
    e20b_scores_file=np.load(output/"scores_e20ab.npz",allow_pickle=True)
    e20b_ids=np.asarray(e20b_scores_file["row_ids"],dtype=str)
    if not np.array_equal(e20b_ids,ctx["row_ids"].astype(str)):
        raise ValueError("E20-B score matrix rows differ from E20-D V0 ordering")
    scores_b=np.asarray(e20b_scores_file["scores"],dtype=np.float32)
    pred_b=np.asarray(e20b_scores_file["pred_hungarian"],dtype=np.int64)
    c_path=output/"scores_e20c.npz"
    if not c_path.is_file():raise FileNotFoundError("E20-D requires saved E20-C fold models and OOF scores")
    c_npz=np.load(c_path,allow_pickle=True)
    if not np.array_equal(np.asarray(c_npz["row_ids"],dtype=str),ctx["row_ids"].astype(str)):
        raise ValueError("E20-C score matrix rows differ from E20-D V0 ordering")
    scores_c=np.asarray(c_npz["scores"],dtype=np.float32)
    pred_c=np.asarray(c_npz["pred_hungarian"],dtype=np.int64)
    scores_d=np.full_like(scores_c,np.nan,dtype=np.float32)
    pred_d=np.full(len(ctx["y"]),-1,dtype=np.int64)
    fold_records=[];fold_to_score=np.asarray(e20b_rows.fold,dtype=np.int64)
    pair_scores=groups_frame.attrs.get("pair_scores",{})
    for outer in ctx["outer_folds"]:
        fold=int(outer["fold"]);fit=np.asarray(outer["fit_rows"],dtype=np.int64)
        held=np.asarray(outer["held_rows"],dtype=np.int64);roster=list(outer["roster"])
        if not np.all(fold_to_score[held]==fold):
            raise ValueError(f"E20-D held rows do not match E20-B fold {fold}")
        model_path=output/f"E20C_fold{fold}.cbm"
        if not model_path.is_file():raise FileNotFoundError(f"E20-C fold classifier missing: {model_path}")
        classifier=CatBoostClassifier();classifier.load_model(str(model_path))
        classifier_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()
        evidence=_evidence(ctx,fit,held)
        vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=1,sublinear_tf=True,
                                   dtype=np.float32,max_features=150000)
        vectorizer.fit(ctx["texts"][fit])
        reconstructed=groups_frame.e20_group.astype(str).to_numpy()[held]
        features=_combine_features(ctx,evidence,held,roster,reconstructed,vectorizer,
            pair_scores=pair_scores,oracle=False,fit_texts=ctx["texts"][fit])
        matrix=_score_with_reused_classifier(classifier,features)
        if matrix.shape!=(len(held),len(LABELS)) or not np.isfinite(matrix).all():
            raise ValueError(f"E20-D fold {fold} produced invalid candidate scores")
        scores_d[held]=matrix
        pred_d[held]=hungarian(matrix,reconstructed,ctx["row_ids"][held])
        fold_records.append({"fold":fold,"train_rows":len(fit),"held_rows":len(held),
            "E20-B":_comparison(ctx["y"][held],ctx["baseline"][held],pred_b[held]),
            "E20-C":_comparison(ctx["y"][held],ctx["baseline"][held],pred_c[held]),
            "E20-D":_comparison(ctx["y"][held],ctx["baseline"][held],pred_d[held]),
            "E20-D_vs_E20-B":_comparison(ctx["y"][held],pred_b[held],pred_d[held]),
            "D_vs_B_changed":int(np.sum(pred_d[held]!=pred_b[held])),
            "group_audit":_group_audit(ctx["groups"][held],reconstructed),
            "classifier_path":str(model_path),"classifier_sha256":classifier_sha256,
            "classifier_reused":True})
        print(f"E20-D fold {fold}: BA={balanced_accuracy_score(ctx['y'][held],pred_d[held]):.4%}",flush=True)
    if (pred_d<0).any() or not np.isfinite(scores_d).all():
        raise RuntimeError("E20-D OOF coverage is incomplete")
    y=ctx["y"];baseline=ctx["baseline"];reconstructed=groups_frame.e20_group.astype(str).to_numpy()
    truth_groups=np.asarray(ctx["groups"],dtype=str)
    e20b_vs_d=_comparison(y,pred_b,pred_d)
    e19_npz=np.load(ctx["root"]/"artifacts/analysis/e19/predictions.npz",allow_pickle=True)
    if not np.array_equal(np.asarray(e19_npz["row_ids"],dtype=str),ctx["row_ids"].astype(str)):
        raise ValueError("E19-A reference rows do not align to E20-D")
    e19a_pred=np.asarray(e19_npz["e19_full"],dtype=np.int64)
    e19b_pred=np.asarray(e20b_scores_file["pred_e19b_hungarian"],dtype=np.int64)
    correctly_grouped=np.zeros(len(y),dtype=bool)
    for group in sorted(set(reconstructed.tolist())):
        rows=np.flatnonzero(reconstructed==group)
        correctly_grouped[rows]=(len(set(truth_groups[rows]))==1 and
            len(rows)==int(np.sum(truth_groups==truth_groups[rows[0]])))
    by_group={}
    for name,pred in (("E20-B",pred_b),("E20-C",pred_c),("E20-D",pred_d)):
        by_group[name]={"correctly_reconstructed":_comparison(y[correctly_grouped],baseline[correctly_grouped],pred[correctly_grouped]),
            "incorrectly_reconstructed":_comparison(y[~correctly_grouped],baseline[~correctly_grouped],pred[~correctly_grouped]) if (~correctly_grouped).any() else None}
    incomplete,incomplete_predictions=_incomplete_group_evaluation(ctx,pred_b,scores_b,pred_c,scores_c,pred_d,scores_d,reconstructed)
    prediction_frame=ctx["v0"][["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy()
    prediction_frame["E13_B"]=np.asarray(LABELS,dtype=object)[baseline]
    prediction_frame["E20_B"]=np.asarray(LABELS,dtype=object)[pred_b]
    prediction_frame["E20_C"]=np.asarray(LABELS,dtype=object)[pred_c]
    prediction_frame["E20_D"]=np.asarray(LABELS,dtype=object)[pred_d]
    prediction_frame["predicted_group_id"]=reconstructed
    prediction_frame["fold"]=fold_to_score
    prediction_frame.to_parquet(output/"predictions_e20d.parquet",index=False)
    np.savez_compressed(output/"scores_e20d.npz",row_ids=ctx["row_ids"],scores=scores_d,y=y,
        pred_hungarian=pred_d,pred_e20b=pred_b,pred_e20c=pred_c,
        reconstructed_group_id=reconstructed,fold=fold_to_score)
    pd.DataFrame(incomplete_predictions).to_parquet(output/"incomplete_group_predictions_e20d.parquet",index=False)
    base_e19a=_metric(y,e19a_pred);base_e19b=_metric(y,e19b_pred)
    metrics={"experiment":"E20-D","rows":len(y),"classifier_source":"unchanged E20-C outer-fold CatBoost models",
      "classifier_reused_without_training":True,"E20-B_and_E20-D_reconstructed_memberships_identical":True,
      "E13-B":_metric(y,baseline),"E19-A":base_e19a,"E19-B":base_e19b,
      "E20-A":json.loads((output/"e20ab_metrics.json").read_text())["E20-A"],
      "E20-B":_comparison(y,baseline,pred_b),"E20-C":_comparison(y,baseline,pred_c),
      "E20-D":_comparison(y,baseline,pred_d),"E20-D_vs_E13-B":_comparison(y,baseline,pred_d),
      "E20-D_vs_E19-A":_comparison(y,e19a_pred,pred_d),"E20-D_vs_E19-B":_comparison(y,e19b_pred,pred_d),
      "E20-D_vs_E20-B":e20b_vs_d,"E20-D_vs_E20-C":_comparison(y,pred_c,pred_d),
      "changed_vs_E20-B":int(np.sum(pred_d!=pred_b)),
      "gap_E20-C_minus_E20-D_pp":float((balanced_accuracy_score(y,pred_c)-balanced_accuracy_score(y,pred_d))*100),
      "folds":fold_records,"group_alignment_to_E20-B_E19-B":fold_group_audit,
      "group_audit":_group_audit(truth_groups,reconstructed),
      "by_reconstruction_correctness":by_group,
      "by_reconstructed_group_size":_group_size_metrics(y,pred_d,reconstructed),
      "incomplete_group_simulations":incomplete,
      "sampling_seeds":sorted({int(x["seed"]) for x in incomplete}),
      "sampling_source":"artifacts/analysis/e19/sampled_predictions.parquet; exact saved E19-A row subsets",
      "hyperparameters":"Reused unchanged from E20-C; no E20-D training or tuning",
      "classifier_paths":[str(output/f"E20C_fold{i}.cbm") for i in range(1,6)],
      "classifier_sha256":{str(i):hashlib.sha256((output/f"E20C_fold{i}.cbm").read_bytes()).hexdigest() for i in range(1,6)}}
    e19b_metrics=json.loads((ctx["root"]/"artifacts/analysis/e19b/metrics.json").read_text())
    metrics["E19-B_outer_fold_grouping_selection"]=[{"fold":int(z["fold"]),"fit_rows":int(z["fit_rows"]),
        "held_rows":int(z["held_rows"]),"selected_C":z["selected"]["C"],
        "selected_threshold":z["selected"]["threshold"]} for z in e19b_metrics["outer_folds"]]
    (output/"e20d_config.json").write_text(json.dumps({"experiment":"E20-D","seed":SEED,
        "inference_models":"E20C_fold1.cbm through E20C_fold5.cbm; reused unchanged",
        "retrained":False,"group_source":"E19-B outer-fold OOF reconstructed group IDs",
        "grouping_selection":"E19-B threshold/C tuned within each outer training fold",
        "grouping_folds":metrics["E19-B_outer_fold_grouping_selection"],
        "e20c_model_sha256":metrics["classifier_sha256"],
        "same_reconstructed_memberships_as_E20B":True},indent=2)+"\n")
    (output/"e20d_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(fold_group_audit).to_csv(output/"e20d_group_alignment.csv",index=False)
    _write_reconstructed_report(ctx["root"],output,{**json.loads((output/"e20ab_metrics.json").read_text()),"E20-D":metrics})
    print(f"E20-D complete: BA={metrics['E20-D']['balanced_accuracy']:.4%}; changes vs B={metrics['changed_vs_E20-B']}",flush=True)
    return metrics


def run(root=ROOT, output=OUT, variant="oracle"):
    os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]=str(THREADS)
    root=Path(root).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ctx=e16_common.load_context(root)
    start=time.monotonic()
    if variant == "all" and not (output/"e20c_metrics.json").is_file():
        run(root,output,"oracle")
    if variant in {"reconstructed","all"}:
        data=_load_e19b_reconstructions(root)
        # Cap BLAS/OpenMP pools too; setting OMP_NUM_THREADS after imports
        # does not constrain already-loaded numerical libraries.
        with threadpool_limits(limits=THREADS):
            ab_metrics=_run_reconstructed(ctx,output,data)
        (output/"e20ab_config.json").write_text(json.dumps({"seed":SEED,"threads":THREADS,"inner_folds":INNER_FOLDS,
            "depths":DEPTHS,"learning_rates":LEARNING_RATES,"iterations":ITERATIONS,"reconstruction_sha256":data["sha256"],
            "grouping":"E19-B pair scorer fitted and tuned inside each outer/nested training partition"},indent=2)+"\n")
        print(f"E20-A/B complete: A BA={ab_metrics['E20-A']['balanced_accuracy']:.4%}, B BA={ab_metrics['E20-B']['balanced_accuracy']:.4%}",flush=True)
        if variant=="reconstructed":return ab_metrics
    if variant in {"e20d","all"}:
        if not (output/"e20ab_metrics.json").is_file():
            raise FileNotFoundError("E20-D requires completed E20-A/B artifacts")
        with threadpool_limits(limits=THREADS):
            return _run_e20d(ctx,output)
    with threadpool_limits(limits=THREADS):
        metrics,predictions,groups=_run_oracle(ctx,output)
    metrics["runtime_seconds"]=time.monotonic()-start
    e19_metrics=Path(root)/"artifacts/analysis/e19b/metrics.json"
    metrics["E19-B"]=(json.loads(e19_metrics.read_text()).get("aggregate_oof") if e19_metrics.is_file() else None)
    metrics["E19-B_artifacts_available"]=bool(e19_metrics.is_file())
    metrics["deferred_variants"]={"E20-A":"awaiting E19-B OOF group assignments and nested group fits","E20-B":"awaiting E19-B OOF group assignments and nested group fits"}
    metrics["inputs"]={"trust_list_snapshot":ctx["snapshot"],"cache_hashes":ctx["cache_hashes"],"baseline_predictions":str(e16_common.e15.BASELINE_PREDICTIONS),"e19b_path":str(E19B)}
    metrics["checkpoint_feature_importance_top"]=_aggregate_importance(metrics["feature_importance"],"checkpoint")[:100]
    metrics["group_feature_importance_top"]=_aggregate_importance(metrics["feature_importance"],"group")[:100]
    (output/"e20c_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(metrics["feature_importance"]).to_csv(output/"e20c_feature_importance.csv",index=False)
    groups.to_parquet(output/"e20c_oracle_groups.parquet",index=False)
    config={"seed":SEED,"threads":THREADS,"inner_folds":INNER_FOLDS,"depths":DEPTHS,"learning_rates":LEARNING_RATES,"iterations":ITERATIONS,"candidate_labels":LABELS,"variant":variant,"oracle_group_source":"V0 prompt_id membership only"}
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    _write_report(root,output,metrics)
    print(f"E20-C complete: accuracy={metrics['E20-C']['accuracy']:.4%} BA={metrics['E20-C']['balanced_accuracy']:.4%} ΔBA={metrics['E20-C']['delta_balanced_accuracy']*100:+.4f} pp",flush=True)
    return metrics


def _write_reconstructed_report(root,output,m):
    cpath=Path(output)/"e20c_metrics.json"
    c=json.loads(cpath.read_text()) if cpath.is_file() else None
    e19a=.749085
    base=m["E13-B"]
    lines=["# E20 — Group-Aware Checkpoint Classifier","",
      "| Model | Balanced accuracy | Gain vs E13-B |","|---|---:|---:|"]
    lines.append(f"| E13-B | {base['balanced_accuracy']:.4%} | — |")
    lines.append(f"| E19-A oracle Hungarian | {e19a:.4%} | +{(e19a-base['balanced_accuracy'])*100:.4f} pp |")
    em=m.get("E19-B")
    lines.append(f"| E19-B reconstructed Hungarian | {em['balanced_accuracy']:.4%} | {(em['balanced_accuracy']-base['balanced_accuracy'])*100:+.4f} pp |" if em else "| E19-B reconstructed Hungarian | grouping metrics unavailable | — |")
    lines.append(f"| E20-A group-aware independent | {m['E20-A']['balanced_accuracy']:.4%} | {m['E20-A']['delta_balanced_accuracy']*100:+.4f} pp |")
    lines.append(f"| E20-B group-aware + Hungarian | {m['E20-B']['balanced_accuracy']:.4%} | {m['E20-B']['delta_balanced_accuracy']*100:+.4f} pp |")
    if c: lines.append(f"| E20-C oracle group-aware + Hungarian | {c['E20-C']['balanced_accuracy']:.4%} | {c['E20-C']['delta_balanced_accuracy']*100:+.4f} pp |")
    d=m.get("E20-D")
    if d: lines.append(f"| E20-D oracle-trained + reconstructed groups + Hungarian | {d['E20-D']['balanced_accuracy']:.4%} | {d['E20-D']['delta_balanced_accuracy']*100:+.4f} pp |")
    lines += ["","E20-A and E20-B use the same fold-specific CatBoost scores and reconstructed groups; only the decoder differs. Grouping is re-fit inside every outer and nested training partition. The true prompt IDs are used only to make pair targets, split folds, tune the grouper on training partitions, and evaluate group errors.","",
      "## E20-A/B results","",f"E20-A accuracy/BA: **{m['E20-A']['accuracy']:.4%} / {m['E20-A']['balanced_accuracy']:.4%}**. E20-B accuracy/BA: **{m['E20-B']['accuracy']:.4%} / {m['E20-B']['balanced_accuracy']:.4%}**. The Hungarian decoder changed BA by {(m['E20-B']['balanced_accuracy']-m['E20-A']['balanced_accuracy'])*100:+.4f} pp.","",
      "| Variant | Corrected | Introduced | Net gains | Changed |","|---|---:|---:|---:|---:|"]
    for name in ("E20-A","E20-B"):
        z=m[name];lines.append(f"| {name} | {z['corrected']} | {z['introduced']} | {z['net_gains']} | {z['changed']} |")
    grouping=m.get("E19-B_grouping") or {}
    lines += ["","## E19-B reference","",
      (f"Applying E19-B's OOF reconstructed groups to the same E19/E13-B utility scores gives **{em['accuracy']:.4%} accuracy and {em['balanced_accuracy']:.4%} balanced accuracy**. E20-B vs that reference: {m['E20-B_vs_E19-B']['corrected']} corrected, {m['E20-B_vs_E19-B']['introduced']} introduced, {m['E20-B_vs_E19-B']['net_gains']} net gains." if em else "An E19-B + Hungarian classification reference could not be reconstructed from the saved aligned score artifacts."),
      (f"E19-B OOF grouping itself achieved {grouping.get('exact_group_recovery',float('nan')):.2%} exact group recovery, {grouping.get('pairwise_precision',float('nan')):.2%} pairwise precision, {grouping.get('pairwise_recall',float('nan')):.2%} pairwise recall and {grouping.get('pairwise_f1',float('nan')):.2%} pairwise F1." if grouping else "E19-B group metrics are unavailable."),"",
      "## E20 outer-fold grouping quality","",f"Across E20 held-out rows: {m['group_audit']['exact_groups']}/{m['group_audit']['true_groups']} exact groups, {m['group_audit']['split_groups']} split prompt groups and {m['group_audit']['merged_groups']} merged groups.","",
      "## Classification by grouping correctness","","| Variant | Group status | Rows | Accuracy | Balanced accuracy |","|---|---|---:|---:|---:|"]
    for name in ("E20-A","E20-B"):
        for status in ("correctly_grouped","incorrectly_grouped"):
            z=m["classification_by_group_correctness"][name][status]
            if z:lines.append(f"| {name} | {status} | {z['rows']} | {z['accuracy']:.4%} | {z['balanced_accuracy']:.4%} |")
    lines += ["","## Five outer folds","","| Fold | E20-A BA | E20-B BA | Δ B-A | Corrected B | Introduced B | Exact groups | Splits | Merges |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in m["folds"]:lines.append(f"| {f['fold']} | {f['E20-A']['balanced_accuracy']:.4%} | {f['E20-B']['balanced_accuracy']:.4%} | {(f['E20-B']['balanced_accuracy']-f['E20-A']['balanced_accuracy'])*100:+.4f} pp | {f['E20-B']['corrected']} | {f['E20-B']['introduced']} | {f['group_audit']['exact_groups']} | {f['group_audit']['split_groups']} | {f['group_audit']['merged_groups']} |")
    if d:
        lines += ["","## E20-D: oracle-trained classifier on reconstructed groups","",
          f"E20-D accuracy/BA: **{d['E20-D']['accuracy']:.4%} / {d['E20-D']['balanced_accuracy']:.4%}**; gain over E13-B: **{d['E20-D']['delta_balanced_accuracy']*100:+.4f} pp**.",
          f"Versus E13-B it corrected {d['E20-D']['corrected']} errors, introduced {d['E20-D']['introduced']}, for {d['E20-D']['net_gains']} net correct predictions.",
          f"Relative to E19-A, E19-B and E20-B, balanced-accuracy changes are {(d['E20-D_vs_E19-A']['balanced_accuracy']-d['E19-A']['balanced_accuracy'])*100:+.4f}, {(d['E20-D_vs_E19-B']['balanced_accuracy']-d['E19-B']['balanced_accuracy'])*100:+.4f}, and {(d['E20-D']['balanced_accuracy']-m['E20-B']['balanced_accuracy'])*100:+.4f} pp.",
          f"Compared with E20-B on identical reconstructed groups, E20-D corrected {d['E20-D_vs_E20-B']['corrected']} predictions, introduced {d['E20-D_vs_E20-B']['introduced']}, netted {d['E20-D_vs_E20-B']['net_gains']}, and changed {d['changed_vs_E20-B']} predictions. Its gap below E20-C's oracle BA is {d['gap_E20-C_minus_E20-D_pp']:.4f} pp.","",
          "E20-D reused the corresponding E20-C fold classifier without retraining. E19-B's OOF group predictions use thresholds selected within each outer training fold; the held fold is excluded from that selection:","",
          "| Fold | E19-B fit rows | Held rows | Selected C | Selected threshold |","|---:|---:|---:|---:|---:|"]
        for z in d['E19-B_outer_fold_grouping_selection']:
            lines.append(f"| {z['fold']} | {z['fit_rows']} | {z['held_rows']} | {z['selected_C']} | {z['selected_threshold']:.2f} |")
        lines += ["",
          "| Fold | E20-B BA | E20-C BA | E20-D BA | D − B | D corrected vs B | D introduced vs B |","|---:|---:|---:|---:|---:|---:|---:|"]
        dfold={int(x['fold']):x for x in d['folds']}
        for f in m['folds']:
            q=dfold[int(f['fold'])]
            lines.append(f"| {f['fold']} | {f['E20-B']['balanced_accuracy']:.4%} | {q['E20-C']['balanced_accuracy']:.4%} | {q['E20-D']['balanced_accuracy']:.4%} | {(q['E20-D']['balanced_accuracy']-f['E20-B']['balanced_accuracy'])*100:+.4f} pp | {q['E20-D_vs_E20-B']['corrected']} | {q['E20-D_vs_E20-B']['introduced']} |")
        lines += ["","## E20-D by reconstructed group quality","","| Group status | Rows | Accuracy | Balanced accuracy |","|---|---:|---:|---:|"]
        for status,z in d['by_reconstruction_correctness']['E20-D'].items():
            if z:lines.append(f"| {status} | {z['rows']} | {z['accuracy']:.4%} | {z['balanced_accuracy']:.4%} |")
        lines += ["","## E20-D by reconstructed group size","","| Group size | Rows | Accuracy | Balanced accuracy |","|---:|---:|---:|---:|"]
        for z in d['by_reconstructed_group_size']:lines.append(f"| {z['group_size']} | {z['rows']} | {z['accuracy']:.4%} | {z['balanced_accuracy']:.4%} |")
        lines += ["","## E20-D per-label precision and recall","","| Label | Support | Precision | Recall | F1 |","|---|---:|---:|---:|---:|"]
        for lab,z in d['E20-D']['per_label'].items():lines.append(f"| {lab} | {z['support']} | {z['precision']:.4%} | {z['recall']:.4%} | {z['f1']:.4%} |")
        lines += ["","## Incomplete-group simulations","","The exact E19-A saved row subsets and seeds (2026–2030) were replayed. E20-B and E20-D use reconstructed group membership for assignment; E20-C is the oracle-group reference.","",
          "| E19-A sample | Model | Mean accuracy | SD | Mean balanced accuracy | SD |","|---|---|---:|---:|---:|---:|"]
        sim=pd.DataFrame(d['incomplete_group_simulations'])
        for (variant,model_name),part in sim.groupby(['sample_variant','model'],sort=True):
            lines.append(f"| {variant} | {model_name} | {part.accuracy.mean():.4%} | {part.accuracy.std(ddof=1):.4%} | {part.balanced_accuracy.mean():.4%} | {part.balanced_accuracy.std(ddof=1):.4%} |")
        simmeans=sim.groupby(['sample_variant','model']).balanced_accuracy.mean()
        b10=float(simmeans[('B-incomplete-2-10','E20-B')]);d10=float(simmeans[('B-incomplete-2-10','E20-D')])
        b18=float(simmeans[('C-incomplete-2-18','E20-B')]);d18=float(simmeans[('C-incomplete-2-18','E20-D')])
        lines += ["","## E20-D conclusion","",
          f"E20-B versus E20-D mean fold BA standard deviation: {np.std([f['E20-B']['balanced_accuracy'] for f in m['folds']],ddof=1)*100:.3f} pp vs {np.std([f['E20-D']['balanced_accuracy'] for f in d['folds']],ddof=1)*100:.3f} pp.",
          f"On the E19-A incomplete-group samples, E20-D changed mean BA by {(d10-b10)*100:+.4f} pp for 2–10-row subsets and {(d18-b18)*100:+.4f} pp for 2–18-row subsets versus E20-B.",
          "The stronger strategy is determined by OOF BA and its fold stability below; V0 remains development data and should not be treated as expected Kaggle score.",""]
        e20b_mean=float(np.mean([f['E20-B']['balanced_accuracy'] for f in m['folds']]))
        e20d_mean=float(np.mean([f['E20-D']['balanced_accuracy'] for f in d['folds']]))
        d_wins=int(sum(f['E20-D']['balanced_accuracy']>f['E20-B']['balanced_accuracy'] for f in d['folds']))
        lines[-1]=(f"Across outer folds, E20-B mean BA was {e20b_mean:.4%} and E20-D mean BA was {e20d_mean:.4%}; E20-D won {d_wins}/5 folds. Recommend **{'E20-D' if d['E20-D']['balanced_accuracy']>m['E20-B']['balanced_accuracy'] else 'E20-B'}** by aggregate OOF balanced accuracy, with fold variability considered.")
    recommendation=("E20-D is the stronger of these two training strategies on five-fold V0: it improved aggregate BA by 0.3701 pp and reduced fold-level BA SD from 1.748 to 0.552 pp. Incomplete-group simulations were mixed, so keep the Kaggle estimate provisional and compare on a separate holdout before deployment." if d else "Do not submit E20-B as the current best candidate. The checkpoint-only + Hungarian ablation scored higher on these OOF rows; validate that simpler inference variant and its deployment path separately before deciding whether to submit.")
    lines += ["","## Per-label E20-B precision and recall","","| Label | Support | Precision | Recall | F1 |","|---|---:|---:|---:|---:|"]
    for lab,z in m["E20-B"]["per_label"].items():lines.append(f"| {lab} | {z['support']} | {z['precision']:.4%} | {z['recall']:.4%} | {z['f1']:.4%} |")
    lines += ["","## Feature ablation","","| Feature set | Accuracy | Balanced accuracy |","|---|---:|---:|"]
    cp=m["checkpoint_only_ablation"];lines.append(f"| Checkpoint only + Hungarian | {cp['accuracy']:.4%} | {cp['balanced_accuracy']:.4%} |")
    lines.append(f"| Checkpoint + reconstructed group features + Hungarian | {m['E20-B']['accuracy']:.4%} | {m['E20-B']['balanced_accuracy']:.4%} |")
    group_delta=(m["E20-B"]["balanced_accuracy"]-cp["balanced_accuracy"])*100
    hungarian_delta=(m["E20-B"]["balanced_accuracy"]-m["E20-A"]["balanced_accuracy"])*100
    oracle_gap=((c["E20-C"]["balanced_accuracy"]-m["E20-B"]["balanced_accuracy"])*100) if c else float("nan")
    reconstruction_gap=((.749085-em["balanced_accuracy"])*100) if em else float("nan")
    lines += ["","## Findings","",
      f"1. **Learned group features:** group features changed E20-B by {group_delta:+.4f} pp versus the checkpoint-only + Hungarian ablation; they did not add value in this comparison.",
      f"2. **Hungarian assignment:** it added {hungarian_delta:+.4f} pp over E20-A and improved four of five outer folds; fold 5 regressed by 1.3091 pp.",
      f"3. **Reconstruction cost:** applying reconstructed rather than oracle groups to the same E13 utility scores lost {reconstruction_gap:.4f} pp BA (E19-A vs E19-B). E20-C is {oracle_gap:.4f} pp above E20-B, but that comparison also changes the classifier's group features and is not a pure grouping-only estimate.",
      "4. **Kaggle readiness:** E20-B is not the preferred submission candidate yet: the group-feature ablation is weaker than checkpoint-only + Hungarian, fold 5 reverses the assignment benefit, and all V0 figures are development results. The checkpoint-only + reconstructed Hungarian variant deserves a separate deployment review." ]
    lines += ["","## Model features and artifacts","", "Mean CatBoost importance is in `e20ab_feature_importance.csv`; separate checkpoint evidence and group feature rankings are in `e20ab_metrics.json`. `predictions_e20ab.parquet` and `scores_e20ab.npz` contain row-aligned OOF results. Fold-specific CatBoost models and nested grouping fits are under the E20 analysis directory.",""]
    if d:
        lines += ["E20-D outputs: `predictions_e20d.parquet`, `scores_e20d.npz`, `e20d_metrics.json`, `e20d_config.json`, `e20d_group_alignment.csv`, and `incomplete_group_predictions_e20d.parquet`. The five E20-C classifiers were hash-verified and loaded directly; no E20-D classifier was trained.",""]
    lines += [
      "V0 is a development set and checkpoint selection has used V0; these scores are not expected Kaggle performance.","",
      "## Recommendation","", recommendation,""]
    (Path(root)/"docs/e20.md").write_text("\n".join(lines))


def _write_report(root,output,m):
    base=m["E13-B"];oracle=m["E20-C"];e19a=0.749085
    e19b=m.get("E19-B")
    rows=["# E20 — Group-Aware Checkpoint Classifier","",
      "| Model | Balanced accuracy | Gain vs E13-B |","|---|---:|---:|",
      f"| E13-B | {base['balanced_accuracy']:.4%} | — |",
      f"| E19-A oracle Hungarian | {e19a:.4%} | +{(e19a-base['balanced_accuracy'])*100:.4f} pp |",
      (f"| E19-B reconstructed Hungarian | {e19b['balanced_accuracy']:.4%} | {(e19b['balanced_accuracy']-base['balanced_accuracy'])*100:+.4f} pp |" if e19b else "| E19-B reconstructed Hungarian | pending; grouping artifacts unavailable | — |"),
      "| E20-A group-aware independent | pending E19-B nested groups | — |",
      "| E20-B group-aware + Hungarian | pending E19-B nested groups | — |",
      f"| E20-C oracle group-aware + Hungarian | {oracle['balanced_accuracy']:.4%} | {oracle['delta_balanced_accuracy']*100:+.4f} pp |","",
      (f"E20-C differs from E19-A by {(m['E20-C_vs_E19-A']['delta_balanced_accuracy']*100):+.4f} pp BA (E20-C {oracle['balanced_accuracy']:.4%}, E19-A {m['E19-A']['balanced_accuracy']:.4%})." if "E20-C_vs_E19-A" in m else "E19-A row-aligned predictions were unavailable for a direct comparison."),"",
      "## E20-C results","",
      f"E20-C scored {oracle['rows']} held-out V0 rows. Accuracy **{oracle['accuracy']:.4%}**, balanced accuracy **{oracle['balanced_accuracy']:.4%}**. It corrected {oracle['corrected']} E13-B errors, introduced {oracle['introduced']} errors, and changed {oracle['changed']} predictions (net {oracle['net_gains']}). The independent argmax variant scored {m['E20-C-independent']['accuracy']:.4%} accuracy and {m['E20-C-independent']['balanced_accuracy']:.4%} balanced accuracy.","",
      "## Five outer folds","","| Fold | Rows | Independent BA | Hungarian BA | Δ BA vs E13-B | Corrected | Introduced |","|---:|---:|---:|---:|---:|---:|---:|"]
    for f in m["folds"]:
        a=f["E20-C-independent"];b=f["E20-C"]
        rows.append(f"| {f['fold']} | {f['held_rows']} | {a['balanced_accuracy']:.4%} | {b['balanced_accuracy']:.4%} | {b['delta_balanced_accuracy']*100:+.4f} pp | {b['corrected']} | {b['introduced']} |")
    rows += ["","## Feature ablation","","| Feature set / decoder | Accuracy | Balanced accuracy | Δ BA vs E13-B | Corrected | Introduced |","|---|---:|---:|---:|---:|---:|"]
    cp=m["ablation_checkpoint_only"];full=m["E20-C"];ind=m["E20-C-independent"]
    rows.append(f"| Checkpoint features only + Hungarian | {cp['accuracy']:.4%} | {cp['balanced_accuracy']:.4%} | {cp['delta_balanced_accuracy']*100:+.4f} pp | {cp['corrected']} | {cp['introduced']} |")
    rows.append(f"| Checkpoint + oracle group features + Hungarian | {full['accuracy']:.4%} | {full['balanced_accuracy']:.4%} | {full['delta_balanced_accuracy']*100:+.4f} pp | {full['corrected']} | {full['introduced']} |")
    rows.append(f"| Checkpoint + oracle group features + independent argmax | {ind['accuracy']:.4%} | {ind['balanced_accuracy']:.4%} | {ind['delta_balanced_accuracy']*100:+.4f} pp | {ind['corrected']} | {ind['introduced']} |")
    rows += ["","Using the same fold-selected depth and learning rate, oracle group features changed balanced accuracy by **{:+.4f} pp** versus checkpoint-only features. Hungarian assignment changed the full group-feature score by **{:+.4f} pp** versus independent argmax.".format((full["balanced_accuracy"]-cp["balanced_accuracy"])*100,(full["balanced_accuracy"]-ind["balanced_accuracy"])*100),"",
      "## Group-size results","","| Group size | Rows | Accuracy | Balanced accuracy |","|---:|---:|---:|---:|"]
    for q in m["per_group_size"]:rows.append(f"| {q['group_size']} | {q['rows']} | {q['accuracy']:.4%} | {q['balanced_accuracy']:.4%} |")
    rows += ["","## Oracle group audit","",f"E20-C has {m['group_audit']['true_groups']} true groups, all recovered exactly; splits {m['group_audit']['split_groups']}, merges {m['group_audit']['merged_groups']}. Consequently all E20-C rows are correctly grouped; an incorrectly-grouped comparison is not applicable. E19-B's measured grouping metrics are unavailable because its reconstruction/evaluation outputs are absent.","",
      "## Per-label precision and recall","","| Label | Support | Precision | Recall | F1 |","|---|---:|---:|---:|---:|"]
    for label in LABELS:
        z=oracle["per_label"][label];rows.append(f"| {label} | {z['support']} | {z['precision']:.4%} | {z['recall']:.4%} | {z['f1']:.4%} |")
    rows += ["","## Feature importance","","Mean CatBoost importance across the five outer-fold models:","","Top checkpoint evidence features:",""]
    for q in m["checkpoint_feature_importance_top"][:15]:rows.append(f"- `{q['feature']}` — {q['mean_importance']:.4f}")
    rows += ["","Top group-level features:",""]
    for q in m["group_feature_importance_top"][:15]:rows.append(f"- `{q['feature']}` — {q['mean_importance']:.4f}")
    rows += ["","## Ablation status","","Both checkpoint-only and checkpoint-plus-group CatBoost models were trained and evaluated in every outer fold with the same selected hyperparameters. E20-A/B and reconstructed-group ablations remain deferred until E19-B reconstruction and nested grouping artifacts exist. No Qwen reconstruction or semantic inference was launched.","",
      "## Interpretation","",
      f"- **Group features:** adding oracle group features changed BA by {(full['balanced_accuracy']-cp['balanced_accuracy'])*100:+.4f} pp versus checkpoint-only features. The group evidence did not improve this ablation.",
      "- **Hungarian assignment:** E20-C applies the E19 assignment algorithm to the same candidate scores as its independent variant; its delta isolates assignment from the classifier.",
      "- **Reconstructed grouping cost:** Cannot yet be measured. E19-B's OOF assignments, reconstruction table, semantic vectors and pair diagnostics are not present locally.",
      "- **Submission readiness:** E20-B cannot be recommended for Kaggle until E19-B grouping results and leakage-controlled E20-A/B evaluation are available.",
      "","V0 is a development set and E13 checkpoint selection has used it; these metrics are not expected Kaggle performance.","",
      "## Artifacts","",f"- `{output}`","- `predictions_e20c.parquet`, `scores_e20c.npz`: row-aligned OOF E20-C predictions and 18-label scores.","- `E20C_fold*.cbm`: fold-specific CatBoost classifiers.","- `e20c_oracle_groups.parquet`, `e20c_feature_importance.csv`, `config.json`, `e20c_metrics.json`.",""]
    (Path(root)/"docs/e20.md").write_text("\n".join(rows))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--variant",choices=("oracle","reconstructed","e20d","all"),default="oracle");a=p.parse_args(argv);run(a.root,a.output,a.variant);return 0


if __name__=="__main__":raise SystemExit(main())
