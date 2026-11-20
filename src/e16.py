"""E16: text-only prompt reconstruction and soft uniqueness decoding."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import e16_common as common
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e16"
SEED = 2026
THREADS = 24
INNER_FOLDS = 3
SIMULATION_REPEATS = 10
PAIR_THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
UNIQUENESS_PENALTIES = (0.05, 0.10, 0.20, 0.35)


def _basic(text):
    text = str(text)
    words = re.findall(r"\b\w+\b", text.casefold())
    return words


def _pair_features(texts, char_matrix, word_matrix, left, right):
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if len(left) != len(right):
        raise ValueError("Pair index vectors must align")
    if not len(left):
        return np.empty((0, 17), dtype=np.float32)
    cs = np.asarray(char_matrix[left].multiply(char_matrix[right]).sum(axis=1)).ravel()
    ws = np.asarray(word_matrix[left].multiply(word_matrix[right]).sum(axis=1)).ravel()
    cache = [_basic(t) for t in texts]
    token_jaccard = np.empty(len(left), dtype=np.float32)
    token_overlap = np.empty(len(left), dtype=np.float32)
    for n, (a, b) in enumerate(zip(left, right)):
        sa, sb = set(cache[a]), set(cache[b])
        token_jaccard[n] = len(sa & sb) / max(1, len(sa | sb))
        token_overlap[n] = len(sa & sb) / max(1, min(len(sa), len(sb)))
    lengths = np.asarray([len(str(t)) for t in texts], dtype=np.float64)
    words = np.asarray([len(x) for x in cache], dtype=np.float64)
    lines = np.asarray([max(1, str(t).count("\n") + 1) for t in texts], dtype=np.float64)
    paras = np.asarray([max(1, len(re.split(r"\n\s*\n", str(t).strip()))) for t in texts], dtype=np.float64)
    punct = np.asarray([sum(ch in ",.;:!?-—–()[]{}\"'“”‘’…" for ch in str(t)) for t in texts], dtype=float)
    upper = np.asarray([sum(ch.isupper() for ch in str(t)) for t in texts], dtype=float)
    alpha = np.asarray([sum(ch.isalpha() for ch in str(t)) for t in texts], dtype=float)
    lf = np.asarray([str(t).count("\n") / max(1, len(str(t))) for t in texts])
    ind = np.asarray([sum(bool(re.match(r"^[ \t]+\S", line)) for line in str(t).splitlines()) for t in texts], dtype=float)
    def pair_ratio(x, a, b):
        return np.minimum(x[a], x[b]) / np.maximum(1.0, np.maximum(x[a], x[b]))
    values = np.column_stack([
        cs, ws, token_jaccard, token_overlap,
        pair_ratio(lengths, left, right), np.abs(np.log1p(lengths[left])-np.log1p(lengths[right])),
        pair_ratio(words, left, right), np.abs(np.log1p(words[left])-np.log1p(words[right])),
        pair_ratio(lines, left, right), pair_ratio(paras, left, right),
        pair_ratio(punct, left, right), pair_ratio(upper, left, right),
        pair_ratio(alpha, left, right), pair_ratio(lf, left, right),
        pair_ratio(ind, left, right),
        np.asarray([len(set(str(texts[a])) & set(str(texts[b]))) /
                    max(1, len(set(str(texts[a])) | set(str(texts[b]))))
                    for a, b in zip(left, right)]),
        np.asarray([abs(str(texts[a]).count("\n")-str(texts[b]).count("\n")) /
                    max(1, str(texts[a]).count("\n")+str(texts[b]).count("\n"))
                    for a, b in zip(left, right)]),
    ]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite E16 pair features")
    return values


def _training_pairs(texts, groups, char_matrix, *, neighbors=10):
    """Build same-prompt positives and high-similarity cross-prompt negatives."""
    groups = np.asarray(groups, dtype=str)
    positives = []
    for group in sorted(set(groups)):
        ix = np.flatnonzero(groups == group)
        positives.extend((int(a), int(b)) for a, b in combinations(ix.tolist(), 2))
    n = len(texts)
    # A full V0 prompt commonly has 18 sibling responses; querying only 11
    # neighbors mostly returns same-prompt positives and starves hard negatives.
    # Search broadly, then retain the closest cross-prompt candidates per row.
    k = min(n, max(40, neighbors * 6 + 1))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=THREADS)
    nn.fit(char_matrix)
    distances, indices = nn.kneighbors(char_matrix)
    neg = set()
    for i in range(n):
        selected = 0
        for distance, j in zip(distances[i], indices[i]):
            j = int(j)
            if i == j or groups[i] == groups[j]:
                continue
            neg.add((min(i, j), max(i, j)))
            selected += 1
            if selected >= neighbors:
                break
    negative = sorted(neg)
    if not positives or not negative:
        raise ValueError("E16 pair construction needs positive and hard-negative pairs")
    return np.asarray(positives, dtype=np.int64), np.asarray(negative, dtype=np.int64)


def fit_pair_model(texts, groups, *, c=1.0):
    """Fit vocabulary and pair classifier using training texts/groups only."""
    texts = np.asarray(texts, dtype=str)
    groups = np.asarray(groups, dtype=str)
    if len(texts) != len(groups) or len(texts) < 20:
        raise ValueError("Invalid E16 pair training rows")
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
                           sublinear_tf=True, dtype=np.float32, max_features=300000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                           sublinear_tf=True, dtype=np.float32, max_features=150000)
    xc, xw = char.fit_transform(texts), word.fit_transform(texts)
    pos, neg = _training_pairs(texts, groups, xc)
    pairs = np.concatenate([pos, neg], axis=0)
    x = _pair_features(texts, xc, xw, pairs[:, 0], pairs[:, 1])
    y = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=float(c), class_weight="balanced",
                               solver="liblinear", max_iter=2000, random_state=SEED)
    model.fit(scaler.transform(x), y)
    return {"char_vectorizer": char, "word_vectorizer": word, "scaler": scaler,
            "model": model, "C": float(c), "training_rows": int(len(texts)),
            "positive_pairs": int(len(pos)), "hard_negative_pairs": int(len(neg)),
            "pair_feature_count": int(x.shape[1])}


def predict_pair_probabilities(model_bundle, texts):
    """Predict all unordered text pairs; prompt IDs are deliberately absent."""
    texts = np.asarray(texts, dtype=str)
    n = len(texts)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    left, right = np.triu_indices(n, k=1)
    xc = model_bundle["char_vectorizer"].transform(texts)
    xw = model_bundle["word_vectorizer"].transform(texts)
    feat = _pair_features(texts, xc, xw, left, right)
    prob = model_bundle["model"].predict_proba(model_bundle["scaler"].transform(feat))[:, 1]
    return left, right, prob.astype(np.float32)


def _joint_objective(scores, assignment, edges, penalty):
    objective = float(scores[np.arange(len(assignment)), assignment].sum())
    if len(edges[0]):
        left, right, prob = edges
        objective -= float(penalty * np.sum(prob * (assignment[left] == assignment[right])))
    return objective


def decode_graph(ids, scores, edge_predictions, threshold, penalty):
    """Soft uniqueness decoding over text-only graph edges.

    Inputs are only IDs, E13-derived label scores, and pair probabilities from
    response text. Prompt IDs never enter this inference function.
    """
    ids = np.asarray(ids, dtype=str)
    scores = common.finite_scores(np.asarray(scores, dtype=np.float64))
    left, right, probs = edge_predictions
    left, right, probs = map(np.asarray, (left, right, probs))
    if scores.shape != (len(ids), len(LABELS)) or len(left) != len(right) or len(left) != len(probs):
        raise ValueError("E16 decoder input dimensions do not align")
    active = probs >= float(threshold)
    edges = (left[active].astype(np.int64), right[active].astype(np.int64), probs[active].astype(float))
    adjacency = [[] for _ in ids]
    for a, b, q in zip(*edges):
        adjacency[a].append((int(b), float(q)))
        adjacency[b].append((int(a), float(q)))
    pred = np.argmax(scores, axis=1).astype(np.int64)
    order = np.argsort(ids, kind="stable")
    for _ in range(12):
        changed = False
        for i in order:
            current = int(pred[i])
            local = scores[i].copy()
            for j, q in adjacency[i]:
                local[pred[j]] -= float(penalty) * q
            best = int(np.argmax(local))
            if local[best] > local[current] + 1e-12:
                pred[i] = best
                changed = True
        if not changed:
            break
    return pred, {"left": edges[0], "right": edges[1], "probability": edges[2],
                  "active_nodes": np.asarray([bool(x) for x in adjacency]),
                  "active_edges": int(active.sum()), "total_pairs": int(len(probs))}


def _graph_metrics(edge_predictions, groups, threshold):
    left, right, prob = edge_predictions
    groups = np.asarray(groups, dtype=str)
    target = groups[left] == groups[right]
    pred = prob >= threshold
    tp = int(np.sum(pred & target)); fp = int(np.sum(pred & ~target)); fn = int(np.sum(~pred & target))
    return {"pairs": int(len(target)), "positive_pairs": int(target.sum()),
            "predicted_edges": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": float(tp / max(1, tp+fp)), "recall": float(tp / max(1, tp+fn)),
            "f1": float(2*tp / max(1, 2*tp+fp+fn))}


def _sample_hidden(rows, groups, seed):
    """Use prompt IDs only for deterministic held-out batch simulation."""
    rng = np.random.default_rng(int(seed))
    groups = np.asarray(groups, dtype=str)
    selected = []
    for group in sorted(set(groups[np.asarray(rows, dtype=np.int64)])):
        ix = np.asarray(rows, dtype=np.int64)[groups[np.asarray(rows, dtype=np.int64)] == group]
        count = int(rng.integers(1, min(3, len(ix)) + 1))
        selected.extend(sorted(rng.choice(ix, size=count, replace=False).tolist()))
    return np.asarray(selected, dtype=np.int64)


def _filter_edges(edge_predictions, global_rows, selected_rows):
    pos = {int(row): i for i, row in enumerate(global_rows)}
    remap = {pos[int(row)]: j for j, row in enumerate(selected_rows)}
    left, right, prob = edge_predictions
    keep = np.asarray([(int(a) in remap and int(b) in remap) for a, b in zip(left, right)])
    return (np.asarray([remap[int(a)] for a in left[keep]], dtype=np.int64),
            np.asarray([remap[int(b)] for b in right[keep]], dtype=np.int64), prob[keep])


def _score_setting(y, base, scores, edge_pred, ids, threshold, penalty):
    pred, graph = decode_graph(ids, scores, edge_pred, threshold, penalty)
    metric = common.comparison(y, base, pred)
    metric.update({"activation_rate": float(graph["active_nodes"].mean()) if len(ids) else 0.0,
                   "active_edges": graph["active_edges"]})
    return pred, graph, metric


def _choose_params(records, setting):
    best = None
    for threshold in PAIR_THRESHOLDS:
        for penalty in UNIQUENESS_PENALTIES:
            y_all, base_all, pred_all = [], [], []
            for rec in records:
                if setting == "complete":
                    rows = rec["rows"]
                    edge = rec["edges"]
                else:
                    for sample in rec["samples"]:
                        rows = sample["rows"]
                        edge = _filter_edges(rec["edges"], rec["rows"], rows)
                        # Tuning needs only decoded labels; computing classification
                        # metrics on small simulated batches emits misleading
                        # missing-class warnings and does not affect selection.
                        p, _ = decode_graph(rec["ids"][sample["take"]],
                            rec["scores"][sample["take"]], edge, threshold, penalty)
                        y_all.extend(rec["y"][sample["take"]].tolist())
                        base_all.extend(rec["base"][sample["take"]].tolist())
                        pred_all.extend(p.tolist())
                    continue
                p, _, _ = _score_setting(rec["y"], rec["base"], rec["scores"], edge,
                                         rec["ids"], threshold, penalty)
                y_all.extend(rec["y"].tolist()); base_all.extend(rec["base"].tolist()); pred_all.extend(p.tolist())
            ba = float(balanced_accuracy_score(y_all, pred_all))
            acc = float(accuracy_score(y_all, pred_all))
            key = (ba, acc, -threshold, -penalty)
            if best is None or key > best[0]:
                best = (key, {"threshold": threshold, "penalty": penalty,
                              "tuning_accuracy": acc, "tuning_balanced_accuracy": ba})
    return best[1]


def _inner_records(ctx, outer_fit, *, output):
    records = []
    y, groups = ctx["y"], ctx["groups"]
    for inner_no, (fit_local, valid_local) in enumerate(GroupKFold(INNER_FOLDS).split(
            outer_fit, y[outer_fit], groups[outer_fit]), 1):
        fit_rows, valid_rows = outer_fit[fit_local], outer_fit[valid_local]
        pair_model = fit_pair_model(ctx["texts"][fit_rows], groups[fit_rows], c=1.0)
        edges = predict_pair_probabilities(pair_model, ctx["texts"][valid_rows])
        est = common.fit_e13_predict(ctx, fit_rows, valid_rows)
        samples = []
        for r in range(SIMULATION_REPEATS):
            chosen = _sample_hidden(valid_rows, groups, SEED + r)
            take = np.asarray([{int(row): i for i, row in enumerate(valid_rows)}[int(x)] for x in chosen])
            samples.append({"rows": chosen, "take": take})
        records.append({"rows": valid_rows, "y": y[valid_rows], "base": est["baseline"],
                        "scores": est["label_scores"], "ids": ctx["row_ids"][valid_rows],
                        "edges": edges, "samples": samples,
                        "pair_train": {k: pair_model[k] for k in ("positive_pairs", "hard_negative_pairs")}})
        print(f"E16 tuning fold {inner_no}/{INNER_FOLDS}: pair data {pair_model['positive_pairs']}+"
              f"{pair_model['hard_negative_pairs']} pairs", flush=True)
    return records


def _metrics_for_batches(ctx, records, params, setting):
    all_y, all_base, all_pred = [], [], []
    all_pair_p, all_pair_target = [], []
    per_batch = []
    for rec in records:
        chunks = ([{"take": np.arange(len(rec["rows"])), "rows": rec["rows"], "seed": None}]
                  if setting == "complete" else
                  [{**s, "seed": s["seed"]} for s in rec["samples"]])
        for sample in chunks:
            take = sample["take"]
            rows = sample["rows"]
            edges = rec["edges"] if setting == "complete" else _filter_edges(rec["edges"], rec["rows"], rows)
            pred, graph = decode_graph(rec["ids"][take], rec["scores"][take], edges,
                                       params["threshold"], params["penalty"])
            yy, bb = rec["y"][take], rec["base"][take]
            all_y.extend(yy.tolist()); all_base.extend(bb.tolist()); all_pred.extend(pred.tolist())
            local_groups = ctx["groups"][rows]
            gm = _graph_metrics(edges, local_groups, params["threshold"])
            per_batch.append({"rows": int(len(rows)), **gm,
                              "activation_rate": float(graph["active_nodes"].mean()) if len(rows) else 0,
                              "active_edges": graph["active_edges"], "seed": sample.get("seed")})
            # Store pair probabilities / truth for aggregate graph metrics.
            l, r, q = edges
            all_pair_p.extend(q.tolist())
            all_pair_target.extend((local_groups[l] == local_groups[r]).astype(int).tolist())
    metric = common.comparison(all_y, all_base, all_pred)
    if all_pair_p:
        pred_pair = np.asarray(all_pair_p) >= params["threshold"]
        truth_pair = np.asarray(all_pair_target, dtype=bool)
        tp = int((pred_pair & truth_pair).sum()); fp = int((pred_pair & ~truth_pair).sum()); fn = int((~pred_pair & truth_pair).sum())
        pair_metrics = {"pairs": len(truth_pair), "predicted_edges": int(pred_pair.sum()),
                        "tp": tp, "fp": fp, "fn": fn,
                        "precision": tp/max(1,tp+fp), "recall": tp/max(1,tp+fn),
                        "f1": 2*tp/max(1,2*tp+fp+fn)}
    else:
        pair_metrics = {"pairs": 0, "predicted_edges": 0, "tp": 0, "fp": 0,
                        "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    metric.update({"pair_graph": pair_metrics, "activation_rate": float(np.mean([x["activation_rate"] for x in per_batch])) if per_batch else 0,
                   "mean_active_edges_per_batch": float(np.mean([x["active_edges"] for x in per_batch])) if per_batch else 0,
                   "batch_count": len(per_batch), "per_batch": per_batch})
    return metric, np.asarray(all_y), np.asarray(all_base), np.asarray(all_pred)


def run(root=ROOT, output=OUT):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ctx = common.load_context(root)
    start = time.monotonic()
    predictions = []
    fold_results = []
    with threadpool_limits(limits=THREADS):
        for outer in ctx["outer_folds"]:
            fold_no, fit_rows, held_rows = outer["fold"], outer["fit_rows"], outer["held_rows"]
            print(f"E16 outer fold {fold_no}/5: tune link/penalty", flush=True)
            tuning_records = _inner_records(ctx, fit_rows, output=output)
            complete_params = _choose_params(tuning_records, "complete")
            simulated_params = _choose_params(tuning_records, "simulated")
            pair_model = fit_pair_model(ctx["texts"][fit_rows], ctx["groups"][fit_rows], c=1.0)
            joblib.dump(pair_model, output / f"E16_pair_fold{fold_no}.joblib", compress=3)
            # Inference signature is ids + raw text + E13 scores; prompt_id is not passed.
            edges = predict_pair_probabilities(pair_model, ctx["texts"][held_rows])
            held_scores = outer["label_scores"]
            held_base = outer["baseline"]
            complete_pred, complete_graph, _ = _score_setting(
                ctx["y"][held_rows], held_base, held_scores, edges,
                ctx["row_ids"][held_rows], complete_params["threshold"], complete_params["penalty"])
            complete_graph_metrics = _graph_metrics(edges, ctx["groups"][held_rows], complete_params["threshold"])
            predictions.extend({"setting": "complete", "fold": fold_no,
                "id": ctx["row_ids"][row], "prompt_id": ctx["groups"][row],
                "y_true": LABELS[ctx["y"][row]], "E13_B": LABELS[held_base[i]],
                "E16": LABELS[complete_pred[i]], "changed": bool(complete_pred[i] != held_base[i]),
                "threshold": complete_params["threshold"], "penalty": complete_params["penalty"],
                "active": bool(complete_graph["active_nodes"][i])}
                for i, row in enumerate(held_rows))
            sim_rows = []
            sim_batches = []
            for rep in range(SIMULATION_REPEATS):
                chosen = _sample_hidden(held_rows, ctx["groups"], SEED + fold_no * 1000 + rep)
                take = np.asarray([{int(row): i for i, row in enumerate(held_rows)}[int(x)] for x in chosen])
                edge_sub = _filter_edges(edges, held_rows, chosen)
                pred, graph, _ = _score_setting(
                    ctx["y"][chosen], held_base[take], held_scores[take], edge_sub,
                    ctx["row_ids"][chosen], simulated_params["threshold"], simulated_params["penalty"])
                graph_m = _graph_metrics(edge_sub, ctx["groups"][chosen], simulated_params["threshold"])
                sim_batches.append({"repeat": rep, "seed": SEED + fold_no * 1000 + rep,
                                    "rows": int(len(chosen)), "pair_graph": graph_m,
                                    "activation_rate": float(graph["active_nodes"].mean()),
                                    "active_edges": graph["active_edges"]})
                sim_rows.extend({"setting": "simulated", "fold": fold_no, "repeat": rep,
                    "id": ctx["row_ids"][row], "prompt_id": ctx["groups"][row],
                    "y_true": LABELS[ctx["y"][row]], "E13_B": LABELS[held_base[take[i]]],
                    "E16": LABELS[pred[i]], "changed": bool(pred[i] != held_base[take[i]]),
                    "threshold": simulated_params["threshold"], "penalty": simulated_params["penalty"],
                    "active": bool(graph["active_nodes"][i])}
                    for i, row in enumerate(chosen))
            predictions.extend(sim_rows)
            # The exact training-to-test setting is reported separately for each mode.
            fold_results.append({"fold": fold_no, "fit_rows": int(len(fit_rows)),
                "held_rows": int(len(held_rows)), "complete_parameters": complete_params,
                "simulated_parameters": simulated_params,
                "complete": {**common.comparison(ctx["y"][held_rows], held_base, complete_pred),
                             "pair_graph": complete_graph_metrics,
                             "activation_rate": float(complete_graph["active_nodes"].mean()),
                             "active_edges": complete_graph["active_edges"]},
                "simulated_batches": sim_batches})
            print(f"E16 fold {fold_no}: complete BA={fold_results[-1]['complete']['balanced_accuracy']:.4%}; "
                  f"threshold={complete_params['threshold']}, penalty={complete_params['penalty']}", flush=True)
    pred_frame = pd.DataFrame(predictions)
    pred_frame.to_parquet(output / "predictions.parquet", index=False)
    summaries = {}
    for setting in ("complete", "simulated"):
        subset = pred_frame[pred_frame.setting == setting]
        y = pd.Categorical(subset.y_true, categories=LABELS).codes
        base = pd.Categorical(subset.E13_B, categories=LABELS).codes
        pred = pd.Categorical(subset.E16, categories=LABELS).codes
        if setting == "complete":
            graph_rows = [f["complete"]["pair_graph"] for f in fold_results]
        else:
            graph_rows = [batch["pair_graph"] for f in fold_results for batch in f["simulated_batches"]]
        tp=sum(x["tp"] for x in graph_rows); fp=sum(x["fp"] for x in graph_rows); fn=sum(x["fn"] for x in graph_rows)
        pair_graph={"pairs":sum(x["pairs"] for x in graph_rows),"predicted_edges":sum(x["predicted_edges"] for x in graph_rows),
                    "tp":tp,"fp":fp,"fn":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),
                    "f1":2*tp/max(1,2*tp+fp+fn)}
        summaries[setting] = {**common.comparison(y, base, pred),
                              "activation_rate": float(subset.active.mean()),
                              "correction_rate": float(subset.changed.mean()),
                              "evaluated_rows": int(len(subset)),
                              "unique_rows": int(subset.id.nunique()), "pair_graph": pair_graph}
    result = {"experiment": "E16", "baseline": "E13-B Trust List 2 v037",
              "validation_rows": len(ctx["v0"]), "checkpoint_ids": ctx["checkpoint_ids"],
              "outer_folds": fold_results, "complete": summaries["complete"],
              "simulated": summaries["simulated"],
              "runtime_seconds": time.monotonic()-start, "threads": THREADS,
              "prompt_id_use": "pair targets, held-out splits and hidden-batch sampling/evaluation only; absent from E16 inference signature",
              "baseline_reproduced_exactly": True,
              "simulation": {"repeats_per_outer_fold": SIMULATION_REPEATS,
                             "seeds": [SEED + fold*1000 + rep for fold in range(1,6) for rep in range(SIMULATION_REPEATS)],
                             "per_prompt_batch_size": "uniform integer 1..3, including singleton prompts"}}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output / "config.json").write_text(json.dumps({"seed":SEED,"threads":THREADS,
        "thresholds":PAIR_THRESHOLDS,"penalties":UNIQUENESS_PENALTIES,
        "inner_folds":INNER_FOLDS,"simulation_repeats":SIMULATION_REPEATS,
        "features":["char/word TF-IDF cosine","token Jaccard/overlap","length/paragraph/line/punctuation/capitalization/indentation structure"],
        "optimizer":"coordinate ascent maximizing E13 label-score sum minus probability-weighted duplicate-label penalties",
        "trust_list_snapshot":ctx["snapshot"],"checkpoint_cache_hashes":ctx["cache_hashes"]},indent=2)+"\n")
    _write_report(output, result)
    print(f"E16 complete: diagnostic BA {summaries['complete']['balanced_accuracy']:.4%}; "
          f"simulated BA {summaries['simulated']['balanced_accuracy']:.4%}", flush=True)
    return result


def _write_report(output, result):
    lines=["# E16 — Prompt Reconstruction and Soft Uniqueness","",
           "E16 predicts same-prompt links from response text, creates a high-confidence text-only graph, then decodes labels jointly while penalizing repeated labels on linked rows. It uses the saved E13-B five-fold predictions and Trust List 2 v037 cached checkpoint probabilities; no base classifier is trained or run.","",
           "V0 is a development evaluation because the E13 checkpoint inventory was selected using V0. `prompt_id` is used only to make positive/negative pair labels, split groups, sample hidden batches, and score graph edges. The E16 inference function accepts only row IDs, raw response text, E13 label scores, and its learned pair model.","",
           "## Results","",
           "| Setting | Rows scored | Unique rows | Accuracy | Balanced accuracy | Δ BA vs E13-B | Corrected | Introduced | Net gains | Switches | Activation | Pair precision | Pair recall |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for setting in ("complete","simulated"):
        m=result[setting]
        # Aggregate edge-level diagnostics are available in fold outputs.
        lines.append(f"| {setting} | {m['evaluated_rows']} | {m['unique_rows']} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['delta_balanced_accuracy']:+.4%} | {m['corrected']} | {m['introduced']} | {m['net_gains']} | {m['switches']} | {m['activation_rate']:.4%} | {m['pair_graph']['precision']:.4%} | {m['pair_graph']['recall']:.4%} |")
    lines += ["","Complete held-out prompts are the diagnostic graph case. Simulated batches randomly retain 1–3 examples for each prompt and mix them into one batch, so prompt IDs are not available to the decoder. Singletons remain present and cannot form edges.","",
              "## Fold-specific tuning and evaluation","","| Fold | Complete threshold | Complete penalty | Complete BA | Simulated threshold | Simulated penalty | Simulated evaluated rows |","|---:|---:|---:|---:|---:|---:|---:|"]
    for f in result["outer_folds"]:
        n=sum(x["rows"] for x in f["simulated_batches"])
        c=f["complete_parameters"]; s=f["simulated_parameters"]
        lines.append(f"| {f['fold']} | {c['threshold']:.2f} | {c['penalty']:.2f} | {f['complete']['balanced_accuracy']:.4%} | {s['threshold']:.2f} | {s['penalty']:.2f} | {n} |")
    lines += ["","## Method and safeguards","",
              "- Pair positives: V0 responses sharing `prompt_id`; negatives: nearest character-TF-IDF neighbors from different prompt groups.",
              "- Pair features: char/word TF-IDF cosine, lexical Jaccard/overlap, text-length and paragraph/line structure, punctuation, capitalization, and indentation similarities.",
              "- Pair vocabulary, pair classifier, E13 reliability router, thresholds, and penalties are fit using outer-training prompt groups only.",
              "- Soft objective: sum of E13 reliability label scores minus `penalty × edge_probability` for each linked pair assigned the same label. Deterministic coordinate ascent performs joint decoding.",
              "- Pair thresholds and uniqueness penalties are selected by three-fold prompt-group validation inside each outer fit.",
              "- E13-B is reproduced from saved fold models; E13-B remains unchanged.","",
              "## Artifacts","",f"- Report: `{output / 'e16.md'}`","- `predictions.parquet`: full-prompt and simulated row-level decisions.","- `E16_pair_fold*.joblib`: fold-local vectorizers, scaler and pair classifier.","- `config.json`, `results.json`: parameters, trust snapshot, fold metrics and simulation seeds.",""]
    report="\n".join(lines); (output/"e16.md").write_text(report); (ROOT/"docs/e16.md").write_text(report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--output",type=Path,default=OUT);a=parser.parse_args(argv);run(a.root,a.output);return 0


if __name__=="__main__":
    raise SystemExit(main())
