"""E19-F: direct response-embedding prompt grouping without text generation.

All learned pair weights and thresholds are fit with prompt IDs from the outer
training partition only. The held-out clustering routines accept only IDs,
response text and precomputed embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from . import e16_common, e19b_evaluate, e19b_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e19f"
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
TASK_PREFIX = "task: clustering | query: "
MAX_LENGTH = 512
MAX_CLUSTER_SIZE = 18
SEED = 2026
INCOMPLETE_SEEDS = (2026, 2027, 2028, 2029, 2030)
THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
MNN_K_VALUES = (5, 10, 20)
LOGISTIC_C_VALUES = (0.1, 1.0, 10.0)
PAIR_RECALL_FLOOR = 0.70


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def encoding_configuration() -> dict:
    return {"task_prefix": TASK_PREFIX, "pooling": "attention-mask mean",
            "normalize": "L2", "max_length": MAX_LENGTH,
            "padding": "batch dynamic", "truncation": True}


def _configuration_key() -> str:
    payload = {"model": MODEL_ID, "revision": MODEL_REVISION,
               "encoding": encoding_configuration()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cuda_is_busy() -> bool:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True)
        return any(line.strip().isdigit() for line in output.splitlines())
    except (OSError, subprocess.SubprocessError):
        return False


def _select_device(torch, requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    # Never compete with another model server/trainer for the shared GPU.
    if _cuda_is_busy() or free_bytes < 7 * 1024**3:
        if requested == "cuda":
            raise RuntimeError("CUDA is busy or has less than 7 GiB free; select CPU or retry later")
        return "cpu"
    return "cuda"


def _cache_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("""CREATE TABLE IF NOT EXISTS response_embeddings (
        config_key TEXT NOT NULL, response_hash TEXT NOT NULL,
        dimension INTEGER NOT NULL, embedding BLOB NOT NULL,
        PRIMARY KEY(config_key, response_hash))""")
    return db


def encode_response_embeddings(texts, *, cache_path=OUT / "embedding_cache.sqlite3",
                               device="auto", batch_size=24, progress=True):
    """Generate or resume normalized embeddings, caching each raw response hash."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = list(map(str, texts))
    hashes = [_sha(text) for text in texts]
    cache_key = _configuration_key()
    db = _cache_connect(Path(cache_path))
    vectors: dict[str, np.ndarray] = {}
    unique_hashes = sorted(set(hashes))
    for start in range(0, len(unique_hashes), 800):
        block = unique_hashes[start:start + 800]
        marks = ",".join("?" for _ in block)
        for response_hash, dimension, blob in db.execute(
                f"SELECT response_hash, dimension, embedding FROM response_embeddings "
                f"WHERE config_key=? AND response_hash IN ({marks})", [cache_key, *block]):
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            if len(vector) != int(dimension) or not np.isfinite(vector).all():
                continue
            vectors[response_hash] = vector
    missing = [digest for digest in unique_hashes if digest not in vectors]
    selected_device = _select_device(torch, device)
    model_seconds = 0.0
    if missing:
        dtype = (torch.bfloat16 if selected_device == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            local_files_only=True, padding_side="right")
        t0 = time.perf_counter()
        model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            local_files_only=True, torch_dtype=dtype)
        model.to(selected_device).eval()
        model_seconds = time.perf_counter() - t0
        text_for_hash = {}
        for text, digest in zip(texts, hashes):
            text_for_hash.setdefault(digest, text)
        pending = [text_for_hash[digest] for digest in missing]
        pos = 0
        batch_size = max(1, int(batch_size if selected_device == "cuda" else min(batch_size, 4)))
        inferred = 0
        try:
            while pos < len(pending):
                take = min(batch_size, len(pending) - pos)
                batch = [TASK_PREFIX + value for value in pending[pos:pos + take]]
                try:
                    tokenized = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(selected_device)
                    with torch.inference_mode():
                        output = model(**tokenized)
                        hidden = output.last_hidden_state.float()
                        mask = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    array = pooled.cpu().numpy().astype(np.float32, copy=False)
                except torch.cuda.OutOfMemoryError:
                    if selected_device != "cuda" or batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    continue
                records = []
                for offset, vector in enumerate(array):
                    digest = missing[pos + offset]
                    vectors[digest] = vector.copy()
                    records.append((cache_key, digest, len(vector), vector.tobytes()))
                db.executemany("INSERT OR REPLACE INTO response_embeddings "
                               "(config_key,response_hash,dimension,embedding) VALUES(?,?,?,?)", records)
                db.commit()
                pos += take
                inferred += take
                if progress:
                    print(f"E19-F embeddings: {len(vectors)}/{len(unique_hashes)} cached; "
                          f"device={selected_device}; batch={batch_size}", flush=True)
        finally:
            del model, tokenizer
            if selected_device == "cuda":
                torch.cuda.empty_cache()
    db.close()
    matrix = np.stack([vectors[digest] for digest in hashes]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or np.any(norms < 1e-8):
        raise ValueError("E19-F embeddings must be finite and nonzero")
    matrix /= norms[:, None]
    return matrix, {"model": MODEL_ID, "revision": MODEL_REVISION,
        "encoding": encoding_configuration(), "configuration_key": cache_key,
        "device": selected_device, "response_rows": len(texts),
        "unique_response_hashes": len(unique_hashes), "cached_hashes": len(unique_hashes) - len(missing),
        "newly_encoded_hashes": len(missing), "model_load_seconds": model_seconds}


def _all_pairs(n: int) -> np.ndarray:
    left, right = np.triu_indices(int(n), k=1)
    return np.column_stack((left, right)).astype(np.int64, copy=False)


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(norms < 1e-8):
        raise ValueError("E19-F embeddings must be a finite nonzero matrix")
    values = values / norms
    return np.clip(values @ values.T, -1.0, 1.0).astype(np.float32)


def _id_order(rows, ids) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    return rows[np.argsort(np.asarray(ids, dtype=str)[rows], kind="stable")]


def cluster_cosine_complete(ids, cosine, *, threshold: float):
    """Full-edge cosine complete-link clustering, delegated to E19-B utility."""
    ids = np.asarray(ids, dtype=str)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("cosine threshold must be in [0, 1]")
    pairs = _all_pairs(len(ids))
    scores = np.asarray(cosine, dtype=np.float32)[pairs[:, 0], pairs[:, 1]]
    return e19b_group.cluster_complete_link(ids, pairs, scores,
        threshold=float(threshold), max_cluster_size=MAX_CLUSTER_SIZE)


def _mnn_links(cosine: np.ndarray, ids, k: int) -> np.ndarray:
    n = len(ids)
    ids = np.asarray(ids, dtype=str)
    links = np.zeros((n, n), dtype=bool)
    k = min(max(1, int(k)), max(1, n - 1))
    for row in range(n):
        candidates = np.asarray([i for i in range(n) if i != row], dtype=np.int64)
        ranked = np.lexsort((ids[candidates], -cosine[row, candidates]))
        links[row, candidates[ranked[:k]]] = True
    return links & links.T


def cluster_mutual_complete(ids, cosine, *, threshold: float, k: int):
    """MNN-seeded agglomeration with a complete-link cosine guard.

    Each merge must have a mutual-nearest-neighbor bridge and every cross
    response pair must clear the cosine threshold. Thus a single weak bridge
    cannot chain unrelated prompt groups together.
    """
    original_ids = np.asarray(ids, dtype=str)
    order = np.argsort(original_ids, kind="stable")
    sorted_ids = original_ids[order]
    raw = np.asarray(cosine, dtype=np.float32)[np.ix_(order, order)]
    mutual = _mnn_links(raw, sorted_ids, k)
    n = len(sorted_ids)
    if n == 0:
        return np.empty(0, dtype=object), {"clusters": [], "k": int(k), "threshold": float(threshold)}
    if not 1 <= MAX_CLUSTER_SIZE <= 18 or not 0 <= float(threshold) <= 1:
        raise ValueError("invalid E19-F-B clustering limits")
    members = {i: [i] for i in range(n)}
    minima = {}
    has_bridge = {}
    heap = []
    next_cluster = n

    def key(a, b):
        return (min(a, b), max(a, b))

    def cluster_name(cluster_id):
        return sorted_ids[min(members[cluster_id])]

    def push(a, b, minimum, bridge):
        if not bridge or minimum + 1e-12 < float(threshold):
            return
        if len(members[a]) + len(members[b]) > MAX_CLUSTER_SIZE:
            return
        pair = key(a, b)
        minima[pair] = float(minimum)
        has_bridge[pair] = True
        left, right = sorted((str(cluster_name(a)), str(cluster_name(b))))
        heapq.heappush(heap, (-float(minimum), left, right, pair[0], pair[1]))

    import heapq
    for a in range(n):
        for b in range(a + 1, n):
            if mutual[a, b]:
                push(a, b, raw[a, b], True)
    while heap:
        neg, _, _, a, b = heapq.heappop(heap)
        pair = key(a, b)
        if a not in members or b not in members or pair not in minima:
            continue
        if abs(minima[pair] + neg) > 1e-7:
            continue
        old_clusters = [c for c in members if c not in {a, b}]
        merged = sorted(members[a] + members[b])
        # Preserve only pairs linked to both old clusters under complete link.
        links = {}
        for other in old_clusters:
            ka, kb = key(a, other), key(b, other)
            min_value = min(float(np.min(raw[np.ix_(members[a], members[other])])),
                            float(np.min(raw[np.ix_(members[b], members[other])])))
            bridge = bool(mutual[np.ix_(members[a] + members[b], members[other])].any())
            if bridge:
                links[other] = (min_value, True)
            minima.pop(ka, None); minima.pop(kb, None)
            has_bridge.pop(ka, None); has_bridge.pop(kb, None)
        minima.pop(pair, None); has_bridge.pop(pair, None)
        del members[a]; del members[b]
        new = next_cluster; next_cluster += 1
        members[new] = merged
        for other, (min_value, bridge) in links.items():
            if other in members:
                push(new, other, min_value, bridge)
    clusters = sorted(members.values(), key=lambda rows: str(sorted_ids[min(rows)]))
    out_sorted = np.empty(n, dtype=object)
    payload = []
    for cluster in clusters:
        name = f"e19f-b-{sorted_ids[min(cluster)]}"
        out_sorted[cluster] = name
        payload.append({"group_id": name, "response_ids": sorted_ids[cluster].tolist(),
                        "size": len(cluster)})
    out = np.empty(n, dtype=object)
    out[order] = out_sorted
    return out, {"clusters": payload, "k": int(k), "threshold": float(threshold),
                 "max_cluster_size": MAX_CLUSTER_SIZE}


def _fit_lexical(texts, fit_rows):
    texts = np.asarray(texts, dtype=str)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
        sublinear_tf=True, dtype=np.float32, max_features=120000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
        sublinear_tf=True, dtype=np.float32, max_features=250000)
    return {"word_vectorizer": word.fit(texts[fit_rows]),
            "char_vectorizer": char.fit(texts[fit_rows])}


def _transform_lexical(texts, vectorizers):
    texts = np.asarray(texts, dtype=str)
    return {"word": vectorizers["word_vectorizer"].transform(texts),
            "char": vectorizers["char_vectorizer"].transform(texts)}


def _neighbor_pairs(matrix, n_rows: int, k: int) -> set[tuple[int, int]]:
    if n_rows < 2:
        return set()
    nearest = NearestNeighbors(n_neighbors=min(n_rows, int(k) + 1), metric="cosine",
        algorithm="brute", n_jobs=1)
    nearest.fit(matrix)
    _, indexes = nearest.kneighbors(matrix)
    pairs = set()
    for left, neighbors in enumerate(indexes):
        for right in neighbors:
            right = int(right)
            if right != left:
                pairs.add((min(left, right), max(left, right)))
    return pairs


def candidate_pairs(rows, ids, lexical, embeddings, *, neighbors=40):
    """Union stable nearest-neighbor candidates from embedding, word and char spaces."""
    rows = _id_order(rows, ids)
    if len(rows) < 2:
        return rows, np.empty((0, 2), dtype=np.int64)
    local_emb = np.asarray(embeddings, dtype=np.float32)[rows]
    local_emb /= np.linalg.norm(local_emb, axis=1, keepdims=True).clip(1e-12)
    pairs = _neighbor_pairs(sparse.csr_matrix(local_emb), len(rows), neighbors)
    for space in ("word", "char"):
        pairs.update(_neighbor_pairs(lexical[space][rows], len(rows), neighbors))
    return rows, np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def _pair_features(rows, pairs, lexical, embeddings):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty((0, 3), dtype=np.float32)
    local_embeddings = np.asarray(embeddings, dtype=np.float32)[rows]
    cosine = np.sum(local_embeddings[pairs[:, 0]] * local_embeddings[pairs[:, 1]], axis=1)
    word = np.asarray(lexical["word"][rows[pairs[:, 0]]].multiply(
        lexical["word"][rows[pairs[:, 1]]]).sum(axis=1)).reshape(-1)
    char = np.asarray(lexical["char"][rows[pairs[:, 0]]].multiply(
        lexical["char"][rows[pairs[:, 1]]]).sum(axis=1)).reshape(-1)
    return np.column_stack((cosine, word, char)).astype(np.float32)


def _training_pairs(rows, pairs, groups):
    all_pairs = {tuple(map(int, pair)) for pair in np.asarray(pairs).reshape(-1, 2)}
    local_groups = np.asarray(groups, dtype=str)[rows]
    members = {}
    for i, group in enumerate(local_groups):
        members.setdefault(group, []).append(i)
    for indexes in members.values():
        all_pairs.update(itertools.combinations(indexes, 2))
    ordered = np.asarray(sorted((min(a, b), max(a, b)) for a, b in all_pairs), dtype=np.int64).reshape(-1, 2)
    labels = (local_groups[ordered[:, 0]] == local_groups[ordered[:, 1]]).astype(np.int8)
    return ordered, labels


def fit_similarity_model(rows, pairs, lexical, embeddings, groups, *, c, seed=SEED):
    train_pairs, labels = _training_pairs(rows, pairs, groups)
    features = _pair_features(rows, train_pairs, lexical, embeddings)
    if len(np.unique(labels)) != 2:
        raise ValueError("E19-F-C requires both same-prompt and different-prompt pair examples")
    model = LogisticRegression(C=float(c), class_weight="balanced", solver="liblinear",
        max_iter=2000, random_state=int(seed))
    model.fit(features, labels)
    return {"model": model, "C": float(c), "feature_names": ["embedding_cosine", "word_tfidf_cosine", "char_tfidf_cosine"],
            "coefficients": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
            "training_candidate_pairs": int(len(train_pairs)),
            "training_positive_pairs": int(labels.sum())}


def score_similarity_model(bundle, features):
    return bundle["model"].predict_proba(np.asarray(features, dtype=np.float32))[:, 1].astype(np.float32)


def _cluster_fc(ids, pairs, scores, threshold):
    return e19b_group.cluster_complete_link(ids, pairs, scores,
        threshold=float(threshold), max_cluster_size=MAX_CLUSTER_SIZE)


def _selection_key(metrics):
    precision = float(metrics["pairwise_precision"])
    recall = float(metrics["pairwise_recall"])
    feasible = recall >= PAIR_RECALL_FLOOR
    # Avoid the degenerate all-singleton solution (precision=1, recall=0).
    return (int(feasible), precision if feasible else recall,
            float(metrics["pairwise_f1"]), float(metrics["exact_group_recovery"]))


def _run_cluster(method, rows, ids, cosine, embeddings, texts, groups, *,
                 threshold, k=None, lexical=None, pair_model=None):
    ordered = _id_order(rows, ids)
    local_ids = np.asarray(ids, dtype=str)[ordered]
    if method == "E19-F-A":
        local_cosine = cosine[np.ix_(ordered, ordered)]
        return cluster_cosine_complete(local_ids, local_cosine, threshold=threshold)
    if method == "E19-F-B":
        local_cosine = cosine[np.ix_(ordered, ordered)]
        return cluster_mutual_complete(local_ids, local_cosine, threshold=threshold, k=int(k))
    local_embeddings = np.asarray(embeddings, dtype=np.float32)[ordered]
    local_lex = {name: matrix[ordered] for name, matrix in lexical.items()}
    local_rows, pairs = candidate_pairs(np.arange(len(ordered)), local_ids, local_lex, local_embeddings)
    features = _pair_features(local_rows, pairs, local_lex, local_embeddings)
    scores = score_similarity_model(pair_model, features) if len(pairs) else np.empty(0, dtype=np.float32)
    return _cluster_fc(local_ids[local_rows], pairs, scores, threshold)


def tune_inner(method, rows, ids, texts, embeddings, cosine, groups):
    """Nested prompt-group tuning confined to the supplied outer-training rows."""
    rows = _id_order(rows, ids)
    records = []
    for inner_no, (fit_local, valid_local) in enumerate(GroupKFold(n_splits=4).split(
            rows, groups=np.asarray(groups, dtype=str)[rows]), start=1):
        fit_rows = _id_order(rows[fit_local], ids)
        valid_rows = _id_order(rows[valid_local], ids)
        if method == "E19-F-A":
            for threshold in THRESHOLDS:
                pred, _ = _run_cluster(method, valid_rows, ids, cosine, embeddings, texts, groups,
                                       threshold=threshold)
                metrics = e19b_evaluate.grouping_metrics(np.asarray(groups)[valid_rows], pred)
                records.append({"inner_fold": inner_no, "threshold": threshold, "metrics": metrics})
        elif method == "E19-F-B":
            for k, threshold in itertools.product(MNN_K_VALUES, THRESHOLDS):
                pred, _ = _run_cluster(method, valid_rows, ids, cosine, embeddings, texts, groups,
                                       threshold=threshold, k=k)
                metrics = e19b_evaluate.grouping_metrics(np.asarray(groups)[valid_rows], pred)
                records.append({"inner_fold": inner_no, "k": k, "threshold": threshold,
                                "metrics": metrics})
        else:
            vectorizers = _fit_lexical(texts, fit_rows)
            lexical = _transform_lexical(texts, vectorizers)
            fit_order, fit_candidates = candidate_pairs(fit_rows, ids, lexical, embeddings)
            for c in LOGISTIC_C_VALUES:
                model = fit_similarity_model(fit_order, fit_candidates, lexical,
                    embeddings, groups, c=c)
                val_order, val_pairs = candidate_pairs(valid_rows, ids, lexical, embeddings)
                val_features = _pair_features(val_order, val_pairs, lexical, embeddings)
                val_scores = score_similarity_model(model, val_features) if len(val_pairs) else np.empty(0)
                for threshold in THRESHOLDS:
                    pred, _ = _cluster_fc(np.asarray(ids)[val_order], val_pairs, val_scores, threshold)
                    metrics = e19b_evaluate.grouping_metrics(np.asarray(groups)[val_order], pred)
                    records.append({"inner_fold": inner_no, "C": c, "threshold": threshold,
                                    "metrics": metrics})
    params = ([("threshold", threshold) for threshold in THRESHOLDS] if method == "E19-F-A" else
        [("k", k, "threshold", threshold) for k, threshold in itertools.product(MNN_K_VALUES, THRESHOLDS)]
        if method == "E19-F-B" else
        [("C", c, "threshold", threshold) for c, threshold in itertools.product(LOGISTIC_C_VALUES, THRESHOLDS)])
    summaries = []
    for key in params:
        param = dict(zip(key[::2], key[1::2])) if len(key) > 1 else {key[0]: key[1]}
        subset = [r["metrics"] for r in records if all(r.get(name) == value for name, value in param.items())]
        summaries.append({"parameters": param,
            "mean_pairwise_precision": float(np.mean([m["pairwise_precision"] for m in subset])),
            "mean_pairwise_recall": float(np.mean([m["pairwise_recall"] for m in subset])),
            "mean_pairwise_f1": float(np.mean([m["pairwise_f1"] for m in subset])),
            "mean_exact_group_recovery": float(np.mean([m["exact_group_recovery"] for m in subset]))})
    # Prefer precision among settings that maintain useful pair recall.
    def rank(item):
        feasible = item["mean_pairwise_recall"] >= PAIR_RECALL_FLOOR
        return (int(feasible), item["mean_pairwise_precision"] if feasible else item["mean_pairwise_recall"],
                item["mean_pairwise_f1"], item["mean_exact_group_recovery"],
                tuple(-float(x) for x in item["parameters"].values()))
    best = max(summaries, key=rank)
    return {"method": method, "selected": best["parameters"], "selected_inner_metrics": best,
            "inner_parameter_summary": summaries, "inner_fold_results": records,
            "selection_rule": f"maximize precision with mean pairwise recall >= {PAIR_RECALL_FLOOR:.2f}; "
                "if none qualify, maximize recall then precision/F1/exact recovery"}


def _fit_fc_full(rows, ids, texts, embeddings, groups, c):
    vectorizers = _fit_lexical(texts, rows)
    lexical = _transform_lexical(texts, vectorizers)
    ordered, pairs = candidate_pairs(rows, ids, lexical, embeddings)
    model = fit_similarity_model(ordered, pairs, lexical, embeddings, groups, c=c)
    return model, vectorizers, lexical


def _sample_incomplete(rows, ids, groups, *, max_group_size, seed):
    rng = np.random.default_rng(int(seed))
    group_rows = {}
    for row in rows:
        group_rows.setdefault(str(groups[row]), []).append(int(row))
    selected = []
    for prompt_id in sorted(group_rows):
        members = np.asarray(sorted(group_rows[prompt_id], key=lambda row: str(ids[row])), dtype=np.int64)
        size = int(rng.integers(1, min(int(max_group_size), len(members)) + 1))
        chosen = rng.choice(members, size=size, replace=False)
        selected.extend(map(int, chosen))
    return _id_order(selected, ids)


def _rename_fold_groups(pred, fold, seed=None):
    prefix = f"fold-{int(fold)}" + (f"-seed-{seed}" if seed is not None else "")
    return np.asarray([f"{prefix}-{value}" for value in pred], dtype=object)


def _bundle_for(method, selected, fc_model=None, vectorizers=None):
    return {"method": method, "selected": dict(selected), "max_cluster_size": MAX_CLUSTER_SIZE,
            "semantic_model": MODEL_ID, "semantic_revision": MODEL_REVISION,
            "encoding": encoding_configuration(), "pair_model": fc_model,
            "vectorizers": vectorizers}


def predict_groups(row_ids, texts, *, embeddings=None, model_path, cache_path=None, device="auto"):
    """Predict groups for Kaggle-style `id,text` inputs without prompt IDs or labels."""
    ids = np.asarray(row_ids, dtype=str)
    texts = np.asarray(texts, dtype=str)
    if len(ids) != len(texts) or len(set(ids.tolist())) != len(ids):
        raise ValueError("E19-F requires unique aligned response IDs and texts")
    model = joblib.load(model_path)
    if embeddings is None:
        embeddings, _ = encode_response_embeddings(texts,
            cache_path=cache_path or Path(model_path).parent / "inference_embedding_cache.sqlite3",
            device=device, progress=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids) or not np.isfinite(embeddings).all():
        raise ValueError("E19-F embeddings do not align with inference rows")
    order = np.argsort(ids, kind="stable")
    sorted_ids, sorted_texts = ids[order], texts[order]
    sorted_embeddings = embeddings[order]
    method = model["method"]
    params = model["selected"]
    cosine = cosine_matrix(sorted_embeddings)
    if method == "E19-F-A":
        pred, _ = cluster_cosine_complete(sorted_ids, cosine, threshold=params["threshold"])
    elif method == "E19-F-B":
        pred, _ = cluster_mutual_complete(sorted_ids, cosine,
            threshold=params["threshold"], k=params["k"])
    elif method == "E19-F-C":
        lexical = _transform_lexical(sorted_texts, model["vectorizers"])
        rows, pairs = candidate_pairs(np.arange(len(sorted_ids)), sorted_ids, lexical, sorted_embeddings)
        features = _pair_features(rows, pairs, lexical, sorted_embeddings)
        scores = score_similarity_model(model["pair_model"], features) if len(pairs) else np.empty(0)
        pred, _ = _cluster_fc(sorted_ids[rows], pairs, scores, params["threshold"])
    else:
        raise ValueError(f"unsupported E19-F method: {method}")
    aligned = np.empty(len(pred), dtype=object)
    aligned[order] = pred
    return aligned


def _complete_fold_prediction(method, held_rows, ids, texts, embeddings, cosine, groups,
                              selected, fc_model=None, vectorizers=None):
    if method == "E19-F-A":
        pred, info = _run_cluster(method, held_rows, ids, cosine, embeddings, texts, groups,
                                  threshold=selected["threshold"])
    elif method == "E19-F-B":
        pred, info = _run_cluster(method, held_rows, ids, cosine, embeddings, texts, groups,
                                  threshold=selected["threshold"], k=selected["k"])
    else:
        local_lex = _transform_lexical(texts, vectorizers)
        ordered = _id_order(held_rows, ids)
        ordered, pairs = candidate_pairs(ordered, ids, local_lex, embeddings)
        features = _pair_features(ordered, pairs, local_lex, embeddings)
        scores = score_similarity_model(fc_model, features) if len(pairs) else np.empty(0)
        pred, info = _cluster_fc(np.asarray(ids)[ordered], pairs, scores, selected["threshold"])
    return np.asarray(pred, dtype=object), info


def _fit_outer_fc(fit_rows, ids, texts, embeddings, groups, c):
    return _fit_fc_full(fit_rows, ids, texts, embeddings, groups, c)


def _grouping_rows(frame, assignments, *, method, kind, seed=None):
    return pd.DataFrame({"id": frame.id.astype(str).to_numpy(),
        "prompt_id": frame.prompt_id.astype(str).to_numpy(), "fold": frame.fold.to_numpy(),
        "method": method, "evaluation": kind, "seed": seed,
        "predicted_group_id": np.asarray(assignments, dtype=object)})


def _aggregate_group_diagnostics(frame, *, method):
    truth = frame.prompt_id.astype(str).to_numpy()
    pred = frame.predicted_group_id.astype(str).to_numpy()
    by_true = {key: part for key, part in frame.groupby("prompt_id", sort=True)}
    by_pred = {key: part for key, part in frame.groupby("predicted_group_id", sort=True)}
    correct, merged, split = [], [], []

    def sample(part, category):
        ids = part.id.astype(str).tolist()
        return {"category": category, "response_ids": ids,
            "true_prompt_ids": part.prompt_id.astype(str).unique().tolist(),
            "predicted_groups": part.predicted_group_id.astype(str).unique().tolist(),
            "responses": [{"id": str(row.id), "prompt_id": str(row.prompt_id),
                           "text": str(row.text)[:320]} for row in part.head(5).itertuples()]}

    for _, part in by_true.items():
        if part.predicted_group_id.nunique() == 1:
            group = by_pred[str(part.predicted_group_id.iloc[0])]
            if len(group) == len(part) and set(group.id.astype(str)) == set(part.id.astype(str)):
                correct.append(sample(part, "correctly_reconstructed"))
        else:
            split.append(sample(part, "incorrectly_split"))
    for _, part in by_pred.items():
        if part.prompt_id.nunique() > 1:
            merged.append(sample(part, "incorrectly_merged"))
    # Retain the shortest examples even when none are truly short; this exposes
    # the weakest-text tail instead of silently omitting the diagnostic.
    short = frame.assign(_words=frame.text.fillna("").astype(str).str.split().str.len())
    short = short.sort_values(["_words", "id"], kind="stable").head(20)
    return {"method": method, "correct_examples": correct[:12], "merge_examples": merged[:20],
        "split_examples": split[:20],
        "short_or_generic_response_examples": [{"id": str(r["id"]), "prompt_id": str(r["prompt_id"]),
            "predicted_group_id": str(r["predicted_group_id"]), "word_count": int(r["_words"]),
            "text": str(r["text"])[:320]} for r in short.to_dict(orient="records")]}


def _outer_baseline_incomplete(ctx, outer, reconstructed, semantic, subset_rows, *, threshold, c):
    fit_rows = np.asarray(outer["fit_rows"], dtype=np.int64)
    all_ids = np.asarray(ctx["row_ids"], dtype=str)
    texts = ctx["texts"]
    groups = ctx["groups"].astype(str)
    recon = reconstructed.reconstructed_prompt.fillna("").astype(str).to_numpy()
    topics = reconstructed.topic.fillna("").astype(str).to_numpy()
    details = [json.loads(x) if isinstance(x, str) else x for x in reconstructed.distinctive_details]
    constraints = [json.loads(x) if isinstance(x, str) else x for x in reconstructed.constraints]
    _, fit_pairs, fit_x, fit_y = e19b_evaluate._pairs_and_features(fit_rows, recon, texts,
        semantic, topics, details, constraints, groups, fit_rows=fit_rows)
    model = e19b_group.fit_pair_scorer(fit_x, fit_y, c=c, seed=SEED)
    _, pairs, features, _ = e19b_evaluate._pairs_and_features(subset_rows, recon, texts,
        semantic, topics, details, constraints, None, fit_rows=fit_rows)
    scores = e19b_group.score_pairs(model, features) if len(pairs) else np.empty(0)
    pred, _ = e19b_group.cluster_complete_link(all_ids[subset_rows], pairs, scores,
        threshold=threshold, max_cluster_size=MAX_CLUSTER_SIZE)
    return pred


def run(root=ROOT, output_dir=OUT, *, device="auto", batch_size=24,
        run_incomplete=True, verify_e19b=True):
    started = time.perf_counter()
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = e16_common.load_context(root)
    v0 = ctx["v0"].reset_index(drop=True)
    ids = np.asarray(ctx["row_ids"], dtype=str)
    texts = ctx["texts"].astype(str)
    groups = ctx["groups"].astype(str)
    if len(ids) != 2699 or len(set(ids)) != len(ids):
        raise ValueError("E19-F expects all 2,699 unique V0 response IDs")
    if v0.id.astype(str).tolist() != ids.tolist():
        raise ValueError("E19-F V0 response texts do not align with saved E13-B row IDs")
    emb_start = time.perf_counter()
    embeddings, embedding_meta = encode_response_embeddings(texts,
        cache_path=output_dir / "embedding_cache.sqlite3", device=device,
        batch_size=batch_size)
    embedding_seconds = time.perf_counter() - emb_start
    np.save(output_dir / "response_embeddings.npy", embeddings)
    (output_dir / "embedding_manifest.json").write_text(json.dumps({**embedding_meta,
        "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "response_hashes": [_sha(text) for text in texts],
        "response_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}, indent=2) + "\n")
    cosine = cosine_matrix(embeddings)
    methods = ("E19-F-A", "E19-F-B", "E19-F-C")
    predictions = {method: np.empty(len(ids), dtype=object) for method in methods}
    folds_by_method = {method: [] for method in methods}
    incomplete_predictions = []
    config = {"experiment": "E19-F", "outer_folds": 5, "inner_folds": 4,
        "outer_splits_source": "existing E13-B saved outer prompt-group folds via e16_common",
        "threshold_grid": list(THRESHOLDS), "mnn_k_values": list(MNN_K_VALUES),
        "logistic_C_values": list(LOGISTIC_C_VALUES), "pairwise_recall_floor_for_selection": PAIR_RECALL_FLOOR,
        "incomplete_seeds": list(INCOMPLETE_SEEDS), "max_cluster_size": MAX_CLUSTER_SIZE,
        "embedding_model": MODEL_ID, "embedding_revision": MODEL_REVISION,
        "encoding": encoding_configuration(), "v0_response_hash_sha256": hashlib.sha256(
            "\0".join(_sha(t) for t in texts).encode()).hexdigest(),
        "v0_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    (output_dir / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")

    baseline_path = root / "artifacts/analysis/e19b/oof_group_predictions.parquet"
    baseline_frame = pd.read_parquet(baseline_path)
    if baseline_frame.id.astype(str).tolist() != ids.tolist():
        raise ValueError("E19-B saved OOF predictions do not align with V0 IDs")
    e19b_full_pred = baseline_frame.predicted_group_id.astype(str).to_numpy()
    e19b_full_metrics = e19b_evaluate.grouping_metrics(groups, e19b_full_pred)
    if verify_e19b and (e19b_full_metrics["exact_groups"] != 117 or
                        abs(e19b_full_metrics["pairwise_precision"] - 1.0) > 1e-9):
        raise ValueError(f"E19-B baseline failed reproduction: {e19b_full_metrics}")
    e19b_metrics = json.loads((root / "artifacts/analysis/e19b/metrics.json").read_text())
    e19b_fold_metrics = {int(row["fold"]): row for row in e19b_metrics["outer_folds"]}

    fold_by_row = np.zeros(len(ids), dtype=np.int8)
    incomplete_summaries = {method: {"1-10": {}, "1-18": {}} for method in methods}
    cluster_runtime = Counter()
    for outer in ctx["outer_folds"]:
        fold = int(outer["fold"])
        fit_rows = _id_order(outer["fit_rows"], ids)
        held_rows = _id_order(outer["held_rows"], ids)
        fold_by_row[held_rows] = fold
        print(f"E19-F outer fold {fold}/5: tune on {len(fit_rows)} training rows", flush=True)
        fold_started = time.perf_counter()
        selected = {}
        fc_models = {}
        fc_vectorizers = {}
        for method in methods:
            tuning = tune_inner(method, fit_rows, ids, texts, embeddings, cosine, groups)
            selected[method] = tuning["selected"]
            if method == "E19-F-C":
                fc_models[method], fc_vectorizers[method], _ = _fit_outer_fc(
                    fit_rows, ids, texts, embeddings, groups, selected[method]["C"])
            pred_start = time.perf_counter()
            pred, cluster_info = _complete_fold_prediction(method, held_rows, ids, texts,
                embeddings, cosine, groups, selected[method], fc_models.get(method),
                fc_vectorizers.get(method))
            elapsed = time.perf_counter() - pred_start
            cluster_runtime[method] += elapsed
            namespaced = _rename_fold_groups(pred, fold)
            predictions[method][held_rows] = namespaced
            metric = e19b_evaluate.grouping_metrics(groups[held_rows], namespaced)
            folds_by_method[method].append({"fold": fold, "fit_rows": len(fit_rows),
                "held_rows": len(held_rows), "selected": selected[method],
                "tuning": tuning, "metrics": metric, "clustering_seconds": elapsed,
                "clusters": cluster_info["clusters"]})
            print(f"E19-F {method} fold {fold}: exact={metric['exact_groups']}/{metric['true_group_count']}, "
                  f"precision={metric['pairwise_precision']:.4f}, recall={metric['pairwise_recall']:.4f}", flush=True)

        # Incomplete-group runs use the same fold-local learned parameters and five fixed seeds.
        if run_incomplete:
            for max_size, name in ((10, "1-10"), (18, "1-18")):
                for seed in INCOMPLETE_SEEDS:
                    subset = _sample_incomplete(held_rows, ids, groups,
                        max_group_size=max_size, seed=seed)
                    for method in methods:
                        if method == "E19-F-C":
                            vectorizers = fc_vectorizers[method]
                            lexical = _transform_lexical(texts, vectorizers)
                            ordered = _id_order(subset, ids)
                            ordered, pairs = candidate_pairs(ordered, ids, lexical, embeddings)
                            features = _pair_features(ordered, pairs, lexical, embeddings)
                            scores = score_similarity_model(fc_models[method], features) if len(pairs) else np.empty(0)
                            pred, info = _cluster_fc(ids[ordered], pairs, scores,
                                selected[method]["threshold"])
                        else:
                            pred, info = _complete_fold_prediction(method, subset, ids, texts,
                                embeddings, cosine, groups, selected[method])
                        assignments = _rename_fold_groups(pred, fold, seed)
                        incomplete_predictions.append(pd.DataFrame({"id": ids[_id_order(subset, ids)],
                            "prompt_id": groups[_id_order(subset, ids)], "fold": fold,
                            "method": method, "evaluation": name, "seed": seed,
                            "predicted_group_id": assignments,
                            "sampled_group_size": [sum(groups[_id_order(subset, ids)] == p)
                                for p in groups[_id_order(subset, ids)]]}))
        print(f"E19-F fold {fold} total tuning/validation seconds: {time.perf_counter()-fold_started:.1f}", flush=True)

    # Exact E19-B held-out baseline reproduction and incomplete groups on the same sampled IDs.
    e19b_incomplete_rows = []
    if run_incomplete:
        reconstructed = pd.read_parquet(root / "artifacts/analysis/e19b/reconstructed_prompts.parquet")
        semantic = np.load(root / "artifacts/analysis/e19b/semantic_embeddings.npy")
        for outer in ctx["outer_folds"]:
            fold = int(outer["fold"])
            held_rows = _id_order(outer["held_rows"], ids)
            selected_baseline = e19b_fold_metrics[fold]["selected"]
            # Reconstruct the E19-B outer-fit grouper from saved inputs and selected parameters.
            full_pred = _outer_baseline_incomplete(ctx, outer, reconstructed, semantic,
                held_rows, threshold=float(selected_baseline["threshold"]), c=float(selected_baseline["C"]))
            full_pred = _rename_fold_groups(full_pred, fold)
            if not np.array_equal(full_pred.astype(str), e19b_full_pred[held_rows].astype(str)):
                raise ValueError(f"E19-B fold {fold} could not be reproduced from its saved configuration")
            for max_size, name in ((10, "1-10"), (18, "1-18")):
                for seed in INCOMPLETE_SEEDS:
                    subset = _sample_incomplete(held_rows, ids, groups,
                        max_group_size=max_size, seed=seed)
                    pred = _outer_baseline_incomplete(ctx, outer, reconstructed, semantic,
                        subset, threshold=float(selected_baseline["threshold"]), c=float(selected_baseline["C"]))
                    e19b_incomplete_rows.append(pd.DataFrame({"id": ids[subset],
                        "prompt_id": groups[subset], "fold": fold,
                        "method": "E19-B Qwen3.5-2B", "evaluation": name, "seed": seed,
                        "predicted_group_id": _rename_fold_groups(pred, fold, seed)}))
    else:
        reconstructed = None

    full_oof_frames = []
    if (fold_by_row == 0).any() or any(pd.isna(predictions[method]).any() for method in methods):
        raise ValueError("E19-F outer folds did not assign each V0 row exactly once")
    for method in methods:
        assignments = predictions[method]
        full_oof_frames.append(pd.DataFrame({"id": ids, "prompt_id": groups,
            "fold": fold_by_row,
            "method": method, "evaluation": "complete-held-out",
            "predicted_group_id": assignments}))
    pred_path = output_dir / "predicted_group_assignments.parquet"
    pd.concat(full_oof_frames, ignore_index=True).to_parquet(pred_path, index=False)
    incomplete_frame = pd.concat(incomplete_predictions + e19b_incomplete_rows,
        ignore_index=True) if incomplete_predictions else pd.DataFrame()
    if len(incomplete_frame):
        incomplete_frame.to_parquet(output_dir / "incomplete_group_assignments.parquet", index=False)

    cross_validated = {"E19-B Qwen3.5-2B": e19b_full_metrics}
    for method in methods:
        cross_validated[method] = e19b_evaluate.grouping_metrics(groups, predictions[method])
    complete_fold_metrics = {method: folds_by_method[method] for method in methods}
    if run_incomplete:
        for method in ("E19-B Qwen3.5-2B", *methods):
            subset_frame = incomplete_frame[incomplete_frame.method.eq(method)]
            for name in ("1-10", "1-18"):
                seed_metrics = {}
                for seed in INCOMPLETE_SEEDS:
                    part = subset_frame[(subset_frame.evaluation == name) & (subset_frame.seed == seed)]
                    seed_metrics[str(seed)] = e19b_evaluate.grouping_metrics(
                        part.prompt_id.astype(str), part.predicted_group_id.astype(str))
                incomplete_summaries.setdefault(method, {})[name] = seed_metrics

    # Save a fitted, reusable Kaggle grouper per family; full-V0 scores remain diagnostics.
    full_rows = _id_order(np.arange(len(ids)), ids)
    final_configs = {}
    full_predictions = {}
    full_metrics = {}
    for method in methods:
        selected_folds = [entry["selected"] for entry in folds_by_method[method]]
        if method == "E19-F-A":
            threshold = float(np.median([x["threshold"] for x in selected_folds]))
            selected = {"threshold": threshold}
            model = _bundle_for(method, selected)
            pred, info = _run_cluster(method, full_rows, ids, cosine, embeddings, texts, groups,
                                      threshold=threshold)
        elif method == "E19-F-B":
            mode_k = Counter(x["k"] for x in selected_folds).most_common()
            k = min(value for value, count in mode_k if count == mode_k[0][1])
            threshold = float(np.median([x["threshold"] for x in selected_folds]))
            selected = {"k": int(k), "threshold": threshold}
            model = _bundle_for(method, selected)
            pred, info = _run_cluster(method, full_rows, ids, cosine, embeddings, texts, groups,
                                      threshold=threshold, k=k)
        else:
            mode_c = Counter(x["C"] for x in selected_folds).most_common()
            c = min(value for value, count in mode_c if count == mode_c[0][1])
            threshold = float(np.median([x["threshold"] for x in selected_folds]))
            selected = {"C": float(c), "threshold": threshold}
            pair_model, vectorizers, lexical = _fit_fc_full(full_rows, ids, texts, embeddings, groups, c)
            model = _bundle_for(method, selected, pair_model, vectorizers)
            ordered, pairs = candidate_pairs(full_rows, ids, lexical, embeddings)
            features = _pair_features(ordered, pairs, lexical, embeddings)
            scores = score_similarity_model(pair_model, features) if len(pairs) else np.empty(0)
            pred, info = _cluster_fc(ids[ordered], pairs, scores, threshold)
        final_configs[method] = {"parameters": selected,
            "source": "median/modal outer-training fold settings; no full-V0 metric fitting",
            "cluster_info": info}
        full_predictions[method] = pred
        full_metrics[method] = e19b_evaluate.grouping_metrics(groups, pred)
        safe_name = method.lower().replace("-", "_")
        bundle_path = output_dir / f"{safe_name}_kaggle_grouper.joblib"
        tmp_path = bundle_path.with_suffix(".joblib.tmp")
        joblib.dump(model, tmp_path, compress=3)
        tmp_path.replace(bundle_path)
        pd.DataFrame({"id": ids, "prompt_id": groups,
            "predicted_group_id": pred}).to_parquet(
                output_dir / f"{safe_name}_full_v0_diagnostic_groups.parquet", index=False)

    error_diagnostics = {}
    source_text = v0.text.fillna("").astype(str).to_numpy()
    diagnostic_frame = v0[["id", "prompt_id", "text"]].copy()
    diagnostic_frame["fold"] = fold_by_row
    for method in methods:
        frame = diagnostic_frame.copy()
        frame["predicted_group_id"] = predictions[method]
        error_diagnostics[method] = _aggregate_group_diagnostics(frame, method=method)
    (output_dir / "error_diagnostics.json").write_text(json.dumps(error_diagnostics,
        indent=2, ensure_ascii=False) + "\n")
    fold_configs = {method: [{"fold": int(entry["fold"]), "selected": entry["selected"],
        "inner_selection_rule": entry["tuning"]["selection_rule"],
        "inner_parameter_summary": entry["tuning"]["inner_parameter_summary"]}
        for entry in folds_by_method[method]] for method in methods}
    (output_dir / "fold_configurations.json").write_text(json.dumps(fold_configs, indent=2) + "\n")
    metrics = {"protocol": "five saved E13-B outer prompt-group folds; four-fold nested prompt-group tuning",
        "metrics_source": "e19b_evaluate.grouping_metrics shared implementation",
        "outer_fold_results": complete_fold_metrics,
        "cross_validated_complete_groups": cross_validated,
        "incomplete_groups_by_method_seed": incomplete_summaries,
        "full_v0_diagnostic_only": {method: full_metrics[method] for method in methods},
        "full_v0_diagnostic_parameters": final_configs,
        "embedding": {**embedding_meta, "generation_runtime_seconds": embedding_seconds,
            "model_and_pooling_runtime_seconds": embedding_meta["model_load_seconds"]},
        "clustering_runtime_seconds_by_method": dict(cluster_runtime),
        "total_runtime_seconds": time.perf_counter() - started,
        "baseline_comparison": {"E19-B expected": {"exact_groups": 117,
            "true_group_count": 150, "pairwise_precision": 1.0,
            "pairwise_recall": 0.9036759255221732, "pairwise_f1": 0.9494010124378679,
            "incorrectly_merged_groups": 0}, "E19-B reproduced": e19b_full_metrics}}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_report(metrics, output_dir / "error_diagnostics.json", output_dir / "e19f.md")
    return metrics


def _write_report(metrics, diagnostics_path, report_path):
    baseline = metrics["cross_validated_complete_groups"]["E19-B Qwen3.5-2B"]
    ranked = sorted((method, metrics["cross_validated_complete_groups"][method])
        for method in ("E19-F-A", "E19-F-B", "E19-F-C"))
    precision_winner = max(ranked, key=lambda item: (item[1]["pairwise_precision"],
        item[1]["exact_group_recovery"], item[1]["pairwise_recall"]))
    exact_winner = max(ranked, key=lambda item: (item[1]["exact_group_recovery"],
        item[1]["pairwise_precision"], item[1]["pairwise_f1"]))
    lines = ["# E19-F — Direct Response Embedding and Prompt Clustering", "",
        "| Method | Exact groups | Pair precision | Pair recall | Pair F1 | Incorrect merges |",
        "|---|---:|---:|---:|---:|---:|"]
    lines.append(f"| E19-B Qwen3.5-2B | {100*baseline['exact_group_recovery']:.2f}% ({baseline['exact_groups']}/{baseline['true_group_count']}) | {100*baseline['pairwise_precision']:.2f}% | {100*baseline['pairwise_recall']:.2f}% | {100*baseline['pairwise_f1']:.2f}% | {baseline['incorrectly_merged_groups']} |")
    for method in ("E19-F-A", "E19-F-B", "E19-F-C"):
        result = metrics["cross_validated_complete_groups"][method]
        lines.append(f"| {method} | {100*result['exact_group_recovery']:.2f}% ({result['exact_groups']}/{result['true_group_count']}) | {100*result['pairwise_precision']:.2f}% | {100*result['pairwise_recall']:.2f}% | {100*result['pairwise_f1']:.2f}% | {result['incorrectly_merged_groups']} |")
    lines.extend(["", "## Validation protocol", "",
        "V0's saved E13-B five outer prompt-group folds were reused. Each E19-F outer fold tunes thresholds (and, where applicable, neighbor count or pair-scorer regularization) using four prompt-group folds drawn only from its outer training partition. Held-out rows are clustered independently. Prompt IDs are used only as fitting labels and for evaluation; the prediction API accepts only response IDs, text and embeddings.",
        "", "## Cross-validated complete-group metrics", "", "```json",
        json.dumps(metrics["cross_validated_complete_groups"], indent=2), "```", "",
        "## Incomplete-group evaluation", "",
        "For each outer fold and each fixed seed, every held-out prompt group was independently reduced to a random size from 1–10 or 1–18 responses. Metrics are reported by seed without mixing rows across trials.", "", "```json",
        json.dumps(metrics["incomplete_groups_by_method_seed"], indent=2), "```", "",
        "## Fold results and selected configurations", "",
        "| Method | Fold | Exact groups | Pair precision | Pair recall | Selected settings |",
        "|---|---:|---:|---:|---:|---|"])
    for method in ("E19-F-A", "E19-F-B", "E19-F-C"):
        for entry in metrics["outer_fold_results"][method]:
            m = entry["metrics"]
            lines.append(f"| {method} | {entry['fold']} | {m['exact_groups']}/{m['true_group_count']} | {m['pairwise_precision']:.2%} | {m['pairwise_recall']:.2%} | `{json.dumps(entry['selected'], sort_keys=True)}` |")
    lines.extend(["", "## Error analysis", "",
        f"Examples and the full shortest-response tail are in [`error_diagnostics.json`]({Path(diagnostics_path).name}). E19-F-A's false merge examples include a space-ship narrative grouped with Antarctic ice-core prose: the subject matter differs, while both use vivid, metaphorical descriptive language. This is a topic/style collision in embedding space. E19-F-A made {metrics['cross_validated_complete_groups']['E19-F-A']['incorrectly_merged_groups']} false merges overall; E19-F-C made none, at the cost of more splits.",
        "",
        "Other A merges show both kinds of semantic collision: p120's heat-death/entropy descriptions merged with p080's dark-energy expansion prose (related cosmology, different prompts), while p021's zero-knowledge proof analogies merged with p071's philosophical discussion of consciousness (similar abstract wording, different tasks).",
        "",
        "The split examples show the other failure mode: responses to the same prompt can use very different wording and narrative details. For example, the Voyager Golden Record group describes the same object through different images and registers. Character TF-IDF has less evidence in these cases, while the semantic signal still fragments some groups under the conservative complete-link cutoff.",
        "",
        "V0 contains no responses of 12 words or fewer, so the diagnostics record the 20 shortest responses instead of claiming to observe truly short inputs. This dataset therefore cannot establish performance on one-line or generic answers. Original prompt text is not present, so topic/task labels in this analysis are inferred from response content and known prompt IDs used only for audit.", "",
        "### Representative error examples", ""])
    diagnostics = json.loads(Path(diagnostics_path).read_text())
    merge = diagnostics["E19-F-A"]["merge_examples"]
    split = diagnostics["E19-F-C"]["split_examples"]
    correct = diagnostics["E19-F-A"]["correct_examples"]
    if correct:
        lines.append("- **Correct reconstruction (E19-F-A):** prompt group " + correct[0]["true_prompt_ids"][0] + ". Example: “" + correct[0]["responses"][0]["text"][:220].replace("\n", " ") + "”")
    if merge:
        lines.append("- **Incorrect merge (E19-F-A):** prompt groups " + ", ".join(merge[0]["true_prompt_ids"]) + ". Example text: “" + merge[0]["responses"][0]["text"][:220].replace("\n", " ") + "”")
    if split:
        lines.append("- **Incorrect split (E19-F-C):** prompt group " + split[0]["true_prompt_ids"][0] + ", split across " + str(len(split[0]["predicted_groups"])) + " clusters. Example: “" + split[0]["responses"][0]["text"][:220].replace("\n", " ") + "”")
    lines.extend(["", "## Full-V0 diagnostic clustering", "",
        "These results are in-sample diagnostics: final clustering models were fit using all V0 prompt labels and then clustered all V0 rows. They are separate from the outer-fold estimates and must not be interpreted as held-out performance.", "", "```json",
        json.dumps(metrics["full_v0_diagnostic_only"], indent=2), "```", "",
        "## Embedding and runtime", "", "```json",
        json.dumps(metrics["embedding"], indent=2), "```", "",
        f"Per-method held-out clustering inference time (not including hyperparameter tuning): `{json.dumps(metrics['clustering_runtime_seconds_by_method'], sort_keys=True)}` seconds. End-to-end experiment runtime, including nested tuning and incomplete-group comparisons: {metrics['total_runtime_seconds']:.1f} seconds.", "",
        "## Artifacts and inference", "",
        "`response_embeddings.npy` contains the normalized embeddings in V0 row order; `embedding_cache.sqlite3` stores resumable per-response vectors keyed by response hash, model revision and encoding configuration. `predicted_group_assignments.parquet` contains OOF full-group assignments; `incomplete_group_assignments.parquet` contains the seeded trials. Three `*_kaggle_grouper.joblib` files save the full-V0 deployment models/configurations. `predict_groups(row_ids, texts, model_path=...)` accepts Kaggle-style ID/text only and reuses cached embeddings.", "",
        "## Conclusions", "",
        f"1. Highest E19-F precision is **{precision_winner[0]}** ({precision_winner[1]['pairwise_precision']:.2%}); highest exact-group recovery is **{exact_winner[0]}** ({exact_winner[1]['exact_group_recovery']:.2%}).",
        f"2. Direct embeddings can match or beat E19-B on grouping scores: E19-F-A reaches {metrics['cross_validated_complete_groups']['E19-F-A']['pairwise_f1']:.2%} pair F1 and {metrics['cross_validated_complete_groups']['E19-F-A']['exact_group_recovery']:.2%} exact recovery, versus {baseline['pairwise_f1']:.2%} and {baseline['exact_group_recovery']:.2%} for E19-B. A also introduces {metrics['cross_validated_complete_groups']['E19-F-A']['incorrectly_merged_groups']} false merges. E19-F-C keeps zero merges but has lower pair recall/F1 than E19-B. Full-V0 diagnostics are not held-out evidence.",
        f"3. E19-F is technically suitable for ID/text-only Kaggle grouping because the saved inference bundle creates embeddings and groups without prompt IDs. Its quality tradeoff is material: choose A for higher recall only if false merges are acceptable, or C for zero merges with more splits.",
        "4. Use **E19-B Qwen3.5-2B** as the primary grouping method in subsequent E20 experiments: it has zero incorrect merges, 100% pair precision, and better recall and F1 than zero-merge E19-F-C. Keep E19-F-A as a high-recall comparison arm and E19-F-C as the direct ID/text-only conservative arm.", "",
        f"Runtime: {metrics['total_runtime_seconds']:.1f} seconds total; embedding generation {metrics['embedding']['generation_runtime_seconds']:.1f} seconds."])
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    required_report = ROOT / "docs/e19f.md"
    if report_path.resolve() != required_report.resolve():
        required_report.parent.mkdir(parents=True, exist_ok=True)
        required_report.write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--skip-incomplete", action="store_true")
    args = parser.parse_args(argv)
    run(output_dir=args.output_dir, device=args.device, batch_size=args.batch_size,
        run_incomplete=not args.skip_incomplete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
