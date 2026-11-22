"""Similarity features and conservative complete-link clustering for E19-B."""
from __future__ import annotations

import heapq
import re
from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import e16

FEATURE_NAMES = (
    "reconstruction_char_cosine", "reconstruction_word_cosine",
    "reconstruction_token_jaccard", "reconstruction_token_overlap",
    "response_char_cosine", "response_word_cosine",
    "response_token_jaccard", "response_token_overlap",
    "semantic_prompt_cosine", "topic_token_jaccard", "topic_exact_match",
    "distinctive_detail_jaccard", "distinctive_detail_overlap",
    "distinctive_entity_jaccard", "distinctive_number_jaccard",
    "constraint_jaccard", "constraint_overlap",
)
DEFAULT_NEIGHBORS = 40
MAX_CLUSTER_SIZE = 18


def fit_similarity_spaces(reconstructed, responses, fit_rows):
    """Fit word/character TF-IDF on training rows; transform all examples."""
    reconstructed = np.asarray(reconstructed, dtype=str)
    responses = np.asarray(responses, dtype=str)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    if len(reconstructed) != len(responses) or not len(fit_rows):
        raise ValueError("E19-B text and fitting rows do not align")
    configs = {
        "reconstruction_char": TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
            min_df=1, sublinear_tf=True, dtype=np.float32, max_features=300000),
        "reconstruction_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
            min_df=1, sublinear_tf=True, dtype=np.float32, max_features=150000),
        "response_char": TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
            min_df=1, sublinear_tf=True, dtype=np.float32, max_features=300000),
        "response_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
            min_df=1, sublinear_tf=True, dtype=np.float32, max_features=150000),
    }
    result = {}
    for name, vectorizer in configs.items():
        source = reconstructed if name.startswith("reconstruction") else responses
        result[name + "_vectorizer"] = vectorizer.fit(source[fit_rows])
        result[name + "_matrix"] = result[name + "_vectorizer"].transform(source)
    return result


def transform_similarity_spaces(reconstructed, responses, vectorizers):
    """Transform new text rows using a fitted grouper's persisted vocabularies."""
    reconstructed=np.asarray(reconstructed,dtype=str)
    responses=np.asarray(responses,dtype=str)
    if len(reconstructed)!=len(responses):
        raise ValueError("E19-B new response and reconstruction rows do not align")
    result={}
    for name in ("reconstruction_char","reconstruction_word","response_char","response_word"):
        vectorizer=vectorizers[name]
        source=reconstructed if name.startswith("reconstruction") else responses
        result[name+"_vectorizer"]=vectorizer
        result[name+"_matrix"]=vectorizer.transform(source)
    return result


def _neighbor_pairs(matrix, rows, neighbors):
    # Stable row ordering makes nearest-neighbor tie handling repeatable even
    # when the caller supplies the same fold rows in a different order.
    rows = np.sort(np.asarray(rows, dtype=np.int64), kind="stable")
    n = len(rows)
    if n < 2:
        return set()
    k = min(n, int(neighbors) + 1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=1)
    local = matrix[rows]
    nn.fit(local)
    _, indices = nn.kneighbors(local)
    pairs = set()
    for i, neighbors_i in enumerate(indices):
        for j in neighbors_i:
            j = int(j)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def _dense_neighbor_pairs_gpu(embeddings, rows, neighbors, *, device="cuda:0", block_size=128):
    """Exact top-k cosine neighbors for dense embeddings, with GPU dot products."""
    import torch

    rows = np.sort(np.asarray(rows, dtype=np.int64), kind="stable")
    values = np.asarray(embeddings, dtype=np.float32)[rows]
    n = len(rows)
    if n < 2:
        return set()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("Semantic embeddings must be finite and nonzero")
    values = values / norms
    matrix = torch.as_tensor(values, dtype=torch.float32, device=device)
    k = min(n, int(neighbors) + 1)
    pairs = set()
    with torch.inference_mode():
        for start in range(0, n, int(block_size)):
            end = min(start + int(block_size), n)
            similarities = (matrix[start:end] @ matrix.T).cpu().numpy()
            for offset, scores in enumerate(similarities):
                row = start + offset
                selected = np.argpartition(-scores, k - 1)[:k]
                selected = selected[np.lexsort((selected, -scores[selected]))]
                for other in selected:
                    other = int(other)
                    if other != row:
                        pairs.add((min(row, other), max(row, other)))
            del similarities
    del matrix
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return pairs


def candidate_pairs(rows, spaces, semantic_embeddings, *, neighbors=DEFAULT_NEIGHBORS,
                    semantic_device=None):
    """Union nearest neighbors from reconstructed prompt and response signals."""
    rows = np.sort(np.asarray(rows, dtype=np.int64), kind="stable")
    if len(rows) < 2:
        return np.empty((0, 2), dtype=np.int64)
    result = set()
    for name in ("reconstruction_char_matrix", "reconstruction_word_matrix",
                 "response_char_matrix", "response_word_matrix"):
        result.update(_neighbor_pairs(spaces[name], rows, neighbors))
    emb = np.asarray(semantic_embeddings, dtype=np.float32)
    if semantic_device is not None:
        result.update(_dense_neighbor_pairs_gpu(emb, rows, neighbors,
                                                device=semantic_device))
    else:
        emb = emb[rows]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        if not np.isfinite(emb).all() or np.any(norms <= 0):
            raise ValueError("E19-B semantic embeddings must be finite and nonzero")
        emb = emb / norms
        result.update(_neighbor_pairs(sparse.csr_matrix(emb), np.arange(len(rows)), neighbors))
    return np.asarray(sorted(result), dtype=np.int64).reshape(-1, 2)


def _overlap_features(values, left, right):
    jaccard = np.zeros(len(left), dtype=np.float32)
    overlap = np.zeros(len(left), dtype=np.float32)
    for ix, (a, b) in enumerate(zip(left, right)):
        aa, bb = values[int(a)], values[int(b)]
        jaccard[ix] = len(aa & bb) / max(1, len(aa | bb))
        overlap[ix] = len(aa & bb) / max(1, min(len(aa), len(bb)))
    return jaccard, overlap


def _terms(items):
    result = set()
    for item in items if isinstance(items, (list, tuple)) else []:
        text = str(item).casefold().strip()
        words = re.findall(r"[\w]+(?:[.][\w]+)*", text, flags=re.UNICODE)
        if words:
            result.add(" ".join(words))
            result.update(word for word in words if len(word) > 1)
    return result


def _structured_detail_terms(items, pattern):
    source=" ".join(str(x) for x in items) if isinstance(items,(list,tuple)) else str(items)
    return {match.casefold() for match in re.findall(pattern,source)}


def pair_features(rows, pairs, spaces, reconstructed, responses, semantic_embeddings,
                  topics, distinctive_details, constraints):
    """Create aligned E16 lexical plus E19-B semantic/detail pair features."""
    rows = np.asarray(rows, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    if np.any(pairs < 0) or np.any(pairs >= len(rows)):
        raise ValueError("E19-B local pair indexes do not align with rows")
    local_recon = np.asarray(reconstructed, dtype=str)[rows]
    local_response = np.asarray(responses, dtype=str)[rows]
    local_topics = np.asarray(topics, dtype=str)[rows]
    left, right = pairs[:, 0], pairs[:, 1]
    recon_lex = e16._pair_features(local_recon,
        spaces["reconstruction_char_matrix"][rows],
        spaces["reconstruction_word_matrix"][rows], left, right)[:, :4]
    response_lex = e16._pair_features(local_response,
        spaces["response_char_matrix"][rows],
        spaces["response_word_matrix"][rows], left, right)[:, :4]
    embeddings = np.asarray(semantic_embeddings, dtype=np.float32)[rows]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    embeddings = embeddings / norms
    semantic = np.sum(embeddings[left] * embeddings[right], axis=1, keepdims=True)
    topic_terms = [set(e16._basic(value)) for value in local_topics]
    topic_jac, _ = _overlap_features(topic_terms, left, right)
    topic_exact = np.asarray([float(local_topics[a].casefold().strip() ==
                                    local_topics[b].casefold().strip() and
                                    bool(local_topics[a].strip())) for a, b in pairs],
                             dtype=np.float32)
    detail_terms = [_terms(distinctive_details[int(row)]) for row in rows]
    entity_terms = [_structured_detail_terms(distinctive_details[int(row)],
        r"\b[A-Z][A-Za-z0-9_-]{2,}\b") for row in rows]
    number_terms = [_structured_detail_terms(distinctive_details[int(row)],
        r"(?<!\w)\d+(?:[.,:/-]\d+)*(?:%|\b)") for row in rows]
    constraint_terms = [_terms(constraints[int(row)]) for row in rows]
    detail_jac, detail_overlap = _overlap_features(detail_terms, left, right)
    entity_jac, _ = _overlap_features(entity_terms, left, right)
    number_jac, _ = _overlap_features(number_terms, left, right)
    constraint_jac, constraint_overlap = _overlap_features(constraint_terms, left, right)
    features = np.column_stack([recon_lex, response_lex, semantic[:, 0], topic_jac,
                                topic_exact, detail_jac, detail_overlap,
                                entity_jac, number_jac, constraint_jac,
                                constraint_overlap]).astype(np.float32)
    if features.shape != (len(pairs), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("E19-B pair features are malformed or non-finite")
    return features


def fit_pair_scorer(features, same_group, *, c=1.0, seed=2026):
    """Learn component weights from fitting-fold pair labels only."""
    features = np.asarray(features, dtype=np.float32)
    same_group = np.asarray(same_group, dtype=np.int8)
    if features.shape != (len(same_group), len(FEATURE_NAMES)) or len(features) < 2:
        raise ValueError("E19-B training pairs/features do not align")
    if len(np.unique(same_group)) != 2:
        raise ValueError("E19-B pair model needs both same-prompt and different-prompt pairs")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=float(c), class_weight="balanced", solver="liblinear",
                               max_iter=2000, random_state=int(seed))
    model.fit(scaler.transform(features), same_group)
    return {"scaler": scaler, "model": model, "C": float(c),
            "feature_names": list(FEATURE_NAMES),
            "coefficients": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0])}


def score_pairs(model_bundle, features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("E19-B score feature matrix has invalid shape")
    model = model_bundle["model"]
    # Joblib models saved by newer scikit-learn releases may omit this
    # deprecated attribute, while Kaggle's older LogisticRegression
    # predict_proba implementation still reads it. Restoring the historical
    # default keeps the binary OVR probability path compatible.
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model.predict_proba(
        model_bundle["scaler"].transform(features))[:, 1].astype(np.float32)


def cluster_complete_link(row_ids, pair_indexes, probabilities, *, threshold,
                          max_cluster_size=MAX_CLUSTER_SIZE):
    """Deterministic agglomerative complete linkage over retrieved edges only.

    A cluster pair is mergeable only if every cross-cluster row pair was
    retrieved and has a probability at least the configured threshold. This
    rejects weak transitive chains. Missing candidate edges have similarity 0.
    """
    row_ids = np.asarray(row_ids, dtype=str)
    pair_indexes = np.asarray(pair_indexes, dtype=np.int64).reshape(-1, 2)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(pair_indexes) != len(probabilities) or len(set(row_ids.tolist())) != len(row_ids):
        raise ValueError("E19-B clustering rows and candidate edges do not align")
    if not np.isfinite(probabilities).all() or not 0 <= float(threshold) <= 1:
        raise ValueError("E19-B edge scores or threshold are invalid")
    n = len(row_ids)
    if n == 0:
        return np.empty(0, dtype=str), {"clusters": []}
    if not 1 <= int(max_cluster_size) <= 18:
        raise ValueError("E19-B maximum cluster size must be between 1 and 18")
    order = np.argsort(row_ids, kind="stable")
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    # Normalize row indexes to ID order, so tie-breaking cannot depend on input order.
    ids = row_ids[order]
    edges = {}
    for (a, b), prob in zip(pair_indexes, probabilities):
        a, b = int(inverse[a]), int(inverse[b])
        if a == b:
            raise ValueError("E19-B candidate edge cannot connect a row to itself")
        key = (min(a, b), max(a, b))
        edges[key] = max(edges.get(key, -np.inf), float(prob))
    members = {i: [i] for i in range(n)}
    neighbors = {i: set() for i in range(n)}
    sims = {}
    heap = []
    next_cluster = n

    def pair_key(a, b):
        return (min(a, b), max(a, b))

    def push(a, b, sim):
        if sim < threshold or len(members[a]) + len(members[b]) > max_cluster_size:
            return
        key = pair_key(a, b)
        sims[key] = float(sim)
        neighbors[a].add(b)
        neighbors[b].add(a)
        first = ids[min(members[a])]
        second = ids[min(members[b])]
        if second < first:
            first, second = second, first
        heapq.heappush(heap, (-float(sim), str(first), str(second), key[0], key[1]))

    for (a, b), prob in sorted(edges.items()):
        if prob >= threshold:
            push(a, b, prob)
    while heap:
        neg, _, _, a, b = heapq.heappop(heap)
        key = (a, b)
        if a not in members or b not in members or key not in sims:
            continue
        sim = sims[key]
        if abs(sim + neg) > 1e-12:
            continue
        if len(members[a]) + len(members[b]) > max_cluster_size:
            continue
        new = next_cluster
        next_cluster += 1
        merged_members = sorted(members[a] + members[b])
        old_neighbors = (neighbors[a] | neighbors[b]) - {a, b}
        old_links = {}
        for other in old_neighbors:
            if other in (a, b) or other not in members:
                continue
            old_a, old_b = sims.get(pair_key(a, other)), sims.get(pair_key(b, other))
            ka, kb = pair_key(a, other), pair_key(b, other)
            sims.pop(ka, None); sims.pop(kb, None)
            neighbors[other].discard(a); neighbors[other].discard(b)
            if old_a is not None and old_b is not None:
                old_links[other] = min(old_a, old_b)
        sims.pop(key, None)
        neighbors[a].clear(); neighbors[b].clear()
        del members[a]; del members[b]
        neighbors.pop(a); neighbors.pop(b)
        members[new] = merged_members
        neighbors[new] = set()
        # Complete linkage requires a retrieved edge from every member to the
        # other cluster. Retain only links present from both former clusters.
        for other, new_sim in old_links.items():
            if other in members:
                push(new, other, new_sim)
    clusters = sorted(members.values(), key=lambda ix: str(ids[min(ix)]))
    assignment_sorted = np.full(n, -1, dtype=np.int64)
    cluster_payload = []
    for cluster_no, rows in enumerate(clusters):
        assignment_sorted[rows] = cluster_no
        cluster_payload.append({"cluster_index": cluster_no,
            "response_ids": [str(ids[i]) for i in rows], "size": len(rows)})
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = assignment_sorted
    cluster_ids = np.asarray([f"e19b-{value:05d}" for value in assignment], dtype=object)
    return cluster_ids, {"clusters": cluster_payload,
                         "max_cluster_size": int(max_cluster_size),
                         "threshold": float(threshold)}


def edge_score_lookup(pair_indexes, probabilities):
    return {(min(int(a), int(b)), max(int(a), int(b))): float(p)
            for (a, b), p in zip(np.asarray(pair_indexes).reshape(-1, 2), probabilities)}
