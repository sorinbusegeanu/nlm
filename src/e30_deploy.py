"""Shared E30 family-max, E19-F clustering and Hungarian deployment helpers."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import e19f, e27
from .v2_checkpoint_trust import LABELS


def cluster_f_candidates(method, ids, ordered, pairs, features, bundle):
    """Apply the frozen E19-F-A or F-C edge score and complete-link grouping."""
    ids = np.asarray(ids, dtype=str)
    ordered = np.asarray(ordered, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    features = np.asarray(features, dtype=np.float32)
    if method not in {"E19-F-A", "E19-F-C"} or len(ids) != len(ordered):
        raise ValueError("Invalid E30 grouping inputs")
    scores = (features[:, 0] if method == "E19-F-A" else
              e19f.score_similarity_model(bundle["pair_model"], features))
    groups_sorted, _ = e19f._cluster_fc(ids[ordered], pairs, scores,
                                        float(bundle["threshold"]))
    result = np.empty(len(ids), dtype=object)
    result[ordered] = groups_sorted
    return result


def predict_f_groups(method, ids, texts, embeddings, bundle):
    ids = np.asarray(ids, dtype=str); texts = np.asarray(texts, dtype=str)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(ids) != len(texts) or len(ids) != len(embeddings) or len(set(ids.tolist())) != len(ids):
        raise ValueError("E30 grouping rows are misaligned or duplicate")
    lexical = e19f._transform_lexical(texts, bundle["vectorizers"])
    _, ordered, pairs, features = e27._candidate_bundle(
        np.arange(len(ids)), ids, texts, embeddings, bundle["vectorizers"], lexical=lexical)
    return cluster_f_candidates(method, ids, ordered, pairs, features, bundle)


def hungarian(scores, groups, row_ids=None):
    score = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    if score.shape != (len(groups), len(LABELS)) or not np.isfinite(score).all():
        raise ValueError("E30 assignment scores and groups are misaligned")
    if row_ids is not None and len(row_ids) != len(groups):
        raise ValueError("E30 row IDs and groups are misaligned")
    pred = np.full(len(groups), -1, dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        if len(rows) > len(LABELS):
            raise ValueError(f"E30 group exceeds 18 responses: {group}")
        rr, cc = linear_sum_assignment(-score[rows])
        pred[rows[rr]] = cc
    if (pred < 0).any():
        raise ValueError("E30 assignment missed rows")
    return pred
