"""E26 external-only ensemble generalization and assignment robustness study.

V0 is used only for final evaluation. Dataset construction, checkpoint scorer,
pair grouper and all policy thresholds are fit/selected on external prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .data_factory import _normal_text
from .e20 import hungarian
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
EXT_PATH = ROOT / "data/processed/dataset_v2.16_openpool16_increment.parquet"
V0_GROUP_PATH = ROOT / "data/processed/dataset_v0_group.parquet"
EXT_GROUP_PATH = ROOT / "data/processed/dataset_v2_16OP_RandomGroups.parquet"
OUT = ROOT / "artifacts/analysis/e26"
SEED = 2026
N_SCENARIOS = 50
TARGET_SCENARIO_ROWS = 500
CHECKPOINTS = {
    "qwen06": "qwen06_v2.16_valv0_data3e3338ca_split323dd2c7_seed2026_cfgbe62927e_code3458579",
    "ultrabert380": "ultrabert380_v2.16_valv0_data3e3338ca_split323dd2c7_seed2026_cfgef8e3fcc_code3458579",
    "modernbert_large": "modernbert_large_v2.16_valv0_data3e3338ca_split323dd2c7_seed2026_cfg6ceaf5ba_code3458579",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stable_seed(seed: int, *parts) -> int:
    payload = "|".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_v0_group(v0: pd.DataFrame, *, n_scenarios=N_SCENARIOS,
                   target_rows=TARGET_SCENARIO_ROWS, seed=SEED) -> pd.DataFrame:
    """Create fixed-size scenarios of true prompt groups with unique labels."""
    required = {"id", "prompt_id", "text", "model"}
    if not required.issubset(v0.columns):
        raise ValueError(f"V0 misses {sorted(required - set(v0.columns))}")
    if v0.id.astype(str).duplicated().any() or set(v0.model) - set(LABELS):
        raise ValueError("V0 IDs/labels fail integrity checks")
    by_prompt = {str(p): g for p, g in v0.groupby("prompt_id", sort=True)}
    prompt_labels = {p: set(g.model.astype(str)) for p, g in by_prompt.items()}
    rows = []
    for scenario in range(int(n_scenarios)):
        rng = np.random.default_rng(_stable_seed(seed, "v0", scenario))
        prompts = list(by_prompt)
        rng.shuffle(prompts)
        used = set()
        chosen_sizes = list(range(1, 19))
        rng.shuffle(chosen_sizes)
        remaining = int(target_rows)
        group_number = 0

        def add_group(size):
            nonlocal remaining, group_number
            candidates = [p for p in prompts if p not in used and len(prompt_labels[p]) >= size]
            if not candidates:
                raise ValueError(f"cannot find unused V0 prompt for group size {size}")
            prompt = candidates[int(rng.integers(len(candidates)))]
            used.add(prompt)
            source = by_prompt[prompt]
            label_pool = sorted(prompt_labels[prompt])
            labels = rng.choice(label_pool, size=size, replace=False).tolist()
            subset = source[source.model.astype(str).isin(labels)].copy()
            subset = subset.set_index(subset.model.astype(str)).loc[labels].reset_index(drop=True)
            gid = f"v0s{scenario:02d}_g{group_number:03d}"
            group_number += 1
            for r in subset.itertuples(index=False):
                rows.append({"scenario_id": f"v0s{scenario:02d}",
                    "scenario_row_id": f"v0s{scenario:02d}_{str(r.id)}",
                    "original_id": str(r.id), "original_prompt_id": prompt,
                    "group_id": gid, "text": str(r.text), "model": str(r.model),
                    "intended_group_size": int(size)})
            remaining -= size

        for size in chosen_sizes:
            if size > remaining:
                raise ValueError("target V0 scenario size too small for required group sizes")
            add_group(size)
        while remaining:
            feasible = [k for k in range(1, min(18, remaining) + 1)
                        if any(p not in used and len(prompt_labels[p]) >= k for p in prompts)]
            if not feasible:
                raise ValueError("V0 prompts exhausted before scenario reached target row count")
            add_group(int(rng.choice(feasible)))
    result = pd.DataFrame(rows)
    validate_v0_group(result, target_rows=target_rows, n_scenarios=n_scenarios)
    return result


def validate_v0_group(frame: pd.DataFrame, *, target_rows=TARGET_SCENARIO_ROWS,
                      n_scenarios=N_SCENARIOS) -> None:
    required = {"scenario_id", "scenario_row_id", "original_id", "original_prompt_id",
                "group_id", "text", "model", "intended_group_size"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("v0_group required fields are missing or null")
    if frame.scenario_row_id.duplicated().any():
        raise ValueError("scenario-specific V0 row IDs are not unique")
    for scenario, part in frame.groupby("scenario_id"):
        if len(part) != target_rows:
            raise ValueError(f"{scenario} has {len(part)} rows, expected {target_rows}")
        if set(part.intended_group_size.astype(int)) < set(range(1, 19)):
            raise ValueError(f"{scenario} does not cover all group sizes")
        if part.groupby("group_id").original_prompt_id.first().duplicated().any():
            raise ValueError(f"{scenario} reuses an original prompt")
        for _, group in part.groupby("group_id"):
            if len(group) != int(group.intended_group_size.iloc[0]):
                raise ValueError("V0 scenario group size is inconsistent")
            if group.model.astype(str).duplicated().any():
                raise ValueError("V0 scenario group contains duplicate labels")
            if group.original_prompt_id.nunique() != 1:
                raise ValueError("V0 scenario group crosses original prompts")
    if frame.scenario_id.nunique() != n_scenarios:
        raise ValueError("wrong number of V0 scenarios")


def _source_prompt_key(row) -> str:
    source = str(row.get("source_dataset") or row.get("source") or "unknown")
    prompt = str(row.get("source_prompt_id") or row.get("prompt_id") or row.get("prompt_group") or "")
    return f"{source}::{prompt}"


def build_external_random_groups(ext: pd.DataFrame, v0: pd.DataFrame, *, seed=SEED):
    """Sample source-prompt groups, one response per distinct label."""
    required = {"id", "text", "model", "source_dataset"}
    if not required.issubset(ext.columns):
        raise ValueError(f"external data misses {sorted(required - set(ext.columns))}")
    data = ext.copy()
    data["original_id"] = data.id.astype(str)
    data["original_prompt_id"] = data.apply(_source_prompt_key, axis=1)
    v0_ids = set(v0.id.astype(str))
    v0_prompt_ids = set(v0.prompt_id.astype(str)) if "prompt_id" in v0 else set()
    v0_text = {_normal_text(x) for x in v0.text.astype(str)}
    id_overlap = data.original_id.isin(v0_ids)
    text_overlap = data.text.astype(str).map(_normal_text).isin(v0_text)
    raw_prompt_columns = [c for c in ["source_prompt_id", "prompt_id", "prompt_group"] if c in data]
    raw_prompt_overlap = pd.Series(False, index=data.index)
    for column in raw_prompt_columns:
        raw_prompt_overlap |= data[column].astype(str).isin(v0_prompt_ids)
    prompt_text_available = "prompt_text" in v0.columns
    known_prompt_text = {_normal_text(x) for x in v0.get("prompt_text", pd.Series([], dtype=str)).dropna()}
    prompt_text_overlap = data.get("prompt_text", pd.Series([""] * len(data), index=data.index)).astype(str).map(_normal_text).isin(known_prompt_text) if prompt_text_available else pd.Series(False, index=data.index)
    if id_overlap.any() or text_overlap.any() or raw_prompt_overlap.any() or prompt_text_overlap.any():
        raise ValueError(f"external source overlaps V0: ids={id_overlap.sum()}, text={text_overlap.sum()}, prompt_id={raw_prompt_overlap.sum()}, prompt_text={prompt_text_overlap.sum()}")
    source_profile_path = ROOT / "data/processed/dataset_profile_v2.16_openpool16_increment.json"
    source_checks = json.loads(source_profile_path.read_text()).get("checks", {}) if source_profile_path.is_file() else {}
    source_groups = {}
    for key, g in data.groupby("original_prompt_id", sort=True):
        labels = g.model.astype(str)
        source_groups[key] = {lab: ix.to_numpy() for lab, ix in g.groupby(labels).groups.items()}
    available = {}
    for size in range(1, 19):
        eligible = [k for k, labels in source_groups.items() if len(labels) >= size]
        if eligible:
            available[size] = eligible
    rng = np.random.default_rng(seed)
    selected = []
    used_prompts = set()
    schedule = []
    for size, eligible in available.items():
        n = len(eligible) if len(eligible) < 4 else int(rng.integers(4, min(10, len(eligible)) + 1))
        schedule.append((size, n))
    # Larger groups are scarcer, reserve those first; prompt pools are disjoint
    # across sizes so all sampled groups have different original prompts.
    for size, n in sorted(schedule, key=lambda x: (-x[0], x[1])):
        eligible = [k for k in available[size] if k not in used_prompts]
        actual = min(n, len(eligible))
        if not actual:
            schedule[schedule.index((size, n))] = (size, 0)
            continue
        chosen_prompts = rng.choice(sorted(eligible), size=actual, replace=False).tolist()
        for group_number, prompt_key in enumerate(chosen_prompts):
            label_map = source_groups[prompt_key]
            labels = sorted(label_map)
            chosen_labels = rng.choice(labels, size=size, replace=False).tolist()
            for label in chosen_labels:
                ix = int(rng.choice(label_map[label]))
                row = data.iloc[ix].to_dict()
                row.update(original_id=str(row["id"]), original_prompt_id=prompt_key,
                           group_id=f"opg_{size:02d}_{group_number:03d}_{hashlib.sha1(prompt_key.encode()).hexdigest()[:8]}",
                           group_size=int(size))
                selected.append(row)
            used_prompts.add(prompt_key)
    out = pd.DataFrame(selected).reset_index(drop=True)
    if out.empty:
        raise ValueError("no external prompt groups were available")
    validate_external_groups(out, v0)
    audit = {"available_prompt_groups_by_size": {str(k): len(v) for k, v in available.items()},
             "requested_sampled_groups_by_size": {str(k): n for k, n in schedule},
             "actual_sampled_groups_by_size": {str(k): int(v) for k, v in out.groupby("group_size").group_id.nunique().items()},
             "unavailable_sizes": sorted(set(range(1, 19)) - set(available)),
             "v0_id_overlap": int(id_overlap.sum()), "v0_normalized_text_overlap": int(text_overlap.sum()),
             "v0_raw_prompt_id_overlap": int(raw_prompt_overlap.sum()),
             "v0_known_prompt_text_overlap": int(prompt_text_overlap.sum()) if prompt_text_available else None,
             "v0_prompt_text_check": "performed" if prompt_text_available else "not available; V0 stores prompt_id but no canonical prompt text",
             "upstream_v0_prompt_group_isolation_verified": bool(source_checks.get("accepted_prompt_groups_disjoint_v0", False))}
    return out, audit


def validate_external_groups(frame: pd.DataFrame, v0: pd.DataFrame):
    if frame.original_id.astype(str).duplicated().any():
        raise ValueError("external random groups repeat a source response")
    if not set(frame.model.astype(str)).issubset(set(LABELS)):
        raise ValueError("external random groups contain invalid labels")
    v0_text = {_normal_text(x) for x in v0.text.astype(str)}
    if frame.text.astype(str).map(_normal_text).isin(v0_text).any():
        raise ValueError("external random groups contain V0 response text")
    for gid, group in frame.groupby("group_id"):
        if len(group) != int(group.group_size.iloc[0]) or group.model.astype(str).duplicated().any():
            raise ValueError(f"invalid unique-label group {gid}")
        if group.original_prompt_id.nunique() != 1:
            raise ValueError(f"group {gid} spans source prompts")
    if frame.groupby("group_id").original_prompt_id.nunique().max() != 1:
        raise ValueError("same original source prompt used for multiple selected groups")
    group_prompt = frame.groupby("group_id").original_prompt_id.first()
    if group_prompt.duplicated().any():
        raise ValueError("source prompt reused across sampled groups")


def _metric_bundle(y, pred):
    y = np.asarray(y, dtype=str); pred = np.asarray(pred, dtype=str)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "per_label": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(s[i])}
                          for i, lab in enumerate(LABELS)}}


def _pair_features(texts, pairs, char_matrix=None, word_matrix=None):
    feats = []
    words = [set(re.findall(r"\w+", _normal_text(t))) for t in texts]
    for a, b in pairs:
        inter = len(words[a] & words[b]); union = max(1, len(words[a] | words[b]))
        la, lb = len(texts[a]), len(texts[b])
        char = float(char_matrix[a].multiply(char_matrix[b]).sum()) if char_matrix is not None else 0.0
        word = float(word_matrix[a].multiply(word_matrix[b]).sum()) if word_matrix is not None else 0.0
        feats.append([char, word, inter / union, min(la, lb) / max(1, max(la, lb)),
                      float(abs(la-lb) / max(1, max(la, lb))),
                      float(texts[a].count("\n") == texts[b].count("\n")),
                      float(texts[a].count("?") == texts[b].count("?"))])
    return np.asarray(feats, dtype=np.float32)


def fit_text_grouper(train: pd.DataFrame, *, seed=SEED):
    """Fit a text-only pair classifier on external prompt identities."""
    texts = train.text.astype(str).tolist()
    char_v = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
                             max_features=180000, sublinear_tf=True, dtype=np.float32)
    word_v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=80000,
                             sublinear_tf=True, dtype=np.float32)
    char_m = char_v.fit_transform(texts); word_m = word_v.fit_transform(texts)
    groups = train.original_prompt_id.astype(str).to_numpy()
    pairs = set()
    nn = NearestNeighbors(n_neighbors=min(15, len(texts)), metric="cosine", n_jobs=1).fit(char_m)
    _, inds = nn.kneighbors(char_m)
    for i, row in enumerate(inds):
        for j in row:
            if i != j:
                pairs.add(tuple(sorted((i, int(j)))))
    # Add same-prompt pairs missed by nearest-neighbor retrieval.
    by_group = pd.Series(np.arange(len(groups))).groupby(groups)
    for _, vals in by_group:
        ix = vals.to_numpy()
        for i in ix:
            for j in ix:
                if i < j: pairs.add((int(i), int(j)))
    pairs = sorted(pairs)
    x = _pair_features(texts, pairs, char_m, word_m)
    y = np.asarray([groups[a] == groups[b] for a, b in pairs], dtype=np.int8)
    if len(np.unique(y)) < 2:
        raise ValueError("external grouping data lacks positive or negative pair examples")
    scaler_model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
                                     random_state=seed, solver="liblinear")
    scaler_model.fit(x, y)
    # Threshold is selected on external training prompt pairs. The validation
    # dataset is never used to tune it.
    probs = scaler_model.predict_proba(x)[:, 1]
    candidates = np.unique(np.quantile(probs, np.linspace(.60, .99, 30)))
    best = (float("-inf"), 0.75)
    for threshold in candidates:
        pred = probs >= threshold
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if f1 > best[0]: best = (f1, float(threshold))
    return {"char_vectorizer": char_v, "word_vectorizer": word_v,
            "pair_model": scaler_model, "threshold": best[1],
            "external_pair_f1": best[0], "feature_names": ["char_cosine", "word_cosine",
                "token_jaccard", "length_ratio", "length_delta", "newline_match", "question_match"]}


def tune_text_grouper_threshold(grouper, validation: pd.DataFrame):
    """Select pair threshold on prompt-held-out external examples only."""
    texts = validation.text.astype(str).tolist()
    groups = validation.original_prompt_id.astype(str).to_numpy()
    cm = grouper["char_vectorizer"].transform(texts)
    wm = grouper["word_vectorizer"].transform(texts)
    nn = NearestNeighbors(n_neighbors=min(15, len(texts)), metric="cosine", n_jobs=1).fit(cm)
    _, indexes = nn.kneighbors(cm)
    pairs = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(indexes) for j in row if i != j})
    if not pairs:
        raise ValueError("external held-out grouping set has no candidate response pairs")
    x = _pair_features(texts, pairs, cm, wm)
    y = np.asarray([groups[a] == groups[b] for a,b in pairs], dtype=bool)
    scores = grouper["pair_model"].predict_proba(x)[:, 1]
    best = (-1.0, .5, 0.0, 0.0)
    for threshold in np.unique(np.quantile(scores, np.linspace(.05, .99, 50))):
        pred = scores >= threshold
        tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if f1 > best[0]: best = (float(f1), float(threshold), precision, recall)
    return {"f1": best[0], "threshold": best[1], "precision": best[2],
            "recall": best[3], "candidate_pairs": len(pairs), "positive_pairs": int(y.sum())}


def reconstruct_groups(grouper, texts, ids, *, threshold=None, max_group=18):
    """Rebuild connected groups from only batch text; returns confidence per row."""
    texts = list(map(str, texts)); ids = np.asarray(ids, dtype=str)
    n = len(texts)
    if n != len(ids) or len(set(ids)) != n:
        raise ValueError("group reconstruction texts and IDs are misaligned")
    if n <= 1:
        return np.asarray([f"singleton::{x}" for x in ids]), np.ones(n, dtype=np.float32), []
    char_m = grouper["char_vectorizer"].transform(texts)
    word_m = grouper["word_vectorizer"].transform(texts)
    nn = NearestNeighbors(n_neighbors=min(15, n), metric="cosine", n_jobs=1).fit(char_m)
    _, inds = nn.kneighbors(char_m)
    pairs = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(inds) for j in row if i != j})
    x = _pair_features(texts, pairs, char_m, word_m)
    probs = grouper["pair_model"].predict_proba(x)[:, 1]
    cutoff = float(grouper["threshold"] if threshold is None else threshold)
    parent = list(range(n)); members = {i: {i} for i in range(n)}
    edges = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ordered = sorted(zip(pairs, probs), key=lambda z: (-float(z[1]), ids[z[0][0]], ids[z[0][1]]))
    for (a, b), score in ordered:
        edges.append((int(a), int(b), float(score)))
        if score < cutoff: continue
        ra, rb = find(a), find(b)
        if ra == rb or len(members[ra]) + len(members[rb]) > max_group: continue
        if ids[min(members[ra] | members[rb])] > ids[max(members[ra] | members[rb])]: ra, rb = rb, ra
        parent[rb] = ra; members[ra] |= members.pop(rb)
    roots = [find(i) for i in range(n)]
    gids = np.asarray([f"recon::{ids[roots[i]]}" for i in range(n)])
    row_conf = np.ones(n, dtype=np.float32)
    for i in range(n):
        inside = [float(p) for (a,b),p in zip(pairs, probs) if roots[a] == roots[b] == roots[i]]
        row_conf[i] = float(np.mean(inside)) if inside else 1.0
    return gids, row_conf, edges


def evidence_features(probs: np.ndarray, labels: np.ndarray, *, groups=None,
                      group_conf=None, include_group=False):
    """Expand row/checkpoint probabilities into 18 candidate-label records."""
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 3 or probs.shape[2] != len(LABELS) or probs.shape[1] < 1:
        raise ValueError("checkpoint probabilities must have shape rows x checkpoints x labels")
    n, ncp, _ = probs.shape
    means = probs.mean(axis=1)
    top = probs.argmax(axis=2)
    entropy = -np.sum(means * np.log(np.clip(means, 1e-8, 1)), axis=1)
    other = np.partition(means, -2, axis=1)[:, -2]
    records = []
    for i in range(n):
        for c, lab in enumerate(LABELS):
            feat = {"candidate_label": lab, "candidate_probability_mean": float(means[i,c]),
                    "candidate_probability_max": float(probs[i,:,c].max()),
                    "candidate_probability_min": float(probs[i,:,c].min()),
                    "candidate_vote_fraction": float(np.mean(top[i] == c)),
                    "candidate_rank": int(1 + np.sum(means[i] > means[i,c])),
                    "mean_probability_margin": float(means[i,c] - other[i]),
                    "probability_entropy": float(entropy[i])}
            for k in range(ncp): feat[f"cp{k}_probability"] = float(probs[i,k,c]); feat[f"cp{k}_top1"] = int(top[i,k] == c)
            if include_group:
                member_rows = np.flatnonzero(groups == groups[i]) if groups is not None else np.asarray([i])
                other_rows = member_rows[member_rows != i]
                feat["group_size"] = int(len(member_rows))
                feat["group_confidence"] = float(group_conf[i]) if group_conf is not None else 1.0
                feat["group_candidate_support"] = int(np.sum(means[other_rows].argmax(axis=1) == c)) if len(other_rows) else 0
                feat["group_candidate_max_other"] = float(means[other_rows,c].max()) if len(other_rows) else 0.0
                feat["group_candidate_mean_other"] = float(means[other_rows,c].mean()) if len(other_rows) else 0.0
                best_other = float(means[other_rows,c].max()) if len(other_rows) else 0.0
                feat["group_candidate_advantage"] = float(means[i,c] - best_other)
                feat["group_labels_supported"] = int(len(set(np.argmax(means[other_rows], axis=1)))) if len(other_rows) else 0
            records.append(feat)
    return pd.DataFrame(records)


def fit_candidate_model(probs, y, *, groups=None, group_conf=None,
                        group_features=False, iterations=500, depth=6, seed=SEED):
    x = evidence_features(probs, y, groups=groups, group_conf=group_conf,
                          include_group=group_features)
    target = np.repeat(np.asarray(y, dtype=str), len(LABELS)) == np.tile(np.asarray(LABELS), len(y))
    model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=.04,
        loss_function="Logloss", eval_metric="Logloss", verbose=False, random_seed=seed,
        thread_count=24, l2_leaf_reg=5, allow_writing_files=False)
    cat_features = ["candidate_label"]
    model.fit(x, target.astype(np.int8), cat_features=cat_features)
    return model, list(x.columns)


def oof_reconstructed_groups(frame: pd.DataFrame, *, n_splits=4, seed=SEED):
    """Reconstruct external training groups from prompt-held-out text only."""
    frame = frame.reset_index(drop=True)
    splits = GroupKFold(n_splits=min(n_splits, frame.original_prompt_id.nunique()))
    groups_out = np.empty(len(frame), dtype=object); confidence = np.zeros(len(frame), dtype=np.float32)
    for fold, (fit_ix, held_ix) in enumerate(splits.split(frame, groups=frame.original_prompt_id)):
        grouper = fit_text_grouper(frame.iloc[fit_ix].reset_index(drop=True), seed=seed + fold)
        pred, conf, _ = reconstruct_groups(grouper, frame.iloc[held_ix].text.astype(str),
                                           frame.iloc[held_ix].original_id.astype(str))
        groups_out[held_ix] = np.asarray([f"oof{fold}::{x}" for x in pred], dtype=object)
        confidence[held_ix] = conf
    return groups_out.astype(str), confidence


def score_candidate_model(model, probs, *, groups=None, group_conf=None,
                          group_features=False, feature_columns=None):
    x = evidence_features(probs, np.repeat(LABELS[0], len(probs)), groups=groups,
                          group_conf=group_conf, include_group=group_features)
    if feature_columns is not None: x = x[feature_columns]
    scores = model.predict_proba(x)[:, 1].reshape(len(probs), len(LABELS))
    return scores


def _hungarian_predictions(scores, groups):
    return hungarian(scores, groups)


def _scenario_metrics(scenarios, predictions: dict[str, np.ndarray], truth, group_sizes):
    out = {}
    truth = np.asarray(truth, dtype=str)
    for name, pred in predictions.items():
        parts = []
        for scenario in sorted(set(scenarios)):
            ix = np.flatnonzero(np.asarray(scenarios) == scenario)
            parts.append({"scenario_id": scenario, **_metric_bundle(truth[ix], np.asarray(pred)[ix])})
        acc = np.asarray([p["accuracy"] for p in parts])
        out[name] = {"mean_scenario_accuracy": float(acc.mean()),
            "minimum_scenario_accuracy": float(acc.min()),
            "p10_scenario_accuracy": float(np.quantile(acc, .10)),
            "scenario_accuracy_std": float(acc.std(ddof=0)),
            "pooled": _metric_bundle(truth, pred), "scenario_metrics": parts,
            "accuracy_by_true_group_size": {
              str(k): float(accuracy_score(truth[np.asarray(group_sizes)==k], np.asarray(pred)[np.asarray(group_sizes)==k]))
              for k in sorted(set(np.asarray(group_sizes).tolist()))}}
    return out


def run_external_experiment(ext_groups: pd.DataFrame, v0_group: pd.DataFrame,
                            external_prob: pd.DataFrame, v0_prob: pd.DataFrame,
                            *, out_dir=OUT):
    """Fit all learnable components externally, then evaluate only on V0 scenarios."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cp_ids = [c for c in external_prob.columns if c.startswith("prob::")]
    class_to_ix = {lab: i for i, lab in enumerate(LABELS)}
    ext = ext_groups.merge(external_prob, on="original_id", validate="one_to_one")
    v0 = v0_group.merge(v0_prob, on="original_id", validate="many_to_one")
    cp_cols = [[f"prob::{lab}::{cp}" for lab in LABELS] for cp in []]
    checkpoint_names = list(CHECKPOINTS)
    checkpoint_provenance = {}
    for model_name, experiment_id in CHECKPOINTS.items():
        manifest_path = _manifest_path(experiment_id)
        model_manifest = json.loads(manifest_path.read_text())
        checkpoint_provenance[model_name] = {"experiment_id": experiment_id,
            "manifest_sha256": sha256(manifest_path),
            "model_revision": model_manifest.get("model_revision"),
            "training_dataset": model_manifest.get("training_dataset"),
            "training_dataset_sha256": model_manifest.get("training_dataset_sha256"),
            "checkpoint_path": model_manifest.get("selected_checkpoint_path") or model_manifest.get("checkpoint_path"),
            "checkpoint_selection_scope": model_manifest.get("checkpoint_selection_scope")}
    availability_path = ROOT / "artifacts/analysis/e25/selection_rosters.json"
    roster_ids = set()
    if availability_path.is_file():
        roster_json = json.loads(availability_path.read_text())
        roster_ids = {str(cid) for fold in roster_json.values()
                      for cid in fold.get("final_roster", [])}
    e22_config_path = ROOT / "artifacts/analysis/e22/configuration.json"
    e22_ids = set(json.loads(e22_config_path.read_text()).get("checkpoint_cache_sha256", {})) if e22_config_path.is_file() else set()
    trust_path = ROOT / "docs/trust-list-2-v039.csv"
    trust = pd.read_csv(trust_path).drop_duplicates("sequence_id").set_index("sequence_id") if trust_path.is_file() else pd.DataFrame()
    selected_by_experiment = {v: k for k,v in CHECKPOINTS.items()}
    availability_rows = []
    for candidate in sorted(roster_ids | e22_ids):
        row = trust.loc[candidate] if candidate in trust.index else {}
        expid = str(row.get("experiment_id", "")) if hasattr(row, "get") else ""
        cp_name = selected_by_experiment.get(expid)
        availability_rows.append({"checkpoint_id": candidate, "experiment_id": expid,
            "model": str(row.get("model", "")) if hasattr(row,"get") else "",
            "training_dataset": str(row.get("training_dataset", "")) if hasattr(row,"get") else "",
            "e22_b_member": candidate in e22_ids,
            "e25_final_roster_member": candidate in roster_ids,
            "external_prediction_available": bool(cp_name is not None),
            "external_prediction_source": f"external_probabilities_{cp_name}.parquet" if cp_name else "",
            "status": "evaluated; no train-ID or normalized-text overlap with V2.16 increment" if cp_name else "not evaluated; no external prediction cache, and no V0 probabilities substituted"})
    pd.DataFrame(availability_rows).to_csv(out_dir / "checkpoint_availability.csv", index=False)
    def tensor(frame):
        return np.stack([frame[[f"prob::{lab}::{cp}" for lab in LABELS]].to_numpy(np.float32)
                         for cp in checkpoint_names], axis=1)
    ep = tensor(ext); vp = tensor(v0)
    yext = ext.model.astype(str).to_numpy(); yv0 = v0.model.astype(str).to_numpy()

    # Prompt-disjoint external holdout drives the only policy selection.
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    fit_ix, tune_ix = next(splitter.split(np.zeros(len(ext)), yext, groups=ext.original_prompt_id))
    external_grouper = fit_text_grouper(ext.iloc[fit_ix].reset_index(drop=True))
    tune_text = ext.iloc[tune_ix].text.astype(str).tolist()
    tune_ids = ext.iloc[tune_ix].original_id.astype(str).to_numpy()
    pair_threshold = tune_text_grouper_threshold(external_grouper,
                                                 ext.iloc[tune_ix].reset_index(drop=True))
    external_grouper["threshold"] = pair_threshold["threshold"]
    tune_groups, tune_conf, tune_edges = reconstruct_groups(external_grouper, tune_text, tune_ids)
    tune_true_groups = ext.iloc[tune_ix].original_prompt_id.astype(str).to_numpy()
    tune_edges_frame = pd.DataFrame(tune_edges, columns=["left_index", "right_index", "confidence"])
    grouper_metrics = {"external_pair_f1": external_grouper["external_pair_f1"],
        "selected_pair_threshold": external_grouper["threshold"],
        "heldout_pair_threshold_metrics": pair_threshold,
        "tune_group_count": int(len(set(tune_groups))),
        "tune_pair_precision_recall": {}}
    # OOF reconstruction prevents the supervised pair grouper from supplying
    # in-sample prompt labels as training features to the group-aware classifier.
    fit_frame = ext.iloc[fit_ix].reset_index(drop=True)
    fit_oof_groups, fit_oof_conf = oof_reconstructed_groups(fit_frame)
    tune_p = ep[tune_ix]
    grid = [(4, 300), (6, 500)]
    def select_model(group_features=False, oracle_train=False):
        best = None
        for depth, iterations in grid:
            if group_features and oracle_train:
                tr_groups = fit_frame.original_prompt_id.astype(str).to_numpy()
                tr_conf = np.ones(len(fit_frame), dtype=np.float32)
            elif group_features:
                tr_groups, tr_conf = fit_oof_groups, fit_oof_conf
            else:
                tr_groups, tr_conf = None, None
            model, columns = fit_candidate_model(ep[fit_ix], yext[fit_ix],
                groups=tr_groups, group_conf=tr_conf, group_features=group_features,
                depth=depth, iterations=iterations)
            tune_scores = score_candidate_model(model, tune_p, groups=tune_groups,
                group_conf=tune_conf, group_features=group_features, feature_columns=columns)
            score = balanced_accuracy_score(yext[tune_ix], np.asarray(LABELS)[tune_scores.argmax(axis=1)])
            result = (float(score), int(depth), int(iterations), model, columns, tune_scores)
            if best is None or result[0] > best[0]: best = result
        return best
    best_ind = select_model(False)
    best_group = select_model(True)
    best_oracle = select_model(True, oracle_train=True)
    # Confidence gates are selected by external held-out balanced accuracy.
    def select_gate(scores):
        independent = scores.argmax(axis=1)
        assigned = _hungarian_predictions(scores, tune_groups)
        best = (-1.0, .5, independent, assigned)
        for cutoff in np.linspace(.1, .95, 18):
            active = np.asarray([np.mean(tune_conf[np.flatnonzero(tune_groups == g)]) >= cutoff
                                 for g in tune_groups])
            pred = np.where(active, assigned, independent)
            metric = balanced_accuracy_score(yext[tune_ix], np.asarray(LABELS)[pred])
            if metric > best[0]: best = (float(metric), float(cutoff), independent, assigned)
        return best
    gate_ind = select_gate(best_ind[5]); gate_group = select_gate(best_group[5])
    gate = gate_ind[1]; gate_group_threshold = gate_group[1]
    tune_ind_pred = best_ind[5].argmax(axis=1)
    tune_group_pred = best_group[5].argmax(axis=1)
    tune_oracle_pred = best_oracle[5].argmax(axis=1)
    tune_ind_h = _hungarian_predictions(best_ind[5], tune_groups)
    tune_true_h = _hungarian_predictions(best_ind[5], tune_true_groups)
    tune_group_h = _hungarian_predictions(best_group[5], tune_groups)
    tune_active = np.asarray([np.mean(tune_conf[np.flatnonzero(tune_groups == g)]) >= gate for g in tune_groups])
    tune_active_group = np.asarray([np.mean(tune_conf[np.flatnonzero(tune_groups == g)]) >= gate_group_threshold for g in tune_groups])
    tune_ind_gate = np.where(tune_active, tune_ind_h, tune_ind_pred)
    tune_group_gate = np.where(tune_active_group, tune_group_h, tune_group_pred)
    tune_family_scores = ep[tune_ix].mean(axis=1)
    tune_family_pred = tune_family_scores.argmax(axis=1)
    tune_family_h = _hungarian_predictions(tune_family_scores, tune_groups)
    tune_results = {
        "E26-A independent CatBoost": tune_ind_pred,
        "E26-B oracle Hungarian": tune_true_h,
        "E26-C reconstructed Hungarian": tune_ind_h,
        "E26-D confidence-gated Hungarian": tune_ind_gate,
        "E26-E group-aware CatBoost": tune_group_pred,
        "E26-F group-aware + gated Hungarian": tune_group_gate,
        "E26-G oracle-trained/reconstructed inference": tune_oracle_pred,
        "family-balanced control": tune_family_pred,
        "family-balanced + reconstructed Hungarian": tune_family_h}
    external_holdout_results = {name: _metric_bundle(yext[tune_ix], np.asarray(LABELS)[pred])
                                for name,pred in tune_results.items()}
    external_tune_frame = ext.iloc[tune_ix][["original_id", "original_prompt_id", "model"]].copy()
    external_tune_frame["reconstructed_group_id"] = tune_groups
    for name,pred in tune_results.items():
        external_tune_frame["pred::" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")] = np.asarray(LABELS)[pred]
    external_tune_frame.to_parquet(out_dir / "external_tune_predictions.parquet", index=False)

    # Fit final text grouper and candidate scorers on all external prompts.
    final_grouper = fit_text_grouper(ext)
    final_grouper["threshold"] = pair_threshold["threshold"]
    ext_rec, ext_conf = oof_reconstructed_groups(ext)
    v0_scen_groups = []
    v0_conf = np.zeros(len(v0), dtype=np.float32)
    for scenario, ix in v0.groupby("scenario_id", sort=True).groups.items():
        ix = np.asarray(list(ix), dtype=np.int64)
        g, c, _ = reconstruct_groups(final_grouper, v0.loc[ix, "text"].astype(str),
                                     v0.loc[ix, "original_id"].astype(str))
        v0_scen_groups.append((ix, g)); v0_conf[ix] = c
    model_ind, col_ind = fit_candidate_model(ep, yext, depth=best_ind[1], iterations=best_ind[2])
    scores_ind = score_candidate_model(model_ind, vp, feature_columns=col_ind)
    model_grp, col_grp = fit_candidate_model(ep, yext, groups=ext_rec, group_conf=ext_conf,
        group_features=True, depth=best_group[1], iterations=best_group[2])
    # Oracle-trained ablation uses true external prompt groups; held-out V0 still
    # gets reconstructed group features, explicitly exposing train/serve mismatch.
    model_oracle, col_oracle = fit_candidate_model(ep, yext, groups=ext.original_prompt_id.astype(str).to_numpy(),
        group_conf=np.ones(len(ext)), group_features=True, depth=best_oracle[1], iterations=best_oracle[2])
    # Build V0 features scenario-by-scenario, never across scenario boundaries.
    scores_group = np.zeros_like(scores_ind); scores_oracle = np.zeros_like(scores_ind)
    for ix, gr in v0_scen_groups:
        conf = v0_conf[ix]
        scores_group[ix] = score_candidate_model(model_grp, vp[ix], groups=gr,
            group_conf=conf, group_features=True, feature_columns=col_grp)
        scores_oracle[ix] = score_candidate_model(model_oracle, vp[ix], groups=gr,
            group_conf=conf, group_features=True, feature_columns=col_oracle)

    pred_ind = scores_ind.argmax(axis=1)
    pred_oracle = np.zeros(len(v0), dtype=np.int64)
    pred_recon = pred_ind.copy(); pred_gate = pred_ind.copy()
    pred_e = scores_group.argmax(axis=1); pred_f = pred_e.copy()
    pred_g = scores_oracle.argmax(axis=1)
    oracle_group_sizes = v0.intended_group_size.to_numpy(np.int64)
    rec_flat = np.empty(len(v0), dtype=object)
    for ix, g in v0_scen_groups:
        rec_flat[ix] = g
        pred_oracle[ix] = _hungarian_predictions(scores_ind[ix], v0.loc[ix, "group_id"].astype(str).to_numpy())
        pred_recon[ix] = _hungarian_predictions(scores_ind[ix], g)
        for group in sorted(set(g)):
            loc = np.flatnonzero(g == group)
            enabled = float(np.mean(v0_conf[ix[loc]])) >= gate
            enabled_group = float(np.mean(v0_conf[ix[loc]])) >= gate_group_threshold
            if enabled:
                pred_gate[ix[loc]] = _hungarian_predictions(scores_ind[ix[loc]], g[loc])
            if enabled_group:
                pred_f[ix[loc]] = _hungarian_predictions(scores_group[ix[loc]], g[loc])
    # Family-balanced control: checkpoint models are three distinct families,
    # therefore this is the equal mean of their probability vectors.
    family_scores = vp.mean(axis=1)
    pred_family = family_scores.argmax(axis=1)
    pred_family_h = pred_family.copy()
    for ix, gr in v0_scen_groups:
        pred_family_h[ix] = _hungarian_predictions(family_scores[ix], gr)
    pred_map = {"E26-A independent CatBoost": np.asarray(LABELS)[pred_ind],
        "E26-B oracle Hungarian": np.asarray(LABELS)[pred_oracle],
        "E26-C reconstructed Hungarian": np.asarray(LABELS)[pred_recon],
        "E26-D confidence-gated Hungarian": np.asarray(LABELS)[pred_gate],
        "E26-E group-aware CatBoost": np.asarray(LABELS)[pred_e],
        "E26-F group-aware + gated Hungarian": np.asarray(LABELS)[pred_f],
        "E26-G oracle-trained/reconstructed inference": np.asarray(LABELS)[pred_g],
        "family-balanced control": np.asarray(LABELS)[pred_family],
        "family-balanced + reconstructed Hungarian": np.asarray(LABELS)[pred_family_h]}
    metrics = _scenario_metrics(v0.scenario_id.astype(str).to_numpy(), pred_map,
                                yv0, oracle_group_sizes)
    # Complete V0 is a distinct reference; it was not used in any fitting.
    full_v0 = pd.read_parquet(V0_PATH)
    full_prob = v0_prob.drop_duplicates("original_id").merge(
        full_v0[["id", "text", "model", "prompt_id"]].rename(columns={"id":"original_id"}),
        on="original_id", validate="one_to_one")
    if len(full_prob) != len(full_v0):
        raise ValueError("full V0 prediction cache did not align to all official rows")
    full_tensor = tensor(full_prob)
    full_text = full_prob.text.astype(str).tolist(); full_ids = full_prob.original_id.astype(str).to_numpy()
    full_rec, full_conf, _ = reconstruct_groups(final_grouper, full_text, full_ids)
    full_ind = score_candidate_model(model_ind, full_tensor, feature_columns=col_ind)
    full_group = score_candidate_model(model_grp, full_tensor, groups=full_rec,
        group_conf=full_conf, group_features=True, feature_columns=col_grp)
    full_oracle_train = score_candidate_model(model_oracle, full_tensor, groups=full_rec,
        group_conf=full_conf, group_features=True, feature_columns=col_oracle)
    full_ind_pred = full_ind.argmax(axis=1)
    full_recon_pred = _hungarian_predictions(full_ind, full_rec)
    full_group_pred = full_group.argmax(axis=1)
    full_gate_pred = full_ind_pred.copy(); full_group_gate = full_group_pred.copy()
    for group in sorted(set(full_rec)):
        ix = np.flatnonzero(full_rec == group)
        active = float(np.mean(full_conf[ix])) >= gate
        active_group = float(np.mean(full_conf[ix])) >= gate_group_threshold
        if active:
            full_gate_pred[ix] = _hungarian_predictions(full_ind[ix], full_rec[ix])
        if active_group:
            full_group_gate[ix] = _hungarian_predictions(full_group[ix], full_rec[ix])
    full_true_group = full_prob.prompt_id.astype(str).to_numpy()
    full_oracle_pred = _hungarian_predictions(full_ind, full_true_group)
    full_family = full_tensor.mean(axis=1); full_family_pred = full_family.argmax(axis=1)
    full_family_h = _hungarian_predictions(full_family, full_rec)
    full_labels = full_prob.model.astype(str).to_numpy()
    full_prediction_map = {"E26-A independent CatBoost": full_ind_pred,
        "E26-B oracle Hungarian": full_oracle_pred,
        "E26-C reconstructed Hungarian": full_recon_pred,
        "E26-D confidence-gated Hungarian": full_gate_pred,
        "E26-E group-aware CatBoost": full_group_pred,
        "E26-F group-aware + gated Hungarian": full_group_gate,
        "E26-G oracle-trained/reconstructed inference": full_oracle_train.argmax(axis=1),
        "family-balanced control": full_family_pred,
        "family-balanced + reconstructed Hungarian": full_family_h}
    full_pred_map = {name: _metric_bundle(full_labels, np.asarray(LABELS)[pred])
                     for name, pred in full_prediction_map.items()}
    full_pred_frame = full_prob[["original_id", "prompt_id", "model"]].copy()
    full_pred_frame["reconstructed_group_id"] = full_rec
    for name, pred in full_prediction_map.items():
        full_pred_frame["pred::" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")] = np.asarray(LABELS)[pred]
    full_pred_frame.to_parquet(out_dir / "full_v0_reference_predictions.parquet", index=False)
    # Group reconstruction diagnostics against randomized intended groups.
    true_groups = v0.group_id.astype(str).to_numpy(); rec_groups = rec_flat.astype(str)
    pair_true = []; pair_pred = []
    for scenario, ix_values in v0.groupby("scenario_id", sort=True).groups.items():
        ix = np.asarray(list(ix_values), dtype=np.int64)
        for i in range(len(ix)):
            for j in range(i+1, len(ix)):
                a,b=ix[i],ix[j]
                same_true = true_groups[a] == true_groups[b]
                same_pred = rec_groups[a] == rec_groups[b]
                pair_true.append(same_true); pair_pred.append(same_pred)
    tp = sum(a and b for a,b in zip(pair_true,pair_pred)); fp = sum((not a) and b for a,b in zip(pair_true,pair_pred)); fn = sum(a and (not b) for a,b in zip(pair_true,pair_pred))
    group_diag = {"pair_precision": tp / max(1,tp+fp), "pair_recall": tp/max(1,tp+fn),
                  "pair_tp": int(tp), "pair_fp": int(fp), "pair_fn": int(fn),
                  "external_heldout_gate_threshold": gate,
                  "external_heldout_gate_accuracy": gate_ind[0]}
    # Save portable, reproducible scores and row predictions.
    pred_frame = v0[["scenario_id","scenario_row_id","original_id","original_prompt_id","group_id","model","intended_group_size"]].copy()
    for k, name in enumerate(checkpoint_names):
        for c, lab in enumerate(LABELS): pred_frame[f"prob::{lab}::{name}"] = vp[:, k, c]
    for name, pred in pred_map.items(): pred_frame["pred::" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")] = pred
    pred_frame.to_parquet(out_dir / "v0_group_predictions.parquet", index=False)
    np.savez_compressed(out_dir / "v0_group_score_matrices.npz", independent=scores_ind,
                        group_aware=scores_group, oracle_group_trained=scores_oracle,
                        family_balanced=family_scores, reconstructed_groups=rec_groups)
    model_ind.save_model(str(out_dir / "e26_candidate_independent.cbm"))
    model_grp.save_model(str(out_dir / "e26_candidate_group_aware.cbm"))
    model_oracle.save_model(str(out_dir / "e26_candidate_oracle_group_train.cbm"))
    # Correction accounting isolates the change introduced by each assignment.
    correction_counts = {}
    independent = pred_map["E26-A independent CatBoost"]
    truth = yv0
    for name, pred in pred_map.items():
        if "Hungarian" in name or "gated" in name:
            changed = np.asarray(pred) != np.asarray(independent)
            correction_counts[name] = {"changed": int(changed.sum()),
                "corrected": int(np.sum(changed & (np.asarray(pred) == truth))),
                "introduced": int(np.sum(changed & (np.asarray(independent) == truth))),
                "net_correct": int(np.sum(changed & (np.asarray(pred) == truth)) - np.sum(changed & (np.asarray(independent) == truth)))}
    rec_size = pd.Series(rec_flat).map(pd.Series(rec_flat).value_counts()).to_numpy()
    reconstructed_size_metrics = {}
    for name, pred in pred_map.items():
        reconstructed_size_metrics[name] = {str(int(k)): float(accuracy_score(truth[rec_size == k], np.asarray(pred)[rec_size == k]))
            for k in sorted(set(rec_size.tolist()))}
    report = {"checkpoint_ids": checkpoint_names, "historical_v0_exposure": True,
        "checkpoint_provenance": checkpoint_provenance,
        "probability_artifact_hashes": {
            "external": {name: sha256(out_dir / f"external_probabilities_{name}.parquet") for name in checkpoint_names},
            "external_merged": sha256(out_dir / "external_checkpoint_probabilities.parquet"),
            "v0_validation": {name: sha256(ROOT / "artifacts/predictions" / f"{CHECKPOINTS[name]}_validation.parquet") for name in checkpoint_names}},
        "external_training_rows": int(len(ext)), "external_prompt_groups": int(ext.original_prompt_id.nunique()),
        "external_fit_prompt_groups": int(ext.iloc[fit_ix].original_prompt_id.nunique()),
        "external_tune_prompt_groups": int(ext.iloc[tune_ix].original_prompt_id.nunique()),
        "grouper": {k:v for k,v in final_grouper.items() if k in {"threshold","external_pair_f1","feature_names"}},
        "external_group_threshold_validation": grouper_metrics,
        "group_diagnostics": group_diag, "randomized_v0_scenarios": metrics,
        "accuracy_by_reconstructed_group_size": reconstructed_size_metrics,
        "assignment_changes_vs_independent": correction_counts,
        "full_v0_reference": full_pred_map,
        "external_heldout_results": external_holdout_results,
        "external_model_selection": {
            "prompt_split_seed": SEED,
            "candidate_grid": [{"depth": d, "iterations": it} for d,it in grid],
            "independent_selected": {"depth": best_ind[1], "iterations": best_ind[2], "heldout_balanced_accuracy": best_ind[0]},
            "group_aware_selected": {"depth": best_group[1], "iterations": best_group[2], "heldout_balanced_accuracy": best_group[0]},
            "oracle_group_train_selected": {"depth": best_oracle[1], "iterations": best_oracle[2], "heldout_balanced_accuracy_on_reconstructed_features": best_oracle[0]},
            "independent_gate_threshold": gate, "independent_gated_heldout_balanced_accuracy": gate_ind[0],
            "group_aware_gate_threshold": gate_group_threshold, "group_aware_gated_heldout_balanced_accuracy": gate_group[0]},
        "notes": ["V0/v0_group was not used to fit or tune any component.",
                  "v0_group scenarios are correlated repeated samples of the official V0 rows.",
                  "The checkpoint base models were selected using V0 in their original training runs."]}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "configuration.json").write_text(json.dumps({"seed": SEED, "scenario_count": N_SCENARIOS,
        "external_dataset": "v2.16OP-RandomGroups", "validation_dataset": "v0_group",
        "models": checkpoint_names, "checkpoint_provenance": checkpoint_provenance,
        "checkpoint_feature_order": list(LABELS),
        "assignment": "maximum score Hungarian, deterministic group ordering",
        "gate_threshold_selected_external_only": gate}, indent=2) + "\n")
    return report


def _manifest_path(experiment_id):
    return ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"


def generate_external_probabilities(ext: pd.DataFrame, *, out_dir=OUT):
    """Run selected V2.16 checkpoints on verified-unseen external responses."""
    from .inference import predict_candidate
    import torch

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Fail closed if a future data refresh introduces any training overlap.
    train = pd.read_parquet(ROOT / "data/processed/dataset_v2.16.parquet", columns=["id", "text"])
    train_ids = set(train.id.astype(str)); train_text = {_normal_text(x) for x in train.text.astype(str)}
    overlap_ids = ext.id.astype(str).isin(train_ids).sum()
    overlap_text = ext.text.astype(str).map(_normal_text).isin(train_text).sum()
    if overlap_ids or overlap_text:
        raise ValueError(f"candidate checkpoints trained on overlapping rows: id={overlap_ids}, text={overlap_text}")
    texts = ext.text.astype(str).tolist(); ids = ext.id.astype(str).tolist()
    outputs = []
    for short_name, experiment_id in CHECKPOINTS.items():
        destination = out_dir / f"external_probabilities_{short_name}.parquet"
        if destination.is_file():
            cached = pd.read_parquet(destination)
            cache_columns = [f"prob::{lab}::{short_name}" for lab in LABELS]
            cache_values = cached[cache_columns].to_numpy(np.float32) if set(cache_columns).issubset(cached.columns) else np.empty((0,0))
            if (cached.original_id.astype(str).tolist() == ids and cache_values.shape == (len(ids),len(LABELS))
                    and np.isfinite(cache_values).all() and np.allclose(cache_values.sum(axis=1),1,atol=2e-3)):
                outputs.append(cached); continue
        manifest = json.loads(_manifest_path(experiment_id).read_text())
        selected = Path(str(manifest.get("selected_checkpoint_path") or manifest.get("checkpoint_path")))
        if not selected.is_absolute(): selected = ROOT / selected
        if not selected.exists(): raise FileNotFoundError(f"missing selected checkpoint {selected}")
        config = dict(manifest.get("config", {}))
        spec = {"candidate_id": short_name, "model": manifest["model"],
                "checkpoint": str(selected), "config": config,
                "model_revision": manifest.get("model_revision"),
                "calibration_temperature": float(manifest.get("calibration_temperature", 1.0)),
                "quantization": manifest.get("quantization"),
                "attention_backend": manifest.get("attention_backend")}
        # predict_candidate handles device batching/OOM retry through the shared
        # inference implementation; per-model caches make this resumable.
        probs = predict_candidate(spec, texts, LABELS, ROOT)
        if (probs.shape != (len(ids), len(LABELS)) or not np.isfinite(probs).all()
                or np.any(probs < -1e-6) or not np.allclose(probs.sum(axis=1), 1, atol=2e-3)):
            raise ValueError(f"{short_name} returned invalid aligned probability vectors")
        frame = pd.DataFrame({"original_id": ids, "checkpoint_id": short_name})
        for i, label in enumerate(LABELS): frame[f"prob::{label}::{short_name}"] = probs[:, i]
        frame.to_parquet(destination, index=False)
        outputs.append(frame)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    merged = outputs[0][["original_id"]].copy()
    for part in outputs:
        merged = merged.merge(part.drop(columns=["checkpoint_id"], errors="ignore"),
                              on="original_id", validate="one_to_one")
    merged.to_parquet(out_dir / "external_checkpoint_probabilities.parquet", index=False)
    return merged


def load_cached_v0_probabilities(v0: pd.DataFrame, *, out_dir=OUT):
    """Align only cached V0 probabilities to scenario original IDs."""
    v0_ids = v0["original_id"] if "original_id" in v0 else v0["id"]
    frames = []
    for short_name, experiment_id in CHECKPOINTS.items():
        path = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"V0 prediction cache unavailable for {short_name}: {path}")
        p = pd.read_parquet(path)
        if "id" not in p or not all(f"prob::{lab}" in p for lab in LABELS):
            raise ValueError(f"invalid V0 prediction cache for {short_name}")
        p = p.rename(columns={"id": "original_id", **{f"prob::{lab}": f"prob::{lab}::{short_name}" for lab in LABELS}})
        values=p[[f"prob::{lab}::{short_name}" for lab in LABELS]].to_numpy(np.float32)
        if (not np.isfinite(values).all() or np.any(values < -1e-6)
                or not np.allclose(values.sum(axis=1),1,atol=2e-3)):
            raise ValueError(f"invalid V0 probabilities in checkpoint cache {short_name}")
        frames.append(p[["original_id", *[f"prob::{lab}::{short_name}" for lab in LABELS]]])
    merged = frames[0]
    for part in frames[1:]: merged = merged.merge(part, on="original_id", validate="one_to_one")
    needed = set(v0_ids.astype(str))
    if not needed.issubset(set(merged.original_id.astype(str))):
        raise ValueError("V0 scenario IDs are absent from cached checkpoint predictions")
    return merged


def write_report(metrics, dataset_audit, root=ROOT):
    path = Path(root) / "docs/e26.md"
    hypotheses = metrics.get("randomized_v0_scenarios", {})
    dataset_manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    group_label_counts = dataset_manifest.get("external_group_size_label_counts", {})
    profile_path = ROOT / "data/processed/dataset_profile_v2_16OP_RandomGroups.json"
    ext_profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
    present_labels = set(ext_profile.get("class_counts", {}))
    missing_labels = [label for label in LABELS if label not in present_labels]
    availability_path = OUT / "checkpoint_availability.csv"
    availability = pd.read_csv(availability_path) if availability_path.is_file() else pd.DataFrame()
    e22_total = int(availability.e22_b_member.sum()) if len(availability) and "e22_b_member" in availability else 0
    e25_total = int(availability.e25_final_roster_member.sum()) if len(availability) and "e25_final_roster_member" in availability else 0
    e22_available = int((availability.e22_b_member & availability.external_prediction_available).sum()) if e22_total else 0
    e25_available = int((availability.e25_final_roster_member & availability.external_prediction_available).sum()) if e25_total else 0
    lines = ["# E26 — Ensemble Generalization and Hungarian Assignment Robustness", "",
        "## Data and leakage controls", "",
        f"- V0 scenarios: {N_SCENARIOS} deterministic scenarios, approximately {TARGET_SCENARIO_ROWS} responses each.",
        f"- External training rows: {metrics.get('external_training_rows', 0)} across {metrics.get('external_prompt_groups', 0)} source prompts.",
        f"- Labels represented: {len(present_labels)}/18. Missing labels: {', '.join(missing_labels) if missing_labels else 'none'}.",
        "- V0 was reserved for evaluation; the three available base checkpoints were trained on V2.16 and selected against V0, so their outputs carry historical V0 exposure.",
        f"- The increment profile records V0 ID, normalized-text and prompt-group isolation as true; E26 independently found {dataset_audit.get('v0_id_overlap', 0)} ID, {dataset_audit.get('v0_normalized_text_overlap', 0)} normalized-text and {dataset_audit.get('v0_raw_prompt_id_overlap', 0)} raw prompt-ID overlaps.",
        "- No V0 prediction was substituted for an unavailable external prediction.", "",
        f"- External probability coverage: E22-B {e22_available}/{e22_total}, E25 final rosters {e25_available}/{e25_total}; remaining IDs are listed in `artifacts/analysis/e26/checkpoint_availability.csv` and were not substituted with V0 probabilities.", "",
        "## External dataset group coverage", "",
        "| Group size | Available source prompts | Selected groups | Label coverage |", "|---:|---:|---:|---:|"]
    for size, count in sorted(dataset_audit.get("available_prompt_groups_by_size", {}).items(), key=lambda x:int(x[0])):
        label_count = len(group_label_counts.get(size, {}))
        lines.append(f"| {size} | {count} | {dataset_audit.get('actual_sampled_groups_by_size',{}).get(size,0)} | {label_count}/18 labels |")
    lines += ["", f"Unavailable group sizes: {dataset_audit.get('unavailable_sizes', [])}.",
        "Full per-size, per-label row counts are in `artifacts/analysis/e26/group_size_label_coverage.csv`.",
        "", "## Hypothesis results on randomized V0 scenarios", "",
        "| Strategy | Mean scenario accuracy | Minimum | 10th percentile | Std. dev. | Pooled balanced accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in sorted(hypotheses.items(), key=lambda kv: kv[1]["mean_scenario_accuracy"], reverse=True):
        pooled = value["pooled"]
        lines.append(f"| {name} | {value['mean_scenario_accuracy']:.4f} | {value['minimum_scenario_accuracy']:.4f} | {value['p10_scenario_accuracy']:.4f} | {value['scenario_accuracy_std']:.4f} | {pooled['balanced_accuracy']:.4f} |")
    lines += ["", "## Group reconstruction and interpretation", "",
        f"Reconstructed-pair precision: {metrics.get('group_diagnostics',{}).get('pair_precision',0):.4f}; recall: {metrics.get('group_diagnostics',{}).get('pair_recall',0):.4f}.",
        f"The confidence gate ({metrics.get('external_model_selection',{}).get('independent_gate_threshold',0):.3f}) was selected on an external prompt-held-out partition only.",
        f"The response-pair grouper threshold was also selected on external held-out prompts (pair F1={metrics.get('external_group_threshold_validation',{}).get('heldout_pair_metrics',{}).get('f1',0):.3f}).",
        f"The confidence gates selected externally were {metrics.get('external_model_selection',{}).get('independent_gate_threshold',0):.2f} for D and {metrics.get('external_model_selection',{}).get('group_aware_gate_threshold',0):.2f} for F.",
        "Oracle Hungarian is an assignment diagnostic using the randomized scenario membership, not a deployable inference strategy.",
        "These results compare held-out V0 scenarios as one correlated benchmark, not as 50 independent test sets.",
        "", "## External held-out prompt results", "",
        "| Strategy | Accuracy | Balanced accuracy |", "|---|---:|---:|"]
    for name, value in sorted(metrics.get("external_heldout_results", {}).items(), key=lambda item:item[1]["balanced_accuracy"], reverse=True):
        lines.append(f"| {name} | {value['accuracy']:.4f} | {value['balanced_accuracy']:.4f} |")
    lines += ["", "## Accuracy by true group size (V0 scenarios)", "",
        "| Group size | Independent A | Oracle Hungarian B | Reconstructed C | Group-aware E | Conditional F | Oracle-trained G |", "|---:|---:|---:|---:|---:|---:|---:|"]
    size_metrics = {name: value.get("accuracy_by_true_group_size", {}) for name,value in hypotheses.items()}
    for size in range(1, 19):
        values = [size_metrics.get(name, {}).get(str(size)) for name in [
            "E26-A independent CatBoost", "E26-B oracle Hungarian", "E26-C reconstructed Hungarian",
            "E26-E group-aware CatBoost", "E26-F group-aware + gated Hungarian",
            "E26-G oracle-trained/reconstructed inference"]]
        lines.append(f"| {size} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in values) + " |")
    helps=[]; hurts=[]
    a_metrics=size_metrics.get("E26-A independent CatBoost",{})
    c_metrics=size_metrics.get("E26-C reconstructed Hungarian",{})
    for size in range(1,19):
        if str(size) in a_metrics and str(size) in c_metrics:
            delta=c_metrics[str(size)]-a_metrics[str(size)]
            (helps if delta>0 else hurts if delta<0 else []).append((size,delta))
    help_text=", ".join(f"{k} ({v*100:+.1f} pp)" for k,v in helps) or "none"
    hurt_text=", ".join(f"{k} ({v*100:+.1f} pp)" for k,v in hurts) or "none"
    oracle_ba=hypotheses.get("E26-B oracle Hungarian",{}).get("pooled",{}).get("balanced_accuracy",0)
    recon_ba=hypotheses.get("E26-C reconstructed Hungarian",{}).get("pooled",{}).get("balanced_accuracy",0)
    lines += ["", f"By true group size, reconstructed Hungarian versus independent is higher at sizes {help_text} and lower at sizes {hurt_text}.",
        f"Oracle versus reconstructed Hungarian pooled balanced accuracy differs by {(oracle_ba-recon_ba)*100:+.2f} percentage points."]
    lines += ["", "## Accuracy by reconstructed group size", "",
        "| Reconstructed size | Independent A | Reconstructed C | Conditional D | Group-aware E | Conditional F |", "|---:|---:|---:|---:|---:|---:|"]
    recon_metrics = metrics.get("accuracy_by_reconstructed_group_size", {})
    for size in sorted({int(k) for value in recon_metrics.values() for k in value}):
        vals = [recon_metrics.get(name, {}).get(str(size)) for name in [
            "E26-A independent CatBoost", "E26-C reconstructed Hungarian",
            "E26-D confidence-gated Hungarian", "E26-E group-aware CatBoost",
            "E26-F group-aware + gated Hungarian"]]
        lines.append(f"| {size} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in vals) + " |")
    lines += ["", "## Complete V0 reference (separate from randomized scenarios)", "",
        "| Strategy | Accuracy | Balanced accuracy |", "|---|---:|---:|"]
    for name,value in sorted(metrics.get("full_v0_reference",{}).items(),key=lambda item:item[1]["balanced_accuracy"],reverse=True):
        lines.append(f"| {name} | {value['accuracy']:.4f} | {value['balanced_accuracy']:.4f} |")
    lines += ["", "## Assignment changes and per-label metrics", "",
        "| Variant | Changed | Corrected | Correct lost / introduced | Net correct |", "|---|---:|---:|---:|---:|"]
    for name,item in metrics.get("assignment_changes_vs_independent",{}).items():
        lines.append(f"| {name} | {item['changed']} | {item['corrected']} | {item['introduced']} | {item['net_correct']:+d} |")
    lines += ["", "Per-label precision and recall for all nine variants are listed below.",
        "", "### Per-label precision and recall", ""]
    for name, value in hypotheses.items():
        lines += [f"#### {name}", "", "| Label | Precision | Recall | Support |", "|---|---:|---:|---:|"]
        for label, item in value["pooled"]["per_label"].items():
            lines.append(f"| {label} | {item['precision']:.3f} | {item['recall']:.3f} | {item['support']} |")
        lines.append("")
    external_best = max(metrics.get("external_heldout_results", {}).items(), key=lambda item:item[1]["balanced_accuracy"])[0]
    scenario_best = max(hypotheses.items(), key=lambda item:item[1]["mean_scenario_accuracy"])[0]
    lines += ["## Recommendation", "",
        f"External held-out prompts select **{external_best}** (balanced accuracy {metrics['external_heldout_results'][external_best]['balanced_accuracy']:.4f}). On randomized V0, the highest mean scenario accuracy is **{scenario_best}** ({hypotheses[scenario_best]['mean_scenario_accuracy']:.4f}; pooled balanced accuracy {hypotheses[scenario_best]['pooled']['balanced_accuracy']:.4f}).",
        "The disagreement means there is no deployment-ready winner: only 502 meta-training rows and 15 labels were available, and all three base checkpoints were selected using V0. Treat E26-F as the externally selected research candidate and family-balanced plus reconstructed Hungarian as the V0 robustness control; validate with a larger, V0-independent checkpoint roster before deployment.", ""]
    lines += ["## Assignment and group-size findings", "",
        "See `artifacts/analysis/e26/metrics.json` for corrected/introduced counts, accuracy by true and reconstructed group size, and per-scenario metrics.",
        "", "## Historical comparison", "",
        f"The historical E20-D result was 78.21% V0 balanced accuracy using V0-exposed E20 classifiers. E26's best pooled score on randomized V0 scenarios is {max(value['pooled']['balanced_accuracy'] for value in hypotheses.values()):.2%} (a nominal {max(value['pooled']['balanced_accuracy'] for value in hypotheses.values())-.782058:+.2%} difference). This comparison is not apples-to-apples because the roster and decoder differ, and E20-D's result is V0-exposed. The E20-D advantage therefore does not survive as an independently validated deployment claim.",
        "", "## Limitations", "",
        "The external increment contains only 15 labels and source prompt groups only up to size 12. The official V0 artifact stores prompt IDs but not canonical prompt text, so exact prompt-text comparison is unavailable; the increment's upstream profile separately records accepted prompt groups as disjoint from V0. Only three V2.16-trained checkpoint probability sets were available; their outputs on the external increment were generated only after confirming no source-response overlap with V2.16. Results should be interpreted as a small external meta-training study, not a clean estimate of Kaggle generalization.", ""]
    path.write_text("\n".join(lines))


def build_datasets():
    v0 = pd.read_parquet(V0_PATH)
    ext = pd.read_parquet(EXT_PATH)
    v0_group = build_v0_group(v0)
    ext_group, audit = build_external_random_groups(ext, v0)
    V0_GROUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    v0_group.to_parquet(V0_GROUP_PATH, index=False)
    ext_group.to_parquet(EXT_GROUP_PATH, index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "source_v0": str(V0_PATH.relative_to(ROOT)),
        "source_v0_sha256": sha256(V0_PATH), "source_external": str(EXT_PATH.relative_to(ROOT)),
        "source_external_sha256": sha256(EXT_PATH), "v0_group_rows": len(v0_group),
        "v0_group_scenarios": int(v0_group.scenario_id.nunique()),
        "v0_group_sha256": sha256(V0_GROUP_PATH), "external_group_rows": len(ext_group),
        "external_group_sha256": sha256(EXT_GROUP_PATH), "external_group_audit": audit,
        "external_label_counts": ext_group.model.value_counts().sort_index().to_dict(),
        "external_group_size_label_counts": {str(size): group.model.value_counts().sort_index().to_dict()
            for size, group in ext_group.groupby("group_size")},
        "v0_scenario_row_counts": v0_group.groupby("scenario_id").size().to_dict(),
        "v0_group_size_row_counts": v0_group.groupby("intended_group_size").size().to_dict(),
        "v0_group_size_group_counts": v0_group.groupby("intended_group_size").group_id.nunique().to_dict()}
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    profile = {"dataset_version": "v2_16OP_RandomGroups", "dataset_name": "v2,16OP-RandomGroups",
        "parent_dataset": "v2.16_openpool16_increment", "rows": len(ext_group),
        "training_only": True, "classes": list(LABELS),
        "class_counts": ext_group.model.value_counts().sort_index().to_dict(),
        "source_counts": ext_group.source_dataset.value_counts().sort_index().to_dict(),
        "prompt_groups": int(ext_group.group_id.nunique()), "group_size_counts": ext_group.groupby("group_size").group_id.nunique().to_dict(),
        "source_sha256": manifest["source_external_sha256"], "dataset_sha256": manifest["external_group_sha256"],
        "source_prompt_namespacing": "source_dataset::source_prompt_id", "seed": SEED,
        "v0_overlap": {k: audit[k] for k in ["v0_id_overlap", "v0_normalized_text_overlap", "v0_raw_prompt_id_overlap", "v0_known_prompt_text_overlap", "v0_prompt_text_check", "upstream_v0_prompt_group_isolation_verified"]},
        "limitations": {"unavailable_group_sizes": audit["unavailable_sizes"],
                        "only_present_source_labels_can_be_selected": True}}
    (ROOT / "data/processed/dataset_profile_v2_16OP_RandomGroups.json").write_text(json.dumps(profile, indent=2) + "\n")
    coverage_rows=[]
    for size in sorted(ext_group.group_size.unique()):
        subset=ext_group[ext_group.group_size==size]
        counts=subset.model.astype(str).value_counts()
        group_count=int(subset.group_id.nunique())
        for label in LABELS:
            coverage_rows.append({"group_size":int(size),"group_count":group_count,
                "label":label,"response_count":int(counts.get(label,0)),
                "label_present":bool(counts.get(label,0))})
    pd.DataFrame(coverage_rows).to_csv(OUT/"group_size_label_coverage.csv",index=False)
    return v0, ext_group, v0_group, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--predict-external", action="store_true", help="Generate incremental-dataset probabilities with verified V2.16 checkpoints")
    parser.add_argument("--run", action="store_true", help="Fit external-only E26 meta models and evaluate V0")
    args = parser.parse_args(argv)
    if args.build_data or not any((args.run, args.predict_external)):
        build_datasets()
    if args.predict_external:
        ext = pd.read_parquet(EXT_GROUP_PATH)
        generate_external_probabilities(ext)
    if args.run:
        ext = pd.read_parquet(EXT_GROUP_PATH)
        vg = pd.read_parquet(V0_GROUP_PATH)
        audit = json.loads((OUT / "dataset_manifest.json").read_text())["external_group_audit"]
        ep = pd.read_parquet(OUT / "external_checkpoint_probabilities.parquet")
        full_v0 = pd.read_parquet(V0_PATH)
        vp = load_cached_v0_probabilities(full_v0)
        metrics = run_external_experiment(ext, vg, ep, vp)
        write_report(metrics, audit)
        # Replace planned catalogue record only after all outputs are complete.
        catalog_path = ROOT / "configs/ensemble_catalog_v2.json"
        catalog = json.loads(catalog_path.read_text())
        for record in catalog.get("experiments", []):
            if record.get("id") == "E26":
                record.update(status="Scored", accuracy=metrics["randomized_v0_scenarios"]["E26-A independent CatBoost"]["pooled"]["accuracy"],
                    balanced_accuracy=metrics["randomized_v0_scenarios"]["E26-A independent CatBoost"]["pooled"]["balanced_accuracy"],
                    primary_score=metrics["randomized_v0_scenarios"]["E26-A independent CatBoost"]["mean_scenario_accuracy"],
                    result_source="artifacts/analysis/e26/metrics.json")
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
