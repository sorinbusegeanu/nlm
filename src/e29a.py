"""E29-A: evaluate E25 checkpoint-pruning stability on v0_group.

This is an evaluation-only replay. It uses cached V0 checkpoint probabilities,
the saved E22-B/E25 CatBoost models and the 50 prebuilt V0 scenarios. V0 is a
repeated, correlated sensitivity benchmark, not independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import e20, e22, e25
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e29a"
V0_GROUP = ROOT / "data/processed/dataset_v0_group.parquet"
V0 = ROOT / "data/processed/dataset_v0.parquet"
BUDGETS = (60, 40, 30, 25, 20)
THREADS = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_scenarios(frame: pd.DataFrame, v0: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"scenario_id", "scenario_row_id", "original_id", "original_prompt_id",
                "group_id", "text", "model", "intended_group_size"}
    if not required.issubset(frame.columns):
        raise ValueError(f"v0_group is missing fields: {sorted(required-set(frame.columns))}")
    if frame.scenario_row_id.duplicated().any():
        raise ValueError("scenario row IDs are not unique")
    if set(frame.model.astype(str)) != set(LABELS):
        raise ValueError("v0_group does not contain all canonical labels")
    if v0.id.astype(str).duplicated().any():
        raise ValueError("V0 original IDs are not unique")
    indexed = v0.copy()
    indexed["id"] = indexed.id.astype(str)
    lookup = indexed.set_index("id")
    if not lookup.index.is_unique:
        raise ValueError("V0 IDs are not unique")
    if not set(frame.original_id.astype(str)).issubset(lookup.index):
        raise ValueError("v0_group contains an original_id absent from V0")
    orig = frame.original_id.astype(str).map(lookup.prompt_id.astype(str))
    if not np.array_equal(orig.to_numpy(), frame.original_prompt_id.astype(str).to_numpy()):
        raise ValueError("v0_group original_prompt_id does not match source V0 rows")
    original_label = frame.original_id.astype(str).map(lookup.model.astype(str))
    if not np.array_equal(original_label.to_numpy(), frame.model.astype(str).to_numpy()):
        raise ValueError("v0_group labels do not match source V0 rows")
    original_text = frame.original_id.astype(str).map(lookup.text.astype(str))
    if not np.array_equal(original_text.to_numpy(), frame.text.astype(str).to_numpy()):
        raise ValueError("v0_group text does not match source V0 rows")
    scenario_ids = sorted(frame.scenario_id.astype(str).unique())
    if len(scenario_ids) != 50:
        raise ValueError(f"expected 50 scenarios, found {len(scenario_ids)}")
    group_size_counts = {}
    label_counts = {}
    for sid, part in frame.groupby("scenario_id", sort=True):
        if part.original_id.astype(str).duplicated().any():
            raise ValueError(f"duplicate original_id in scenario {sid}")
        if set(part.model.astype(str)) != set(LABELS):
            raise ValueError(f"scenario {sid} does not include all 18 labels")
        if set(part.intended_group_size.astype(int)) < set(range(1, 19)):
            raise ValueError(f"scenario {sid} misses one or more intended group sizes")
        for gid, group in part.groupby("group_id", sort=False):
            if group.original_prompt_id.astype(str).nunique() != 1:
                raise ValueError(f"true group {sid}/{gid} spans original prompts")
            if group.model.astype(str).duplicated().any():
                raise ValueError(f"true group {sid}/{gid} repeats a label")
            size = int(group.intended_group_size.iloc[0])
            if size != len(group) or not 1 <= size <= 18:
                raise ValueError(f"true group {sid}/{gid} size mismatch")
            group_size_counts[str(size)] = group_size_counts.get(str(size), 0) + 1
        label_counts[sid] = part.model.astype(str).value_counts().to_dict()
    frame = frame.copy().reset_index(drop=True)
    frame["original_id"] = frame.original_id.astype(str)
    return frame, {"scenario_count": len(scenario_ids), "rows": len(frame),
                   "unique_original_ids": int(frame.original_id.nunique()),
                   "all_scenarios_cover_all_labels": True,
                   "all_scenarios_cover_group_sizes_1_to_18": True,
                   "true_groups_by_size": group_size_counts,
                   "scenario_label_counts": label_counts}


def _fold_alignment(ctx):
    fold_for_row = np.full(len(ctx["y"]), -1, dtype=np.int16)
    for fold in ctx["folds"]:
        held = np.asarray(fold["held_rows"], dtype=np.int64)
        if (fold_for_row[held] >= 0).any():
            raise ValueError("saved E25 held-out folds overlap")
        fold_for_row[held] = int(fold["fold"])
    if (fold_for_row < 0).any():
        raise ValueError("saved folds do not cover every V0 row")
    return fold_for_row


def _load_and_verify_scores(ctx, out: Path, fold_for_row: np.ndarray):
    """Load baseline/budget OOF matrices and reproduce E25 saved fold outputs."""
    checkpoint_ids = list(ctx["checkpoint_ids"])
    id_to_idx = {sid: i for i, sid in enumerate(checkpoint_ids)}
    matrices = {}
    baseline_file = np.load(out.parent / "e22" / "predictions_E22-B.npz", allow_pickle=True)
    _verify_aggregate(baseline_file, ctx, "E22-B")
    matrices["E22-B-89"] = baseline_file["scores"].astype(np.float32)
    baseline_saved_h = baseline_file["pred_E19D"].astype(np.int64)
    baseline_h = e20.hungarian(matrices["E22-B-89"], ctx["group_d"], ctx["row_ids"])
    if not np.array_equal(baseline_h, baseline_saved_h):
        raise ValueError("E22-B cached scores do not reproduce saved E19-D predictions")
    if not np.array_equal(matrices["E22-B-89"].argmax(axis=1), baseline_file["pred_independent"].astype(np.int64)):
        raise ValueError("E22-B independent predictions do not match cached scores")

    # Exact replay of full-roster E22-B OOF scores from its saved fold models.
    for fold in ctx["folds"]:
        fold_id, held = int(fold["fold"]), np.asarray(fold["held_rows"], dtype=np.int64)
        model = e20.CatBoostClassifier()
        model.load_model(str(out.parent / "e22" / f"E22-B_fold{fold_id}.cbm"))
        recomputed = e25._scores(model, ctx, held, list(range(len(checkpoint_ids))), list(model.feature_names_))
        if not np.allclose(recomputed, matrices["E22-B-89"][held], rtol=1e-6, atol=1e-7):
            raise ValueError(f"E22-B saved fold model {fold_id} does not reproduce OOF scores")

    rosters = json.loads((out / "selection_rosters.json").read_text())
    for budget in BUDGETS:
        agg = np.load(out / f"predictions_E25_{budget}.npz", allow_pickle=True)
        _verify_aggregate(agg, ctx, f"E25-{budget}")
        score = agg["scores"].astype(np.float32)
        if not np.isfinite(score).all() or score.shape != (len(ctx["y"]), len(LABELS)):
            raise ValueError(f"invalid E25-{budget} score matrix")
        if not np.array_equal(e20.hungarian(score, ctx["group_d"], ctx["row_ids"]),
                              agg["pred_E19D"].astype(np.int64)):
            raise ValueError(f"E25-{budget} aggregate scores fail Hungarian reproduction")
        if not np.array_equal(score.argmax(axis=1), np.load(out / f"predictions_E25_{budget}.npz")["scores"].argmax(axis=1)):
            raise ValueError(f"E25-{budget} independent score alignment mismatch")
        for fold in ctx["folds"]:
            fid, held = int(fold["fold"]), np.asarray(fold["held_rows"], dtype=np.int64)
            fold_npz = np.load(out / f"E25_{budget}_fold{fid}.npz", allow_pickle=True)
            if not np.array_equal(fold_npz["row_ids"].astype(str), ctx["row_ids"][held].astype(str)):
                raise ValueError(f"E25-{budget} fold {fid} row IDs misaligned")
            if not np.allclose(score[held], fold_npz["scores"], rtol=1e-6, atol=1e-7):
                raise ValueError(f"E25-{budget} fold {fid} differs from aggregate matrix")
            if not np.array_equal(e20.hungarian(fold_npz["scores"], ctx["group_d"][held],
                                                ctx["row_ids"][held]), fold_npz["pred"].astype(np.int64)):
                raise ValueError(f"E25-{budget} fold {fid} saved predictions fail reproduction")
            # Recompute from its exact fold-local roster and saved classifier.
            roster = rosters[str(fid)]["target_rosters"][str(budget)]
            if len(roster) != budget or len(set(roster)) != budget:
                raise ValueError(f"invalid E25-{budget} roster for fold {fid}")
            model = e20.CatBoostClassifier()
            model.load_model(str(out / f"E25_{budget}_fold{fid}.cbm"))
            pool = [id_to_idx[sid] for sid in roster]
            replay = e25._scores(model, ctx, held, pool, list(model.feature_names_))
            if not np.allclose(replay, score[held], rtol=1e-6, atol=1e-7):
                raise ValueError(f"E25-{budget} fold model {fid} does not reproduce its score matrix")
        matrices[f"E25-{budget}"] = score
    return matrices


def _verify_aggregate(data, ctx, name):
    if not np.array_equal(data["row_ids"].astype(str), ctx["row_ids"].astype(str)):
        raise ValueError(f"{name} row IDs are not aligned with E22 context")
    if not np.array_equal(data["y"].astype(np.int64), ctx["y"].astype(np.int64)):
        raise ValueError(f"{name} labels are not aligned with E22 context")
    scores = data["scores"]
    if scores.shape != (len(ctx["y"]), len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError(f"{name} score matrix has invalid shape or values")


def _score_cache_path(out, config_id):
    return out / "score_matrices" / f"{config_id}.npz"


def _build_drop_scores(ctx, out: Path, drop_ids: tuple[str, ...], cache_key: str):
    path = _score_cache_path(out, cache_key)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if (np.array_equal(z["row_ids"].astype(str), ctx["row_ids"].astype(str)) and
                tuple(z["dropped_ids"].astype(str)) == tuple(drop_ids) and
                z["scores"].shape == (len(ctx["y"]), len(LABELS)) and np.isfinite(z["scores"]).all()):
            return z["scores"].astype(np.float32)
    n = len(ctx["y"])
    scores = np.full((n, len(LABELS)), np.nan, dtype=np.float32)
    cp_index = {sid: i for i, sid in enumerate(ctx["checkpoint_ids"])}
    pool = [i for sid, i in cp_index.items() if sid not in set(drop_ids)]
    if len(pool) != len(cp_index) - len(set(drop_ids)) or not pool:
        raise ValueError(f"invalid reduced roster {cache_key}")
    e22dir = out.parent / "e22"
    for fold in ctx["folds"]:
        fid, held = int(fold["fold"]), np.asarray(fold["held_rows"], dtype=np.int64)
        model = e20.CatBoostClassifier()
        model.load_model(str(e22dir / f"E22-B_fold{fid}.cbm"))
        scores[held] = e25._scores(model, ctx, held, pool, list(model.feature_names_))
    if not np.isfinite(scores).all():
        raise ValueError(f"incomplete reduced score matrix for {cache_key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, row_ids=ctx["row_ids"], dropped_ids=np.asarray(drop_ids), scores=scores)
    return scores


def build_all_score_matrices(ctx, out: Path, base_matrices):
    matrices = dict(base_matrices)
    ids = list(ctx["checkpoint_ids"])
    for index, sid in enumerate(ids, 1):
        key = f"drop_{sid}"
        matrices[key] = _build_drop_scores(ctx, out, (sid,), key)
        if index % 10 == 0:
            print(f"E29-A cached {index}/{len(ids)} individual-removal score matrices", flush=True)
    clusters = json.loads((out.parent / "e25" / "near_duplicate_clusters.json").read_text())
    for c in clusters:
        cids = tuple(sorted(map(str, c["checkpoint_ids"])))
        matrices[f"drop_{c['cluster_id']}"] = _build_drop_scores(ctx, out, cids, f"drop_{c['cluster_id']}")
    return matrices, clusters


def _scenario_inputs(frame, ctx, fold_for_row):
    id_to_index = {str(sid): i for i, sid in enumerate(ctx["row_ids"])}
    scenarios = []
    for sid, part in frame.groupby("scenario_id", sort=True):
        rows = np.asarray([id_to_index[x] for x in part.original_id.astype(str)], dtype=np.int64)
        if len(set(rows.tolist())) != len(rows):
            raise ValueError(f"scenario {sid} maps multiple views to one V0 row")
        if not np.array_equal(ctx["y"][rows], np.asarray([LABELS.index(x) for x in part.model.astype(str)])):
            raise ValueError(f"scenario {sid} label mapping changed during alignment")
        true_groups = part.group_id.astype(str).to_numpy()
        fold_groups = np.asarray([f"fold{fold_for_row[row]}::{ctx['group_d'][row]}" for row in rows], dtype=str)
        scenarios.append({"scenario_id": str(sid), "source_rows": rows,
                          "scenario_row_ids": part.scenario_row_id.astype(str).to_numpy(),
                          "y": ctx["y"][rows], "true_groups": true_groups,
                          "frozen_groups": fold_groups,
                          "true_group_size": part.intended_group_size.astype(int).to_numpy()})
    return scenarios


def _metric(y, pred):
    labels = np.arange(len(LABELS))
    support = np.bincount(y, minlength=len(LABELS))
    recall = np.divide(np.bincount(y[pred == y], minlength=len(LABELS)), support,
                       out=np.full(len(LABELS), np.nan), where=support > 0)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(np.nanmean(recall)),
            "labels_present": int(np.sum(support > 0)),
            "min_label_support": int(support[support > 0].min()) if np.any(support) else 0}


def decode_scenario(score_matrix, row_ids, true_groups, frozen_groups):
    """Decode only the supplied scenario rows; never combine scenario groups."""
    scores = np.asarray(score_matrix)
    row_ids = np.asarray(row_ids, dtype=str)
    true_groups = np.asarray(true_groups, dtype=str)
    frozen_groups = np.asarray(frozen_groups, dtype=str)
    if scores.shape != (len(row_ids), len(LABELS)):
        raise ValueError("scenario scores and row IDs are misaligned")
    if len(set(row_ids.tolist())) != len(row_ids):
        raise ValueError("scenario row IDs must be unique")
    if not (len(true_groups) == len(frozen_groups) == len(row_ids)):
        raise ValueError("scenario groups are not aligned to scenario rows")
    return {"independent": scores.argmax(axis=1).astype(np.int16),
            "true_group_hungarian": e20.hungarian(scores, true_groups, row_ids),
            "frozen_E19D_subset_hungarian": e20.hungarian(scores, frozen_groups, row_ids)}


def evaluate_scenarios(ctx, scenarios, matrices, clusters):
    config_scores = dict(matrices)
    config_metadata = {k: {"roster_type": "E22-B full" if k == "E22-B-89" else
                           ("E25 budget" if k.startswith("E25-") else
                            ("checkpoint removal" if k.startswith("drop_N") else "cluster removal"))}
                       for k in config_scores}
    methods = ("independent", "true_group_hungarian", "frozen_E19D_subset_hungarian")
    scenario_rows, group_rows, label_rows, prediction_cache, scenario_prediction_cache = [], [], [], {}, {}
    for config_id, score in config_scores.items():
        pred_by_method = {m: np.full(len(ctx["y"]), -1, dtype=np.int16) for m in methods}
        scenario_prediction_cache[config_id] = {}
        for scenario in scenarios:
            ix = scenario["source_rows"]
            local_scores = score[ix]
            if not np.isfinite(local_scores).all():
                raise ValueError(f"nonfinite scenario scores: {config_id}/{scenario['scenario_id']}")
            decoded = decode_scenario(local_scores, scenario["scenario_row_ids"],
                                      scenario["true_groups"], scenario["frozen_groups"])
            for method in methods:
                pred_by_method[method][ix] = decoded[method]
            scenario_prediction_cache[config_id][scenario["scenario_id"]] = {
                "row_ids": scenario["scenario_row_ids"], "y": scenario["y"],
                "original_ids": ctx["row_ids"][ix].astype(str), **decoded}
            for method in methods:
                pred = pred_by_method[method][ix]
                metric = _metric(scenario["y"], pred)
                baseline_config = "E22-B-89"
                base_pred = None
                if config_id != baseline_config:
                    base_score = config_scores[baseline_config][ix]
                    if method == "independent":
                        base_pred = base_score.argmax(axis=1)
                    elif method == "true_group_hungarian":
                        base_pred = e20.hungarian(base_score, scenario["true_groups"], scenario["scenario_row_ids"])
                    else:
                        base_pred = e20.hungarian(base_score, scenario["frozen_groups"], scenario["scenario_row_ids"])
                else:
                    base_pred = pred
                corrected = int(np.sum((base_pred != scenario["y"]) & (pred == scenario["y"])))
                introduced = int(np.sum((base_pred == scenario["y"]) & (pred != scenario["y"])))
                scenario_rows.append({"configuration": config_id, "method": method,
                    "scenario_id": scenario["scenario_id"], **metric,
                    "corrected_vs_89": corrected, "introduced_vs_89": introduced,
                    "net_correct_vs_89": corrected-introduced})
                for size in range(1, 19):
                    mask = scenario["true_group_size"] == size
                    if not mask.any():
                        continue
                    gm = _metric(scenario["y"][mask], pred[mask])
                    group_rows.append({"configuration": config_id, "method": method,
                        "scenario_id": scenario["scenario_id"], "true_group_size": size,
                        "rows": int(mask.sum()), **gm})
        prediction_cache[config_id] = pred_by_method
    scenario_df = pd.DataFrame(scenario_rows)
    group_df = pd.DataFrame(group_rows)
    # Per-label results pooled over scenario rows. This is descriptive only;
    # repeated V0 records are not independent observations.
    for config_id, scenarios_pred in scenario_prediction_cache.items():
        for method in methods:
            for label_idx, label in enumerate(LABELS):
                tp = fp = support = 0
                for scenario_record in scenarios_pred.values():
                    y = np.asarray(scenario_record["y"],dtype=np.int64)
                    p = np.asarray(scenario_record[method],dtype=np.int64)
                    support += int((y == label_idx).sum())
                    tp += int(((y == label_idx) & (p == label_idx)).sum())
                    fp += int(((y != label_idx) & (p == label_idx)).sum())
                label_rows.append({"configuration": config_id, "method": method,
                    "label": label, "support_repeated_rows": support, "true_positive": tp,
                    "false_positive": fp, "precision": tp/max(1,tp+fp), "recall": tp/max(1,support)})
    return scenario_df, group_df, pd.DataFrame(label_rows), prediction_cache, scenario_prediction_cache


def summarize_rosters(scenario_df: pd.DataFrame):
    rows = []
    baseline = {(method, scenario): row for (method, scenario), row in
                scenario_df[scenario_df.configuration == "E22-B-89"].set_index(
                    ["method", "scenario_id"]).iterrows()}
    for (config, method), group in scenario_df.groupby(["configuration", "method"], sort=False):
        values = group.balanced_accuracy.to_numpy(float)
        deltas = np.asarray([float(row.balanced_accuracy - baseline[(method,row.scenario_id)].balanced_accuracy)
                             for row in group.itertuples()], dtype=float)
        rows.append({"configuration": config, "method": method,
            "scenario_count": len(values), "mean_balanced_accuracy": float(values.mean()),
            "std_balanced_accuracy": float(values.std(ddof=1)),
            "minimum_balanced_accuracy": float(values.min()),
            "p10_balanced_accuracy": float(np.quantile(values, .10)),
            "mean_balanced_accuracy_delta_vs_89": float(deltas.mean()),
            "scenarios_balanced_accuracy_improved_vs_89": int((deltas > 0).sum()),
            "scenarios_balanced_accuracy_worsened_vs_89": int((deltas < 0).sum()),
            "mean_accuracy": float(group.accuracy.mean()),
            "mean_corrected_vs_89": float(group.corrected_vs_89.mean()),
            "mean_introduced_vs_89": float(group.introduced_vs_89.mean()),
            "mean_net_correct_vs_89": float(group.net_correct_vs_89.mean()),
            "scenarios_improved_vs_89": int((group.net_correct_vs_89 > 0).sum())})
    return pd.DataFrame(rows)


def checkpoint_stability(ctx, scenario_df, prediction_cache, rankings):
    original = {x["checkpoint_id"]: x for x in rankings}
    result = []
    base_methods = prediction_cache["E22-B-89"]
    cp_ids = list(ctx["checkpoint_ids"])
    scenario_ids = scenario_df.scenario_id.unique()
    indexed = {(config, method): part.set_index("scenario_id")
               for (config, method), part in scenario_df.groupby(["configuration", "method"], sort=False)}
    scenario_rankings = {}
    for method in base_methods:
        scenario_rankings[method] = {}
        for scenario_id in scenario_ids:
            baseline_ba = indexed[("E22-B-89", method)].loc[scenario_id, "balanced_accuracy"]
            values = [(float(baseline_ba-indexed[(f"drop_{cp}",method)].loc[scenario_id,"balanced_accuracy"]), cp)
                      for cp in cp_ids]
            values.sort(key=lambda x: (-x[0], x[1]))
            scenario_rankings[method][scenario_id] = {cp: rank+1 for rank,(_,cp) in enumerate(values)}
    for method in base_methods:
        bdf = indexed[("E22-B-89", method)]
        for sid in cp_ids:
            ddf = indexed[(f"drop_{sid}", method)]
            losses, ranks, improves = [], [], []
            for scenario_id in scenario_ids:
                loss = float(bdf.loc[scenario_id, "balanced_accuracy"] - ddf.loc[scenario_id, "balanced_accuracy"])
                losses.append(loss)
                improves.append(loss < 0)
            ranks = [scenario_rankings[method][s][sid] for s in scenario_ids]
            result.append({"checkpoint_id": sid, "decoder": method,
                "original_e25_importance_rank": 1 + sorted(original, key=lambda k: (-float(original[k]["balanced_accuracy_loss"]),k)).index(sid),
                "original_e25_ba_loss": float(original[sid]["balanced_accuracy_loss"]),
                "mean_scenario_ba_loss_when_removed": float(np.mean(losses)),
                "std_scenario_ba_loss_when_removed": float(np.std(losses,ddof=1)),
                "p10_ba_loss": float(np.quantile(losses,.1)), "p90_ba_loss": float(np.quantile(losses,.9)),
                "mean_scenario_importance_rank": float(np.mean(ranks)),
                "std_scenario_importance_rank": float(np.std(ranks,ddof=1)),
                "min_scenario_rank": int(min(ranks)), "max_scenario_rank": int(max(ranks)),
                "removal_improves_scenario_pct": float(np.mean(improves)),
                "spearman_vs_original_rank": float(spearmanr(
                    [1+sorted(original,key=lambda k:(-float(original[k]["balanced_accuracy_loss"]),k)).index(cp) for cp in cp_ids],
                    [np.mean([scenario_rankings[method][s][cp] for s in scenario_ids]) for cp in cp_ids]).statistic)})
    return pd.DataFrame(result)


def group_size_summary(group_df):
    rows=[]
    for (config,method,size),part in group_df.groupby(["configuration","method","true_group_size"]):
        rows.append({"configuration":config,"method":method,"true_group_size":int(size),
            "mean_scenario_accuracy":float(part.accuracy.mean()),
            "std_scenario_accuracy":float(part.accuracy.std(ddof=1)),
            "mean_scenario_balanced_accuracy":float(part.balanced_accuracy.mean()),
            "rows_per_scenario_mean":float(part.rows.mean())})
    return pd.DataFrame(rows)


def checkpoint_group_size_stability(group_df):
    """Estimate each checkpoint's removal effect separately by true group size."""
    base = group_df[group_df.configuration == "E22-B-89"][
        ["method", "scenario_id", "true_group_size", "balanced_accuracy"]].rename(
            columns={"balanced_accuracy": "baseline_balanced_accuracy"})
    drops = group_df[group_df.configuration.str.startswith("drop_N")].copy()
    drops["checkpoint_id"] = drops.configuration.str.removeprefix("drop_")
    joined = drops.merge(base, on=["method", "scenario_id", "true_group_size"],
                         how="left", validate="many_to_one")
    joined["balanced_accuracy_loss_when_removed"] = (
        joined.baseline_balanced_accuracy - joined.balanced_accuracy)
    result=[]
    for (checkpoint_id,method,size),part in joined.groupby(
            ["checkpoint_id","method","true_group_size"],sort=False):
        vals=part.balanced_accuracy_loss_when_removed.to_numpy(float)
        result.append({"checkpoint_id":checkpoint_id,"method":method,
            "true_group_size":int(size),"scenario_count":len(vals),
            "mean_ba_loss_when_removed":float(vals.mean()),
            "std_ba_loss_when_removed":float(vals.std(ddof=1)),
            "p10_ba_loss":float(np.quantile(vals,.1)),"p90_ba_loss":float(np.quantile(vals,.9)),
            "removal_improves_scenario_pct":float(np.mean(vals<0))})
    return pd.DataFrame(result)


def run(root=ROOT, output=OUT):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "score_matrices").mkdir(exist_ok=True)
    with threadpool_limits(limits=THREADS):
        ctx = e22.load_context(root)
        v0 = pd.read_parquet(root / "data/processed/dataset_v0.parquet")
        scenarios_frame = pd.read_parquet(root / "data/processed/dataset_v0_group.parquet")
        scenarios_frame, data_audit = validate_scenarios(scenarios_frame, v0)
        fold_for_row = _fold_alignment(ctx)
        matrices = _load_and_verify_scores(ctx, output.parent / "e25", fold_for_row)
        matrices, clusters = build_all_score_matrices(ctx, output, matrices)
        scenarios = _scenario_inputs(scenarios_frame, ctx, fold_for_row)
        scenario_df, group_df, labels_df, predictions, scenario_predictions = evaluate_scenarios(
            ctx, scenarios, matrices, clusters)
        roster_df = summarize_rosters(scenario_df)
        roster_df.to_csv(output / "roster_stability.csv", index=False)
        group_summary = group_size_summary(group_df)
        group_summary.to_csv(output / "group_size_performance.csv", index=False)
        cp_group_size = checkpoint_group_size_stability(group_df)
        cp_group_size.to_csv(output / "checkpoint_group_size_stability.csv",index=False)
        cp_size_variation=(cp_group_size.groupby(["checkpoint_id","method"])
            .mean_ba_loss_when_removed.agg(["min","max"]).reset_index())
        cp_size_variation["group_size_effect_range"] = cp_size_variation["max"]-cp_size_variation["min"]
        cp_size_variation=cp_size_variation.sort_values("group_size_effect_range",ascending=False)
        labels_df.to_csv(output / "per_label_metrics.csv", index=False)
        scenario_df.to_csv(output / "scenario_metrics.csv", index=False)
        checkpoint_df = checkpoint_stability(ctx, scenario_df, predictions,
            json.loads((output.parent / "e25" / "contribution_rankings.json").read_text()))
        checkpoint_df.to_csv(output / "checkpoint_stability.csv", index=False)
        ranking_rows=[]
        for method, part in checkpoint_df.groupby("decoder"):
            ranks_orig=part.original_e25_importance_rank.to_numpy(float)
            ranks_new=part.mean_scenario_importance_rank.to_numpy(float)
            rho=float(spearmanr(ranks_orig,ranks_new).statistic)
            ranking_rows.append({"decoder":method,"spearman_e25_vs_v0_group":rho,
                "n_checkpoints":len(part),"top10_original":",".join(part.nsmallest(10,"original_e25_importance_rank").checkpoint_id),
                "top10_v0_group":",".join(part.nsmallest(10,"mean_scenario_importance_rank").checkpoint_id)})
        ranking_df=pd.DataFrame(ranking_rows)
        ranking_df.to_csv(output / "ranking_comparison.csv",index=False)
        fold_rows=[]
        for fold in ctx["folds"]:
            fid=int(fold["fold"])
            for row in np.asarray(fold["held_rows"],dtype=np.int64):
                fold_rows.append({"v0_id":str(ctx["row_ids"][row]),"fold":fid,"role":"held_out"})
        pd.DataFrame(fold_rows).to_csv(output/"fold_assignments.csv",index=False)
        cluster_rows=[]
        for cluster in clusters:
            cfg=f"drop_{cluster['cluster_id']}"
            part=roster_df[roster_df.configuration==cfg]
            for row in part.to_dict("records"):
                cluster_rows.append({"cluster_id":cluster["cluster_id"],"checkpoint_ids":";".join(cluster["checkpoint_ids"]),**row})
        pd.DataFrame(cluster_rows).to_csv(output / "cluster_removal_stability.csv",index=False)
        # Persist each scenario in its own slice. Repeated original V0 rows remain
        # separate records and group assignments never cross scenario boundaries.
        pred_dir=output / "scenario_predictions"; pred_dir.mkdir(exist_ok=True)
        for config_id, per_scenario in scenario_predictions.items():
            ids=[]; original_ids=[]; yy=[]; scenario_ids=[]; decoded={m:[] for m in
                ("independent","true_group_hungarian","frozen_E19D_subset_hungarian")}
            for sid, record in per_scenario.items():
                n=len(record["row_ids"]); ids.extend(record["row_ids"]); original_ids.extend(record["original_ids"]); yy.extend(record["y"])
                scenario_ids.extend([sid]*n)
                for method in decoded: decoded[method].extend(record[method])
            np.savez_compressed(pred_dir / f"{config_id}.npz", scenario_row_ids=np.asarray(ids),
                original_row_ids=np.asarray(original_ids),
                scenario_ids=np.asarray(scenario_ids),y=np.asarray(yy),
                **{method:np.asarray(value,dtype=np.int16) for method,value in decoded.items()})
        meta={"experiment":"E29-A","seed":2026,"cpu_threads":THREADS,
            "evaluation_only":True,"neural_inference":False,"model_training":False,
            "scenario_reuse_note":"50 randomized scenarios reuse official V0 rows and are correlated; results are sensitivity analysis, not independent validation.",
            "reconstructed_group_note":"E19-D IDs are frozen full-V0 groups restricted to the scenario, namespaced by the original held-out fold. This is not fresh reconstruction.",
            "canonical_label_order":list(LABELS),
            "v0_group_sha256":sha256(root/"data/processed/dataset_v0_group.parquet"),
            "v0_sha256":sha256(root/"data/processed/dataset_v0.parquet"),
            "e22_predictions_sha256":sha256(root/"artifacts/analysis/e22/predictions_E22-B.npz"),
            "e25_selection_rosters_sha256":sha256(root/"artifacts/analysis/e25/selection_rosters.json"),
            "e25_contribution_rankings_sha256":sha256(root/"artifacts/analysis/e25/contribution_rankings.json"),
            "e25_near_duplicate_clusters_sha256":sha256(root/"artifacts/analysis/e25/near_duplicate_clusters.json"),
            "e22_fold_model_sha256":{str(int(f["fold"])):sha256(root/"artifacts/analysis/e22"/f"E22-B_fold{int(f['fold'])}.cbm") for f in ctx["folds"]},
            "e25_budget_model_sha256":{f"{budget}-fold{int(f['fold'])}":sha256(root/"artifacts/analysis/e25"/f"E25_{budget}_fold{int(f['fold'])}.cbm") for budget in BUDGETS for f in ctx["folds"]},
            "data_validation":data_audit,"checkpoint_count":len(ctx["checkpoint_ids"]),
            "checkpoint_ids":list(ctx["checkpoint_ids"]),"budgets":list(BUDGETS),
            "cluster_removals":[{"cluster_id":c["cluster_id"],"checkpoint_ids":c["checkpoint_ids"]} for c in clusters],
            "scenario_rows":len(scenarios_frame),"scenario_count":len(scenarios)}
        summary={"metadata":meta,"roster_stability":roster_df.to_dict("records"),
            "ranking_comparison":ranking_rows,
            "checkpoint_group_size_top_variation": cp_size_variation.head(20).to_dict("records"),
            "baseline_scenario_results":roster_df[roster_df.configuration=="E22-B-89"].to_dict("records"),
            "budget_scenario_results":roster_df[roster_df.configuration.str.startswith("E25-")].to_dict("records"),
            "pruning_removal_results":roster_df[roster_df.configuration.str.startswith("drop_")].to_dict("records")}
        (output/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    summary=run(args.root,args.output)
    print(f"E29-A complete: {len(summary['metadata']['checkpoint_ids'])} checkpoints, "
          f"{summary['metadata']['scenario_count']} scenarios; output={args.output}",flush=True)


if __name__=="__main__":
    main()
