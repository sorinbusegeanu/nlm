"""Fit full-data ORACLE02/ORACLE04 deployment meta-models from cached V0 outputs.

Base checkpoints are not retrained. ORACLE04 decision models are fit using the
saved cross-fitted ORACLE02 probabilities as their training features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from . import oracle02, oracle04
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/deployment/oracle04"
ORACLE02_ROWS = ROOT / "artifacts/analysis/oracle02/row_predictions.parquet"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _predict_binary(bundle, X):
    return oracle04._binary_predict(bundle, X)


def _cv_binary_scores(idx, X, target, eligible, groups, kind):
    """Nested-style OOF scores and full-data bundle using the validated tuner."""
    n = len(target)
    scores = np.full(n, np.nan, dtype=float)
    configs = []
    folds = GroupKFold(n_splits=5)
    for fold, (tr_local, va_local) in enumerate(folds.split(np.zeros(len(idx)), groups=groups[idx]), 1):
        tr, va = idx[tr_local], idx[va_local]
        # _fit_oof deliberately receives a matrix local to train_idx, while
        # labels/groups/eligibility remain globally indexed (same convention
        # as the validated ORACLE04 outer-fold loop).
        Xtr = X[tr]
        _, selection, bundle = oracle04._fit_oof(tr, Xtr, target, eligible, groups, kind)
        config = selection.get("selected")
        configs.append(config)
        if bundle is None:
            scores[va] = 0.5
        else:
            scores[va] = _predict_binary(bundle, X[va])
    if np.isnan(scores[idx]).any():
        raise RuntimeError("Cross-fitted deployment scores have gaps")
    # Select final hyperparameter by the modal inner-fold choice; deterministic
    # serialization resolves ties by JSON lexical order.
    serialized = [json.dumps(x, sort_keys=True) for x in configs]
    counts = {k: serialized.count(k) for k in set(serialized)}
    selected_json = min(counts, key=lambda k: (-counts[k], k))
    selected = json.loads(selected_json)
    local = idx[eligible[idx]]
    final = oracle04._binary_fit(kind, X[local], target[local], selected)
    return scores, final, selected, configs


def verify_saved_oof(root: Path = ROOT) -> dict:
    """Replay the three requested saved fold-model strategies exactly."""
    rows = pd.read_parquet(root / "artifacts/analysis/oracle02/row_predictions.parquet")
    oracle04_rows = pd.read_parquet(root / "artifacts/analysis/oracle04/row_predictions.parquet")
    labels = list(LABELS)
    pcols = [f"prob_oof::all78::catboost_equal9_blend::{label}" for label in labels]
    pbase = rows[pcols].to_numpy(dtype=float)
    metrics = json.loads((root / "artifacts/analysis/oracle04/metrics.json").read_text())
    fold_decisions = {(int(d["fold"]), d["strategy"]): d for d in metrics["fold_rows"]}
    observed = {name: np.column_stack([
        oracle04_rows[f"prob::{name}::{label}"].to_numpy(dtype=float) for label in labels
    ]) for name in ("combined_logistic_core", "global_logistic_core", "global_catboost_core")}
    replayed = {name: np.zeros_like(pbase) for name in observed}
    for fold in range(1, 6):
        z = np.load(root / f"artifacts/analysis/oracle04/features_outer_{fold}.npz")
        held = z["held_indexes"].astype(int)
        X = z["held_core"].astype(np.float32)
        base = pbase[held]
        t1, t2 = base.argmax(1), np.argsort(-base, axis=1, kind="stable")[:, 1]

        # Global logistic and CatBoost strategies.
        for name, kind in (("global_logistic_core", "logistic"),
                           ("global_catboost_core", "catboost")):
            item = fold_decisions[(fold, name)]
            model_path = root / "artifacts/analysis/oracle04/models" / f"outer_{fold}" / name
            if kind == "logistic":
                bundle = joblib.load(model_path / "model.joblib")
            else:
                model = CatBoostClassifier()
                model.load_model(str(model_path / "model.cbm"))
                bundle = {"kind": "catboost", "model": model}
            score = oracle04._binary_predict(bundle, X)
            threshold = item["threshold"]
            fire = np.zeros(len(held), dtype=bool) if threshold is None else score >= float(threshold)
            replayed[name][held] = oracle04._swap_rows(base, fire)

        # Combined logistic strategy uses A then B then global; validation
        # records the three independently tuned thresholds for each held fold.
        combined_info = fold_decisions[(fold, "combined_logistic_core")]["threshold_train"]
        result = base.copy(); claimed = np.zeros(len(held), dtype=bool)
        for spec, strategy in (("group_A", "group_A_logistic_core"),
                                ("group_B", "group_B_logistic_core"),
                                ("global", "global_logistic_core")):
            path = root / "artifacts/analysis/oracle04/models" / f"outer_{fold}" / strategy / "model.joblib"
            bundle = joblib.load(path); score = oracle04._binary_predict(bundle, X)
            if spec == "group_A":
                eligible = oracle04._eligible(t1, t2, oracle04.GROUP_A)
            elif spec == "group_B":
                eligible = oracle04._eligible(t1, t2, oracle04.GROUP_B)
            else:
                eligible = np.ones(len(held), dtype=bool)
            threshold = combined_info[spec]["threshold"]
            fire = (eligible & ~claimed & (score >= float(threshold))) if threshold is not None else np.zeros(len(held), bool)
            result = oracle04._swap_rows(result, fire); claimed |= fire
        replayed["combined_logistic_core"][held] = result
    report = {}
    for name, pred in replayed.items():
        exact = bool(np.array_equal(pred, observed[name]))
        max_abs = float(np.max(np.abs(pred - observed[name])))
        report[name] = {"exact_probability_match": exact, "max_abs_probability_delta": max_abs}
        if not exact:
            raise RuntimeError(f"Saved fold predictions did not replay for {name}: max delta {max_abs}")
    return report


def fit(output_dir: Path = OUT):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_reproduction = verify_saved_oof(ROOT)
    (ids, matrices, v0, row_ids, prompts, y_text, y, fold_ids, c3_pred,
     equal_p, input_hashes) = oracle02._source_inputs(ROOT, output_dir, True)
    if len(ids) != 78:
        raise ValueError(f"Expected the ORACLE02 all78 roster, found {len(ids)}")

    # Fit the final ORACLE02 CatBoost stack on all V0 rows. Hyperparameters and
    # EQUAL9 blend weight are selected from grouped OOF predictions only.
    Xbase_path = ROOT / "artifacts/analysis/oracle02/features.npz"
    Xbase = np.load(Xbase_path)["X_all78"]
    if Xbase.shape != (len(y), 3078):
        raise ValueError(f"Unexpected ORACLE02 all78 feature matrix: {Xbase.shape}")
    cfg, alpha, baseline_tuning = oracle02._tune_catboost(
        Xbase, y, np.asarray(prompts), np.arange(len(y)), equal_p
    )
    base_model = oracle02._fit_catboost(Xbase, y, cfg)
    base_model.save_model(str(output_dir / "oracle02_catboost.cbm"))

    source = pd.read_parquet(ORACLE02_ROWS)
    pcols = [f"prob_oof::all78::catboost_equal9_blend::{label}" for label in LABELS]
    p_oof = source[pcols].to_numpy(dtype=np.float64)
    if source.id.astype(str).tolist() != row_ids or not np.isfinite(p_oof).all():
        raise ValueError("ORACLE02 saved OOF probabilities do not align to V0")
    y_from_rows = np.asarray([LABELS.index(x) for x in source.y_true.astype(str)])
    if not np.array_equal(y, y_from_rows):
        raise ValueError("ORACLE02 OOF labels differ from the full V0 dataset")

    # The ORACLE04 training features use OOF ORACLE02 probabilities, never the
    # full-fit ORACLE02 model's in-sample probabilities.
    trust = pd.read_csv(ROOT / "docs/trust-list-2-v025.csv", dtype=str, keep_default_na=False)
    model_map = trust.drop_duplicates("sequence_id").set_index("sequence_id").model.to_dict()
    families = {sid: oracle04._family(model_map.get(sid, sid)) for sid in ids}
    X, feature_names = oracle04._base_features(p_oof, matrices, ids, equal_p, families)
    top1, top2, target, _ = oracle04._swap_targets(y, p_oof)
    n = len(y); all_idx = np.arange(n); groups = np.asarray(prompts)
    eligible = {
        "global": np.ones(n, dtype=bool),
        "group_A": oracle04._eligible(top1, top2, oracle04.GROUP_A),
        "group_B": oracle04._eligible(top1, top2, oracle04.GROUP_B),
    }

    strategies = {
        "global_logistic_core": ("global", "logistic"),
        "global_catboost_core": ("global", "catboost"),
        "group_A_logistic_core": ("group_A", "logistic"),
        "group_B_logistic_core": ("group_B", "logistic"),
    }
    bundles = {}
    scores = {}
    selected = {}
    cv_choices = {}
    for name, (spec, kind) in strategies.items():
        scores[name], bundles[name], selected[name], cv_choices[name] = _cv_binary_scores(
            all_idx, X, target, eligible[spec], groups, kind
        )
        if kind == "catboost":
            bundles[name]["model"].save_model(str(output_dir / f"{name}.cbm"))
            bundles[name] = {k: v for k, v in bundles[name].items() if k != "model"}
            bundles[name]["model_path"] = f"{name}.cbm"
        else:
            joblib.dump(bundles[name], output_dir / f"{name}.joblib", compress=3)
            bundles[name] = {"kind": "logistic", "model_path": f"{name}.joblib"}

    # Cross-fitted probability scores select all decision thresholds. The
    # combined policy applies A, then B, then global, each against the residual
    # unclaimed rows and updated probability vector.
    thresholds = {}
    for name, (spec, _) in strategies.items():
        th, info = oracle04._threshold_scores(all_idx, p_oof, y, eligible[spec], scores[name])
        thresholds[name] = {"threshold": th, "selection": info}

    working = p_oof.copy(); claimed = np.zeros(n, dtype=bool); combined = {}
    for spec, name in (("group_A", "group_A_logistic_core"),
                       ("group_B", "group_B_logistic_core"),
                       ("global", "global_logistic_core")):
        residual = eligible[spec] & ~claimed
        th, info = oracle04._threshold_scores(all_idx, working, y, residual, scores[name])
        fire = residual & (scores[name] >= th) if th is not None else np.zeros(n, dtype=bool)
        working = oracle04._swap_rows(working, fire)
        claimed |= fire
        combined[spec] = {"threshold": th, "selection": info,
                          "crossfit_rows_fired": int(fire.sum())}
    thresholds["combined_logistic_core"] = combined

    manifest = {
        "status": "fitted_from_cached_oof_features",
        "strategies": ["combined_logistic_core", "global_logistic_core", "global_catboost_core"],
        "trust_list": "trust-list-2-v025",
        "checkpoint_ids": ids,
        "checkpoint_families": families,
        "class_order": LABELS,
        "v0_rows": len(y),
        "v0_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
        "input_hashes": input_hashes,
        "feature_variant": "core",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "baseline": {"kind": "CatBoost + EQUAL9 blend", "selected_catboost_config": cfg,
                     "catboost_weight": alpha, "equal9_weight": 1.0-alpha,
                     "inner_group_selection": baseline_tuning,
                     "model_path": "oracle02_catboost.cbm"},
        "decision_models": bundles,
        "decision_hyperparameters": selected,
        "decision_cv_choices": cv_choices,
        "thresholds": thresholds,
        "saved_oof_reproduction": fold_reproduction,
        "score_selection": "5-fold prompt-group cross-fitted meta-classifier scores; thresholds maximize training V0 balanced accuracy",
        "deployment_script_sha256": _sha(Path(__file__).resolve()),
        "known_limitation": "Base checkpoints and roster were selected using V0; these final deployment meta-models and thresholds are V0-exposed and are not an independent Kaggle generalization estimate.",
    }
    (output_dir / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n")
    return manifest


def predict_strategies(output_dir: Path, checkpoint_ids: list[str],
                       matrices: dict[str, np.ndarray], equal9_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the fitted ORACLE04 deployment models to already-computed test probabilities.

    ``matrices`` must contain normalized 18-class probabilities for each saved
    checkpoint ID, with rows already aligned to the same test ID order.
    """
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "deployment_manifest.json").read_text())
    expected_ids = list(manifest["checkpoint_ids"])
    if list(checkpoint_ids) != expected_ids or set(matrices) != set(expected_ids):
        raise ValueError("Checkpoint ID order/set differs from the fitted ORACLE02 stack")
    equal9 = np.asarray(equal9_probabilities, dtype=np.float64)
    if equal9.ndim != 2 or equal9.shape[1] != len(LABELS):
        raise ValueError("EQUAL9 probabilities must be [rows, 18]")
    n = len(equal9)
    for sid in expected_ids:
        p = np.asarray(matrices[sid], dtype=np.float64)
        if p.shape != (n, len(LABELS)) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError(f"Invalid test probability matrix for {sid}")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Unnormalized test probabilities for {sid}")
    if not np.allclose(equal9.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("EQUAL9 probabilities are not normalized")

    X02, names02 = oracle02._build_features(expected_ids, matrices, equal9)
    if X02.shape[1] != 3078:
        raise ValueError(f"Unexpected ORACLE02 test feature count: {X02.shape[1]}")
    from catboost import CatBoostClassifier
    baseline = CatBoostClassifier()
    baseline.load_model(str(output_dir / manifest["baseline"]["model_path"]))
    raw = np.asarray(baseline.predict_proba(X02), dtype=np.float64)
    baseline_probs = np.zeros((n, len(LABELS)), dtype=np.float64)
    for j, cls in enumerate(baseline.classes_):
        baseline_probs[:, int(cls)] = raw[:, j]
    baseline_probs = manifest["baseline"]["catboost_weight"] * baseline_probs + manifest["baseline"]["equal9_weight"] * equal9
    baseline_probs /= np.maximum(baseline_probs.sum(axis=1, keepdims=True), 1e-15)
    X04, _ = oracle04._base_features(baseline_probs, matrices, expected_ids, equal9,
                                      manifest["checkpoint_families"])
    if X04.shape[1] != manifest["feature_count"]:
        raise ValueError("ORACLE04 feature count differs from fitted models")

    def score(name: str) -> np.ndarray:
        spec = manifest["decision_models"][name]
        path = output_dir / spec["model_path"]
        if spec["kind"] == "logistic":
            model = joblib.load(path)
            return oracle04._binary_predict(model, X04)
        model = CatBoostClassifier(); model.load_model(str(path))
        return oracle04._binary_predict({"kind": "catboost", "model": model}, X04)

    out = {}
    for name in ("global_logistic_core", "global_catboost_core"):
        threshold = manifest["thresholds"][name]["threshold"]
        fire = np.zeros(n, dtype=bool) if threshold is None else score(name) >= float(threshold)
        out[name] = oracle04._swap_rows(baseline_probs, fire)

    combined = baseline_probs.copy(); claimed = np.zeros(n, dtype=bool)
    top1 = baseline_probs.argmax(1)
    top2 = np.argsort(-baseline_probs, axis=1, kind="stable")[:, 1]
    rules = manifest["thresholds"]["combined_logistic_core"]
    for spec, model_name, group in (("group_A", "group_A_logistic_core", oracle04.GROUP_A),
                                    ("group_B", "group_B_logistic_core", oracle04.GROUP_B),
                                    ("global", "global_logistic_core", None)):
        eligible = np.ones(n, dtype=bool) if group is None else oracle04._eligible(top1, top2, group)
        threshold = rules[spec]["threshold"]
        fire = (eligible & ~claimed & (score(model_name) >= float(threshold))) if threshold is not None else np.zeros(n, bool)
        combined = oracle04._swap_rows(combined, fire); claimed |= fire
    out["combined_logistic_core"] = combined
    for name, p in out.items():
        if p.shape != (n, len(LABELS)) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError(f"Invalid deployed probability output for {name}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = fit(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "strategies": result["strategies"],
                      "checkpoint_count": len(result["checkpoint_ids"]),
                      "baseline_config": result["baseline"]["selected_catboost_config"],
                      "catboost_weight": result["baseline"]["catboost_weight"],
                      "thresholds": result["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
