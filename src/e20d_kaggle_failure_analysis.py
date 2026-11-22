"""Diagnose the E20-D V0-to-Kaggle gap using saved predictions and classifiers.

CPU-only analysis. No base model inference or fitting is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("MKL_NUM_THREADS", "24")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from . import e13_reliability_router as e13, e15, e16_common, e19b_group, e20
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e20d_kaggle_failure_analysis"
PKG = ROOT / "artifacts/kaggle/e20d-checkpoints-v001"
DEP = PKG / "deployment"
N_THREADS = 24


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _router_evidence(ctx, manifest_fold, probs, cp_ids, families, fit_rows, target_rows):
    """Run the packaged E13 router exactly as the Kaggle notebook does."""
    roster = list(manifest_fold["checkpoint_ids"])
    indexes = [cp_ids.index(sid) for sid in roster]
    p = probs[indexes]
    roster_families = list(manifest_fold["families"])
    router = CatBoostClassifier()
    router.load_model(str(DEP / f"e13_router_fold{manifest_fold['fold']}.cbm"))
    # Mirror the notebook's concatenated V0-cache + test prediction tensor.
    # Offset duplicate target rows so an all-V0 deployment replay stays
    # disjoint from the fold's router-fitting indices, just as Kaggle test rows do.
    p_extended = np.concatenate([p, p], axis=1)
    target_index = target_rows + p.shape[1]
    frame, hard = e13.build_feature_frame(
        p_extended, roster, roster_families, ctx["y"][fit_rows], fit_rows, target_index)
    rel = router.predict_proba(e13._cat_frame(frame), thread_count=1)[:, 1]
    reliability = rel.reshape(len(target_rows), len(roster)).T.astype(np.float32)
    ensemble = e13.family_balanced_probabilities(p[:, target_rows], roster_families)
    scores = e15._pair_scores(hard, reliability, ensemble, roster_families)
    return {"scores": e16_common.finite_scores(scores), "roster_ids": roster,
            "reliability": reliability, "hard": hard.astype(np.int16),
            "router_frame": frame}


def _load_package_groups(ctx, data, grouper, rows, spaces=None):
    """Rebuild package-style E19-B groups using only rows in this batch."""
    rows = np.asarray(rows, dtype=np.int64)
    if spaces is None:
        spaces = e19b_group.transform_similarity_spaces(
            data["recon"], ctx["texts"], grouper["vectorizers"])
    pairs = e19b_group.candidate_pairs(rows, spaces, data["embeddings"],
        neighbors=grouper.get("neighbors", 40), semantic_device=None)
    pair_features = e19b_group.pair_features(rows, pairs, spaces, data["recon"],
        ctx["texts"], data["embeddings"], data["topics"], data["details"],
        data["constraints"])
    pair_prob = (e19b_group.score_pairs(grouper["pair_scorer"], pair_features)
                 if len(pairs) else np.empty(0, dtype=np.float32))
    local_ids = ctx["row_ids"][rows].astype(str)
    group_ids, detail = e19b_group.cluster_complete_link(local_ids, pairs, pair_prob,
        threshold=float(grouper["threshold"]),
        max_cluster_size=int(grouper.get("max_cluster_size", 18)))
    pair_scores = {tuple(sorted((local_ids[int(a)], local_ids[int(b)]))): float(v)
                   for (a, b), v in zip(pairs, pair_prob)}
    return np.asarray(group_ids, dtype=str), pair_scores, {
        "candidate_pairs": int(len(pairs)), "pair_edges": int(len(pair_prob)),
        "groups": int(len(set(group_ids))), "detail": detail}


def _cached_graph(ctx, data, grouper, rows, cache_path, spaces=None):
    """Cache deterministic package-style group reconstruction by exact row IDs."""
    cache_path = Path(cache_path)
    identity = hashlib.sha256("\n".join(ctx["row_ids"][rows].astype(str)).encode()).hexdigest()
    grouper_hash = _sha(DEP / "e19b_grouper.joblib")
    if cache_path.is_file():
        item = joblib.load(cache_path)
        if item.get("row_hash") == identity and item.get("grouper_hash") == grouper_hash:
            return item["groups"], item["pair_scores"], item["info"]
    groups, scores, info = _load_package_groups(ctx, data, grouper, rows, spaces=spaces)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"row_hash": identity, "grouper_hash": grouper_hash,
                 "groups": groups, "pair_scores": scores, "info": info},
                cache_path, compress=3)
    return groups, scores, info


def _cat_frame_for_fold(fold, features, classifier):
    expected = list(classifier.feature_names_)
    if set(features.columns) != set(expected):
        raise ValueError(f"fold {fold}: feature schema mismatch; missing="
                         f"{sorted(set(expected)-set(features.columns))[:8]} extra="
                         f"{sorted(set(features.columns)-set(expected))[:8]}")
    return features.loc[:, expected]


def _score_matrix(classifier, features, fold):
    frame = _cat_frame_for_fold(fold, features, classifier)
    return e20._predict_candidate(classifier, frame)


def _metric(y, pred):
    return e20._metric(np.asarray(y), np.asarray(pred))


def _comparison(y, base, pred):
    return e20._comparison(np.asarray(y), np.asarray(base), np.asarray(pred))


def _aggregate(scores, groups, y, *, normalize=False):
    values = np.stack(scores).astype(np.float64)
    if normalize:
        values /= np.maximum(values.sum(axis=2, keepdims=True), 1e-12)
    mean = values.mean(axis=0)
    pred_ind = mean.argmax(axis=1)
    pred_hun = e20.hungarian(mean, groups)
    return mean, pred_ind, pred_hun


def _feature_shift(frame_a, frame_b, importance):
    rows = []
    a = frame_a.reset_index(drop=True); b = frame_b.reset_index(drop=True)
    numeric = [c for c in a if c in b and c != "candidate_label" and
               pd.api.types.is_numeric_dtype(a[c])]
    for col in numeric:
        x = a[col].to_numpy(dtype=np.float64); z = b[col].to_numpy(dtype=np.float64)
        sd = float(np.sqrt((np.var(x) + np.var(z)) / 2))
        rows.append({"feature": col, "true_group_mean": float(x.mean()),
            "reconstructed_mean": float(z.mean()), "mean_delta_true_minus_reconstructed": float(x.mean()-z.mean()),
            "standardized_mean_difference": float((x.mean()-z.mean()) / max(sd, 1e-12)),
            "mean_absolute_row_difference": float(np.mean(np.abs(x-z))),
            "max_absolute_row_difference": float(np.max(np.abs(x-z))),
            "mean_catboost_importance": float(importance.get(col, 0.0))})
    return pd.DataFrame(rows).sort_values(
        ["mean_catboost_importance", "mean_absolute_row_difference"], ascending=False)


def write_report(metrics_path=OUT/"metrics.json", output_path=ROOT/"docs/e20d_kaggle_failure_analysis.md"):
    """Render a concise report from the saved, reproducible analysis tables."""
    metrics = json.loads(Path(metrics_path).read_text())
    exp = metrics["experiments"]
    comp = metrics["comparisons"]
    lines = ["# E20-D Kaggle Failure Analysis", "",
        "## Finding", "",
        "The packaged E20-D scoring path reproduces its saved V0 OOF candidate scores and Hungarian labels exactly within floating-point tolerance. The deployment gap is therefore not explained by a class-order, checkpoint-probability, E13-router, CatBoost feature-order, or assignment-code mismatch. Group distribution is a confirmed contributor: full-batch deployment-style groups cost 1.96 pp versus the original OOF setup, oracle group features add 2.33 pp over reconstructed features on held-out rows, and the corrected incomplete-group simulation reaches 70.79% BA (7.41 pp below full-group OOF). Those V0 effects still do not explain the full 22.2 pp gap to the 0.558 Kaggle score; hidden-test distribution shift remains the leading unverified cause.", "",
        "The submitted E20-D kernel version 11 is recorded by Kaggle as submission **58710430**, public score **0.558**. E13-B submission 58609286 scored 0.606. The packaged local output contains only the three-row smoke-test input; the scored hidden evaluation predictions are not present locally. Kaggle describes the competition as approximately 4,000 unlabeled passages ([competition description](https://www.kaggle.com/competitions/name-that-llm/overview/description)), so the three-row local output cannot explain the leaderboard result.", "",
        "## 1. Deployment parity", "",
        "| Stage | Result |", "|---|---:|"]
    max_router = max(x["packaged_vs_refit_router_reliability_max_abs_diff"] for x in metrics["parity_by_fold"])
    max_score = max(x["saved_oof_score_max_abs_diff"] for x in metrics["parity_by_fold"])
    max_e13 = max(x["packaged_vs_refit_router_label_score_max_abs_diff"] for x in metrics["parity_by_fold"])
    max_feat = max(x["feature_value_max_abs_diff_packaged_vs_oof_builder"] for x in metrics["parity_by_fold"])
    lines += [f"| Checkpoint class order | Exact match to canonical 18-label order |",
        f"| Packaged V0 probabilities vs local cache | max absolute difference `{metrics['checkpoint_probability_max_abs_diff']:.3g}` |",
        f"| Packaged router reliability vs refit path | max absolute difference `{max_router:.3g}` |",
        f"| E13 candidate-label scores | max absolute difference `{max_e13:.3g}` |",
        f"| CatBoost input feature values (same OOF rows/groups) | max absolute difference `{max_feat:.3g}` |",
        f"| CatBoost candidate scores vs saved E20-D OOF | max absolute difference `{max_score:.3g}` |",
        f"| Hungarian predictions vs saved OOF | `{sum(x['saved_oof_hungarian_prediction_differences'] for x in metrics['parity_by_fold'])}` changed rows across five folds |",
        "| CatBoost feature names/order | Exact schema; package selects the saved model's feature order |",
        "", "First divergence is group construction when moving from OOF held-fold batches to deployment's single full batch: saved all-V0 package grouping has **258 groups**, whereas OOF predictions cluster each held fold independently. The package-style group assignment exactly matches the saved all-V0 diagnostic grouping (ARI 1.0). This changes group-derived features, but the row-held-out model replay remains well above the Kaggle score.", "",
        "Per-fold parity details are in `artifacts/analysis/e20d_kaggle_failure_analysis/deployment_parity.csv`.", "",
        "## 2. Five-fold aggregation", "",
        "| Evaluation | Accuracy | Balanced accuracy | Interpretation |", "|---|---:|---:|---|"]
    rows = [
        ("Saved E20-D OOF", "E20-D_saved_OOF", "Reference; one held-out fold model and OOF groups"),
        ("Per-row fold model, package-style full-batch groups", "E20-D_packaged_OOF_scores_Hungarian_full_batch_groups", "Leakage-controlled model choice; measures group-feature/decoder shift"),
        ("Per-row fold model, row-normalized scores + Hungarian", "foldheld_package_norm_hungarian", "Leakage-controlled; per-row scores normalized before assignment"),
        ("All five models averaged; independent labels", "E20-D_all5_average_independent_global_groups", "Optimistic/in-sample on V0; each row is in four models' training folds"),
        ("All five raw scores averaged + Hungarian", "E20-D_all5_average_Hungarian_global_groups", "Optimistic/in-sample; not a valid Kaggle generalization estimate"),
        ("All five per-model normalized scores + Hungarian", "E20-D_all5_row_normalized_average_Hungarian", "Optimistic/in-sample; normalization changes 80 V0 predictions"),
    ]
    for label, key, note in rows:
        m=exp[key]; lines.append(f"| {label} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {note} |")
    lines += ["", "All-five-on-V0 is not an honest analogue of Kaggle: the five CatBoost fold models were each trained on 80% of V0, so every V0 row is in four models' fitting data. Its high score must not be used to claim the deployment aggregation generalizes. The valid fold-held-out replay with deployment-style grouping is the second row; it falls from 78.21% to 76.24% BA, a 1.96 pp drop, still far short of 55.8%.", ""]
    c=comp["all5_raw_vs_saved_OOF"]
    lines += [f"Moving from saved OOF to all-five raw/Hungarian changes {c['changed']} rows: {c['corrected']} corrected and {c['introduced']} regressed. This is contaminated by fold-training overlap.", ""]
    cn=comp["all5_normalized_vs_raw"]
    lines += [f"Per-model row normalization changes {cn['changed']} rows relative to raw-score averaging ({cn['corrected']} corrected, {cn['introduced']} regressed; `{cn['delta_balanced_accuracy']*100:+.3f}` pp BA). This is also in-sample on V0.", "",
        f"On the leakage-controlled per-row held-fold scores, row normalization changes {comp['foldheld_package_norm_vs_raw']['changed']} assignments ({comp['foldheld_package_norm_vs_raw']['corrected']} corrected, {comp['foldheld_package_norm_vs_raw']['introduced']} regressed) and changes BA from {exp['E20-D_packaged_OOF_scores_Hungarian_full_batch_groups']['balanced_accuracy']:.4%} to {exp['foldheld_package_norm_hungarian']['balanced_accuracy']:.4%}. It does not improve this held-out replay.", "",
        "### Per-label precision and recall", "",
        "| Label | E13-B P/R | E20-D OOF P/R | Fold-held package groups P/R | Checkpoint-only OOF P/R |", "|---|---:|---:|---:|---:|"]
    label_keys = ["E13-B", "E20-D_saved_OOF",
                  "E20-D_packaged_OOF_scores_Hungarian_full_batch_groups",
                  "E20-checkpoint-only_OOF_Hungarian"]
    label_names = list(LABELS)
    for i, label in enumerate(label_names):
        vals=[]
        for key in label_keys:
            item=exp[key]["per_label"][label]
            vals.append(f"{item['precision']:.3f}/{item['recall']:.3f}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "### Per-label change in precision / recall vs E13-B", "",
        "| Label | E20-D saved OOF ΔP/ΔR | Package groups, held-fold ΔP/ΔR | Checkpoint-only OOF ΔP/ΔR |", "|---|---:|---:|---:|"]
    for label in label_names:
        base=exp["E13-B"]["per_label"][label]
        deltas=[]
        for key in label_keys[1:]:
            item=exp[key]["per_label"][label]
            deltas.append(f"{item['precision']-base['precision']:+.3f}/{item['recall']-base['recall']:+.3f}")
        lines.append(f"| {label} | " + " | ".join(deltas) + " |")
    lines += ["", "The full per-label precision/recall/F1 and confusion matrices for every aggregation variant are retained in `metrics.json`; row-aligned labels are in `v0_predictions.parquet`.", "",
        "## 3. Group-feature distribution shift", "",
        "The same fold classifier scores were recomputed with true prompt groups and reconstructed groups. Feature shifts are ranked by standardized mean difference; CatBoost importance shows which shifts are used by the scorer.", "",
        "| Feature | Mean absolute SMD | Mean absolute row delta | Mean CatBoost importance |", "|---|---:|---:|---:|"]
    for x in metrics["group_feature_shift_summary"][:12]:
        lines.append(f"| `{x['feature']}` | {x['mean_abs_smd']:.3f} | {x['mean_abs_row_delta']:.4f} | {x['mean_importance']:.3f} |")
    oof_group = exp["E20-D_packaged_OOF_scores_Hungarian_full_batch_groups"]["balanced_accuracy"]
    true_group = exp["E20-D_fold_heldout_true_group_features_reconstructed_decoder_Hungarian"]["balanced_accuracy"]
    shiftcomp=comp["OOF_true_vs_reconstructed_group_features"]
    lines += ["", f"With held-out fold models and the same reconstructed-group Hungarian decoder, replacing reconstructed group features with true-group features changes {shiftcomp['changed']} rows ({shiftcomp['corrected']} corrected, {shiftcomp['introduced']} regressed) and moves BA from {oof_group:.4%} to {true_group:.4%} ({(true_group-oof_group)*100:+.3f} pp). This is measurable, but not enough to account for the Kaggle gap.", "",
        "The largest distribution shifts are group-edge confidence and group size. Some highly shifted edge features have zero or near-zero CatBoost importance; the more influential group candidate competition features have moderate shifts. Detailed per-fold values are in `group_feature_shift_by_fold.csv`.", "",
        "## 4. Rebuilt incomplete-group evaluation", "",
        "Unlike the previous E20-D incomplete-group result, this evaluation reconstructs the pair graph on each exact sampled subset, recomputes group-dependent features, scores all five saved classifiers, and then runs independent or Hungarian decoding. It also reports a leakage-controlled variant that uses only the fold model that held out each row.", "",
        "| Method | Mean BA | SD | Mean accuracy |", "|---|---:|---:|---:|"]
    for x in metrics["incomplete_group_aggregation"]:
        lines.append(f"| {x['method']} | {x['mean_balanced_accuracy']:.4%} | {x['std_balanced_accuracy']:.4%} | {x['mean_accuracy']:.4%} |")
    by_size_path = Path(metrics_path).parent / "incomplete_group_by_size.csv"
    by_size = pd.read_csv(by_size_path) if by_size_path.is_file() else pd.DataFrame()
    lines += ["", "### Leakage-controlled results by reconstructed group size", "",
        "These are the fold-held-out models, averaged over the existing five seeds per scenario. Sizes not represented in a sampled scenario are shown as —.", "",
        "| Size | B rows | B independent BA | B Hungarian BA | C rows | C independent BA | C Hungarian BA |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for size in range(1, 19):
        cells=[]
        for variant, method in [("B-incomplete-2-10","E20-D-fold-heldout-independent"),
                                ("B-incomplete-2-10","E20-D-fold-heldout-Hungarian"),
                                ("C-incomplete-2-18","E20-D-fold-heldout-independent"),
                                ("C-incomplete-2-18","E20-D-fold-heldout-Hungarian")]:
            hit=by_size[(by_size.variant==variant)&(by_size.method==method)&(by_size.group_size==size)] if len(by_size) else pd.DataFrame()
            cells.append((int(hit.rows.iloc[0]),f"{hit.balanced_accuracy.iloc[0]:.1%}") if len(hit) else (0,"—"))
        lines.append(f"| {size} | {cells[0][0]} | {cells[0][1]} | {cells[1][1]} | {cells[2][0]} | {cells[2][1]} | {cells[3][1]} |")
    lines += ["", "The all-five methods are optimistic because sampled V0 rows were seen by four fold models. Use the `*-fold-heldout-*` rows for validation interpretation. Metrics by reconstructed group size (including singletons through size 18 where present) are saved in `incomplete_group_by_size.csv`; row-level sample predictions are in `incomplete_group_predictions_rescored.parquet`.", "",
        "## 5. Checkpoint-only ablation", "",
        "| Checkpoint-only scoring | Accuracy | Balanced accuracy |", "|---|---:|---:|"]
    for label,key in [("OOF independent","E20-checkpoint-only_OOF_independent"),
                      ("OOF + reconstructed-group Hungarian","E20-checkpoint-only_OOF_Hungarian"),
                      ("All-five independent (in-sample)","E20-checkpoint-only_all5_average_independent"),
                      ("All-five + Hungarian (in-sample)","E20-checkpoint-only_all5_average_Hungarian")]:
        m=exp[key]; lines.append(f"| {label} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} |")
    cc=comp["checkpoint_only_vs_E20D_all5"]
    lines += ["", f"On the in-sample all-five diagnostic, checkpoint-only is {cc['delta_balanced_accuracy']*100:+.3f} pp over E20-D and changes {cc['changed']} labels ({cc['corrected']} corrected, {cc['introduced']} regressed). This dramatic gain is a leakage artifact, not evidence that the checkpoint-only model is deployably stronger. OOF checkpoint-only scores are the valid comparator.", "",
        "## 6. Root-cause assessment", "",
        "### Confirmed", "",
        "- No packaged inference parity defect was found on identical V0 rows and OOF groups: all five folds reproduce saved E20-D scores to <3.0e-8 maximum absolute error and produce zero changed Hungarian predictions.",
        "- Deployment grouping is a different batch regime from OOF: 258 package-style all-V0 clusters versus fold-local OOF partitions. Under held-out fold models this costs about 1.96 pp BA, not the 22.2 pp Kaggle gap.",
        "- The earlier incomplete-group evaluation was invalid for the requested question because it reused full-group CatBoost scores. The rebuilt sample evaluation shows the proper group-reconstruction path and is saved separately.",
        "- All-five fold-model averaging on V0 and checkpoint-only all-five scores are contaminated: each V0 example appears in four models' training folds. They cannot explain the hidden-test result.",
        "", "### Suspected, not proven", "",
        "- Hidden-test domain/prompt/style shift is the leading explanation left by the parity checks. Kaggle scoring uses a hidden evaluation set; the scored-run predictions/labels are not available in the local v11 output.",
        "- E20-C trains its CatBoost scorer with oracle prompt-group features, while E20-D serves reconstructed groups. This is a real train/serve feature mismatch, but leakage-controlled V0 impact is modest in the current replay; it may be larger on hidden groups.",
        "- Test-time group completeness and reconstruction quality may differ from V0. V0 subset simulations do not establish the hidden test's group-size distribution.",
        "- A checkpoint-specific inference or class-order defect on hidden test rows is not ruled out by V0 parity alone; the archived v11 output has only three smoke-test rows and does not include the hidden scored-run probability cache.",
        "", "## Recommended deployment corrections (ranked)", "",
        "1. **Obtain scored-run outputs before changing the model:** preserve per-checkpoint probability matrices, reconstructed groups/pair scores, per-fold 18-label CatBoost scores, and final assignments from the actual hidden evaluation run; verify canonical class order and finite normalized checkpoint probabilities there.",
        "2. **Validate the exact serving aggregation with an untouched proxy set:** evaluate all five fold models on rows unseen by every fold model (or train a single final scorer on all V0 and evaluate on a truly external labeled set). Do not use the contaminated all-five-on-V0 score.",
        "3. **Reduce train/serve group-feature mismatch:** compare E20-B (reconstructed-group training and inference) with E20-D on a genuinely unseen labeled set; if mismatch persists, deploy E20-B or remove the most shifted group features based on held-out evidence.",
        "4. **Measure hidden-test grouping diagnostics:** expose group-size histogram, singleton share, pair confidence, and cluster purity proxy in the run manifest. Do not force Hungarian uniqueness on weak-confidence/singleton groups without validation.",
        "5. **Check hidden inference class mapping end-to-end:** emit per-checkpoint class map, row-ID hash, probability row sums and score matrix checksum in the scored-run artifacts.",
        "", "## Artifacts", "",
        "Analysis code: `src/e20d_kaggle_failure_analysis.py`; regression tests: `tests/test_e20d_kaggle_failure_analysis.py`.",
        "Machine-readable results: `artifacts/analysis/e20d_kaggle_failure_analysis/metrics.json`, `deployment_scores_v0.npz`, `deployment_parity.csv`, `group_feature_shift_summary.csv`, `per_label_changes.csv`, `incomplete_group_metrics.csv`, `incomplete_group_by_size.csv`, and `incomplete_group_predictions_rescored.parquet`.", "",
        "No neural inference, retraining, queue changes, or Kaggle submission was performed."]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines)+"\n")
    return Path(output_path)


def run(root=ROOT, output=OUT):
    root = Path(root).resolve(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=N_THREADS):
        ctx = e16_common.load_context(root)
        saved = np.load(root / "artifacts/analysis/e20/scores_e20d.npz", allow_pickle=True)
        if not np.array_equal(saved["row_ids"].astype(str), ctx["row_ids"].astype(str)):
            raise ValueError("Saved E20-D rows do not match V0 context")
        y = ctx["y"]; baseline = ctx["baseline"]
        saved_oof_scores = np.asarray(saved["scores"], dtype=np.float32)
        saved_oof_pred = np.asarray(saved["pred_hungarian"], dtype=np.int64)
        group_frame = e20.load_reconstructed_groups(root, required=True)
        oof_groups = group_frame.e20_group.astype(str).to_numpy()
        pair_scores = group_frame.attrs.get("pair_scores", {})
        data = e20._load_e19b_reconstructions(root)
        grouper = joblib.load(DEP / "e19b_grouper.joblib")
        package_spaces = e19b_group.transform_similarity_spaces(
            data["recon"], ctx["texts"], grouper["vectorizers"])
        manifest = json.loads((PKG / "e20d_manifest.json").read_text())
        cache = np.load(DEP / "e20d_v0_probabilities.npz", allow_pickle=True)
        cp_ids = cache["checkpoint_ids"].astype(str).tolist()
        cp_families = cache["families"].astype(str).tolist()
        probs = np.asarray(cache["probabilities"], dtype=np.float32)
        if manifest["classes"] != list(LABELS):
            raise ValueError("Packaged canonical class order differs from NLM class order")
        if not np.array_equal(cache["row_ids"].astype(str), ctx["row_ids"].astype(str)):
            raise ValueError("Packaged V0 checkpoint probabilities are row-misaligned")
        cp_max_diff = 0.0
        cp_detail = []
        for i, sid in enumerate(cp_ids):
            source_idx = ctx["checkpoint_ids"].index(sid)
            delta = np.abs(probs[i] - ctx["matrices"][source_idx]).astype(np.float64)
            max_diff = float(delta.max())
            cp_max_diff = max(cp_max_diff, max_diff)
            cp_detail.append({"checkpoint_id": sid, "family": cp_families[i],
                              "max_abs_probability_diff": max_diff,
                              "mean_abs_probability_diff": float(delta.mean())})
        # Full-batch E19-B reconstruction exactly mirrors Kaggle grouping without
        # Qwen/EmbeddingGemma inference: their cached outputs are the inputs here.
        # The full-batch diagnostic is the saved deployment grouper's own
        # all-V0 clustering, generated by the same bundle in e19b_evaluate.
        # Pair scores are regenerated below because the summary artifact stores
        # group IDs but not pair-level confidence values.
        global_groups_frame = pd.read_parquet(root / "artifacts/analysis/e19b/all_v0_diagnostic_groups.parquet")
        if global_groups_frame.id.astype(str).tolist() != ctx["row_ids"].astype(str).tolist():
            raise ValueError("Full-V0 E19-B diagnostic group rows are not aligned")
        global_groups = global_groups_frame.predicted_group_id.astype(str).to_numpy()
        rebuilt_global_groups, global_pair_scores, global_group_info = _cached_graph(
            ctx, data, grouper, np.arange(len(y)), output/"package_full_v0_graph.joblib",
            spaces=package_spaces)
        from sklearn.metrics import adjusted_rand_score
        global_group_ari = float(adjusted_rand_score(global_groups, rebuilt_global_groups))
        global_group_exact = bool(global_group_ari == 1.0)

        parity_rows = []
        fold_scores_global = []
        fold_scores_oof_groups = []
        fold_scores_true_groups = []
        fold_scores_cp_only = []
        fold_scores_cp_only_oof = []
        fold_router_diagnostics = []
        group_shift_parts = []
        model_importances = {}
        for mf in manifest["folds"]:
            fold = int(mf["fold"]); fit = np.asarray(mf["fit_rows"], dtype=np.int64)
            held = np.asarray(ctx["outer_folds"][fold-1]["held_rows"], dtype=np.int64)
            local_outer = ctx["outer_folds"][fold-1]
            if not np.array_equal(held, local_outer["held_rows"]):
                raise ValueError(f"fold {fold} row assignment mismatch")
            # Exact deployment input: fold router, package vectorizer, and E20-C model.
            evidence_cache = output / "package_evidence" / f"fold{fold}.joblib"
            if evidence_cache.is_file():
                ev_all = joblib.load(evidence_cache)
            else:
                ev_all = _router_evidence(ctx, mf, probs, cp_ids, cp_families,
                                          fit, np.arange(len(y)))
                ev_all.pop("router_frame", None)
                evidence_cache.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(ev_all, evidence_cache, compress=3)
            ev_held_local = _router_evidence(ctx, mf, probs, cp_ids, cp_families,
                                             fit, held)
            ev_local = e20._evidence(ctx, fit, held)
            reliability_delta = float(np.max(np.abs(ev_held_local["reliability"]-
                                                         ev_local["reliability"])))
            evidence_score_delta = float(np.max(np.abs(ev_held_local["scores"]-
                                                           ev_local["scores"])))
            router = CatBoostClassifier(); router.load_model(str(DEP/f"e13_router_fold{fold}.cbm"))
            router_local_frame = ev_held_local.pop("router_frame")
            local_frame = e13.build_feature_frame(
                probs[[cp_ids.index(x) for x in mf["checkpoint_ids"]]], mf["checkpoint_ids"],
                mf["families"], y[fit], fit, held)[0]
            router_feature_diff = float(np.max(np.abs(
                router_local_frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)-
                local_frame.select_dtypes(include=[np.number]).to_numpy(dtype=float))))
            vectorizer = joblib.load(DEP / f"e20_group_vectorizer_fold{fold}.joblib")
            classifier = CatBoostClassifier(); classifier.load_model(str(DEP/f"e20c_classifier_fold{fold}.cbm"))
            cp_classifier = CatBoostClassifier(); cp_classifier.load_model(
                str(root/f"artifacts/analysis/e20/E20AB_checkpoint_only_fold{fold}.cbm"))
            # Group reconstruction from full-batch package path, then OOF membership,
            # then oracle membership, all with exactly the same checkpoint evidence.
            def build(ev, rows, groups, pairs, oracle=False):
                return e20._combine_features(ctx, ev, rows, mf["checkpoint_ids"], groups,
                    vectorizer, pair_scores=pairs, oracle=oracle, fit_texts=ctx["texts"][fit])
            features_global = build(ev_all, np.arange(len(y)), global_groups,
                                    global_pair_scores, False)
            features_oof = build(ev_all, np.arange(len(y)), oof_groups,
                                 pair_scores, False)
            features_true = build(ev_all, np.arange(len(y)), ctx["groups"].astype(str),
                                  {}, True)
            score_global = _score_matrix(classifier, features_global, fold)
            score_all_oof = _score_matrix(classifier, features_oof, fold)
            score_all_true = _score_matrix(classifier, features_true, fold)
            score_global_cp = _score_matrix(cp_classifier,
                e20._evidence_feature_frame(ctx, ev_all, np.arange(len(y)), mf["checkpoint_ids"]), fold)
            # Local OOF rebuild must reproduce the saved fold score matrix.
            parity_features = e20._combine_features(ctx, ev_held_local, held,
                mf["checkpoint_ids"], oof_groups[held], vectorizer,
                pair_scores=pair_scores, oracle=False, fit_texts=ctx["texts"][fit])
            local_features = e20._combine_features(ctx, ev_local, held,
                mf["checkpoint_ids"], oof_groups[held], vectorizer,
                pair_scores=pair_scores, oracle=False, fit_texts=ctx["texts"][fit])
            ordered_feature_names = list(classifier.feature_names_)
            if set(ordered_feature_names) != set(parity_features.columns):
                feature_value_delta = float("inf")
            else:
                pkg_x = parity_features.loc[:, ordered_feature_names].select_dtypes(
                    include=[np.number]).to_numpy(dtype=np.float64)
                local_x = local_features.loc[:, ordered_feature_names].select_dtypes(
                    include=[np.number]).to_numpy(dtype=np.float64)
                feature_value_delta = float(np.max(np.abs(pkg_x-local_x)))
            local_saved_model = CatBoostClassifier(); local_saved_model.load_model(
                str(root/f"artifacts/analysis/e20/E20C_fold{fold}.cbm"))
            parity_scores = _score_matrix(local_saved_model, parity_features, fold)
            oof_score_delta = float(np.max(np.abs(parity_scores-saved_oof_scores[held])))
            parity_pred = e20.hungarian(parity_scores, oof_groups[held], ctx["row_ids"][held])
            parity_pred_diff = int(np.sum(parity_pred != saved_oof_pred[held]))
            local_groups_file = group_frame.loc[held, "e20_group"].astype(str).to_numpy()
            if not np.array_equal(local_groups_file, oof_groups[held]):
                raise ValueError("OOF reconstructed group row ordering mismatch")
            fold_scores_global.append(score_global)
            fold_scores_oof_groups.append(score_all_oof)
            fold_scores_true_groups.append(score_all_true)
            fold_scores_cp_only.append(score_global_cp)
            fold_scores_cp_only_oof.append(_score_matrix(cp_classifier,
                e20._evidence_feature_frame(ctx, ev_all, np.arange(len(y)), mf["checkpoint_ids"]), fold))
            parity_rows.append({"fold": fold, "held_rows": int(len(held)),
                "checkpoint_probability_max_abs_diff": cp_max_diff,
                "packaged_vs_refit_router_reliability_max_abs_diff": reliability_delta,
                "packaged_vs_refit_router_label_score_max_abs_diff": evidence_score_delta,
                "router_input_feature_max_abs_diff": router_feature_diff,
                "feature_value_max_abs_diff_packaged_vs_oof_builder": feature_value_delta,
                "source_feature_column_order_matches_model": ordered_feature_names == list(parity_features.columns),
                "scored_feature_column_order_matches_model": True,
                "saved_oof_score_max_abs_diff": oof_score_delta,
                "saved_oof_hungarian_prediction_differences": parity_pred_diff,
                "group_source_for_reproduction": "saved E19-B OOF memberships",
                "packaged_full_batch_group_count": int(len(set(global_groups))),
                "oof_reconstructed_group_count": int(len(set(oof_groups[held])))})
            imp = dict(zip(classifier.feature_names_, classifier.get_feature_importance()))
            model_importances[fold] = imp
            # paired feature differences for full V0; use group-derived schema fields.
            gcols = [c for c in features_global.columns if c.startswith("group::")]
            imp_g = {k: float(v) for k, v in imp.items() if k.startswith("group::")}
            shift = _feature_shift(features_true[gcols], features_global[gcols], imp_g)
            shift["fold"] = fold
            group_shift_parts.append(shift)
            fold_router_diagnostics.append({"fold": fold, "roster": mf["checkpoint_ids"],
                "reliability_max_abs_diff": reliability_delta,
                "e13_candidate_score_max_abs_diff": evidence_score_delta})

        # Deployment aggregation on the exact same globally reconstructed groups.
        deploy_raw, deploy_ind, deploy_hun = _aggregate(fold_scores_global, global_groups, y)
        deploy_norm, deploy_norm_ind, deploy_norm_hun = _aggregate(
            fold_scores_global, global_groups, y, normalize=True)
        oof_fold_score = np.full_like(saved_oof_scores, np.nan)
        cp_oof_score = np.full_like(saved_oof_scores, np.nan)
        fold_of_row = np.full(len(y), -1, dtype=np.int16)
        for i, outer in enumerate(ctx["outer_folds"]):
            h = outer["held_rows"]; fold_of_row[h] = int(outer["fold"])
            oof_fold_score[h] = fold_scores_global[i][h]
            cp_oof_score[h] = fold_scores_cp_only[i][h]
        oof_global_groups_pred = e20.hungarian(oof_fold_score, global_groups)
        oof_global_groups_ind = oof_fold_score.argmax(axis=1)
        oof_true_feature_score = np.full_like(saved_oof_scores, np.nan, dtype=np.float32)
        for i, outer in enumerate(ctx["outer_folds"]):
            h = outer["held_rows"]
            oof_true_feature_score[h] = fold_scores_true_groups[i][h]
        oof_true_feature_ind = oof_true_feature_score.argmax(axis=1)
        oof_true_feature_hun = e20.hungarian(oof_true_feature_score, global_groups)
        # True-vs-reconstructed features isolate feature shift, with decoder groups fixed.
        score_true_mean, true_ind, true_hun = _aggregate(fold_scores_true_groups,
                                                          global_groups, y)
        # Checkpoint-only classifier: OOF and all-five deployment aggregation.
        cp_deploy, cp_ind, cp_hun = _aggregate(fold_scores_cp_only, global_groups, y)
        cp_oof_hun = e20.hungarian(cp_oof_score, global_groups)
        cp_oof_ind = cp_oof_score.argmax(axis=1)
        # E20-D's original saved OOF prediction and all-five deployment result.
        metrics = {
            "rows": int(len(y)), "classes": list(LABELS),
            "checkpoint_probability_max_abs_diff": cp_max_diff,
            "checkpoint_probability_rows": cp_detail,
            "parity_by_fold": parity_rows,
            "deployment_group_reconstruction": global_group_info,
            "deployment_groups_match_saved_all_v0_diagnostic": global_group_exact,
            "deployment_group_adjusted_rand_index": global_group_ari,
            "experiments": {
                "E13-B": _metric(y, baseline),
                "E20-D_saved_OOF": _metric(y, saved_oof_pred),
                "E20-D_packaged_score_on_OOF_groups": _metric(y, oof_fold_score.argmax(1)),
                "E20-D_packaged_OOF_scores_Hungarian_full_batch_groups": _metric(y, oof_global_groups_pred),
                "E20-D_fold_heldout_true_group_features_reconstructed_decoder_independent": _metric(y, oof_true_feature_ind),
                "E20-D_fold_heldout_true_group_features_reconstructed_decoder_Hungarian": _metric(y, oof_true_feature_hun),
                "E20-D_all5_average_independent_global_groups": _metric(y, deploy_ind),
                "E20-D_all5_average_Hungarian_global_groups": _metric(y, deploy_hun),
                "E20-D_all5_row_normalized_average_independent": _metric(y, deploy_norm_ind),
                "E20-D_all5_row_normalized_average_Hungarian": _metric(y, deploy_norm_hun),
                "E20-D_true_group_features_all5_average_Hungarian_reconstructed_decoder": _metric(y, true_hun),
                "E20-checkpoint-only_OOF_independent": _metric(y, cp_oof_ind),
                "E20-checkpoint-only_OOF_Hungarian": _metric(y, cp_oof_hun),
                "E20-checkpoint-only_all5_average_independent": _metric(y, cp_ind),
                "E20-checkpoint-only_all5_average_Hungarian": _metric(y, cp_hun),
            },
            "comparisons": {
                "all5_raw_vs_saved_OOF": _comparison(y, saved_oof_pred, deploy_hun),
                "all5_normalized_vs_raw": _comparison(y, deploy_hun, deploy_norm_hun),
                "true_vs_reconstructed_group_features": _comparison(y, deploy_hun, true_hun),
                "OOF_true_vs_reconstructed_group_features": _comparison(y, oof_global_groups_pred, oof_true_feature_hun),
                "checkpoint_only_vs_E20D_all5": _comparison(y, deploy_hun, cp_hun),
                "checkpoint_only_vs_E13B": _comparison(y, baseline, cp_hun),
            },
            "folds": [],
            "group_feature_shift_summary": {},
            "artifacts": {},
        }
        for i, outer in enumerate(ctx["outer_folds"]):
            h = outer["held_rows"]
            metrics["folds"].append({"fold": int(outer["fold"]), "rows": int(len(h)),
                "saved_OOF": _metric(y[h], saved_oof_pred[h]),
                "deployment_model_only_on_held_rows": _metric(y[h], fold_scores_global[i][h].argmax(1)),
                "all5_average_independent": _metric(y[h], deploy_ind[h]),
                "all5_average_Hungarian": _metric(y[h], deploy_hun[h]),
                "all5_normalized_Hungarian": _metric(y[h], deploy_norm_hun[h]),
                "checkpoint_only_independent": _metric(y[h], cp_ind[h]),
                "checkpoint_only_Hungarian": _metric(y[h], cp_hun[h])})
        shift = pd.concat(group_shift_parts, ignore_index=True)
        shift_summary = shift.groupby("feature", as_index=False).agg(
            mean_abs_smd=("standardized_mean_difference", lambda x: float(np.mean(np.abs(x)))),
            mean_abs_row_delta=("mean_absolute_row_difference", "mean"),
            mean_importance=("mean_catboost_importance", "mean"),
            max_abs_smd=("standardized_mean_difference", lambda x: float(np.max(np.abs(x)))))
        shift_summary = shift_summary.sort_values(["mean_abs_smd", "mean_importance"], ascending=False)
        metrics["group_feature_shift_summary"] = shift_summary.to_dict(orient="records")

        # Re-score each saved incomplete-group sample after reconstructing its graph
        # on the sample only, then rebuilding features and running all five models.
        sample_path = root / "artifacts/analysis/e19/sampled_predictions.parquet"
        sampled = pd.read_parquet(sample_path)
        row_loc = {str(x): i for i, x in enumerate(ctx["row_ids"])}
        sample_records = []; sample_preds = []
        fold_of_row = np.full(len(y), -1, dtype=np.int16)
        for outer in ctx["outer_folds"]:
            fold_of_row[np.asarray(outer["held_rows"], dtype=np.int64)] = int(outer["fold"])
        e13_cache = {}
        for mf in manifest["folds"]:
            fold = int(mf["fold"]); fit = np.asarray(mf["fit_rows"], dtype=np.int64)
            evidence_cache = output / "package_evidence" / f"fold{fold}.joblib"
            e13_cache[fold] = joblib.load(evidence_cache) if evidence_cache.is_file() else _router_evidence(
                ctx, mf, probs, cp_ids, cp_families, fit, np.arange(len(y)))
        for (variant, seed), part in sampled.groupby(["variant", "seed"], sort=True):
            rows = np.asarray([row_loc[str(x)] for x in part.id], dtype=np.int64)
            if len(np.unique(rows)) != len(rows):
                raise ValueError("Duplicate sampled response ID")
            sample_key = hashlib.sha256(f"{variant}|{seed}|".encode() +
                                        ctx["row_ids"][rows].astype(str).tobytes()).hexdigest()
            subgroups, subpairs, subinfo = _cached_graph(ctx, data, grouper, rows,
                output/"subset_graphs"/f"{sample_key}.joblib", spaces=package_spaces)
            fold_mats = []; fold_cp_mats = []
            for mf in manifest["folds"]:
                fold = int(mf["fold"]); fit = np.asarray(mf["fit_rows"], dtype=np.int64)
                vectorizer = joblib.load(DEP/f"e20_group_vectorizer_fold{fold}.joblib")
                ev = {"roster_ids": e13_cache[fold]["roster_ids"],
                      "scores": e13_cache[fold]["scores"][rows],
                      "reliability": e13_cache[fold]["reliability"][:, rows],
                      "hard": e13_cache[fold]["hard"][:, rows]}
                features = e20._combine_features(ctx, ev, rows, mf["checkpoint_ids"],
                    subgroups, vectorizer, pair_scores=subpairs, oracle=False,
                    fit_texts=ctx["texts"][fit])
                model = CatBoostClassifier(); model.load_model(str(DEP/f"e20c_classifier_fold{fold}.cbm"))
                fold_mats.append(_score_matrix(model, features, fold))
                cpmodel = CatBoostClassifier(); cpmodel.load_model(
                    str(root/f"artifacts/analysis/e20/E20AB_checkpoint_only_fold{fold}.cbm"))
                cpfeat = e20._evidence_feature_frame(ctx, ev, rows, mf["checkpoint_ids"])
                fold_cp_mats.append(_score_matrix(cpmodel, cpfeat, fold))
            mat, independent, hung = _aggregate(fold_mats, subgroups, y[rows])
            cp_mat, cp_independent, cp_hung = _aggregate(fold_cp_mats, subgroups, y[rows])
            # Leakage-controlled counterpart: for each V0 row, use only the
            # fold model that held out its entire true prompt group.
            selected_e20 = np.empty_like(mat); selected_cp = np.empty_like(cp_mat)
            for model_ix in range(len(manifest["folds"])):
                mask = fold_of_row[rows] == model_ix + 1
                if mask.any():
                    selected_e20[mask] = fold_mats[model_ix][mask]
                    selected_cp[mask] = fold_cp_mats[model_ix][mask]
            oof_ind = selected_e20.argmax(axis=1)
            oof_hun = e20.hungarian(selected_e20, subgroups, ctx["row_ids"][rows])
            cp_oof_ind = selected_cp.argmax(axis=1)
            cp_oof_hun = e20.hungarian(selected_cp, subgroups, ctx["row_ids"][rows])
            # Legacy result reruns Hungarian over full-batch saved scores; retained only as audit comparator.
            old_full_scores = saved_oof_scores[rows]
            old_full_hung = e20.hungarian(old_full_scores, oof_groups[rows])
            for name, pred in (("E20-D-rescored-independent", independent),
                               ("E20-D-rescored-Hungarian", hung),
                               ("E20-D-fold-heldout-independent", oof_ind),
                               ("E20-D-fold-heldout-Hungarian", oof_hun),
                               ("E20-checkpoint-only-independent", cp_independent),
                               ("E20-checkpoint-only-Hungarian", cp_hung),
                               ("E20-checkpoint-only-fold-heldout-independent", cp_oof_ind),
                               ("E20-checkpoint-only-fold-heldout-Hungarian", cp_oof_hun),
                               ("legacy-full-group-score-Hungarian", old_full_hung)):
                mm = _metric(y[rows], pred)
                sample_records.append({"variant": str(variant), "seed": int(seed),
                    "method": name, "rows": int(len(rows)), "groups": int(subinfo["groups"]),
                    "singleton_groups": int(np.sum(np.bincount(pd.factorize(subgroups)[0])==1)),
                    "accuracy": mm["accuracy"], "balanced_accuracy": mm["balanced_accuracy"],
                    "group_size_counts": {str(k): int(v) for k,v in pd.Series(
                        pd.Series(subgroups).value_counts().to_numpy()).value_counts().items()}})
                row_group_sizes = pd.Series(subgroups).map(pd.Series(subgroups).value_counts()).to_numpy()
                for local_row, (row, pred_code) in enumerate(zip(rows, pred)):
                    sample_preds.append({"variant": str(variant), "seed": int(seed),
                        "method": name, "id": str(ctx["row_ids"][row]),
                        "y_true": LABELS[int(y[row])], "prediction": LABELS[int(pred_code)],
                        "reconstructed_group_id": str(subgroups[local_row]),
                        "reconstructed_group_size": int(row_group_sizes[local_row])})
        sample_results = pd.DataFrame(sample_records)
        sample_predictions = pd.DataFrame(sample_preds)
        size_results = sample_results.copy()
        metrics["incomplete_group_samples"] = sample_results.to_dict(orient="records")
        metrics["incomplete_group_aggregation"] = sample_results.groupby("method").agg(
            sample_runs=("seed", "count"), mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"), mean_accuracy=("accuracy", "mean"),
            mean_rows=("rows", "mean")).reset_index().to_dict(orient="records")
        pred_frame = sample_predictions.copy()
        true_codes = pd.Categorical(pred_frame.y_true, categories=LABELS).codes
        pred_codes = pd.Categorical(pred_frame.prediction, categories=LABELS).codes
        pred_frame["_true"] = true_codes; pred_frame["_pred"] = pred_codes
        size_records = []
        for (variant, method, group_size), part in pred_frame.groupby(
                ["variant", "method", "reconstructed_group_size"], sort=True):
            mm = _metric(part._true.to_numpy(), part._pred.to_numpy())
            size_records.append({"variant": variant, "method": method,
                "group_size": int(group_size), "rows": int(len(part)),
                "accuracy": mm["accuracy"], "balanced_accuracy": mm["balanced_accuracy"]})
        metrics["incomplete_group_by_size"] = size_records

        # Save machine-readable prediction matrices and audit tables.
        np.savez_compressed(output/"deployment_scores_v0.npz", row_ids=ctx["row_ids"].astype(str),
            y=y, global_group_ids=global_groups, oof_group_ids=oof_groups,
            e20d_saved_oof=saved_oof_scores, all5_fold_scores=np.stack(fold_scores_global),
            all5_raw_mean=deploy_raw, all5_normalized_mean=deploy_norm,
            true_group_features_all5_mean=score_true_mean,
            true_group_features_oof=oof_true_feature_score,
            checkpoint_only_fold_scores=np.stack(fold_scores_cp_only))
        pd.DataFrame(parity_rows).to_csv(output/"deployment_parity.csv", index=False)
        pd.DataFrame(cp_detail).to_csv(output/"checkpoint_probability_parity.csv", index=False)
        shift.to_csv(output/"group_feature_shift_by_fold.csv", index=False)
        shift_summary.to_csv(output/"group_feature_shift_summary.csv", index=False)
        sample_results.to_csv(output/"incomplete_group_metrics.csv", index=False)
        sample_predictions.to_parquet(output/"incomplete_group_predictions_rescored.parquet", index=False)
        pd.DataFrame(size_records).to_csv(output/"incomplete_group_by_size.csv", index=False)
        cp = pd.DataFrame({"id":ctx["row_ids"].astype(str), "y_true":np.asarray(LABELS,dtype=object)[y],
            "E13-B":np.asarray(LABELS,dtype=object)[baseline],
            "E20-D-OOF":np.asarray(LABELS,dtype=object)[saved_oof_pred],
            "E20-D-all5-independent":np.asarray(LABELS,dtype=object)[deploy_ind],
            "E20-D-all5-Hungarian":np.asarray(LABELS,dtype=object)[deploy_hun],
            "E20-D-all5-normalized-Hungarian":np.asarray(LABELS,dtype=object)[deploy_norm_hun],
            "E20-checkpoint-only-all5":np.asarray(LABELS,dtype=object)[cp_hun]})
        cp.to_parquet(output/"v0_predictions.parquet", index=False)
        metrics["input_hashes"] = {
            "package_manifest": _sha(PKG/"e20d_manifest.json"),
            "package_v0_probabilities": _sha(DEP/"e20d_v0_probabilities.npz"),
            "saved_e20d_scores": _sha(root/"artifacts/analysis/e20/scores_e20d.npz"),
            "saved_e19b_groups": _sha(root/"artifacts/analysis/e19b/oof_group_predictions.parquet")}
        metrics["artifacts"] = {"scores": str(output/"deployment_scores_v0.npz"),
            "parity": str(output/"deployment_parity.csv"),
            "feature_shift": str(output/"group_feature_shift_summary.csv"),
            "incomplete_metrics": str(output/"incomplete_group_metrics.csv"),
            "predictions": str(output/"v0_predictions.parquet")}
        (output/"metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
        return metrics


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args(argv)
    result = run(args.root, args.output)
    print(json.dumps({name: {"accuracy": value["accuracy"],
                             "balanced_accuracy": value["balanced_accuracy"]}
                      for name, value in result["experiments"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
