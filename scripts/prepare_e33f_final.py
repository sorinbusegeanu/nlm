"""Fit the deployable E33-F CatBoost rescoring model from prompt-OOF features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src import e33, e34, ensemble_catalog as catalog
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]


def main():
    v0 = catalog.cascades._v0(ROOT)
    _, members, cached, excluded, _ = catalog._load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    manifest = json.loads((ROOT / "artifacts/analysis/e31/e31_manifest.json").read_text())
    roster = list(map(str, manifest["roster"]))
    if len(roster) != 40 or set(roster) - set(cached):
        raise ValueError("E31 fixed 40-checkpoint roster caches are incomplete")
    ids = v0.id.astype(str).to_numpy()
    y = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    prompts = v0.prompt_id.astype(str).to_numpy()
    probabilities = np.stack([cached[sid] for sid in roster]).astype(np.float32)
    ranks = e33._rank_positions(probabilities)
    score_cache = np.load(ROOT / "artifacts/analysis/e31/roster_scenario_evaluation/predictions_and_scores.npz",
                          allow_pickle=True)
    if score_cache["row_ids"].astype(str).tolist() != ids.tolist():
        raise ValueError("E31 OOF score cache row ordering changed")
    scores = score_cache["score_40"].astype(np.float64)
    families = [_model_family(members[sid]["model"]) for sid in roster]
    e33._BUILD_CP_IDS = roster
    final = e34._fit_final_deployment(probabilities, ranks, scores, y, prompts,
        families, roster, ROOT / "artifacts/analysis/e33", tag="e33")
    final["baseline_router"] = "artifacts/analysis/e31/e31_router.cbm"
    final["roster_source"] = "artifacts/analysis/e31/e31_manifest.json"
    final["roster"] = roster
    final["base_score_source"] = "artifacts/analysis/e31/roster_scenario_evaluation/predictions_and_scores.npz:score_40"
    final["purpose"] = "Deployment classifier only; reported E33-F OOF metrics remain the saved outer-fold results."
    model = CatBoostClassifier()
    model.load_model(str(ROOT / "artifacts/analysis/e33/e33_final.cbm"))
    schema = {
        "schema_version": 1,
        "experiment": "E33-F",
        "checkpoint_order": roster,
        "class_order": LABELS,
        "feature_order": list(model.feature_names_),
        "categorical_features": ["candidate_label"],
        "model_class_order": [int(x) for x in model.classes_],
        "target_semantics": "1=candidate_label_is_incorrect; p_correct=1-p(class=1)",
        "adjustment": "adjusted_score = E31_score * predicted_candidate_correctness",
        "reliability_stat_keys": sorted(np.load(ROOT / "artifacts/analysis/e33/e33_final_reliability.npz").files),
    }
    schema_path = ROOT / "artifacts/analysis/e33/e33_final_feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    final["feature_schema_path"] = str(schema_path.relative_to(ROOT))
    final["feature_schema_sha256"] = e33._sha(schema_path)
    final["class_order"] = LABELS
    final["model_class_order"] = schema["model_class_order"]
    path = ROOT / "artifacts/analysis/e33/e33_final_deployment.json"
    path.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
