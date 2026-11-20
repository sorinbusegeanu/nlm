"""Fold-local selection rules for the E-series ensemble catalog."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import ensemble_cascade_comparison as cascades
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

LABEL_ARRAY = np.asarray(LABELS, dtype=object)


def _select_mean(algorithm, ranked, probabilities, actual, support, f1, families, limit):
    if algorithm == "family_champions":
        champions = {}
        for index in ranked:
            champions.setdefault(families[index], index)
        return list(champions.values())[:limit]
    if algorithm == "label_coverage":
        covered = np.zeros(len(LABELS))
        roster = []
        for _ in range(limit):
            gain = np.maximum(f1 - covered, 0).mean(axis=1)
            chosen = max((index for index in ranked if index not in roster), key=gain.__getitem__)
            roster.append(chosen)
            covered = np.maximum(covered, f1[chosen])
        return roster
    if algorithm == "greedy_mean":
        roster = ranked[:1]
        summed = probabilities[roster[0]].copy()
        row_weights = 1 / (len(LABELS) * support[actual])
        current = (summed.argmax(axis=1) == actual) @ row_weights
        while len(roster) < limit:
            scores = ((probabilities + summed).argmax(axis=2) == actual) @ row_weights
            chosen = max((index for index in ranked if index not in roster), key=scores.__getitem__)
            if scores[chosen] <= current + 1e-12:
                break
            roster.append(chosen)
            summed += probabilities[chosen]
            current = scores[chosen]
        return roster
    return ranked[:limit]


def _specialist_switches(top_two, hard, truth, fit, held, ids, min_support):
    fitting, testing = top_two[fit], top_two[held]
    predicted = testing[:, 0].copy()
    rules = []
    for first, second in np.unique(fitting, axis=0):
        pair_rows = fit[(fitting[:, 0] == first) & (fitting[:, 1] == second)]
        fires = hard[:, pair_rows] == second
        support = fires.sum(axis=1)
        delta = fires @ ((truth[pair_rows] == second).astype(int) - (truth[pair_rows] == first))
        eligible = np.flatnonzero((support >= min_support) & (delta > 0))
        if not len(eligible):
            continue
        chosen = min(eligible, key=lambda index: (-delta[index], -support[index], index))
        switch = (testing[:, 0] == first) & (testing[:, 1] == second) & (hard[chosen, held] == second)
        predicted[switch] = second
        rules.append({"checkpoint_id": ids[chosen], "from": LABELS[first], "to": LABELS[second],
                      "fit_support": int(support[chosen]), "fit_net_correct": int(delta[chosen])})
    return LABEL_ARRAY[predicted], rules


def evaluate_rule(strategy, folds, v0, members, probabilities):
    ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    if not ids:
        raise ValueError("No eligible cached checkpoints")
    prob = np.stack([probabilities[sid] for sid in ids])
    hard = prob.argmax(axis=2)
    families = [_model_family(members[sid]["model"]) for sid in ids]
    hard_labels = {sid: LABEL_ARRAY[hard[index]] for index, sid in enumerate(ids)}
    truth = pd.Categorical(v0.model, categories=LABELS).codes
    y = v0.model.to_numpy(dtype=object)
    predicted = np.empty(len(v0), dtype=object)
    algorithm = strategy["algorithm"]
    records = []
    splits = list(GroupKFold(n_splits=folds).split(v0, truth, v0.prompt_id))
    all_rows = np.arange(len(v0))
    splits.append((all_rows, all_rows))
    for split_index, (fit, held) in enumerate(splits):
        if split_index == folds:
            held_out_predictions = predicted.copy()
        actual = truth[fit]
        support = np.bincount(actual, minlength=len(LABELS))
        if not support.all():
            raise ValueError("The fitting partition must contain all canonical labels")
        counts = np.stack([np.bincount(row, minlength=len(LABELS)) for row in hard[:, fit]])
        correct = np.stack([np.bincount(row[row == actual], minlength=len(LABELS)) for row in hard[:, fit]])
        f1 = 2 * correct / (counts + support)
        ba = (correct / support).mean(axis=1)
        ranked = sorted(range(len(ids)), key=lambda index: (-ba[index], -correct[index].sum(), index))
        record = {"held_ids": v0.id.iloc[held].tolist()}

        if algorithm == "label_f1_mean":
            pool = _select_mean("label_coverage", ranked, prob[:, fit], actual, support, f1, families, min(strategy["max_checkpoints"], len(ids)))
            scores = np.empty((len(held), len(LABELS)))
            rules = []
            for index, label in enumerate(LABELS):
                eligible = [i for i in pool if counts[i, index] >= strategy["min_support"]]
                selected = sorted(eligible, key=lambda i: (-f1[i, index], i))[:strategy["rules_per_label"]]
                weights = f1[selected, index]
                if len(selected) < strategy["rules_per_label"] or weights.sum() <= 0:
                    raise ValueError(f"Not enough supported positive F1 weights for {label}")
                scores[:, index] = np.average(prob[selected][:, held, index], axis=0, weights=weights)
                rules.extend({"checkpoint_id": ids[i], "label": label, "weight": float(w)}
                             for i, w in zip(selected, weights))
            predicted[held] = LABEL_ARRAY[scores.argmax(axis=1)]
            record.update(rules=rules, checkpoint_ids=sorted({rule["checkpoint_id"] for rule in rules}))
        else:
            limit = min(strategy["max_checkpoints"], len(ids))
            selected = _select_mean(algorithm, ranked, prob[:, fit], actual, support, f1, families, limit)
            mean = prob[selected].mean(axis=0)
            roster = [ids[index] for index in selected]
            if algorithm == "top_mean_cascade":
                candidates = [(ids[i], LABELS[j]) for i in selected for j in range(len(LABELS))
                              if counts[i, j] >= strategy["min_support"]]
                fallback = LABEL_ARRAY[mean.argmax(axis=1)]
                cascade = strategy["cascade"]
                if cascade == "C4":
                    rules, _, rounds = cascades._select_two(
                        cascade, "precision", fit, y, hard_labels, candidates, fallback)
                else:
                    rules, _ = cascades._select_single(cascade, "precision", fit, y, hard_labels, candidates)
                    rounds = {}
                predicted[held] = cascades._apply(rules, held, y, hard_labels, fallback, rounds)[0]
                record["rules"] = [{"checkpoint_id": sid, "label": label, "round": rounds.get((sid, label), 1)}
                                   for sid, label in rules]
            elif algorithm == "top_two_specialist":
                top_two = np.argsort(-mean, axis=1, kind="stable")[:, :2]
                predicted[held], record["rules"] = _specialist_switches(
                    top_two, hard[selected], truth, fit, held, roster, strategy["min_support"])
            else:
                predicted[held] = LABEL_ARRAY[mean[held].argmax(axis=1)]
            if "rules" in record:
                record["fallback_checkpoint_ids"] = roster
            record["checkpoint_ids"] = list(dict.fromkeys(
                roster + [rule["checkpoint_id"] for rule in record.get("rules", [])]))
        records.append(record)
    deployment = records.pop()
    deployment.pop("held_ids")
    # The last pass resolves the deployment rules on full V0. Restore the
    # held-out predictions from the separate fold passes.
    return held_out_predictions, records, deployment
