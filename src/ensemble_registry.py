"""Human-readable ensemble history and machine-readable active candidate roster."""
from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


REGISTRY_START = "<!-- ensemble-registry:start -->"
REGISTRY_END = "<!-- ensemble-registry:end -->"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_ensemble_registry(path="docs/ensembles.md"):
    """Read the YAML registry embedded in docs/ensembles.md."""
    path = Path(path)
    if not path.exists():
        return {"ensembles": []}
    text = path.read_text(encoding="utf-8")
    match = re.search(
        re.escape(REGISTRY_START) + r"\s*```yaml\s*\n(.*?)\n```\s*" + re.escape(REGISTRY_END),
        text, flags=re.DOTALL)
    if not match:
        return {"ensembles": []}
    value = yaml.safe_load(match.group(1)) or {}
    ensembles = value.get("ensembles", [])
    if not isinstance(ensembles, list):
        raise ValueError(f"ensemble registry in {path} must contain an ensembles list")
    return {**{key: item for key, item in value.items() if key != "ensembles"},
            "ensembles": ensembles}


def write_ensemble_registry(ensembles, path="docs/ensembles.md", *, metadata=None):
    """Atomically replace the structured registry block, preserving prose."""
    path = Path(path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = "# Ensemble registry\n\n"
    value = dict(metadata or {})
    value["ensembles"] = list(ensembles)
    block = (f"{REGISTRY_START}\n```yaml\n" +
             yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip() +
             f"\n```\n{REGISTRY_END}")
    pattern = re.escape(REGISTRY_START) + r".*?" + re.escape(REGISTRY_END)
    if re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, lambda _: block, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    _atomic_write(path, text)


def read_active_roster(path="artifacts/predictions/active_model_roster.json"):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("candidates"), list):
        raise ValueError(f"active ensemble roster in {path} has no candidates list")
    return value


def write_active_roster(roster, path="artifacts/predictions/active_model_roster.json"):
    _atomic_write(Path(path), json.dumps(roster, indent=2, ensure_ascii=False) + "\n")


def _roster_hash(candidates):
    identity = sorted((item["model"], item["candidate_id"]) for item in candidates)
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def _candidate_record(candidate):
    manifest = candidate["model_manifest"]
    checkpoint = Path(candidate.get("checkpoint_path") or "")
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    return {
        "model": str(candidate["model"]).strip().lower(),
        "candidate_id": candidate["candidate_id"],
        "training_dataset": candidate["dataset"],
        "source_manifest": candidate["manifest_path"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_filename": checkpoint.name,
        "selected_epoch": manifest.get("selected_epoch"),
        "validation_balanced_accuracy": float(candidate["validation_score"]),
    }


def refresh_prediction_factory(*, artifact_dir="artifacts", data_dir="data/processed",
                                reference_dataset="v0", completed_manifest=None,
                                evaluate_if_unchanged=True, registry_path="docs/ensembles.md",
                                roster_path="artifacts/predictions/active_model_roster.json"):
    """Update one checkpoint per model, evaluate the pool, and append a version on change."""
    from .prediction_factory import (_align_candidates, _read_candidates,
                                     evaluate_prediction_strategies)

    artifact_dir = Path(artifact_dir)
    predictions_dir = artifact_dir / "predictions"
    all_candidates, read_problems = _read_candidates(predictions_dir)
    if read_problems:
        # Ignore malformed unrelated history, but do not hide missing candidates from the active pool.
        read_problems = [item for item in read_problems if "invalid probabilities" in item]
    if not all_candidates:
        raise FileNotFoundError(f"no readable validation prediction manifests in {predictions_dir}")
    _, _, _, aligned, excluded, _ = _align_candidates(
        all_candidates, reference_dataset=reference_dataset)
    if excluded and completed_manifest:
        completed_id = json.loads(Path(completed_manifest).read_text()).get("experiment_id")
        excluded_ids = {item["candidate_id"] for item in excluded}
        if completed_id in excluded_ids:
            raise ValueError(f"new model {completed_id} does not align to the frozen {reference_dataset} validation rows")

    best_by_model = {}
    for candidate in aligned:
        key = str(candidate["model"]).strip().lower()
        prior = best_by_model.get(key)
        if prior is None or (candidate["validation_score"], candidate["candidate_id"]) > (
                prior["validation_score"], prior["candidate_id"]):
            best_by_model[key] = candidate

    active = read_active_roster(roster_path)
    if active is None:
        active_candidates = {model: candidate for model, candidate in best_by_model.items()}
    else:
        active_candidates = {}
        for item in active["candidates"]:
            candidate_id = str(item["candidate_id"])
            candidate = next((value for value in aligned if value["candidate_id"] == candidate_id), None)
            if candidate is None:
                raise FileNotFoundError(f"active candidate {candidate_id} is no longer readable or validation-aligned")
            active_candidates[str(item["model"]).strip().lower()] = candidate
        for model, candidate in best_by_model.items():
            incumbent = active_candidates.get(model)
            if incumbent is None or candidate["validation_score"] > incumbent["validation_score"]:
                active_candidates[model] = candidate

    candidate_records = [_candidate_record(active_candidates[key]) for key in sorted(active_candidates)]
    roster_hash = _roster_hash(candidate_records)
    changed = active is None or roster_hash != active.get("roster_hash")
    timestamp = datetime.now(timezone.utc).isoformat()
    roster = {
        "schema_version": 1,
        "reference_dataset": reference_dataset,
        "updated_at_utc": timestamp,
        "roster_hash": roster_hash,
        "candidates": candidate_records,
        "last_prediction_experiment_id": active.get("last_prediction_experiment_id") if active else None,
        "last_recorded_roster_hash": active.get("last_recorded_roster_hash") if active else None,
    }
    if changed:
        write_active_roster(roster, roster_path)

    latest_registry = read_ensemble_registry(registry_path)
    latest_entries = latest_registry["ensembles"]
    last_hash = latest_registry.get("last_recorded_active_roster_hash")
    optimizer_protocol = "forward_group_cv_weighted_v1"
    record_needed = (last_hash != roster_hash or
                     latest_registry.get("optimizer_protocol") != optimizer_protocol)
    if not changed and not evaluate_if_unchanged and not record_needed:
        return {"status": "unchanged", "roster_changed": False,
                "candidate_count": len(candidate_records), "ensemble_version": None}

    report = evaluate_prediction_strategies(
        artifact_dir, reference_dataset=reference_dataset,
        candidate_ids=[item["candidate_id"] for item in candidate_records],
        data_dir=data_dir,
        previous_champion=latest_entries[-1] if latest_entries else None)
    roster["last_prediction_experiment_id"] = report["experiment_id"]
    candidate_by_id = {item["candidate_id"]: item for item in candidate_records}

    if record_needed:
        next_number = 1 + max((int(str(item.get("version", "v0")).lstrip("v"))
                               for item in latest_entries), default=0)
        selected = [candidate_by_id[item] for item in report["selected_members"]]
        completed_id = None
        if completed_manifest:
            completed_id = json.loads(Path(completed_manifest).read_text()).get("experiment_id")
        entry = {
            "version": f"v{next_number}",
            "created_at_utc": roster["updated_at_utc"],
            "trigger_experiment_id": completed_id,
            "candidate_roster_hash": roster_hash,
            "candidate_checkpoints": candidate_records,
            "selected_strategy": report["selected_strategy"],
            "selected_comparison": report.get("selected_comparison", report["selected_strategy"]),
            "selected_members": selected,
            "predicted_balanced_accuracy": float(report["selected_validation_balanced_accuracy"]),
            "validation_accuracy": float(report["selected_validation_accuracy"]),
            "group_oof_metrics": report["strategy_metrics"],
            "selected_weights": report["selected_weights"],
            "individual_accuracy": report["individual_accuracy"],
            "forward_selection_steps": [{key: step[key] for key in
                ("step", "added_candidate_id", "accuracy", "log_loss")}
                for step in report["selection_steps"]],
            "optimization_summary": report["summary_path"],
            "forward_steps_artifact": report["forward_steps_path"],
            "submitted_balanced_accuracy": 0.0,
            "prediction_experiment_id": report["experiment_id"],
            "prediction_manifest": report["manifest_path"],
            "validation_rows": int(report["validation_rows"]),
            "optimizer_protocol": optimizer_protocol,
        }
        latest_entries.append(entry)
        metadata = {key: value for key, value in latest_registry.items() if key != "ensembles"}
        metadata.update({
            "active_candidate_roster_hash": roster_hash,
            "active_candidate_roster": candidate_records,
            "last_recorded_active_roster_hash": roster_hash,
            "optimizer_protocol": optimizer_protocol,
            "checkpoint_replacement_policy": "replace a model checkpoint only when validation balanced accuracy is higher",
        })
        write_ensemble_registry(latest_entries, registry_path, metadata=metadata)
        roster["last_recorded_roster_hash"] = roster_hash
    elif latest_entries:
        roster["last_recorded_roster_hash"] = last_hash or roster_hash
        latest = latest_entries[-1]
        if (latest.get("candidate_roster_hash") == roster_hash and
                latest.get("optimizer_protocol", optimizer_protocol) == optimizer_protocol):
            latest.update({
                "selected_strategy": report["selected_strategy"],
                "selected_comparison": report.get("selected_comparison", report["selected_strategy"]),
                "selected_members": [candidate_by_id[item] for item in report["selected_members"]],
                "predicted_balanced_accuracy": float(report["selected_validation_balanced_accuracy"]),
                "validation_accuracy": float(report["selected_validation_accuracy"]),
                "group_oof_metrics": report["strategy_metrics"],
                "selected_weights": report["selected_weights"],
                "individual_accuracy": report["individual_accuracy"],
                "forward_selection_steps": [{key: step[key] for key in
                    ("step", "added_candidate_id", "accuracy", "log_loss")}
                    for step in report["selection_steps"]],
                "optimization_summary": report["summary_path"],
                "forward_steps_artifact": report["forward_steps_path"],
                "prediction_experiment_id": report["experiment_id"],
                "prediction_manifest": report["manifest_path"],
                "optimizer_protocol": optimizer_protocol,
            })
            metadata = {key: value for key, value in latest_registry.items() if key != "ensembles"}
            metadata["optimizer_protocol"] = optimizer_protocol
            write_ensemble_registry(latest_entries, registry_path, metadata=metadata)

    write_active_roster(roster, roster_path)
    return {"status": "recorded" if record_needed else "evaluated",
            "roster_changed": changed, "candidate_count": len(candidate_records),
            "ensemble_version": latest_entries[-1]["version"] if record_needed else None,
            "prediction_experiment_id": report["experiment_id"],
            "predicted_balanced_accuracy": report["selected_validation_balanced_accuracy"],
            "selected_strategy": report["selected_strategy"],
            "selected_comparison": report.get("selected_comparison", report["selected_strategy"]),
            "selected_members": report["selected_members"],
            "excluded_candidates": excluded}
