"""SQLite experiment registry with reproducibility and artifact metadata."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class Registry:
    def __init__(self, path="experiments/registry.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY, model TEXT NOT NULL, split TEXT NOT NULL,
                dataset_hash TEXT, split_hash TEXT, model_revision TEXT, code_revision TEXT, config_hash TEXT, seed INTEGER,
                fold_id INTEGER, runtime_seconds REAL, artifact_path TEXT,
                config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            # Add new metadata columns when opening registries created by earlier versions.
            columns = {row[1] for row in db.execute("PRAGMA table_info(experiments)")}
            for name, kind in [("split_hash", "TEXT"), ("model_revision", "TEXT"), ("config_hash", "TEXT"), ("seed", "INTEGER"), ("fold_id", "INTEGER"),
                               ("runtime_seconds", "REAL"), ("artifact_path", "TEXT")]:
                if name not in columns: db.execute(f"ALTER TABLE experiments ADD COLUMN {name} {kind}")

    def connect(self): return sqlite3.connect(self.path)

    def add(self, experiment_id, model, split, config, metrics, dataset_hash=None, code_revision=None,
            seed=None, fold_id=None, runtime_seconds=None, artifact_path=None, config_hash=None,
            split_hash=None, model_revision=None):
        with self.connect() as db:
            db.execute("""INSERT INTO experiments(experiment_id,model,split,dataset_hash,split_hash,
                model_revision,code_revision,config_hash,seed,fold_id,runtime_seconds,artifact_path,config_json,metrics_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(experiment_id) DO UPDATE SET
                model=excluded.model,split=excluded.split,dataset_hash=excluded.dataset_hash,
                split_hash=excluded.split_hash,model_revision=excluded.model_revision,
                code_revision=excluded.code_revision,config_hash=excluded.config_hash,seed=excluded.seed,
                fold_id=excluded.fold_id,runtime_seconds=excluded.runtime_seconds,
                artifact_path=excluded.artifact_path,config_json=excluded.config_json,
                metrics_json=excluded.metrics_json,created_at=CURRENT_TIMESTAMP""",
                (experiment_id, model, split, dataset_hash, split_hash, model_revision, code_revision, config_hash, seed, fold_id,
                 runtime_seconds, artifact_path, json.dumps(config, sort_keys=True, default=str),
                 json.dumps(metrics, sort_keys=True, default=str)))

    def rows(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM experiments ORDER BY created_at DESC")]
