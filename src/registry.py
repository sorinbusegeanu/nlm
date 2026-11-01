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
                sequence_number INTEGER UNIQUE, description TEXT NOT NULL DEFAULT '',
                dataset_hash TEXT, split_hash TEXT, model_revision TEXT, code_revision TEXT, config_hash TEXT, seed INTEGER,
                fold_id INTEGER, runtime_seconds REAL, artifact_path TEXT,
                config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            # Add new metadata columns when opening registries created by earlier versions.
            columns = {row[1] for row in db.execute("PRAGMA table_info(experiments)")}
            for name, kind in [("split_hash", "TEXT"), ("model_revision", "TEXT"), ("config_hash", "TEXT"), ("seed", "INTEGER"), ("fold_id", "INTEGER"),
                               ("runtime_seconds", "REAL"), ("artifact_path", "TEXT"),
                               ("sequence_number", "INTEGER"), ("description", "TEXT NOT NULL DEFAULT ''")]:
                if name not in columns: db.execute(f"ALTER TABLE experiments ADD COLUMN {name} {kind}")
            # Give existing rows stable sequence numbers in their original order.
            missing = db.execute("SELECT experiment_id FROM experiments WHERE sequence_number IS NULL ORDER BY created_at, experiment_id").fetchall()
            next_number = db.execute("SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM experiments").fetchone()[0]
            for offset, (experiment_id,) in enumerate(missing):
                db.execute("UPDATE experiments SET sequence_number=? WHERE experiment_id=?", (next_number + offset, experiment_id))
            db.execute("UPDATE experiments SET description='Registered ' || model || ' experiment on ' || split || ' data.' WHERE description IS NULL OR description=''")

    def connect(self): return sqlite3.connect(self.path)

    def add(self, experiment_id, model, split, config, metrics, dataset_hash=None, code_revision=None,
            seed=None, fold_id=None, runtime_seconds=None, artifact_path=None, config_hash=None,
            split_hash=None, model_revision=None, description=""):
        if not description:
            description = f"Registered {model} experiment on {split} data."
        with self.connect() as db:
            existing = db.execute("SELECT sequence_number FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            sequence_number = existing[0] if existing else db.execute("SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM experiments").fetchone()[0]
            db.execute("""INSERT INTO experiments(experiment_id,model,split,sequence_number,description,dataset_hash,split_hash,
                model_revision,code_revision,config_hash,seed,fold_id,runtime_seconds,artifact_path,config_json,metrics_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(experiment_id) DO UPDATE SET
                model=excluded.model,split=excluded.split,dataset_hash=excluded.dataset_hash,
                description=excluded.description,
                split_hash=excluded.split_hash,model_revision=excluded.model_revision,
                code_revision=excluded.code_revision,config_hash=excluded.config_hash,seed=excluded.seed,
                fold_id=excluded.fold_id,runtime_seconds=excluded.runtime_seconds,
                artifact_path=excluded.artifact_path,config_json=excluded.config_json,
                metrics_json=excluded.metrics_json,created_at=CURRENT_TIMESTAMP""",
                (experiment_id, model, split, sequence_number, description, dataset_hash, split_hash, model_revision, code_revision, config_hash, seed, fold_id,
                 runtime_seconds, artifact_path, json.dumps(config, sort_keys=True, default=str),
                 json.dumps(metrics, sort_keys=True, default=str)))

    def rows(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM experiments ORDER BY sequence_number")]
