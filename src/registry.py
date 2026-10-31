"""Small SQLite experiment and metric registry."""
import json, sqlite3
from pathlib import Path


class Registry:
    def __init__(self, path="experiments/registry.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY, model TEXT NOT NULL, split TEXT NOT NULL,
                dataset_hash TEXT, code_revision TEXT, config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def connect(self):
        return sqlite3.connect(self.path)

    def add(self, experiment_id, model, split, config, metrics, dataset_hash=None, code_revision=None):
        with self.connect() as db:
            db.execute("INSERT INTO experiments(experiment_id,model,split,dataset_hash,code_revision,config_json,metrics_json) VALUES(?,?,?,?,?,?,?)",
                       (experiment_id, model, split, dataset_hash, code_revision, json.dumps(config, sort_keys=True), json.dumps(metrics, sort_keys=True)))

    def rows(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT * FROM experiments ORDER BY created_at DESC")]
