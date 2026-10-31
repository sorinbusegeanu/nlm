# Name That LLM

Reproducible experiments for identifying the model that generated a passage. The initial dataset and experiment definitions are in [`docs/`](docs/mvp_design.md).

## Local workflow

Use Python 3.10 or newer. Install dependencies on the machine used for development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,dashboard]'
```

For neural training, install a PyTorch build suited to the local hardware and then run `python -m pip install -e '.[neural]'`. The transformer checkpoint IDs and immutable revisions must be configured after their availability is confirmed on the target machine.

Prepare frozen dataset V0 artifacts, with a prompt-grouped 20% holdout and five development folds:

```bash
python -m src.pipeline data prepare --config configs/data/official_v0.yaml
python -m src.pipeline models train --model svc --dataset v0
```

Run the untouched holdout evaluation once after choosing the model recipe:

```bash
python -m src.pipeline models evaluate-holdout --model svc --dataset v0
```

Start the dashboard with `streamlit run dashboard/app.py`. Run the regression suite with `python -m unittest discover -s tests`.

Artifacts and the SQLite experiment registry are written under `data/processed/`, `artifacts/`, and `experiments/`. Raw official inputs live under `data/raw/`.
