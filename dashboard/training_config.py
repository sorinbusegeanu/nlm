"""Resolve the effective configuration shown for a live training process."""
from pathlib import Path


def live_training_config(root, action, model, options):
    """Load the run's config file, then apply only explicit CLI overrides."""
    root = Path(root)
    config_path = Path(options.get("--config", f"configs/models/{model}.yaml"))
    if not config_path.is_absolute():
        config_path = root / config_path

    config = {}
    if config_path.is_file():
        import yaml
        loaded = yaml.safe_load(config_path.read_text())
        if isinstance(loaded, dict):
            config.update(loaded)
    config["model"] = model

    training_dataset = options.get("--dataset", "v1")
    validation_dataset = options.get("--validation-dataset", "v0")
    config["training_data"] = training_dataset
    config["validation_data"] = validation_dataset

    defaults = {
        "train": {"epochs": 10},
        "continue-qwen": {"epochs": 4},
        "continue-encoder": {
            "epochs": 10,
            "learning_rate": 5e-5,
            "learning_rate_decay": 0.9,
        },
    }
    for key, value in defaults.get(action, {}).items():
        config.setdefault(key, value)

    explicit_options = {
        "--epochs": ("epochs", int),
        "--learning-rate": ("learning_rate", float),
        "--learning-rate-decay": ("learning_rate_decay", float),
    }
    for option, (key, convert) in explicit_options.items():
        if option in options:
            config[key] = convert(options[option])
    return config
