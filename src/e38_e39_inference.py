"""Inference helpers for the E38/E39 probability members used by E42/E43."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


def _align(values: np.ndarray, source_classes: list[str], classes: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros((len(values), len(classes)), dtype=np.float64)
    indexes = {str(label): i for i, label in enumerate(classes)}
    for j, label in enumerate(source_classes):
        if str(label) not in indexes:
            raise ValueError(f"unknown class in feature model output: {label}")
        out[:, indexes[str(label)]] = values[:, j]
    if (not np.isfinite(out).all() or (out < 0).any()
            or not np.allclose(out.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("feature model returned invalid class probabilities")
    return out


def load_e38_bundle(bundle_path: str | Path):
    """Load the fitted E38 artifact after its src.feature_factory module is installed."""
    return joblib.load(bundle_path)


def predict_e38(bundle_path_or_bundle, texts: list[str], classes: list[str]) -> np.ndarray:
    """Run the saved enhanced SVC bundle with its original fitted feature views."""
    bundle = (joblib.load(bundle_path_or_bundle)
              if isinstance(bundle_path_or_bundle, (str, Path)) else bundle_path_or_bundle)
    if list(map(str, bundle["classes"])) != list(map(str, classes)):
        raise ValueError("E38 class order differs from requested canonical ordering")
    features = bundle["features"].transform([str(t) for t in texts])
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(features), dtype=np.float64)
    else:
        from scipy.special import softmax
        values = np.asarray(model.decision_function(features), dtype=np.float64)
        if values.ndim == 1:
            values = np.column_stack([-values, values])
        values = softmax(values, axis=1)
    return _align(values, list(map(str, model.classes_)), list(map(str, classes)))


def predict_e39(model_dir: str | Path, texts: list[str], classes: list[str], feature_factory) -> np.ndarray:
    """Run the saved E39 CatBoost tokenizer-fingerprint classifier."""
    from catboost import CatBoostClassifier
    from scipy.special import softmax

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "stylometry.json").read_text(encoding="utf-8"))
    if list(map(str, metadata["classes"])) != list(map(str, classes)):
        raise ValueError("E39 class order differs from requested canonical ordering")
    recipe = dict(metadata["feature_recipe"])
    if recipe.get("tokenizer_paths"):
        recipe["tokenizer_paths"] = {
            key: str((model_dir / path).resolve()) if not Path(path).is_absolute() else path
            for key, path in recipe["tokenizer_paths"].items()
        }
    feature_state = None
    state_name = metadata.get("feature_state_file")
    if state_name:
        feature_state = json.loads((model_dir / state_name).read_text(encoding="utf-8"))
    features = feature_factory.stylometry(texts, recipe, feature_state=feature_state)
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "model.cbm"))
    raw = np.asarray(model.predict(features, prediction_type="RawFormulaVal"), dtype=np.float64)
    if raw.ndim == 1:
        raw = np.column_stack([-raw, raw])
    aligned_logits = np.zeros((len(texts), len(classes)), dtype=np.float64)
    indexes = {str(label): i for i, label in enumerate(classes)}
    for j, label in enumerate(model.classes_):
        if str(label) not in indexes:
            raise ValueError(f"unknown class in E39 output: {label}")
        aligned_logits[:, indexes[str(label)]] = raw[:, j]
    values = softmax(aligned_logits, axis=1)
    return _align(values, list(classes), list(classes))


def blend_equal(base_scores: np.ndarray, auxiliary_probabilities: np.ndarray) -> np.ndarray:
    """Average row-normalized host scores with an already normalized E41 vector."""
    scores = np.asarray(base_scores, dtype=np.float64)
    aux = np.asarray(auxiliary_probabilities, dtype=np.float64)
    if scores.shape != aux.shape or scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("host and E41 score matrices must be finite and aligned")
    if not np.isfinite(aux).all() or (aux < 0).any() or not np.allclose(aux.sum(1), 1, atol=1e-5):
        raise ValueError("E41 probabilities are invalid")
    scores = np.maximum(scores, 0.0)
    den = scores.sum(axis=1, keepdims=True)
    if (den <= 0).any():
        raise ValueError("host score row has no nonnegative support")
    base = scores / den
    blended = 0.5 * base + 0.5 * aux
    if not np.isfinite(blended).all() or not np.allclose(blended.sum(1), 1, atol=1e-10):
        raise ValueError("blended probabilities are invalid")
    return blended
