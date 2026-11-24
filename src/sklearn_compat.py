"""Narrow compatibility repairs for serialized sklearn inference artifacts."""
from __future__ import annotations


def patch_serialized_sklearn_estimators(root):
    """Restore known deprecated sklearn attributes and smoke-test classifiers.

    scikit-learn 1.9 stopped serializing ``LogisticRegression.multi_class``;
    1.6's ``predict_proba`` still reads it. For a fitted LogisticRegression,
    ``auto`` is the legacy default and preserves the trained decision function.
    The walk supports estimator bundles nested in dicts, sequences and sklearn
    meta-estimators. It returns a JSON-safe audit record.
    """
    visited = set()
    repaired = []
    smoke_tests = []

    def visit(value, path):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple, set)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return

        module = type(value).__module__
        name = type(value).__name__
        if module.startswith("sklearn") and name == "LogisticRegression":
            classes = getattr(value, "classes_", ())
            if len(classes) >= 2 and not hasattr(value, "multi_class"):
                value.multi_class = "auto"
                repaired.append({"path": path, "estimator": name,
                                 "attribute": "multi_class", "value": "auto"})
            if hasattr(value, "predict_proba") and hasattr(value, "n_features_in_"):
                import numpy as np

                probabilities = value.predict_proba(np.zeros((1, int(value.n_features_in_)), dtype=np.float32))
                if probabilities.shape != (1, len(classes)) or not np.isfinite(probabilities).all():
                    raise ValueError(f"invalid LogisticRegression predict_proba result at {path}: {probabilities.shape}")
                smoke_tests.append({"path": path, "estimator": name,
                                    "predict_proba": "PASS"})

        if module.startswith("sklearn"):
            for key, child in vars(value).items():
                if isinstance(child, (dict, list, tuple, set)) or type(child).__module__.startswith("sklearn"):
                    visit(child, f"{path}.{key}")

    visit(root, "root")
    return {"patched_attributes": repaired, "smoke_tests": smoke_tests}
