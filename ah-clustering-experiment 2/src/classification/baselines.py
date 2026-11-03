"""Pooled single-vector baseline arm.

Token selection is shared with the retained arm, so the only differences between
the two arms are the reduction and the decision rule.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid

from ..clustering.similarity import select_vectors
from ..contracts import document_ids, require


DEFINITION = {
    "name": "pooled_single_vector_v1",
    "token_selection": "attention_mask & image_mask (identical to the retained-vector arm)",
    "reduction": "arithmetic mean over selected tokens in float64",
    "normalization": "L2 after reduction",
    "output_shape": "[dimension]; one vector per document",
    "training": "none for the reduction itself; the classifier is fit on training vectors only",
}


def pool_document(document: dict) -> np.ndarray:
    """Reduce one document's selected tokens to a single L2-normalized vector."""
    tokens = select_vectors(document)
    require(tokens.ndim == 2 and tokens.shape[0] > 0, "Expected at least one selected token")
    mean = tokens.mean(axis=0, dtype=np.float64)
    require(np.isfinite(mean).all(), "Pooled vector is not finite")
    norm = float(np.linalg.norm(mean))
    require(norm > 0, "Pooled vector has zero norm")
    return (mean / norm).astype(np.float32)


def pooled_matrix(documents: dict, ids) -> np.ndarray:
    """Stack pooled vectors in the given ID order."""
    order = document_ids(ids)
    require(isinstance(documents, dict) and set(order) <= set(documents),
            "Every requested document ID must be present in the embedding cache")
    matrix = np.stack([pool_document(documents[document_id]) for document_id in order])
    require(matrix.ndim == 2 and np.isfinite(matrix).all(), "Invalid pooled matrix")
    return matrix


class PooledClassifier:
    """Fit a linear or centroid decision rule on pooled document vectors."""

    METHODS = ("logistic_regression", "nearest_centroid")

    def __init__(self, method: str = "logistic_regression", C: float = 1.0, random_state: int = 42):
        require(method in self.METHODS, f"Unknown method {method!r}; choose from {self.METHODS}")
        require(type(C) in (int, float) and C > 0, "C must be a positive number")
        self.method = method
        self.C = float(C)
        self.random_state = random_state
        self.model = None
        self.classes_ = None

    def fit(self, features: np.ndarray, labels) -> "PooledClassifier":
        require(isinstance(features, np.ndarray) and features.ndim == 2, "Expected a feature matrix")
        require(np.isfinite(features).all(), "Features must be finite")
        values = np.asarray([str(label) for label in labels])
        require(len(values) == len(features), "Feature and label counts differ")
        self.classes_ = np.unique(values)
        require(len(self.classes_) >= 2, "At least two classes are required")
        if self.method == "logistic_regression":
            self.model = LogisticRegression(C=self.C, max_iter=5000, random_state=self.random_state)
        else:
            self.model = NearestCentroid()
        self.model.fit(features, values)
        return self

    def predict(self, features: np.ndarray) -> list[str]:
        require(self.model is not None, "Fit the classifier before predicting")
        require(isinstance(features, np.ndarray) and features.ndim == 2, "Expected a feature matrix")
        require(np.isfinite(features).all(), "Features must be finite")
        return [str(value) for value in self.model.predict(features)]


def retained_bytes(manifest: dict) -> int:
    """Bytes of the retained token vectors in a saved manifest."""
    files = manifest["files"]
    return sum(int(item["bytes"]) for name, item in files.items() if name.endswith("-vectors"))


def pooled_bytes(document_count: int, dimension: int, dtype_itemsize: int = 4) -> int:
    """Bytes needed if each document kept only one pooled vector."""
    require(document_count > 0 and dimension > 0, "Positive document count and dimension required")
    return int(document_count) * int(dimension) * int(dtype_itemsize)
