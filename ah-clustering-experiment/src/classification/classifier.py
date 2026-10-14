"""Multiclass classifier and unified two-stage intake router for legal documents."""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid

from ..detection.detector import OutOfScopeDetector


class LegalDocumentClassifier:
    """Multiclass classifier over in-scope legal document types."""

    def __init__(
        self,
        method: Literal["logistic_regression", "nearest_centroid"] = "logistic_regression",
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.method = method
        self.C = C
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.model = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if (norms <= 0).any() or not np.isfinite(norms).all():
            raise ValueError("All vectors must have positive finite L2 norm")
        return (X / norms).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray | list[str]) -> LegalDocumentClassifier:
        """Trains multiclass classifier on in-scope legal documents."""
        X_norm = self._normalize(X)
        y_arr = np.asarray(y)

        if len(X_norm) != len(y_arr):
            raise ValueError(f"Mismatch: {len(X_norm)} vectors vs {len(y_arr)} labels")

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Multiclass classification requires at least 2 distinct classes")

        if self.method == "logistic_regression":
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
            self.model.fit(X_norm, y_arr)

        elif self.method == "nearest_centroid":
            self.model = NearestCentroid(metric="euclidean")
            self.model.fit(X_norm, y_arr)

        else:
            raise ValueError(f"Unknown classifier method: {self.method}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the legal class for each document embedding."""
        if self.model is None:
            raise RuntimeError("Classifier must be fit before calling predict()")
        X_norm = self._normalize(X)
        return self.model.predict(X_norm)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities (or softmax-normalized centroid similarities)."""
        if self.model is None:
            raise RuntimeError("Classifier must be fit before calling predict_proba()")
        X_norm = self._normalize(X)

        if self.method == "logistic_regression":
            return self.model.predict_proba(X_norm)
        elif self.method == "nearest_centroid":
            # Cosine similarity to centroids converted to softmax probabilities
            centroids = self.model.centroids_
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norm_centroids = centroids / c_norms
            sims = np.dot(X_norm, norm_centroids.T)
            # Softmax
            exp_sims = np.exp(sims * 5.0)  # temperature scaling
            return exp_sims / np.sum(exp_sims, axis=1, keepdims=True)


class TwoStageIntakeRouter:
    """Complete Two-Stage Legal Intake Router.

    Stage 1: Out-of-Scope Detector checks if document is legal or non-legal.
             If score < threshold, it is flagged as 'OUT_OF_SCOPE' (rejected for human review).
    Stage 2: If accepted as legal, LegalDocumentClassifier routes it to its exact
             legal document type (e.g. COMMERCIAL_LEASE_AGREEMENT, PATENT).
    """

    REJECT_LABEL = "OUT_OF_SCOPE"

    def __init__(
        self,
        detector: OutOfScopeDetector,
        classifier: LegalDocumentClassifier,
    ):
        self.detector = detector
        self.classifier = classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Routes documents: returns specific legal class or 'OUT_OF_SCOPE'."""
        is_rejected = self.detector.predict(X)
        class_preds = self.classifier.predict(X)

        final_preds = np.array(class_preds, dtype=object)
        final_preds[is_rejected] = self.REJECT_LABEL
        return final_preds

    def predict_with_details(self, X: np.ndarray) -> list[dict]:
        """Provides full routing decision details including detector similarity and confidence."""
        is_rejected = self.detector.predict(X)
        scores = self.detector.score(X)
        class_preds = self.classifier.predict(X)
        probas = self.classifier.predict_proba(X)
        max_probas = np.max(probas, axis=1)

        results = []
        for i in range(len(X)):
            status = "REJECTED" if is_rejected[i] else "ACCEPTED"
            assigned_label = self.REJECT_LABEL if is_rejected[i] else class_preds[i]
            results.append({
                "status": status,
                "assigned_label": str(assigned_label),
                "predicted_legal_class": str(class_preds[i]),
                "classifier_confidence": float(max_probas[i]),
                "detector_similarity": float(scores[i]),
                "detector_threshold": float(self.detector.threshold),
            })
        return results
