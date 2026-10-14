"""Out-of-scope detector implementing Nearest-Training-Doc and Class-Centroid similarity."""
from __future__ import annotations

import numbers
from typing import Literal

import numpy as np


class OutOfScopeDetector:
    """Detects out-of-scope documents by evaluating similarity to known in-scope legal documents.

    Supported methods:
      - 'nearest': Compares query embedding to each individual legal training document
                   and takes the maximum cosine similarity.
      - 'centroid': Computes the normalized mean embedding (centroid) for each legal class
                    and takes the maximum cosine similarity to any class centroid.
    """

    def __init__(self, method: Literal["nearest", "centroid"] = "nearest"):
        if method not in {"nearest", "centroid"}:
            raise ValueError(f"Unknown method '{method}'; expected 'nearest' or 'centroid'")
        self.method = method
        self.train_vectors: np.ndarray | None = None
        self.train_labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.centroid_classes: list[str] | None = None
        self.threshold: float | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"Expected 2D numpy array [n_samples, n_features], got shape {getattr(X, 'shape', None)}")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if (norms <= 0).any() or not np.isfinite(norms).all():
            raise ValueError("All vectors must have positive finite L2 norm")
        return (X / norms).astype(np.float32)

    def fit(self, X_train_legal: np.ndarray, y_train_legal_classes: np.ndarray | list[str] | None = None) -> OutOfScopeDetector:
        """Fit detector on known in-scope legal training documents.

        Args:
            X_train_legal: 2D array of legal document embeddings [n_train_legal, dim].
            y_train_legal_classes: Class labels for each training document (required for 'centroid' method).
        """
        X_norm = self._normalize(X_train_legal)
        self.train_vectors = X_norm

        if self.method == "centroid":
            if y_train_legal_classes is None:
                raise ValueError("y_train_legal_classes is required when method='centroid'")
            labels = np.asarray(y_train_legal_classes)
            if len(labels) != len(X_norm):
                raise ValueError(f"Length mismatch: {len(labels)} labels for {len(X_norm)} vectors")

            classes = sorted(list(set(labels)))
            centroids = []
            for c in classes:
                mask = (labels == c)
                c_mean = np.mean(X_norm[mask], axis=0, dtype=np.float64)
                c_norm = np.linalg.norm(c_mean)
                if c_norm <= 0 or not np.isfinite(c_norm):
                    raise ValueError(f"Failed to compute valid centroid for class '{c}'")
                centroids.append((c_mean / c_norm).astype(np.float32))

            self.centroids = np.stack(centroids, axis=0)
            self.centroid_classes = classes

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute legal similarity score for queries (higher = more similar to legal documents).

        Returns:
            1D array of shape [n_samples] with cosine similarities in [-1.0, 1.0].
        """
        if self.train_vectors is None:
            raise RuntimeError("Detector must be fit before calling score()")

        X_norm = self._normalize(X)

        if self.method == "nearest":
            # [n_samples, n_train_legal]
            sims = np.dot(X_norm, self.train_vectors.T)
            return np.max(sims, axis=1)

        elif self.method == "centroid":
            # [n_samples, n_classes]
            sims = np.dot(X_norm, self.centroids.T)
            return np.max(sims, axis=1)

        raise RuntimeError(f"Unsupported method: {self.method}")

    def tune_threshold(
        self,
        X_val: np.ndarray,
        is_out_of_scope_val: np.ndarray,
        policy: Literal["max_false_rejection", "target_recall", "f1"] = "max_false_rejection",
        target_value: float = 0.05,
    ) -> float:
        """Tune rejection threshold tau on validation data.

        Decision rule:
            Reject as out-of-scope if score < tau
            Accept as legal if score >= tau

        Policies:
            - 'max_false_rejection': Sets tau to the alpha percentile of legal validation scores,
              ensuring at most target_value fraction of legal documents are rejected (e.g. target_value=0.05).
            - 'target_recall': Sets tau to ensure at least target_value fraction of out-of-scope documents
              are rejected (e.g. target_value=0.95).
            - 'f1': Maximizes the F1 score for detecting out-of-scope documents.
        """
        scores = self.score(X_val)
        oos_mask = np.asarray(is_out_of_scope_val, dtype=bool)
        if len(scores) != len(oos_mask):
            raise ValueError("Length mismatch between X_val and is_out_of_scope_val")
        if not oos_mask.any():
            raise ValueError("Validation set contains no out-of-scope documents")
        if oos_mask.all():
            raise ValueError("Validation set contains no in-scope legal documents")

        legal_scores = scores[~oos_mask]
        oos_scores = scores[oos_mask]

        if policy == "max_false_rejection":
            # Target alpha fraction of legal documents mistakenly rejected
            # A legal document is rejected if score < tau.
            # So tau is the (target_value * 100)-th percentile of legal scores.
            self.threshold = float(np.percentile(legal_scores, target_value * 100))

        elif policy == "target_recall":
            # Target beta fraction of OOS documents rejected (score < tau).
            # So tau is the (target_value * 100)-th percentile of OOS scores.
            self.threshold = float(np.percentile(oos_scores, target_value * 100))

        elif policy == "f1":
            # Sweep all candidate thresholds across score spectrum
            candidate_thresholds = np.unique(scores)
            best_f1, best_tau = -1.0, float(candidate_thresholds[0])
            for tau in candidate_thresholds:
                pred_oos = scores < tau
                tp = np.sum(pred_oos & oos_mask)
                fp = np.sum(pred_oos & ~oos_mask)
                fn = np.sum(~pred_oos & oos_mask)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = float(tau)
            self.threshold = best_tau

        else:
            raise ValueError(f"Unknown threshold tuning policy: '{policy}'")

        return self.threshold

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predict whether documents are Out-of-Scope (Rejected).

        Returns:
            Boolean array: True if Rejected (out-of-scope), False if Accepted (in-scope legal).
        """
        tau = threshold if threshold is not None else self.threshold
        if tau is None:
            raise RuntimeError("Threshold is not set; call tune_threshold() or pass threshold explicitly")
        scores = self.score(X)
        return scores < tau
