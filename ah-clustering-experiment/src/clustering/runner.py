"""Validated clustering inputs and explicit scikit-learn baseline settings."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN, SpectralClustering
from threadpoolctl import threadpool_limits

from ..contracts import require


def validate_matrix(data, expected_count, precomputed=False, affinity=False):
    require(isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype.kind in "fiu",
            "Expected a numeric two-dimensional matrix")
    require(data.shape[0] == expected_count and expected_count >= 2 and data.shape[1] > 0,
            "Matrix shape differs from document count")
    require(np.isfinite(data).all(), "Matrix must contain only finite values")
    if precomputed:
        require(data.shape == (expected_count, expected_count), "Precomputed matrix must be square")
        require(np.allclose(data, data.T, rtol=0, atol=1e-6), "Precomputed matrix must be symmetric")
        require((data >= 0).all(), "Precomputed values must be nonnegative")
        if not affinity:
            require(np.allclose(np.diag(data), 0, rtol=0, atol=1e-7), "Distance diagonal must be zero")


def fit_clusters(data, algorithm, parameters, seed, *, precomputed=False):
    require(algorithm in {"kmeans", "agglomerative", "hdbscan", "spectral"}, "Unsupported clustering algorithm")
    require(type(seed) is int and seed >= 0 and isinstance(parameters, dict), "Invalid seed or parameters")
    validate_matrix(data, len(data), precomputed, affinity=algorithm == "spectral")
    params = dict(parameters)
    require("random_state" not in params, "Use the declared seed, not a parameter override")
    if algorithm != "hdbscan":
        count = params.get("n_clusters")
        require(type(count) is int and 2 <= count < len(data), "n_clusters must be between 2 and n_documents-1")
    if algorithm == "kmeans":
        require(not precomputed, "K-means is unavailable for precomputed similarities or distances")
        estimator = KMeans(random_state=seed, **params)
    elif algorithm == "agglomerative":
        require(params.get("linkage") in {"average", "complete", "single", "ward"}, "Declare linkage")
        require(params.get("metric") == ("precomputed" if precomputed else "euclidean"), "Incompatible agglomerative metric")
        require(not precomputed or params["linkage"] != "ward", "Ward needs Euclidean feature vectors")
        estimator = AgglomerativeClustering(**params)
    elif algorithm == "hdbscan":
        require(params.get("metric") == ("precomputed" if precomputed else "euclidean"), "Incompatible HDBSCAN metric")
        estimator = HDBSCAN(**params)
    else:
        require(params.get("affinity") == ("precomputed" if precomputed else "rbf"), "Incompatible spectral affinity")
        estimator = SpectralClustering(random_state=seed, **params)
    # Bound CPU use and make the numerical execution policy explicit.
    with threadpool_limits(limits=1):
        labels = estimator.fit_predict(np.array(data, dtype=np.float64, copy=True))
    require(labels.shape == (len(data),) and labels.dtype.kind in "iu" and (labels >= -1).all(), "Invalid cluster labels")
    return labels.astype(np.int64)
