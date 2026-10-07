"""Label-free, deterministic two-dimensional visualizations of saved features."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from ..contracts import require


def _matrix(data, precomputed):
    require(isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype.kind in "fiu",
            "Projection input must be a numeric two-dimensional array")
    require(data.shape[0] >= 2 and data.shape[1] >= 1, "Projection input is too small")
    require(np.isfinite(data).all(), "Projection input must contain only finite values")
    if precomputed:
        require(data.shape[0] == data.shape[1], "Precomputed dissimilarities must be square")
        require((data >= 0).all(), "Precomputed dissimilarities must be nonnegative")
        require(np.allclose(data, data.T, rtol=0, atol=1e-6), "Precomputed dissimilarities must be symmetric")
        require(np.allclose(np.diag(data), 0, rtol=0, atol=1e-7), "Precomputed diagonal must be zero")


def project(data, method, *, precomputed=False, seed=42, parameters=None):
    """Return coordinates in input-row order and a JSON-compatible construction record.

    The caller supplies already transformed feature vectors or validated nonmetric
    dissimilarities, and owns document-ID alignment. No labels, clustering, scaling,
    or embedding inference are used here. Coordinates are for visualization only.
    """
    require(isinstance(method, str) and method in {"pca", "umap"}, "Unsupported projection method")
    require(type(precomputed) is bool, "precomputed must be boolean")
    require(type(seed) is int and 0 <= seed <= 2**32 - 1, "Invalid projection seed")
    require(parameters is None or isinstance(parameters, dict), "Projection parameters must be a mapping")
    _matrix(data, precomputed)
    params = {} if parameters is None else dict(parameters)
    details = {"method": method, "seed": seed, "input_count": len(data),
               "input_dimensions": data.shape[1], "visualization_only": True,
               "input_geometry": "precomputed_nonmetric_dissimilarity" if precomputed else "euclidean_features",
               "row_order": "unchanged_from_input", "additional_normalization": "none"}
    if method == "pca":
        require(not precomputed, "PCA requires fixed feature vectors, not precomputed dissimilarities")
        require(data.shape[1] >= 2, "PCA requires at least two feature dimensions")
        allowed = {"n_components": 2, "svd_solver": "full", "whiten": False}
        require(set(params) <= set(allowed), "Unsupported PCA parameter")
        require(all(type(value) is type(allowed[key]) and value == allowed[key]
                    for key, value in params.items()), "PCA construction parameters are fixed")
        params = allowed
        require(float(np.var(data.astype(np.float64), axis=0).sum()) > 0,
                "PCA is undefined for features with zero total variance")
        estimator = PCA(**params)
        details["construction"] = "Centered feature PCA with exact full SVD, two components, no whitening"
    else:
        fixed = {"n_components": 2, "metric": "precomputed" if precomputed else "euclidean",
                 "init": "random", "random_state": seed, "transform_seed": seed, "n_jobs": 1}
        variable = {"n_neighbors": 15, "min_dist": 0.1, "n_epochs": 500}
        require(set(params) <= set(fixed) | set(variable), "Unsupported UMAP parameter")
        require(all(type(params[key]) is type(value) and params[key] == value
                    for key, value in fixed.items() if key in params), "UMAP construction parameters are fixed")
        params = {**variable, **params, **fixed}
        require(type(params["n_neighbors"]) is int and 2 <= params["n_neighbors"] < len(data),
                "UMAP n_neighbors must be between 2 and n_documents-1")
        require(type(params["n_epochs"]) is int and params["n_epochs"] >= 10,
                "UMAP n_epochs must be an integer of at least 10")
        require(isinstance(params["min_dist"], numbers.Real) and not isinstance(params["min_dist"], bool)
                and np.isfinite(params["min_dist"]) and 0 <= params["min_dist"] <= 1,
                "UMAP min_dist must be finite and between zero and the default spread of one")
        params["min_dist"] = float(params["min_dist"])
        # Keep PCA usable without UMAP/Numba or optional UMAP dependencies installed.
        from umap.umap_ import UMAP
        estimator = UMAP(**params)
        details["construction"] = ("UMAP on supplied nonmetric dissimilarity; no triangle-inequality assumption claimed"
                                   if precomputed else "UMAP on supplied feature vectors with Euclidean distance")
    with threadpool_limits(limits=1):
        coordinates = estimator.fit_transform(np.array(data, dtype=np.float64, copy=True))
    require(coordinates.shape == (len(data), 2) and np.isfinite(coordinates).all(),
            "Projection must return two finite coordinates for every input row")
    details["parameters"] = params
    if method == "pca":
        ratios = estimator.explained_variance_ratio_
        require(np.isfinite(ratios).all(), "PCA explained variance must be finite")
        details["explained_variance_ratio"] = ratios.tolist()
    return coordinates, details
