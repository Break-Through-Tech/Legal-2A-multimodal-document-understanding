"""Evaluate saved partitions and select label-independent document examples."""

from __future__ import annotations

from collections import Counter
import numbers

import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from ..contracts import document_ids, require


AGREEMENT = {
    "ari": metrics.adjusted_rand_score,
    "nmi": metrics.normalized_mutual_info_score,
    "ami": metrics.adjusted_mutual_info_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "v_measure": metrics.v_measure_score,
}


def _value(value=None, reason=None):
    if value is not None:
        value = float(value)
        if not np.isfinite(value):
            return {"value": None, "reason": "Calculation returned a nonfinite value."}
    return {"value": value, "reason": reason}


def _truth(values, count):
    if values is None:
        return [None] * count
    values = list(values)
    require(len(values) == count, "Truth length differs from document IDs")
    result = []
    for value in values:
        if value is None or (isinstance(value, numbers.Real) and np.isnan(value)):
            result.append(None)
        else:
            require(isinstance(value, str) and bool(value.strip()),
                    "Truth labels must be nonempty strings or missing (None/NaN)")
            result.append(value)
    return result


def _distribution(truth):
    counts = Counter(value for value in truth if value is not None)
    if not counts:
        return {}, [], None
    maximum = max(counts.values())
    return dict(sorted(counts.items())), sorted(k for k, v in counts.items() if v == maximum), maximum / sum(counts.values())


def _scope_scores(mask, truth, labels, features, distances, precomputed):
    indices = np.flatnonzero(mask)
    scoped_labels = labels[indices]
    labeled = np.array([index for index in indices if truth[index] is not None], dtype=np.int64)
    scores = {}
    for name, function in AGREEMENT.items():
        scores[name] = (_value(function([truth[i] for i in labeled], labels[labeled]))
                        if len(labeled) >= 2 else _value(reason="Fewer than two labeled documents in this scope."))
    if len(labeled):
        majority_count = sum(max(Counter(truth[i] for i in labeled if labels[i] == cluster).values())
                             for cluster in np.unique(labels[labeled]))
        scores["purity"] = _value(majority_count / len(labeled))
    else:
        scores["purity"] = _value(reason="No labeled documents in this scope.")

    group_count = len(np.unique(scoped_labels))
    condition = 2 <= group_count < len(indices)
    undefined = "Requires between 2 and n_documents - 1 groups in this scope."
    scores["silhouette"] = (_value(metrics.silhouette_score(distances[np.ix_(indices, indices)],
                                                          scoped_labels, metric="precomputed"))
                            if condition else _value(reason=undefined))
    coincident_centroids = False
    within_sum_squares = None
    if condition and not precomputed:
        groups = [features[indices[scoped_labels == group]] for group in np.unique(scoped_labels)]
        centroids = np.stack([group.mean(axis=0) for group in groups])
        coincident_centroids = any(np.array_equal(centroids[i], centroids[j])
                                   for i in range(len(centroids)) for j in range(i))
        within_sum_squares = sum(float(np.square(group - centroid).sum())
                                 for group, centroid in zip(groups, centroids))
    for name, function in (("davies_bouldin", metrics.davies_bouldin_score),
                           ("calinski_harabasz", metrics.calinski_harabasz_score)):
        if precomputed:
            scores[name] = _value(reason="Unsupported for precomputed dissimilarity: requires original Euclidean features.")
        elif not condition:
            scores[name] = _value(reason=undefined)
        elif name == "davies_bouldin" and coincident_centroids:
            scores[name] = _value(reason="Undefined: two cluster centroids coincide, giving a zero separation denominator.")
        elif name == "calinski_harabasz" and within_sum_squares == 0:
            scores[name] = _value(reason="Undefined: within-cluster sum of squares is zero, giving a zero denominator.")
        else:
            scores[name] = _value(function(features[indices], scoped_labels))
    return {"document_count": len(indices), "coverage": len(indices) / len(labels),
            "labeled_document_count": len(labeled),
            "labeled_coverage": len(labeled) / len(indices) if len(indices) else 0.0,
            "metrics": scores}


def evaluate(ids, truth, labels, data, *, precomputed=False, examples_per_cluster=3):
    """Return JSON-compatible metrics/examples in input ID order, without fitting.

    ``data`` is the exact fixed feature representation used to cluster, or a
    validated symmetric nonnegative dissimilarity with zero diagonal. Reference
    labels may be missing, and never affect representative/outlier selection.
    """
    ids = document_ids(ids)
    require(type(precomputed) is bool, "precomputed must be boolean")
    require(type(examples_per_cluster) is int and examples_per_cluster > 0,
            "examples_per_cluster must be a positive integer")
    labels = np.asarray(labels)
    require(labels.shape == (len(ids),) and labels.dtype.kind in "iu" and (labels >= -1).all(),
            "Cluster labels must be a one-dimensional integer array >= -1 aligned with IDs")
    truth = _truth(truth, len(ids))
    require(isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype.kind in "fiu",
            "Expected a numeric two-dimensional matrix")
    require(data.shape[0] == len(ids) and data.shape[1] > 0, "Matrix shape differs from document IDs")
    require(np.isfinite(data).all(), "Matrix must contain only finite values")
    features = np.array(data, dtype=np.float64, copy=True)
    if precomputed:
        require(features.shape == (len(ids), len(ids)), "Precomputed matrix must be square")
        require((features >= 0).all(), "Dissimilarity must be nonnegative")
        require(np.allclose(features, features.T, rtol=0, atol=1e-6), "Dissimilarity must be symmetric")
        require(np.allclose(np.diag(features), 0, rtol=0, atol=1e-7), "Dissimilarity diagonal must be zero")
        distances = (features + features.T) / 2
    else:
        with threadpool_limits(limits=1):
            distances = pairwise_distances(features, metric="euclidean")
    require(np.isfinite(distances).all(), "Pairwise distances must remain finite")
    np.fill_diagonal(distances, 0)
    scores = {}
    with threadpool_limits(limits=1):
        for name, mask, policy in (
            ("including_noise", np.ones(len(ids), dtype=bool), "Noise label -1 participates as one group."),
            ("excluding_noise", labels != -1, "Noise label -1 is excluded from all metrics in this scope."),
        ):
            scores[name] = _scope_scores(mask, truth, labels, features, distances, precomputed)
            scores[name]["noise_policy"] = policy

    documents = [{"document_id": document_id, "cluster_id": int(cluster), "label": value,
                  "is_noise": bool(cluster == -1), "majority_label_mismatch": False,
                  "representative_rank": None, "outlier_rank": None, "distance_to_representative": None}
                 for document_id, cluster, value in zip(ids, labels, truth)]
    clusters = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        counts, dominant, purity = _distribution([truth[i] for i in members])
        representatives, outliers = [], []
        if cluster != -1:
            within = distances[np.ix_(members, members)]
            medoid_local = min(range(len(members)), key=lambda j: (float(within[j].sum()), ids[members[j]]))
            medoid = int(members[medoid_local])
            nearest = sorted((int(i) for i in members if i != medoid), key=lambda i: (distances[i, medoid], ids[i]))
            representatives = ([medoid] + nearest)[:examples_per_cluster]
            outliers = sorted((int(i) for i in members), key=lambda i: (-distances[i, medoid], ids[i]))[:examples_per_cluster]
            for i in members:
                documents[i]["distance_to_representative"] = float(distances[i, medoid])
                documents[i]["majority_label_mismatch"] = bool(truth[i] is not None and truth[i] not in dominant)
            for rank, i in enumerate(representatives, 1):
                documents[i]["representative_rank"] = rank
            for rank, i in enumerate(outliers, 1):
                documents[i]["outlier_rank"] = rank
        clusters.append({"cluster_id": int(cluster), "size": len(members), "is_noise": bool(cluster == -1),
                         "labeled_document_count": sum(counts.values()), "purity": purity,
                         "dominant_labels": dominant, "label_counts": counts,
                         "representative_ids": [ids[i] for i in representatives],
                         "outlier_ids": [ids[i] for i in outliers]})
    return {"scores": scores, "clusters": clusters, "documents": documents, "policy": {
        "distance": "validated precomputed nonmetric dissimilarity" if precomputed else "Euclidean on supplied clustering features",
        "precomputed_tolerance": {"symmetry_atol": 1e-6, "diagonal_atol": 1e-7,
                                  "roundoff_handling": "Average transpose and zero diagonal after validation."} if precomputed else None,
        "representatives": "Minimum within-cluster sum-distance medoid, then nearest to medoid; ties use ascending document ID.",
        "outliers": "Farthest from medoid; ties use ascending document ID. Review candidates, not proven errors.",
        "examples_per_cluster": examples_per_cluster,
        "example_overlap": "Representatives and outliers may overlap, especially for small clusters.",
        "noise_examples": "No representatives, outliers, or majority-label mismatch for noise.",
        "dominant_labels": "All labels tied for the largest count; all are accepted for mismatch flags.",
        "truth": "Supervised metrics and purity use labeled documents; internal metrics use all documents in each scope.",
        "coverage": "Scope document count / input count; labeled_coverage is labeled count / scope count (zero when empty).",
        "normalization": "No normalization is applied; caller supplies the exact clustering representation.",
        "silhouette": "Full-data Euclidean pairwise distances for fixed features, supplied dissimilarity otherwise; singleton-group samples score zero.",
        "agreement_averaging": "NMI and AMI use scikit-learn arithmetic averaging; V-measure beta=1.",
        "degenerate_geometry": "Davies-Bouldin is unavailable for coincident centroids; Calinski-Harabasz is unavailable for zero within-cluster dispersion.",
    }}
