"""Top-k label voting over a saved similarity matrix.

Ranking uses similarity only; labels enter at the vote.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..contracts import document_ids, require


DEFINITION = {
    "name": "topk_label_vote_v1",
    "k_values": [1, 3, 5],
    "ranking": "descending similarity, restricted to the neighbour cohort",
    "vote": "plurality of the k neighbour labels",
    "tie_break": "highest summed similarity among tied labels, then lexicographically smallest label",
    "self_match": "query and neighbour ID sets must be disjoint",
    "abstain": "never; every query receives one of its neighbours' labels",
    "uncovered_labels": "labels absent from the neighbour cohort can never be predicted",
}


def _aligned(labels, ids) -> list[str]:
    require(isinstance(labels, dict), "Labels must map document ID to label name")
    require(set(labels) == set(ids), "Label coverage must equal the neighbour ID set exactly")
    values = [labels[i] for i in ids]
    require(all(isinstance(v, str) and v for v in values), "Labels must be nonempty strings")
    return values


def topk_labels(similarity: np.ndarray, query_ids, neighbour_ids,
                neighbour_labels: dict, k: int) -> list[str]:
    """Predict one label per query from its k most similar labelled documents."""
    queries = document_ids(query_ids)
    neighbours = document_ids(neighbour_ids)
    require(not (set(queries) & set(neighbours)),
            "Query and neighbour ID sets must be disjoint; a document cannot vote for itself")
    require(isinstance(similarity, np.ndarray) and similarity.shape == (len(queries), len(neighbours)),
            "Similarity must be shaped query-by-neighbour")
    require(similarity.dtype.kind == "f" and np.isfinite(similarity).all(),
            "Similarity must contain finite floating point values")
    require(type(k) is int and 1 <= k <= len(neighbours),
            f"k must be an integer between 1 and the {len(neighbours)} available neighbours")
    values = _aligned(neighbour_labels, neighbours)
    predictions = []
    for row in similarity:
        # Stable sort keeps equal scores in declared neighbour order.
        order = np.argsort(-row, kind="stable")[:k]
        counts, totals = defaultdict(int), defaultdict(float)
        for index in order:
            counts[values[index]] += 1
            totals[values[index]] += float(row[index])
        ranked = sorted(counts, key=lambda name: (-counts[name], -totals[name], name))
        predictions.append(ranked[0])
    return predictions


def macro_f1(reference, predicted, labels=None) -> tuple[float, dict]:
    """Macro F1 over the observed label set, or over `labels` when supplied.

    Resampling can drop a class from a replicate. Passing the full label set keeps the
    average over the same classes in every replicate, so replicates stay comparable.
    """
    truth, guess = list(reference), list(predicted)
    require(len(truth) == len(guess) and bool(truth), "Reference and prediction must be nonempty and aligned")
    names = sorted(set(labels) if labels is not None else set(truth) | set(guess))
    require(bool(names), "At least one label is required")
    per_label, scores = {}, []
    for name in names:
        hits = sum(1 for t, g in zip(truth, guess) if t == name and g == name)
        predicted_n = sum(1 for g in guess if g == name)
        actual_n = sum(1 for t in truth if t == name)
        precision = hits / predicted_n if predicted_n else 0.0
        recall = hits / actual_n if actual_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {"f1": f1, "precision": precision, "recall": recall, "support": actual_n}
        scores.append(f1)
    return sum(scores) / len(scores), per_label


def evaluate(similarity: np.ndarray, query_ids, reference_labels: dict, neighbour_ids,
             neighbour_labels: dict, k_values=(1, 3, 5)) -> dict:
    """Score each k on the same query set."""
    queries = document_ids(query_ids)
    require(isinstance(reference_labels, dict) and set(reference_labels) == set(queries),
            "Reference label coverage must equal the query ID set exactly")
    ks = [int(k) for k in k_values]
    require(bool(ks) and len(ks) == len(set(ks)) and min(ks) >= 1, "k values must be unique positive integers")
    truth = [reference_labels[i] for i in queries]
    results = []
    for k in ks:
        predicted = topk_labels(similarity, queries, neighbour_ids, neighbour_labels, k)
        score, per_label = macro_f1(truth, predicted)
        results.append({"k": k, "macro_f1": score,
                        "accuracy": sum(t == p for t, p in zip(truth, predicted)) / len(truth),
                        "per_label": per_label, "predictions": predicted})
    return {"definition": DEFINITION, "query_count": len(queries),
            "neighbour_count": len(document_ids(neighbour_ids)), "results": results}
