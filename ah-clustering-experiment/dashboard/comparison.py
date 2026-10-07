"""Compare saved partitions after a one-to-one maximum-overlap alignment.

Noise never takes part in alignment. Ground-truth labels validate shared metadata
and are displayed only; they have no influence on the cluster mapping.
"""

from __future__ import annotations

import json
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


POLICY = {
    "alignment": "Hungarian one-to-one maximum document overlap",
    "population": "documents assigned to non-noise clusters in both runs",
    "noise_cluster": -1,
    "zero_overlap_matches": "unmatched",
    "unmatched_right_cluster": "aligned_right_cluster is null; disagreement when both non-noise",
    "agreement_denominator": "documents assigned to non-noise clusters in both runs",
    "comparison_coverage_denominator": "all cohort documents",
    "ties": "sorted cluster IDs and scipy deterministic assignment; equally optimal mappings may exist",
}
_IDENTITY = ("dataset_id", "cohort", "designation", "manifest_sha256", "dataset_identity_sha256")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integers(values, name: str, minimum: int) -> list[int]:
    result = list(values)
    _require(all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
                 and value >= minimum for value in result), f"Invalid {name}: expected integers >= {minimum}")
    return [int(value) for value in result]


def _validate(run: dict, side: str) -> tuple[pd.DataFrame, dict]:
    _require(isinstance(run, dict), f"Invalid {side} run")
    config = run.get("config")
    _require(isinstance(config, dict) and isinstance(config.get("cohort"), dict),
             f"Missing {side} cohort metadata")
    cohort = config["cohort"]
    _require(all(isinstance(cohort.get(key), str) and cohort[key] for key in _IDENTITY),
             f"Incomplete {side} cohort identity")
    _require(isinstance(cohort.get("document_ids"), list), f"Missing {side} cohort document IDs")
    expected = _integers(cohort["document_ids"], f"{side} cohort document IDs", 0)
    _require(bool(expected) and len(set(expected)) == len(expected), f"Duplicate or empty {side} cohort IDs")
    frame = run.get("documents")
    _require(isinstance(frame, pd.DataFrame), f"Missing {side} documents")
    _require(frame.columns.is_unique and {"document_id", "cluster_id", "label", "split"}.issubset(frame.columns),
             f"Missing or duplicate {side} document columns")
    ids = _integers(frame.document_id, f"{side} document IDs", 0)
    _require(len(set(ids)) == len(ids), f"Duplicate {side} document IDs")
    _require(set(ids) == set(expected), f"{side} document IDs differ from declared cohort")
    _integers(frame.cluster_id, f"{side} cluster IDs", -1)
    _require(not frame[["label", "split"]].isna().any().any(), f"Missing {side} label or split metadata")
    if cohort["cohort"] in {"train", "val", "validation", "test"}:
        _require((frame.split == cohort["cohort"]).all(), f"{side} split differs from declared cohort")
    summary = run.get("summary", {})
    _require(isinstance(summary, dict), f"Invalid {side} summary")
    if "document_count" in summary:
        _require(summary["document_count"] == len(ids), f"{side} summary document count differs")
    return frame.set_index("document_id").sort_index(), cohort


def compare_runs(left_run: dict, right_run: dict) -> dict:
    """Align completed saved runs by ID and report explicit noise/coverage states.

    Positive-overlap mappings are one-to-one. An unmatched right cluster has a
    null aligned ID, so a numeric label collision cannot manufacture agreement.
    """
    left, left_cohort = _validate(left_run, "left")
    right, right_cohort = _validate(right_run, "right")
    _require(all(left_cohort[key] == right_cohort[key] for key in _IDENTITY),
             "Incompatible dataset or cohort identity")
    _require(left.index.equals(right.index), "Incompatible document ID sets")
    for name in ("label", "split"):
        _require((left[name] == right[name]).all(), f"Conflicting {name} metadata for aligned document IDs")
    a, b = left.cluster_id.to_numpy(), right.cluster_id.to_numpy()
    both = (a != -1) & (b != -1)
    pairs = pd.DataFrame({"left_cluster": a[both], "right_cluster": b[both]})
    contingency = pairs.groupby(["left_cluster", "right_cluster"], sort=True).size().reset_index(name="count")
    left_ids, right_ids = sorted(set(a) - {-1}), sorted(set(b) - {-1})
    counts = np.zeros((len(left_ids), len(right_ids)), dtype=np.int64)
    left_position, right_position = ({value: index for index, value in enumerate(ids)}
                                     for ids in (left_ids, right_ids))
    for row in contingency.itertuples(index=False):
        counts[left_position[row.left_cluster], right_position[row.right_cluster]] = row.count
    mapped = {}
    if counts.size:
        rows, cols = linear_sum_assignment(counts, maximize=True)
        mapped = {right_ids[col]: (left_ids[row], int(counts[row, col]))
                  for row, col in zip(rows, cols) if counts[row, col] > 0}
    mapping = [{"right_cluster": int(value),
                "left_cluster": int(mapped[value][0]) if value in mapped else None,
                "overlap": mapped[value][1] if value in mapped else 0}
               for value in right_ids]
    aligned = pd.array([mapped[value][0] if value in mapped else None for value in b], dtype="Int64")
    agrees = np.asarray([bool(both[index] and value in mapped and a[index] == mapped[value][0])
                         for index, value in enumerate(b)])
    statuses = np.full(len(a), "disagreement", dtype=object)
    statuses[agrees] = "agreement"
    statuses[(a == -1) & (b != -1)] = "left_noise"
    statuses[(a != -1) & (b == -1)] = "right_noise"
    statuses[(a == -1) & (b == -1)] = "both_noise"
    documents = pd.DataFrame({"document_id": left.index.to_numpy(), "left_cluster": a,
                              "right_cluster": b, "aligned_right_cluster": aligned,
                              "status": statuses, "label": left.label.to_numpy()})
    compared, agreement = int(both.sum()), int(agrees.sum())
    return {
        "documents": documents,
        "contingency": contingency,
        "mapping": mapping,
        "unmatched_left_clusters": [int(value) for value in left_ids
                                    if value not in {match[0] for match in mapped.values()}],
        "unmatched_right_clusters": [int(value) for value in right_ids if value not in mapped],
        "summary": {
            "document_count": len(a), "compared_count": compared,
            "agreement_count": agreement, "disagreement_count": compared - agreement,
            "agreement_fraction": agreement / compared if compared else None,
            "both_noise_count": int(((a == -1) & (b == -1)).sum()),
            "left_noise_count": int(((a == -1) & (b != -1)).sum()),
            "right_noise_count": int(((a != -1) & (b == -1)).sum()),
            "comparison_coverage": compared / len(a),
        },
        "policy": json.dumps(POLICY, sort_keys=True),
    }
