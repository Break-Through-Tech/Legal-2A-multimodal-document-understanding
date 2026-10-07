"""Small, hand-specified partitions independently determine expected overlap."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.comparison import compare_runs


def run(labels):
    ids = list(range(len(labels)))
    return {
        "config": {"cohort": {"dataset_id": "dataset-fixture", "cohort": "train",
            "designation": "initial_training_partition", "manifest_sha256": "manifest-sha",
            "dataset_identity_sha256": "dataset-sha", "document_ids": ids}},
        "documents": pd.DataFrame({"document_id": ids, "cluster_id": labels,
            "label": ["A"] * len(ids), "split": ["train"] * len(ids)}),
        "summary": {"document_count": len(ids)},
    }


def set_column(item, name, values):
    frame = item["documents"]
    frame[name] = values


class ComparisonTests(unittest.TestCase):
    def test_cluster_numbering_and_row_order_do_not_change_partition(self):
        left, right = run([0, 0, 1, 1, 2, 2]), run([9, 9, 7, 7, 8, 8])
        right["documents"] = right["documents"].iloc[[5, 0, 2, 4, 1, 3]]
        right["config"]["cohort"]["document_ids"].reverse()
        result = compare_runs(left, right)
        self.assertEqual(result["summary"]["agreement_fraction"], 1)
        self.assertEqual(result["documents"].aligned_right_cluster.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(result["documents"].document_id.tolist(), list(range(6)))

    def test_unequal_clusters_remain_unmatched_despite_numeric_collision(self):
        result = compare_runs(run([0, 0, 0, 1, 1, 1]), run([4, 4, 0, 5, 5, 5]))
        self.assertEqual(result["summary"]["agreement_count"], 5)
        self.assertEqual(result["summary"]["disagreement_count"], 1)
        self.assertEqual(result["mapping"], [
            {"right_cluster": 0, "left_cluster": None, "overlap": 0},
            {"right_cluster": 4, "left_cluster": 0, "overlap": 2},
            {"right_cluster": 5, "left_cluster": 1, "overlap": 3}])
        self.assertTrue(pd.isna(result["documents"].iloc[2].aligned_right_cluster))
        self.assertEqual(result["documents"].iloc[2].status, "disagreement")

    def test_altered_partition_cannot_be_erased_by_alignment(self):
        result = compare_runs(run([0, 0, 0, 1, 1, 1]), run([7, 7, 8, 8, 8, 7]))
        self.assertEqual(result["summary"]["agreement_count"], 4)
        self.assertEqual(result["summary"]["disagreement_count"], 2)
        self.assertEqual(sorted(result["contingency"]["count"].tolist()), [1, 1, 2, 2])

    def test_hungarian_uses_global_optimum(self):
        # Counts [[3, 2], [2, 0]]: greedily pairing the largest cell gets 3;
        # the unique globally optimal mapping gets 2+2=4.
        result = compare_runs(run([0] * 5 + [1] * 2), run([8] * 3 + [9] * 2 + [8] * 2))
        self.assertEqual(result["summary"]["agreement_count"], 4)
        self.assertEqual(result["mapping"], [
            {"right_cluster": 8, "left_cluster": 1, "overlap": 2},
            {"right_cluster": 9, "left_cluster": 0, "overlap": 2}])

    def test_noise_is_separate_and_excluded_from_coverage(self):
        result = compare_runs(run([-1, -1, 1, 1, 2]), run([-1, 8, -1, 8, 9]))
        self.assertEqual(result["documents"].status.tolist(),
                         ["both_noise", "left_noise", "right_noise", "agreement", "agreement"])
        self.assertEqual(result["summary"], {
            "document_count": 5, "compared_count": 2, "agreement_count": 2,
            "disagreement_count": 0, "agreement_fraction": 1,
            "both_noise_count": 1, "left_noise_count": 1, "right_noise_count": 1,
            "comparison_coverage": 0.4})
        self.assertTrue(result["documents"].loc[[0, 2], "aligned_right_cluster"].isna().all())

    def test_all_noise_and_disjoint_assigned_documents_are_undefined(self):
        for left, right in [([-1, -1], [-1, -1]), ([0, -1], [-1, 9])]:
            with self.subTest(left=left):
                result = compare_runs(run(left), run(right))
                self.assertIsNone(result["summary"]["agreement_fraction"])
                self.assertEqual(result["summary"]["comparison_coverage"], 0)
                self.assertTrue(result["contingency"].empty)
                self.assertTrue(all(item["left_cluster"] is None for item in result["mapping"]))

    def test_label_values_do_not_influence_mapping(self):
        left, right = run([0, 0, 1, 1]), run([8, 8, 9, 9])
        expected = compare_runs(left, right)["mapping"]
        for current in (left, right):
            current["documents"]["label"] = ["B", "A", "B", "A"]
        self.assertEqual(compare_runs(left, right)["mapping"], expected)

    def test_input_frames_are_not_mutated(self):
        left, right = run([0, 0, 1]), run([7, 7, 8])
        snapshot = deepcopy(left)
        compare_runs(left, right)
        pd.testing.assert_frame_equal(left["documents"], snapshot["documents"])
        self.assertEqual(left["config"], snapshot["config"])

    def test_conflicting_dataset_cohort_metadata_rejected(self):
        for key in ("dataset_id", "cohort", "designation", "manifest_sha256", "dataset_identity_sha256"):
            with self.subTest(key=key):
                left, right = run([0, 1]), run([7, 8])
                right["config"]["cohort"][key] = "different"
                with self.assertRaises(ValueError):
                    compare_runs(left, right)

    def test_duplicate_missing_or_changed_document_ids_rejected(self):
        mutations = [lambda item: set_column(item, "document_id", [0, 0]),
                     lambda item: item.__setitem__("documents", item["documents"].iloc[:1]),
                     lambda item: set_column(item, "document_id", [0, 3]),
                     lambda item: item["config"]["cohort"].__setitem__("document_ids", [0, 0])]
        for mutate in mutations:
            right = run([7, 8])
            mutate(right)
            with self.assertRaises(ValueError):
                compare_runs(run([0, 1]), right)

    def test_same_size_different_declared_document_set_rejected(self):
        right = run([7, 8])
        right["documents"]["document_id"] = [2, 3]
        right["config"]["cohort"]["document_ids"] = [2, 3]
        with self.assertRaisesRegex(ValueError, "document ID sets"):
            compare_runs(run([0, 1]), right)

    def test_corrupt_labels_splits_summary_and_cluster_values_rejected(self):
        mutations = [lambda item: set_column(item, "label", ["B", "A"]),
                     lambda item: set_column(item, "split", ["test", "train"]),
                     lambda item: item["summary"].__setitem__("document_count", 99),
                     lambda item: set_column(item, "cluster_id", [-2, 0]),
                     lambda item: set_column(item, "cluster_id", [1.0, 2.0]),
                     lambda item: set_column(item, "cluster_id", [np.nan, 0]),
                     lambda item: set_column(item, "document_id", [False, True]),
                     lambda item: item["config"]["cohort"].pop("manifest_sha256")]
        for mutate in mutations:
            right = run([7, 8])
            mutate(right)
            with self.assertRaises(ValueError):
                compare_runs(run([0, 1]), right)


if __name__ == "__main__":
    unittest.main()
