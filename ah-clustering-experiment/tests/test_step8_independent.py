"""Step 8 requirement-derived oracles, specified before candidate inspection."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.comparison import compare_runs


def run(labels, ids=None):
    ids = list(range(1, len(labels) + 1)) if ids is None else ids
    return {
        "documents": pd.DataFrame({"document_id": ids, "cluster_id": labels,
                                   "label": ["A"] * len(ids), "split": ["train"] * len(ids)}),
        "config": {"cohort": {"cohort": "train", "designation": "initial_training_cohort", "dataset_id": "fixture-dataset",
                              "document_ids": list(ids), "manifest_sha256": "a" * 64,
                              "dataset_identity_sha256": "b" * 64}},
        "summary": {"document_count": len(ids)},
    }


class IndependentComparison(unittest.TestCase):
    def test_cluster_renumbering_is_perfect(self):
        result = compare_runs(run([0, 0, 1, 1]), run([98, 98, 7, 7]))
        self.assertEqual(result["summary"]["compared_count"], 4)
        self.assertEqual(result["summary"]["agreement_fraction"], 1)
        self.assertEqual(result["summary"]["comparison_coverage"], 1)

    def test_document_order_is_not_alignment(self):
        left = run([0, 0, 1, 1])
        right = run([7, 98, 7, 98], ids=[4, 1, 3, 2])
        result = compare_runs(left, right)
        self.assertEqual(result["summary"]["agreement_fraction"], 1)
        self.assertEqual(set(result["documents"]["document_id"]), {1, 2, 3, 4})

    def test_global_assignment_beats_greedy(self):
        # Contingency [[4, 3], [3, 0]]: optimum is off-diagonal 6, not greedy 4.
        left = run([0] * 7 + [1] * 3)
        right = run([8] * 4 + [9] * 3 + [8] * 3)
        result = compare_runs(left, right)
        self.assertEqual(result["summary"]["compared_count"], 10)
        self.assertAlmostEqual(result["summary"]["agreement_fraction"], 0.6)

    def test_unequal_group_counts_leave_unmatched_groups(self):
        result = compare_runs(run([0, 0, 0, 1, 1, 1]), run([7, 7, 8, 9, 9, 9]))
        self.assertAlmostEqual(result["summary"]["agreement_fraction"], 5 / 6)

    def test_noise_coverage_and_denominator(self):
        left = run([0, 0, 1, 1, -1, -1, 2, -1])
        right = run([8, 8, 9, 9, -1, 10, -1, -1])
        result = compare_runs(left, right)
        self.assertEqual(result["summary"]["compared_count"], 4)
        self.assertEqual(result["summary"]["agreement_fraction"], 1)
        self.assertEqual(result["summary"]["comparison_coverage"], 0.5)
        self.assertEqual(result["documents"]["status"].tolist(),
                         ["agreement"] * 4 + ["both_noise", "left_noise", "right_noise", "both_noise"])
        self.assertTrue(all(row["right_cluster"] != -1 and row["left_cluster"] != -1
                            for row in result["mapping"]))

    def test_noise_cannot_manufacture_cluster_agreement(self):
        result = compare_runs(run([-1, -1, 0, 0]), run([7, 7, -1, -1]))
        self.assertEqual(result["summary"]["compared_count"], 0)
        self.assertEqual(result["summary"]["comparison_coverage"], 0)
        self.assertIsNone(result["summary"]["agreement_fraction"])

    def test_all_noise_is_not_perfect_agreement(self):
        result = compare_runs(run([-1, -1]), run([-1, -1]))
        self.assertIsNone(result["summary"]["agreement_fraction"])
        self.assertEqual(result["summary"]["compared_count"], 0)

    def test_one_assignment_change_is_detected(self):
        result = compare_runs(run([0, 0, 1, 1]), run([8, 9, 9, 9]))
        self.assertEqual(result["summary"]["agreement_fraction"], 0.75)

    def test_mismatched_id_sets_rejected(self):
        with self.assertRaises(ValueError):
            compare_runs(run([0, 1], [1, 2]), run([8, 9], [1, 3]))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            compare_runs(run([0, 1], [1, 1]), run([8, 9], [1, 2]))

    def test_cohort_provenance_mismatch_rejected(self):
        for field, bad in (("cohort", "test"), ("dataset_id", "other"),
                           ("manifest_sha256", "c" * 64), ("dataset_identity_sha256", "d" * 64)):
            with self.subTest(field=field):
                right = run([8, 9])
                right["config"]["cohort"][field] = bad
                with self.assertRaises(ValueError):
                    compare_runs(run([0, 1]), right)

    def test_declared_cohort_ids_must_match_actual_documents(self):
        right = run([8, 9])
        right["config"]["cohort"]["document_ids"] = [1, 99]
        with self.assertRaises(ValueError):
            compare_runs(run([0, 1]), right)

    def test_inputs_remain_unchanged(self):
        left, right = run([0, 0, 1]), run([7, 7, 8])
        before = copy.deepcopy((left, right))
        compare_runs(left, right)
        for actual, expected in zip((left, right), before):
            pd.testing.assert_frame_equal(actual["documents"], expected["documents"])
            self.assertEqual(actual["config"], expected["config"])


if __name__ == "__main__":
    unittest.main()
