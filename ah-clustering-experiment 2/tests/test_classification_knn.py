"""Contract tests for top-k label voting on a fixed, hand-computed example."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True

from src.classification.knn import evaluate, macro_f1, topk_labels


class TopKVote(unittest.TestCase):
    def setUp(self):
        self.neighbours = [10, 11, 12]
        self.labels = {10: "B", 11: "A", 12: "A"}
        self.similarity = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)

    def test_top1_takes_the_single_closest_neighbour(self):
        self.assertEqual(topk_labels(self.similarity, [0], self.neighbours, self.labels, 1), ["B"])

    def test_top3_takes_the_plurality(self):
        self.assertEqual(topk_labels(self.similarity, [0], self.neighbours, self.labels, 3), ["A"])

    def test_two_way_tie_breaks_on_summed_similarity(self):
        # k=2 gives one B (0.9) and one A (0.8); the tie must go to B.
        self.assertEqual(topk_labels(self.similarity, [0], self.neighbours, self.labels, 2), ["B"])

    def test_equal_counts_and_equal_sums_break_on_label_name(self):
        neighbours = [20, 21]
        labels = {20: "Z", 21: "A"}
        similarity = np.array([[0.5, 0.5]], dtype=np.float32)
        self.assertEqual(topk_labels(similarity, [0], neighbours, labels, 2), ["A"])

    def test_self_match_is_rejected(self):
        with self.assertRaises(ValueError):
            topk_labels(self.similarity, [10], self.neighbours, self.labels, 1)

    def test_k_above_neighbour_count_is_rejected(self):
        with self.assertRaises(ValueError):
            topk_labels(self.similarity, [0], self.neighbours, self.labels, 4)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            topk_labels(np.zeros((1, 2), dtype=np.float32), [0], self.neighbours, self.labels, 1)

    def test_non_finite_similarity_is_rejected(self):
        bad = np.array([[np.nan, 0.8, 0.7]], dtype=np.float32)
        with self.assertRaises(ValueError):
            topk_labels(bad, [0], self.neighbours, self.labels, 1)

    def test_label_coverage_must_match(self):
        with self.assertRaises(ValueError):
            topk_labels(self.similarity, [0], self.neighbours, {10: "B", 11: "A"}, 1)


class MacroF1(unittest.TestCase):
    def test_hand_computed_two_label_case(self):
        score, per_label = macro_f1(["A", "A", "B", "B"], ["A", "A", "A", "B"])
        self.assertAlmostEqual(per_label["A"]["f1"], 0.8, places=6)
        self.assertAlmostEqual(per_label["B"]["f1"], 2 / 3, places=6)
        self.assertAlmostEqual(score, (0.8 + 2 / 3) / 2, places=6)

    def test_unpredicted_label_scores_zero(self):
        # A: precision 1/2, recall 1 -> 0.667. B: never predicted -> 0. macro 0.333.
        score, per_label = macro_f1(["A", "B"], ["A", "A"])
        self.assertAlmostEqual(per_label["A"]["f1"], 2 / 3, places=6)
        self.assertEqual(per_label["B"]["f1"], 0.0)
        self.assertAlmostEqual(score, 1 / 3, places=6)

    def test_misalignment_is_rejected(self):
        with self.assertRaises(ValueError):
            macro_f1(["A", "B"], ["A"])

    def test_explicit_labels_keep_absent_classes_in_the_average(self):
        score, per_label = macro_f1(["A", "A"], ["A", "A"], labels=["A", "B"])
        self.assertEqual(per_label["B"]["f1"], 0.0)
        self.assertEqual(per_label["B"]["support"], 0)
        self.assertAlmostEqual(score, 0.5, places=6)

    def test_explicit_labels_can_exclude_an_observed_class(self):
        score, per_label = macro_f1(["A", "B"], ["A", "B"], labels=["A"])
        self.assertEqual(set(per_label), {"A"})
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_empty_label_set_is_rejected(self):
        with self.assertRaises(ValueError):
            macro_f1(["A"], ["A"], labels=[])


class Evaluate(unittest.TestCase):
    def test_returns_one_row_per_k_and_matches_direct_calls(self):
        neighbours = [10, 11, 12]
        labels = {10: "B", 11: "A", 12: "A"}
        similarity = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.95]], dtype=np.float32)
        reference = {0: "A", 1: "A"}
        result = evaluate(similarity, [0, 1], reference, neighbours, labels, k_values=(1, 3))
        self.assertEqual([row["k"] for row in result["results"]], [1, 3])
        self.assertEqual(result["results"][0]["predictions"], ["B", "A"])
        self.assertEqual(result["results"][1]["predictions"], ["A", "A"])
        self.assertAlmostEqual(result["results"][0]["accuracy"], 0.5, places=6)
        self.assertAlmostEqual(result["results"][1]["accuracy"], 1.0, places=6)
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["neighbour_count"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
