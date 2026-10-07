"""Builder checks from declared four-point geometry and partition invariants."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import evaluate


class EvaluationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.ids = [40, 20, 30, 10]
        self.features = np.array([[0.], [2.], [10.], [12.]])
        self.truth = ["a", "a", "b", "b"]
        self.labels = [5, 5, 9, 9]

    def result(self, **kwargs):
        args = dict(ids=self.ids, truth=self.truth, labels=self.labels, data=self.features)
        args.update(kwargs)
        return evaluate(**args)

    def test_hand_geometry_and_perfect_agreement(self):
        scores = self.result()["scores"]["including_noise"]["metrics"]
        for name in ("ari", "nmi", "ami", "homogeneity", "completeness", "v_measure", "purity"):
            self.assertAlmostEqual(scores[name]["value"], 1.)
            self.assertIsNone(scores[name]["reason"])
        self.assertAlmostEqual(scores["silhouette"]["value"], 1 - 1/11 - 1/9)
        self.assertAlmostEqual(scores["davies_bouldin"]["value"], .2)
        self.assertAlmostEqual(scores["calinski_harabasz"]["value"], 50.)

    def test_partition_permutation_and_perturbation_control(self):
        original = self.result()["scores"]
        self.assertEqual(original, self.result(labels=[12, 12, 0, 0])["scores"])
        broken = self.result(labels=[0, 1, 0, 1])["scores"]["including_noise"]["metrics"]
        self.assertAlmostEqual(broken["ari"]["value"], -.5)
        self.assertNotEqual(broken["ari"], original["including_noise"]["metrics"]["ari"])

    def test_medoid_example_ranks_tie_ids_and_truth_independence(self):
        result = self.result(examples_per_cluster=2)
        self.assertEqual(result["clusters"][0]["representative_ids"], [20, 40])
        self.assertEqual(result["clusters"][0]["outlier_ids"], [40, 20])
        self.assertEqual(result["clusters"][1]["representative_ids"], [10, 30])
        self.assertEqual(result["clusters"][1]["outlier_ids"], [30, 10])
        self.assertEqual([d["document_id"] for d in result["documents"]], self.ids)
        self.assertEqual([d["distance_to_representative"] for d in result["documents"]], [2., 0., 2., 0.])
        changed = self.result(truth=["z", "x", "x", "z"], examples_per_cluster=2)
        for first, second in zip(result["clusters"], changed["clusters"]):
            for key in ("representative_ids", "outlier_ids"):
                self.assertEqual(first[key], second[key])

    def test_tied_majorities_and_real_mismatch(self):
        tied = self.result(truth=["a", "b", "b", "b"])
        self.assertEqual(tied["clusters"][0]["dominant_labels"], ["a", "b"])
        self.assertEqual(tied["scores"]["including_noise"]["metrics"]["purity"]["value"], .75)
        self.assertFalse(any(d["majority_label_mismatch"] for d in tied["documents"]))
        unequal = self.result(truth=["a", "a", "b", "b"], labels=[0, 0, 0, 1])
        self.assertEqual([d["majority_label_mismatch"] for d in unequal["documents"]], [False, False, True, False])

    def test_precomputed_matches_silhouette_not_feature_scores(self):
        distances = np.abs(self.features - self.features.T)
        result = self.result(data=distances, precomputed=True)
        metrics = result["scores"]["including_noise"]["metrics"]
        self.assertAlmostEqual(metrics["silhouette"]["value"], 1 - 1/11 - 1/9)
        for name in ("davies_bouldin", "calinski_harabasz"):
            self.assertIsNone(metrics[name]["value"])
            self.assertIn("requires original Euclidean features", metrics[name]["reason"])
        self.assertEqual(result["clusters"], self.result()["clusters"])

    def test_missing_truth_preserves_internal_metrics_and_coverage(self):
        result = self.result(truth=["a", None, "b", np.nan])
        scope = result["scores"]["including_noise"]
        self.assertEqual(scope["labeled_document_count"], 2)
        self.assertEqual(scope["labeled_coverage"], .5)
        self.assertEqual(scope["coverage"], 1.)
        self.assertEqual(scope["metrics"]["ari"]["value"], 1.)
        absent = self.result(truth=None)["scores"]["including_noise"]["metrics"]
        self.assertIsNone(absent["ari"]["value"])
        self.assertIsNotNone(absent["ari"]["reason"])
        self.assertAlmostEqual(absent["silhouette"]["value"], 1 - 1/11 - 1/9)
        self.assertIsNone(absent["purity"]["value"])

    def test_noise_scopes_and_all_noise(self):
        result = self.result(labels=[0, 0, -1, -1])
        self.assertEqual(result["scores"]["excluding_noise"]["coverage"], .5)
        self.assertEqual(result["scores"]["including_noise"]["metrics"]["ari"]["value"], 1.)
        self.assertIsNone(result["scores"]["excluding_noise"]["metrics"]["silhouette"]["value"])
        self.assertEqual(result["clusters"][0]["representative_ids"], [])
        for document in result["documents"][2:]:
            self.assertIsNone(document["distance_to_representative"])
            self.assertFalse(document["majority_label_mismatch"])
        all_noise = self.result(labels=[-1] * 4)
        excluded = all_noise["scores"]["excluding_noise"]
        self.assertEqual(excluded["document_count"], 0)
        self.assertEqual(excluded["coverage"], 0.)
        self.assertTrue(all(value["value"] is None and value["reason"] for value in excluded["metrics"].values()))

    def test_singleton_input_all_singletons_one_group_and_singleton_sample(self):
        single = evaluate([3], ["a"], [0], np.array([[2.]]))
        self.assertEqual(single["clusters"][0]["representative_ids"], [3])
        self.assertIsNone(single["scores"]["including_noise"]["metrics"]["silhouette"]["value"])
        for labels in ([0, 0, 0, 0], [0, 1, 2, 3]):
            scope = self.result(labels=labels)["scores"]["including_noise"]["metrics"]
            for name in ("silhouette", "davies_bouldin", "calinski_harabasz"):
                self.assertIsNone(scope[name]["value"])
        with_singleton = evaluate([0, 1, 2], ["a", "a", "b"], [0, 0, 1], np.array([[0.], [2.], [10.]]))
        self.assertAlmostEqual(with_singleton["scores"]["including_noise"]["metrics"]["silhouette"]["value"], (.8 + .75)/3)

    def test_invalid_identity_shape_and_labels_negative_controls(self):
        for changes in ({"ids": [1, 1, 2, 3]}, {"ids": [False, 1, 2, 3]},
                        {"labels": [0., 0., 1., 1.]}, {"labels": [0, 0, -2, 1]},
                        {"labels": [[0, 0, 1, 1]]}, {"truth": ["a"]},
                        {"truth": ["a", "", "b", "b"]}, {"data": np.ones((3, 2))},
                        {"data": np.full((4, 2), np.nan)}, {"examples_per_cluster": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.result(**changes)

    def test_coincident_centroids_and_zero_within_cluster_dispersion(self):
        constant = self.result(data=np.zeros((4, 2)))["scores"]["including_noise"]["metrics"]
        self.assertEqual(constant["silhouette"]["value"], 0.)
        self.assertIsNone(constant["davies_bouldin"]["value"])
        self.assertIn("centroids coincide", constant["davies_bouldin"]["reason"])
        self.assertIsNone(constant["calinski_harabasz"]["value"])
        self.assertIn("sum of squares is zero", constant["calinski_harabasz"]["reason"])
        zero_scatter = self.result(data=np.array([[0.], [0.], [10.], [10.]]))["scores"]["including_noise"]["metrics"]
        self.assertEqual(zero_scatter["davies_bouldin"]["value"], 0.)
        self.assertIsNone(zero_scatter["davies_bouldin"]["reason"])
        self.assertIsNone(zero_scatter["calinski_harabasz"]["value"])
        nonzero_scatter = self.result(data=np.array([[-1.], [1.], [-2.], [2.]]))["scores"]["including_noise"]["metrics"]
        self.assertIsNone(nonzero_scatter["davies_bouldin"]["value"])
        self.assertEqual(nonzero_scatter["calinski_harabasz"]["value"], 0.)

    def test_invalid_dissimilarity_negative_controls(self):
        distances = np.abs(self.features - self.features.T)
        asymmetric = distances.copy()
        asymmetric[0, 1] += .1
        negative = distances.copy()
        negative[0, 1] = negative[1, 0] = -.1
        diagonal = distances.copy()
        diagonal[0, 0] = .1
        for data in (asymmetric, negative, diagonal, np.ones((4, 2))):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.result(data=data, precomputed=True)


if __name__ == "__main__":
    unittest.main()
