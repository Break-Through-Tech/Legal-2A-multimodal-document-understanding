"""Step 7 independent mathematical oracles, frozen before candidate inspection."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate
from src.evaluation.projections import project


EXTERNAL = ("ari", "nmi", "ami", "homogeneity", "completeness", "v_measure", "purity")


class IndependentMetrics(unittest.TestCase):
    def setUp(self):
        self.ids = [1, 2, 3, 4]
        self.truth = ["A", "A", "B", "B"]
        self.labels = np.array([0, 0, 1, 1])
        self.features = np.array([[0.0], [2.0], [8.0], [10.0]])

    def run_example(self, **changes):
        args = dict(ids=self.ids, truth=self.truth, labels=self.labels, data=self.features)
        args.update(changes)
        return evaluate(**args)

    def scores(self, result, scope="including_noise"):
        return result["scores"][scope]["metrics"]

    def assert_scalar(self, result, metric, expected, scope="including_noise"):
        score = self.scores(result, scope)[metric]
        self.assertAlmostEqual(score["value"], expected, places=9)
        self.assertIsNone(score["reason"])

    def test_literal_euclidean_geometry_and_perfect_agreement(self):
        result = self.run_example()
        for name in EXTERNAL:
            self.assert_scalar(result, name, 1)
        self.assert_scalar(result, "silhouette", 47 / 63)
        self.assert_scalar(result, "davies_bouldin", 1 / 4)
        self.assert_scalar(result, "calinski_harabasz", 32)

    def test_crossed_contingency_literal_scores(self):
        result = self.run_example(truth=["A", "B", "A", "B"])
        for name in ("ari", "ami"):
            self.assert_scalar(result, name, -0.5)
        for name in ("nmi", "homogeneity", "completeness", "v_measure"):
            self.assert_scalar(result, name, 0)
        self.assert_scalar(result, "purity", 0.5)

    def test_cluster_label_permutation_invariance(self):
        original = self.run_example()
        permuted = self.run_example(labels=np.array([37, 37, 8, 8]))
        for scope in ("including_noise", "excluding_noise"):
            for name in self.scores(original, scope):
                self.assertEqual(self.scores(original, scope)[name], self.scores(permuted, scope)[name])

    def test_noise_coverage_and_no_noise_examples(self):
        result = self.run_example(ids=list(range(1, 7)), truth=list("AABBBA"), labels=np.array([0, 0, 1, 1, -1, -1]), data=np.array([[0.], [2.], [8.], [10.], [12.], [20.]]))
        self.assert_scalar(result, "purity", 5 / 6)
        for name in EXTERNAL:
            self.assert_scalar(result, name, 1, "excluding_noise")
        self.assertEqual(result["scores"]["excluding_noise"]["document_count"], 4)
        self.assertAlmostEqual(result["scores"]["excluding_noise"]["coverage"], 2 / 3)
        self.assertEqual(result["scores"]["including_noise"]["coverage"], 1)
        self.assertTrue(result["scores"]["including_noise"]["noise_policy"])
        noise = next(row for row in result["clusters"] if row["cluster_id"] == -1)
        self.assertTrue(noise["is_noise"])
        self.assertEqual(noise["representative_ids"], [])
        self.assertEqual(noise["outlier_ids"], [])

    def test_precomputed_has_same_silhouette_but_no_feature_scores(self):
        distance = np.abs(self.features - self.features.T)
        result = self.run_example(data=distance, precomputed=True)
        self.assert_scalar(result, "silhouette", 47 / 63)
        for name in ("davies_bouldin", "calinski_harabasz"):
            self.assertIsNone(self.scores(result)[name]["value"])
            self.assertTrue(self.scores(result)[name]["reason"])

    def test_medoid_nearest_farthest_and_id_ties(self):
        result = self.run_example(ids=[26, 1, 13, 17, 18], truth=list("AAABB"), labels=np.array([0, 0, 0, 1, 1]), data=np.array([[0.], [2.], [9.], [20.], [22.]]), examples_per_cluster=3)
        first = next(row for row in result["clusters"] if row["cluster_id"] == 0)
        self.assertEqual(first["representative_ids"], [1, 26, 13])
        self.assertEqual(first["outlier_ids"], [13, 26, 1])
        second = next(row for row in result["clusters"] if row["cluster_id"] == 1)
        self.assertEqual(second["representative_ids"], [17, 18])
        docs = {row["document_id"]: row for row in result["documents"]}
        self.assertEqual(set(docs), {26, 1, 13, 17, 18})
        for identity, distance in ((1, 0), (26, 2), (13, 7)):
            self.assertAlmostEqual(docs[identity]["distance_to_representative"], distance)

    def test_majority_ties_are_not_forced_mismatches(self):
        result = self.run_example(truth=["A", "B", "B", "B"])
        first = next(row for row in result["clusters"] if row["cluster_id"] == 0)
        self.assertEqual(set(first["dominant_labels"]), {"A", "B"})
        self.assertTrue(all(not row["majority_label_mismatch"] for row in result["documents"]))

    def test_unknown_is_an_ordinary_label(self):
        result = self.run_example(truth=["Unknown", "Unknown", "B", "B"])
        for name in EXTERNAL:
            self.assert_scalar(result, name, 1)

    def test_missing_truth_is_explicit(self):
        result = self.run_example(truth=[None] * 4)
        for name in EXTERNAL:
            self.assertIsNone(self.scores(result)[name]["value"])
            self.assertTrue(self.scores(result)[name]["reason"])
        self.assert_scalar(result, "silhouette", 47 / 63)

    def test_degenerate_geometry_is_explicit(self):
        for labels in (np.zeros(4, dtype=int), np.arange(4), np.full(4, -1)):
            with self.subTest(labels=labels.tolist()):
                result = self.run_example(labels=labels)
                for name in ("silhouette", "davies_bouldin", "calinski_harabasz"):
                    self.assertIsNone(self.scores(result)[name]["value"])
                    self.assertTrue(self.scores(result)[name]["reason"])
                if np.all(labels == -1):
                    self.assertEqual(result["scores"]["excluding_noise"]["document_count"], 0)
                    self.assertEqual(result["scores"]["excluding_noise"]["coverage"], 0)

    def test_coincident_centroids_make_davies_bouldin_undefined(self):
        # Two positive-scatter clusters both have centroid zero: DB divides by zero.
        result = self.run_example(data=np.array([[-1.], [1.], [-2.], [2.]]))
        score = self.scores(result)["davies_bouldin"]
        self.assertIsNone(score["value"])
        self.assertTrue(score["reason"])
        self.assert_scalar(result, "calinski_harabasz", 0)

    def test_zero_within_dispersion_has_undefined_ch_but_valid_zero_db(self):
        # Distinct centroids and zero scatter: DB=0; CH has zero denominator.
        result = self.run_example(data=np.array([[0.], [0.], [2.], [2.]]))
        score = self.scores(result)["calinski_harabasz"]
        self.assertIsNone(score["value"])
        self.assertTrue(score["reason"])
        self.assert_scalar(result, "davies_bouldin", 0)

    def test_all_identical_features_have_undefined_db_ch(self):
        result = self.run_example(data=np.zeros((4, 1)))
        for name in ("davies_bouldin", "calinski_harabasz"):
            self.assertIsNone(self.scores(result)[name]["value"])
            self.assertTrue(self.scores(result)[name]["reason"])

    def test_invalid_ids_labels_features_rejected(self):
        cases = [dict(ids=[1, 1, 3, 4]), dict(ids=[1]), dict(truth=["A"]), dict(labels=np.array([0, 0, -2, -2])), dict(labels=np.array([0., 0., 1.5, 1.5])), dict(data=np.array([[np.nan], [2.], [8.], [10.]]))]
        for change in cases:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                self.run_example(**change)

    def test_invalid_precomputed_rejected(self):
        good = np.abs(self.features - self.features.T)
        cases = []
        for index, value in (((0, 1), 99), ((0, 0), 1), ((0, 1), -1)):
            bad = good.copy()
            bad[index] = value
            cases.append(bad)
        for matrix in cases:
            with self.subTest(matrix=matrix.tolist()), self.assertRaises((ValueError, TypeError)):
                self.run_example(data=matrix, precomputed=True)

    def test_oracle_detects_wrong_metric_and_id_negative_controls(self):
        result = self.run_example()
        broken = copy.deepcopy(result)
        broken["scores"]["including_noise"]["metrics"]["silhouette"]["value"] = 0.75
        with self.assertRaises(AssertionError):
            self.assert_scalar(broken, "silhouette", 47 / 63)
        broken["documents"][0]["document_id"] = 999
        with self.assertRaises(AssertionError):
            self.assertEqual({row["document_id"] for row in broken["documents"]}, set(self.ids))


class IndependentProjections(unittest.TestCase):
    def setUp(self):
        self.features = np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.], [5., 0.], [5., 2.], [7., 0.], [7., 2.]])

    def test_pca_preserves_full_2d_geometry(self):
        coords, details = project(self.features, "pca", seed=42)
        self.assertEqual(coords.shape, (8, 2))
        source_squared = ((self.features[:, None] - self.features[None]) ** 2).sum(axis=2)
        projected_squared = ((coords[:, None] - coords[None]) ** 2).sum(axis=2)
        np.testing.assert_allclose(projected_squared, source_squared, atol=1e-6)
        self.assertTrue(details)

    def test_seeded_umap_features_and_precomputed(self):
        distance = np.linalg.norm(self.features[:, None] - self.features[None], axis=2)
        for precomputed, data in ((False, self.features), (True, distance)):
            with self.subTest(precomputed=precomputed):
                args = dict(precomputed=precomputed, seed=42, parameters={"n_neighbors": 3, "n_epochs": 50, "init": "random"})
                first, details = project(data, "umap", **args)
                second, _ = project(data, "umap", **args)
                self.assertEqual(first.shape, (8, 2))
                self.assertTrue(np.isfinite(first).all())
                np.testing.assert_array_equal(first, second)
                self.assertTrue(details)

    def test_precomputed_pca_and_bad_distances_rejected(self):
        distance = np.linalg.norm(self.features[:, None] - self.features[None], axis=2)
        with self.assertRaises((ValueError, TypeError)):
            project(distance, "pca", precomputed=True)
        distance[0, 1] += 1
        with self.assertRaises((ValueError, TypeError)):
            project(distance, "umap", precomputed=True)


if __name__ == "__main__":
    unittest.main()
