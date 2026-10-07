"""Projection boundary checks and real PCA/UMAP numerical execution."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.projections import project


def distances(values):
    return np.sqrt(((values[:, None] - values[None, :]) ** 2).sum(axis=2))


class ProjectionTests(unittest.TestCase):
    def test_pca_literal_rectangle_preserves_distances_and_variance(self):
        # Rank-two rectangle in three dimensions. Centered axis variances 4 and 1
        # imply variance fractions .8 and .2 independently of PCA axis signs.
        points = np.array([[-2., -1., 7.], [-2., 1., 7.], [2., -1., 7.], [2., 1., 7.]])
        coords, details = project(points, "pca")
        np.testing.assert_allclose(distances(coords), distances(points), atol=1e-12)
        np.testing.assert_allclose(details["explained_variance_ratio"], [.8, .2], atol=1e-12)
        np.testing.assert_allclose(coords.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_array_equal(project(points, "pca")[0], coords)
        self.assertTrue(details["visualization_only"])
        self.assertEqual(details["parameters"]["svd_solver"], "full")
        json.dumps(details, allow_nan=False)
        changed = coords.copy()
        changed[0, 0] += 1
        self.assertFalse(np.allclose(distances(changed), distances(points)))

    def test_pca_rejects_distance_matrix_and_degenerate_features(self):
        for data, precomputed in [(np.eye(4), True), (np.ones((4, 3)), False), (np.ones((4, 1)), False)]:
            with self.subTest(precomputed=precomputed), self.assertRaises(ValueError):
                project(data, "pca", precomputed=precomputed)

    def test_invalid_input_boundaries(self):
        for data in [np.ones(3), np.ones((1, 3)), np.empty((4, 0)), np.ones((4, 2), dtype=bool),
                     np.array([[1., 2], [np.nan, 1]]), np.array([[1., 2], [np.inf, 1]])]:
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                project(data, "pca")
        for kwargs in [{"seed": -1}, {"seed": True}, {"seed": 2**32}, {"parameters": []},
                       {"precomputed": 1}, {"parameters": {"random_state": 1}},
                       {"parameters": {"whiten": True}}, {"parameters": {"n_components": True}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                project(np.eye(4), "pca", **kwargs)
        with self.assertRaises(ValueError):
            project(np.eye(4), "tsne")

    def test_precomputed_negative_controls(self):
        valid = np.ones((20, 20)) - np.eye(20)
        asymmetric, negative, diagonal = (valid.copy() for _ in range(3))
        asymmetric[1, 2] = 0
        negative[1, 2] = negative[2, 1] = -1
        diagonal[1, 1] = 1
        for matrix in [np.ones((20, 3)), asymmetric, negative, diagonal]:
            with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                project(matrix, "umap", precomputed=True)

    def test_umap_parameter_negative_controls(self):
        for parameters in [{"n_neighbors": 20}, {"n_neighbors": 1}, {"n_neighbors": True},
                           {"min_dist": -1}, {"min_dist": np.nan}, {"min_dist": 1.1},
                           {"n_epochs": 0}, {"n_jobs": 2}, {"random_state": 41},
                           {"metric": "cosine"}, {"init": "spectral"}, {"target_weight": .9}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                project(np.eye(20), "umap", parameters=parameters)

    def test_actual_fixedseed_umap_features_and_nonmetric_matrix_repeat(self):
        points = np.random.default_rng(133).normal(size=(24, 4))
        dissimilarity = distances(points)
        # Deliberately violate a triangle inequality; it is a dissimilarity, not
        # represented or tested as Euclidean geometry.
        dissimilarity[0, 1] = dissimilarity[1, 0] = 100
        self.assertGreater(dissimilarity[0, 1], dissimilarity[0, 2] + dissimilarity[2, 1])
        for data, precomputed in [(points, False), (dissimilarity, True)]:
            with self.subTest(precomputed=precomputed):
                original = data.copy()
                params = {"n_neighbors": 5, "n_epochs": 30}
                first, details = project(data, "umap", precomputed=precomputed, parameters=params)
                second, _ = project(data, "umap", precomputed=precomputed, parameters=params)
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(data, original)
                self.assertEqual(first.shape, (24, 2))
                self.assertTrue(np.isfinite(first).all())
                self.assertEqual(details["parameters"]["metric"], "precomputed" if precomputed else "euclidean")
                self.assertEqual(details["parameters"]["n_jobs"], 1)
                json.dumps(details, allow_nan=False)

    def test_pca_does_not_import_umap_or_encoders(self):
        code = "import sys, numpy as np; from src.evaluation.projections import project; project(np.eye(4), 'pca'); assert not any(x in sys.modules for x in ('umap', 'transformers', 'tensorflow', 'src.embeddings.models'))"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
