"""Independent step 6 oracle: geometry, literal MaxSim values, and ID invariants.

Scenarios were fixed from IMPLEMENTATION_PLAN step 6 and the interface contract
before reading the clustering implementation. No encoder or storage mock is used.
"""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def separated_clouds():
    # Two identical 4x4 grids, separated by over 7 units; grid spacing is .1.
    left = np.array([[-4 + x / 10, y / 10] for x in range(4) for y in range(4)], dtype=np.float64)
    return np.concatenate([left, left + np.array([8., 0.])])


def assert_partition(testcase, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertTrue(np.issubdtype(actual.dtype, np.integer))
    # Compare membership directly; arbitrary cluster-number permutations are valid.
    np.testing.assert_array_equal(actual[:, None] == actual[None, :], expected[:, None] == expected[None, :])


class IndependentClusteringGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.clustering.runner import fit_clusters, validate_matrix
        cls.fit = staticmethod(fit_clusters)
        cls.validate = staticmethod(validate_matrix)

    def test_all_cls_algorithms_recover_known_separated_groups_and_repeat(self):
        data = separated_clouds()
        expected = [0] * 16 + [1] * 16
        configurations = {
            "kmeans": {"n_clusters": 2, "n_init": 10},
            "agglomerative": {"n_clusters": 2, "linkage": "average", "metric": "euclidean"},
            "hdbscan": {"min_cluster_size": 5, "min_samples": 2, "metric": "euclidean"},
            "spectral": {"n_clusters": 2, "affinity": "rbf", "gamma": 1.0, "n_init": 10},
        }
        for name, parameters in configurations.items():
            with self.subTest(algorithm=name):
                first = self.fit(data, name, parameters, 42)
                second = self.fit(data, name, parameters, 42)
                assert_partition(self, first, expected)
                assert_partition(self, second, first)

    def test_precomputed_compatible_algorithms_recover_same_geometry(self):
        data = separated_clouds()
        distances = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))
        affinity = np.exp(-distances ** 2)
        for name, parameters, matrix in [
            ("agglomerative", {"n_clusters": 2, "linkage": "average", "metric": "precomputed"}, distances),
            ("hdbscan", {"min_cluster_size": 5, "min_samples": 2, "metric": "precomputed"}, distances),
            ("spectral", {"n_clusters": 2, "affinity": "precomputed", "n_init": 10}, affinity),
        ]:
            with self.subTest(algorithm=name):
                actual = self.fit(matrix, name, parameters, 42, precomputed=True)
                assert_partition(self, actual, [0] * 16 + [1] * 16)

    def test_matrix_shape_count_and_nonfinite_boundaries(self):
        self.validate(separated_clouds(), 32)
        for invalid, expected in [
            (np.ones((4, 2)), 5), (np.ones(4), 4), (np.ones((4, 2, 1)), 4),
            (np.empty((4, 0)), 4), (np.array([[1., np.nan], [2., 3.]]), 2),
            (np.array([[1., np.inf], [2., 3.]]), 2),
        ]:
            with self.subTest(shape=invalid.shape, count=expected), self.assertRaises(ValueError):
                self.validate(invalid, expected)

    def test_precomputed_distance_and_affinity_semantics(self):
        distance = np.array([[0., .3, .7], [.3, 0., .5], [.7, .5, 0.]])
        self.validate(distance, 3, precomputed=True)
        self.validate(np.exp(-distance), 3, precomputed=True, affinity=True)
        invalids = [np.ones((3, 2)), np.array([[0., 1.], [.5, 0.]]),
                    np.array([[0., -1.], [-1., 0.]]), np.eye(2)]
        for invalid in invalids:
            with self.subTest(matrix=invalid.tolist()), self.assertRaises(ValueError):
                self.validate(invalid, len(invalid), precomputed=True)
        with self.assertRaises(ValueError):
            self.validate(np.array([[1., -.1], [-.1, 1.]]), 2, precomputed=True, affinity=True)

    def test_incompatible_precomputed_algorithms_are_rejected(self):
        distance = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
        for name, parameters in [
            ("kmeans", {"n_clusters": 2}),
            ("agglomerative", {"n_clusters": 2, "linkage": "ward", "metric": "precomputed"}),
        ]:
            with self.subTest(algorithm=name), self.assertRaises(ValueError):
                self.fit(distance, name, parameters, 42, precomputed=True)

    def test_partition_oracle_rejects_wrong_assignment_negative_control(self):
        expected = np.array([0, 0, 1, 1])
        assert_partition(self, np.array([7, 7, 9, 9]), expected)
        with self.assertRaises(AssertionError):
            assert_partition(self, np.array([7, 9, 7, 9]), expected)

    def test_saved_feature_clustering_never_imports_encoder_module(self):
        code = (
            "import sys,numpy as np; "
            "from src.clustering.runner import fit_clusters; "
            "fit_clusters(np.array([[0.,0.],[0.,.1],[9.,9.],[9.,9.1]]),"
            "'kmeans',{'n_clusters':2,'n_init':10},42); "
            "assert 'src.embeddings.models' not in sys.modules, 'Encoder module imported during clustering'"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class IndependentMultiVectorSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.clustering.similarity import select_vectors, compute_similarity, validate_matrices
        cls.select = staticmethod(select_vectors)
        cls.compute = staticmethod(compute_similarity)
        cls.validate = staticmethod(validate_matrices)

    def fixture(self):
        return [np.array([[1., 0.]], dtype=np.float32),
                np.array([[1., 0.], [0., 1.]], dtype=np.float32),
                np.array([[-1., 0.]], dtype=np.float32)]

    def test_hand_calculated_directionality_symmetry_scale_and_transforms(self):
        actual = self.compute(self.fixture(), device="cpu")
        repeated = self.compute(self.fixture(), device="cpu")
        expected = {
            "directed": [[1., 1., -1.], [.5, 1., -.5], [-1., 0., 1.]],
            "similarity": [[1., .75, -1.], [.75, 1., -.25], [-1., -.25, 1.]],
            "distance": [[0., .25, 2.], [.25, 0., 1.25], [2., 1.25, 0.]],
            "affinity": [[1., .875, 0.], [.875, 1., .375], [0., .375, 1.]],
        }
        for name, golden in expected.items():
            np.testing.assert_allclose(actual[name], golden, rtol=0, atol=1e-6, err_msg=name)
            np.testing.assert_array_equal(actual[name], repeated[name], err_msg="repeat " + name)
        self.validate(actual, [19, 4, 22], [19, 4, 22])
        self.assertGreater(actual["distance"][0, 2], actual["distance"][0, 1] + actual["distance"][1, 2])

    def test_token_normalization_makes_positive_rescaling_invariant(self):
        original = self.fixture()
        changed = [original[0] * 7, original[1] * np.array([[2.], [13.]], dtype=np.float32), original[2] * .25]
        first, second = self.compute(original, device="cpu"), self.compute(changed, device="cpu")
        for name in ("directed", "similarity", "distance", "affinity"):
            np.testing.assert_allclose(second[name], first[name], rtol=0, atol=1e-6)

    def test_mean_maxsim_records_length_effect_without_sum_bias(self):
        inputs = self.fixture()
        duplicated = [inputs[0], np.concatenate([inputs[1], inputs[1]]), inputs[2]]
        result = self.compute(duplicated, device="cpu")
        np.testing.assert_allclose(result["directed"][1], [.5, 1., -.5], rtol=0, atol=1e-6)
        duplicated[1] = np.array([[1., 0.], [0., 1.], [0., 1.]], dtype=np.float32)
        unequal = self.compute(duplicated, device="cpu")
        self.assertAlmostEqual(float(unequal["directed"][1, 0]), 1 / 3, places=6)
        self.assertEqual(float(unequal["directed"][0, 1]), 1.)

    def test_attended_image_tokens_include_special_image_ids_and_exclude_prompts(self):
        document = {
            "document_id": 19,
            "tensors": {
                "vectors": np.array([[3., 4.], [100., 0.], [0., 200.], [300., 0.], [-4., 3.]], dtype=np.float32),
                "image_mask": np.array([True, False, True, True, True]),
                "attention_mask": np.array([True, True, False, True, True]),
                "special_mask": np.array([False, False, False, True, False]),
            },
            "token_metadata": {"fixture": True},
        }
        selected = self.select(document)
        # Selection can preserve magnitudes or perform the declared normalization;
        # only the retained row directions constitute this selection invariant.
        self.assertEqual(selected.shape, (3, 2))
        np.testing.assert_allclose(selected / np.linalg.norm(selected, axis=1, keepdims=True),
                                   [[.6, .8], [1., 0.], [-.8, .6]], rtol=0, atol=1e-6)
        empty = copy.deepcopy(document)
        empty["tensors"]["image_mask"][:] = False
        with self.assertRaises(ValueError):
            self.select(empty)
        bad_mask = copy.deepcopy(document)
        bad_mask["tensors"]["image_mask"] = np.array([True])
        with self.assertRaises(ValueError):
            self.select(bad_mask)

    def test_invalid_vectors_are_rejected(self):
        for invalid in (np.empty((0, 2), dtype=np.float32),
                        np.array([[0., 0.]], dtype=np.float32),
                        np.array([[np.nan, 0.]], dtype=np.float32),
                        np.ones((1, 3), dtype=np.float32)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.compute([self.fixture()[0], invalid], device="cpu")

    def test_exact_id_order_missing_duplicates_and_permutation_negative_control(self):
        matrices = self.compute(self.fixture(), device="cpu")
        expected = [19, 4, 22]
        self.validate(matrices, expected, expected)
        for ids in ([4, 19, 22], [19, 4], [19, 4, 4], [19, 4, 999]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.validate(matrices, ids, expected)

    def test_matrix_corruption_is_rejected(self):
        matrices = self.compute(self.fixture(), device="cpu")
        for name, matrix in [("similarity", np.eye(2)),
                             ("distance", np.full((3, 3), np.nan)),
                             ("affinity", -np.ones((3, 3))),
                             ("similarity", np.array([[1., 0., 0.], [.2, 1., 0.], [0., 0., 1.]]))]:
            mutated = {key: value.copy() for key, value in matrices.items()}
            mutated[name] = matrix
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(mutated, [19, 4, 22], [19, 4, 22])


if __name__ == "__main__":
    unittest.main()
