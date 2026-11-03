"""Contract tests for the pooled single-vector baseline arm."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True

from src.classification.baselines import (
    PooledClassifier,
    pool_document,
    pooled_bytes,
    pooled_matrix,
    retained_bytes,
)


def document(vectors, attention, image, special=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    count = len(vectors)
    return {
        "document_id": 0,
        "tensors": {
            "vectors": vectors,
            "attention_mask": np.asarray(attention, dtype=bool),
            "image_mask": np.asarray(image, dtype=bool),
            "special_mask": np.asarray(special if special is not None else [False] * count, dtype=bool),
        },
    }


class Pooling(unittest.TestCase):
    def test_mean_then_normalize_matches_hand_computation(self):
        # Selected rows are [1,0,0] and [1,1,0]; their mean normalizes to the bisector.
        doc = document([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                       attention=[True, True, False, True],
                       image=[True, False, False, True])
        pooled = pool_document(doc)
        expected = np.array([math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0], dtype=np.float32)
        self.assertEqual(pooled.shape, (3,))
        np.testing.assert_allclose(pooled, expected, atol=1e-6)

    def test_output_is_unit_length(self):
        doc = document([[3, 4, 0], [0, 5, 12]], attention=[True, True], image=[True, True])
        self.assertAlmostEqual(float(np.linalg.norm(pool_document(doc))), 1.0, places=6)

    def test_special_mask_does_not_affect_selection(self):
        plain = document([[1, 0], [0, 1]], attention=[True, True], image=[True, True])
        special = document([[1, 0], [0, 1]], attention=[True, True], image=[True, True],
                           special=[True, True])
        np.testing.assert_allclose(pool_document(plain), pool_document(special))

    def test_no_selected_token_is_rejected(self):
        doc = document([[1, 0], [0, 1]], attention=[False, False], image=[True, True])
        with self.assertRaises(ValueError):
            pool_document(doc)

    def test_zero_norm_mean_is_rejected(self):
        doc = document([[1, 0], [-1, 0]], attention=[True, True], image=[True, True])
        with self.assertRaises(ValueError):
            pool_document(doc)

    def test_matrix_preserves_requested_order(self):
        docs = {7: document([[1, 0], [0, 1]], [True, True], [True, True]),
                3: document([[0, 1], [1, 0]], [True, True], [True, True])}
        matrix = pooled_matrix(docs, [7, 3])
        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_allclose(matrix[0], pool_document(docs[7]))
        np.testing.assert_allclose(matrix[1], pool_document(docs[3]))

    def test_missing_document_is_rejected(self):
        with self.assertRaises(ValueError):
            pooled_matrix({1: document([[1, 0]], [True], [True])}, [1, 2])


class Classifier(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = rng.normal(loc=[2.0, 0.0], scale=0.1, size=(20, 2)).astype(np.float32)
        self.b = rng.normal(loc=[0.0, 2.0], scale=0.1, size=(20, 2)).astype(np.float32)
        self.X = np.vstack([self.a, self.b])
        self.y = ["A"] * 20 + ["B"] * 20

    def test_both_methods_separate_easy_clusters(self):
        for method in ("logistic_regression", "nearest_centroid"):
            model = PooledClassifier(method=method).fit(self.X, self.y)
            predicted = model.predict(self.X)
            self.assertEqual(predicted, self.y, msg=method)

    def test_predictions_are_strings(self):
        model = PooledClassifier().fit(self.X, self.y)
        self.assertTrue(all(isinstance(value, str) for value in model.predict(self.X)))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            PooledClassifier(method="random_forest")

    def test_non_positive_C_is_rejected(self):
        with self.assertRaises(ValueError):
            PooledClassifier(C=0)

    def test_single_class_is_rejected(self):
        with self.assertRaises(ValueError):
            PooledClassifier().fit(self.a, ["A"] * len(self.a))

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(ValueError):
            PooledClassifier().predict(self.X)

    def test_label_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            PooledClassifier().fit(self.X, ["A"] * 3)


class Cost(unittest.TestCase):
    def test_retained_bytes_sums_only_vector_shards(self):
        manifest = {"files": {
            "shard-00000-vectors": {"bytes": 100},
            "shard-00000-attention_mask": {"bytes": 10},
            "shard-00001-vectors": {"bytes": 250},
            "shard-00001-image_mask": {"bytes": 12},
        }}
        self.assertEqual(retained_bytes(manifest), 350)

    def test_pooled_bytes_is_count_times_dimension(self):
        self.assertEqual(pooled_bytes(1000, 128), 1000 * 128 * 4)
        self.assertEqual(pooled_bytes(10, 8, dtype_itemsize=2), 10 * 8 * 2)

    def test_pooled_bytes_rejects_empty(self):
        with self.assertRaises(ValueError):
            pooled_bytes(0, 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
