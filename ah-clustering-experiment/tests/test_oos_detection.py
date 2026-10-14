"""Unit and invariant tests for Out-of-Scope Detection."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.detector import OutOfScopeDetector
from src.detection.evaluation import compute_rejection_metrics, format_comparison_table, sweep_thresholds
from src.detection.pooling import pool_colpali_tokens, pool_tokens


class TestTokenPooling(unittest.TestCase):
    def test_pool_tokens_2d_and_3d(self):
        # 2D case
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pooled = pool_tokens(vecs)
        expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)
        np.testing.assert_allclose(pooled, expected, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(pooled)), 1.0, places=5)

        # 3D case (batch = 2)
        vecs_batch = np.stack([vecs, vecs * 2], axis=0)
        pooled_batch = pool_tokens(vecs_batch)
        self.assertEqual(pooled_batch.shape, (2, 2))
        np.testing.assert_allclose(pooled_batch[0], expected, atol=1e-6)
        np.testing.assert_allclose(pooled_batch[1], expected, atol=1e-6)

    def test_pool_colpali_tokens_with_masks(self):
        # 4 tokens: token 0 (image & attended), token 1 (image but unattended), token 2 (attended but not image), token 3 (image & attended)
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 5.0],
            [9.0, 9.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        attn_mask = np.array([True, False, True, True])
        img_mask = np.array([True, True, False, True])

        doc = {
            "tensors": {
                "vectors": vectors,
                "attention_mask": attn_mask,
                "image_mask": img_mask,
            }
        }
        # Tokens 0 and 3 are selected: mean is [0.5, 0.5], normalized is [1/sqrt(2), 1/sqrt(2)]
        pooled = pool_colpali_tokens(doc)
        expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.float32)
        np.testing.assert_allclose(pooled, expected, atol=1e-6)


class TestDetectorAlgorithms(unittest.TestCase):
    def setUp(self):
        # Class A: vectors along X axis [1, 0]
        # Class B: vectors along Y axis [0, 1]
        self.X_train = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ], dtype=np.float32)
        self.y_train = ["class_a", "class_a", "class_b", "class_b"]

    def test_nearest_similarity_hand_calculation(self):
        detector = OutOfScopeDetector(method="nearest")
        detector.fit(self.X_train, self.y_train)

        # Query close to [1, 0]
        query = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        scores = detector.score(query)

        # Max dot product with X_train for query[0]: dot([1,0], [1,0]) = 1.0
        self.assertAlmostEqual(float(scores[0]), 1.0, places=5)
        # Query[1] is opposite [-1, 0], max dot product should be max(-1, -0.9, 0, -0.1) = 0.0
        self.assertAlmostEqual(float(scores[1]), 0.0, places=5)

    def test_centroid_similarity_hand_calculation(self):
        detector = OutOfScopeDetector(method="centroid")
        detector.fit(self.X_train, self.y_train)

        # Two centroids should exist
        self.assertEqual(len(detector.centroid_classes), 2)
        self.assertEqual(detector.centroids.shape, (2, 2))

        # Centroid for class_a should be normalized mean of [1, 0] and [0.9, 0.1]
        mean_a = (np.array([1.0, 0.0]) + np.array([0.9, 0.1]) / np.linalg.norm([0.9, 0.1])) / 2.0
        mean_a /= np.linalg.norm(mean_a)
        np.testing.assert_allclose(detector.centroids[0], mean_a, atol=1e-5)

        # Query aligned with class_a
        score = detector.score(mean_a.reshape(1, -1))
        self.assertAlmostEqual(float(score[0]), 1.0, places=5)

    def test_missing_labels_for_centroid_raises_error(self):
        detector = OutOfScopeDetector(method="centroid")
        with self.assertRaisesRegex(ValueError, "y_train_legal_classes is required"):
            detector.fit(self.X_train, y_train_legal_classes=None)

    def test_invalid_method_raises_error(self):
        with self.assertRaises(ValueError):
            OutOfScopeDetector(method="unsupported")


class TestThresholdTuning(unittest.TestCase):
    def test_tune_threshold_max_false_rejection(self):
        # 100 legal documents with high similarity (0.8 to 1.0)
        # 100 OOS documents with low similarity (0.0 to 0.4)
        rng = np.random.default_rng(42)
        legal_scores = rng.uniform(0.8, 1.0, size=100)
        oos_scores = rng.uniform(0.0, 0.4, size=100)

        scores = np.concatenate([legal_scores, oos_scores])
        is_oos = np.array([False] * 100 + [True] * 100)

        # Mock detector with pre-fixed score
        detector = OutOfScopeDetector(method="nearest")
        detector.train_vectors = np.eye(2, dtype=np.float32)
        detector.score = lambda X: scores

        # Max false rejection = 5%
        tau = detector.tune_threshold(
            X_val=np.zeros((200, 2)),
            is_out_of_scope_val=is_oos,
            policy="max_false_rejection",
            target_value=0.05,
        )

        metrics = compute_rejection_metrics(is_oos, scores, threshold=tau)
        # At most 5% of legal documents rejected
        self.assertLessEqual(metrics["legal_false_rejection_rate"], 0.06)
        # 100% of out-of-scope should be rejected since distributions don't overlap
        self.assertEqual(metrics["oos_recall"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_sweep_thresholds(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        is_oos = np.array([True, True, False, False])
        curve = sweep_thresholds(is_oos, scores, n_points=5)
        self.assertEqual(len(curve), 5)
        self.assertIn("oos_recall", curve[0])
        self.assertIn("legal_false_rejection_rate", curve[0])


class TestMetricsAndTable(unittest.TestCase):
    def test_format_comparison_table(self):
        results = {
            "ColPali - Nearest Doc": {
                "roc_auc": 0.9850,
                "pr_auc": 0.9700,
                "threshold": 0.7250,
                "oos_recall": 0.9500,
                "legal_false_rejection_rate": 0.0400,
                "oos_f1": 0.9550,
            }
        }
        table = format_comparison_table(results)
        self.assertIn("ColPali - Nearest Doc", table)
        self.assertIn("0.9850", table)
        self.assertIn("95.0%", table)


if __name__ == "__main__":
    unittest.main()
