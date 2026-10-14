"""Unit tests for Multiclass Classification and Two-Stage Intake Routing."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification.classifier import LegalDocumentClassifier, TwoStageIntakeRouter
from src.classification.evaluation import compute_multiclass_metrics, format_multiclass_table
from src.detection.detector import OutOfScopeDetector


class TestLegalDocumentClassifier(unittest.TestCase):
    def setUp(self):
        # 3 classes in 2D space
        # Class A: [1, 0]
        # Class B: [0, 1]
        # Class C: [-1, 0]
        self.X_train = np.array([
            [1.0, 0.0], [0.95, 0.05], [0.9, -0.1],
            [0.0, 1.0], [0.05, 0.95], [-0.1, 0.9],
            [-1.0, 0.0], [-0.95, 0.05], [-0.9, -0.1],
        ], dtype=np.float32)
        self.y_train = [
            "LEASE", "LEASE", "LEASE",
            "PATENT", "PATENT", "PATENT",
            "REAL_ESTATE", "REAL_ESTATE", "REAL_ESTATE",
        ]

    def test_logistic_regression_fit_and_predict(self):
        clf = LegalDocumentClassifier(method="logistic_regression")
        clf.fit(self.X_train, self.y_train)

        # Query very close to PATENT ([0, 1])
        query = np.array([[0.02, 0.98]], dtype=np.float32)
        pred = clf.predict(query)
        self.assertEqual(pred[0], "PATENT")

        probas = clf.predict_proba(query)
        self.assertEqual(probas.shape, (1, 3))
        # PATENT should have highest probability
        patent_idx = list(clf.classes_).index("PATENT")
        self.assertEqual(np.argmax(probas[0]), patent_idx)

    def test_nearest_centroid_fit_and_predict(self):
        clf = LegalDocumentClassifier(method="nearest_centroid")
        clf.fit(self.X_train, self.y_train)

        query = np.array([[-0.98, 0.01]], dtype=np.float32)
        pred = clf.predict(query)
        self.assertEqual(pred[0], "REAL_ESTATE")

    def test_insufficient_classes_fails(self):
        clf = LegalDocumentClassifier()
        with self.assertRaisesRegex(ValueError, "at least 2 distinct classes"):
            clf.fit(self.X_train[:3], ["LEASE", "LEASE", "LEASE"])


class TestTwoStageIntakeRouter(unittest.TestCase):
    def test_routing_rejects_oos_and_classifies_legal(self):
        # Legal training docs along X-axis
        X_train_legal = np.array([
            [1.0, 0.0], [0.95, 0.05],   # LEASE
            [0.0, 1.0], [0.05, 0.95],   # PATENT
        ], dtype=np.float32)
        y_train_legal = ["LEASE", "LEASE", "PATENT", "PATENT"]

        detector = OutOfScopeDetector(method="nearest")
        detector.fit(X_train_legal, y_train_legal)
        detector.threshold = 0.5  # Reject if similarity < 0.5

        classifier = LegalDocumentClassifier(method="logistic_regression")
        classifier.fit(X_train_legal, y_train_legal)

        router = TwoStageIntakeRouter(detector, classifier)

        # 3 Test queries:
        # 1. Clear LEASE: [0.99, 0.01] (similarity ~ 1.0 >= 0.5 -> Accepted -> LEASE)
        # 2. Clear PATENT: [0.01, 0.99] (similarity ~ 1.0 >= 0.5 -> Accepted -> PATENT)
        # 3. Clear OOS (pizza label pointing opposite): [-1.0, -1.0] (similarity ~ -0.7 < 0.5 -> REJECTED)
        X_test = np.array([
            [0.99, 0.01],
            [0.01, 0.99],
            [-1.0, -1.0],
        ], dtype=np.float32)

        preds = router.predict(X_test)
        self.assertEqual(preds[0], "LEASE")
        self.assertEqual(preds[1], "PATENT")
        self.assertEqual(preds[2], "OUT_OF_SCOPE")

        details = router.predict_with_details(X_test)
        self.assertEqual(details[0]["status"], "ACCEPTED")
        self.assertEqual(details[0]["assigned_label"], "LEASE")
        self.assertEqual(details[2]["status"], "REJECTED")
        self.assertEqual(details[2]["assigned_label"], "OUT_OF_SCOPE")


class TestMulticlassMetrics(unittest.TestCase):
    def test_metrics_calculation_and_table(self):
        y_true = ["LEASE", "LEASE", "PATENT", "PATENT"]
        y_pred = ["LEASE", "PATENT", "PATENT", "PATENT"]
        m = compute_multiclass_metrics(y_true, y_pred)
        self.assertEqual(m["accuracy"], 0.75)
        self.assertIn("LEASE", m["per_class"])
        self.assertIn("PATENT", m["per_class"])

        table = format_multiclass_table(m)
        self.assertIn("**Overall Accuracy:** 75.00%", table)
        self.assertIn("LEASE", table)


if __name__ == "__main__":
    unittest.main()
