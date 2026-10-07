"""Hand-derived cosine expectations and real NumPy/CUDA differential checks."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering.similarity import DEFINITION, compute_similarity, select_vectors, validate_matrices


def double_precision_oracle(documents):
    """Small direct cosine definition; independent of tiled implementation."""
    normalized = [array.astype(np.float64) / np.linalg.norm(array.astype(np.float64), axis=1)[:, None]
                  for array in documents]
    return np.array([[np.mean([max(float(np.dot(a, b)) for b in target) for a in source])
                      for target in normalized] for source in normalized])


class Similarity(unittest.TestCase):
    def test_direction_symmetry_and_transforms_match_hand_calculation(self):
        a = np.eye(2, dtype=np.float32)
        b = np.array([[1, 0]], dtype=np.float32)
        result = compute_similarity([a, b])
        np.testing.assert_array_equal(result["directed"], [[1, .5], [1, 1]])
        np.testing.assert_array_equal(result["similarity"], [[1, .75], [.75, 1]])
        np.testing.assert_array_equal(result["distance"], [[0, .25], [.25, 0]])
        np.testing.assert_array_equal(result["affinity"], [[1, .875], [.875, 1]])
        validate_matrices(result, [9, 3], [9, 3])

    def test_attended_image_specials_are_kept_prompt_and_padding_excluded(self):
        document = {"tensors": {
            "vectors": np.array([[0, 3], [4, 0], [10, 10], [-10, 10]], dtype=np.float32),
            "attention_mask": np.array([True, True, True, False]),
            "image_mask": np.array([True, True, False, True]),
            "special_mask": np.array([True, False, True, True]),
        }}
        np.testing.assert_array_equal(select_vectors(document), [[0, 1], [1, 0]])
        self.assertEqual(DEFINITION["selection"], "attention_mask & image_mask")
        self.assertEqual(select_vectors(document).dtype, np.float32)

    def test_missing_misaligned_nonboolean_and_empty_masks_fail(self):
        base = {"tensors": {"vectors": np.eye(2, dtype=np.float32),
                            "attention_mask": np.array([True, True]),
                            "image_mask": np.array([True, False]),
                            "special_mask": np.array([True, False])}}
        for invalid in (None, np.array([True]), np.array([1, 0]), np.array([False, False])):
            with self.subTest(invalid=invalid):
                document = copy.deepcopy(base)
                document["tensors"]["image_mask"] = invalid
                with self.assertRaises(ValueError):
                    select_vectors(document)

    def test_positive_scaling_and_whole_document_duplication_do_not_change_scores(self):
        vectors = [np.array([[1, 2], [3, -1]], dtype=np.float32),
                   np.array([[-1, 2], [2, 0], [0, -4]], dtype=np.float32)]
        original = compute_similarity(vectors)
        scaled = compute_similarity([vectors[0] * np.array([[1e30], [1e-30]], dtype=np.float32), vectors[1] * 5])
        repeated = compute_similarity([np.tile(vectors[0], (3, 1)), vectors[1]])
        for name in original:
            np.testing.assert_allclose(scaled[name], original[name], atol=2e-6, rtol=0)
            np.testing.assert_allclose(repeated[name], original[name], atol=2e-6, rtol=0)

    def test_target_duplication_preserves_max_but_source_distribution_can_change_mean(self):
        source = np.array([[1, 0]], dtype=np.float32)
        target = np.eye(2, dtype=np.float32)
        original = compute_similarity([source, target])
        duplicate = compute_similarity([source, np.vstack([target, target[0]])])
        self.assertEqual(original["directed"][0, 1], duplicate["directed"][0, 1])
        self.assertNotEqual(original["directed"][1, 0], duplicate["directed"][1, 0])

    def test_padding_cannot_win_against_negative_cosines(self):
        result = compute_similarity([np.array([[1., 0.]], dtype=np.float32),
                                     np.array([[-1., 0.]], dtype=np.float32),
                                     np.array([[-1., 0.], [-1., 0.]], dtype=np.float32)])
        np.testing.assert_array_equal(result["directed"][0], [1, -1, -1])
        np.testing.assert_array_equal(result["distance"][0], [0, 2, 2])

    def test_nonmetric_triangle_violation_is_explicit(self):
        documents = [np.array([value], dtype=np.float32) for value in ([1, 0], [1, 1], [0, 1])]
        distance = compute_similarity(documents)["distance"]
        self.assertGreater(distance[0, 2], distance[0, 1] + distance[1, 2])
        self.assertTrue(DEFINITION["nonmetric"])

    def test_invalid_vectors_fail(self):
        for value in (np.empty((0, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
                      np.array([[float("nan"), 1]], dtype=np.float32), np.array([[1, 2]]),
                      np.array([[float("inf"), 0]], dtype=np.float32)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_similarity([value])
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            compute_similarity([np.ones((1, 2), dtype=np.float32), np.ones((1, 3), dtype=np.float32)])

    def test_determinism_and_independent_double_precision_oracle(self):
        random = np.random.default_rng(312)
        documents = [random.normal(size=(n, 7)).astype(np.float32) for n in (1, 3, 7, 2, 4, 9, 1, 5, 3, 2)]
        events = []
        result = compute_similarity(documents, progress=events.append)
        np.testing.assert_allclose(result["directed"], double_precision_oracle(documents), atol=2e-6, rtol=0)
        repeated = compute_similarity(documents)
        for name in result:
            np.testing.assert_array_equal(result[name], repeated[name])
        self.assertEqual(events[-1]["completed_pairs"], 45)
        self.assertEqual(events[-1]["total_pairs"], 45)

    def test_invalid_matrix_shapes_ids_ranges_diagonals_and_transforms_fail(self):
        result = compute_similarity([np.eye(2, dtype=np.float32), np.ones((1, 2), dtype=np.float32)])
        with self.assertRaisesRegex(ValueError, "order"):
            validate_matrices(result, [1, 2], [2, 1])
        for name, change in (("similarity", "nan"), ("directed", "range"), ("distance", "diagonal"),
                             ("distance", "wrong_transform"), ("affinity", "shape")):
            altered = copy.deepcopy(result)
            if change == "shape":
                altered[name] = altered[name][:1]
            elif change == "diagonal":
                altered[name][0, 0] = .1
            elif change == "wrong_transform":
                altered[name][0, 1] = altered[name][1, 0] = .5
            else:
                altered[name][0, 1] = float("nan") if change == "nan" else 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_matrices(altered, [1, 2], [1, 2])

    def test_cuda_and_numpy_agree_across_token_chunk_boundaries(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable; CUDA differential unverified")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; GPU differential unverified")
        random = np.random.default_rng(63)
        documents = [random.normal(size=(n, 8)).astype(np.float32) for n in (1025, 1029, 3)]
        cpu = compute_similarity(documents)
        cuda = compute_similarity(documents, device="cuda")
        repeat = compute_similarity(documents, device="cuda")
        for name in cpu:
            np.testing.assert_allclose(cuda[name], cpu[name], atol=3e-6, rtol=0)
            np.testing.assert_array_equal(cuda[name], repeat[name])


if __name__ == "__main__":
    unittest.main()
