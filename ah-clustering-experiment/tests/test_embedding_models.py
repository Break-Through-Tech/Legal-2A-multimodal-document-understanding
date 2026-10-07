"""Boundary negative controls. These do not certify pretrained model inference."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embeddings.models import NativeEncoder, validate_output_alignment, validate_settings


SETTINGS = {"family": "dinov3", "checkpoint": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "revision": "114c1379950215c8b35dfcd4e90a5c251dde0d32", "dtype": "float32",
            "input_mode": "image_only", "batch_size": 1, "processor_options": {}}


class ModelBoundaries(unittest.TestCase):
    def test_mutable_revision_rejected(self):
        for revision in ("main", "v1.0", "1234567"):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "Immutable"):
                validate_settings({**SETTINGS, "revision": revision})

    def test_unserializable_native_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "numpy-compatible"):
            validate_settings({**SETTINGS, "dtype": "bfloat16"})

    def test_nonlayout_ocr_and_invalid_batch_rejected(self):
        for update in ({"input_mode": "image_and_ocr"}, {"batch_size": 0}, {"batch_size": True}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_settings({**SETTINGS, **update})

    def test_processor_cannot_reenable_ocr_or_override_revision(self):
        for options in ({"apply_ocr": True}, {"revision": "main"}, {"text": "reference transcription"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "processor_options"):
                validate_settings({**SETTINGS, "processor_options": options})

    def test_mask_length_checked_against_returned_vectors(self):
        # An input mask with four tokens cannot certify an output with six tokens.
        with self.assertRaisesRegex(ValueError, "token/mask alignment"):
            validate_output_alignment(np.ones((2, 6, 128), dtype=np.float32),
                                      np.ones((2, 4), dtype=np.int64), np.ones((2, 4), dtype=np.int64))

    def test_mask_batch_axis_checked(self):
        with self.assertRaisesRegex(ValueError, "token/mask alignment"):
            validate_output_alignment(np.ones((2, 6, 128), dtype=np.float32),
                                      np.ones((1, 6), dtype=np.int64), np.ones((1, 6), dtype=np.int64))

    def test_token_ids_alignment_checked_separately(self):
        with self.assertRaisesRegex(ValueError, "token/ID alignment"):
            validate_output_alignment(np.ones((2, 6, 128), dtype=np.float32),
                                      np.ones((2, 6), dtype=np.int64), np.ones((2, 5), dtype=np.int64))

    def test_nonbinary_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_output_alignment(np.ones((1, 2, 128), dtype=np.float32),
                                      np.array([[1, 2]]), np.array([[3, 4]]))

    def test_nonfinite_native_outputs_rejected(self):
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                validate_output_alignment(np.full((1, 2, 128), invalid), np.ones((1, 2)), np.ones((1, 2), dtype=np.int64))

    def test_ocr_boundary_negative_controls(self):
        valid = {"document_id": 3, "image_sha256": "a" * 64, "words": ["Invoice"],
                 "boxes": [[0, 0, 1000, 1000]], "word_confidences": [95.0], "status": "ok"}
        for field, value in (("boxes", []), ("boxes", [[0, 0, 1001, 1000]]),
                             ("boxes", [[500, 0, 100, 1000]]), ("word_confidences", [float("nan")]),
                             ("status", "failed"), ("status", "empty"), ("words", [""]),
                             ("image_sha256", "missing")):
            record = copy.deepcopy(valid)
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                NativeEncoder._validate_ocr(record)


if __name__ == "__main__":
    unittest.main()
