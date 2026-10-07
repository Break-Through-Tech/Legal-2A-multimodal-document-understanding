"""Independent requirement-derived boundaries; these are not model-inference evidence."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from src.embeddings.ocr import parse_tsv, validate_ocr_record
from src.embeddings.models import validate_output_alignment


HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


class IndependentOCRBoundaries(unittest.TestCase):
    """Tesseract TSV geometry is specified in original image pixel coordinates."""

    def test_asymmetric_image_coordinates_and_fractional_confidence(self):
        # Hand-calculated normalized XYXY: [10/200, 15/300, 50/200, 75/300].
        actual = parse_tsv(HEADER + "5\t1\t1\t1\t1\t1\t10\t15\t40\t60\t91.125\tContract\n", 200, 300)
        self.assertEqual(actual, (["Contract"], [[50, 50, 250, 250]], [91.125]))

    def test_floor_not_round_and_full_image_edge(self):
        text = HEADER + "5\t1\t1\t1\t1\t1\t1\t1\t2\t6\t100\tX\n"
        self.assertEqual(parse_tsv(text, 3, 7)[1], [[333, 142, 1000, 1000]])

    def test_successful_empty_page_does_not_invent_words(self):
        text = HEADER + "1\t1\t0\t0\t0\t0\t0\t0\t200\t300\t-1\t\n"
        self.assertEqual(parse_tsv(text, 200, 300), ([], [], []))

    def test_bad_pixel_geometry_is_rejected(self):
        for left, top, width, height in [(-1, 0, 3, 5), (0, 0, 0, 3), (199, 0, 2, 3), (0, 299, 2, 2)]:
            with self.subTest(rect=(left, top, width, height)), self.assertRaises(ValueError):
                parse_tsv(HEADER + f"5\t1\t1\t1\t1\t1\t{left}\t{top}\t{width}\t{height}\t90\tX\n", 200, 300)

    def test_nonfinite_or_out_of_range_confidence_is_rejected(self):
        for confidence in ["nan", "inf", "-1", "101"]:
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                parse_tsv(HEADER + f"5\t1\t1\t1\t1\t1\t1\t1\t2\t2\t{confidence}\tX\n", 200, 300)

    def test_word_row_on_second_page_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_tsv(HEADER + "5\t2\t1\t1\t1\t1\t1\t1\t2\t2\t90\tX\n", 200, 300)

    def test_record_negative_controls(self):
        document = {"document_id": 19, "image_path": "outputs/images/19.png", "image_sha256": "a" * 64}
        record = {**document, "image_width": 200, "image_height": 300,
                  "words": ["Contract"], "boxes": [[50, 50, 250, 250]],
                  "word_confidences": [91.125], "confidence": 91.125, "status": "ok"}
        validate_ocr_record(record, document)
        for field, value in [("document_id", 20), ("image_sha256", "b" * 64),
                             ("boxes", []), ("boxes", [[50, 50, 1001, 250]]),
                             ("status", "empty"), ("words", [""]), ("confidence", 0.0)]:
            mutated = copy.deepcopy(record)
            mutated[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_ocr_record(mutated, document)


class IndependentOutputMaskBoundaries(unittest.TestCase):
    """Literal arrays test validation only; no claim of model execution."""

    def setUp(self):
        self.vectors = np.ones((2, 7, 128), dtype=np.float32)
        self.mask = np.array([[1] * 7, [0, 0, 1, 1, 1, 1, 1]], dtype=np.int64)
        self.ids = np.arange(14, dtype=np.int64).reshape(2, 7)

    def test_aligned_variable_padding_is_accepted(self):
        validate_output_alignment(self.vectors, self.mask, self.ids)

    def test_input_mask_missing_returned_visual_positions_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "alignment"):
            validate_output_alignment(self.vectors, self.mask[:, :3], self.ids)

    def test_nonbinary_mask_is_rejected(self):
        self.mask[1, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_output_alignment(self.vectors, self.mask, self.ids)

    def test_nonfinite_output_is_rejected(self):
        self.vectors[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_output_alignment(self.vectors, self.mask, self.ids)

    def test_nonintegral_token_identifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_output_alignment(self.vectors, self.mask, self.ids.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
