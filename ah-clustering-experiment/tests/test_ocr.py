"""TSV and persistence boundary checks; optional live tests use actual Tesseract.

Hand-specified TSV cases test parsing only and do not stand in for engine output.
"""

from __future__ import annotations

import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import file_sha256, fingerprint
from src.embeddings.ocr import (_run, parse_tsv, prepare_ocr, render_ocr_overlay,
                                validate_ocr_config, validate_ocr_record)

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


class OCRBoundaries(unittest.TestCase):
    def test_official_tsv_coordinates_and_confidence(self):
        # Official TSV example coordinates: The at (65,41), size (46,20),
        # page size 640x500. Floor scaling is independently calculated here.
        tsv = HEADER + "1\t1\t0\t0\t0\t0\t0\t0\t640\t500\t-1\t\n5\t1\t1\t1\t1\t1\t65\t41\t46\t20\t96.063751\tThe\n"
        self.assertEqual(parse_tsv(tsv, 640, 500), (["The"], [[101, 82, 173, 122]], [96.063751]))

    def test_blank_page_has_no_invented_text(self):
        self.assertEqual(parse_tsv(HEADER + "1\t1\t0\t0\t0\t0\t0\t0\t640\t500\t-1\t\n", 640, 500), ([], [], []))

    def test_invalid_boxes_and_confidence_rejected(self):
        for left, width, confidence in ((-1, 20, "50"), (639, 20, "50"), (2, 0, "50"), (2, 20, "nan"), (2, 20, "101")):
            with self.subTest(left=left, width=width, confidence=confidence), self.assertRaises(ValueError):
                parse_tsv(HEADER + f"5\t1\t1\t1\t1\t1\t{left}\t20\t{width}\t20\t{confidence}\tword\n", 640, 500)

    def test_config_rejects_unsupported_options(self):
        for config in ({"reference_text": "Never use this"}, {"psm": 0}, {"oem": True},
                       {"language": "eng --psm 6"}, {"timeout_seconds": float("nan")}, {"timeout_seconds": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_ocr_config(config)

    def test_engine_failure_is_explicit(self):
        with patch("src.embeddings.ocr.subprocess.run", side_effect=OSError("engine unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Actual OCR engine failed"):
                _run(["missing-engine"], timeout=1)

    def test_records_reject_alignment_status_and_checksum_errors(self):
        document = {"document_id": 4, "image_path": "outputs/images/4.png", "image_sha256": "a" * 64}
        valid = {**document, "words": ["word"], "boxes": [[0, 0, 100, 100]], "word_confidences": [50.0],
                 "confidence": 50.0, "status": "ok", "image_width": 500, "image_height": 500}
        validate_ocr_record(valid, document)
        for change in ({"boxes": []}, {"boxes": [[0, 0, 1001, 20]]}, {"boxes": [[50, 0, 20, 20]]},
                       {"status": "empty"}, {"image_sha256": "b" * 64}, {"confidence": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_ocr_record({**valid, **change}, document)
        changed = copy.deepcopy(valid)
        changed["words"] = ["different"]
        self.assertNotEqual(fingerprint([valid]), fingerprint([changed]))


@unittest.skipUnless(shutil.which("tesseract"), "Live Tesseract executable unavailable")
class OCRLive(unittest.TestCase):
    def test_actual_engine_roundtrip_overlay_empty_and_checksum_guard(self):
        cache = ROOT / "cache/test-ocr"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            root = Path(temporary) / "ah-clustering-experiment"
            (root / "outputs/images").mkdir(parents=True)
            image = Image.new("RGB", (900, 200), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=44)
            draw.text((30, 65), "LEGAL DOCUMENT 12345", fill="black", font=font)
            image.save(root / "outputs/images/4.png")
            Image.new("RGB", (900, 200), "white").save(root / "outputs/images/5.png")
            docs = [{"document_id": key, "image_path": f"outputs/images/{key}.png",
                     "image_sha256": file_sha256(root / f"outputs/images/{key}.png")} for key in (4, 5)]
            identity, records = prepare_ocr(root, docs, source_root=ROOT, config={"psm": 6})
            self.assertEqual(records[4]["words"], ["LEGAL", "DOCUMENT", "12345"])
            self.assertEqual(records[5]["status"], "empty")
            self.assertEqual(records[5]["confidence"], 0.0)
            self.assertEqual((identity, records), prepare_ocr(root, docs, source_root=ROOT, config={"psm": 6}))
            self.assertTrue(render_ocr_overlay(root, records[4], identity["artifact_id"]).is_file())
            Image.new("RGB", (900, 200), "black").save(root / "outputs/images/4.png")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_ocr(root, docs, source_root=ROOT, config={"psm": 6})


if __name__ == "__main__":
    unittest.main()
