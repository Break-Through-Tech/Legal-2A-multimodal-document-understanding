"""Local filesystem recovery checks with an explicitly substituted OCR engine.

These fixtures certify checkpoint reuse and validation, not Tesseract accuracy
or the locking semantics of a remote mount. A failed engine call supplies the
interruption; the journal, image decoding, artifact writer and readers are real.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import artifact_identity, read_artifact
from src.dataset import file_sha256
from src.embeddings.ocr import COORDINATES, DEFAULT_CONFIG, prepare_ocr
from src.provenance import source_identity
from src.recovery import RecoveryJournal


HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
ENGINE = {"engine": "tesseract", "engine_version": "fixture-engine-for-unit-test",
          "executable_sha256": "a" * 64, "traineddata_sha256": {"eng": "b" * 64}}


class OCRRecovery(unittest.TestCase):
    def setUp(self):
        cache = ROOT / "cache/test-ocr-recovery"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "ah-clustering-experiment"
        (self.root / "outputs/images").mkdir(parents=True)
        self.source = Path(self.temporary.name) / "source"
        self.source.mkdir()
        self.documents = []
        for document_id in (4, 8, 12):
            relative = f"outputs/images/{document_id}.png"
            Image.new("RGB", (100, 80), (document_id, 0, 0)).save(self.root / relative)
            self.documents.append({"document_id": document_id, "image_path": relative,
                                   "image_sha256": file_sha256(self.root / relative)})
        self.attempts = []
        self.fail_on = None
        self.addCleanup(patch.stopall)
        patch("src.embeddings.ocr._engine", return_value=(["fixture-engine"], ENGINE)).start()
        patch("src.embeddings.ocr._run", side_effect=self.engine).start()
        # The expected recovery identity includes every input affecting OCR.
        self.identity = {**ENGINE, "documents": self.documents,
                         "image_decoder": {"Pillow": importlib.metadata.version("Pillow")},
                         "options": DEFAULT_CONFIG.copy(), "coordinate_transform": COORDINATES,
                         "image_transform": "single-page decoded RGB PNG; original size; no EXIF rotation, resize or deskew",
                         "source_digest": source_identity(self.source)["source_digest"]}
        self.key = artifact_identity("ocr", self.identity)[0]

    def engine(self, command, *, timeout, payload):
        with Image.open(io.BytesIO(payload)) as image:
            document_id = image.getpixel((0, 0))[0]
        self.attempts.append(document_id)
        if document_id == self.fail_on:
            raise RuntimeError("injected engine interruption")
        return HEADER + f"5\t1\t1\t1\t1\t1\t10\t10\t20\t20\t95\tword-{document_id}\n"

    def prepare(self):
        return prepare_ocr(self.root, self.documents, source_root=self.source)

    def interrupt(self):
        self.fail_on = 8
        with self.assertRaisesRegex(RuntimeError, "injected engine interruption"):
            self.prepare()
        self.assertEqual(self.attempts, [4, 8])
        self.assertFalse((self.root / f"outputs/metadata/{self.key}/artifact.json").exists())
        self.fail_on = None
        self.attempts.clear()

    def test_completed_document_reused_after_interruption_and_exact_ids_published(self):
        self.interrupt()
        identity, records = self.prepare()
        self.assertEqual(self.attempts, [8, 12])
        self.assertEqual(list(records), [4, 8, 12])
        self.assertEqual([records[key]["words"] for key in records],
                         [["word-4"], ["word-8"], ["word-12"]])
        measurements = read_artifact(self.root, identity["artifact_id"])["manifest"]["provenance"]["measurements"]
        self.assertEqual(measurements["reused_documents"], 1)
        self.assertEqual(measurements["processed_documents"], 2)
        self.attempts.clear()
        self.assertEqual(self.prepare(), (identity, records))
        self.assertEqual(self.attempts, [])
        clean_root = Path(self.temporary.name) / "uninterrupted/ah-clustering-experiment"
        shutil.copytree(self.root / "outputs/images", clean_root / "outputs/images")
        _, uninterrupted = prepare_ocr(clean_root, self.documents, source_root=self.source)
        self.assertEqual(uninterrupted, records)
        self.assertEqual(self.attempts, [4, 8, 12])

    def test_checksum_corrupt_checkpoint_is_recomputed(self):
        self.interrupt()
        with RecoveryJournal(self.root, self.key, self.identity) as journal:
            self.assertIsNotNone(journal.get("document-4"))
            # Deliberately change bytes without their receipt, like a torn write.
            (journal.directory / "document-4.bin").write_bytes(b"corrupt")
            self.assertIsNone(journal.get("document-4"))
        self.prepare()
        self.assertEqual(self.attempts, [4, 8, 12])

    def test_invalid_checksum_valid_checkpoint_is_recomputed(self):
        self.interrupt()
        with RecoveryJournal(self.root, self.key, self.identity) as journal:
            record = json.loads(journal.get("document-4"))
            record["document_id"] = 999
            journal.put("document-4", json.dumps(record).encode())
        _, records = self.prepare()
        self.assertEqual(self.attempts, [4, 8, 12])
        self.assertEqual(list(records), [4, 8, 12])

    def test_malformed_json_checkpoint_is_recomputed(self):
        self.interrupt()
        with RecoveryJournal(self.root, self.key, self.identity) as journal:
            journal.put("document-4", b"{broken")
        self.prepare()
        self.assertEqual(self.attempts, [4, 8, 12])

    def test_dimensions_are_checked_against_original_image(self):
        self.interrupt()
        with RecoveryJournal(self.root, self.key, self.identity) as journal:
            record = json.loads(journal.get("document-4"))
            record["image_width"] = 999
            journal.put("document-4", json.dumps(record).encode())
        _, records = self.prepare()
        self.assertEqual(self.attempts, [4, 8, 12])
        self.assertEqual(records[4]["image_width"], 100)

    def test_changed_source_image_fails_before_checkpoint_reuse(self):
        self.interrupt()
        Image.new("RGB", (100, 80), "white").save(self.root / self.documents[0]["image_path"])
        with self.assertRaisesRegex(ValueError, "source image checksum mismatch"):
            self.prepare()
        self.assertEqual(self.attempts, [])

    def test_changed_ocr_configuration_does_not_reuse_partial_documents(self):
        self.interrupt()
        identity, _ = prepare_ocr(self.root, self.documents, source_root=self.source, config={"psm": 6})
        self.assertNotEqual(identity["artifact_id"], self.key)
        self.assertEqual(self.attempts, [4, 8, 12])

    def test_failure_log_preserves_progress_and_resume_has_separate_completed_log(self):
        self.interrupt()
        log_paths = list((self.root / "outputs/logs").glob("ocr-request-*.json"))
        self.assertEqual(len(log_paths), 1)
        failed = json.loads(log_paths[0].read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["artifact_id"], self.key)
        self.assertEqual(failed["completed_document_ids"], [4])
        self.assertEqual(failed["completed_document_count"], 1)
        self.assertEqual(failed["expected_document_count"], 3)
        self.assertEqual(failed["current_document_id"], 8)
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertIn("injected engine interruption", failed["error"])
        self.prepare()
        logs = [json.loads(path.read_text()) for path in (self.root / "outputs/logs").glob("ocr-request-*.json")]
        self.assertEqual(len(logs), 2)
        self.assertIn(failed, logs)
        complete = next(log for log in logs if log["status"] == "complete")
        self.assertEqual(complete["completed_document_ids"], [4, 8, 12])
        self.assertEqual(complete["completed_document_count"], 3)
        self.assertEqual(complete["reused_document_count"], 1)
        self.assertIsNone(complete["current_document_id"])

    def test_engine_discovery_failure_is_logged_before_artifact_identity_exists(self):
        with patch("src.embeddings.ocr._engine", side_effect=RuntimeError("engine missing")):
            with self.assertRaisesRegex(RuntimeError, "engine missing"):
                self.prepare()
        logs = list((self.root / "outputs/logs").glob("ocr-request-*.json"))
        self.assertEqual(len(logs), 1)
        failed = json.loads(logs[0].read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["artifact_id"])
        self.assertEqual(failed["completed_document_count"], 0)
        self.assertEqual(failed["error"], "engine missing")


if __name__ == "__main__":
    unittest.main()
