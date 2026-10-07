"""Regression checks for independently reviewed restart edge cases.

The existing synthetic encoder supplies numeric fixtures; journal locks,
checkpoint validation, publication, and cache readers use real local files.
"""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_step5_independent as independent
from src.artifacts import ArtifactWriter
from src.dataset import fingerprint
from src.embeddings.pipeline import extract_embeddings
from src.recovery import RecoveryJournal


class RecoveryEdges(unittest.TestCase):
    setUp = independent.IndependentStep5Recovery.setUp
    extract = independent.IndependentStep5Recovery.extract
    inferred_ids = independent.IndependentStep5Recovery.inferred_ids
    assert_native = independent.IndependentStep5Recovery.assert_native

    def test_old_writer_receipt_cannot_delete_new_writer_in_same_process(self):
        with RecoveryJournal(self.root, "ownership-edge", {"case": "ownership"}) as journal:
            with ArtifactWriter(self.root, "ocr", {"case": "nonce"}) as writer:
                journal.mark_artifact_writer(writer.artifact_id)
                first_lock = writer.lock_path.read_bytes()
            with ArtifactWriter(self.root, "ocr", {"case": "nonce"}) as next_writer:
                self.assertNotEqual(first_lock, next_writer.lock_path.read_bytes())
                with self.assertRaisesRegex(ValueError, "Unknown artifact writer lock"):
                    journal.clear_artifact_lock(next_writer.artifact_id)
                self.assertTrue(next_writer.lock_path.is_file())

    def test_unknown_writer_without_receipt_is_not_removed(self):
        with RecoveryJournal(self.root, "unknown-writer", {"case": "unknown"}) as journal:
            with ArtifactWriter(self.root, "ocr", {"case": "unknown"}) as writer:
                with self.assertRaisesRegex(ValueError, "Unknown artifact writer lock"):
                    journal.clear_artifact_lock(writer.artifact_id)
                self.assertTrue(writer.lock_path.is_file())

    def repair_malformed_configuration(self, *, nested):
        with self.assertRaisesRegex(RuntimeError, "independent injected"):
            self.extract(fail_call=2)
        configs = list((self.root / "outputs/metadata").glob("embeddings-*/config.json"))
        self.assertEqual(len(configs), 1)
        config = json.loads(configs[0].read_text())
        request = config["processor"]["execution"]
        malformed = copy.deepcopy(config) if nested else []
        if nested:
            malformed["model"] = []
        with RecoveryJournal(self.root, "extraction-" + fingerprint(request)[:20], request) as journal:
            journal.put("embedding-config", json.dumps(malformed).encode())
        result = self.extract()
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 4, 5])
        self.assert_native(result)

    def test_nonobject_configuration_checkpoint_rebuilds_without_repeating_saved_batch(self):
        self.repair_malformed_configuration(nested=False)

    def test_invalid_nested_configuration_rebuilds_without_repeating_saved_batch(self):
        self.repair_malformed_configuration(nested=True)

    def test_published_artifact_without_index_recovers_without_loading_model(self):
        result = self.extract()
        indices = list((self.root / "outputs/metadata").glob("extraction-*.json"))
        self.assertEqual(len(indices), 1)
        indices[0].unlink()
        with patch("src.embeddings.pipeline.importlib.metadata.version", return_value="fixture-version"), \
             patch("src.embeddings.models.load_encoder", side_effect=AssertionError("Publication recovery loaded model")):
            restored = extract_embeddings(independent.SETTINGS, self.root, self.directory,
                                          source_root=self.source, document_ids=list(independent.GOLDEN))
        self.assertTrue(restored["reused"])
        self.assertEqual(restored["artifact_id"], result["artifact_id"])
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 4, 5])
        self.assertTrue(indices[0].is_file())
        self.assert_native(restored)


if __name__ == "__main__":
    unittest.main()
