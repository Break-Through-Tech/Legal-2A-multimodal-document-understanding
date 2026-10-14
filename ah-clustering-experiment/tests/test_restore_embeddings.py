"""Publication checks for corruption, existing data, and archive path handling."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location(
    "restore_embeddings", Path(__file__).resolve().parents[1] / "experiments/restore_embeddings.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "ah-clustering-experiment"
        self.root.mkdir()
        self.name = "outputs/embeddings/example/tensor.npy"
        self.payload = b"original tensor bytes"
        self.archive = self.base / "example.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(self.name, self.payload)
        self.entry = {
            "artifact_id": "example", "document_count": 1, "archive": self.archive.name,
            "archive_bytes": self.archive.stat().st_size,
            "archive_sha256": module.sha256(self.archive),
            "files": {self.name: {"bytes": len(self.payload),
                                   "sha256": hashlib.sha256(self.payload).hexdigest()}},
        }

    def restore(self):
        return module.restore("example", self.entry, self.root, self.base)

    def test_restore_and_reuse(self):
        self.assertEqual(self.restore()["restored_files"], 1)
        self.assertEqual((self.root / self.name).read_bytes(), self.payload)
        self.assertEqual(self.restore()["reused_files"], 1)

    def test_different_existing_file_is_preserved(self):
        target = self.root / self.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"local results")
        with self.assertRaisesRegex(ValueError, "left untouched"):
            self.restore()
        self.assertEqual(target.read_bytes(), b"local results")

    def test_corrupt_archive_rejected_before_writing(self):
        data = bytearray(self.archive.read_bytes())
        data[40] ^= 1
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Archive checksum"):
            self.restore()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_corrupt_payload_rejected_before_writing(self):
        self.entry["files"][self.name]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Expanded checksum"):
            self.restore()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_path_traversal_rejected(self):
        self.entry["files"]["outputs/../../escaped"] = self.entry["files"].pop(self.name)
        with self.assertRaisesRegex(ValueError, "Invalid snapshot path"):
            self.restore()
        self.assertFalse((self.base / "escaped").exists())

    def test_lfs_pointer_has_actionable_error(self):
        self.archive.write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "git lfs pull"):
            self.restore()

    def test_split_archive_restore_and_corrupt_part(self):
        data = self.archive.read_bytes()
        self.entry["parts"] = []
        for number, payload in enumerate((data[:50], data[50:])):
            path = self.base / f"example.zip.part-{number:03d}"
            path.write_bytes(payload)
            self.entry["parts"].append({"name": path.name, "bytes": len(payload),
                                         "sha256": module.sha256(path)})
        self.archive.unlink()
        self.assertEqual(self.restore()["restored_files"], 1)
        self.assertEqual((self.root / self.name).read_bytes(), self.payload)
        first = self.base / self.entry["parts"][0]["name"]
        first.write_bytes(b"x" * first.stat().st_size)
        with self.assertRaisesRegex(ValueError, "part checksum"):
            self.restore()

    def test_missing_split_archive_part(self):
        self.entry["parts"] = [{"name": "missing.zip.part-000", "bytes": 5, "sha256": "0" * 64}]
        with self.assertRaisesRegex(ValueError, "git lfs pull"):
            self.restore()


if __name__ == "__main__":
    unittest.main()
