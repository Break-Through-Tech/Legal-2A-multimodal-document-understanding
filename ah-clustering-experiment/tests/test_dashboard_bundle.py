"""The distributed snapshot restores intact without replacing local data."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dashboard_bundle import BUNDLE, INDEX, digest, restore


class DashboardBundleTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'cache' / 'test-dashboard-bundle'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'ah-clustering-experiment'

    def test_real_snapshot_and_repeat(self):
        result = restore(self.root)
        self.assertEqual((result['runs'], result['projections']), (14, 6))
        self.assertGreater(result['written_files'], 0)
        self.assertEqual(restore(self.root)['written_files'], 0)
        for name, entry in json.loads(INDEX.read_text())['files'].items():
            self.assertEqual(digest((self.root / name).read_bytes()), entry['sha256'])
        self.assertFalse((self.root / 'outputs/embeddings').exists())
        self.assertFalse((self.root / 'outputs/images').exists())

    def test_conflict_rejected_before_any_write(self):
        names = list(json.loads(INDEX.read_text())['files'])
        conflict = self.root / names[-1]
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b'existing local result')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            restore(self.root)
        self.assertEqual(conflict.read_bytes(), b'existing local result')
        self.assertFalse((self.root / names[0]).exists())

    def test_corrupt_archive_rejected(self):
        bundle = self.base / 'broken.zip'
        bundle.write_bytes(BUNDLE.read_bytes() + b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            restore(self.root, bundle, INDEX)
        self.assertFalse(self.root.exists())

    def test_traversal_rejected_even_with_matching_checksum(self):
        bundle, index = self.base / 'bad.zip', self.base / 'bad.json'
        name, payload = '../outside.json', b'{}'
        with zipfile.ZipFile(bundle, 'w') as archive:
            archive.writestr(name, payload)
        index.write_text(json.dumps({'schema_version': 1, 'archive_sha256': digest(bundle.read_bytes()),
                                     'files': {name: {'bytes': len(payload), 'sha256': digest(payload)}}}))
        with self.assertRaisesRegex(ValueError, 'Unexpected snapshot path'):
            restore(self.root, bundle, index)
        self.assertFalse((self.base / 'outside.json').exists())


if __name__ == '__main__':
    unittest.main()
