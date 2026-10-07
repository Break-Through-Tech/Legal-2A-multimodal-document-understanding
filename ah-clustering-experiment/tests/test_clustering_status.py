"""Configuration omissions must not be presented as missing cache evidence."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_clustering import unconfigured_models


class ClusteringStatusChecks(unittest.TestCase):
    def test_unconfigured_model_can_have_an_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / 'configs/models'
            configs.mkdir(parents=True)
            for name in ('layoutlmv3-ocr', 'dinov3', 'another-model'):
                (configs / f'{name}.json').write_text('{}')
            cache = root / 'outputs/metadata/embeddings-existing'
            cache.mkdir(parents=True)
            (cache / 'artifact.json').write_text('{}')
            pending = unconfigured_models(root, {'dinov3': 'embeddings-dino'})
            self.assertEqual(set(pending), {'layoutlmv3-ocr', 'another-model'})
            self.assertEqual(set(pending.values()), {'No embedding artifact is configured for clustering.'})
            self.assertEqual(unconfigured_models(root, {
                'dinov3': 'embeddings-dino', 'layoutlmv3-ocr': 'embeddings-existing',
                'another-model': 'embeddings-other'}), {})


if __name__ == '__main__':
    unittest.main()
