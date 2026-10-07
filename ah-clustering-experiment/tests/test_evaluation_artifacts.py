"""Real publication/reuse boundaries; sentinels forbid recomputation only."""

import copy
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.artifacts import ArtifactWriter, read_artifact
from src.evaluation.experiment import save_evaluation, save_projection, validate_evaluation, validate_projection
from src.provenance import capture_provenance


class EvaluationArtifactChecks(unittest.TestCase):
    def setUp(self):
        cache = ROOT / 'cache/test-evaluation'
        cache.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'ah-clustering-experiment'
        self.source = Path(temporary.name) / 'source'
        for name in ('src/evaluation/projections.py', 'src/evaluation/experiment.py', 'src/artifacts.py',
                     'src/contracts.py', 'src/dataset.py', 'src/embeddings/__init__.py'):
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.ids = [3, 7, 11, 19]
        self.data = np.array([[0., 0.], [2., 0.], [8., 0.], [10., 0.]])
        self.input = {'cohort': {'document_ids': self.ids}, 'precomputed': False,
                      'distance': 'euclidean', 'representation': 'cls'}
        self.frame = pd.DataFrame({'document_id': self.ids, 'label': ['A', 'A', 'B', 'B'],
                                   'image_path': [f'outputs/images/{i}.png' for i in self.ids]})
        cluster_config = {'cohort': self.input['cohort'], 'algorithm': 'fixture partition',
                          'model': {'checkpoint': 'literal geometry'},
                          'cluster_count_policy': {'mode': 'independent fixture'}}
        with ArtifactWriter(self.root, 'clusters', cluster_config) as writer:
            writer.table('assignments', pd.DataFrame({'document_id': self.ids, 'cluster_id': [0, 0, 1, 1]}))
            writer.json('summary', {'document_count': 4, 'noise_count': 0, 'cluster_count': 2})
            writer.complete(capture_provenance(self.source, time.perf_counter()))
        self.cluster = read_artifact(self.root, writer.artifact_id)

    def test_scores_and_examples_roundtrip_and_reuse_without_computation(self):
        first = save_evaluation(self.root, self.source, self.cluster, self.frame, self.data, self.input, [], {'examples_per_cluster': 1})
        bundle = read_artifact(self.root, first['artifact_id'])
        self.assertAlmostEqual(bundle['json']['scores']['excluding_noise']['metrics']['silhouette']['value'], 47/63)
        self.assertEqual(bundle['tables']['documents'].document_id.tolist(), self.ids)
        self.assertEqual(bundle['tables']['clusters'].representative_ids.map(list).tolist(), [[3], [11]])
        with patch('src.evaluation.experiment.evaluate', side_effect=AssertionError('Recomputed metrics')):
            second = save_evaluation(self.root, self.source, self.cluster, self.frame, self.data, self.input, [], {'examples_per_cluster': 1})
        self.assertTrue(second['reused'])
        self.assertEqual(first['artifact_id'], second['artifact_id'])
        changed = save_evaluation(self.root, self.source, self.cluster, self.frame, self.data, self.input, [], {'examples_per_cluster': 2})
        self.assertNotEqual(first['artifact_id'], changed['artifact_id'])

    def test_projection_reuse_after_unrelated_cluster_setting_edit(self):
        first = save_projection(self.root, self.source, self.data, self.input, 'pca', {}, 42)
        (self.source / 'configs').mkdir()
        (self.source / 'configs/clustering.json').write_text('{"n_clusters":3}')
        with patch('src.evaluation.experiment.project', side_effect=AssertionError('Recomputed coordinates')):
            second = save_projection(self.root, self.source, self.data, self.input, 'pca', {}, 42)
        self.assertTrue(second['reused'])
        self.assertEqual(first['artifact_id'], second['artifact_id'])
        bundle = read_artifact(self.root, first['artifact_id'])
        bundle['tables']['coordinates'] = bundle['tables']['coordinates'].iloc[::-1]
        with self.assertRaisesRegex(ValueError, 'ID order'):
            validate_projection(bundle)

    def test_wrong_example_membership_and_document_order_rejected(self):
        saved = save_evaluation(self.root, self.source, self.cluster, self.frame, self.data, self.input, [], {'examples_per_cluster': 1})
        bundle = read_artifact(self.root, saved['artifact_id'])
        bad = copy.deepcopy(bundle)
        bad['tables']['documents'] = bad['tables']['documents'].iloc[::-1]
        with self.assertRaisesRegex(ValueError, 'ID order'):
            validate_evaluation(bad)
        bad = copy.deepcopy(bundle)
        bad['tables']['clusters']['representative_ids'] = pd.Series([[19], [11]], dtype=object)
        with self.assertRaisesRegex(ValueError, 'outside cluster'):
            validate_evaluation(bad)
        bad = copy.deepcopy(bundle)
        bad['json']['scores']['excluding_noise']['coverage'] = 0.5
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            validate_evaluation(bad)
        bad = copy.deepcopy(bundle)
        bad['tables']['documents'].loc[0, 'representative_rank'] = 3
        with self.assertRaisesRegex(ValueError, 'ranks differ'):
            validate_evaluation(bad)

    def test_undefined_pca_does_not_discard_metrics(self):
        projection = save_projection(self.root, self.source, self.data[:, :1], self.input, 'pca', {}, 42)
        self.assertEqual(projection['status'], 'unavailable')
        saved = save_evaluation(self.root, self.source, self.cluster, self.frame, self.data[:, :1], self.input,
                                [projection], {'examples_per_cluster': 1})
        self.assertEqual(saved['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
