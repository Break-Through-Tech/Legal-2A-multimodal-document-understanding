"""Actual tiny artifact publications exercise the dashboard's read-only boundary."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.data import image_bytes, load_projection, load_run, scan_catalog
from src.artifacts import ArtifactWriter, read_artifact
from src.dataset import file_sha256, fingerprint


class DashboardDataChecks(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'cache/test-dashboard'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'ah-clustering-experiment'
        self.provenance = {'source': {'source_digest': fingerprint({}), 'source_files_sha256': {},
            'git_commit': None, 'repository_dirty': None, 'experiment_dirty': None, 'git_status': None},
            'dependencies': {'fixture': '1'}, 'runtime': {'elapsed_seconds': 1.0}, 'hardware': {'fixture': True}, 'measurements': {}}
        self.cohort = {'document_ids': [2, 7, 11], 'cohort': 'train'}
        self.cluster_config = {'cohort': self.cohort, 'algorithm': 'fixture', 'model': {'checkpoint': 'saved fixture'},
                               'representation': 'cls', 'seed': 42}
        self.assignments = pd.DataFrame({'document_id': [11, 2, 7], 'cluster_id': [-1, 0, 0]})
        counts = {'document_count': 3, 'cluster_count': 1, 'noise_count': 1}
        with ArtifactWriter(self.root, 'clusters', self.cluster_config) as writer:
            writer.table('assignments', self.assignments)
            writer.json('summary', counts)
            writer.complete(self.provenance)
        cluster = read_artifact(self.root, writer.artifact_id)
        self.cluster_id = writer.artifact_id
        self.config = {'clustering_artifact': self.cluster_id,
            'clustering_manifest_sha256': cluster['manifest']['manifest_sha256'], 'cohort': self.cohort,
            'input': {'cohort': self.cohort, 'representation': 'cls'}, 'projections': []}
        self.documents = pd.DataFrame({'document_id': [7, 11, 2], 'cluster_id': [0, -1, 0],
            'label': ['A', 'B', 'A'], 'is_noise': [False, True, False],
            'representative_rank': [None, None, 1], 'outlier_rank': [1, None, None]})
        self.clusters = pd.DataFrame([
            {'cluster_id': 0, 'size': 2, 'is_noise': False, 'labeled_document_count': 2, 'purity': 1.,
             'label_counts': {'A': 2}, 'dominant_labels': ['A'], 'representative_ids': [2], 'outlier_ids': [7]},
            {'cluster_id': -1, 'size': 1, 'is_noise': True, 'labeled_document_count': 1, 'purity': 1.,
             'label_counts': {'B': 1}, 'dominant_labels': ['B'], 'representative_ids': [], 'outlier_ids': []}])
        self.summary = {**counts, 'noise_percentage': 100/3, 'algorithm': 'fixture',
            'model': self.cluster_config['model'], 'representation': 'cls', 'cluster_sizes': {'0': 2, '-1': 1}, 'projections': []}
        self.scores = {name: {'document_count': size, 'coverage': size/3, 'labeled_document_count': size,
            'labeled_coverage': 1., 'metrics': {'ari': {'value': 1., 'reason': None},
                'silhouette': {'value': None, 'reason': 'Fixture does not compute distances.'}}}
            for name, size in [('including_noise', 3), ('excluding_noise', 2)]}
        (self.root / 'configs').mkdir()
        (self.root / 'configs/evaluation.json').write_text(json.dumps({'runs': [{'artifact_id': self.cluster_id, 'model': 'fixture'}],
            'pending': {'layoutlmv3-ocr': 'Missing full embeddings'},
            'unavailable': [{'model': 'colpali', 'algorithm': 'kmeans', 'reason': 'Needs fixed vectors'}]}))

    def publish(self):
        with ArtifactWriter(self.root, 'metrics', self.config) as writer:
            writer.table('documents', self.documents, unique_key=None)
            writer.table('clusters', self.clusters, unique_key=None)
            writer.json('summary', self.summary)
            writer.json('scores', self.scores)
            writer.complete(self.provenance)
        return writer.artifact_id

    def test_joins_permuted_rows_by_id_and_catalog_reports_pending(self):
        artifact = self.publish()
        result = load_run(self.root, artifact)
        self.assertEqual(result['documents'].document_id.tolist(), [2, 7, 11])
        self.assertEqual(result['documents'].cluster_id.tolist(), [0, 0, -1])
        catalog = scan_catalog(self.root)
        self.assertEqual(len(catalog['runs']), 1)
        self.assertEqual(catalog['runs'][0]['model'], 'fixture')
        self.assertEqual({item['status'] for item in catalog['issues']}, {'pending', 'unavailable'})
        self.assertIn({'status': 'pending', 'artifact_id': None, 'model': 'layoutlmv3-ocr',
                       'reason': 'Missing full embeddings'}, catalog['issues'])

    def test_completed_configured_runs_supersede_model_pending(self):
        settings_path = self.root / 'configs/evaluation.json'
        settings = json.loads(settings_path.read_text())
        settings['pending']['fixture'] = 'Missing full embeddings'
        settings_path.write_text(json.dumps(settings))
        self.publish()
        catalog = scan_catalog(self.root)
        self.assertEqual(len(catalog['runs']), 1)
        self.assertFalse(any(item['model'] == 'fixture' and item['status'] == 'pending'
                             for item in catalog['issues']))
        self.assertTrue(any(item['model'] == 'layoutlmv3-ocr' and item['status'] == 'pending'
                            for item in catalog['issues']))

    def test_partial_configured_runs_retain_model_pending(self):
        with ArtifactWriter(self.root, 'clusters', {**self.cluster_config, 'seed': 43}) as writer:
            writer.table('assignments', self.assignments)
            writer.json('summary', {'document_count': 3, 'cluster_count': 1, 'noise_count': 1})
            writer.complete(self.provenance)
        settings_path = self.root / 'configs/evaluation.json'
        settings = json.loads(settings_path.read_text())
        settings['runs'].append({'artifact_id': writer.artifact_id, 'model': 'fixture'})
        settings['pending']['fixture'] = 'Missing full embeddings'
        settings_path.write_text(json.dumps(settings))
        self.publish()
        catalog = scan_catalog(self.root)
        self.assertEqual(len(catalog['runs']), 1)
        self.assertIn({'status': 'pending', 'artifact_id': None, 'model': 'fixture',
                       'reason': 'Missing full embeddings'}, catalog['issues'])
        self.assertTrue(any(item['artifact_id'] == writer.artifact_id and item['status'] == 'pending'
                            for item in catalog['issues']))

    def test_invalid_evaluation_retains_model_pending(self):
        settings_path = self.root / 'configs/evaluation.json'
        settings = json.loads(settings_path.read_text())
        settings['pending']['fixture'] = 'Missing full embeddings'
        settings_path.write_text(json.dumps(settings))
        artifact = self.publish()
        (self.root / f'outputs/metrics/{artifact}/scores.json').write_text('{}')
        catalog = scan_catalog(self.root)
        self.assertEqual(catalog['runs'], [])
        self.assertIn({'status': 'pending', 'artifact_id': None, 'model': 'fixture',
                       'reason': 'Missing full embeddings'}, catalog['issues'])
        self.assertTrue(any(item['status'] == 'invalid' and 'checksum' in item['reason']
                            for item in catalog['issues']))

    def test_wrong_assignment_labels_rejected(self):
        self.documents.loc[0, 'cluster_id'] = 1
        with self.assertRaisesRegex(ValueError, 'cluster labels differ'):
            load_run(self.root, self.publish())

    def test_malformed_settings_are_an_issue_not_dashboard_failure(self):
        for settings in [[], {'runs': [{}]}, {'pending': []}, {'unavailable': ['bad']}]:
            with self.subTest(settings=settings):
                (self.root / 'configs/evaluation.json').write_text(json.dumps(settings))
                catalog = scan_catalog(self.root)
                self.assertTrue(any(item['status'] == 'invalid' and item['artifact_id'] == 'configs/evaluation.json'
                                    for item in catalog['issues']))

    def test_duplicate_document_ids_rejected(self):
        self.documents.loc[0, 'document_id'] = 11
        with self.assertRaisesRegex(ValueError, 'Duplicate document'):
            load_run(self.root, self.publish())

    def test_wrong_cohort_membership_rejected(self):
        self.documents.loc[0, 'document_id'] = 99
        with self.assertRaisesRegex(ValueError, 'document IDs differ'):
            load_run(self.root, self.publish())

    def test_wrong_noise_coverage_rejected(self):
        self.scores['excluding_noise']['coverage'] = 1.
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            load_run(self.root, self.publish())

    def test_example_from_other_cluster_rejected(self):
        self.clusters.at[0, 'representative_ids'] = [11]
        with self.assertRaisesRegex(ValueError, 'example IDs'):
            load_run(self.root, self.publish())

    def test_corrupt_payload_is_separate_issue(self):
        artifact = self.publish()
        path = self.root / f'outputs/metrics/{artifact}/scores.json'
        path.write_text('{}')
        catalog = scan_catalog(self.root)
        self.assertEqual(catalog['runs'], [])
        self.assertTrue(any(item['status'] == 'invalid' and 'checksum' in item['reason'] for item in catalog['issues']))

    def test_missing_completion_marker_is_separate_issue(self):
        artifact = self.publish()
        (self.root / f'outputs/metadata/{artifact}/artifact.json').unlink()
        catalog = scan_catalog(self.root)
        self.assertEqual(catalog['runs'], [])
        self.assertTrue(any(item['status'] == 'incomplete' for item in catalog['issues']))

    def test_failed_current_model_log_and_completed_supersession(self):
        logs = self.root / 'outputs/logs'
        (logs / 'step6-100.json').write_text(json.dumps({'results': [
            {'model': 'fixture', 'status': 'failed', 'error': 'Saved upstream failure'}]}))
        catalog = scan_catalog(self.root)
        self.assertTrue(any(item['status'] == 'failed' for item in catalog['issues']))
        self.publish()
        self.assertFalse(any(item['status'] == 'failed' for item in scan_catalog(self.root)['issues']))

    def test_missing_projection_does_not_hide_valid_metrics(self):
        ref = {'artifact_id': 'projections-' + 'a'*20, 'manifest_sha256': 'b'*64, 'method': 'pca', 'status': 'complete'}
        self.config['projections'] = [ref]
        self.summary['projections'] = [ref]
        artifact = self.publish()
        self.assertEqual(len(load_run(self.root, artifact)['documents']), 3)
        with self.assertRaises(ValueError):
            load_projection(self.root, ref, [2, 7, 11])

    def test_projection_rows_align_and_mismatched_ids_fail(self):
        config = {'input': {'cohort': self.cohort}, 'method': 'pca'}
        with ArtifactWriter(self.root, 'projections', config) as writer:
            writer.table('coordinates', pd.DataFrame({'document_id': [11, 2, 7], 'x': [3., 1., 2.], 'y': [6., 4., 5.]}))
            manifest = writer.complete(self.provenance)
        ref = {'artifact_id': writer.artifact_id, 'manifest_sha256': manifest['manifest_sha256'], 'method': 'pca', 'status': 'complete'}
        result = load_projection(self.root, ref, [2, 7, 11])
        self.assertEqual(result.x.tolist(), [1., 2., 3.])
        with self.assertRaisesRegex(ValueError, 'document IDs differ'):
            load_projection(self.root, ref, [2, 7, 99])
        ref['manifest_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            load_projection(self.root, ref, [2, 7, 11])

    def test_image_decode_checksum_missing_and_escape(self):
        path = self.root / 'sample.png'
        Image.new('RGB', (4, 5), 'red').save(path)
        row = {'image_path': 'sample.png', 'image_sha256': file_sha256(path)}
        self.assertTrue(image_bytes(self.root, row).startswith(b'\x89PNG'))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            image_bytes(self.root, {**row, 'image_sha256': '0'*64})
        for bad in [{'image_path': '../sample.png'}, {'image_path': 'missing.png'}]:
            with self.assertRaises(ValueError):
                image_bytes(self.root, bad)
        path.write_text('not an image')
        with self.assertRaises(ValueError):
            image_bytes(self.root, {'image_path': 'sample.png'})


if __name__ == '__main__':
    unittest.main()
