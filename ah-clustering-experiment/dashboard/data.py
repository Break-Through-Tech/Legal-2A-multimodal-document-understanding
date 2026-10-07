"""Read-only, checksummed dashboard boundary; never imports experiment runners."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.artifacts import read_artifact
from src.contracts import document_ids, portable_path, require
from src.dataset import file_sha256, owned_path, validate_storage_root


def _aligned(frame: pd.DataFrame, expected_ids, name: str) -> pd.DataFrame:
    expected = document_ids(expected_ids)
    actual = document_ids(frame.document_id)
    require(set(actual) == set(expected), f'{name} document IDs differ from cohort')
    return frame.set_index('document_id').loc[expected].reset_index()


def _counts(frame, summary, name):
    labels = frame.cluster_id.to_numpy()
    require(labels.dtype.kind in 'iu' and (labels >= -1).all(), f'{name} has invalid cluster labels')
    require(summary['document_count'] == len(frame), f'{name} document count differs')
    require(summary['noise_count'] == int((labels == -1).sum()), f'{name} noise count differs')
    require(summary['cluster_count'] == len(set(labels) - {-1}), f'{name} cluster count differs')


def load_run(root: Path, artifact_id: str) -> dict:
    """Validate metrics and upstream assignments, aligning every join by document ID."""
    try:
        bundle = read_artifact(root, artifact_id)
        require(bundle['manifest']['kind'] == 'metrics', 'Expected a metrics artifact')
        require(set(bundle['tables']) == {'documents', 'clusters'} and not bundle['arrays'], 'Unexpected metrics payloads')
        config = bundle['config']
        expected = config['cohort']['document_ids']
        documents = _aligned(bundle['tables']['documents'], expected, 'Metrics')
        cluster = read_artifact(root, config['clustering_artifact'])
        require(cluster['manifest']['kind'] == 'clusters', 'Expected a clustering artifact')
        require(cluster['manifest']['manifest_sha256'] == config['clustering_manifest_sha256'], 'Clustering manifest differs')
        require(cluster['config']['cohort'] == config['cohort'], 'Clustering cohort differs')
        assignments = _aligned(cluster['tables']['assignments'], expected, 'Assignments')
        require(np.array_equal(documents.cluster_id, assignments.cluster_id), 'Evaluation cluster labels differ from assignments')
        summary, scores = bundle['json']['summary'], bundle['json']['scores']
        _counts(assignments, cluster['json']['summary'], 'Clustering')
        _counts(documents, summary, 'Metrics')
        require(documents.is_noise.dtype.kind == 'b' and (documents.is_noise == (documents.cluster_id == -1)).all(), 'Noise flags differ')
        require(np.isclose(summary['noise_percentage'], 100 * documents.is_noise.mean(), atol=1e-10), 'Noise percentage differs')
        for key in ('algorithm', 'model', 'representation'):
            require(summary[key] == cluster['config'][key], f'Summary {key} differs from clustering')
        require(config['input']['cohort'] == config['cohort'], 'Evaluation input cohort differs')
        for key in ('representation', 'embedding_artifact', 'embedding_identity', 'embedding_manifest_sha256', 'normalization', 'pooling', 'token_selection'):
            if key in cluster['config']:
                require(config['input'].get(key) == cluster['config'][key], f'Evaluation input {key} differs')
        clusters = bundle['tables']['clusters'].copy()
        require(not clusters.cluster_id.isna().any() and not clusters.cluster_id.duplicated().any(), 'Duplicate or null cluster IDs')
        require(set(clusters.cluster_id) == set(documents.cluster_id), 'Cluster table coverage differs')
        for row in clusters.to_dict('records'):
            members = documents.loc[documents.cluster_id == row['cluster_id']]
            require(row['size'] == len(members), 'Cluster size differs')
            require(row['is_noise'] == (row['cluster_id'] == -1), 'Cluster noise flag differs')
            counts = members.label.dropna().value_counts().to_dict()
            saved_counts = {key: value for key, value in row['label_counts'].items() if value is not None and not pd.isna(value) and value != 0}
            require(saved_counts == counts, 'Cluster label distribution differs')
            require(row['labeled_document_count'] == sum(counts.values()), 'Cluster labeled count differs')
            if counts:
                require(np.isclose(row['purity'], max(counts.values()) / sum(counts.values())), 'Cluster purity differs')
                require(set(row['dominant_labels']) == {label for label, count in counts.items() if count == max(counts.values())}, 'Dominant labels differ')
            else:
                require(pd.isna(row['purity']), 'Unlabeled cluster purity must be unavailable')
            for name, rank in [('representative_ids', 'representative_rank'), ('outlier_ids', 'outlier_rank')]:
                examples = list(row[name])
                require(len(set(examples)) == len(examples) and set(examples) <= set(members.document_id), 'Invalid example IDs')
                ranked = members.loc[members[rank].notna()].sort_values(rank)
                require(ranked.document_id.tolist() == examples and ranked[rank].tolist() == list(range(1, len(examples) + 1)), 'Example ranks differ')
                require(row['cluster_id'] != -1 or not examples, 'Noise cannot have cluster examples')
        require(summary['cluster_sizes'] == {str(row['cluster_id']): row['size'] for row in clusters.to_dict('records')}, 'Cluster size summary differs')
        require(set(scores) == {'including_noise', 'excluding_noise'}, 'Missing score noise scope')
        for name, scope in scores.items():
            selected = documents if name == 'including_noise' else documents.loc[~documents.is_noise]
            labeled = int(selected.label.notna().sum())
            require(scope['document_count'] == len(selected) and np.isclose(scope['coverage'], len(selected) / len(documents)), 'Score scope coverage differs')
            require(scope['labeled_document_count'] == labeled and np.isclose(scope['labeled_coverage'], labeled / len(selected) if len(selected) else 0), 'Labeled score coverage differs')
            require(bool(scope['metrics']), 'Missing metrics')
            for result in scope['metrics'].values():
                value, reason = result['value'], result['reason']
                require((value is None and isinstance(reason, str) and bool(reason)) or
                        (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and reason is None), 'Metric needs finite value or explicit reason')
        require(summary['projections'] == config['projections'], 'Projection references differ')
        return {'artifact_id': artifact_id, 'config': config, 'manifest': bundle['manifest'],
                'documents': documents, 'clusters': clusters, 'scores': scores, 'summary': summary,
                'cluster_config': cluster['config'], 'cluster_manifest': cluster['manifest'],
                'projections': config['projections']}
    except (KeyError, TypeError, OSError, AttributeError) as error:
        raise ValueError(f'Invalid dashboard run {artifact_id}: {error}') from error


def load_projection(root: Path, projection_ref: dict, expected_ids) -> pd.DataFrame:
    """Load saved coordinates only; a missing projection leaves its metrics usable."""
    try:
        require(projection_ref.get('status') == 'complete', projection_ref.get('reason', 'Projection is incomplete'))
        bundle = read_artifact(root, projection_ref['artifact_id'])
        require(bundle['manifest']['kind'] == 'projections', 'Expected projection artifact')
        require(bundle['manifest']['manifest_sha256'] == projection_ref['manifest_sha256'], 'Projection manifest differs')
        require(bundle['config']['method'] == projection_ref['method'], 'Projection method differs')
        require(set(bundle['tables']) == {'coordinates'} and not bundle['arrays'], 'Unexpected projection payloads')
        require(list(bundle['tables']['coordinates'].columns) == ['document_id', 'x', 'y'], 'Invalid coordinate schema')
        frame = _aligned(bundle['tables']['coordinates'], expected_ids, 'Projection')
        require(document_ids(bundle['config']['input']['cohort']['document_ids']) == document_ids(expected_ids), 'Projection cohort order differs')
        require(np.isfinite(frame[['x', 'y']].to_numpy()).all(), 'Nonfinite projection coordinates')
        return frame
    except (KeyError, TypeError, OSError, AttributeError) as error:
        raise ValueError(f'Invalid projection: {error}') from error


def image_bytes(root: Path, row) -> bytes:
    """Check the owned original image, decode fully, then return captured PNG bytes."""
    try:
        root = validate_storage_root(root)
        portable_path(row['image_path'])
        path = owned_path(root, row['image_path'])
        digest = row.get('image_sha256')
        if digest is not None and not pd.isna(digest):
            require(file_sha256(path) == digest, 'Image checksum mismatch')
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as original:
            original.load()
            output = io.BytesIO()
            original.convert('RGB').save(output, format='PNG')
        return output.getvalue()
    except (OSError, KeyError, TypeError) as error:
        raise ValueError(f'Image unavailable: {error}') from error


def scan_catalog(root: Path) -> dict:
    """Discover saved metrics and expose relevant missing, incomplete and failed work."""
    root = validate_storage_root(root)
    issues, runs = [], []
    settings_path = owned_path(root, 'configs/evaluation.json')
    try:
        settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
        require(isinstance(settings, dict), 'Evaluation settings must be an object')
        configured_runs = settings.get('runs', [])
        require(isinstance(configured_runs, list) and all(isinstance(item, dict)
                and isinstance(item.get('artifact_id'), str) and isinstance(item.get('model'), str)
                for item in configured_runs), 'Evaluation runs must list artifact IDs and models')
        pending = settings.get('pending', {})
        require(isinstance(pending, dict) and all(isinstance(model, str) and isinstance(reason, str)
                for model, reason in pending.items()), 'Pending evaluations must map model names to reasons')
        unavailable = settings.get('unavailable', [])
        require(isinstance(unavailable, list) and all(isinstance(item, dict)
                and all(isinstance(item.get(key), str) for key in ('model', 'algorithm', 'reason'))
                for item in unavailable), 'Unavailable evaluations need model, algorithm and reason')
    except (ValueError, OSError) as error:
        settings = {}
        issues.append({'status': 'invalid', 'artifact_id': 'configs/evaluation.json', 'model': 'unknown', 'reason': str(error)})
    expected = {item['artifact_id']: item['model'] for item in settings.get('runs', [])}
    encountered, successful = set(), set()
    metadata = owned_path(root, 'outputs/metadata')
    for directory in sorted(metadata.glob('metrics-*')):
        cluster_id, model = None, 'unknown'
        try:
            config_path = owned_path(root, f'outputs/metadata/{directory.name}/config.json')
            config = json.loads(config_path.read_text(encoding='utf-8'))
            cluster_id = config.get('clustering_artifact')
            model = expected.get(cluster_id, 'unknown')
            # The current matrix owns the dashboard; historical pilot cohorts are separate work.
            if expected and cluster_id not in expected:
                continue
            encountered.add(cluster_id)
            run = load_run(root, directory.name)
            cluster_config = run['cluster_config']
            model = expected.get(cluster_id, cluster_config['model'].get('checkpoint', 'unknown'))
            successful.add(cluster_id)
            runs.append({'artifact_id': directory.name, 'clustering_artifact': cluster_id, 'model': model,
                         'algorithm': cluster_config['algorithm'], 'representation': cluster_config['representation'],
                         'cohort': cluster_config['cohort'].get('cohort', 'custom'), 'seed': cluster_config.get('seed'),
                         'document_count': run['summary']['document_count'], 'cluster_count': run['summary']['cluster_count'],
                         'noise_percentage': run['summary']['noise_percentage'],
                         'clustering_seconds': run['cluster_manifest']['provenance']['runtime']['elapsed_seconds'],
                         'evaluation_seconds': run['manifest']['provenance']['runtime']['elapsed_seconds']})
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
            issues.append({'status': 'incomplete' if not (directory / 'artifact.json').exists() else 'invalid',
                           'artifact_id': directory.name, 'model': model, 'reason': str(error)})
    for cluster_id, model in expected.items():
        if cluster_id not in encountered:
            issues.append({'status': 'pending', 'artifact_id': cluster_id, 'model': model, 'reason': 'No saved evaluation artifact for this configured clustering run.'})
    # Last attempts supersede older failures; successful publications supersede all attempt logs.
    latest, model_failures = {}, {}
    logs = owned_path(root, 'outputs/logs')
    for path in sorted([*logs.glob('step6-*.json'), *logs.glob('step7-*.json')]):
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            for entry in report.get('results', report.get('runs', [])):
                cluster_id = entry.get('clustering_artifact', entry.get('artifact_id'))
                if cluster_id in expected:
                    latest[cluster_id] = entry
                elif entry.get('model') in expected.values() and entry.get('status') == 'failed':
                    model_failures[entry['model']] = entry
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    for cluster_id, entry in latest.items():
        if entry.get('status') == 'failed' and cluster_id not in successful:
            issues = [item for item in issues if not (item['artifact_id'] == cluster_id and item['status'] == 'pending')]
            issues.append({'status': 'failed', 'artifact_id': cluster_id, 'model': expected[cluster_id], 'reason': entry.get('error', 'Recorded experiment failure')})
    for model, entry in model_failures.items():
        if any(cluster_id not in successful for cluster_id, name in expected.items() if name == model):
            issues = [item for item in issues if not (item['model'] == model and item['status'] == 'pending' and item['artifact_id'] in expected)]
            issues.append({'status': 'failed', 'artifact_id': None, 'model': model, 'reason': entry.get('error', 'Recorded experiment failure')})
    for model, reason in settings.get('pending', {}).items():
        model_runs = {cluster_id for cluster_id, name in expected.items() if name == model}
        if not model_runs or not model_runs <= successful:
            issues.append({'status': 'pending', 'artifact_id': None, 'model': model, 'reason': reason})
    for entry in settings.get('unavailable', []):
        issues.append({'status': 'unavailable', 'artifact_id': None, 'model': entry['model'], 'reason': f"{entry['algorithm']}: {entry['reason']}"})
    unique = {tuple(item.values()): item for item in issues}
    return {'runs': runs, 'issues': list(unique.values())}
