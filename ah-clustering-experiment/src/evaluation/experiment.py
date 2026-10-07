"""Publish evaluation and shared projections without fitting any clusters."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..artifacts import ArtifactWriter, artifact_identity, read_artifact
from ..clustering.experiment import load_cohort, validate_assignments, _similarity_bundle
from ..clustering.runner import validate_matrix
from ..contracts import document_ids, require
from ..dataset import file_sha256, fingerprint, owned_path
from ..embeddings import _decode
from ..provenance import capture_provenance, source_identity
from .metrics import evaluate
from .projections import project


def load_labels(root: Path, source_root: Path, config: dict) -> pd.DataFrame:
    """Return the exact clustering cohort, checking original split/label authority."""
    expected = config['cohort']
    directory = owned_path(root, f"outputs/metadata/{expected['dataset_id']}")
    cohort, _ = load_cohort(root, directory, expected['cohort'])
    require(cohort == expected, 'Evaluation cohort differs from clustering')
    metadata = json.loads((directory / 'dataset.json').read_text())
    shared_path = source_root.parent / 'data/split.csv'
    require(file_sha256(shared_path) == metadata['identity']['shared_csv_sha256'],
            'Shared CSV changed since dataset preparation')
    shared = pd.read_csv(shared_path).rename(columns={'id': 'document_id'}).set_index('document_id')
    frame = pd.read_parquet(directory / 'manifest.parquet').set_index('document_id').loc[cohort['document_ids']].reset_index()
    for column in ('label', 'split', 'original_label', 'document_quality'):
        require(frame[column].tolist() == shared.loc[frame.document_id, column].tolist(),
                f'Shared CSV {column} differs from metadata')
    return frame


def load_representation(root: Path, config: dict, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    ids = document_ids(frame.document_id)
    require(ids == config['cohort']['document_ids'], 'Representation ID order differs')
    base = {key: config[key] for key in ('embedding_artifact', 'embedding_identity',
            'embedding_manifest_sha256', 'representation', 'token_selection', 'pooling', 'normalization', 'cohort')}
    if config['similarity'] is not None:
        similarity = read_artifact(root, config['similarity']['artifact_id'])
        _similarity_bundle(similarity)
        require(similarity['manifest']['identity_sha256'] == config['similarity']['identity_sha256'],
                'Similarity identity differs from clustering')
        require(similarity['config']['cohort'] == config['cohort']
                and similarity['config']['embedding_identity'] == config['embedding_identity']
                and similarity['config']['embedding_artifact'] == config['embedding_artifact']
                and similarity['config']['embedding_manifest_sha256'] == config['embedding_manifest_sha256'],
                'Similarity upstream identity differs')
        base.update(precomputed=True, similarity=config['similarity'],
                    similarity_manifest_sha256=similarity['manifest']['manifest_sha256'],
                    distance='1 - symmetric mean-MaxSim; nonmetric dissimilarity')
        data = np.asarray(similarity['arrays']['distance'], dtype=np.float64)
    else:
        embedding = read_artifact(root, config['embedding_artifact'])
        require(embedding['manifest']['identity_sha256'] == config['embedding_identity']
                and embedding['manifest']['manifest_sha256'] == config['embedding_manifest_sha256'],
                'Embedding identity differs from clustering')
        require(config['representation'] == 'cls' and config['pooling'] == {'kind': 'none'},
                'Unsupported fixed representation')
        expected_images = frame[['document_id', 'image_path', 'image_sha256']].to_dict('records')
        images = {row['document_id']: row for row in embedding['config']['documents']}
        require(all(images.get(row['document_id']) == row for row in expected_images), 'Embedding image identity differs')
        native = _decode(embedding)
        require(all(native[key]['tensors']['cls'].shape[0] == 1 for key in ids), 'Expected one CLS vector per document')
        data = np.vstack([native[key]['tensors']['cls'][0] for key in ids]).astype(np.float64)
        normalization = config['normalization']['kind']
        require(normalization in {'none', 'l2'}, 'Unsupported CLS normalization')
        if normalization == 'l2':
            lengths = np.linalg.norm(data, axis=1, keepdims=True)
            require((lengths > 0).all(), 'Cannot normalize zero CLS vector')
            data /= lengths
        base.update(precomputed=False, distance='euclidean')
    validate_matrix(data, len(ids), precomputed=base['precomputed'])
    return data, base


def validate_projection(bundle: dict) -> None:
    require(bundle['manifest']['kind'] == 'projections', 'Expected projection artifact')
    require(set(bundle['tables']) == {'coordinates'} and not bundle['arrays'], 'Unexpected projection payloads')
    frame = bundle['tables']['coordinates']
    require(list(frame.columns) == ['document_id', 'x', 'y'], 'Invalid projection schema')
    require(document_ids(frame.document_id) == bundle['config']['input']['cohort']['document_ids'], 'Projection ID order differs')
    require(np.isfinite(frame[['x', 'y']].to_numpy()).all(), 'Nonfinite projection')
    require(bundle['json']['details']['repeat_coordinates_equal'] is True, 'Projection repeat check missing')


def save_projection(root, source_root, data, input_config, method, parameters, seed):
    if method == 'pca' and not input_config['precomputed']:
        if data.shape[1] < 2 or float(np.var(data, axis=0).sum()) == 0:
            return {'method': method, 'status': 'unavailable',
                    'reason': 'Two-dimensional PCA requires at least two feature dimensions and positive total variance.'}
    # Algorithm choices, labels and metric options do not invalidate coordinates.
    config = {'input': input_config, 'method': method, 'parameters': parameters, 'seed': seed,
              'implementation_files_sha256': {name: file_sha256(source_root / name) for name in (
                  'src/evaluation/projections.py', 'src/evaluation/experiment.py', 'src/artifacts.py',
                  'src/contracts.py', 'src/dataset.py', 'src/embeddings/__init__.py')},
              'dependencies': {name: importlib.metadata.version(name) for name in
                  ('numpy', 'scipy', 'scikit-learn', 'umap-learn', 'numba', 'pynndescent')}}
    artifact_id, _ = artifact_identity('projections', config)
    marker = owned_path(root, f'outputs/metadata/{artifact_id}/artifact.json')
    reused = marker.exists()
    if not reused:
        started = time.perf_counter()
        with ArtifactWriter(root, 'projections', config) as writer:
            coordinates, details = project(data, method, precomputed=input_config['precomputed'], seed=seed, parameters=parameters)
            repeated, _ = project(data, method, precomputed=input_config['precomputed'], seed=seed, parameters=parameters)
            require(np.allclose(coordinates, repeated, atol=1e-5, rtol=0), 'Repeated projection differs')
            details.update(repeat_coordinates_equal=True, repeat_absolute_tolerance=1e-5,
                           repeat_max_error=float(np.max(np.abs(coordinates - repeated))),
                           purpose='visualization only; not clustering features')
            writer.table('coordinates', pd.DataFrame({'document_id': input_config['cohort']['document_ids'],
                         'x': coordinates[:, 0], 'y': coordinates[:, 1]}))
            writer.json('details', details)
            writer.complete(capture_provenance(source_root, started, details), validator=validate_projection)
    bundle = read_artifact(root, artifact_id, expected_config=config)
    validate_projection(bundle)
    return {'artifact_id': artifact_id, 'manifest_sha256': bundle['manifest']['manifest_sha256'],
            'method': method, 'reused': reused, 'status': 'complete'}


def validate_evaluation(bundle: dict) -> None:
    require(bundle['manifest']['kind'] == 'metrics', 'Expected metrics artifact')
    require(set(bundle['tables']) == {'documents', 'clusters'} and not bundle['arrays'], 'Unexpected evaluation payloads')
    frame = bundle['tables']['documents']
    config = bundle['config']
    require(document_ids(frame.document_id) == config['cohort']['document_ids'], 'Evaluation ID order differs')
    labels = frame.cluster_id.to_numpy()
    require(labels.dtype.kind in 'iu' and (labels >= -1).all(), 'Invalid evaluation assignments')
    require((frame.is_noise == (labels == -1)).all(), 'Noise flags differ')
    summary = bundle['json']['summary']
    require(summary['document_count'] == len(frame) and summary['noise_count'] == int((labels == -1).sum()), 'Evaluation counts differ')
    require(summary['cluster_count'] == len(set(labels) - {-1}), 'Cluster count differs')
    clusters = bundle['tables']['clusters']
    require(set(clusters.cluster_id) == set(labels), 'Cluster table coverage differs')
    for row in clusters.to_dict('records'):
        ids = set(frame.loc[frame.cluster_id == row['cluster_id'], 'document_id'])
        require(row['size'] == len(ids), 'Cluster size differs')
        for name in ('representative_ids', 'outlier_ids'):
            require(isinstance(row[name], (list, np.ndarray)) and np.ndim(row[name]) == 1,
                    'Example IDs must be a sequence')
            require(set(row[name]).issubset(ids), 'Example points outside cluster')
            require(len(set(row[name])) == len(row[name]), 'Duplicate example IDs')
            rank_column = 'representative_rank' if name == 'representative_ids' else 'outlier_rank'
            ranked = frame.loc[(frame.cluster_id == row['cluster_id']) & frame[rank_column].notna()].sort_values(rank_column)
            require(ranked.document_id.tolist() == list(row[name])
                    and ranked[rank_column].tolist() == list(range(1, len(row[name]) + 1)), 'Example ranks differ')
        if row['cluster_id'] == -1:
            require(len(row['representative_ids']) == len(row['outlier_ids']) == 0, 'Noise has cluster examples')
    require(set(bundle['json']['scores']) == {'including_noise', 'excluding_noise'}, 'Missing noise score scope')
    for name, scope in bundle['json']['scores'].items():
        selected = frame if name == 'including_noise' else frame.loc[~frame.is_noise]
        labeled_count = int(selected.label.notna().sum())
        require(scope['document_count'] == len(selected) and scope['coverage'] == len(selected)/len(frame),
                'Score scope count or coverage differs')
        require(scope['labeled_document_count'] == labeled_count
                and scope['labeled_coverage'] == (labeled_count/len(selected) if len(selected) else 0),
                'Labeled coverage differs')
        for result in scope['metrics'].values():
            require((result['value'] is None and bool(result['reason']))
                    or (isinstance(result['value'], (float, int)) and np.isfinite(result['value']) and result['reason'] is None),
                    'Metric must have a finite value or explicit reason')


def save_evaluation(root, source_root, cluster, frame, data, input_config, projections, settings):
    started = time.perf_counter()
    config = {'clustering_artifact': cluster['manifest']['artifact_id'],
              'clustering_manifest_sha256': cluster['manifest']['manifest_sha256'],
              'cohort': cluster['config']['cohort'], 'input': input_config,
              'projections': [{key: value for key, value in item.items() if key != 'reused'} for item in projections],
              'policy': {'noise_scopes': ['including_noise', 'excluding_noise'],
                         'examples_per_cluster': settings['examples_per_cluster'],
                         'representatives': 'medoid then nearest; ties by document ID',
                         'outliers': 'farthest from medoid; ties by document ID',
                         'label_source': 'checksummed dataset manifest verified against shared CSV'},
              'source_digest': source_identity(source_root)['source_digest'],
              'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}}
    artifact_id, _ = artifact_identity('metrics', config)
    reused = owned_path(root, f'outputs/metadata/{artifact_id}/artifact.json').exists()
    if not reused:
        ids = frame.document_id.tolist()
        labels = cluster['tables']['assignments'].cluster_id.to_numpy()
        result = evaluate(ids, frame.label.tolist(), labels, data, precomputed=input_config['precomputed'],
                          examples_per_cluster=settings['examples_per_cluster'])
        documents = pd.DataFrame(result['documents'])
        require(documents.document_id.tolist() == ids, 'Evaluator changed document order')
        require(np.array_equal(documents.cluster_id.to_numpy(), labels), 'Evaluator changed assignments')
        extras = frame.drop(columns=['label'])
        documents = documents.merge(extras, on='document_id', validate='one_to_one', sort=False)
        summary = {'document_count': len(ids), 'cluster_count': int(len(set(labels) - {-1})),
                   'noise_count': int((labels == -1).sum()), 'noise_percentage': float(100 * (labels == -1).mean()),
                   'cluster_sizes': {str(row['cluster_id']): row['size'] for row in result['clusters']},
                   'distance': input_config['distance'], 'representation': input_config['representation'],
                   'model': cluster['config']['model'], 'algorithm': cluster['config']['algorithm'],
                   'cluster_count_policy': cluster['config']['cluster_count_policy'],
                   'policy': result['policy'], 'projections': config['projections']}
        with ArtifactWriter(root, 'metrics', config) as writer:
            writer.table('documents', documents)
            writer.table('clusters', pd.DataFrame(result['clusters']), unique_key='cluster_id')
            writer.json('scores', result['scores'])
            writer.json('summary', summary)
            writer.complete(capture_provenance(source_root, started, summary), validator=validate_evaluation)
    bundle = read_artifact(root, artifact_id, expected_config=config)
    validate_evaluation(bundle)
    return {'artifact_id': artifact_id, 'reused': reused, 'clustering_artifact': cluster['manifest']['artifact_id'],
            'algorithm': cluster['config']['algorithm'], 'document_count': len(frame), 'status': 'complete',
            'scores': bundle['json']['scores'], 'elapsed_seconds': time.perf_counter() - started}


def run_evaluation(root, source_root, entries, settings):
    """One owner per representation; output algorithms reuse the same projections."""
    representations = {}
    results, projection_results = [], {}
    for entry in entries:
        try:
            cluster = read_artifact(root, entry['artifact_id'])
            validate_assignments(cluster)
            frame = load_labels(root, source_root, cluster['config'])
            key = fingerprint({key: cluster['config'][key] for key in ('embedding_artifact', 'embedding_identity',
                'embedding_manifest_sha256', 'representation', 'token_selection', 'pooling', 'normalization', 'cohort', 'similarity')})
            if key not in representations:
                data, input_config = load_representation(root, cluster['config'], frame)
                projections = []
                for method in (['umap'] if input_config['precomputed'] else ['pca', 'umap']):
                    projection = save_projection(root, source_root, data, input_config, method,
                                                 settings['projections'][method], settings['seed'])
                    projections.append(projection)
                    projection_results[projection.get('artifact_id', f'{key}-{method}')] = {'model': entry['model'], **projection}
                    print(json.dumps({'projection': projection.get('artifact_id'), 'model': entry['model'], **projection}), flush=True)
                representations[key] = data, input_config, projections
            data, input_config, projections = representations[key]
            result = save_evaluation(root, source_root, cluster, frame, data, input_config, projections, settings)
            result['model'] = entry['model']
        except Exception as error:
            result = {'model': entry['model'], 'clustering_artifact': entry['artifact_id'], 'status': 'failed',
                      'error_type': type(error).__name__, 'error': str(error)}
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != 'scores'}), flush=True)
    return results, list(projection_results.values())
