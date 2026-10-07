"""Restore the committed dashboard results, or export a new verified snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / 'dashboard' / 'saved-results.zip'
INDEX = BUNDLE.with_suffix('.json')
ALLOWED = re.compile(r'outputs/(?:metadata|metrics|clusters|projections)/(?:metrics|clusters|projections)-[0-9a-f]{20}/[a-z][a-z0-9_-]*\.(?:json|parquet)')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(root=ROOT, bundle=BUNDLE, index=INDEX):
    """Validate the whole snapshot and conflicts before writing any files."""
    from src.dataset import _atomic_bytes, owned_path, validate_storage_root

    root = validate_storage_root(root)
    metadata = json.loads(index.read_text(encoding='utf-8'))
    if metadata['schema_version'] != 1 or digest(bundle.read_bytes()) != metadata['archive_sha256']:
        raise ValueError('Saved-results archive checksum or version differs')
    pending = []
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(metadata['files']):
            raise ValueError('Archive file list differs from the snapshot manifest')
        for name in names:
            if not ALLOWED.fullmatch(name) or '..' in PurePosixPath(name).parts:
                raise ValueError(f'Unexpected snapshot path: {name}')
            path = owned_path(root, name)
            payload = archive.read(name)
            expected = metadata['files'][name]
            if len(payload) != expected['bytes'] or digest(payload) != expected['sha256']:
                raise ValueError(f'Snapshot payload checksum differs: {name}')
            if path.exists():
                if digest(path.read_bytes()) != expected['sha256']:
                    raise ValueError(f'Existing file differs; refusing to overwrite: {path}')
            else:
                pending.append((path, payload))
    # Publish completion markers last so interrupted restoration is retryable.
    pending.sort(key=lambda item: item[0].name == 'artifact.json')
    for path, payload in pending:
        _atomic_bytes(path, payload)
    return {'written_files': len(pending), 'runs': metadata['runs'], 'projections': metadata['projections']}


def export(root=ROOT, bundle=BUNDLE, index=INDEX):
    """Collect only verified UI dependencies, without images or model caches."""
    from dashboard.data import load_projection, load_run, scan_catalog
    from src.artifacts import read_artifact
    from src.dataset import owned_path

    catalog = scan_catalog(root)
    if not catalog['runs']:
        raise ValueError('No completed dashboard runs to export')
    artifact_ids, projection_ids = set(), set()
    for row in catalog['runs']:
        run = load_run(root, row['artifact_id'])
        artifact_ids.update([row['artifact_id'], row['clustering_artifact']])
        for ref in run['projections']:
            if ref['status'] == 'complete':
                load_projection(root, ref, run['documents'].document_id)
                projection_ids.add(ref['artifact_id'])
    artifact_ids.update(projection_ids)
    paths = set()
    for artifact_id in sorted(artifact_ids):
        artifact = read_artifact(root, artifact_id)
        paths.update(item['path'] for item in artifact['manifest']['files'].values())
        paths.update(f'outputs/metadata/{artifact_id}/{name}' for name in ('config.json', 'artifact.json'))
    files = {}
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(paths):
            if not ALLOWED.fullmatch(name):
                raise ValueError(f'Unexpected dashboard dependency: {name}')
            payload = owned_path(root, name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
            files[name] = {'bytes': len(payload), 'sha256': digest(payload)}
    metadata = {'schema_version': 1, 'archive_sha256': digest(bundle.read_bytes()),
                'runs': len(catalog['runs']), 'projections': len(projection_ids),
                'description': 'Saved training-cohort dashboard results. Images are downloaded separately from the pinned upstream dataset. No embeddings or model weights.',
                'files': files}
    index.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return {'archive_bytes': bundle.stat().st_size, 'files': len(files),
            'runs': metadata['runs'], 'projections': metadata['projections']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true', help='Maintainer: replace the committed snapshot from current verified results')
    args = parser.parse_args()
    try:
        result = export() if args.export else restore()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Dashboard snapshot failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
