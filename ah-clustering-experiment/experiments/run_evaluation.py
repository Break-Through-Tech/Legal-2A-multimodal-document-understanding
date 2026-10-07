"""Evaluate saved Step 6 assignments and cache Step 7 visualization inputs."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class NoEncoderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'src.embeddings.models' or fullname.split('.')[0] in {'transformers', 'colpali_engine', 'tensorflow'}:
            raise ImportError('Evaluation consumes saved artifacts; encoder loading is disabled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation.json')
    parser.add_argument('--models', nargs='+', help='Subset of configured model names')
    args = parser.parse_args()
    sys.meta_path.insert(0, NoEncoderImports())
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    from src.dataset import _atomic_json, configure_cache, owned_path, validate_storage_root
    root = validate_storage_root(args.storage_root)
    configure_cache(root)
    os.environ['NUMBA_CACHE_DIR'] = str(root / 'cache/numba')
    os.environ['MPLCONFIGDIR'] = str(root / 'cache/matplotlib')
    from src.evaluation.experiment import run_evaluation
    from src.provenance import source_identity
    from threadpoolctl import threadpool_limits
    settings = json.loads(args.config.read_text())
    entries = settings['runs']
    if len({row['artifact_id'] for row in entries}) != len(entries):
        parser.error('Duplicate clustering artifact IDs')
    if args.models:
        if len(args.models) != len(set(args.models)) or not set(args.models).issubset({row['model'] for row in entries}):
            parser.error('Select unique configured model names')
        entries = [row for row in entries if row['model'] in args.models]
    source = source_identity(ROOT)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        results, projections = run_evaluation(root, ROOT, entries, settings)
    summary = {'source_digest': source['source_digest'], 'settings': settings,
               'encoder_imports_blocked': True, 'compute': 'local CPU', 'clustering_refit': False,
               'results': results, 'projections': projections, 'pending': settings['pending'],
               'unavailable': settings['unavailable'], 'elapsed_seconds': time.perf_counter() - started}
    if source_identity(ROOT)['source_digest'] != source['source_digest']:
        summary['source_changed_during_run'] = True
    report = owned_path(root, f'outputs/logs/step7-{time.time_ns()}.json')
    _atomic_json(report, summary)
    print(json.dumps({'report': str(report), 'completed': sum(row['status'] == 'complete' for row in results),
                      'failed': sum(row['status'] == 'failed' for row in results), 'projections': len(projections)}), flush=True)
    if summary.get('source_changed_during_run') or any(row['status'] == 'failed' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
