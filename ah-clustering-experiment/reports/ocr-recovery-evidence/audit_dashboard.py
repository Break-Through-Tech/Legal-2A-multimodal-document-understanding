"""Verify the portable snapshot and OCR views in Streamlit's Python test runner."""

import json
from pathlib import Path
import shutil
import sys
import tempfile

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard.data import scan_catalog
from experiments.dashboard_bundle import restore


def main():
    current = scan_catalog(ROOT)
    assert len(current['runs']) == 18
    assert len(current['issues']) == 2
    assert {item['model'] for item in current['issues']} == {'colpali', 'colqwen2'}
    assert all(item['status'] == 'unavailable' for item in current['issues'])
    with tempfile.TemporaryDirectory(dir=ROOT / 'cache') as directory:
        restored_root = Path(directory) / 'ah-clustering-experiment'
        restored = restore(restored_root)
        assert (restored['runs'], restored['projections']) == (18, 8)
        (restored_root / 'configs').mkdir()
        shutil.copyfile(ROOT / 'configs/evaluation.json', restored_root / 'configs/evaluation.json')
        catalog = scan_catalog(restored_root)
        assert {r['artifact_id'] for r in catalog['runs']} == {r['artifact_id'] for r in current['runs']}
        assert catalog['issues'] == current['issues']
        assert restore(restored_root)['written_files'] == 0

    app = AppTest.from_file(str(ROOT / 'dashboard/app.py'), default_timeout=90).run()
    checks = ['portable snapshot restores 18 runs and 8 projections', 'snapshot restore is idempotent']

    def clean(name):
        assert not app.exception, [entry.message for entry in app.exception]
        assert not app.error, [entry.value for entry in app.error]
        checks.append(name)

    clean('overview')
    assert next(m.value for m in app.metric if m.label == 'Completed runs') == '18'
    assert next(m.value for m in app.metric if m.label == 'Pending / unavailable') == '2'
    app.multiselect(key='models').set_value(['layoutlmv3-ocr']).run()
    clean('OCR filter')
    assert next(m.value for m in app.metric if m.label == 'Completed runs') == '4'
    app.radio(key='page').set_value('Embedding explorer').run()
    clean('OCR embedding explorer')
    selector = next(widget for widget in app.selectbox if widget.label == 'Projection')
    selector.select('UMAP').run()
    clean('OCR UMAP')
    selector = next(widget for widget in app.selectbox if widget.label == 'Projection')
    selector.select('PCA').run()
    clean('OCR PCA')
    app.radio(key='page').set_value('Cluster explorer').run()
    clean('OCR cluster explorer')
    app.radio(key='page').set_value('Experiment details').run()
    clean('OCR details')
    assert not any(name in sys.modules for name in ('torch', 'transformers', 'sklearn', 'umap'))
    result = {'status': 'pass', 'surface': 'Streamlit AppTest; browser rendering not tested',
              'runs': 18, 'projections': 8, 'issues': current['issues'], 'checks': checks,
              'inference_and_fitting_modules_loaded': False}
    (Path(__file__).parent / 'dashboard-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
