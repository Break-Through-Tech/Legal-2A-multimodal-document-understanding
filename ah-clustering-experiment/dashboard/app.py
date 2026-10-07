"""Document Atlas: read-only Streamlit dashboard for saved clustering results."""

from __future__ import annotations

import importlib.abc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SavedResultsOnly(importlib.abc.MetaPathFinder):
    """The dashboard cannot import any inference, fitting or projection runner."""
    dashboard_guard = True

    def find_spec(self, fullname, path=None, target=None):
        blocked = ('src.embeddings', 'src.clustering', 'src.evaluation')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked) or fullname.split('.')[0] in {
                'torch', 'transformers', 'colpali_engine', 'tensorflow', 'umap', 'sklearn'}:
            raise ImportError('Dashboard reads saved results only; inference and fitting imports are disabled')


if not any(getattr(finder, 'dashboard_guard', False) for finder in sys.meta_path):
    sys.meta_path.insert(0, SavedResultsOnly())

import streamlit as st
from src.dataset import validate_storage_root
from dashboard.ui import MODEL_NAMES, catalog, run_data, run_name
from dashboard.overview import render as overview_view
from dashboard.explorer import embedding_view, cluster_view
from dashboard.review import comparison_view, details_view


def main():
    st.set_page_config(page_title='Document Atlas · Clustering', page_icon='▦', layout='wide', initial_sidebar_state='expanded')
    st.html('''<style>
        :root { --atlas-ink: #253A39; --atlas-accent: #147D73; }
        .stApp { background: #FAFBF8; color: var(--atlas-ink); }
        [data-testid="stSidebar"] { background: #EDF2EC; }
        [data-testid="stMetricValue"] { color: var(--atlas-accent); }
        h1,h2,h3 { color: var(--atlas-ink); letter-spacing: -0.025em; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        </style>''')
    try:
        root = validate_storage_root(os.environ.get('AH_CLUSTERING_STORAGE_ROOT', str(ROOT)))
    except ValueError as error:
        st.error(str(error))
        return
    with st.sidebar:
        st.caption('LEGAL · DOCUMENT UNDERSTANDING')
        st.header('Document Atlas')
        st.caption('Saved clustering experiments')
        page = st.radio('View', ['Overview', 'Embedding explorer', 'Cluster explorer', 'Experiment comparison', 'Experiment details'], key='page')
        st.divider()
        if st.button('Refresh saved results', width='stretch'):
            st.cache_data.clear()
        st.caption('Read-only · No model or clustering jobs')
    try:
        saved = catalog(root)
    except (ValueError, OSError) as error:
        st.error(f'Saved results unavailable: {error}')
        return
    if not saved['runs']:
        st.title('No completed evaluations')
        st.info('Run Step 7 to publish evaluation artifacts, then refresh saved results.')
        if saved['issues']:
            st.dataframe(saved['issues'], hide_index=True, width='stretch')
        return
    all_rows = sorted(saved['runs'], key=lambda row: (row['model'], row['algorithm'], row['artifact_id']))
    with st.sidebar:
        models = st.multiselect('Input modes', sorted({row['model'] for row in all_rows}),
                               default=sorted({row['model'] for row in all_rows}), format_func=lambda value: MODEL_NAMES.get(value, value), key='models')
        rows = [row for row in all_rows if row['model'] in models]
        if not rows:
            st.info('Select at least one input mode.')
            return
        options = [row['artifact_id'] for row in rows]
        lookup = {row['artifact_id']: row for row in rows}
        if st.session_state.get('active_run') not in options:
            preferred = next((row['artifact_id'] for row in rows if row['model'] == 'dinov3' and row['algorithm'] == 'kmeans'), options[0])
            st.session_state['active_run'] = preferred
        active_id = st.selectbox('Active run', options, format_func=lambda key: run_name(lookup[key]), key='active_run')
        scope_label = st.radio('Metric noise policy', ['Include noise', 'Exclude noise'], key='noise_policy')
        scope = 'including_noise' if scope_label == 'Include noise' else 'excluding_noise'
        st.caption('Excluding noise changes the scored cohort; coverage is shown alongside scores.')
        with st.expander('Storage location'):
            st.code(str(root), language=None)
    try:
        if page == 'Overview':
            overview_view(root, rows, saved['issues'], scope)
            return
        run = run_data(root, active_id)
        st.caption(run_name(lookup[active_id]) + f" · {len(run['documents'])} saved documents · {run['cluster_config']['cohort']['cohort']} cohort")
        if page == 'Embedding explorer':
            embedding_view(root, run)
        elif page == 'Cluster explorer':
            cluster_view(root, run)
        elif page == 'Experiment comparison':
            comparison_view(root, run, rows, active_id)
        else:
            details_view(run, scope)
    except (ValueError, OSError) as error:
        st.error(f'Saved run unavailable: {error}')
        st.info('Refresh saved results to recheck available artifacts.')


if __name__ == '__main__':
    main()
