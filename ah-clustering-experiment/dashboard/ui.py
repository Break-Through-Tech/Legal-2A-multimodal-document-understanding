"""Shared display helpers. All values come from saved artifacts."""

import pandas as pd
import streamlit as st

from dashboard.data import image_bytes, load_projection, load_run, scan_catalog

MODEL_NAMES = {'dinov3': 'DINOv3', 'layoutlmv3-image': 'LayoutLMv3 · image',
               'layoutlmv3-ocr': 'LayoutLMv3 · OCR', 'colpali': 'ColPali', 'colqwen2': 'ColQwen2'}
ALGORITHM_NAMES = {'kmeans': 'K-means', 'agglomerative': 'Agglomerative',
                   'hdbscan': 'HDBSCAN', 'spectral': 'Spectral'}
METRICS = {'ari': 'ARI', 'nmi': 'NMI', 'ami': 'AMI', 'homogeneity': 'Homogeneity',
           'completeness': 'Completeness', 'v_measure': 'V-measure', 'purity': 'Purity',
           'silhouette': 'Silhouette', 'davies_bouldin': 'Davies–Bouldin',
           'calinski_harabasz': 'Calinski–Harabasz'}
PALETTE = ['#147D73', '#D08835', '#546CA3', '#C56561', '#8B72A5', '#568C58', '#B58B58',
           '#5D9DA5', '#AC658D', '#8D954B', '#356F9B', '#CD975C', '#737373', '#9D5046',
           '#488C89', '#7A82AF', '#A67A42', '#727D47', '#AD799C', '#55716F']


@st.cache_data(ttl=30, show_spinner='Checking saved experiment files…')
def catalog(root):
    return scan_catalog(root)


@st.cache_data(ttl=30, show_spinner=False)
def run_data(root, artifact_id):
    return load_run(root, artifact_id)


@st.cache_data(ttl=30, show_spinner=False)
def projection_data(root, reference, ids):
    return load_projection(root, reference, ids)


def run_name(row):
    return f"{MODEL_NAMES.get(row['model'], row['model'])} / {ALGORITHM_NAMES.get(row['algorithm'], row['algorithm'])} · {row['artifact_id'][-6:]}"


def score_table(run, scope):
    return pd.DataFrame([{'Metric': METRICS.get(name, name), 'Value': value['value'],
                          'Unavailable reason': value['reason'] or ''}
                         for name, value in run['scores'][scope]['metrics'].items()])


def document_panel(root, documents, document_id):
    row = documents.set_index('document_id').loc[int(document_id)]
    st.header(f'Document {int(document_id)}')
    cluster = 'Noise (−1)' if row.cluster_id == -1 else f'Cluster {int(row.cluster_id)}'
    st.caption(f"{row.label} · {cluster} · {row.get('document_quality', 'Quality unavailable')}")
    if row.get('majority_label_mismatch', False):
        st.info('Review candidate: its label differs from the cluster’s dominant label(s). This is not proof of an error.')
    try:
        content = image_bytes(root, row.to_dict())
        st.image(content, caption=f"Original image · document {int(document_id)}", width='stretch')
    except (ValueError, OSError) as error:
        st.warning(f'Original image unavailable: {error}')
    with st.expander('Document record'):
        st.json({'document_id': int(document_id), 'label': row.label, 'cluster_id': int(row.cluster_id),
                 'split': row.get('split'), 'original_label': row.get('original_label'),
                 'document_quality': row.get('document_quality'), 'image_path': row.get('image_path')})


def plot_style(figure, height=440):
    figure.update_layout(height=height, margin=dict(l=12, r=12, t=20, b=12),
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(family='Arial, sans-serif', color='#253A39'),
                         legend=dict(title_text='', orientation='h', y=-0.15),
                         xaxis=dict(gridcolor='#E6EBE8'), yaxis=dict(gridcolor='#E6EBE8'))
    return figure
