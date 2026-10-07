"""Saved-coordinate exploration and cluster-level document review."""

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import PALETTE, document_panel, plot_style, projection_data


def embedding_view(root, run):
    st.title('Embedding explorer')
    st.write('Explore a saved projection. Select a point or choose a document to inspect its original image.')
    references = run['projections']
    available = [ref for ref in references if ref.get('status', 'complete') == 'complete']
    for ref in references:
        if ref.get('status', 'complete') != 'complete':
            st.info(f"{ref['method'].upper()} unavailable: {ref.get('reason', 'No completed projection')}")
    if not available:
        st.warning('No completed projection is available for this run.')
        return
    controls = st.columns(3)
    ref = controls[0].selectbox('Projection', available, format_func=lambda item: item['method'].upper(), key='projection_method')
    color = controls[1].selectbox('Color by', ['Ground truth', 'Cluster', 'Source', 'Document quality', 'Review status'], key='projection_color')
    include_noise = controls[2].checkbox('Show noise documents', True, key='explorer_noise')
    docs = run['documents'].copy()
    selected_labels = st.multiselect('Ground-truth labels', sorted(docs.label.dropna().unique()),
                                    placeholder='All labels', key='explorer_labels')
    try:
        coordinates = projection_data(root, ref, docs.document_id.tolist())
        docs = docs.merge(coordinates, on='document_id', validate='one_to_one', sort=False)
    except (ValueError, OSError) as error:
        st.warning(f'Saved projection unavailable: {error}')
        return
    docs['Cluster'] = docs.cluster_id.map(lambda value: 'Noise (−1)' if value == -1 else f'Cluster {value}')
    docs['Ground truth'] = docs.label.fillna('Label unavailable')
    docs['Run'] = run['config']['clustering_artifact']
    docs['Algorithm'] = run['cluster_config']['algorithm']
    docs['Checkpoint'] = run['cluster_config']['model']['checkpoint']
    docs['Source'] = 'getomni-ai/ocr-benchmark'
    docs['Document quality'] = docs.document_quality.fillna('Unavailable')
    docs['Review status'] = ['Noise' if noise else ('Majority-label mismatch' if mismatch else 'Matches dominant label')
                             for noise, mismatch in zip(docs.is_noise, docs.majority_label_mismatch)]
    if selected_labels:
        docs = docs.loc[docs.label.isin(selected_labels)]
    if not include_noise:
        docs = docs.loc[~docs.is_noise]
    st.caption(f'{len(docs)} of {len(run["documents"])} documents shown · coordinates are for visualization only; saved clustering does not change.')
    if color == 'Source':
        st.caption('Source denotes the recorded benchmark dataset. Template or organization sources are not available.')
    if color == 'Review status':
        st.caption('A majority-label mismatch is a review candidate, not a proven label error. Unknown is an ordinary benchmark label.')
    if docs.empty:
        st.info('No documents match these filters. Include noise or clear the label filter.')
        return
    chart, preview = st.columns([1.7, 1], gap='large')
    signature = hashlib.sha256(str((run['artifact_id'], ref['artifact_id'], color, docs.document_id.tolist())).encode()).hexdigest()[:12]
    picker_key = f'explorer_document_{run["artifact_id"]}'
    ids = docs.document_id.tolist()
    if st.session_state.get(picker_key) not in ids:
        st.session_state[picker_key] = ids[0]
    with chart:
        figure = px.scatter(docs, x='x', y='y', color=color, custom_data=['document_id'],
                            hover_name='document_id', hover_data=['Ground truth', 'Cluster', 'Document quality', 'Algorithm', 'Checkpoint', 'Run'],
                            color_discrete_sequence=PALETTE)
        figure.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.4, color='#FFFFFF')))
        figure.update_layout(dragmode='select', clickmode='event+select')
        figure.update_xaxes(title=f'{ref["method"].upper()} 1')
        figure.update_yaxes(title=f'{ref["method"].upper()} 2')
        event = st.plotly_chart(plot_style(figure, 540), width='stretch', on_select='rerun',
                                selection_mode=('points', 'box', 'lasso'), key=f'projection_chart_{signature}',
                                config={'displaylogo': False, 'scrollZoom': False}, alt='Document projection colored by '+color.lower())
        points = event.selection.points
        if points:
            document_id = int(points[0]['customdata'][0])
            token = (signature, document_id)
            if st.session_state.get('last_plot_selection') != token and document_id in ids:
                st.session_state[picker_key] = document_id
                st.session_state['last_plot_selection'] = token
        st.caption('For a box or lasso selection, the first selected document opens on the right. Use the picker to inspect any visible document.')
        st.download_button('Download visible coordinates', docs[['document_id', 'x', 'y', 'label', 'cluster_id']].to_csv(index=False),
                           'projection-documents.csv', 'text/csv')
    with preview:
        document_id = st.selectbox('Document', ids, key=picker_key)
        document_panel(root, run['documents'], document_id)


def cluster_view(root, run):
    st.title('Cluster explorer')
    st.write('Inspect label distributions, medoid representatives, and candidate outliers.')
    clusters = run['clusters'].sort_values('cluster_id')
    cluster_id = st.selectbox('Cluster', clusters.cluster_id.tolist(),
                             format_func=lambda value: 'Noise (−1)' if value == -1 else f'Cluster {value}', key='cluster_choice')
    row = clusters.loc[clusters.cluster_id == cluster_id].iloc[0]
    members = run['documents'].loc[run['documents'].cluster_id == cluster_id]
    cols = st.columns(3)
    cols[0].metric('Documents', int(row['size']))
    cols[1].metric('Purity', 'Unavailable' if pd.isna(row.purity) else f'{row.purity:.1%}')
    cols[2].metric('Majority-label review candidates', int(members.majority_label_mismatch.sum()))
    st.caption('Dominant label(s): ' + ', '.join(row.dominant_labels))
    counts = pd.DataFrame([{'Label': key, 'Documents': value} for key, value in row.label_counts.items() if value is not None and value > 0])
    if not counts.empty:
        fig = px.bar(counts.sort_values('Documents'), x='Documents', y='Label', orientation='h', color_discrete_sequence=PALETTE)
        st.plotly_chart(plot_style(fig, min(450, max(200, 26 * len(counts)))), width='stretch',
                        config={'displaylogo': False}, key='cluster_labels', alt='Ground-truth label counts in this cluster')
    if cluster_id == -1:
        st.info('These documents were marked as noise. They have no cluster representative or majority-label mismatch designation.')
    else:
        st.caption('Representatives: medoid, then nearest members. Candidate outliers: farthest from that medoid. Ties use document ID; small clusters may have overlapping lists.')
        a, b = st.columns(2)
        a.write('**Representative documents**')
        a.write(', '.join(map(str, row.representative_ids)))
        b.write('**Candidate outlier documents**')
        b.write(', '.join(map(str, row.outlier_ids)))
    selection = st.radio('Show documents', ['All members', 'Representatives', 'Candidate outliers', 'Majority-label mismatches'],
                         horizontal=True, key='cluster_subset')
    subset = members
    if selection == 'Representatives':
        subset = members.loc[members.representative_rank.notna()].sort_values('representative_rank')
    elif selection == 'Candidate outliers':
        subset = members.loc[members.outlier_rank.notna()].sort_values('outlier_rank')
    elif selection == 'Majority-label mismatches':
        subset = members.loc[members.majority_label_mismatch]
    left, right = st.columns([1.25, 1], gap='large')
    with left:
        st.dataframe(subset[['document_id', 'label', 'distance_to_representative', 'representative_rank', 'outlier_rank']],
                     hide_index=True, width='stretch')
        with st.expander('All cluster sizes and purity'):
            st.dataframe(clusters[['cluster_id', 'size', 'purity', 'is_noise']], hide_index=True, width='stretch')
    with right:
        if subset.empty:
            st.info('No documents in this selection. Choose All members to browse this group.')
        else:
            document_id = st.selectbox('Cluster document', subset.document_id.tolist(), key=f'cluster_doc_{run["artifact_id"]}_{cluster_id}_{selection}')
            document_panel(root, run['documents'], document_id)
