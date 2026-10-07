"""Aligned partition comparison and saved experiment provenance."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.comparison import compare_runs
from dashboard.ui import document_panel, plot_style, run_data, run_name, score_table


def comparison_view(root, run, rows, active_id):
    st.title('Experiment comparison')
    st.write('Align document IDs and cluster numbering before inspecting disagreements.')
    other_rows = [row for row in rows if row['artifact_id'] != active_id] or rows
    selected = st.selectbox('Compare against', other_rows, format_func=run_name, key='comparison_run')
    right = run_data(root, selected['artifact_id'])
    try:
        result = compare_runs(run, right)
    except ValueError as error:
        st.warning(f'These runs cannot be compared: {error}')
        return
    summary = result['summary']
    columns = st.columns(3)
    fraction = summary['agreement_fraction']
    columns[0].metric('Aligned agreement', 'Unavailable' if fraction is None else f'{fraction:.1%}')
    columns[1].metric('Comparison coverage', f"{summary['comparison_coverage']:.1%}")
    columns[2].metric('Disagreements', summary['disagreement_count'])
    st.caption(f"Agreement uses {summary['compared_count']} documents assigned to a non-noise cluster in both runs. "
               f"Both noise: {summary['both_noise_count']}; noise only in active run: {summary['left_noise_count']}; "
               f"noise only in comparison: {summary['right_noise_count']}.")
    st.caption('Cluster numbers are aligned by a one-to-one maximum-overlap assignment. Unmatched groups disagree; noise is never mapped to a regular cluster.')
    table = result['contingency']
    if not table.empty:
        matrix = table.pivot(index='left_cluster', columns='right_cluster', values='count').fillna(0)
        figure = px.imshow(matrix, aspect='auto', color_continuous_scale='Teal',
                            labels={'x': 'Comparison cluster (original number)', 'y': 'Active cluster', 'color': 'Shared documents'})
        st.plotly_chart(plot_style(figure, 390), width='stretch', config={'displaylogo': False},
                        key='comparison_heatmap', alt='Cluster overlap counts before number alignment')
    else:
        st.info('No document belongs to a regular cluster in both runs; aligned agreement is undefined.')
    documents = result['documents']
    states = st.multiselect('Comparison statuses', sorted(documents.status.unique()),
                            default=[value for value in ['disagreement', 'left_noise', 'right_noise'] if value in set(documents.status)],
                            key='comparison_status')
    shown = documents.loc[documents.status.isin(states)] if states else documents
    st.caption(f'{len(shown)} document records shown. Clear the status filter to show all documents.')
    a, b = st.columns([1.3, 1], gap='large')
    with a:
        st.dataframe(shown, hide_index=True, width='stretch')
        with st.expander('Cluster-number alignment'):
            st.dataframe(pd.DataFrame(result['mapping']), hide_index=True, width='stretch')
    with b:
        if not shown.empty:
            document_id = st.selectbox('Comparison document', shown.document_id.tolist(), key=f'comparison_doc_{active_id}_{selected["artifact_id"]}_{str(states)}')
            document_panel(root, run['documents'], document_id)


def details_view(run, scope):
    st.title('Experiment details')
    st.write('Inspect the exact checkpoint, representation, parameters and provenance attached to this saved result.')
    config = run['cluster_config']
    st.success('Completed artifact · payload checksums and document mappings validated')
    st.code(run['artifact_id'], language=None)
    st.header('Checkpoint and representation')
    st.json({key: config[key] for key in ('model', 'representation', 'token_selection', 'pooling', 'normalization',
                                         'algorithm', 'parameters', 'seed', 'cluster_count_policy')})
    st.header('Metric values and availability')
    scores = run['scores'][scope]
    st.caption(f"{scores['noise_policy']} Coverage: {scores['coverage']:.1%}; labeled coverage: {scores['labeled_coverage']:.1%}.")
    st.dataframe(score_table(run, scope), hide_index=True, width='stretch')
    with st.expander('Similarity definition and projection references'):
        st.json({'similarity': config['similarity'], 'projections': run['projections']})
    with st.expander('Cohort and dataset identity'):
        st.json(config['cohort'])
    with st.expander('Clustering execution provenance'):
        st.json(run['cluster_manifest']['provenance'])
    with st.expander('Evaluation execution provenance'):
        st.json(run['manifest']['provenance'])
    with st.expander('Evaluation policy and artifact identity'):
        st.json(run['config'])
        st.json(run['summary']['policy'])
