"""Cross-run scores, coverage and visible unavailable results."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import ALGORITHM_NAMES, METRICS, MODEL_NAMES, PALETTE, plot_style, run_data


def render(root, rows, issues, scope):
    st.title('Experiment overview')
    st.write('Compare the saved baselines, then inspect the documents behind each cluster.')
    columns = st.columns(4)
    columns[0].metric('Completed runs', len(rows))
    columns[1].metric('Input modes', len({row['model'] for row in rows}))
    counts = sorted({row['document_count'] for row in rows})
    columns[2].metric('Documents per run', ' / '.join(map(str, counts)))
    columns[3].metric('Pending / unavailable', len(issues))
    st.caption('Training-cohort results. Fixed-count algorithms use 19 class-count-informed groups; HDBSCAN discovers its group count.')
    records, reasons = [], []
    for row in rows:
        run = run_data(root, row['artifact_id'])
        scores = run['scores'][scope]
        record = {'Model': MODEL_NAMES.get(row['model'], row['model']),
                  'Algorithm': ALGORITHM_NAMES.get(row['algorithm'], row['algorithm']),
                  'Representation': row['representation'], 'Cohort': row['cohort'], 'Seed': row['seed'],
                  'Documents': row['document_count'], 'Clusters': row['cluster_count'],
                  'Noise %': row['noise_percentage'], 'Coverage %': scores['coverage'] * 100,
                  'Clustering seconds': row['clustering_seconds'], 'Evaluation seconds': row['evaluation_seconds'],
                  'Artifact': row['artifact_id']}
        for name, result in scores['metrics'].items():
            record[METRICS.get(name, name)] = result['value']
            if result['reason']:
                reasons.append({'Model': record['Model'], 'Algorithm': record['Algorithm'],
                                'Metric': METRICS.get(name, name), 'Reason': result['reason']})
        records.append(record)
    frame = pd.DataFrame(records)
    metric = st.selectbox('Compare metric', list(METRICS.values()), key='overview_metric')
    st.caption(('Lower is better.' if metric == 'Davies–Bouldin' else 'Higher is better.')
               + ' Scores describe the selected noise policy; excluding noise can change coverage substantially.')
    available = frame.loc[frame[metric].notna()]
    if available.empty:
        st.info('This metric is unavailable for the selected runs. Reasons appear below.')
    else:
        figure = px.bar(available, x='Model', y=metric, color='Algorithm', barmode='group',
                        color_discrete_sequence=PALETTE, hover_data=['Coverage %', 'Clusters', 'Noise %'])
        st.plotly_chart(plot_style(figure, 360), width='stretch', config={'displaylogo': False},
                        key='overview_chart', alt=f'{metric} by model and clustering algorithm')
    st.header('Saved run scores')
    st.dataframe(frame[['Model', 'Algorithm', metric, 'Coverage %', 'Noise %', 'Clusters', 'Cohort', 'Seed']],
                 hide_index=True, width='stretch')
    with st.expander('All metrics, representations and runtime'):
        st.dataframe(frame, hide_index=True, width='stretch')
        st.caption('Runtime is the observed saved invocation, not a controlled throughput benchmark. Internal geometry scores depend on representation.')
        st.download_button('Download displayed scores', frame.to_csv(index=False), 'clustering-scores.csv', 'text/csv')
    if reasons:
        with st.expander('Unavailable metric reasons'):
            st.dataframe(pd.DataFrame(reasons), hide_index=True, width='stretch')
    st.header('Pending and unavailable results')
    if issues:
        st.dataframe(pd.DataFrame(issues), hide_index=True, width='stretch')
    else:
        st.success('No missing or failed artifacts were found.')
