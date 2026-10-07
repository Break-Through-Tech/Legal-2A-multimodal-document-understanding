# Step 8 verification

Completed 2026-10-06 (America/Chicago). Evidence grade: **LIVE_VERIFIED** for the four available modes and their 14 saved runs. The dashboard is running locally at http://127.0.0.1:8501. It reads saved results on the CPU; dashboard interactions require no local or Colab GPU.

The frozen source digest is `0fc7619c291a2af6cfcc4e692b5307c6f1a597ea797ae30d2fbfa3578be666db`. [Source identity](step8-evidence/source-identity.json) includes uncommitted source, tests, configurations and dependency declarations. Documentation and generated evidence are excluded. The acceptance contract was recorded in [STEP_8_METHODS.md](STEP_8_METHODS.md) before implementation. Independent expectations and review findings are recorded in [STEP_8_INDEPENDENT.md](STEP_8_INDEPENDENT.md).

## Evidence ledger

| Claim | Evidence | Result |
|---|---|---|
| Saved results are displayed faithfully | Independent raw-file audit of 14 runs, 56 payload checksums, 280 metric slots, six projections, exact training IDs and sampled decoded images | Passed |
| Comparison handles arbitrary cluster numbering, noise and incompatible cohorts | Literal partition tests specified independently of the implementation, plus real-run renumbering | Passed |
| Five views and dynamic controls execute correctly | 48 Streamlit AppTest checks, including both noise scopes, empty filters, mode switches and blocked computation imports | Passed |
| Existing functionality remains valid | Full unittest discovery: 248 tests in 239.420 seconds, zero failures, one optional OCR-engine skip | Passed |
| Real UI supports exploration and document selection | Live browser observations below and saved screenshots | Passed |
| Missing and corrupt inputs are explicit | Four independent missing-file checks plus live browser controls using disposable copies | Passed |

Machine-readable evidence: [artifact audit](step8-evidence/independent-artifact-audit.json), [AppTest](step8-evidence/independent-apptest.json), [missing files](step8-evidence/independent-missing-files.json), and [browser checks](step8-evidence/browser-verification.json). The three independent audit files record hashes of the final dashboard candidate.

Run the full suite from the experiment directory with `.venv/Scripts/python.exe -m unittest discover -s tests -q`. The independent report lists the additional reproducible audit scripts. Dependency versions are pinned in `requirements-dashboard.txt`: Streamlit 1.65.0, Plotly 7.1.0 and SciPy 1.17.1.

## Live browser observations

- **Overview:** 14 completed runs, four input modes, 700 documents per run, and three pending/unavailable entries. Metric charts, saved score tables, runtime metadata and unavailable reasons render. Noise policy changes through the keyboard as well as the Python interaction checks.
- **Embedding explorer:** document 76 opens its equipment-inspection image. A real chart-point selection opens document 0 and its image. Switching PCA to UMAP and coloring by cluster works. Filtering to EQUIPMENT_INSPECTION displays 35 of 700 documents.
- **Cluster explorer:** DINOv3 K-means cluster 0 contains 36 documents with 94.4% majority-label purity (NUTRITION). Representatives are 356, 260 and 445; candidate outliers are 886, 976 and 45. Selecting the representatives subset opens document 356 and its original image.
- **Experiment comparison:** DINOv3 K-means versus ColPali agglomerative clustering displays 39.3% aligned agreement across 700 documents and 425 disagreements. Switching to ColPali HDBSCAN compares 625 documents assigned to regular clusters on both sides, reports 75 right-side noise documents and 89.3% comparison coverage. The aligned contingency chart and document review remain usable.
- **Experiment details:** the actual DINOv3 checkpoint `facebook/dinov3-vits16-pretrain-lvd1689m`, revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`, L2 normalization, K-means group count 19 and seed 42 are visible alongside configuration and provenance.
- **Missing-file controls:** a disposable copy of real DINOv3 artifacts on port 8502 shows an explicit missing-projection warning. The cluster view remains usable while warning about a missing image. Corrupting only the copied score payload and refreshing removes the invalid run from completed results and displays its issue. The disposable server was stopped after verification; published artifacts were not changed.
- **Layout and keyboard:** tested CSS viewport widths of 320, 768, 1024 and 1440 pixels without outer-page horizontal overflow. The mobile details view remains readable with navigation collapsed. Space selects the metric-noise radio option and Tab advances focus. Temporary viewport overrides were reset. This is a focused usability check, not a full accessibility audit.
- **Console:** server restarts produced expected WebSocket disconnect messages during development. No warnings or errors appeared in the final live-browser checks after `2026-10-07T00:13:00Z`.

Screenshots: [overview](step8-evidence/overview.jpg), [point selection](step8-evidence/explorer-point-selection.jpg), [filtered UMAP](step8-evidence/filtered-umap.jpg), [representative](step8-evidence/cluster-representative.jpg), [comparison](step8-evidence/comparison.jpg), [missing projection](step8-evidence/missing-projection.jpg), [missing image](step8-evidence/missing-image.jpg), [corrupt artifact](step8-evidence/corrupt-artifact.jpg), [mobile details](step8-evidence/mobile-details.jpg).

## Implementation and limits

The dashboard validates artifacts, joins projections by document ID, checks image paths and checksums, and caches saved results for up to 30 seconds with manual refresh. Comparison uses a one-to-one maximum-overlap assignment among non-noise clusters and reports the compared population explicitly. A runtime import guard rejects encoder, clustering, evaluation and fitting modules; independent checks attempted these imports and confirmed rejection. UI filtering does not fit models or overwrite saved experiment scores.

Review and browser testing repaired malformed-settings handling, the overview error boundary, missing hover provenance and Plotly point-click selection. The final source was frozen before the final audit and full test run.

Full OCR-mode extraction and the Drive/Colab recovery gate remain pending from Step 5. Retained-vector K-means remains unsupported and is shown with a reason. The Step 9 research findings review and later classification work are not included in Step 8. The server is local and remains running for the user; changes are uncommitted.
