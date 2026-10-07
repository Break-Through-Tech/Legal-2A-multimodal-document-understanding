# Step 8 dashboard acceptance

The dashboard reads the verified Step 7 artifacts for four modes and 14 clustering runs, with six saved projections and 700 training documents per run. It has five views: overview, embedding explorer, cluster explorer, experiment comparison and experiment details. Incomplete or failed results, the pending full OCR mode and unsupported retained-vector K-means are shown separately.

```text
CLAIM: The real Streamlit interface displays saved experiments accurately,
  supports filtering/comparison/image selection, and handles missing files
  without running encoders, clustering, evaluation or projection fitting.
ORACLE_SOURCE: IMPLEMENTATION_PLAN.md Step 8; raw Step 7 artifacts and shared
  CSV; independently specified partition-permutation and noise examples.
REAL_COMPONENTS: artifact readers, Streamlit server and browser, Plotly
  selections, scipy assignment alignment, saved Parquet/JSON and images.
ALLOWED_MOCKS: none in real-artifact/browser acceptance. Deliberately altered
  disposable artifacts are negative controls, not substitutes for real data.
NEGATIVE_CONTROL: corrupt/missing artifact or image; incompatible cohorts;
  changed IDs; altered partition; arbitrary cluster-number permutation.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED for the available four modes.
```

## Display policies

Both metric noise scopes remain selectable and show coverage. Undefined scores retain their recorded reasons. `Unknown` is a ground-truth benchmark class, whereas `-1` is clustering noise. Overview scores describe whole saved runs; visual document filters never silently recompute or replace them. Coordinates remain visualization-only.

Ground-truth labels, cluster assignments, dataset source, document quality and declared majority-label mismatch can color saved coordinates. The source is the recorded benchmark dataset, not an inferred organization/template family. Plot selections and a keyboard-accessible document picker resolve explicit document IDs before opening an image. Images must stay within the experiment root and pass their saved checksum; missing or corrupt images show a reason.

Run comparison requires matching dataset/cohort identity and the exact document ID set. It aligns IDs before computing a contingency table. A one-to-one maximum-overlap assignment aligns non-noise cluster numbers. Unmatched groups remain disagreements. Noise is handled separately and never matched to a regular cluster. Agreement is reported over documents assigned to non-noise groups in both runs, together with comparison coverage and the three noise-state counts. The alignment is descriptive, not a ground-truth classifier.

The UI caches read results for at most 30 seconds and provides an explicit Refresh saved results button. It validates files again after refresh and displays unavailable status instead of retaining an unverified active run. It never writes experiment artifacts. Only local loopback access is needed.

## Verification plan

1. Test independent partition examples, document mapping, artifact validation, and missing-file behavior.
2. Audit every overview value and document table against raw Step 7 outputs.
3. Exercise all five views with Streamlit AppTest while computation imports are blocked.
4. Inspect the running browser: filters and noise policies, real projections, selected documents/images, cluster examples, aligned run comparison, provenance, and isolated missing-file states. Check responsive layout and keyboard controls.
5. Record the exact source digest, dependencies, browser evidence, test results and remaining limits before completion.

[Streamlit chart selection API](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart) supports rerunning the display on selected points; point custom data carries document IDs. [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) complements, but does not replace, browser verification.
