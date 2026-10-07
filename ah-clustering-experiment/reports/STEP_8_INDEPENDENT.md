# Step 8 independent verification

## Oracle frozen before candidate inspection

The verifier read Step 8 of `IMPLEMENTATION_PLAN.md` before any dashboard directory existed. The oracle is the requirement to display completed saved artifacts, provide five views, align documents by ID, account for arbitrary cluster numbering, distinguish failed or unavailable results, and avoid implicit model inference or clustering.

Comparison expectations are literal partitions, independent of any candidate code: renumbering identical groups gives full agreement; shuffled document rows have no effect; a globally optimal one-to-one matching can differ from greedy matching; unequal group counts leave unmatched groups; noise is never matched to a regular cluster; and coverage reports the fraction of documents assigned to non-noise clusters in both runs. Different ID sets, duplicate IDs, or incompatible cohort identities must be rejected.

Real-artifact expectations come directly from the saved Step 7 JSON and Parquet files, the shared training CSV and dataset manifest. Dashboard catalog values and document/projection joins must agree with those sources. Missing files and deliberately corrupted fixture artifacts must become explicit unavailable states rather than plausible results.

```text
CLAIM: Saved-result dashboard data, comparison semantics and displayed artifact identities are correct.
ORACLE_SOURCE: IMPLEMENTATION_PLAN.md Step 8; literal partitions; saved Step 7 files; shared split CSV.
REAL_COMPONENTS: pandas/Parquet readers, checksummed artifact readers, comparison implementation, Streamlit application.
ALLOWED_MOCKS: None for artifact/comparison claims. Disposable local fixture files may exercise missing/corrupt inputs.
NEGATIVE_CONTROL: Mismatch IDs/cohort, inject duplicate IDs, corrupt a saved artifact fixture, remove an expected file.
REQUIRED_EVIDENCE_GRADE: Independent checks plus actual Streamlit browser execution owned by the lead.
```

AppTest checks, if run, certify Python-side Streamlit behavior only. They do not substitute for browser inspection of chart interaction, images, navigation, or layout. Results follow after executing the frozen scenarios.

## Executed checks

The 13 requirement-derived comparison tests pass. Literal expected results include perfect agreement after renumbering or row shuffling; 6/10 agreement for the globally optimal non-greedy assignment; 5/6 for unequal group counts; 1/2 coverage when only four of eight documents belong to regular clusters on both sides; and undefined agreement for zero compared documents. An assignment change reduces agreement to 3/4. Incompatible identities and duplicate or mismatched IDs are rejected. Comparison leaves its input records unchanged.

The independent raw-file audit loaded all 14 real evaluations, verified 56 payload sizes and SHA-256 checksums directly, and compared all 280 metric slots and cluster/document tables with the dashboard reader. Every run resolves to the exact 700 training IDs and labels in the shared CSV. All six unique projections preserve exact saved coordinates and IDs. Renumbering every non-noise cluster in each real run preserves full covered-population agreement. The decoded dashboard image has the same RGB pixels as the original in all 14 sampled run records.

The Streamlit AppTest sweep passes 48 checks. It executes all five views and changes projection, mode, color, noise policy, selected document, cluster subset, comparison run, comparison status and model filters. It includes the transition from an HDBSCAN noise group to K-means, empty model filters followed by restoration, and unavailable multivector feature-space metrics. Overview ARI and coverage are compared directly to all 14 saved score files under both noise scopes. Plotly payloads include the correct clustering artifact, checkpoint and algorithm in hover details and enable point selection. Actual import attempts for encoder and fitting packages/runners fail under the application's saved-results-only guard; none are loaded.

Disposable copies of real artifact files exercise corruption and missing-file controls without touching published results. A corrupt or missing score payload is excluded from the catalog. Streamlit displays a warning for missing projection coordinates, keeps a cluster usable when its image is missing, and displays the invalid artifact state after refreshing a corrupted score payload.

## Review findings

Two failure-path gaps found during independent review were repaired by their owners: malformed but valid-JSON evaluation settings now become an explicit invalid issue; and the overview now uses the same missing/corrupt-run exception boundary as the other views. Projection hover details now include the checkpoint, algorithm and clustering artifact, as required by Step 8. The scripts below exercise the resulting candidate.

```text
python -m unittest discover -s tests -p test_step8_independent.py -v
python reports/step8-evidence/audit_dashboard.py
python reports/step8-evidence/audit_streamlit.py
python reports/step8-evidence/audit_missing_files.py
```

Run from `ah-clustering-experiment/` using its `.venv/Scripts/python.exe`. The AppTest scripts run in independent processes because the import guard is process-wide.

The JSON records under `reports/step8-evidence/` contain exact dashboard file hashes, the checked real run identities and individual interaction/negative-control outcomes. These checks establish `INDEPENDENTLY_CHECKED` behavior for comparison and saved-result handling, with real Streamlit Python execution. Browser rendering and point selection must additionally be verified on the live interface by the lead before claiming Step 8 is complete. The full OCR input mode remains pending; unsupported retained-vector K-means remains visibly unavailable. No encoder inference, clustering fitting, evaluation recomputation or artifact publication occurred during this audit.
