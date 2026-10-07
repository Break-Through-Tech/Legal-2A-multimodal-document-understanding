# Step 7 independent verification

Oracle expectations below were recorded from `IMPLEMENTATION_PLAN.md` Step 7 and mathematical definitions before the verifier inspected the candidate implementation or its tests.

For Euclidean scalar points 0, 2, 8, 10 partitioned into the first and last pairs, within-pair distance is 2 and the mean distances to the other pair are 9, 7, 7, 9. Silhouette is therefore `(7/9 + 5/7)/2 = 47/63`. Centroids 1 and 9 have average absolute scatter 1, so Davies–Bouldin is `2/8 = 1/4`. Between-cluster sum of squares is 64 and within-cluster sum is 4; Calinski–Harabasz is `(64/1)/(4/2) = 32`.

With truth AABB all agreement metrics and purity equal 1. With truth ABAB the contingency table consists of four ones: ARI and AMI are -1/2, NMI, homogeneity, completeness and V-measure are zero, and purity is 1/2. These constants are literal independent expectations, not outputs captured from the candidate or its dependencies.

Other frozen checks cover arbitrary cluster-number permutations; two noise documents excluded from a six-document cohort with coverage 2/3; Unknown as an ordinary class; missing labels; explicit undefined geometry; medoid/nearest/farthest examples and document-ID ties; preservation of full 2D pair distances by 2D PCA; seeded UMAP execution; invalid IDs, labels, feature values and dissimilarities; rejection of precomputed PCA. Negative controls deliberately corrupt a metric and a document ID in an in-memory result and require the independent assertion to fail. They do not change production files or substitute numerical components.

The verifier owns `tests/test_step7_independent.py`, this report, and the separate artifact-audit script/evidence. Candidate metrics, projection libraries and later published artifacts must execute without mocked numerical outputs. Actual run evidence and residual risks will be recorded after execution.

The first fixture used string document IDs; the existing shared contract correctly rejected them. The fixture was adapted to nonnegative integer IDs and the agreed `scores` result container without changing numerical expectations. The corrected initial 16 tests passed in 50.641 seconds using the actual scikit-learn and UMAP implementations, including repeated feature and precomputed-distance UMAP fits. No numerical component was mocked.

Review added three independently specified degeneracy cases. Points -1, 1 in one group and -2, 2 in another have equal centroids: Davies–Bouldin divides by zero and is unavailable, while Calinski–Harabasz is zero. Two repeated points at 0 and two at 2 have zero within-group scatter: Calinski–Harabasz divides by zero and is unavailable, while Davies–Bouldin is valid zero. If all four points coincide, both metrics are undefined. These checks intentionally require an explicit reason instead of library convenience fallback values.

All 16 metric scenarios, including these three degeneracy cases, passed in 1.215 seconds after the correction. Alongside the three projection scenarios already executed, the independent test file contains 19 scenarios. Its source was then frozen before the full run.

The direct artifact auditor in `outputs/logs/step7-independent-audit.py` imports no project code and does not call the evaluator, sklearn metrics, model inference, clustering, or projection fitting. Its supervised reference computes contingency-table probabilities and entropy, ARI pair counts, and AMI's expected mutual information from the hypergeometric distribution using log-factorials. Its internal reference computes explicit Euclidean differences, silhouette sample means, centroid scatter ratios, and sums of squares; multi-vector inputs use only the saved validated dissimilarity. It also recomputes every cluster's label distribution, medoid, ordered examples, document ranks, majority-label review flags and distances. PCA is checked through a separate eigen-decomposition of the centered feature Gram matrix. UMAP artifact checks cover coordinates, IDs, parameters, source hashes and recorded repeat-run evidence; seeded UMAP itself was executed in the independent small examples.

## Final saved-artifact audit

The complete audit passed for all **14 metric artifacts and six projection artifacts** from `outputs/logs/step7-1791331356802849500.json`. Every artifact records source digest `1a77022f866aec4931fa67603c155401a0aaa20ed6cddcebe930f73c1963b71a`; participating source files and their hashes were checked against the current files. The command was:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
ah-clustering-experiment/.venv/Scripts/python.exe ah-clustering-experiment/outputs/logs/step7-independent-audit.py ah-clustering-experiment/outputs/logs/step7-1791331356802849500.json
```

The independent audit confirmed:

- Exactly the shared CSV's 700 training IDs in the correct order, with all 300 held-out IDs excluded. Saved labels, original labels, split and quality metadata agree with that authority.
- All original Step 6 cluster assignments are unchanged. Evaluation input identities, normalization, representation, pooling and similarity agree with their upstream clustering configurations.
- All 280 metric slots across both noise scopes agree with the independent formulas or the declared unsupported result: 256 finite values and 24 explicit unavailable DB/CH values for precomputed dissimilarities. Scope counts, labeled coverage and noise counts/percentages agree directly with assignment rows.
- Every cluster size, label distribution, purity, dominant-label tie, representative, outlier, document rank, distance and majority-label review flag agrees with the independently reconstructed geometry. Noise has no representative/outlier ranks or distances.
- All six coordinate tables contain exactly those 700 IDs and finite coordinates, and every metric artifact references the correct projection input and manifest. Both PCA coordinate Gram matrices match independent top-two eigenspaces. Seeds, configured parameters and repeat-coordinate evidence are present and consistent.
- Raw payload sizes and SHA-256 hashes, configuration checksums and manifest checksums pass. The independent negative controls reject a deliberately wrong silhouette and a permuted ID order.

The compact machine-readable proof is `reports/step7-evidence/independent-artifact-audit.json` (including all artifact IDs, source identity, score observations, sizes, projection details and negative-control outcomes). No saved source/output artifact was modified to introduce a negative control.

```text
REVISION: 1a77022f866aec4931fa67603c155401a0aaa20ed6cddcebe930f73c1963b71a
CLAIM: Correct saved evaluation, examples and projection identity for the 14 available Step 6 runs.
ORACLE_SOURCE: Step 7 requirements; literal mathematical examples; shared CSV; raw saved assignments/vectors; independent contingency, geometry and eigen-decomposition formulas.
EVIDENCE_GRADE: LIVE_VERIFIED
CHECKS_RUN: 19 independent scenarios; all 14 metric and six projection artifacts audited; 280 metric slots checked.
REAL_COMPONENTS_EXECUTED: NumPy, scikit-learn evaluation, PCA and UMAP, raw Parquet/NPY/JSON artifacts and actual saved vectors.
MOCKS_AND_LOST_COVERAGE: No mocked numerical behavior. Full-cohort UMAP was not independently refit by the artifact auditor; it checks saved repeat evidence, while small independent UMAP cases execute twice.
NEGATIVE_CONTROL_RESULT: Wrong numerical output, document-ID changes, invalid geometry/labels and incompatible PCA input rejected.
CONTRADICTORY_EVIDENCE: None remaining after integer-fixture adaptation and explicit degeneracy handling.
RESIDUAL_RISKS: Full OCR mode lacks full embeddings. This certifies saved evaluation and examples, not optimal model quality or the Step 8 dashboard.
VERDICT: PASS for the four available modes.
```
