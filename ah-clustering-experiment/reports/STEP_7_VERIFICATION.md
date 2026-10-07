# Step 7 evaluation verification

Execution date: 2026-10-06. Acceptance covers the 14 verified Step 6 clustering artifacts from DINOv3, image-only LayoutLMv3, ColPali and ColQwen2. Each uses the original 700 training documents. Full OCR-mode embeddings remain pending, as does the separate Step 5 Drive recovery gate. No held-out document inspection, new extraction, or clustering fit is part of this step.

## Frozen source and command

The candidate source digest is `1a77022f866aec4931fa67603c155401a0aaa20ed6cddcebe930f73c1963b71a`. `step7-evidence/source-identity.json` preserves every source/configuration/test hash. Outputs identify their older immutable Step 6 inputs separately; adding evaluation code does not alter those saved assignments.

```powershell
ah-clustering-experiment/.venv/Scripts/python.exe -u ah-clustering-experiment/experiments/run_evaluation.py
ah-clustering-experiment/.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -q
```

Step 7 runs on the local CPU with one numerical thread. UMAP 0.5.12, Numba 0.68.0, pynndescent 0.6.0 and llvmlite 0.50.0 are pinned alongside scikit-learn 1.9.1. Artifact provenance records the complete observed environment. The CLI blocks encoder and optional TensorFlow imports. Source changes remain uncommitted, consistent with the implementation plan.

## Acceptance evidence

The predeclared contract and score/projection choices are in `STEP_7_METHODS.md`. Independent tests derive expected numbers from literal geometry and contingency tables before inspecting implementation. For points 0, 2, 8, 10 in two paired groups, silhouette is 47/63, Davies–Bouldin is 1/4 and Calinski–Harabasz is 32. Crossed labels give ARI and AMI -1/2. Tests also cover arbitrary cluster renumbering, missing labels, Unknown/noise separation, coverage, degenerate geometry, ID ties and deliberately corrupted outputs.

Builder tests supplement that oracle with feature/precomputed comparisons, missing-label handling and actual repeated seeded UMAP. Four artifact checks exercise real Parquet/JSON/numeric publication, cache reuse with computation forbidden, shared projection reuse after a clustering-only config edit, invalid ID/example/rank/coverage rejection, and metric preservation when PCA is unavailable. Failing sentinels in reuse checks supply no substitute numerical outputs.

Review resolved two issues before freezing the source. Degenerate Davies–Bouldin and Calinski–Harabasz cases now return unavailable reasons instead of library fallback values. Undefined PCA no longer discards valid metrics. Artifact validation checks exact coverage denominators and example-rank mappings. See `STEP_7_REVIEW.md` and `STEP_7_INDEPENDENT.md` for roles, checks and limitations.

## Interpretation limits

Agreement scores describe training-cohort partitions. Fixed-count algorithms used 19 groups informed by the known class count; HDBSCAN used its separately declared density settings. Excluding-noise scores must be read together with coverage. `Unknown` merges rare benchmark formats and is unrelated to clustering noise. The benchmark includes nonlegal business documents and does not group template families across partitions.

Internal geometry scores depend on representation. The retained-token `1 - symmetric mean-MaxSim` dissimilarity is nonmetric; only precomputed silhouette is used for it. Davies–Bouldin and Calinski–Harabasz remain unavailable. UMAP/PCA coordinates are visualizations and never replace clustering features. Majority-label mismatches and farthest-member examples are review candidates rather than verified errors. Streamlit implementation remains Step 8.

## Actual results

All 14 metrics artifacts and six shared projections completed with zero failures. The invocation took 240.7 seconds on the local CPU, including upstream validation and repeated projection fits. No source change occurred during execution. The complete invocation is preserved in `step7-evidence/run-summary.json`.

Scores below are rounded for display. Both noise policies and full-precision values for all requested metrics remain in each saved `scores.json`.

| Model | Algorithm | ARI, all | NMI, all | AMI, all | ARI, excluding noise | Coverage excluding noise |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dinov3 | kmeans | 0.7029 | 0.8324 | 0.8156 | 0.7029 | 100.0% |
| dinov3 | agglomerative | 0.0001 | 0.0846 | 0.0222 | 0.0001 | 100.0% |
| dinov3 | hdbscan | 0.4862 | 0.8102 | 0.7893 | 0.9111 | 74.3% |
| dinov3 | spectral | 0.7410 | 0.8469 | 0.8315 | 0.7410 | 100.0% |
| layoutlmv3-image | kmeans | 0.2153 | 0.4501 | 0.3980 | 0.2153 | 100.0% |
| layoutlmv3-image | agglomerative | 0.0035 | 0.1358 | 0.0685 | 0.0035 | 100.0% |
| layoutlmv3-image | hdbscan | 0.0481 | 0.3567 | 0.3143 | 0.4775 | 33.1% |
| layoutlmv3-image | spectral | 0.2090 | 0.4479 | 0.3936 | 0.2090 | 100.0% |
| colpali | agglomerative | 0.1880 | 0.6454 | 0.6151 | 0.1880 | 100.0% |
| colpali | hdbscan | 0.9048 | 0.9620 | 0.9580 | 0.9809 | 89.3% |
| colpali | spectral | 0.8995 | 0.9645 | 0.9609 | 0.8995 | 100.0% |
| colqwen2 | agglomerative | 0.0961 | 0.5183 | 0.4792 | 0.0961 | 100.0% |
| colqwen2 | hdbscan | 0.8997 | 0.9595 | 0.9550 | 0.9710 | 89.1% |
| colqwen2 | spectral | 0.8774 | 0.9453 | 0.9397 | 0.8774 | 100.0% |

HDBSCAN scores excluding noise describe a selected subset, so higher agreement at lower coverage does not establish a better system. The report makes no held-out performance claim.

| Model | Algorithm | Metrics artifact |
| --- | --- | --- |
| dinov3 | kmeans | `metrics-caa6e7d343a717723785` |
| dinov3 | agglomerative | `metrics-71967027d3b1c04fcd18` |
| dinov3 | hdbscan | `metrics-017648c85fd92d988bb0` |
| dinov3 | spectral | `metrics-2abb4654f1be19776302` |
| layoutlmv3-image | kmeans | `metrics-67d2b7b2919d072373d1` |
| layoutlmv3-image | agglomerative | `metrics-059e6287ca1b8d5d3f00` |
| layoutlmv3-image | hdbscan | `metrics-47a7fef83bccfe8950fe` |
| layoutlmv3-image | spectral | `metrics-da8f99af889c4ea9e822` |
| colpali | agglomerative | `metrics-925a5dbbdc796edc8422` |
| colpali | hdbscan | `metrics-36b2ec46394c4b78e326` |
| colpali | spectral | `metrics-4abc90e447daf4c326b9` |
| colqwen2 | agglomerative | `metrics-ca3f65b30ad9565e6fee` |
| colqwen2 | hdbscan | `metrics-54e1fcbaf1a0bc914c94` |
| colqwen2 | spectral | `metrics-3026701ed2d79b33bfc8` |

| Model | Projection | Artifact | Repeated coordinate error |
| --- | --- | --- | ---: |
| dinov3 | pca | `projections-3e26f408fd49e60a16cc` | 0 |
| dinov3 | umap | `projections-c579da635a1bd5c492d7` | 0 |
| layoutlmv3-image | pca | `projections-24c44b03a03e2cb16728` | 0 |
| layoutlmv3-image | umap | `projections-10d88662f6c1bb1d1582` | 0 |
| colpali | umap | `projections-07d761c4bba0ce5c33a4` | 0 |
| colqwen2 | umap | `projections-296636281ddd354aecdc` | 0 |

Every projection contains 700 explicit document IDs. All four algorithms for each CLS model reuse its two projection artifacts; all three compatible algorithms for each retained-token model reuse its UMAP artifact. Precomputed UMAP reports that inverse transformation is unavailable; no inverse transformation is requested or needed.

## Final verification

The full frozen-source suite ran **210 tests in 285.672 seconds: 209 passed, one unavailable-Tesseract check skipped, zero failures or errors**. All 41 new Step 7 tests passed. The OCR skip belongs to earlier extraction work. The command and counts are saved in `step7-evidence/test-summary.json`.

The independent raw-file audit passed all 14 evaluation artifacts and six projections. It verified all 280 metric slots using independently implemented contingency/geometry formulas: 256 finite scores and 24 explicit unsupported Davies–Bouldin/Calinski–Harabasz entries. It checked every cluster example and rank, exact shared-CSV labels and training IDs, exclusion of all 300 held-out IDs, unchanged original assignments, and upstream/source checksums. Both PCA coordinate Gram matrices matched a separate eigendecomposition. Full-cohort UMAP auditing checked saved repeat evidence; the independent small UMAP cases executed twice. See `step7-evidence/independent-artifact-audit.json`.

All 14 real metrics artifacts were reopened with the evaluator replaced by a failing sentinel. Every artifact was reused without numerical evaluation or replacement outputs. `step7-evidence/cache-reuse.json` records that check. Projection reuse is separately tested through actual storage with projection computation forbidden. The final source digest remained unchanged and `git diff --check` passed.

```text
REVISION: 1a77022f866aec4931fa67603c155401a0aaa20ed6cddcebe930f73c1963b71a
CLAIM: Step 7 executed correctly for 14 available-mode training-cohort runs,
  saving evaluation, cluster examples and six shared 2D projections.
ORACLE_SOURCE: Step 7 requirements; independent literal metric examples;
  shared CSV; independent contingency/geometry/eigendecomposition formulas.
EVIDENCE_GRADE: LIVE_VERIFIED
CHECKS_RUN: 210 tests, one earlier OCR skip; 14 metrics and six projections;
  repeated projections; 280-slot independent audit; 14 real cache reuses.
REAL_COMPONENTS_EXECUTED: Actual saved vectors/assignments/dissimilarities,
  NumPy/sklearn/UMAP, Parquet/JSON/numeric artifact publication and readers.
MOCKS_AND_LOST_COVERAGE: No replacement numerical outputs. Failing sentinels
  forbid recomputation in reuse checks. Auditor did not refit full UMAP.
NEGATIVE_CONTROL_RESULT: Incorrect numbers, invalid IDs/geometry, reversed
  rows, wrong examples/ranks and altered coverage were detected.
CONTRADICTORY_EVIDENCE: No unresolved finding for the accepted scope.
RESIDUAL_RISKS: Full OCR mode and Drive recovery remain pending. Scores are
  descriptive training results; projections are visualizations. Streamlit
  is not implemented by this step.
VERDICT: PASS for Step 7 on the four available modes.
```
