# Step 6 clustering execution

Run date: 2026-10-06. The runner uses the four available full embedding caches and exactly the 700 training IDs from the shared CSV. The full LayoutLMv3 OCR cache is still unavailable. Step 5's Drive recovery gate remains separate; no new encoder extraction or Colab compute is used here.

The implementation adds configured CLS clustering, validated retained-token document similarities, reusable similarity artifacts, checksummed cluster assignments, failure logs, and repeated-fit checks. Encoder imports are blocked by the CLI. Labels are excluded from fitting. The 19-cluster settings are explicitly class-count-informed; HDBSCAN receives no fixed cluster count.

## Frozen oracle contract

```text
ORACLE_SOURCE: IMPLEMENTATION_PLAN.md step 6; scikit-learn API contracts;
  independently specified token geometry, known separated groups and CSV IDs.
REAL_COMPONENTS: native artifact readers, NumPy/CUDA saved-token arithmetic,
  scikit-learn estimators, Parquet/JSON/numeric publication and cache readers.
ALLOWED_MOCKS: none in independent numerical/algorithm acceptance and real runs.
  A self-check forbids computation during similarity-cache reuse.
NEGATIVE_CONTROL: wrong partitions, permuted IDs, invalid masks/nonfinite or
  malformed matrices must fail; nonmetric distance has a triangle counterexample.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED for saved-artifact clustering.
```

The frozen source digest is `f90312ada4ecf46676ca601f23a1bdc74bbcb1e8a9a4bab6d3762da1dc344b7c`. Full file hashes are recorded under `step6-evidence/source-identity.json`. Source changes are uncommitted, as requested by the implementation plan. The unrelated earlier changes remain in place.

## Commands and environment

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
ah-clustering-experiment/.venv/Scripts/python.exe -u ah-clustering-experiment/experiments/run_clustering.py --similarity-device cuda:0
ah-clustering-experiment/.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -q
```

The local environment uses Python 3.11, scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, and torch 2.6.0+cu124. Saved-token arithmetic runs on the RTX 3050 Ti laptop GPU; fitting uses one CPU thread. Complete installed dependencies and runtime observations are preserved in each artifact manifest. This is not the separate Colab extraction environment.

## Verification design

The integrated frozen-source suite ran 169 tests in 243.213 seconds and passed with one unavailable-Tesseract skip. That skip concerns earlier OCR extraction; it does not skip any Step 6 clustering or CUDA similarity check.

- Fourteen independent tests use literal cosine geometry, a triangle-inequality counterexample, known separated groups, and exact ID order. All passed without mocks. They verify all four CLS algorithms and the three compatible precomputed algorithms, repeated partitions, matrix rejection and negative controls.
- Eleven similarity checks passed, including real CUDA versus NumPy, repeated GPU results, image tokens marked special, and token-chunk boundaries. A measured tile adjustment reduced the fixture computation time about 25% with less than 61 MiB peak tensor allocation.
- Two artifact tests passed. One consumes the 700-ID metadata cohort when only two original images exist, proving clustering does not depend on pixels. The other changes clustering configuration and reuses a real similarity artifact with recomputation forbidden.
- Each real similarity artifact checks all 700 ordered IDs, finite values, bounds, diagonals, symmetry and transforms. Nine selected directed scores are compared to a direct float64 NumPy calculation over actual saved tokens.
- Every new clustering fit runs twice. Partition agreement must be exact, allowing arbitrary cluster-number permutations, and noise membership must match.
- Independent output auditing reads the shared CSV separately, checks exact training coverage and held-out exclusion, reads the Parquet/JSON files directly, and verifies hashes, configuration policies and summary counts.

The scoped implementation review found that the initial broad similarity source identity would invalidate expensive matrices after a clustering-only config edit. The corrected identity includes relevant similarity implementation files and arithmetic dependencies; full execution provenance is still retained. `STEP_6_REVIEW.md` records the review, and `STEP_6_INDEPENDENT.md` records independent evidence.

See `STEP_6_METHODS.md` for defaults, token selection, length effects, and upstream documentation links. No ranking of clustering quality is claimed. Label-agreement metrics, projections and qualitative examples belong to Step 7; the dashboard belongs to Step 8.

## Execution results

All **14 requested available-mode runs completed**, with no failures. Each contains exactly 700 training IDs, and every repeated fit reproduced its partition and noise membership. Two retained-vector K-means combinations are explicitly unavailable. The full OCR mode remains pending its embedding cache.

The complete invocation took 815.2 seconds, including native-cache validation, similarities, repeated fits and artifact publication. This is an observed runtime, not a throughput benchmark.

| Mode | Algorithm | Clusters | Noise documents | Artifact |
| --- | --- | ---: | ---: | --- |
| dinov3 | kmeans | 19 | 0 | `clusters-7c58a2ef2e2dc1e1560a` |
| dinov3 | agglomerative | 19 | 0 | `clusters-40b280874929384e5bc9` |
| dinov3 | hdbscan | 19 | 180 | `clusters-2c570644206a69347818` |
| dinov3 | spectral | 19 | 0 | `clusters-a3b97f5bd553ae9cc906` |
| layoutlmv3-image | kmeans | 19 | 0 | `clusters-5faff8c976b54b2a2fbd` |
| layoutlmv3-image | agglomerative | 19 | 0 | `clusters-39ac0cd9385da467f171` |
| layoutlmv3-image | hdbscan | 9 | 468 | `clusters-baecfbbcb1204706c2c5` |
| layoutlmv3-image | spectral | 19 | 0 | `clusters-c876fa642aa6c4948b1b` |
| colpali | agglomerative | 19 | 0 | `clusters-79f57ed40739b91cf392` |
| colpali | hdbscan | 19 | 75 | `clusters-4700e2bad113d169c12c` |
| colpali | spectral | 19 | 0 | `clusters-b187c9c9938cc1b5bc38` |
| colqwen2 | agglomerative | 19 | 0 | `clusters-dbe077efe92a963b03df` |
| colqwen2 | hdbscan | 20 | 76 | `clusters-f0085da7b8db3aa2a971` |
| colqwen2 | spectral | 19 | 0 | `clusters-cc13d69204a20acd6d26` |

The fixed-count methods request 19 groups. HDBSCAN discovered its counts using density settings alone. Noise percentages are 25.7% for DINOv3, 66.9% for image-only LayoutLMv3, 10.7% for ColPali and 10.9% for ColQwen2. These figures do not measure class agreement or rank model quality.

| Similarity model | Artifact | Selected tokens per document | Largest reference error |
| --- | --- | ---: | ---: |
| colpali | `similarities-db1ed3d1ed9349d36b5d` | 1024 to 1024 | 2.11e-08 |
| colqwen2 | `similarities-d60d7b3eb5c6cf3b6f8d` | 12 to 768 | 3.61e-08 |

Both matrices have shape 700 by 700 with exact cohort order. The direct-score tolerance was 0.00002. ColPali directional asymmetry reached 0.2804 and ColQwen2 0.2971, so treating the raw directed score as symmetric would be wrong. Both published artifacts retain the directed scores and explicit symmetric conversion.

Four real DINO clustering artifacts were reopened with `fit_clusters` replaced by a failing sentinel. All four returned verified saved results without fitting; `step6-evidence/cache-reuse.json` records that check. The independent matrix-cache test separately proves reuse after a clustering-only configuration edit.

The compact run summary, full source identity, test summary, cache-reuse result and matrix validation records are preserved in `step6-evidence/`. Actual numeric matrices and assignment Parquet files remain ignored generated outputs.

The final independent audit passed for all 14 clustering artifacts and both similarity artifacts. It checked the shared CSV directly: exactly 700 training IDs in order, with all 300 held-out IDs excluded. It also verified file checksums, integer assignments, seed and count policies, matrix relationships, and source provenance without fitting or inference. The complete audit record is `step6-evidence/independent-artifact-audit.json`.

## Final evidence ledger

```text
REVISION: f90312ada4ecf46676ca601f23a1bdc74bbcb1e8a9a4bab6d3762da1dc344b7c
CLAIM: Step 6 executed on the four available full embedding caches,
  producing 14 clustering artifacts for the 700-document training cohort.
ORACLE_SOURCE: Step 6 requirements, upstream estimator APIs, independent
  literal geometry, shared CSV cohort, direct NumPy saved-token references.
EVIDENCE_GRADE: LIVE_VERIFIED for saved-artifact clustering.
CHECKS_RUN: 169 tests (one unrelated OCR-engine skip); repeated real fits;
  independent audit of 14 assignments and two matrices; cache-reuse checks.
REAL_COMPONENTS: saved embedding artifacts, CUDA/NumPy arithmetic,
  scikit-learn fits, Parquet assignments, numeric matrices and manifests.
MOCKS: none in real runs or independent numerical acceptance; failing
  computation sentinels in reuse checks provide no replacement outputs.
NEGATIVE_CONTROL: invalid matrices, permuted IDs and perturbed partitions
  rejected; nonmetric dissimilarity demonstrated by a triangle counterexample.
CONTRADICTORY_EVIDENCE: none found for the execution claim.
RESIDUAL_RISK: full OCR embedding cache remains unavailable; clustering
  quality and projections require Step 7. Similarity-derived dissimilarity
  is explicitly nonmetric. Successful execution does not establish a winner.
VERDICT: PASS for Step 6 on the four available modes; OCR mode pending.
```
