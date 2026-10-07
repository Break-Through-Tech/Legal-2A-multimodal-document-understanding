# Independent step 6 verification

## Frozen oracle

Expected behavior was derived from `IMPLEMENTATION_PLAN.md`, step 6, and the
public API/policy contract supplied by the integration owner before inspecting
candidate clustering or similarity source. The verifier did not inspect the fit
or similarity implementation. After tests were frozen and passed, a separate
read-only integration review inspected `src/clustering/experiment.py` and
`experiments/run_clustering.py`; that review did not change the oracle or tests.
Tests live in `tests/test_step6_independent.py`.

- **ORACLE_SOURCE:** configured CLS clustering and compatible precomputed
  clustering; explicit token masking, directionality, symmetrization, score
  transforms and document-length effects; finite matrices with exact document
  order; reproducible partitions; no model inference during clustering.
- **REAL_COMPONENTS:** public clustering functions and sklearn algorithms,
  similarity computation and validation on CPU, fresh-process module import.
- **ALLOWED_MOCKS:** none. Input arrays are deliberately synthetic independent
  mathematical fixtures, not presented as extracted model embeddings.
- **NEGATIVE_CONTROL:** permuted ordered IDs must be rejected even when matrix
  shape is valid; a wrong cluster-membership assignment must fail the partition
  oracle while arbitrary cluster-number changes must pass.
- **REQUIRED_EVIDENCE_GRADE:** independent mathematical/local behavior evidence,
  with saved-artifact end-to-end, full-cohort, GPU and live-backend evidence
  explicitly separate.

The initial interface packet proposed excluding tokenizer-special image IDs.
The integration owner then reported an observation from existing ColPali and
ColQwen2 caches: image token IDs themselves carry the special-token flag. The
selected policy was corrected before candidate inspection to `attention_mask &
image_mask`, retaining selected image IDs regardless of `special_mask`. The
independent fixture distinguishes attended special image tokens, padded image
tokens and non-image prompt tokens. This correction is grounded in the reported
artifact observation, not inferred from the candidate implementation.

## Mathematical expectations

For unit-token documents `A=[(1,0)]`, `B=[(1,0),(0,1)]`, `C=[(-1,0)]`, mean
query-to-document MaxSim must be:

```text
       A     B     C
A     1     1    -1
B     .5    1    -.5
C    -1     0     1
```

The symmetric average has off-diagonal scores `S(A,B)=.75`, `S(A,C)=-1`,
`S(B,C)=-.25`. Therefore `D=1-S` has off-diagonal values `.25`, `2`, `1.25`,
and `affinity=(S+1)/2` gives `.875`, `0`, `.375`. The diagonal is one for
similarity/affinity and zero for distance. Values must agree with these literal
expectations within absolute tolerance `1e-6`, with zero relative tolerance.
Repeated CPU computations must be elementwise identical.

This fixture proves that the declared transformed distance is **not a metric**:
`D(A,C)=2 > D(A,B)+D(B,C)=1.5`. Duplicating every token in B must leave its mean
score unchanged, while duplicating only its `(0,1)` token changes `B→A` from
`1/2` to `1/3`. Positive per-token rescaling must not change normalized scores.

Clustering uses two identical 4-by-4 point grids separated by eight units.
Expected membership is specified directly from construction; pairwise
co-membership comparisons permit arbitrary cluster-label permutations. K-means,
average-linkage agglomerative, HDBSCAN and RBF spectral clustering must recover
the two groups and repeat that partition for seed 42. Compatible precomputed
agglomerative, HDBSCAN and spectral paths must recover the same groups. Standard
K-means and Ward linkage must reject precomputed-distance mode.

Validation checks cover matrix dimensionality, document count, empty features,
NaN/infinity, distance symmetry/nonnegativity/zero diagonal, nonnegative affinity,
empty or zero-norm tokens, mismatched vector dimensions, malformed masks,
missing/duplicate/permuted IDs, and corrupted matrix results. A fresh Python
process checks that saved-feature K-means never imports `src.embeddings.models`.

## Execution and limits

Command, from the experiment directory:

```text
.venv/Scripts/python.exe -m unittest discover -s tests -p test_step6_independent.py -v
```

Execution on 2026-10-06: **all 14 tests passed in 10.430 seconds** using the
experiment `.venv`, sklearn 1.9.1, NumPy 2.4.6 and torch 2.6.0+cu124. Computation
in these tests was CPU-only. sklearn emitted its forward-looking HDBSCAN `copy`
default warning; no test failed or skipped. The tests are frozen for integration
and real saved-artifact runs.

Frozen test SHA-256:
`f9bd38ea77eb14effb71a2b2cf7ee65f9824f49332abeeb437ddd6cc2adcb080`.
Experiment source digest observed after the run:
`ca6c878d80b9c0453b5ec0af200e43840decddc58fa5ad16d2ae8f716d25eb92`.
Other workstreams were active, so final integration must record its final source
identity; this digest identifies the observation point only.

The subsequent integration review found explicit source/configuration identities,
saved-artifact readers, ordered-ID validation, failed-run records, a repeated-fit
partition check and an encoder-import blocker in the CLI. This is source review
evidence, not an independently executed saved-artifact workflow. It found no
additional actionable issue within that limited inspection.

## Completed real-artifact audit

After the live runner began, the verifier independently checked the completed
artifacts without fitting models, running encoders, or opening document images.
The authoritative expected IDs came directly from the shared `data/split.csv`
training rows, not the candidate cohort selector. A deliberately permuted ID
list was rejected by that independent order check.

The final audit covers **all 14 completed clustering artifacts and both saved
similarity artifacts** from `outputs/logs/step6-1791329587419145600.json`. It
supersedes the earlier six-artifact snapshot. The summary contains no failed
run and correctly marks K-means unavailable for ColPali and ColQwen2. Every
clustering artifact passed:

- `read_artifact` integrity validation, plus direct Parquet and JSON reads;
- independently recomputed payload byte counts and SHA-256 checksums;
- exact ordered membership of all 700 training IDs, no duplicates, and no IDs
  from the 300 validation/test rows;
- integer cluster labels greater than or equal to -1;
- seed 42 and declared class-count-informed 19-cluster settings where applicable;
- HDBSCAN `min_cluster_size=10`, `min_samples=5`, with no fixed cluster count;
- independently recomputed cluster/noise/document counts matching saved summaries
  and completion logs.

HDBSCAN summaries were independently recomputed from the stored assignments:

| Model | Clusters | Noise documents |
| --- | ---: | ---: |
| DINOv3 | 19 | 180 |
| LayoutLMv3 image-only | 9 | 468 |
| ColPali | 19 | 75 |
| ColQwen2 | 20 | 76 |

The other ten clustering artifacts each contain 19 clusters and no noise. These
are observations, not quality claims.

The similarity audit read each numeric `.npy` file directly with pickle disabled
and independently checked all four 700-by-700 matrices for finite floating-point
values. Direct Parquet reads establish that matrix positions 0 through 699 map
exactly to the shared CSV's sorted 700 training IDs. Each artifact's raw sizes and
SHA-256 values match its manifest; all three downstream clustering runs for that
model reference that same independently audited similarity artifact.

- ColPali: `similarities-db1ed3d1ed9349d36b5d`.
- ColQwen2: `similarities-d60d7b3eb5c6cf3b6f8d`.

For every matrix entry, absolute tolerance `2e-5` with zero relative tolerance
verifies `S=(directed+directed.T)/2`, `D=1-S`, and `affinity=(S+1)/2`. Symmetry,
declared score bounds, self similarity one and exact distance diagonal zero all
pass. The maximum directed asymmetry is approximately 0.280426 for ColPali and
0.297095 for ColQwen2, confirming that symmetrization is material on these data.
The saved validation metadata explicitly labels the distance nonmetric.

All artifact provenance records match the full run source digest
`f90312ada4ecf46676ca601f23a1bdc74bbcb1e8a9a4bab6d3762da1dc344b7c`.
Clustering configurations carry that digest; similarity configurations instead
identify their participating implementation files, whose byte hashes were also
independently checked against the actual files. No fitting, inference or pixel
loading occurred during this audit.

Machine-readable proof is preserved in
`reports/step6-evidence/independent-artifact-audit.json` and the ignored
`outputs/logs/step6-independent-artifact-audit.json`. Both contain the full artifact
IDs, checksums, observed matrix ranges, count summaries, source identity and the
rejected ID-permutation negative control. Multimodal LayoutLMv3 remains explicitly
pending because a complete expected-cohort OCR embedding cache is unavailable.

These checks certify the audited saved contents and their declared relationships,
not real-document cluster quality, cache-reuse behavior, changed-run identity
isolation, mounted-backend behavior, independent GPU-versus-CPU numerical agreement,
or dashboard results. They do not substitute for later step 7 metrics.
