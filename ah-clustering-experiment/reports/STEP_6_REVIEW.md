# Step 6 read-only implementation review

Reviewed 2026-10-06: `src/clustering/runner.py`, `experiment.py`, `similarity.py`, `experiments/run_clustering.py`, `configs/clustering.json`, the existing artifact/configuration contracts, and implementation-plan step 6. No implementation or test source was changed by this review.

## Resolution at frozen source

The actionable P2 below is **resolved** at source digest `f90312ada4ecf46676ca601f23a1bdc74bbcb1e8a9a4bab6d3762da1dc344b7c`. Similarity identity now records the relevant implementation-file hashes and NumPy/torch versions; it excludes `configs/clustering.json`, the clustering algorithm runner and scikit-learn algorithm settings. The full source inventory remains in the publication provenance.

The reviewer reran `python -m unittest discover -s ah-clustering-experiment/tests -p test_clustering_artifacts.py -v` in the experiment environment: **2 tests passed** in 2.376 seconds. The first used actual similarity computation and checksummed artifact publication, changed the cluster-count configuration, then reused the identical artifact while a sentinel prohibited recomputation. The second loaded the 700-document training cohort from complete metadata despite only two original images being present. This directly verifies the fixed cache-identity and pixel-dependence behaviors.

The conditional OCR pending report was also re-read and is corrected for a configured OCR mode. No remaining blocking correctness finding was identified. The per-algorithm configuration-error reporting observation below remains a non-blocking limitation for invalid custom configurations.

## Original actionable finding — resolved

**P2 — Algorithm-only settings invalidate the separately saved similarity cache.**

At `src/clustering/experiment.py:61`, the similarity identity embeds the digest returned by `source_identity(source_root)`. That source inventory includes `configs/clustering.json`, including its K-means, agglomerative, HDBSCAN and spectral settings. Consequently, changing only a clustering parameter in the documented configuration creates a different similarity artifact ID and recomputes the full document-pair matrix before fitting. Even changing K-means `n_init` invalidates the ColPali/ColQwen2 similarity although K-means is unavailable for those representations. This conflicts with step 6's explicit requirement that algorithm changes reuse independently saved similarities.

A read-only in-memory reproduction changed only the K-means `n_init` value in the configuration content hash. The source digest changed from `ca6c878d80b9c0453b5ec0af200e43840decddc58fa5ad16d2ae8f716d25eb92` to `70161c9799d49064e7a966d0b2f2644ac01c4c39e37dd7321421969eb1c24ec4`; no filesystem edit was used. Because the source digest is an input to `artifact_identity`, the similarity ID necessarily changes as well.

The resolution above implements the requested separation and verifies exact artifact reuse with computation disabled.

## Additional operational observations

- **Resolved:** `load_cohort` now validates the checksummed dataset manifest and metadata directly. It preserves dataset and embedding document identity checks while allowing clustering without original image files. The regression test passed.
- Algorithm configuration construction happens outside the per-algorithm `try` block. An invalid algorithm's count policy can abort the remaining algorithms for that model; the outer runner records a model failure rather than returning any earlier successful result rows. Published earlier artifacts remain safe. Consider validating the complete configuration before starting expensive similarity computation, or include configuration validation within per-algorithm failure recording.
- **Resolved:** the run report marks `layoutlmv3-ocr` pending only when that mode is absent from the configured model mapping. Configured execution failures remain represented in the result rows.

## Verified by inspection

- The 700-document training cohort is selected explicitly. No ground-truth label array is passed to clustering; the 19-cluster choice is disclosed as class-count-informed. HDBSCAN uses its separately declared density parameters.
- Fixed CLS features use the configured L2/no-normalization policy. K-means is refused for precomputed inputs; precomputed agglomerative rejects Ward; HDBSCAN receives dissimilarities; spectral clustering receives affinities.
- Retained retrieval tokens are selected by attention and image masks. Image placeholders remain included even if marked special. Prompt and padding outputs are excluded.
- Directed scores use the mean of per-source-token maximum normalized dot products. Mean symmetrization, `1 - similarity` dissimilarities and shifted nonnegative affinity are explicitly defined. The nonmetric limitation and token-length effects are disclosed.
- Padding cannot beat negative cosine values, because padded target scores are set to negative infinity. Pair tiling computes both directions. CUDA TF32 is disabled during similarity computation and restored afterward.
- Matrix validation checks exact document ordering, dimensions, finiteness, ranges, diagonals, symmetry and the declared transforms. The independent test sources include literal expected matrices, length effects, negative controls and CPU/CUDA differential checks.
- Published clustering and similarity artifacts use checksummed immutable storage with expected configuration checks. Embedding identities include both immutable artifact identity and manifest checksum. Dataset/image identity is checked before use. Source changes during publication are rejected by the artifact writer.
- The command-line entry point blocks imports of encoder implementations, Transformers and ColPali Engine. Similarity GPU arithmetic does not invoke an encoder.

This review does not substitute for execution evidence. Full-corpus clustering, actual saved-token validation and repeat/cache reuse results are owned by the main task.
