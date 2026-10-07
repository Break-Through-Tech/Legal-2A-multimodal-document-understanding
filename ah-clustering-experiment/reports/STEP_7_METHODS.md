# Step 7 evaluation protocol

Declared before viewing results, 2026-10-06. Evaluate the 14 completed Step 6 runs on their original 700 training IDs. Do not fit clusters, extract embeddings, or inspect held-out examples. The full OCR mode remains pending. Labels come from the checksummed dataset manifest, cross-checked against the shared CSV; `Unknown` remains an ordinary benchmark label, distinct from noise (`-1`).

## Oracle contract

```text
CLAIM: saved assignments produce reproducible evaluation, examples and
  visualization coordinates with exact document mapping.
ORACLE_SOURCE: implementation-plan Step 7; independently calculated metric
  examples, mathematical invariants, shared CSV, official sklearn/UMAP APIs.
SCENARIOS: fixed features, precomputed dissimilarity, noise, absent labels,
  degenerate partitions, label renumbering, invalid IDs and cache reuse.
REAL_COMPONENTS: saved embeddings/assignments/matrices, sklearn metrics/PCA,
  UMAP, checksummed Parquet/JSON artifact publication and reading.
ALLOWED_MOCKS: none for numerical acceptance or real runs; failing sentinels
  may prohibit recomputation when checking completed-artifact reuse.
NEGATIVE_CONTROL: altered partitions, invalid matrices and mismatched IDs
  must be detected. Permuting cluster numbers must preserve scores.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED on the four available modes.
```

## Metrics and examples

Report both including-noise and excluding-noise scopes. The former treats every `-1` assignment as one group for scoring, not a discovered cluster. The latter excludes those documents and reports retained coverage. Supervised metrics use available labels, with labeled coverage reported separately. Internal metrics use every document within the selected scope. Undefined values are JSON null with a reason, never NaN or an invented score.

ARI, NMI, AMI, homogeneity, completeness and V-measure compare the saved partition with labels. NMI and AMI use arithmetic averaging. Purity is the sum of per-group majority-label counts divided by labeled documents in the scope. These are descriptive training-cohort results, not held-out classification performance. The fixed 19-cluster settings were class-count-informed before evaluation.

CLS metrics use the same saved vectors and normalization as clustering: Euclidean silhouette, Davies–Bouldin and Calinski–Harabasz. Multi-vector runs use silhouette on the saved `1 - symmetric mean-MaxSim` dissimilarity; this is explicitly nonmetric. Davies–Bouldin and Calinski–Harabasz are unavailable for that input. Scores computed in different representation spaces need that context when compared. [scikit-learn evaluation definitions](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation), [silhouette API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

For each non-noise cluster, select a medoid minimizing total within-cluster Euclidean distance or saved dissimilarity. Representatives are the three closest documents to that medoid; candidate outliers are the three farthest. Break ties by document ID. These selections do not use labels. A document can appear in both lists in small clusters. Record all tied dominant labels; a mismatch against them is a review candidate, not proof of an annotation error. Noise receives no within-cluster representatives or mismatch claims.

## Projections and storage

Cache two-dimensional PCA and UMAP for each fixed CLS representation, and UMAP with `metric=precomputed` for each retained-token dissimilarity. Do not apply PCA to distance-matrix rows. UMAP defaults: 15 neighbors, minimum distance 0.1, 500 epochs, random initialization, seed 42, one CPU thread. PCA uses the full SVD solver. Projections receive neither labels nor assignments and are shared across algorithms consuming the same representation and cohort. Both original and repeated projection coordinates must agree at absolute tolerance 1e-5. These coordinates are visualizations; clustering stays in its original space. [UMAP precomputed input contract](https://umap-learn.readthedocs.io/en/latest/api.html), [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).

Metrics artifacts contain score JSON, per-cluster summaries and per-document records linked to image paths, labels, assignments, example ranks and projection artifact IDs. Projection artifacts contain explicit document IDs and coordinates. Configuration, upstream checksums, source identity, seed, dependencies and runtime are preserved with every artifact. Completed outputs are validated and reused. No Streamlit interface is built in this step.
