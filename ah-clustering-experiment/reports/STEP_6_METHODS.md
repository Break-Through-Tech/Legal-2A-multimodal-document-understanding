# Step 6 baseline methods

The baseline configuration was fixed before inspecting clustering results. Inputs are the four verified full native embedding artifacts listed in `configs/clustering.json`. Each run selects exactly the 700 shared training IDs. Labels do not enter feature construction, document similarity or fitting. The full LayoutLMv3 OCR cache remains unavailable.

## CLS representations

DINOv3 and image-only LayoutLMv3 use their saved CLS vector, converted to float64 and normalized to unit L2 length. No patches are pooled. K-means uses k-means++ initialization, 20 restarts, Lloyd iterations, maximum 300 iterations and tolerance 0.0001. Agglomerative clustering uses average linkage and Euclidean distance. Spectral clustering uses RBF affinity with gamma 1, ARPACK, 20 K-means restarts and seed 42. These three algorithms request 19 clusters, explicitly informed by the known class count rather than discovered from the vectors.

HDBSCAN uses Euclidean distance, the brute-force implementation, `min_cluster_size=10`, `min_samples=5`, excess-of-mass cluster selection and no single-cluster override. It receives no label-derived count. Noise remains cluster ID -1. All algorithms run with one CPU thread and preserve their explicit parameters in each configuration.

## Retained image-token representations

ColPali and ColQwen2 use `attention_mask & image_mask`. The stored tokenizer `special_mask` marks image placeholders as special too: the first ColPali document has 1,024 image tokens and 1,025 special tokens; the first ColQwen2 document has 540 image tokens and 545 special tokens. Excluding all special tokens would remove every image token in those examples. The policy therefore retains attended image tokens even when their special mask is true; image-mask selection excludes the non-image prompts and specials.

Selected vectors are normalized individually to unit L2 length in float32. For token matrices A and B, the directed score is `mean_i max_j dot(A_i, B_j)`. This is a document-to-document experiment definition, not a claim that the models were trained for this symmetric document task. The saved artifact preserves both directed scores and their arithmetic-mean symmetrization S. The distance is `1-S`, with zero diagonal; spectral affinity is `(1+S)/2`. Cosine bounds imply S in [-1,1], distance in [0,2], and affinity in [0,1]. Rounding is clipped to the declared score bounds.

The distance is explicitly non-metric. An independently calculated token example violates the triangle inequality. Repeating every source token or duplicating target tokens preserves mean-MaxSim; changing token diversity or relative token frequency can change it. The mean removes the direct source-length sum factor, but it does not eliminate document-length effects. No accuracy improvement is assumed.

Compatible algorithms are average-linkage agglomerative and brute-force HDBSCAN on the precomputed distance, and spectral clustering on the precomputed affinity. K-means is unavailable because matrix rows are not the original feature vectors. Ward linkage is rejected for this representation. No mean-pooling ablation is included.

The CUDA implementation computes both directed scores for each pair from bounded token tiles, with TF32 disabled. Tests compare it to direct NumPy calculations and verify repeatability. The actual 700-document matrix also checks nine selected directed scores against an independent float64 NumPy calculation before publication, with tolerance 0.00002. Exact matrix shape, ID order, finite values, score ranges, diagonals, symmetry and transforms are required.

## Persistence and reproduction

Similarity artifacts contain ordered IDs, directed scores, symmetric similarities, nonmetric distances, affinities and their validation record. Their identity includes embedding content, cohort, token policy, arithmetic settings, relevant implementation files, device and arithmetic dependency versions. Clustering-only parameter edits do not invalidate similarities. Full execution provenance remains in their manifests.

Clustering artifacts contain ID-keyed assignments and basic cluster/noise counts. Their identity includes the saved representation, cohort, algorithm parameters, normalization, similarity identity when applicable, seed, dependencies and full source digest. Each newly fitted result is fitted again, and the two partitions and noise memberships must match. Compatible completed artifacts are read and verified without another fit. The CLI blocks encoder module imports. Original pixels are not required to consume saved vectors; the metadata manifest and embedding image identities must agree.

Labels are reserved for Step 7 evaluation. This step does not rank model quality, calculate label-agreement metrics, build projections or launch a dashboard.

## Upstream contracts

- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) defines the seeded feature-vector baseline and explicit initialization policy.
- [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) accepts precomputed distances with compatible linkage; Ward requires Euclidean features.
- [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) accepts a square precomputed distance matrix. The brute-force path is used for the nonmetric experimental dissimilarity; no metric-space interpretation is claimed.
- [SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) accepts a precomputed nonnegative affinity that increases with similarity.

The execution uses scikit-learn 1.9.1. These contracts were inspected before the real runs.
