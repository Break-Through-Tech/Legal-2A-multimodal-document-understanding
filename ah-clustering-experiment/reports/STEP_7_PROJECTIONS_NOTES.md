# Step 7 projection construction

The projection interface takes a numeric matrix and returns two coordinates per input row, in the same order. It has no label inputs. The orchestration layer owns document-ID joins, feature transformations, artifact provenance and coordinate caching. Neither projection changes the saved clustering assignments or becomes the clustering feature space.

- Fixed-size vectors: centered scikit-learn PCA with two components, `svd_solver=full`, no whitening; explained-variance fractions are saved. Input vectors are used as supplied, with no further normalization.
- Fixed-size vectors: UMAP with Euclidean distance.
- Multi-vector artifacts: UMAP with `metric=precomputed` on the saved symmetric nonnegative zero-diagonal dissimilarities. This construction does not claim a metric or triangle inequality. PCA on these matrices is rejected.

UMAP defaults are `n_neighbors=15`, `min_dist=0.1`, `n_epochs=500`, `init=random`, `n_components=2`, `random_state=42`, `transform_seed=42`, and `n_jobs=1`. The three neighborhood/optimization parameters may be explicitly overridden; the other construction choices are fixed. BLAS/OpenMP fitting is bounded to one thread. Invalid matrices, unsupported parameter overrides and nonfinite outputs are rejected; the implementation never silently truncates the requested neighborhood size.

Imports of UMAP are deferred until a UMAP projection is requested, so PCA does not require UMAP, Numba or encoder modules. The UMAP package itself attempts an optional TensorFlow import; the command-line import guard can forbid that import and UMAP handles its absence. No parametric UMAP model is requested.

## Checks

`tests/test_evaluation_projections.py` contains seven checks with real numerical components and no output mocks. PCA is checked against a literal rank-two rectangle: all pairwise distances must survive the projection and variance fractions must be 0.8/0.2, regardless of axis signs. Perturbing one coordinate fails that geometry check. Real UMAP is fit twice with the same seed on both fixed vectors and an explicitly nonmetric precomputed matrix. Those coordinate arrays must match exactly. Additional controls reject malformed/asymmetric/negative/nonzero-diagonal/nonfinite inputs, invalid parameters, precomputed PCA and zero-variance PCA. A subprocess verifies that PCA imports neither UMAP nor encoders.

These implementation-authored checks are self-check evidence; the separate Step 7 independent oracle and saved-artifact audit determine the integrated verification grade. Numerical UMAP repeatability is scoped to the pinned local runtime, not promised across package/platform changes.

Observed local result on 2026-10-06: all seven tests passed in 80.698 seconds using the experiment Python environment, scikit-learn 1.9.1 and umap-learn 0.5.12. Both repeated UMAP coordinate comparisons passed exactly. The precomputed-metric warning only states that inverse transformation is unavailable; no inverse transformation is requested.

## Primary references

- [scikit-learn PCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html): centering, SVD solver and explained variance.
- [UMAP API](https://umap-learn.readthedocs.io/en/latest/api.html): precomputed metric and constructor parameters.
- [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html): fixed random state and single-thread reproducibility.
- [umap-learn 0.5.12 release](https://pypi.org/project/umap-learn/0.5.12/): pinned implementation distribution.
