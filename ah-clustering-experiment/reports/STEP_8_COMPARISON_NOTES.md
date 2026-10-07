# Step 8 partition comparison

`dashboard/comparison.py` compares saved partitions without importing an encoder,
clustering runner or scikit-learn. It checks dataset identity, manifest identity,
cohort designation and membership before aligning rows by document ID. Duplicate
IDs, missing documents, invalid cluster IDs, conflicting labels or splits, and
incompatible cohorts raise `ValueError` with a reason.

The non-noise overlap contingency supplies a Hungarian one-to-one assignment
maximizing document overlap. The mapping is determined only by saved assignments,
never by ground-truth labels or cluster purity. A positive overlap is required;
zero-overlap Hungarian assignments are discarded. All unmatched cluster IDs are
listed, and an unmatched right cluster has a nullable aligned ID. Its original
number is never reused as if it were aligned.

Noise (`-1`) is excluded from matching. Each document is explicitly classified as
agreement, disagreement, both noise, left noise or right noise. Agreement uses
the population assigned to regular clusters on both sides, with its coverage
shown separately. An empty comparison population produces an undefined agreement
fraction (`None`), not zero or perfect agreement. Equal-optimum cluster mappings
can occur; sorted cluster IDs provide deterministic SciPy tie handling, and the
policy discloses this ambiguity. Comparison is descriptive partition overlap,
not a ground-truth quality metric.

## Verification

Command from repository root:

```text
ah-clustering-experiment/.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -p test_dashboard_comparison.py -q
```

Result on 2026-10-06: 12 tests passed. Fixtures specify hand-computed outcomes for
perfect renumbering, arbitrary row ordering, unequal cluster counts, altered
partitions, global versus greedy alignment (contingency `[[3,2],[2,0]]` must match
4 documents, not 3), noise on each side, no jointly assigned documents and
metadata corruption. Inputs remain unchanged. These worker-authored tests are
self-check evidence; the separate independent verifier owns stronger oracle
claims.

A real-artifact smoke check loaded the 14-run catalog through `dashboard.data`,
then compared the first and last saved runs: 700 documents, 520 jointly assigned,
221 aligned agreements, 299 disagreements and 180 left-only noise documents.
Comparing the first run to itself produced 520/520 agreements and 180 both-noise
documents. No model inference, clustering or projection fitting ran. The lead
owns the final real-interface verification and overall Step 8 completion claim.
