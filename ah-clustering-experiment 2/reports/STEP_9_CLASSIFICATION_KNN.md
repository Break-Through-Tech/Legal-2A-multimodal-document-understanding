# Step 9: retained-vector versus pooled-vector classification

## Answer

**ColPali: no.** Retaining every token beats pooled logistic regression by 0.0059 macro F1
on the shared random split, with a paired 95 % bootstrap interval of [−0.0130, +0.0256].
That interval contains zero, and 67 % of resamples favour the retained arm — a coin flip.
Under a template-grouped split the difference is exactly 0.0000, interval
[−0.0176, +0.0198]. Cost: 515× storage, 339× preparation time.

**ColQwen2: significant only while the leak is present.** The random split gives +0.0259
with interval [+0.0066, +0.0510] and 99.8 % of resamples favouring the retained arm, so
that gap is real. Remove the template leak and it falls to +0.0137 (interval
[−0.0008, +0.0363], 88 % of resamples) at threshold 0.80 and +0.0058 (interval
[−0.0011, +0.0203], 66 %) at 0.85. Neither grouped interval clears zero. Cost: 372×
storage, 250× preparation time.

**A pooled nearest-centroid classifier is competitive throughout.** It holds 19 prototype
vectors in total — one per class — against 752 to 1030 vectors per document. On the grouped
splits it reaches 0.9799 (ColPali, τ=0.85) and 0.9882 (ColQwen2, τ=0.85), which brackets the
retained arm rather than losing to it. No confidence interval was computed for that pair;
those are point estimates.

**So the honest verdict is that neither model shows a statistically significant retained
advantage once template leakage is removed.** The answer still depends on the encoder, but
the dependency is now visible as a difference in how much of the random-split gain survives,
not as a difference in whether an advantage exists.

**The validation split reverses the sign.** Scored on the 150 validation documents, pooled
logistic regression reaches 0.9871 (ColPali) and 0.9810 (ColQwen2) against 0.9696 and 0.9752
for retained top-1: −0.0174 and −0.0058, both favouring the pooled arm. The repository's plan
says to select settings on validation and score test once. Applied literally, validation
selects the pooled arm for both models, and the ColQwen2 test-set advantage would never have
been selected. That is the strongest argument in this report that the advantage is not
stable.

Everything below is produced inside this branch and depends on no unmerged work from another
branch.

## Scope

The task is a retrieval classifier. For each held-out document, rank the 700 training
documents by similarity, take the labels of the top *k* neighbours, and predict the
plurality label for *k* = 1, 3 and 5. The comparison arm collapses the same tokens to one
vector and fits a linear boundary.

Both arms read the same embedding cache, the same split file, the same label column and
the same metric. Only the representation and the decision rule differ.

## Conventions

| Item | Value |
| --- | --- |
| Embedding artifacts | `embeddings-d987f31cbdb073fed394` (ColPali v1.3), `embeddings-17d7e1b5ec380770c824` (ColQwen2) |
| Split file | `data/split.csv`, 700 train / 150 validation / 150 test, unmodified |
| Label column | `label` (19 classes: 18 head types plus one `Unknown` bucket) |
| Query set | `test` |
| Neighbour set | `train` |
| Primary metric | macro F1 over the union of reference and predicted labels, zero division yielding 0 |
| Similarity matrix | `directed` (`mean_i max_j dot(A_i, B_j)`), query to neighbour, unsymmetrised |
| Seed | 42 |

`directed` is used deliberately. `similarity` is `(directed + directedᵀ)/2`, which discards
the direction retrieval needs. Both are stored in the same artifact, so this is recorded
rather than discovered.

## Method

**Token selection.** `src.clustering.similarity.select_vectors` selects
`attention_mask & image_mask` and applies per-token L2 normalization. The pooled arm calls
the same function, so both arms see identical tokens. Comparing a pooled vector built from
different tokens would confound representation with token choice.

**Retained arm.** Every selected token is kept and documents are compared with the
validated mean-MaxSim definition already in the repository. Ranking uses similarity alone;
labels enter only at the vote.

**Vote rule** (`src/classification/knn.py`, `DEFINITION`). Plurality of the *k* neighbour
labels. Ties resolve by highest summed similarity among tied labels, then lexicographically
smallest label name. Without a fixed order the reported score would change between runs on
tied votes.

**Pooled arm** (`src/classification/baselines.py`). Arithmetic mean over selected tokens in
float64, then L2 normalization, then `LogisticRegression` (lbfgs, `max_iter=5000`) or
`NearestCentroid`.

**Template grouping** (`experiments/run_template_aware_comparison.py`). Within one label,
documents are connected when their similarity is at least a threshold; connected components
are template groups. Whole groups are then assigned to one side, stratified by label. The
threshold is swept (0.80, 0.85, 0.90) instead of chosen, because grouping is a modelling
decision and its effect should be visible.

## Results on the shared random split

| Model | Arm | Representation | macro F1 | accuracy |
| --- | --- | --- | ---: | ---: |
| ColPali | pooled + LR (C=1) | mean-pool → 1 × 128 | 0.9687 | 0.9667 |
| ColPali | pooled + LR (C=10) | mean-pool → 1 × 128 | 0.9620 | 0.9600 |
| ColPali | pooled + nearest centroid | mean-pool → 1 × 128 | 0.9626 | 0.9600 |
| ColPali | **retained + top-1** | 1030 × 128 | **0.9745** | 0.9733 |
| ColPali | retained + top-3 | 1030 × 128 | 0.9745 | 0.9733 |
| ColPali | retained + top-5 | 1030 × 128 | 0.9549 | 0.9533 |
| ColQwen2 | pooled + LR (C=1) | mean-pool → 1 × 128 | 0.9618 | 0.9600 |
| ColQwen2 | pooled + LR (C=10) | mean-pool → 1 × 128 | 0.9618 | 0.9600 |
| ColQwen2 | pooled + nearest centroid | mean-pool → 1 × 128 | 0.9634 | 0.9600 |
| ColQwen2 | **retained + top-1/3/5** | 752 × 128 | **0.9877** | 0.9867 |

Baselines on the same split: majority class (`Unknown`) 0.0072; random over 19 classes 0.0526.

## Template leakage is real

Share of the 150 test documents that have a same-template document in training:

| Model | τ=0.80 | τ=0.85 | τ=0.90 |
| --- | ---: | ---: | ---: |
| ColPali | **69 / 150 = 46.0 %** | 19 / 150 = 12.7 % | 5 / 150 = 3.3 % |
| ColQwen2 | 24 / 150 = 16.0 % | 9 / 150 = 6.0 % | 0 / 150 = 0.0 % |

Under the grouped split the leak is 0 % by construction at every threshold.

## Results on the template-grouped split

| Model | τ | test docs | pooled | retained k=1 | retained k=3 | retained k=5 | best retained − pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ColPali | 0.80 | 101 | **0.9875** | 0.9801 | 0.9736 | 0.9487 | **−0.0075** |
| ColPali | 0.85 | 136 | 0.9737 | 0.9737 | 0.9675 | 0.9463 | **0.0000** |
| ColPali | 0.90 | 146 | 0.9801 | 0.9802 | 0.9740 | 0.9528 | +0.0000 |
| ColQwen2 | 0.80 | 132 | 0.9741 | 0.9815 | **0.9877** | 0.9816 | **+0.0137** |
| ColQwen2 | 0.85 | 146 | 0.9820 | 0.9815 | **0.9877** | 0.9816 | +0.0058 |
| ColQwen2 | 0.90 | 146 | 0.9820 | 0.9815 | **0.9877** | 0.9816 | +0.0058 |

The last column is computed from unrounded scores, so it can differ from subtracting the
four-decimal values in the same row by 0.0001.

At τ=0.90 ColQwen2 has 1000 groups of size 1, so the grouped split carries no grouping and
that row is a control rather than an independent test.

## Results on the validation split

Same arms, same training documents, scored against the 150 validation documents. This is the
split the plan reserves for selecting settings.

| Model | Arm | Representation | macro F1 | accuracy |
| --- | --- | --- | ---: | ---: |
| ColPali | pooled + LR (C=1) | mean-pool → 1 × 128 | **0.9871** | 0.9867 |
| ColPali | pooled + LR (C=10) | mean-pool → 1 × 128 | 0.9871 | 0.9867 |
| ColPali | pooled + nearest centroid | mean-pool → 1 × 128 | 0.9697 | 0.9667 |
| ColPali | retained + top-1 | 1030 × 128 | 0.9696 | 0.9667 |
| ColPali | retained + top-3 | 1030 × 128 | 0.9473 | 0.9467 |
| ColPali | retained + top-5 | 1030 × 128 | 0.9304 | 0.9333 |
| ColQwen2 | pooled + LR (C=1) | mean-pool → 1 × 128 | 0.9754 | 0.9733 |
| ColQwen2 | pooled + LR (C=10) | mean-pool → 1 × 128 | **0.9810** | 0.9800 |
| ColQwen2 | pooled + nearest centroid | mean-pool → 1 × 128 | 0.9763 | 0.9733 |
| ColQwen2 | retained + top-1 | 752 × 128 | 0.9752 | 0.9733 |
| ColQwen2 | retained + top-3 | 752 × 128 | 0.9484 | 0.9467 |
| ColQwen2 | retained + top-5 | 752 × 128 | 0.9381 | 0.9400 |

Validation favours the pooled arm for both models, by 0.0174 (ColPali) and 0.0058 (ColQwen2).
It also orders `k` differently from test: on validation `k=1` beats `k=3` beats `k=5` for both
models, while on test `k=1` and `k=3` tie. A setting chosen on validation would therefore not
be the setting that looks best on test, which is the expected behaviour when the differences
are noise.

## Cost

| Quantity | ColPali | ColQwen2 |
| --- | ---: | ---: |
| Vectors per document (median) | 1030 | 752 |
| Retained vector bytes, 1000 documents | 251.6 MiB | 181.5 MiB |
| Pooled vector bytes | 0.488 MiB | 0.488 MiB |
| Storage ratio | **515×** | **372×** |
| Similarity matrix preparation | 101.6 s (499,500 pairs) | 75.1 s (499,500 pairs) |
| Pooling preparation | 0.3 s | 0.2 s |

## Confidence intervals

Paired bootstrap, 2000 replicates, resampling stratified by label, models held fixed. Both
arms are scored on the same resampled documents, so the interval is on the difference.
Reference arm is pooled logistic regression (C=1).

| Model | Split | Arm | Δ macro F1 | 95 % CI | P(Δ>0) |
| --- | --- | --- | ---: | --- | ---: |
| ColPali | random | top-1 | +0.0059 | [−0.0130, +0.0256] | 0.673 |
| ColPali | random | top-5 | −0.0137 | [−0.0423, +0.0129] | 0.160 |
| ColPali | grouped τ=0.85 | top-1 | 0.0000 | [−0.0176, +0.0198] | 0.495 |
| ColPali | grouped τ=0.85 | top-5 | −0.0273 | [−0.0541, −0.0002] | 0.025 |
| **ColQwen2** | **random** | **top-1/3/5** | **+0.0259** | **[+0.0066, +0.0510]** | **0.998** |
| ColQwen2 | grouped τ=0.80 | top-3 | +0.0137 | [−0.0008, +0.0363] | 0.883 |
| ColQwen2 | grouped τ=0.80 | top-1 | +0.0074 | [−0.0135, +0.0344] | 0.726 |
| ColQwen2 | grouped τ=0.85 | top-3 | +0.0058 | [−0.0011, +0.0203] | 0.663 |
| ColQwen2 | grouped τ=0.85 | top-1 | −0.0005 | [−0.0205, +0.0194] | 0.361 |

Unstratified resampling gives intervals of the same width and the same verdicts in every
row, so the conclusion is not an artifact of the resampling scheme.

Only one row clears zero: ColQwen2 on the random split. Every grouped interval contains
zero. ColPali's only interval that excludes zero is `top-5` at τ=0.85, and it excludes zero
on the losing side.

## Reading the result

**The `k` sweep is the sharpest evidence in this report.** For ColQwen2, `top-1` drops from
0.9877 to 0.9815 once template groups are removed, while `top-3` stays at exactly 0.9877
across all four splits. The single nearest neighbour was leaning on the near-duplicate;
aggregating three neighbours recovers the accuracy. That is what type signal looks like, as
opposed to copy signal. ColPali shows the opposite pattern: `k=1` is its best setting on
grouped splits and raising `k` hurts, so its neighbours do not carry recoverable signal.

**The surviving ColQwen2 margin is one to two documents.** The grouped test sets hold
132–146 documents across 19 classes, so one flipped document moves macro F1 by roughly
0.004–0.010. +0.0137 is about two documents; +0.0058 is about one. The bootstrap interval
agrees: neither grouped estimate is separable from zero.

**`k` should not be shared across models.** A single `k` picked for both would misreport at
least one of them.

## Verification

```text
python -m unittest tests.test_classification_knn tests.test_classification_baselines -v
```

30 tests pass, covering the vote contract (plurality, both tie-break levels, disjoint ID
sets, `k` bounds, shape and finiteness rejection), the macro-F1 definition against
hand-computed values, pooling (mean-then-normalise against a hand-computed bisector, unit
length, zero-norm rejection, mask agreement with the retained arm), the classifier
(separable clusters for both methods, unknown method, non-positive `C`, single class,
predict-before-fit), and the byte accounting helpers.

Reproduce:

```text
python experiments/run_clustering.py --models colpali colqwen2 --cohort all --similarity-device cuda:0
python experiments/run_representation_comparison.py --label-column label --C 1.0 10.0 \
    --out results/representation-comparison.json
python experiments/run_template_aware_comparison.py --label-column label \
    --thresholds 0.80 0.85 0.90 --out results/template-aware-comparison.json
python experiments/run_bootstrap_ci.py --mode random --replicates 2000 \
    --out results/bootstrap-random.json
python experiments/run_bootstrap_ci.py --mode grouped --threshold 0.85 --replicates 2000 \
    --out results/bootstrap-grouped.json
```

Pass `--embedding-id` and `--similarity-id` to select ColQwen2.

## Limitations

- **Grouped splits are not comparable to each other or to the random split in absolute
  terms.** Group assignment changes train and test sizes (101–146 test documents) and, at
  τ=0.80 for ColPali, leaves one class without test documents so macro F1 averages over 18
  classes. Only the within-split arm comparison is meaningful.
- **The intervals cover test-set variance only.** The bootstrap resamples the test documents
  and holds both fitted models fixed, so it answers "how uncertain is this test-set
  difference" and not "how much would this vary across training sets". Training-set variance
  is unmeasured.
- **One grouping rule.** Connected components at a similarity threshold is one definition of
  "same template". Agglomerative clustering within a class, or a learned template model,
  would group differently. The sweep shows the direction of the effect but not its exact
  size.
- **Thresholds are not calibrated.** 0.80 / 0.85 / 0.90 were chosen to span the observed
  similarity range (ColPali nearest-neighbour median 0.7795, maximum 0.9266), not derived
  from a template model.
- **One grouping per threshold, one bootstrap seed.** Replicates share the point-estimate
  split; the interval does not include variation from re-grouping at a different seed.
- **One pooling rule.** Mean pooling was chosen as the cheapest defensible reduction. Max
  pooling, first-token, or a learned attention pool could move the pooled arm.
- **The nearest-centroid comparison has no interval.** It is reported as a point estimate
  only; the paired bootstrap was run against pooled logistic regression.
- **Held-out use is disclosed.** Test was scored after the method and metric were fixed.
  Earlier exploratory use of validation and test documents during clustering is recorded in
  the Step 7 and Step 8 reports.
- **`Unknown` is included.** The 19-class target treats 75 rare documents as one bucket.
  Excluding it would change every row.
- **The two embedding families use different token counts** (ColPali 1030, ColQwen2 752), so
  their similarity matrices are not on a common scale. Cross-model score comparison is not
  supported; only within-model arm comparisons are.
