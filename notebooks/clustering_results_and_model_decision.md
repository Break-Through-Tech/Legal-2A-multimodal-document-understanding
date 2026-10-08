# Clustering Evaluation and Embedding Model Decision

**Project:** Legal Document Intake (Break Through Tech AI Studio, Fall 2026)
**Stage:** September deliverable, embedding head-to-head
**Decision:** Use **SigLIP (`google/siglip-base-patch16-224`)** as the working embedding model. The decision is provisional until the template-aware check (see "Next steps").

---

## 1. What we did, in plain language

Each of the 1,000 document images was turned into a list of numbers (an *embedding*) by two vision models, so that documents that look alike get similar numbers. The embeddings are cached to disk, so nothing is recomputed.

| Model | Checkpoint | Embedding size |
|---|---|---|
| SigLIP | `google/siglip-base-patch16-224` | 768 numbers per document |
| Jina v4 | `jinaai/jina-embeddings-v4` (~3.8B parameters) | 2,048 numbers per document |

Then we asked a simple question: **do these embeddings already group documents by type, without being told the types?** That is what clustering tests. It is a test of the embeddings, not a model we deploy.

## 2. Data and labels

- **Source:** OmniAI OCR Benchmark, 1,000 document images.
- **36 raw document types.** 18 common types hold 925 documents (49-59 each). The other 18 rare types share 75 documents, and several have a single example.
- **19 working classes.** The 18 common types stay as they are. The 75 rare documents are grouped into one class called `Unknown`. The raw type is kept in `original_label`.
- **Split:** `split.csv` assigns every document to train (700), val (150) or test (150), stratified by class. Its row order matches the embedding files (`ids.npy`), which we verified with an assertion before running anything.

### Does the split matter for clustering? No.

Clustering never learns from labels. Labels are used only afterward, to score how well the groups match the true types. Nothing can leak, so we cluster all documents and score against all labels. The split matters from classification onward.

### 18 classes or 19?

`Unknown` is a mix of about 18 unrelated rare types. It is not a real document type, so an algorithm has no reason to find it as one group. We therefore report two setups:

- **18 classes (925 documents), the headline result.** A clean test of whether the embeddings separate real document types.
- **19 classes (all 1,000 documents), a robustness check.** Scores are lower for both models, as expected.

## 3. Algorithms and metrics

**Algorithms**

| Algorithm | Needs the number of clusters? | Notes |
|---|---|---|
| K-means | Yes (k = 18 or 19) | Run on the normalized embeddings |
| Agglomerative (Ward) | Yes (k = 18 or 19) | Run on the normalized embeddings |
| HDBSCAN | No | Run on a 10-dimensional UMAP reduction (cosine metric), `min_cluster_size=10`. It can label documents as noise |

**Metrics**

| Metric | Uses labels? | What it tells us | Better is |
|---|---|---|---|
| ARI (adjusted Rand index) | Yes | Agreement between clusters and true types, corrected for chance | Higher (random is about 0) |
| NMI (normalized mutual information) | Yes | How much knowing the cluster tells us about the type | Higher |
| Purity | Yes | Share of documents that belong to their cluster's majority type | Higher |
| Silhouette (cosine) | No | How tight and well separated the clusters are | Higher |
| Davies-Bouldin | No | Another tightness and separation measure | Lower |

For HDBSCAN we also report the number of clusters found and the noise percentage, because a good-looking score can hide documents the algorithm refused to group.

**Random baseline:** assigning random clusters gives ARI of about 0.001-0.002 in both setups, so every result below is far above chance.

All runs use seed 42.

## 4. Results

### 18 classes (925 documents)

| Model | Algorithm | Clusters | Noise % | ARI | NMI | Purity | Silhouette | Davies-Bouldin |
|---|---|---|---|---|---|---|---|---|
| SigLIP | K-means | 18 | 0.0 | 0.884 | 0.941 | 0.919 | 0.392 | 1.889 |
| SigLIP | Agglomerative | 18 | 0.0 | **0.895** | **0.944** | **0.952** | 0.388 | 1.824 |
| SigLIP | HDBSCAN | 20 | 1.5 | 0.869 | 0.938 | 0.925 | 0.382 | 1.712 |
| Jina v4 | K-means | 18 | 0.0 | 0.813 | 0.896 | 0.884 | 0.603 | 1.072 |
| Jina v4 | Agglomerative | 18 | 0.0 | 0.813 | 0.910 | 0.885 | 0.583 | 1.137 |
| Jina v4 | HDBSCAN | 26 | 0.5 | 0.770 | 0.904 | 0.925 | 0.588 | 1.084 |

### 19 classes (1,000 documents)

| Model | Algorithm | Clusters | Noise % | ARI | NMI | Purity | Silhouette | Davies-Bouldin |
|---|---|---|---|---|---|---|---|---|
| SigLIP | K-means | 19 | 0.0 | 0.808 | 0.901 | 0.886 | 0.348 | 1.949 |
| SigLIP | Agglomerative | 19 | 0.0 | 0.839 | 0.920 | 0.922 | 0.343 | 2.070 |
| SigLIP | HDBSCAN | 20 | 0.6 | 0.693 | 0.873 | 0.827 | 0.325 | 1.901 |
| Jina v4 | K-means | 19 | 0.0 | 0.719 | 0.845 | 0.828 | 0.536 | 1.163 |
| Jina v4 | Agglomerative | 19 | 0.0 | 0.690 | 0.845 | 0.797 | 0.522 | 1.318 |
| Jina v4 | HDBSCAN | 26 | 0.9 | 0.679 | 0.849 | 0.862 | 0.533 | 1.188 |

## 5. How to read the results

1. **SigLIP agrees better with the true types.** It has higher ARI and NMI than Jina in every row, in both setups and with all three algorithms. The ARI gap is about 0.08, which is large. The differences between k-means and agglomerative within one model (about 0.01) are too small to interpret.
2. **Best single result:** SigLIP with agglomerative clustering (ARI 0.895, NMI 0.944, purity 0.952).
3. **Jina makes tighter clusters.** It has higher silhouette and lower Davies-Bouldin. These two metrics do not use labels, so tight clusters can still be the wrong groups. We weight the label-based metrics (ARI, NMI, purity) more, and we report Jina's advantage here honestly.
4. **HDBSCAN works well with UMAP, and it was never told the number of types.** SigLIP with HDBSCAN reaches ARI 0.869 with only 1.5% noise. HDBSCAN over-splits: it found 20 clusters for SigLIP and 26 for Jina, where the truth is 18. The extra clusters are probably separate templates of the same document type, and Jina splits the most.
5. **Both models are weaker with the `Unknown` class included.** This is expected, because `Unknown` is a mixture and not a coherent group.
6. **Preliminary classification check.** A simple logistic regression on either embedding scores about 0.98-0.99 macro-F1 on the test split, against 0.007 for a majority-class baseline. That means the task is close to saturated on a random split and cannot separate the two models. These numbers need to be reproduced in the team notebook.

## 6. Decision

**Use SigLIP (`google/siglip-base-patch16-224`) as the working model.**

- It has higher ARI, NMI and purity across all three algorithms and both setups.
- It is much cheaper to run. Jina v4 is a ~3.8B-parameter model that needs an ~8 GB GPU footprint, while SigLIP-base runs quickly on a free Colab T4 (fill in the actual run times from your logs).
- Jina's only advantage is cluster tightness, which does not measure agreement with document types.

The decision is provisional. It is confirmed or reversed by the template-aware check below.

## 7. Limitations and risks

- **Near-duplicate templates.** Many documents of one type come from the same template. In a quick check, the median similarity between a Jina test document and its closest training document was about 0.995, which suggests near-copies across the split. Scores on a random split may therefore reflect template recognition more than true document understanding. Similarity values from different models are not directly comparable, so treat this as a warning sign, not a measurement.
- **Low resolution in SigLIP.** SigLIP resizes each page to a small square (224x224), so fine text is largely lost and it likely separates documents by layout. That works on this dataset but may fail on legal documents that share a layout. This connects to the project's open question of vision embeddings versus OCR text.
- **Small test classes.** Each class has only about 7-9 test documents, so one document moves a class score by more than 10 points. Single-class results are noisy, so report aggregate metrics and use cross-validation alongside the held-out test.
- **Single seed.** Results use one seed (42). Differences of about 0.01 should not be treated as meaningful.
- **Scope of the dataset.** The benchmark is general business documents. Only part of it is legal, and several common types (for example charts and nutrition labels) are not legal documents.

## 8. Next steps

1. **Classification on the fixed split.** Train on train, tune on val, score test once. Report macro-F1, per-class F1, a confusion matrix, and a slice by `document_quality`. Baselines: majority class, 1-nearest-neighbor, then logistic regression.
2. **Template-aware check.** Cluster within each class to form template groups, keep each group on one side of the split, and compare random-split and grouped-split macro-F1 for both models. This is the experiment that can confirm or reverse the SigLIP decision.
3. **Out-of-scope detection.** Agree as a team on which types are in scope. Train only on in-scope types, then test whether the other types get flagged. Report ROC-AUC, precision and recall, with the threshold chosen on val and reported on test.
4. **Extraction and validation.** Prompt a vision-language model with each document's schema, validate the output with `jsonschema`, and score field-level accuracy.
5. **Front end.** A small app that shows the predicted type, confidence, a recognized or needs-review flag, and the extracted fields next to the document.

## 9. Reproducibility checklist

- Seed: 42 for every run.
- Embedding files: `siglip_siglip-base-patch16-224.npy` (1000 x 768), `jina_v4.npy` (1000 x 2048), `ids.npy` (1000), all L2-normalized before clustering.
- Labels and split: `split.csv` (`label` = 19 classes, `original_label` = 36 raw types), row order verified against `ids.npy`.
- UMAP for HDBSCAN: 10 components, cosine metric, seed 42.
- **To fill in before submitting:** exact versions of `transformers`, `torch`, `scikit-learn` and `umap-learn` from your Colab runtime, and the embedding run times.
