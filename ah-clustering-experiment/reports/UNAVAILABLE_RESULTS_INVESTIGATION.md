# Investigation of pending and unavailable results

Investigated October 6, 2026 (America/Chicago). No pipeline code, configuration, or experiment outputs were changed.

## Findings

### LayoutLMv3 OCR: stale configuration after cache recovery

The full cache `embeddings-14f0c9c4cc8cb38ae601` is present and valid. A fresh `read_artifact` and `_decode` check verified 1,817,351,161 payload bytes, checksums, tensor schemas, exactly document IDs 0–999, and all 700 training image records against the prepared dataset. Each document has a `(1, 768)` CLS tensor. The dataset identity and manifest hashes match.

The recovery report records publication at 2026-10-07T00:18:21Z, which is October 6 at 7:18 p.m. Chicago time. The earlier clustering and evaluation configuration was not updated afterward.

The cause is visible at three boundaries:

- `configs/clustering.json:5` lists four models and omits `layoutlmv3-ocr`.
- `experiments/run_clustering.py:60` labels the OCR cache unavailable solely because the model is absent from configuration; it does not inspect cache availability.
- `configs/evaluation.json:75` retains the old pending message, and `dashboard/data.py:214` appends that message unconditionally.

No saved clustering artifact currently references the recovered OCR embedding ID. Extraction is complete; downstream clustering and evaluation remain pending.

Recovery requires registering the existing embedding ID, running only the OCR model's clustering, registering its resulting clustering IDs for evaluation, and evaluating those runs. The existing CLS path supports this cache. OCR and embedding inference do not need repeating. Status reporting should distinguish a missing cache from an unconfigured or unfinished downstream run.

### ColPali and ColQwen2: deliberate K-means restriction

The current experiment retains image-token vectors and converts token comparisons into pairwise mean-MaxSim similarities, distances, and affinities. It does not pool each document into one feature vector. `src/clustering/experiment.py:154` explicitly skips standard K-means for this representation, and `src/clustering/runner.py:36` also rejects precomputed input for K-means.

This follows `IMPLEMENTATION_PLAN.md:126`. Standard [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) expects a sample-by-feature matrix. Treating a pairwise matrix as features would change the experiment's geometry.

Fresh checksummed reads confirmed all six saved runs below, each covering exactly the 700 training IDs:

| Model | Agglomerative | HDBSCAN | Spectral |
| --- | --- | --- | --- |
| ColPali | 19 clusters, 0 noise | 19 clusters, 75 noise | 19 clusters, 0 noise |
| ColQwen2 | 19 clusters, 0 noise | 20 clusters, 76 noise | 19 clusters, 0 noise |

Mean-pooled, normalized features could enable a separately named K-means ablation, as permitted by the plan. This requires an explicit representation path and provenance; changing K-means parameters alone will not enable it. No quality improvement is established. Spectral clustering's internal K-means assignment uses a spectral embedding and does not contradict the restriction.

## Evidence ledger

- REVISION: Git `d0650e86581b6ca89eb1ae2b6f0bef45c73727b7`, with existing uncommitted work; runnable source digest `9815a43e8ac096b331e5a32811894b47fb541f4dd76e428a578fc6774797d83a`.
- CLAIM: Each displayed row is explained by either stale configuration or a deliberate representation restriction.
- ORACLE_SOURCE: Prepared dataset IDs/image identities, preexisting cache and run artifacts, recorded implementation requirements, and the official KMeans input contract.
- REAL_COMPONENTS: Actual saved payloads, artifact reader, embedding decoder, cohort loader, dashboard catalog reader, and clustering input validator.
- ALLOWED_MOCKS: None.
- NEGATIVE_CONTROL: Remove a training document from the in-memory cache document map; reject pairwise distance input for direct K-means.
- REQUIRED_EVIDENCE_GRADE: INDEPENDENTLY_CHECKED for the diagnosis.
- EVIDENCE_GRADE: INDEPENDENTLY_CHECKED.
- CHECKS_RUN: Full recovered OCR artifact validation; live `scan_catalog` reproduced exactly the three reported issues and validated 14 completed runs; six retrieval-model assignment artifacts checked; direct float64 saved-token pair comparisons agreed with persisted scores within 2.74e-8; fixed-feature K-means positive control passed.
- NEGATIVE_CONTROL_RESULT: Missing training document rejected with `Cohort images differ from embedding cache`; the three-document pilot also lacked 697 training IDs; direct K-means rejected precomputed distances with the expected compatibility error.
- MOCKS_AND_LOST_COVERAGE: No mocks. No model inference, full new clustering runs, or browser rendering was performed.
- CONTRADICTORY_EVIDENCE: The old missing-cache message conflicts with the now-valid recovered artifact; the configuration/display path explains the contradiction.
- RESIDUAL_RISKS: OCR clustering quality and future repair behavior remain untested because this was an investigation. The saved OCR cache is usable, but its downstream results have not been generated.
- VERDICT: PASS for diagnosis; no repair claimed.
