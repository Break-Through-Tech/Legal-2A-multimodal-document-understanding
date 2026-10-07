# Implementation plan for embeddings, clustering, and Streamlit

The first delivery extracts reusable embeddings, runs clustering experiments, and displays their results in Streamlit. Classification starts after we review those clustering results. Field extraction is later work.

All experiment-specific source files, configurations, caches, logs, results, and dashboard files stay under `ah-clustering-experiment/`. Persistent storage uses the same experiment directory name. The shared `data/split.csv` supplies document IDs, labels, and partition assignments.

Steps 1 through 4 are complete. Step 5's restart implementation and real local GPU interruption pilot are verified. Full multimodal LayoutLMv3 extraction and Google Drive/Colab recovery remain pending. Steps 6 and 7 have run on the four available full caches: 14 clustering experiments, 14 evaluation artifacts and six shared projections, each covering the 700 training documents. Step 8's five-view Streamlit dashboard is implemented and verified against these saved results in a real browser. See `reports/STEP_8_VERIFICATION.md` for dashboard evidence and `reports/STEP_5_VERIFICATION.md` for the remaining extraction gates. Step 9's findings review remains next.

## Execution order

1. Prepare the dataset and artifact formats.
2. Build the embedding extraction code and the OCR inputs required by multimodal LayoutLMv3.
3. Extract and persist embeddings, with recovery after interruption.
4. Run clustering on explicitly configured representations.
5. Save clustering metrics, assignments, projections, and qualitative examples.
6. Build Streamlit views and inspect the real clustering results.
7. Use the findings to plan classification in a later phase.

## Representations to preserve

| Model or input mode | Saved embeddings | Initial clustering path |
| --- | --- | --- |
| DINOv3 ViT | Native CLS and patch vectors, with special-token metadata | Standard clustering on CLS |
| LayoutLMv3 with image and actual OCR | Multimodal CLS and relevant token outputs, masks, and OCR alignment | Standard clustering on CLS |
| LayoutLMv3 image-only | Verified image-only CLS and relevant token outputs | Standard clustering on CLS |
| ColPali | Retained multi-vector output, masks, and token metadata | Validate document similarity, then use compatible clustering algorithms |
| ColQwen2 | Retained multi-vector output, masks, and token metadata | Validate document similarity, then use compatible clustering algorithms |

Mean pooling is an optional named ablation. It is not required to save embeddings or to fit a fixed experiment matrix. No representation is assumed to improve accuracy or clustering quality before measurement.

## Step 1. Establish isolation and preserve old work. Complete

The `archive/old-experiments` branch already preserves earlier experiment work. The `ah-clustering-experiment/` directory exists. Keep subsequent changes inside that directory and shared data files unchanged.

Existing root caches and environments have unverified ownership. Leave them in place. Inspect archived code before reusing it and record the source revision. Leave changes uncommitted until a commit is requested.

Completion evidence: the archive branch and isolated experiment directory already exist.

## Step 2. Prepare the dataset and define the clustering cohort. Complete

Load the dataset revision documented in `data/README.md`, currently `4ed0d95271ca00107726230f7a0944ed9e90d897`. Join upstream records to `data/split.csv` by document ID. Validate 1,000 unique IDs, 19 final labels, and the existing 700 training, 150 validation, and 150 test assignments.

Preserve original images, final labels, original labels, document quality, and shared partitions. Do not regenerate labels or splits. Keep dataset reference transcriptions separate from actual OCR.

Extract embeddings for all 1,000 documents. Use the 700 training documents as the default initial clustering cohort, so the first dashboard review does not require further inspection of held-out documents. Keep the cohort configurable. Any clustering over all documents must have a separate run identity and an explicit exploratory designation.

Labels evaluate and explain clusters. They do not enter embedding extraction or clustering fitting. Declare how cluster counts are chosen. A class-count-informed setting is acceptable if recorded as such rather than presented as a discovered number of groups.

Record that the test partition has already seen exploratory use. The `Unknown` label merges rare document formats; it is distinct from clustering noise and does not establish unseen-class detection. The benchmark includes business documents beyond the legal subset, and its splits do not group template families.

Completion check: the manifest records exact IDs, partitions, labels, image references, dataset revision, and shared CSV checksum. Duplicate IDs, missing records, and overlapping partitions fail validation.

Completed on 2026-10-04. `experiments/prepare_dataset.py` produced `outputs/metadata/dataset-9c71de8a0c0e5e4aab1d/manifest.parquet` and the dataset and cohort sidecars. All 1,000 saved images match the pinned source byte for byte. The 22 contract and persistence tests pass, and offline reuse passed. The shared CSV is unchanged. The experiment-specific ignore rules and dataset dependencies were added because preparation needed them; the full artifact contracts in step 3 remain pending.

A dedicated audit on the same date fixed acceptance of incomplete provenance/policy metadata and recovery from non-object JSON sidecars. Its 19 independent dataset tests and the combined 81-test suite passed. Fresh preparation, source-unavailable reuse, corruption detection, interrupted repair and full repair were also verified using all 1,000 actual source images in isolated storage. See `reports/STEP_2_AUDIT.md` for current evidence and limits.

## Step 3. Define storage and provenance. Complete

Add the experiment's `.gitignore` before generating data. Track source files, configurations, dependencies, tests, and documentation. Ignore model downloads, OCR caches, numeric arrays, generated images, and large experiment outputs.

Use numeric array shards for embeddings, Parquet for document records and cluster assignments, and JSON for configuration snapshots and provenance. Save offsets and lengths for variable-length embeddings. Keep records joined by document ID rather than relying on row order alone.

Use these output directories under the experiment root:

- `outputs/embeddings/` contains reusable native model outputs.
- `outputs/ocr/` contains actual OCR words, locations, and processing status.
- `outputs/similarities/` contains validated document similarities or distances and their ordered IDs.
- `outputs/clusters/` contains assignments and cluster summaries.
- `outputs/metrics/` contains metrics, coverage, and reasons for undefined values.
- `outputs/projections/` contains PCA or UMAP coordinates and their configuration.
- `outputs/metadata/` contains dataset, artifact, and run manifests.
- `outputs/logs/` contains extraction and experiment logs.

An embedding artifact records the model checkpoint and immutable revision, processor configuration, dataset identity, image identity, OCR identity where relevant, output tensors, dtype, masks, and token metadata. Record transformations applied to images and word coordinates.

A clustering run identifies the checkpoint, saved representation, token selection or pooling policy, normalization, similarity definition if used, algorithm, parameters, document cohort, and seed. Store the Git commit, source digest and dirty state, dependency versions, runtime, and available hardware measurements. A commit hash alone cannot reproduce uncommitted source changes.

Completion check: arrays and document records round-trip with the same IDs and values. Readers reject corrupt shards and incompatible configurations. Changed OCR or preprocessing invalidates the affected embedding cache.

Completed on 2026-10-04. `src/artifacts.py`, `src/contracts.py`, `src/provenance.py` and `src/embeddings/` implement versioned numeric/Parquet/JSON storage, native tensor records, configuration identities, checksums, single-writer publication and execution provenance. `ARTIFACT_FORMAT.md` documents the contract. Local storage verification uses explicit numeric fixtures and direct readers; real model output and inference resume on the intended persistent backend remain steps 4 and 5.

An independent audit on the same date reproduced and fixed missing-shard document loss and non-native byte-order conversion. The final 62-test suite, all-1,000-image source comparison, and offline dataset reuse passed. `reports/STEP_3_AUDIT.md` contains the current evidence ledger and limits.

## Step 4. Build the embedding extraction pipeline. Complete

Implement model loading, input preparation, batched inference, output collection, and artifact writing under `src/embeddings/`. Add model configurations and `experiments/extract_embeddings.py`. Use frozen pretrained models in evaluation mode with gradients disabled.

Start with DINOv3 to prove image loading, embedding extraction, persistence, and cache reading. Add ColPali, ColQwen2, and both LayoutLMv3 modes. Verify official checkpoint documentation and installed APIs before pinning compatible dependencies and model revisions.

Preserve DINOv3 CLS and patch vectors. Record special and register tokens where present. Preserve ColPali and ColQwen2 vectors with masks aligned to their actual outputs. Do not assume an input attention mask matches returned embedding tokens. Record whether prompt and special tokens are present.

For multimodal LayoutLMv3, prepare and cache actual OCR words and boxes with engine version, configuration, confidence where available, coordinate conventions, and image references. Inspect word boxes overlaid on source images. Define empty-OCR handling and truncation or chunking explicitly. Retain parent document IDs for all chunks.

Verify the image-only LayoutLMv3 input path and the exact output used as CLS. Disable automatic processor OCR and exclude document text. Changing cached OCR must leave this mode's embeddings unchanged. Keep this mode unresolved if its intended input path cannot be verified.

OCR is an input preparation dependency for multimodal LayoutLMv3. The other image-input extraction paths can proceed independently. Never substitute reference transcriptions when actual OCR fails.

Completion check: each supported model produces finite outputs with verified shapes, masks, and document mappings on real images. The cache preserves native outputs without silently averaging or normalizing them.

Completed on 2026-10-04. `src/embeddings/models.py`, `pipeline.py`, and `ocr.py` implement all five modes, configured batched inference, actual Tesseract caching and overlays, native output storage, and validated completed-cache reuse. `experiments/extract_embeddings.py`, `prepare_ocr.py`, and `verify_step4.py` provide the runners and independent differential oracle. Immutable checkpoint presets and compatible encoder dependencies are pinned.

All five modes passed real pretrained forward comparisons on training documents using a Colab T4 and batch size 2. Frozen inference, output masks, exact document mapping, image-only OCR invariance, empty OCR and truncation were checked. OCR overlays were inspected and quality limitations recorded. The local suite ran 118 tests successfully with one unavailable-engine skip; all 37 step 4 tests passed on Colab with actual Tesseract. The final source digest and artifact IDs are preserved in `reports/step4-evidence/` and the evidence ledger in `reports/STEP_4_VERIFICATION.md`. All-1,000-document extraction and mounted-storage recovery remain step 5.

## Step 5. Extract embeddings and prove recovery

Run small real-image pilots on the intended compute. Measure memory, throughput, and retained output size. Verify T4 compatibility, precision, and batch limits before full extraction. Estimate persistent-storage needs from the actual output sizes.

Write shards incrementally into the persistent experiment directory. Publish completion manifests only after validating document counts, configuration fingerprints, and checksums. Use one writer per artifact. Treat incomplete or corrupt shards as unfinished.

Interrupt a pilot during a write, reconnect, and resume. Confirm that completed documents do not require inference again. Test recovery on the actual storage backend. Keep temporary files, model caches, and logs inside the experiment directory. Document clone, install, mount, and resume commands. Release compute after use.

After the pilot passes, extract all 1,000 documents for each model and input mode. A complete system can proceed to clustering while another model awaits extraction. Record failures without silently dropping documents.

Completion check: each completed cache contains every expected document ID, with native vectors, masks, and provenance. The interrupted and uninterrupted pilots agree within a declared numerical tolerance and have no duplicate or missing IDs.

Implementation added on 2026-10-06. Extraction commits native batches and OCR commits individual records to checksummed restart journals. Process-lifetime locks exclude concurrent writers; restart repairs unpublished output shards from validated checkpoints, preserves original inference batch boundaries, and recovers the completion/index gap. Corrupt published artifacts remain immutable and fail validation. Per-attempt OCR failures and extraction measurements are retained.

The independent requirement-derived tests and real local pretrained LayoutLMv3 image-only pilot verify restart without repeating completed inference. Real interrupted and uninterrupted tensors matched exactly at declared tolerance `rtol=atol=1e-5`. The local filesystem probe verified writer exclusion, kill/restart and corruption detection. These results do not complete the selected Google Drive remount/reconnect gate. Full OCR extraction still needs to be regenerated and retained on persistent storage. Step 5 remains open until those gates pass.

## Step 6. Implement clustering for the saved representations. Implemented and run on available modes

For fixed-size CLS representations, implement K-means, agglomerative clustering, HDBSCAN, and spectral clustering. Keep normalization and algorithm parameters in configuration. Record defaults before comparing results, and postpone broad hyperparameter searches until baseline runs work.

For ColPali and ColQwen2, define and validate a document-to-document similarity before clustering their retained vectors. Check token masking, directionality, symmetry, self-comparison behavior, score scale, document-length effects, and deterministic results. Specify any symmetrization or conversion to a distance or affinity. Do not call an arbitrary transformed score a metric distance without evidence.

Use only clustering algorithms and linkage or affinity modes compatible with that validated input. Verify the chosen library's requirements. Standard K-means does not become applicable merely because a similarity matrix exists. Unsupported model-algorithm combinations remain explicitly unavailable. Mean-pooled K-means can be added only as a separately configured ablation.

Save document similarities independently from cluster assignments so algorithm changes can reuse them. The runner skips compatible completed runs and records failures. A changed representation, similarity, cohort, or algorithm configuration creates a distinct run.

Completion check: clustering consumes saved artifacts with model inference disabled. Identical inputs reproduce the declared behavior. Invalid matrix shapes, ID order, non-finite values, and incompatible algorithm settings fail validation. ColPali and ColQwen2 clustering results require the similarity checks to pass.

Completed for the four available full caches on 2026-10-06. DINOv3 and image-only LayoutLMv3 each ran K-means, average-linkage agglomerative, HDBSCAN and spectral clustering. ColPali and ColQwen2 each ran the three compatible precomputed algorithms after their 700-document mean-MaxSim similarities passed validation. Standard retained-vector K-means remains explicitly unavailable. All 14 runs contain the exact training IDs and reproduced their partitions and noise membership on repeat fits. The full suite passed 169 tests with one earlier OCR-engine skip. The CLI blocked encoder imports. `reports/STEP_6_METHODS.md` records the predeclared defaults and `reports/STEP_6_VERIFICATION.md` lists artifacts and evidence. Full OCR-mode clustering awaits its missing embedding cache. Step 7 results for the available modes are recorded below.

## Step 7. Save evaluation, projections, and examples. Complete for available modes

Save ARI, NMI, AMI, homogeneity, completeness, and V-measure where labels are available. Record cluster counts, size distributions, and HDBSCAN noise counts and percentages.

For fixed-size features, calculate silhouette, Davies-Bouldin, and Calinski-Harabasz scores when their conditions hold. State the feature representation and distance used. For a validated precomputed distance, use only metrics that support that input. Never compute feature-space scores on arbitrary distance-matrix rows. Undefined or unsupported metrics get a reason rather than an invented number.

State whether noise participates in each score, and show coverage next to results that exclude noise. Save cluster purity, dominant labels, representative documents, and candidate outliers. Declare how representatives and outliers are chosen for each representation. A majority-label mismatch is a review candidate, not proof of an error.

Cache PCA and UMAP coordinates for fixed-size features. For multi-vector results, use only a projection compatible with the validated similarity or distance and label its construction. Record projection parameters and seed. A projection is a visualization, not the feature space for clustering unless explicitly configured as a separate experiment.

Completion check: metrics agree with independently specified examples and can be recreated from saved assignments. Cluster-label permutations preserve agreement scores. Every displayed coordinate and example resolves to the correct document ID.

Completed on 2026-10-06 for the four available modes. `experiments/run_evaluation.py` consumed the exact 14 Step 6 assignments with no cluster fitting or encoder inference. It published 14 metrics/example artifacts and six shared projections: PCA and UMAP for each CLS model and precomputed-dissimilarity UMAP for each retained-token model. Both noise scopes include explicit coverage; unsupported metrics and degenerate geometry have reasons. All repeated projection coordinates matched. The full suite ran 210 tests with no failures and one earlier OCR-engine skip. An independent raw-file audit checked all 280 metric slots, exact training IDs, example selections, projection mappings and source provenance. All 14 metrics artifacts reused with evaluation disabled. See `reports/STEP_7_VERIFICATION.md` for the results and limits. Full OCR-mode evaluation awaits extraction and clustering; Streamlit remains Step 8.

## Step 8. Build the Streamlit clustering dashboard

Build the dashboard after a real extraction and clustering run has saved results. Read completed artifacts under `outputs/`. Display incomplete or failed runs separately.

Provide five views:

- The experiment overview compares models, representations, algorithms, cohorts, seeds, metrics, coverage, and runtime.
- The embedding explorer displays saved PCA or UMAP coordinates with colors for ground truth, cluster, source, or a declared disagreement status. Hover shows document and run details. Selecting a document displays its original image when available.
- The cluster explorer displays sizes, purity, label distributions, representative documents, and candidate outliers.
- The experiment comparison aligns document IDs, checks cohort compatibility, and accounts for arbitrary cluster numbering before showing disagreements.
- The experiment details show checkpoints, representation choices, similarity definitions, parameters, provenance, and artifact status.

Keep figures dynamic. Changing colors, filters, or selected documents must not rerun embedding inference or clustering. Show unavailable metrics and projections with their recorded reasons.

Completion check: use the real Streamlit interface to inspect saved experiments, compare clusters, select images, and handle missing files. The dashboard works while encoder models are unloaded.

Completed on 2026-10-06 for the four available modes. The live interface loads all 14 evaluations and six projections, aligns comparisons by document ID and maximum cluster overlap, and exposes unavailable results and missing-file reasons. All five views, chart selection, real images, filters, representative documents and isolated missing-file controls passed browser checks. The full suite ran 248 tests with no failures and one OCR-engine skip; an independent audit checked 280 saved metric slots, and 48 Streamlit AppTest checks passed. Runtime import guards reject encoder and fitting modules. See `reports/STEP_8_VERIFICATION.md` for the frozen source identity and evidence. Full OCR input remains visibly pending.

## Step 9. Run the clustering comparison and review findings

Complete embedding extraction and the configured clustering runs. Compare the CLS-based baselines first, then add ColPali and ColQwen2 results after their similarity checks pass. Do not force a rectangular model-by-algorithm table when some combinations are unsupported.

Use Streamlit to inspect representative clusters, outliers, visually similar documents with different labels, and semantically related documents with different layouts. Save a findings report under `reports/` with run IDs and document examples. Discuss OCR errors, merged rare labels, cohort limitations, resource costs, and the effects of representation and similarity choices.

Completion check: the dashboard contains real clustering results for each supported model or input mode, with reproducible metrics and examples. Any unresolved model mode or multi-vector similarity remains visibly incomplete. Classification work begins after this review.

## Later work. Classification and field extraction

Retain the proposed classification comparison for a later implementation plan. DINOv3 CLS, multimodal LayoutLMv3 CLS, and image-only LayoutLMv3 CLS will feed logistic regression. ColPali and ColQwen2 retained vectors will feed small attention MIL classifiers. All vectors from a document remain in its shared partition.

Train on the 700 training documents, select settings using the 150 validation documents, and evaluate selected systems on the 150 test documents. Record any additional exploratory use of validation or test documents during clustering. Compare complete model-representation-classifier systems, and save classifier checkpoints, predictions, macro F1, accuracy, per-class metrics, and confusion matrices in that later phase.

Field extraction follows its own model-selection and evaluation protocol. Preserve original images and actual OCR for that work. Classification embeddings do not replace source documents.

## Dependencies and verification

The lead owns the dataset, artifact, configuration, and result contracts. Freeze those contracts after steps 2 and 3. Independent workers can then implement disjoint encoder modules, clustering and evaluation modules, and dashboard components. Assign exact files and checks to each worker. Use one owner for dependencies, GPU sessions, and writes to each artifact.

Prove one DINOv3 embedding-to-clustering-to-dashboard workflow before expanding the comparison. OCR can proceed alongside image-only embedding work. Full multi-vector clustering waits for its similarity validation. Classifier training and classification dashboard views are deferred because the user wants to review clustering first.

```text
ORACLE_SOURCE: User requirements, the shared dataset manifest, official model
              and algorithm contracts, independently specified metric examples,
              and mathematical invariants.
REAL_COMPONENTS: Frozen models, actual OCR where required, persistent storage,
                 clustering algorithms, evaluation, and the Streamlit application.
ALLOWED_MOCKS: Fixtures for isolated development checks. Mocked outputs cannot
               certify real model, OCR, storage, or dashboard integration.
NEGATIVE_CONTROL: Interrupt a write, corrupt a shard, mismatch document IDs,
                  change a cache fingerprint, permute cluster labels, and provide
                  an incompatible similarity matrix or algorithm configuration.
REQUIRED_EVIDENCE_GRADE: Real execution against independent expectations and
                         negative controls. SELF_CHECKED alone is insufficient.
```

Keep an evidence ledger under `reports/` with the claim, expected behavior, source of that expectation, code and configuration identity, verification command, observed result, and remaining limitation.

Checkpoint access, actual OCR support, LayoutLMv3 image-only behavior, compute capacity, persistent storage, and multi-vector similarity validation remain explicit gates. Record failed gates with evidence. A fixture-based dashboard or smoke test alone does not complete this delivery.

## Proposed commands

Dataset preparation, OCR preparation, extraction, clustering, evaluation and Streamlit entry points are implemented. See the README's Step 8 section for dashboard installation and launch instructions. Run these commands from the repository root as their inputs become available.

```text
python ah-clustering-experiment/experiments/prepare_dataset.py
python ah-clustering-experiment/experiments/prepare_ocr.py --manifest ah-clustering-experiment/outputs/metadata/dataset-9c71de8a0c0e5e4aab1d/manifest.parquet --limit 3
python ah-clustering-experiment/experiments/extract_embeddings.py --model dinov3
python ah-clustering-experiment/experiments/run_clustering.py --models dinov3
python ah-clustering-experiment/experiments/run_evaluation.py
streamlit run ah-clustering-experiment/dashboard/app.py
```

The clustering runner and dashboard consume saved embeddings and results. They never trigger encoder inference implicitly. The configured clustering matrix includes only supported representation-algorithm combinations. Missing embedding or similarity artifacts produce an actionable status.
