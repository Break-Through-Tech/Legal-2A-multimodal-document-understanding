# Anh's document clustering experiment

To clone this repository and explore the saved results without a GPU, follow [the dashboard quickstart](DASHBOARD_QUICKSTART.md). The repository includes the saved score tables, assignments and projections; the setup command downloads document images separately.

Steps 2 and 3 prepare the shared dataset and define storage and provenance. Steps 4 and 5 add pretrained extraction, actual OCR, and restart checkpoints. Steps 6 through 8 provide 18 clustering experiments, saved evaluations, eight projections and a verified Streamlit dashboard for five available modes using the 700 training documents. See [step 8 verification](reports/STEP_8_VERIFICATION.md) and the [OCR clustering recovery](reports/OCR_CLUSTERING_RECOVERY.md). Full OCR embeddings and downstream results are complete. Drive backend verification remains a separate [step 5 gate](reports/STEP_5_VERIFICATION.md); the Step 9 findings review and classification remain later work.

## Install the dataset dependencies

Use Python 3.12 or newer and an environment inside this directory for new installations. Dataset preparation was verified with Python 3.14. From the repository root on Windows:

```powershell
python -m venv ah-clustering-experiment/.venv
ah-clustering-experiment/.venv/Scripts/python.exe -m pip install -r ah-clustering-experiment/requirements.txt
ah-clustering-experiment/.venv/Scripts/python.exe ah-clustering-experiment/experiments/prepare_dataset.py
```

With an existing environment that has these dependencies, use:

```text
python ah-clustering-experiment/experiments/prepare_dataset.py
```

The same command works from the experiment directory as `python experiments/prepare_dataset.py`. These dependencies cover dataset preparation and artifact storage. Encoder installation is described below.

## Dataset and clustering cohort

The source is `getomni-ai/ocr-benchmark` at revision `4ed0d95271ca00107726230f7a0944ed9e90d897`. Its upstream split named `test` contains all 1,000 documents. The team's partitions come from the shared `data/split.csv` and contain 700 training, 150 validation, and 150 test documents.

The preparation command joins records by ID and preserves the final label, original label, document quality, and partition. It checks normalized source metadata against the shared CSV. It never recreates the shared split.

The manifest covers all 1,000 documents so embedding extraction can process each image once. The default clustering cohort contains only the 700 training IDs. Labels support evaluation and inspection; they must not be passed into model inference or clustering fitting. The number of classes does not silently set the cluster count. Clustering configuration must state whether that count was chosen independently or with class-count information.

Choosing `--cohort all` creates a separate exploratory cohort. It does not change the team's saved assignments. Inspection of that cohort includes validation and test documents and must be disclosed in later supervised results.

Settings live in `configs/dataset.json`. To use mounted persistent storage, pass `--storage-root` with a directory named `ah-clustering-experiment`. For example, on Colab:

```text
python ah-clustering-experiment/experiments/prepare_dataset.py --storage-root /content/drive/MyDrive/ah-clustering-experiment
```

The preparation command assumes a single writer per storage directory. The shared CSV stays in the repository. `--split-csv` can point to an explicit copy when the repository layout differs.

## Saved data

All caches and outputs stay under this experiment directory or a configured persistent directory named `ah-clustering-experiment`.

- Original image files retain their source bytes. The manifest records relative file paths, checksums, dimensions, and document IDs.
- `outputs/metadata/` contains the document manifest in Parquet, dataset provenance in JSON, and clustering cohort IDs in JSON.
- `cache/` contains the pinned upstream data and download caches.

Dataset reference transcriptions and extraction targets remain reference data in the upstream cache. They are excluded from the prepared document manifest and cannot stand in for actual OCR.

Generated data stays out of Git. Source, configuration, this README, and verification reports remain reviewable. Repeating preparation verifies compatible saved artifacts before reusing them.

The current dataset artifact is `outputs/metadata/dataset-9c71de8a0c0e5e4aab1d/`. It contains `manifest.parquet`, `dataset.json`, and separate cohort JSON files. The preparation command prints the exact paths and whether it reused the dataset. Image paths in the manifest are relative to the storage root.

## Verify the preparation code

From the repository root with the dataset dependencies installed:

```text
python -m unittest discover -s ah-clustering-experiment/tests -v
```

The tests exercise ID joins, partition validation, cohort selection, byte preservation, reuse, and repairs after deliberate corruption. Small fixture images test failure handling. The separate `tests/verify_real_dataset.py` checks actual saved images against the pinned source and the shared CSV. Exact commands and results are in the step 2 evidence report.

## Research limitations

The test partition has already seen exploratory use. The merged `Unknown` class contains 75 documents across multiple rare formats; it is not clustering noise or proof of unseen-class rejection. The dataset includes business documents beyond the legal subset. The shared partitions do not group related template families.

See [the artifact format](ARTIFACT_FORMAT.md) for storage APIs, tensor layouts, cache identities and provenance. Recovery now skips validated inference batches and OCR records. Corrupt partial checkpoints are unfinished and recomputed; corrupt published artifacts fail validation.

See [the implementation plan](IMPLEMENTATION_PLAN.md) for later steps, [the dedicated step 2 audit](reports/STEP_2_AUDIT.md) for dataset results, and [the independent step 3 audit](reports/STEP_3_AUDIT.md) for storage verification. The step 3 combined suite passed 81 tests; the step 4 report records the expanded suite. The step 2 audit also rebuilt and repaired all 1,000 real images in isolated storage, verifying exact original bytes and shared assignments.

## Step 4: extract pretrained embeddings

Use Python 3.13 and an environment inside the experiment directory. Install `requirements-embeddings.txt`; it includes the dataset requirements. The verified GPU runtime is Colab T4 with PyTorch 2.11.0, Transformers 5.17.0 and CUDA 13.0. The retrieval models use float16 and batch size 1 by default. DINOv3 and LayoutLMv3 use float32. Native Transformers classes handle every model; no ColPali engine or adapter checkpoint is required.

```text
python -m pip install -r ah-clustering-experiment/requirements-embeddings.txt
python ah-clustering-experiment/experiments/extract_embeddings.py --model dinov3 --device cuda:0 --document-ids 0 1
python ah-clustering-experiment/experiments/extract_embeddings.py --model colpali --device cuda:0 --document-ids 0 1
python ah-clustering-experiment/experiments/extract_embeddings.py --model colqwen2 --device cuda:0 --document-ids 0 1
python ah-clustering-experiment/experiments/extract_embeddings.py --model layoutlmv3-image --device cuda:0 --document-ids 0 1
python ah-clustering-experiment/experiments/extract_embeddings.py --model layoutlmv3-ocr --device cuda:0 --document-ids 0 1
```

The immutable checkpoint pins and processor overrides live in `configs/models/`. DINOv3 requires approved access to its official Hugging Face repository and an authenticated environment. Do not put credentials in configuration files. For CPU pilots use `--device cpu`; large retrieval models need substantially more RAM and time. `--batch-size` and `--dtype` change the artifact identity. Omitting `--document-ids` requests every manifest ID, including held-out partitions, for the full extraction in step 5. Pilot IDs 0, 1 and 2 are training documents.

Multimodal LayoutLMv3 requires Tesseract and English traineddata. On Colab these are supplied by `tesseract-ocr` and `tesseract-ocr-eng`; Windows can use an installed executable through an OCR JSON configuration (`executable`, optional `tessdata_dir`, `language`, `psm`, `oem`, `timeout_seconds`). Inspect actual OCR before interpreting multimodal results:

```text
python ah-clustering-experiment/experiments/prepare_ocr.py --manifest ah-clustering-experiment/outputs/metadata/dataset-9c71de8a0c0e5e4aab1d/manifest.parquet --limit 3
```

OCR records preserve words, normalized XYXY boxes, per-word confidence, empty status, engine version, executable/traineddata checksums and original image identity. Overlays live beside the OCR artifact. A failed OCR call fails extraction; reference text is never substituted. Successful empty OCR uses text special tokens plus visual tokens. Text truncates on the right to the first 512 subtokens including special tokens; original word count, retained word alignment, truncation status and parent ID are saved. There is no chunk expansion in this version.

Image-only LayoutLMv3 disables processor OCR and passes only pixels. Its CLS is the visual token at index 0. Multimodal CLS is the text token at index 0; the full saved sequence also retains the visual CLS and patches. DINOv3 stores separate CLS, register and patch arrays. ColPali and ColQwen2 retain native projected vectors, prompt/special tokens, input IDs, image/special masks and an output-aligned attention mask. Their upstream forward already normalizes vectors; this pipeline adds no pooling or normalization.

Completed extraction requests reuse validated artifacts without loading encoders. Changed checkpoint, preprocessing, OCR, selected IDs, dependencies, device, batch size or source digest creates a new request. Image-only identities ignore OCR configuration. Restart the same command with the same source and environment to reuse completed batches from an interrupted run. Direct cache reading with `src.embeddings.read_embeddings` needs only storage dependencies; the extraction runner also checks installed encoder dependency versions.

The independent real-component check reloads the official model separately and compares tensors, masks and IDs, then runs negative controls and cache checks:

```text
python ah-clustering-experiment/experiments/verify_step4.py --config ah-clustering-experiment/configs/models/layoutlmv3-image.json --dataset-dir ah-clustering-experiment/outputs/metadata/dataset-9c71de8a0c0e5e4aab1d --ids 0 1 --batch-size 2 --device cuda:0 --report ah-clustering-experiment/outputs/logs/oracle-layoutlmv3-image.json
```

Use `--storage-root` for a prepared persistent directory named `ah-clustering-experiment`. [Model API notes](reports/MODEL_API_NOTES.md) and [OCR API notes](reports/OCR_API_NOTES.md) link the authoritative upstream contracts.

## Step 5: restart and persistent storage

Run the same extraction command after an interruption. A complete checkpoint batch requires no encoder call. Its original batch boundaries remain fixed, so padding and output masks match uninterrupted extraction. OCR checkpoints commit one document at a time. A killed write without a valid checksum receipt is unfinished. Resume replays valid checkpoints into final shards, validates exact IDs and checksums, then publishes completion. If publication finished before the extraction index was written, restart restores that index without inference.

Keep the exact source snapshot and dependency versions for a resumed run. The source digest includes tests and configuration, so even an unrelated source edit creates a new request. Use one compute host and one writer for an artifact. Restart checkpoints remain beside final outputs, so reserve roughly twice the retained tensor bytes, plus images, OCR, model downloads, metadata and temporary write space. Logs record inference duration, reused/inferred counts, tensor sizes and available CUDA peak memory. A tensor-only size projection is not the total disk requirement.

For a fresh Colab runtime, clone into an experiment-owned work directory. An ordinary clone contains only committed files; this uncommitted implementation must first be copied as an exact source snapshot, including tests and configuration. Do not expect a clone of the current remote revision to contain it.

```bash
mkdir -p /content/ah-clustering-experiment/cache
git clone https://github.com/Break-Through-Tech/Legal-2A-multimodal-document-understanding.git /content/ah-clustering-experiment/cache/repository
cd /content/ah-clustering-experiment/cache/repository
python -m venv ah-clustering-experiment/.venv --system-site-packages
ah-clustering-experiment/.venv/bin/python -m pip install -r ah-clustering-experiment/requirements-embeddings.txt
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng
```

Mount Drive in a notebook cell and complete Google's interactive authorization:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then prepare the persistent dataset and test that backend's filesystem operations:

```bash
export OMP_THREAD_LIMIT=1
PY=ah-clustering-experiment/.venv/bin/python
STORE=/content/drive/MyDrive/ah-clustering-experiment
$PY ah-clustering-experiment/experiments/prepare_dataset.py --storage-root "$STORE"
$PY ah-clustering-experiment/experiments/verify_recovery_storage.py --storage-root "$STORE"
$PY ah-clustering-experiment/experiments/extract_embeddings.py --model dinov3 --device cuda:0 --batch-size 1 --document-ids 0 1 2 --storage-root "$STORE"
```

The filesystem probe kills a real writer process, checks exclusion of a second writer, resumes, and rejects an interrupted and a deliberately corrupt payload. Passing it proves same-host process recovery only. It does not certify Drive remounts, power-loss durability or writers on different hosts. Validate real model interruption on the intended backend before beginning full extraction.

After the pilot passes, omit `--document-ids` to request all 1,000 documents. Use a 600-second OCR timeout for this dataset, as the prior full OCR run exceeded 120 seconds on one document:

```bash
printf '{"timeout_seconds":600}\n' > "$STORE/cache/ocr-options.json"
for MODEL in dinov3 colpali colqwen2 layoutlmv3-image layoutlmv3-ocr; do
  $PY ah-clustering-experiment/experiments/extract_embeddings.py --model "$MODEL" --device cuda:0 --batch-size 1 --storage-root "$STORE" --ocr-config "$STORE/cache/ocr-options.json" || break
done
```

After reconnecting, remount the same Drive location, restore the same source snapshot and dependencies, and repeat the interrupted command. The CLI prints the completed artifact ID only after full validation. Failures remain in `outputs/logs/`. Never delete a writer lock based only on its age: the pipeline removes only a lock matching its recorded ownership. An unknown legacy lock requires confirming its owner has stopped before manual removal. Release Colab compute after verifying that outputs are on persistent storage.

## Step 6: cluster saved embeddings

Clustering uses the five completed full caches already on disk. The default cohort is exactly the shared 700 training documents. Install the additional clustering dependency in the experiment environment and run:

```powershell
ah-clustering-experiment/.venv/Scripts/python.exe -m pip install -r ah-clustering-experiment/requirements-clustering.txt
ah-clustering-experiment/.venv/Scripts/python.exe ah-clustering-experiment/experiments/run_clustering.py --similarity-device cuda:0
```

CUDA here computes similarities from saved token vectors; encoder module loading is disabled. With no GPU use `--similarity-device cpu`, which is slower for retained multi-vectors. Run only CLS baselines with `--models dinov3 layoutlmv3-image`. Configuration, normalization and algorithm parameters are explicit in `configs/clustering.json`. `--cohort all`, `val` or `test` creates a separate exploratory identity; the default remains `train`.

CLS baselines support K-means, average-linkage agglomerative clustering, HDBSCAN and spectral clustering. Fixed-count methods use 19 clusters, recorded as class-count-informed. HDBSCAN discovers its count using predeclared density parameters. ColPali and ColQwen2 use validated attended-image-token mean-MaxSim, symmetric averaging, nonmetric `1-S` distance and `(S+1)/2` affinity. Compatible precomputed algorithms are agglomerative, HDBSCAN and spectral; standard K-means is unavailable for those retained-vector inputs.

Similarities are saved independently under `outputs/similarities/`; assignment tables live under `outputs/clusters/<artifact_id>/assignments.parquet`, with configurations and provenance in `outputs/metadata/`. The runner reuses verified completed artifacts, records failures, and repeats each new fit to verify its partition and noise membership. A clustering-only configuration edit can reuse the same similarity matrices. Original pixels are not required after extraction.

See [step 6 methods](reports/STEP_6_METHODS.md) for the mask policy, length effects, matrix validation and upstream API contracts. Results and limits are recorded in [step 6 verification](reports/STEP_6_VERIFICATION.md).

## Step 7: evaluate saved assignments

Install the evaluation dependencies in the experiment environment and run:

```powershell
ah-clustering-experiment/.venv/Scripts/python.exe -m pip install -r ah-clustering-experiment/requirements-evaluation.txt
ah-clustering-experiment/.venv/Scripts/python.exe ah-clustering-experiment/experiments/run_evaluation.py
```

The configuration in `configs/evaluation.json` names the 18 verified clustering artifacts explicitly. `--models dinov3` selects a subset; `--storage-root` selects a prepared directory named `ah-clustering-experiment`. The shared CSV remains at the repository's `data/split.csv`. This runner uses local CPU computation, blocks encoder imports, and never refits clusters. It checks upstream artifacts before consuming them, so the first read of a large native embedding cache can take time.

Metrics and examples live under `outputs/metrics/<artifact_id>/`: `scores.json` records both noise policies and coverage, `clusters.parquet` contains label distributions and representative/outlier IDs, and `documents.parquet` joins IDs to labels, assignments, image references and example ranks. Undefined or unsupported scores have a null value and an explicit reason. Noise is distinct from the benchmark's `Unknown` label. Majority-label mismatches and farthest-from-medoid examples are review candidates, not proven errors.

Reusable projections live under `outputs/projections/<artifact_id>/coordinates.parquet`, with explicit document IDs and `x`, `y` columns. CLS models have PCA and UMAP; retained-token models have UMAP over their validated nonmetric dissimilarities. These visualizations never replace clustering features. Every new projection repeats with the same seed and checks its coordinates. Algorithms sharing a representation share projection artifacts, and completed results are validated before reuse.

The invocation prints its summary path under `outputs/logs/step7-*.json`. Configuration, checksums and provenance remain under `outputs/metadata/`. See [evaluation methods](reports/STEP_7_METHODS.md) and [verification](reports/STEP_7_VERIFICATION.md). Streamlit consumes these saved files in Step 8.

## Step 8: inspect the Streamlit dashboard

```powershell
ah-clustering-experiment/.venv/Scripts/python.exe -m pip install -r ah-clustering-experiment/requirements-dashboard.txt
ah-clustering-experiment/.venv/Scripts/python.exe -m streamlit run ah-clustering-experiment/dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none --theme.base light --theme.primaryColor '#147D73'
```

Open **http://127.0.0.1:8501**. The dashboard reads completed results from this experiment directory. To use another prepared store, set `$env:AH_CLUSTERING_STORAGE_ROOT='D:/path/ah-clustering-experiment'` before launching. The storage directory must retain that final name. Keep its configs, output payloads and metadata together. To stop the local server, press Ctrl+C in its terminal.

Use the sidebar to switch among the overview, embedding explorer, cluster explorer, experiment comparison and experiment details. Select input modes and an active run. The metric noise policy controls saved scores and shows coverage; it does not alter the original clustering. Overview filters select whole runs. Document filters in the explorer affect only what is displayed.

In the embedding explorer, select PCA or UMAP, choose a color mapping, and click a point or use the document picker to open the original image. Cluster review shows label distributions, medoid representatives and farthest-member candidates. Comparison checks exact cohort identity and aligns cluster numbers by maximum overlap, with noise and unmatched groups handled separately. Details preserve configuration, unavailable-score reasons and execution provenance.

The app blocks encoder and fitting imports and never recomputes embeddings, clustering, metrics or coordinates. It caches reads for up to 30 seconds; **Refresh saved results** immediately rechecks the files. Missing or corrupt outputs are shown as unavailable, and an absent image does not hide its document record. OCR-mode results are included. Standard retained-vector K-means remains unsupported for ColPali and ColQwen2.

The independent AppTest checks and raw artifact audit are documented in [Step 8 verification](reports/STEP_8_VERIFICATION.md), alongside the real browser evidence. Results interpretation and a findings report remain Step 9.
