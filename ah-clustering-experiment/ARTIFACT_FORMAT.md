# Artifact storage contract, version 1

Step 3 defines persistence and provenance. It does not load encoders, perform OCR, or fit clustering algorithms. The existing step 2 dataset format remains unchanged.

## Layout and publication

`src.artifacts.initialize_storage(root)` creates the eight output directories in the implementation plan. The storage root must be named `ah-clustering-experiment`. Generated images remain in `outputs/images/` as established in step 2. All output paths are relative to that root.

| Directory | Contents |
| --- | --- |
| `outputs/embeddings/<artifact_id>/` | Native numeric tensor shards and document records |
| `outputs/ocr/<artifact_id>/` | Actual OCR words, boxes, confidence and processing status |
| `outputs/similarities/<artifact_id>/` | Numeric matrices and explicit ordered document IDs |
| `outputs/clusters/<artifact_id>/` | Parquet assignments and summaries |
| `outputs/metrics/<artifact_id>/` | JSON metrics, coverage and reasons for undefined results |
| `outputs/projections/<artifact_id>/` | Parquet coordinates joined by document ID |
| `outputs/metadata/<artifact_id>/` | `config.json`, `artifact.json` and a writer lock while active |
| `outputs/logs/` | Extraction and experiment logs |
| `outputs/checkpoints/<request_or_ocr_id>/` | Restart payloads, checksum receipts and a process-lifetime lock |

`artifact_identity(kind, config)` hashes canonical JSON with SHA-256. Its input is `{"schema_version": 1, "kind": kind, "config": config}`. Canonical JSON sorts keys, uses compact separators, and rejects non-finite numbers. The directory ID is `<kind>-<first 20 hex characters>`; the manifest retains the full identity hash and checks it on read. Configuration is copied on entry so caller mutations cannot change a running writer.

`ArtifactWriter` writes each payload through a flushed, synchronized sibling temporary file and atomic replacement. An exclusive `writer.lock` prevents two writers from publishing the same identity. Before publication, it rereads all files, checks checksums and schemas, and runs an optional domain validator. The completion manifest is the last write. Readers require that manifest and reject partial artifacts.

Completed artifacts are immutable. Call the reader to validate and reuse them. A failed context-managed write removes its lock and leaves unpublished files; a retry may overwrite those files with the same configuration. A killed process can leave an artifact lock. The extraction journal can recover a lock only when its recorded checksum matches the lock's unique acquisition token. Unknown locks require stopped-owner confirmation before manual removal. Filesystem assumptions must be verified on the intended mount.

## File formats

- Numeric arrays are `.npy`, written and loaded with `allow_pickle=False`. Supported dtypes are NumPy boolean, integer and floating types. Arrays must have at least one axis and contain finite values. There is no implicit dtype conversion, normalization or pooling. A future encoder emitting a dtype NumPy cannot represent must define an explicit encoding before use.
- Document records, assignments and coordinates are Parquet. Tables default to a required, unique, non-null `document_id` key. Tables with another natural key must declare it; word-level OCR tables can use a composite key encoded as a separate column. No downstream component may infer document identity from row order.
- Configurations, provenance, summaries and metric values are JSON. An undefined metric uses `value: null` and a `reason`; its coverage and noise policy belong in the same result.

Each payload descriptor records a relative path, byte count and SHA-256. Numeric descriptors also record dtype and shape; Parquet descriptors record columns, row count and key. The completion manifest has its own canonical digest, the configuration file checksum, full artifact identity and execution provenance. These hashes detect corruption; they are not cryptographic signatures.

Readers validate every referenced payload before returning. Numeric arrays are memory-mapped read-only; document tensor slices retain those mappings. Callers must release returned arrays before replacing files or deleting their storage directory on Windows.

The generic bundle layer enforces storage integrity. OCR semantics, similarity mathematics, clustering algorithm compatibility, and metric validity are separate domain checks for steps 4, 6 and 7. Those producers should supply a validator to `ArtifactWriter.complete`. The generic writer alone does not certify such results.

## Embedding identity and records

Use `src.contracts.embedding_config` to build the configuration and `src.embeddings.write_embeddings` / `read_embeddings` to persist or read it. The configuration includes:

- Model checkpoint and immutable 40- or 64-character hexadecimal revision.
- Full processor settings, image transformations, and word-coordinate transformations for multimodal OCR inputs.
- Dataset ID, full dataset identity and manifest checksum from step 2.
- Sorted document IDs with original image paths and SHA-256 values. Labels, reference transcriptions and targets are excluded from these input records.
- Input mode (`image_only` or `image_and_ocr`), output tensor schema, static token policy, and source digest.
- For multimodal inputs, the actual OCR artifact ID, content digest, engine, version and configuration. Use the validated OCR completion manifest's `manifest_sha256` as `content_sha256`, so changes in words, boxes, processing status or configuration invalidate embeddings. Never identify OCR using its filename alone.

Image-only identity deliberately excludes OCR. Changing OCR alone therefore changes multimodal identity and leaves image-only identity unchanged. Image checksums, preprocessing, model revision, tensor schema and source changes create different embedding identities. IDs are sorted for configuration identity, while tensor records remain joined explicitly by ID.

Each output tensor declares `dtype`, `tail_shape`, `role` and `alignment`. Axis 0 is the retained sequence length. A single CLS vector of width D has shape `[1, D]` and `tail_shape: [D]`; patches have shape `[patch_count, D]`. Output masks have boolean dtype and `tail_shape: []`. Tensors in the same alignment group must have identical sequence lengths for each document. Input masks cannot be substituted for output masks without checking that alignment.

Each input to the writer has exactly this structure:

```python
{
    "document_id": 19,
    "tensors": {"cls": cls_array, "patches": patch_array, "patch_mask": mask_array},
    "token_metadata": {
        "special_indices": [],
        "register_indices": [],
        "prompt_indices": [],
        "ocr_word_alignment": None,
    },
}
```

The actual tensor names must match the configured schema. Token metadata must describe the actual outputs. Empty sequences are supported and should record a reason. Future chunked outputs must keep the parent document ID and record chunk boundaries and OCR alignment in token metadata; one document remains one record in this format.

Within each shard, a tensor's sequences are concatenated along axis 0. Files are named `shard-00000-<tensor>.npy`. `documents.parquet` stores `document_id`, image identity, shard index, JSON token metadata, and `<tensor>__offset` / `<tensor>__length` for each tensor. Record order is irrelevant. Readers verify exact expected IDs, image identities, contiguous spans without overlaps, complete shard coverage, tensor schemas and alignment groups. The writer snapshots producer buffers before advancing its iterator.

```python
from pathlib import Path
from src.embeddings import write_embeddings, read_embeddings
from src.provenance import source_identity

source_root = Path("ah-clustering-experiment").resolve()
# Build config with embedding_config(...,
#     source_digest=source_identity(source_root)["source_digest"]).
# Build native_outputs from a verified encoder in step 4.
artifact_id = write_embeddings(
    source_root, config, native_outputs, source_root=source_root, shard_size=32,
)
outputs_by_id = read_embeddings(source_root, config)
```

Shard size is a storage choice, not a representation change. All expected documents must be present before completion. The low-level writer consumes an iterable; the extraction pipeline supplies validated restart batches from `src.embeddings.checkpoints`.

## Restart checkpoint format

`RecoveryJournal` holds an OS advisory lock on `process.lock` through checkpoint reading, inference and publication. The file remains on disk but its lock releases when the process ends. The identity file binds the journal to a full configuration fingerprint. This assumes one host and a backend supporting advisory locks and atomic replacement; there is no cross-host distributed lease.

Embedding batches use `batch-<offset>.bin`, an uncompressed NPZ with pickle-free native arrays and UTF-8 JSON metadata encoded as a uint8 array. The companion JSON receipt records payload bytes, SHA-256 and identity SHA-256. Payload replacement precedes receipt replacement. No receipt, a checksum mismatch, invalid tensors, wrong IDs, or invalid masks makes the batch unfinished. The original batch grouping remains fixed on retry. The configuration snapshot uses `embedding-config.bin` and the same receipt protocol.

OCR uses `document-<id>.bin` containing JSON words, boxes, confidences, status and image identity. A valid receipt alone is insufficient: the OCR validator also checks the record and original image dimensions before reuse. OCR failure logs preserve the completed IDs and current failing ID for each attempt.

Restart checkpoints are retained after publication. Readers of completed artifacts do not depend on them. They currently duplicate retained tensors; storage planning must include both checkpoint and final payload bytes. Checkpoints are not advertised as complete embedding or OCR artifacts.

## Clustering and downstream results

`src.contracts.clustering_config` requires the checkpoint identity, embedding artifact and full identity, representation, token selection, pooling, normalization, optional similarity identity and definition, algorithm, parameters, cohort, seed, count-selection policy and source digest. Explicit `{"kind": "none"}` policies distinguish no transformation from missing configuration. If parameters specify `n_clusters`, it must agree with the declared count policy.

The cohort carries its dataset ID, partition selector and exact document IDs. Validation, test and all-document cohorts must declare `exploratory`. Sorted membership enters the run identity. Labels remain evaluation metadata. A similarity artifact must store a separate Parquet `ordered_ids` table with unique IDs and positions; row and column position must resolve through that table. Its producer must verify matrix shape, order and mathematical properties before completion.

Use `ArtifactWriter(root, "clusters", run_config)` with a Parquet assignments table keyed by `document_id`, then `complete(provenance, validator=...)`. Cluster summaries may use `cluster_id` as their declared unique key. Metrics and projections should identify the originating run, cohort and their own calculation settings in configuration; changing those settings creates a distinct artifact. Algorithm and evaluation implementations remain later work.

## Provenance

`source_identity(source_root)` records content hashes for source, entry points, configs, tests, dashboard code and dependency files, including untracked files. Generated outputs, caches, environments and reports are excluded. The digest currently covers the whole experiment implementation; changes in unrelated experiment code conservatively invalidate new identities. Recorded file hashes allow comparison even when no commit exists.

`capture_provenance(source_root, started, measurements)` records that identity, Git commit, repository and experiment dirty states, all installed package versions, Python and platform, elapsed runtime, timestamp, CPU information and an available GPU query. Unavailable Git or GPU observations are explicit. `started` comes from `time.perf_counter()`; callers can add measured peak memory and throughput. No GPU model is loaded to collect provenance. A writer rejects publication if the recorded source digest differs from its configuration.

## Verification

```text
python -m unittest discover -s ah-clustering-experiment/tests -v
```

The step 3 tests write real numeric, Parquet and JSON files locally. They use hand-specified tensor fixtures and direct independent reads, and inject corrupt payloads, invalid offsets, IDs, dtypes, configuration changes and producer interruption. They establish the storage contract, not real model extraction or persistent cloud-storage recovery. See `reports/STEP_3_VERIFICATION.md` for the evidence and limitations.
