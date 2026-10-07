# Step 3 verification

Historical implementation report. The subsequent [independent audit](STEP_3_AUDIT.md) found and repaired two missed edge cases, added independent and real-process checks, and passed the integrated 62-test suite. Use that audit for the current completion assessment and source identity.

Status: complete on 2026-10-04 for local storage and provenance contracts. No model inference, OCR or clustering ran. The implementation remains uncommitted on `anh-embedding-experiments`.

## Scope and expectations

The acceptance source is step 3 of `IMPLEMENTATION_PLAN.md`: native arrays and document records must round-trip by ID, readers must reject corruption and incompatible configurations, and relevant OCR or preprocessing changes must invalidate embedding caches.

```text
ORACLE_SOURCE: Step 3 requirements, exact hand-specified numeric values,
              independently read Parquet records, and ID/offset invariants.
REAL_COMPONENTS: Windows filesystem, NumPy .npy serialization and mmap reader,
                 Parquet writer/reader, JSON, SHA-256, Git and runtime queries.
ALLOWED_MOCKS: Small tensor/image fixtures and an interrupted producer iterator.
NEGATIVE_CONTROL: Corrupt/missing shards, rewritten checksums with invalid spans,
                  wrong IDs/dtypes, non-finite values, changed configurations,
                  duplicate writers and incomplete publication.
EVIDENCE_GRADE: Locally executed storage integration with independent expected
                values and direct format reads. Model and cloud-storage
                integration are not certified by these checks.
```

The new storage tests use document IDs 19, 7 and 42 with deliberately different values and lengths. The first vector shard must contain exactly `[[1, 2], [3, 4], [5, 6]]`, with document 19 at offset 0 / length 2 and document 7 at offset 2 / length 1. Document 42 has a declared empty sequence in a second shard. CLS vectors and boolean masks are preserved separately. These expected values are specified in the tests rather than produced by the artifact reader.

## Implementation identity

Base Git commit: `9483f437d48d8b8f4b780c6c2afa4507879ebb85`.

Final experiment source digest: `c20e6845df5f4fcd767ae1b27b7ba1ee8ed0f223c63759716fbb1fee5a7e8241`. Both repository and experiment dirty states are `true`. This digest covers source, tests, configurations and dependency declarations; generated files and this report are excluded.

| File | SHA-256 |
| --- | --- |
| `src/artifacts.py` | `63bdcd7f370fbaceb317221a7755ef09218607a4a791d5ca36148e44661cc5f1` |
| `src/contracts.py` | `80930efd988d4b9a8dceca2faf628b0da1f461df946e8bb8ff951529e5fd5204` |
| `src/embeddings/__init__.py` | `89b34a2009e320b940b3c2ea066e522f8e82c6b472831b3c22062f0545b7ab18` |
| `src/provenance.py` | `e15b742422488cb09cc9f54003da3101f4c41711edbb58f14853b512bd23a043` |
| `tests/test_artifact_contract.py` | `bfb4af9d6968e206bc0e7cf8420c8488aa2f4e2c987fd38a1e8d660a845bf717` |
| `requirements.txt` | `5264f5336bea601cd9444a4d3bc3c438b8f6d5b99dd0d388674f1fbae4fef901` |
| `.gitignore` | `9a11f48aed9f26050f01d7757b844a85d2c4fc2a500b3a4d9d644bf10aa85df2` |

The existing environment supplied Python 3.14.0, NumPy 2.5.3, pandas 3.0.5 and pyarrow 25.0.1. NumPy is now a pinned direct dependency. No packages were installed and no existing environment was modified.

## Verification command and observations

From the repository root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -v
```

Result: **41 tests passed in 37.246 seconds**, comprising 19 new artifact tests and the 22 existing dataset tests. Temporary storage stays inside `ah-clustering-experiment/cache/test-artifacts/` and is cleaned up by the tests. Tests directly load `.npy` and Parquet files in addition to exercising the public artifact reader.

| Claim | Expected behavior and observed result | Remaining limit |
| --- | --- | --- |
| Native output preservation | Exact values, dtypes, variable lengths, empty sequences, CLS and masks round-trip. Independent direct reads confirm concatenation and offsets. PASS. | These are explicit numeric fixtures, not encoder outputs. |
| ID-based joins | Shuffling configuration records does not change identity; shuffling persisted rows preserves the correct tensors for each ID. Duplicate, missing and unexpected IDs fail. PASS. | Chunk policies remain an encoder responsibility. |
| Corruption detection | Corrupt/missing payloads, changed config snapshots, changed manifests and unsupported schema versions fail. PASS. | Checksums are integrity checks, not signatures. |
| Semantic tensor checks | Even with recomputed checksums, wrong IDs, overlapping spans, wrong dtype and non-finite values fail. Misaligned masks fail before publication. PASS. | Correct model token semantics must be checked during extraction. |
| Cache invalidation | Changed OCR content changes multimodal identity; OCR changes leave image-only identity unchanged. Processor, transformation and checkpoint changes alter identity. Floating revisions and label-bearing input records fail. PASS. | OCR producers must hash real OCR contents and preserve status. |
| Publication | Interrupted production after the first shard leaves no completion marker; reads fail; a fresh same-config retry completes. Duplicate writers and overwriting completed artifacts fail. Reused producer buffers are copied correctly. PASS. | This does not skip inference for completed documents and does not simulate process termination during an OS write. |
| Result formats | OCR records, similarity matrices with ordered IDs, assignments, metric null/reason/coverage fields and projection records write and read through actual files. PASS. | Generic storage checks do not validate algorithm or metric semantics. |
| Run and source identity | Representation, selection, pooling, normalization, algorithm, parameters, seed and similarity affect run identity. Held-out cohorts require exploratory designation. Untracked source changes alter source digest; generated files do not. A source mismatch prevents publication. PASS. | Whole-experiment source hashing conservatively invalidates caches after unrelated code changes. |

## Isolation and ignore checks

`git diff --check` passed. Final `git status --short --untracked-files=all` showed changes only beneath `ah-clustering-experiment/`. The implementation plan, README and step 2 report were already untracked when this work began; the step 2 report was preserved.

The root ignore rule `embeddings/` also matched the new source package. A scoped exception in the experiment's `.gitignore` now exposes `src/embeddings/`. Final Git checks confirm generated outputs, model cache files and Python bytecode remain ignored. The ignore exception was narrowed after the full test run; no executable source changed afterward.

The shared CSV SHA-256 before and after implementation is unchanged:

`0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c`.

The original dataset preparation source is unchanged, and the existing dataset regression checks all pass. Step 3 did not rerun a dataset download or modify the saved production dataset.

## Remaining work

Step 4 must verify model loading and output semantics, including actual OCR and LayoutLMv3 input modes. Step 5 must test GPU throughput, storage requirements, process interruption and document-level inference resume on the intended persistent backend. Real model extraction and Drive-mounted recovery have not been claimed or marked complete here.
