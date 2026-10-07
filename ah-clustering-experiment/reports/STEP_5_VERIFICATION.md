# Step 5 recovery implementation and evidence

Date: 2026-10-06. The recovery implementation is built and verified locally. Step 5 as a whole remains open: the user selected Google Drive with Colab for the live persistent-backend test, and its mount/authorization is pending. The full 1,000-document LayoutLMv3 OCR artifact is also still missing. No clustering ran and no commit was created.

## Completion predicate and oracle contract

The authority is `IMPLEMENTATION_PLAN.md`, step 5. Completion requires every expected document ID in each of five native-output caches, checksummed incremental persistence, one writer per artifact, and an interrupted pilot that resumes without re-inferencing completed documents. Interrupted and uninterrupted native values, masks and IDs must agree within a declared tolerance. The intended persistent backend must actually execute the recovery test.

```text
ORACLE_SOURCE: documented step 5 requirements; literal independently specified
  tensors/masks; direct uninterrupted pretrained inference for the real pilot.
REAL_COMPONENTS: dataset/image validation, filesystem checkpoint/publication,
  fresh child processes, native readers, and real GPU inference for the live pilot.
ALLOWED_MOCKS: deterministic encoder and dependency lookup in unit scenarios only.
  The real model pilot has no encoder or storage substitution.
NEGATIVE_CONTROL: numeric +1 perturbation must fail the independent tensor oracle;
  corrupt/missing payloads must be rejected; exit during checkpoint publication.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED for the selected persistent backend.
```

## Implementation

`src/recovery.py` owns a process-lifetime OS lock and payload/receipt protocol. `src/embeddings/checkpoints.py` stores native batch tensors without pickle, pooling or dtype conversion. The pipeline checks receipt hashes and validates IDs, schemas and masks before reuse. It preserves the original batch boundaries, so restart does not change dynamic padding. A partial batch is inferred again; a valid completed batch is not. Corrupt final unpublished shards are rebuilt from intact checkpoints. A completed artifact with a missing request index is validated and indexed without loading a model.

OCR saves each validated actual-engine record immediately, rechecks source-image identity and dimensions on resume, and logs every attempt's progress or failure. Invalid cached records are unfinished. Reference transcriptions remain excluded. Published artifacts remain immutable, including when a payload is corrupt.

Artifact locks include a random acquisition token. Automatic stale-lock removal requires the exact checksum recorded under the journal's exclusive lock. Unknown legacy locks fail safely. The protocol assumes a single compute host and a filesystem supporting advisory locks and atomic replacement. It does not implement cross-host leases or certify power-loss durability.

The README supplies clone/source-snapshot, install, Drive mount, storage probe, full extraction and resume commands. Original images, model caches, checkpoints, temporary writes and logs remain under the experiment directory. Checkpoint tensors remain after publication and duplicate the final numeric payloads; capacity planning includes both copies.

## Evidence

The final integrated command `.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -q` ran 142 tests in 120.638 seconds and passed with one unavailable-Tesseract skip. `git diff --check` passed. The source digest remained unchanged from the real local GPU pilot.

The source digest for the final local live pilot is
`19f95fe14111b8abff529bad79c673e054befb50859b71e68efec76c5e4c63d6`.
Source identity covers uncommitted source, tests, configurations and dependency declarations. Reports and generated outputs do not alter it.

Compact machine-readable evidence and the complete source identity are retained under `reports/step5-evidence/`. `notebooks/step5-drive-recovery.ipynb` embeds the exact tested source snapshot and checks its SHA-256 before restoring it into a versioned Drive folder. Notebook code cells compile locally; they have not yet executed on Drive.

- The ten independent requirement-derived tests cover real process death before and during checkpoint publication, selective recomputation after corruption, exact native arrays/masks, configuration isolation and a numeric-error negative control. Their encoder is an explicit fixture. See `STEP_5_INDEPENDENT.md`.
- OCR checks cover per-document restart, malformed records, image identity/dimensions, failed-attempt logs and preservation of completed-artifact integrity. Actual Tesseract is unavailable locally, so the engine-dependent test is skipped.
- Five recovery edge tests cover stale-lock acquisition tokens, preservation of unknown locks, malformed saved configuration, and completed-artifact/index-gap recovery with model loading forbidden.
- `experiments/verify_recovery_storage.py` passed on the actual local D: volume. It excluded a concurrent writer, killed the writer process tree, reopened a committed checkpoint, rejected a payload lacking a receipt, repaired it, and rejected deliberate checksum corruption. This is same-host process recovery, not a Drive or remount test.
- `experiments/verify_step5_live.py` passed with real pretrained LayoutLMv3 image-only on training IDs 0, 1 and 2. It exited with code 74 during the second checkpoint write, restarted in a fresh process, and reused document 0 without another encoder call. All values matched direct uninterrupted inference with maximum absolute error 0.0 at declared `rtol=atol=1e-5`; masks and IDs matched exactly. A third process reused the completed artifact with model loading forbidden.

The real pilot artifact is `embeddings-324ac06ad4d360c926c7`. Peak GPU allocated memory was 528,624,128 bytes and reserved memory was 566,231,040 bytes. Two resumed inference calls measured 4.736 documents/second, excluding model load. Three documents retained 1,825,359 tensor bytes, 1,835,531 published payload bytes and 2,445,384 checkpoint-directory bytes including an interrupted temporary write. This small pilot does not establish maximum batch size or corpus throughput.

Four historical full caches were independently reopened and validated against exact IDs 0 through 999. They total 1,393,015,305 published payload bytes. These were produced by earlier code, and do not certify the new recovery implementation. The historical full OCR transfer is not a valid ZIP and was not reused. See `STEP_5_COMPUTE.md` for artifact IDs, hardware, dependency differences and the completed cache audit.

## Evidence ledger

```text
REVISION: uncommitted source digest 19f95fe14111b8abff529bad79c673e054befb50859b71e68efec76c5e4c63d6
CLAIM: native batch and OCR restart implementation; completed inference reuse.
ORACLE_SOURCE: requirements, independent literal fixtures, uninterrupted pretrained outputs.
EVIDENCE_GRADE: LIVE_VERIFIED for local image-only process recovery;
  INDEPENDENTLY_CHECKED for fixture-based orchestration; Drive gate pending.
REAL_COMPONENTS_EXECUTED: D: filesystem, subprocess termination/restart, native
  artifact readers/writers, pretrained LayoutLMv3 on RTX 3050 Ti.
MOCKS_AND_LOST_COVERAGE: OCR fixtures do not prove actual engine resume; model
  fixtures do not certify all five pretrained paths on mounted storage.
NEGATIVE_CONTROL_RESULT: independently specified tensor oracle rejects +1 error;
  checkpoint readers reject corrupted or unfinished payloads.
CONTRADICTORY_EVIDENCE: none for the scoped local claims.
RESIDUAL_RISKS: Drive locking/remount/reconnect unverified; full OCR cache missing;
  local torch/numpy versions differ from pinned Colab stack; all-model full rerun pending.
VERDICT: PASS for scoped local recovery; full step 5 completion remains pending.
```
