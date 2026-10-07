# Independent audit through step 3

Audit date: 2026-10-04. Starting revision: `c9861bea102ace2106870076e46481d5a9fc55b3`.

This audit was requested after the initial implementation. The original 41 tests were written alongside the candidate, so their passing result alone was not independent certification. The original verification report remains a historical record; this report supersedes its completion assessment.

## Frozen verification contract

Completion requires the step 2 dataset and step 3 local artifact contracts to pass fresh requirement-derived checks, with deliberate corruption detected and no unresolved defect in those contracts.

```text
ORACLE_SOURCE: IMPLEMENTATION_PLAN.md steps 1–3; shared data/split.csv;
              original images at the pinned dataset revision; exact numeric
              fixtures; ID coverage and tensor preservation invariants.
REAL_COMPONENTS: Local filesystem, actual saved dataset and source Arrow,
                 NumPy, Parquet, JSON, real child processes, Git provenance.
ALLOWED_MOCKS: Synthetic numeric inputs instead of future model outputs.
              No replacement of storage, serialization or process boundaries.
NEGATIVE_CONTROL: Corrupt shards; semantically invalid records with valid
                  checksums; interrupted producer; killed child writer.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED for the local steps 2–3 boundaries.
```

The lead owns production fixes and the final ledger. Two fresh reviewers inspect storage and identity contracts read-only. A separate verifier owns requirement-derived tests and their report, freezing its invariants before reading the implementation and without reading the original tests. Each writer uses its own scratch directory beneath the experiment cache.

GPU inference, actual OCR generation, Drive-mounted durability, and inference-level resume are skipped because the plan assigns them to steps 4 and 5. Their absence must remain visible in the final verdict.

## Checks run before fixes

- The original 19 step 3 tests passed on the starting candidate in 20.374 seconds.
- `tests/verify_real_dataset.py` read the saved Parquet manifest and pinned source Arrow independently of the dataset implementation. All 1,000 original images were byte-equal to source, totaling 378,089,858 bytes. The 19 labels, 700/150/150 assignments, and both cohort identities matched expectations.
- `experiments/prepare_dataset.py` ran with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` and returned `reused: true`, 1,000 documents and a 700-document default cohort.
- The archive branch `archive/old-experiments` exists. The shared CSV checksum remains `0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c`.
- `tests/test_publication_process.py` passed against real processes: a live second writer could not acquire the artifact; killing the first writer left no completion marker; readers rejected the partial files. After confirming the child exited and removing its stale lock, a same-config retry completed successfully. This tests local publication, not skipping previous inference.

## Independent review findings

Both reviewers found that null shard keys were dropped by Pandas grouping after the reader had checked expected IDs. The returned mapping could silently omit expected documents. The storage reviewer also found that NumPy concatenation converted explicitly non-native byte-order arrays to native byte order; the writer accepted their input dtype but failed on read-back. Both findings concern the stated step 3 contract and require fixes.

The identity reviewer found no other actionable defect in current checkpoint/config identities, source/runtime provenance, or the shared dataset join. The storage reviewer found no additional actionable publication, checksum, or path-ownership defect. These reviews supplement execution evidence; reviewer agreement alone is not a correctness oracle.

Reviewers: `storage_review` and `identity_review`, both with fresh context and read-only scope. Independent verifier: `independent_checks`.

Agreement map: both reviewers raised missing shard validation; the storage reviewer additionally raised dtype byte-order conversion. Both were classified **Act on**, reproduced through public APIs by the independent verifier, and fixed by the lead. The storage reviewer then rechecked the actual fix and found no remaining issue in those paths. **Consider:** none. **Noted:** model-specific token-index and OCR-word alignment semantics are not defined by this generic storage format. **Dismissed:** none. Review verdict: `INDEPENDENTLY_REVIEWED`; the execution evidence below determines the functional verdict.

## Reproduction, repairs and final checks

The independent verifier froze expectations from the plan and format document before inspecting code. It did not use the original builder-authored tests as an oracle. Its initial 19-case run passed 17 cases and exposed exactly the two reported failures:

- `test_rehashed_nullable_shard_cannot_silently_drop_empty_document` failed because no `ValueError` was raised after the test introduced a null shard reference and recomputed the file and manifest checksums.
- `test_embedding_preserves_nonnative_endian_dtype` errored at completion with `Tensor dtype/shape mismatch: vectors` for accepted `>f4` arrays.

The baseline output is retained in `cache/independent-step3/baseline.log`. The two test assertions were not relaxed or changed after the production repairs. An additional producer-buffer reuse case, specified before repair, expanded the final independent suite to 20 cases.

The repaired reader validates every shard reference before grouping and checks decoded ID coverage before returning. The writer explicitly supplies the configured dtype and prohibits conversion during concatenation. The final read-only review also checked byte preservation for `>f4`, `>f8`, `>i8`, `<f4` and boolean arrays.

The independent verifier's final run passed **20 tests in 22.835 seconds, exit 0**. Its own detailed coverage and source hashes are in [STEP_3_INDEPENDENT_CHECKS.md](STEP_3_INDEPENDENT_CHECKS.md).

The lead then verified the integrated candidate:

```powershell
.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -v > ah-clustering-experiment/cache/step3-audit-final.log 2>&1
$auditExit = $LASTEXITCODE
Get-Content ah-clustering-experiment/cache/step3-audit-final.log -Tail 12
exit $auditExit
```

Result: **62 tests passed in 49.743 seconds, exit 0**. This includes the 22 dataset tests, 19 original artifact tests, 20 independent artifact tests and one real-process publication test. Corruption controls operate on real temporary numeric, Parquet and JSON files. None replaces the writer, reader, serialization libraries or process boundary with mocks.

Final `git diff --check` passed. Changes remain confined to `ah-clustering-experiment/`. The shared CSV checksum is unchanged. The saved production dataset remains valid, and the offline preparation command returned a cache hit. No GPU, network download or package installation was needed. Deliberately corrupt test artifacts are confined to ignored scratch under `cache/independent-step3/`; production artifacts are not corrupted.

## Final evidence ledger

```text
REVISION: c9861bea102ace2106870076e46481d5a9fc55b3 plus uncommitted repairs/tests.
SOURCE_DIGEST: fc463d0f2f7f5873e60ddc3790f9faf001e4f8d8a6874af99c17c545a65f22ec
CLAIM: Steps 1–3 local isolation, prepared data, numeric persistence,
       document mapping, integrity rejection and provenance are verified.
ORACLE_SOURCE: Frozen plan requirements, shared CSV, pinned original image
               bytes, independent exact numeric fixtures and invariants.
EVIDENCE_GRADE: LIVE_VERIFIED for these local boundaries.
CHECKS_RUN: Two read-only reviews; 20 independent cases; integrated 62-test
            suite; all-1000-image comparison; offline dataset reuse; Git checks.
REAL_COMPONENTS_EXECUTED: Local disk, source Arrow and saved data, NumPy,
                         Parquet, JSON, Git/runtime capture, actual child process.
MOCKS_AND_LOST_COVERAGE: Only tensor inputs are synthetic. No encoder or OCR
                       result semantics are certified.
NEGATIVE_CONTROL_RESULT: Baseline detected two actual defects. Repaired code
                         passes unchanged assertions and rejects deliberate
                         corruption and partial process-interrupted output.
CONTRADICTORY_EVIDENCE: Two baseline defects found and repaired; none unresolved
                       within the declared local storage contract.
RESIDUAL_RISKS: Actual model/OCR semantics, GPU execution, persistent cloud
                storage locking/durability and inference resume remain later
                steps. Token-metadata JSON validity does not certify the
                meaning of model-specific indices.
VERDICT: PASS within the stated local steps 1–3 scope.
```

Source SHA-256 values at the final integrated run:

| File | SHA-256 |
| --- | --- |
| `src/embeddings/__init__.py` | `ac2bbcbe399d610a0dd5940dab1085022ed8490c12ae7710a325d1f69d86f078` |
| `tests/test_step3_independent.py` | `569035f40f12adabddb0740050105acb995d2993b7c2c3b25c81d4d9e08c0950` |
| `tests/test_publication_process.py` | `aab5daf9ac08fc1decda0afd37e6c54c60b0b089d14cc4b13df3833916a13306` |

The final source digest also covers the unchanged dataset/storage modules, configurations, dependency declarations and prior tests. Documentation and this ledger are excluded from that digest. Audit fixes and new tests remain uncommitted; the audit did not create or amend any commit.
