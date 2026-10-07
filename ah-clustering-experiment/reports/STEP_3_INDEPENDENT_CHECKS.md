# Independent Step 3 checks

## Oracle freeze (before implementation inspection)

Base revision: `c9861bea102ace2106870076e46481d5a9fc55b3`.
Oracle sources: `IMPLEMENTATION_PLAN.md`, Step 3 completion requirements, and
`ARTIFACT_FORMAT.md` version 1. Existing artifact tests and the builder's
verification report were not read or used as an oracle.

Expected invariants frozen before inspecting implementation APIs:

1. Real NumPy, Parquet, and JSON payloads retain exact hand-written numeric
   values, dtype, shape, ID association, and token metadata. No normalization,
   pooling, silent integer rounding, or implicit row-order identity is allowed.
2. Unordered, noncontiguous IDs and zero-length sequences round-trip; aligned
   masks have equal lengths. Concatenated spans partition each shard exactly.
   Record order and numerical shard index gaps do not change meaning.
3. Corrupt payload bytes, malformed offsets/lengths/keys/alignment metadata,
   missing/duplicate IDs, and incompatible configurations must be rejected
   before publication or during reading. Merely rebuilding integrity hashes
   must not bypass semantic validation.
4. Config snapshots are independent of caller mutation. Changing image,
   preprocessing, model, source, and relevant OCR identity invalidates cache;
   changing OCR alone does not change image-only identity.
5. Completion is published last; interrupted writes are unreadable, release
   context-managed writer locks, and permit a same-config retry. Completed
   bundles are immutable and concurrent same-identity writers are rejected.
6. Source provenance changes for tracked/untracked implementation changes and
   excludes generated cache/output/report content. A configured source digest
   mismatch prevents completion.

Real components: local disk, NumPy arrays, Parquet tables, JSON, actual source
files, and artifact writer/reader. Only input tensors are synthetic; these
checks make no claim about model inference, OCR quality, cloud durability, or
Step 5 inference resume.

Execution results and any refinements follow below.

## Independent execution and negative controls

The fresh suite is `tests/test_step3_independent.py`. It imports only public
writer/reader/config/provenance APIs. Expected tensor values are literal NumPy
fixtures. It reads saved NumPy, Parquet and JSON directly and computes SHA-256
and canonical JSON hashes independently with Python `hashlib`/`json`. Rehashed
malicious records exercise semantic validation beyond accidental byte damage.
The source fixture is a real local directory with actual source file edits;
provenance collection, installed dependency inspection, and hardware queries run
normally. No implementation functions are mocked or replaced.

Command from repository root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -p test_step3_independent.py -v
```

Baseline against the original Step 3 implementation: **19 tests, 29.467 seconds,
17 passed, 1 failure, 1 error**. Captured output:
`cache/independent-step3/baseline.log`. The initial exploratory run used long
test names in scratch paths and hit Windows path limits; those harness paths
were shortened before the reported baseline. Two exploratory assertions about
token-index semantics were also removed for the contract reason below.

The logged baseline was invoked with output redirection followed by
`Get-Content`; that shell wrapper returned zero because `Get-Content` succeeded.
The Python result is explicitly `FAILED (failures=1, errors=1)` in the log;
the wrapper status is not passing test evidence.

| Requirement or invariant | Independent evidence | Baseline result |
| --- | --- | --- |
| Exact native numeric values, dtype, shape and metadata | Hand-written float16/float32, boolean, int64 tensors; rank-3 tiles; read-only mapped slices; direct NPY/Parquet/JSON reads | Pass |
| Broad numeric storage types | uint64 maximum, signed int8 bounds, float64, big-endian float32 through generic bundle | Pass |
| Endian dtype preserved by embedding writer | Configured and supplied `>f4` tensors, exact dtype expected on read | **Error: valid input cannot publish** |
| IDs independent of row order and shard numbering | IDs 901, 4, 72; reverse physical Parquet order; rename shards to 3 and 41 and independently rehash | Pass |
| Empty retained outputs | Empty document within mixed shard and all documents empty in separate shards | Pass |
| Full document coverage after decoding | Nullable integer shard column; set empty document 4's shard to null; independently rehash | **Failure: invalid artifact accepted** |
| Reject corrupt bytes | Flip a payload byte on disk after completion; damage JSON before completion | Pass |
| Reject malicious spans and keys despite valid checksums | Negative, fractional, boolean, huge or gapped bounds; duplicate/unexpected/negative IDs | Pass |
| Validate mask alignment | Reassign mask spans while keeping complete contiguous coverage, yielding per-document length mismatch | Pass |
| Validate metadata container | Rehashed empty object and list instead of nonempty metadata object | Pass |
| Interruption/publication/retry | Raise `InterruptedError` after first real shard; require absent completion/lock and rejected read; retry identical config | Pass |
| Single writer and immutable completion | Competing same-identity context; attempt reopen completed artifact; compare manifest bytes | Pass |
| Immutable config snapshot and compatible reads | Mutate nested caller config after writer construction; reject mismatching expected config | Pass |
| Cache invalidation | Change image hash, preprocessing, transformation, model revision, source digest; compare identity hashes | Pass |
| OCR identity separation | Different OCR content hashes affect multimodal identity; image-only identity unchanged | Pass |
| Actual source provenance | New untracked source and changed source bytes alter digest; cache/output/report files do not; changed source prevents publication | Pass |

## Defects reproduced before repair

1. **Nullable shard silently loses a document.** Public `read_embeddings` accepts
   a checksum-valid table whose zero-length document has a null shard. The
   table-level IDs still include that document, but group iteration drops null
   values and the decoded result omits it. Reproducer:
   `test_rehashed_nullable_shard_cannot_silently_drop_empty_document`.
   Observed assertion: `AssertionError: ValueError not raised`.
2. **Nonnative numeric endian rejected after writing.** `>f4` is an accepted
   numeric dtype, but concatenation converts it to native endian. Completion
   then errors with `ValueError: Tensor dtype/shape mismatch: vectors`.
   Reproducer: `test_embedding_preserves_nonnative_endian_dtype`.

Both failing controls were reported to the lead before any production repair.
The independent worker edits only its own test/report; the lead owns repairs.

## Repaired implementation: fresh verification

The lead repaired shard-key validation and final decoded ID coverage, and made
embedding concatenation preserve the declared dtype with `casting="no"`.
The two regression assertions were unchanged. One additional contract-derived
test was added before repairs: an iterator reuses the same tensor buffers,
changes them for each document, then zeros them at exhaustion before a delayed
shard flush. Exact saved values must still match each separately specified
document. Its initial focused execution passed (1 test, 4.578 seconds).

The frozen final suite passed **20 tests in 22.835 seconds, process exit 0**.
Both previously failing regression cases passed. Raw output is retained in
`cache/independent-step3/repaired.log`. Exact PowerShell invocation preserved
the Python exit status across log display:

```powershell
.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -p test_step3_independent.py -v *> ah-clustering-experiment/cache/independent-step3/repaired.log
$testExit = $LASTEXITCODE
Get-Content ah-clustering-experiment/cache/independent-step3/repaired.log
exit $testExit
```

Verified revision: `c9861bea102ace2106870076e46481d5a9fc55b3` plus the lead's
uncommitted embedding repairs and the independent test file. SHA-256 of the
actual working files at this passing execution:

| File under experiment root | SHA-256 |
| --- | --- |
| `src/embeddings/__init__.py` | `ac2bbcbe399d610a0dd5940dab1085022ed8490c12ae7710a325d1f69d86f078` |
| `src/artifacts.py` | `63bdcd7f370fbaceb317221a7755ef09218607a4a791d5ca36148e44661cc5f1` |
| `src/contracts.py` | `80930efd988d4b9a8dceca2faf628b0da1f461df946e8bb8ff951529e5fd5204` |
| `src/provenance.py` | `e15b742422488cb09cc9f54003da3101f4c41711edbb58f14853b512bd23a043` |
| `src/dataset.py` | `e51fcbabe9716ca3ecb87eba200a8cf1df267962896ad3e6035a93d44f5e4802` |
| `tests/test_step3_independent.py` | `569035f40f12adabddb0740050105acb995d2993b7c2c3b25c81d4d9e08c0950` |

The test source was not edited after repair; only this report was finalized.
No production file was edited by the independent worker, and no commits were
created. The lead runs broader integration checks separately.

## Scope refinements and remaining limits

The initial oracle's reference to malformed alignment metadata means declared
tensor alignment groups and required token-metadata container shape. Version 1
does not define which tensor each `special_indices` list refers to, or a common
model-independent semantic schema for OCR word alignment. Exploratory probes
confirmed `special_indices: [-1]` and `[999]` are accepted as JSON metadata.
That observation is a limitation for later encoder validation, not a Step 3
failure based on an invented index contract. The tests retain validation of
the explicitly required nonempty metadata object and tensor/mask alignment.

The evidence grade is **INDEPENDENTLY_CHECKED** for local persistence and the
listed invariants. Synthetic tensors do not establish model-output semantics,
actual OCR accuracy, inference resume, or cloud-mount crash/locking behavior.
Scratch is retained only under `cache/independent-step3`.
