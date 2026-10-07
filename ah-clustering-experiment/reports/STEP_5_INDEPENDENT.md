# Independent step 5 verification

The acceptance scenarios were derived from `IMPLEMENTATION_PLAN.md`, step 5,
and the pre-existing `ARTIFACT_FORMAT.md` before inspecting candidate recovery
implementation. The verifier inspected the original public extraction pipeline,
native tensor reader/writer interface, and existing dataset fixture format.
It did not inspect the new recovery journal implementation.

## Oracle contract

- **ORACLE_SOURCE:** step 5 requires resumable incremental extraction, no repeated
  inference for completed documents, incomplete or corrupt shards treated as
  unfinished, exact document membership, and interrupted/uninterrupted native
  outputs agreeing within a declared tolerance. The artifact contract requires
  identity isolation for preprocessing, native dtype/mask preservation, and
  document-ID joins.
- **REAL_COMPONENTS:** `extract_embeddings`, manifest validation, original image
  loading, filesystem persistence, journal recovery, publication, and completed
  artifact readers. Two checks start real child Python processes: one terminates
  with `os._exit(73)` after a completed batch, and another terminates with
  `os._exit(74)` immediately before the second batch payload's atomic replacement.
- **ALLOWED_MOCKS:** the pretrained encoder loader returns deterministic synthetic
  outputs; package-version lookup returns a fixed identity. No checkpoint,
  publication, filesystem, manifest-validation or reader component is mocked in
  positive acceptance tests. The write-interruption check wraps `os.replace`,
  preserves every normal call, and kills its process at the specified second
  batch payload destination. The explicit negative control alone replaces the
  observable reader result to inject a shape-valid numeric error.
- **NEGATIVE_CONTROL:** perturb one retained vector by +1 while keeping IDs,
  shapes, masks and dtypes unchanged; the independently specified literal oracle
  must reject it. Original pipeline execution also failed the newly authored
  replay and process-death requirements.
- **REQUIRED_EVIDENCE_GRADE:** independently specified and locally verified for
  orchestration/storage only. Live pretrained inference, mounted-backend crash
  recovery and full-cohort extraction require separate evidence.

## Scenarios and execution

`tests/test_step5_independent.py` creates six synthetic PNG inputs with a real
1,000-row metadata manifest based on the shared assignment CSV. Only six selected
images are required by the public pilot API. The fixed variable-length vector
oracle is literal; masks and metadata are also checked. All expected tensors
must match exactly, with **zero numerical tolerance**. Both uninterrupted and
resumed runs must satisfy that same oracle.

The ten scenarios cover uninterrupted native-output persistence, exception
recovery, real child-process death between batches and during payload publication,
corrupt and missing checkpoint payloads, damaged unpublished final shards,
changed preprocessing configuration, duplicate/missing returned IDs, and a
numeric perturbation negative control. Exception recovery also checks subsequent
completed-cache reuse without encoder loading.
Inference traces are synchronized to actual local files and assert that valid
completed documents are never inferred twice. A damaged batch must be recomputed
while another intact completed batch remains reusable. A damaged unpublished
final `.npy` shard is rebuilt from the intact journal without repeated inference.
The contract owner supplied the public journal layout
`outputs/checkpoints/extraction-*/batch-*.bin`; candidate implementation source
was not inspected to derive corruption targets.

Command, from the experiment directory:

```text
../.venv/Scripts/python.exe -m unittest discover -s tests -p test_step5_independent.py -v
```

Initial execution before recovery integration: seven scenarios ran; four passed,
two failed, and one errored. Exception restart repeated the first two IDs; partial
corruption caused both completed batches to replay; process death left an
unrecoverable writer lock. Source edits overlapped this baseline run, so this is
behavioral baseline evidence rather than an exact-revision certification.
A missing-file, write-interruption, and unpublished-final-shard scenario were
subsequently added. Final candidate execution on 2026-10-06: **10 tests passed in
17.540 seconds**. Both real child processes exited with their expected injected
codes; all resumed native outputs matched the literal oracle exactly.

The test file SHA-256 was
`1d99162559f6a142734fb9211ded6f898f4b16cd0928156eca658dd713059403`.
The experiment source digest observed immediately after execution was
`b1d2d155b6918acbacd5fd135b6fdb676cf66219bdbad2247a3b3076bab6578d`.
Other workstreams were active; final integration must rerun the suite and capture
its final source identity. This digest identifies the observation point, not a
claim that all concurrent work was frozen throughout the run.

## Limits

The final run used the repository `.venv/Scripts/python.exe`, with NumPy 2.5.3,
pandas 3.0.5, Pillow 12.3.0 and PyArrow 25.0.1, matching the storage dependency
pins. The initial baseline used the unrelated Anaconda interpreter and its older
storage libraries; it is not final candidate validation.
The tests do not certify T4 precision, memory limits, throughput, actual output
size, model correctness, actual OCR execution, Drive mount locking or durability,
all-1,000-document extraction, or release of live compute. Synthetic fixtures are
never presented as real-image model pilots. Step 5 is not fully complete based
on these checks alone.
