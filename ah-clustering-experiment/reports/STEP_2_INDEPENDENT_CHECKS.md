# Step 2 independent checks — 2026-10-04

Evidence grade: **INDEPENDENTLY_CHECKED** for the fixture-backed preparation,
join, persistence, cache validation, and recovery checks described here.
This report does not by itself certify that the production images are the pinned
upstream images. The lead's separate real-source verification supplies that gate.

## Oracle fixed before implementation inspection

The verifier first read `IMPLEMENTATION_PLAN.md`, `../data/README.md`, and
`../data/split.csv`, then inspected `src/dataset.py` to learn its public call
surface. The existing `tests/test_dataset_contract.py` was neither read nor
imported. Expected IDs, assignment values, counts, revision, and image bytes are
derived independently, never from a candidate-produced expected manifest.

- Original integer IDs 0–999 occur once each, joined by ID rather than position.
- The CSV remains the authority for every final label, original label, document
  quality, and partition: 700 train, 150 val, 150 test; 19 final labels in each
  partition; 75 documents labeled exactly `Unknown`.
- Save the original encoded images for all 1,000 documents without replacing
  reference transcriptions with OCR. Each independently generated fixture image
  has distinct bytes and a known ID, so exchanging images cannot pass by using
  one identical image for every record.
- The default cohort contains exactly the 700 CSV training IDs. Train, val,
  test, and all cohorts select the exact respective IDs. All-document use has
  a separate cohort identity and an explicit exploratory designation.
- Pinned dataset revision and CSV checksum accompany the artifact. Labels are
  evaluation-only; cluster-count selection is declared.
- Incomplete/corrupt artifacts cannot be reused as complete. Sound artifacts
  remain usable with an unavailable source. Interrupted repair cannot retain a
  completion marker. Required reference-text policy, interpretation limits,
  and preparation provenance cannot silently disappear.

The plan explicitly requires the reference-text distinction and benchmark
limitations. Preparation source/runtime provenance is also required by the
plan's storage/provenance section and evidence ledger. The exact JSON field
names used in assertions were learned from the API after freezing those
semantic requirements. Timestamp presence is treated as part of the published
preparation provenance contract, not a new dataset-label requirement.

## Components and isolation

ORACLE_SOURCE: Shared CSV, data README, plan original-image/ID and provenance
requirements. Independent SHA-256, direct Parquet/JSON reads, and PIL decode.

REAL_COMPONENTS: Python, NumPy, Pandas, PIL, Parquet, JSON, actual local disk,
candidate preparation and validation APIs. No storage or decoder mocks.

ALLOWED_MOCKS: Tiny independently encoded PNG images plus source iterables
that fail or become unavailable. No inference, OCR, or source-authenticity
claim is made from those fixtures.

NEGATIVE_CONTROL: Missing/duplicate/invalid IDs and metadata; invalid config;
missing/corrupt/swapped image bytes; corrupt Parquet; semantic manifest edits
with a freshly recomputed manifest hash; stripped provenance; null/nonobject
or truncated JSON; interrupted/unavailable source; malformed source images.

The reusable pristine fixture contains 1,000 distinct PNGs. Each corruption
case restores the pristine manifest, sidecar, and changed images in a `finally`
block. A process owns its own short temporary directory beneath
`cache/independent-step2`; separate processes do not share fixture artifacts.
Cleanup resolves and checks confinement before deleting. The harness restores
environment variables after preparation. It does not modify the shared CSV,
production source, or saved production artifacts.

## Baseline evidence retained before fixes

Baseline Git HEAD: `c9861bea102ace2106870076e46481d5a9fc55b3`.

| Input | SHA-256 |
| --- | --- |
| `../data/split.csv` | `0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c` |
| `../data/README.md` | `9c5acad1bbe491b55468de79bdcfdc516e249da6de0ceb1367a868a254d38e95` |
| `IMPLEMENTATION_PLAN.md` at oracle read | `daa90fa6ae6bd2879c2777e06915d4e652862c1ca6fc321a5071ff3f632c806f` |
| `src/dataset.py` before lead fixes | `e51fcbabe9716ca3ecb87eba200a8cf1df267962896ad3e6035a93d44f5e4802` |
| `cache/independent-step2/baseline.log` | `a2bb4a186b1fbded8410c9efd3f25d7529809c9d64761ea3ea972e7cb20e4e30` |

Command, from repository root:

```powershell
& .venv/Scripts/python.exe ah-clustering-experiment/tests/test_step2_independent.py
```

The first run used 16 test methods. Its complete output is preserved in
`cache/independent-step2/baseline.log`, with process exit code `1` separately
preserved in `cache/independent-step2/baseline.exit.txt`.

Observed: **16 tests in 25.954 seconds; six failed subtests and one error**.

- Deleting `reference_text_policy`, `limitations`, `preparation_provenance`,
  `dependencies`, `created_at`, or `python_version` did not cause validation to
  reject the artifact. Each deletion was independently exercised.
- Replacing `dataset.json` with JSON `null` raised
  `AttributeError: 'NoneType' object has no attribute 'get'` inside
  `validate_artifact`, preventing the intended repair path.
- The other 14 test methods passed: exact 1,000-ID image-byte mapping,
  assignments/cohorts, shuffle invariance, ID type boundaries, malformed
  assignments/source/configuration, image corruption, semantic corruption with
  recomputed hash, interrupted repair, malformed image failure, offline reuse,
  and correct originally written provenance.

These failures were reported to the lead before any production changes.
Assertions were not weakened. Two complementary methods were then added for
policy deletion repair/unavailable source and nonobject/truncated sidecar
recovery, plus an explicit confined-cleanup guard. At the lead's request a
nineteenth method adds ten structurally invalid provenance cases, including an
inconsistent source digest. The original failing assertions remain unchanged.

## Final verification

The lead repaired production validation. The verifier reran the stable
independent suite with the same command above: **19 tests passed in 37.165
seconds, process exit code 0**. The complete result is retained in
`cache/independent-step2/final.log`, with `final.exit.txt` recording `0`.
The baseline remains preserved.

| Verified final input/evidence | SHA-256 |
| --- | --- |
| `src/dataset.py` | `62e47bfa9a8f3a94bfd10dafdf6bb53f26ac32dbba4f78c3ee7c92a696a273ba` |
| `experiments/prepare_dataset.py` | `75f3775b8a7e4535c46f4dd289435dffadfab559e5b777fb283c8f914ad272b5` |
| `tests/test_step2_independent.py` | `a3cefd3defc79a5db480a0ad1f360a34d8358324ec721eed97407cc197a326da` |
| `cache/independent-step2/final.log` | `407a121dc7ac32bcc4716d9131a1829a6fe2a48a51d7d1bef0e42de97d7d5377` |

The shared CSV retained its baseline SHA-256 after the final run. Environment:
Windows; repository `.venv/Scripts/python.exe`; Python 3.14.0; NumPy 2.5.3;
Pandas 3.0.5; Pillow 12.3.0; PyArrow 25.0.1; datasets 5.0.1. No installation or
commit was performed by this verifier.

| Claim | Independent observation |
| --- | --- |
| Exact 1,000 document mapping | All 1,000 distinct original fixture byte strings and image metadata matched their CSV IDs and assignments. |
| Configurable cohort | All four exact ID sets passed; default training count 700; distinct cohort identities; exploratory all designation. |
| Order-independent join | Deterministically permuted source rows, shuffled assignment rows, and reordered Parquet rows retained correct IDs. |
| Type and source boundaries | Boolean, float, string, negative and missing IDs failed; NumPy integer assignments passed; missing/duplicate source records and mismatched metadata failed. |
| Corrupt artifact detection | Broken/missing/swapped images, corrupt Parquet, and semantically invalid records with a fresh hash were rejected. |
| Provenance enforcement | Six baseline deletion failures now reject; missing and semantically invalid policy/runtime/source provenance all reject. |
| Offline reuse | All cohorts were read with an iterable that raises if source access occurs. |
| Recovery | JSON null repairs; nonobject/truncated sidecars with unavailable source leave no marker; failed image decoding and interrupted repair leave no marker; subsequent valid-source repair succeeds. |

## Limits

The fixture suite proves behavior against independent expectations on real
local disk, but tiny PNGs do not cover all real image formats, sizes, or source
metadata. It intentionally leaves actual pinned-source image comparison to
the separate real-source check. Source interruption is injected while
iterating records; this is not an operating-system crash or power-loss test.
Concurrent writers, hostile sidecar forgery with every dependent hash updated,
and remote/persistent compute storage are outside this Step 2 fixture check.
No model inference, OCR, clustering, or dashboard execution occurred.
