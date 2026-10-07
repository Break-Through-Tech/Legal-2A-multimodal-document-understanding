# Step 2 verification

Historical implementation report. The later [dedicated step 2 audit](STEP_2_AUDIT.md) found and repaired incomplete-provenance acceptance and null-sidecar recovery, passed 81 integrated tests, and rebuilt/repaired all 1,000 real images in isolated storage. Use that audit for the current completion assessment and source identity.

Status: complete on 2026-10-04. The acceptance criteria below were recorded before implementation. Evidence grade: `LIVE_VERIFIED`.

The dataset preparation command must preserve the shared document assignments and save a complete manifest with real image references. Default clustering membership is the training partition. Preparation must not run embeddings, OCR, clustering, or classification.

```text
ORACLE_SOURCE: data/split.csv, data/README.md, and step 2 of IMPLEMENTATION_PLAN.md.
REAL_COMPONENTS: Pinned Hugging Face dataset download, shared CSV, original image
                 bytes, local filesystem, Parquet writer/reader, and preparation CLI.
ALLOWED_MOCKS: Small source fixtures only for invalid-input checks. Real dataset
              acquisition and final artifact checks must use the pinned source.
NEGATIVE_CONTROL: Duplicate or missing document IDs, conflicting partition
                  assignments, source metadata mismatch, and corrupt artifacts.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED.
```

Independent expected values are 1,000 documents, 19 final labels, and partition counts of 700 training, 150 validation, and 150 test. The shared CSV SHA-256 before implementation is `0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c`.

The dataset revision is `4ed0d95271ca00107726230f7a0944ed9e90d897`. Its upstream split named `test` contains the full dataset and does not define the team's test partition.

The implementation and verification workers own separate source and test files. The lead owns integration, actual dataset loading, documentation, and final evidence. Test expectations come from the shared CSV and requirement-defined invariants.

Official API references checked before implementation:

- [Loading a pinned dataset revision](https://huggingface.co/docs/datasets/loading).
- [Reading original image bytes without automatic decoding](https://huggingface.co/docs/datasets/image_load).
- [Controlling dataset and Hub caches](https://huggingface.co/docs/datasets/cache).

## Verified source and artifacts

The base Git commit is `e3828ef286eebf285eaa77bf59cc940e8cbf858b`. The implementation is uncommitted, and the artifact records that state. No shared tracked files changed.

| File or artifact | SHA-256 |
| --- | --- |
| `src/dataset.py` | `e51fcbabe9716ca3ecb87eba200a8cf1df267962896ad3e6035a93d44f5e4802` |
| `experiments/prepare_dataset.py` | `75f3775b8a7e4535c46f4dd289435dffadfab559e5b777fb283c8f914ad272b5` |
| Saved `manifest.parquet` | `42df06d3477c3c4017436255af30b2343f995c8d77171472a53b65f4f6a8139e` |

The preparation source digest is `962568fbccc7a2ee6bb8cd800257c7a68ae56ea1d86bed7a7ee580bbbca5a803`. Dataset metadata stores the source hashes, Git state, Python version, and dependency versions.

The dataset artifact is `outputs/metadata/dataset-9c71de8a0c0e5e4aab1d/`. It contains:

- `manifest.parquet` with 1,000 rows and relative original-image references.
- `dataset.json` with the dataset revision, CSV checksum, counts, limitations, and preparation provenance.
- `cohorts/cohort-2718f59cc8eefe37e844/cohort.json` with the 700 default training IDs.
- `cohorts/cohort-0fa4feb2ff2a3eb41e89/cohort.json` with 1,000 IDs and an explicit exploratory designation.

The all-document cohort was created to verify cohort identity. No clustering or visual analysis of that cohort ran in this step.

## Coverage results

| Slice | Status | Evidence | Gaps |
| --- | --- | --- | --- |
| Preparation implementation | PASS | Real CLI produced the manifest, original images, and both cohorts. Source/configuration changes stay within the experiment directory. | Concurrent writers are unsupported. |
| Independent contract checks | PASS | 22 tests cover shared assignments, shuffled source order, invalid inputs, persistence, and repair. | Small fixture images exercise injected failures. |
| Real dataset integration | PASS | A separate reader verified all 1,000 images byte for byte against the pinned source and checked every shared label and partition. | Windows local storage was exercised; Colab and mounted cloud storage were not. |
| Repeat execution | PASS | Experiment-directory invocation with Hub and dataset offline modes returned `reused: true`, 1,000 documents, and a 700-document cohort. | Model extraction and its recovery belong to later steps. |

The saved images total 378,089,858 bytes. The independent reader confirmed 19 labels, train 700, validation 150, and test 150. All image paths resolve inside the experiment storage root. Reference transcriptions, extraction targets, schemas, and OCR words are absent from the prepared manifest.

## Commands and observations

The existing repository environment supplied Python 3.14.0, datasets 5.0.1, pandas 3.0.5, pyarrow 25.0.1, Pillow 12.3.0, and huggingface_hub 1.30.0. No packages were installed into that shared environment. The experiment requirements record the tested direct dependencies.

From the repository root, the real preparation command was:

```powershell
$env:PYTHONPYCACHEPREFIX = "$PWD/ah-clustering-experiment/cache/pycache"
.venv/Scripts/python.exe ah-clustering-experiment/experiments/prepare_dataset.py
```

It downloaded the pinned source and produced the 1,000-document manifest and the 700-document cohort. Initial dataset loading exposed a Windows path-length failure in a library cache lock. The implementation now loads with a short relative dataset-cache path from the experiment directory. Short artifact directory IDs and temporary filenames also avoid excessive path lengths while full hashes remain in metadata.

The final test run was:

```powershell
.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -v
```

Result: 22 tests passed in 16.123 seconds. The tests preserve the shared CSV and create temporary artifacts only inside the experiment cache.

The lead ran this independent artifact reader against the final recovered dataset:

```powershell
$env:HF_HOME = "$PWD/ah-clustering-experiment/cache/huggingface"
$env:HF_HUB_OFFLINE = '1'
$artifactRoot = 'ah-clustering-experiment/outputs/metadata/dataset-9c71de8a0c0e5e4aab1d'
.venv/Scripts/python.exe ah-clustering-experiment/tests/verify_real_dataset.py `
  --storage-root ah-clustering-experiment `
  --manifest "$artifactRoot/manifest.parquet" `
  --metadata "$artifactRoot/dataset.json" `
  --train-cohort "$artifactRoot/cohorts/cohort-2718f59cc8eefe37e844/cohort.json" `
  --all-cohort "$artifactRoot/cohorts/cohort-0fa4feb2ff2a3eb41e89/cohort.json" `
  --source-arrow ah-clustering-experiment/cache/datasets/getomni-ai___ocr-benchmark/default/0.0.0/4ed0d95271ca00107726230f7a0944ed9e90d897/ocr-benchmark-test.arrow
```

Result: `PASS`, 1,000 matching original images, exact shared assignments, and distinct correct training and exploratory cohorts. This verifier reads Parquet and the source Arrow file directly rather than using the implementation's artifact reader.

The final repeat run used the experiment directory as its working directory:

```powershell
$env:PYTHONPYCACHEPREFIX = "$PWD/cache/pycache"
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
../.venv/Scripts/python.exe experiments/prepare_dataset.py
```

Result: `reused: true`, `document_count: 1000`, `cohort_count: 700`, with the same dataset and training-cohort IDs. No source download was needed.

## Negative controls and repairs

Duplicate and missing IDs, unexpected source IDs, conflicting partitions, and mismatched source metadata all fail validation. Shuffling source and assignment order preserves the correct ID-based join.

Fixture-based tests corrupt an image, a Parquet file, and completion metadata. Preparation rejects those cached artifacts and repairs them from the supplied fixture source. A valid artifact can be reused even when the supplied source iterator would fail on access.

The independent reviewer also changed only a sidecar's document count from 1,000 to 999. The original candidate incorrectly reused that sidecar, so the negative-control test failed. The fix validates identity fields and count summaries against the manifest. The test then passed.

The lead repeated that document-count corruption on the real generated sidecar. Running `prepare_dataset.py --cohort all` returned `reused: false`, repaired the count to 1,000, and published a valid completion marker. The independent artifact reader passed afterward. No deliberate corruption remains.

## Final verdict

`PASS`, with evidence grade `LIVE_VERIFIED`. The real source, images, filesystem, Parquet boundaries, preparation command, and offline reuse executed successfully. Independent expected values and negative controls support the result. The final shared CSV checksum equals the checksum recorded before implementation.

Colab storage, GPU inference, OCR, clustering, and classification were not exercised. They are outside step 2. New files remain uncommitted.
