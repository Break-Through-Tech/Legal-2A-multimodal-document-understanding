# Dedicated step 2 audit

Audit date: 2026-10-04. Starting Git revision: `c9861bea102ace2106870076e46481d5a9fc55b3`, with the prior step 3 audit's uncommitted changes preserved.

## Frozen acceptance contract

The audit must establish exact ID-based preservation of all 1,000 documents, final and original labels, document quality, 700/150/150 partition assignments and original encoded image bytes. The default cohort is the 700 training documents; other cohorts are separately identified and exploratory. Reference transcriptions cannot enter the prepared manifest or substitute for OCR. Valid caches must reuse without upstream access, and invalid or incomplete artifacts must fail validation or be safely repaired before completion is published.

```text
ORACLE_SOURCE: Step 2 of IMPLEMENTATION_PLAN.md; data/README.md; data/split.csv;
              pinned source Arrow/images at revision
              4ed0d95271ca00107726230f7a0944ed9e90d897.
REAL_COMPONENTS: Actual source images and metadata, local files, PIL,
                 Pandas/Parquet, JSON and preparation/reuse/repair paths.
ALLOWED_MOCKS: Small valid encoded source-image fixtures for injected failure
              tests. Real-data integration may not replace original source.
NEGATIVE_CONTROL: Corrupt bytes, invalid records with recomputed checksums,
                  incomplete/invalid metadata and interrupted repair.
REQUIRED_EVIDENCE_GRADE: LIVE_VERIFIED for local dataset preparation.
```

The lead owns production changes, real-source integration and synthesis. Two fresh read-only reviewers inspect joins/cohorts and persistence/recovery. A separate verifier freezes new tests from the requirements before reading implementation APIs; the original dataset tests are not its oracle. Every writer uses separate scratch under `ah-clustering-experiment/cache/`. The prepared production dataset and shared CSV are preserved.

The prior step 3 audit reconfirmed saved images and offline reuse, but did not provide this dedicated adversarial examination of step 2. Existing builder-authored tests alone cannot certify the implementation independently. Concurrent dataset preparation and cloud-mount behavior are not claimed by the current step 2 API.

## Independent review and baseline

Reviewers: `dataset_join_review` and `dataset_cache_review`. Fresh verifier: `dataset_independent_checks`.

`dataset_join_review` reported no actionable findings. It independently compared all 1,000 cached pinned-source images to saved bytes and hashes, checked every CSV assignment and source metadata normalization, reversed the source-record order, checked all four cohort selectors, and exercised invalid source/assignment cases in memory. The shared data and saved original images matched exactly.

`dataset_cache_review` found that a cached metadata object could lose `preparation_provenance`, `dependencies`, `python_version`, `created_at`, `limitations` and `reference_text_policy` yet still pass full validation. Its read-only probe supplied an in-memory stripped sidecar against the real saved manifest and images; all 1,000 records were accepted. This violates complete provenance and required research-context preservation.

The independent verifier froze requirements before inspecting the implementation and did not read the original tests. Its baseline ran 16 test methods in 25.954 seconds, exit 1: six deletion subcases failed because missing metadata was accepted, and JSON `null` caused `AttributeError` instead of entering the repair path. All other methods passed. The failing baseline is retained in `cache/independent-step2/baseline.log` with its actual exit code in `baseline.exit.txt`.

Agreement/classification:

- **Act on, resolved:** incomplete policy/provenance accepted. Found by the cache reviewer and independently reproduced by the verifier.
- **Act on, resolved:** non-object JSON bypasses repair via `AttributeError`. Found by the verifier's public preparation test.
- **Consider:** none.
- **Noted:** the step 2 API explicitly supports one dataset writer; concurrent preparation and remote storage are outside this audit.
- **Dismissed:** none. The real-data harness initially supplied source-cache file paths outside its isolated output root, which the implementation correctly rejected. The harness now reads those exact original bytes and supplies encoded bytes to the isolated writer; production isolation was not weakened.

Review verdict: `INDEPENDENTLY_REVIEWED`. Functional certification uses the execution evidence below, not reviewer agreement.

## Repairs and independent rerun

The lead changed only the relevant dataset validator/writer path:

- Reject any non-object metadata value with `ValueError`, which preparation already handles as a repairable cache failure.
- Require the reference-text policy and research limitations, Python/dependency information, timezone-aware creation timestamp, source hashes and Git-state fields.
- Check the recorded source digest against its recorded file hashes, preserving the validity of historical artifacts whose source differs from today's code.
- Use one constant for the written and validated reference-text policy.

No original failure assertion was relaxed. Three additional methods cover missing-policy repair with unavailable source, non-object/truncated JSON and semantically invalid provenance. The final independent suite passed **19 tests in 37.165 seconds, exit 0**. Its full coverage, commands and hashes are recorded in [STEP_2_INDEPENDENT_CHECKS.md](STEP_2_INDEPENDENT_CHECKS.md).

The cache reviewer then examined the actual repair and ran 52 malformed metadata cases in memory. Every invalid case raised `ValueError`; the existing complete historical sidecar remained valid. No further actionable issue was found.

## Real-source integration

`tests/verify_step2_real.py` creates an isolated temporary experiment directory and rebuilds all original images from the cached pinned source in reversed ID order. It uses `tests/verify_real_dataset.py` as an independent reader: labels and partitions come directly from the shared CSV, while encoded bytes come from the original Arrow/source cache.

The harness checks reuse when source iteration would raise, then deliberately corrupts one isolated image. The independent reader must reject that image. An interrupted repair must remove the completion marker. A subsequent valid-source repair must restore every original byte, and a second full comparison checks all 1,000 documents. Scratch is cleaned up afterward. The production prepared dataset is never corrupted.

The pre-fix real-source baseline passed this full rebuild/reuse/corruption/interruption/repair scenario. These positive image checks did not expose the metadata defects; the independent malformed-input cases did. The final repaired-source result is recorded in the ledger below.

Run from the repository root:

```powershell
$env:HF_HOME = "$PWD/ah-clustering-experiment/cache/huggingface"
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
.venv/Scripts/python.exe ah-clustering-experiment/tests/verify_step2_real.py --source-arrow ah-clustering-experiment/cache/datasets/getomni-ai___ocr-benchmark/default/0.0.0/4ed0d95271ca00107726230f7a0944ed9e90d897/ocr-benchmark-test.arrow > ah-clustering-experiment/cache/step2-real-final.log 2>&1
$step2Exit = $LASTEXITCODE
Get-Content ah-clustering-experiment/cache/step2-real-final.log -Tail 50
exit $step2Exit
```

## Exact candidate identity

Final experiment source digest: `8a0f3263af86bcb9d6070990c41d8bdc9ac131b1eb52937fb33cf40a2526a1e2`. Git commit remains `c9861bea102ace2106870076e46481d5a9fc55b3`; source and tests have uncommitted changes. The earlier step 3 repairs are preserved.

| File | SHA-256 |
| --- | --- |
| `src/dataset.py` | `62e47bfa9a8f3a94bfd10dafdf6bb53f26ac32dbba4f78c3ee7c92a696a273ba` |
| `tests/test_step2_independent.py` | `a3cefd3defc79a5db480a0ad1f360a34d8358324ec721eed97407cc197a326da` |
| `tests/verify_step2_real.py` | `10bdc82f2e3ac711c3a0d3bb025ecab2281ceb92f69b9d8e9d6c3becccb2cdec` |

The environment remains Python 3.14.0, NumPy 2.5.3, pandas 3.0.5, Pillow 12.3.0, pyarrow 25.0.1 and datasets 5.0.1 in the existing repository environment. No packages were installed.

## Final execution and evidence ledger

The repaired-source real integration returned **PASS, exit 0, in 156.890 seconds**. Both the fresh build and the post-interruption repair matched all 1,000 original encoded images byte for byte, totaling 378,089,858 bytes. The 19 labels, 700/150/150 assignments, training cohort and exploratory all-document cohort matched the independent CSV/source expectations. Source-unavailable reuse passed. Deliberate corruption was detected, interrupted repair left no completion marker, and valid-source repair succeeded. Temporary original-image copies were cleaned up; no production artifact was changed.

The integrated regression command was:

```powershell
.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -v > ah-clustering-experiment/cache/step2-audit-final.log 2>&1
$step2SuiteExit = $LASTEXITCODE
Get-Content ah-clustering-experiment/cache/step2-audit-final.log -Tail 12
exit $step2SuiteExit
```

Result: **81 tests passed in 107.856 seconds, exit 0**. This includes the original 22 dataset tests, 19 new independent dataset tests, and all 40 artifact/process tests retained from the step 3 audit. No production or test source changed after the final runs. `git diff --check` passed, and all changes remain inside the experiment directory.

The shared CSV checksum before and after this audit is unchanged:

`0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c`.

```text
REVISION: c9861bea102ace2106870076e46481d5a9fc55b3 plus uncommitted audit repairs/tests.
SOURCE_DIGEST: 8a0f3263af86bcb9d6070990c41d8bdc9ac131b1eb52937fb33cf40a2526a1e2
CLAIM: Step 2 local preparation preserves exact original images and shared
       assignments, creates the intended cohorts, validates complete metadata,
       reuses valid artifacts and safely repairs invalid/incomplete artifacts.
ORACLE_SOURCE: Shared CSV, plan requirements, pinned source bytes/metadata,
               independent distinct-image fixtures and failure invariants.
EVIDENCE_GRADE: LIVE_VERIFIED for local step 2 preparation/reuse/repair.
CHECKS_RUN: Two fresh reviews, independent 19-case suite, integrated 81-case
            suite, real-source fresh build and full corruption/repair workflow.
REAL_COMPONENTS_EXECUTED: Actual cached source Arrow and original images, local
                         filesystem, PIL decode, Parquet, JSON, cache paths.
MOCKS_AND_LOST_COVERAGE: Tiny synthetic images support isolated failure tests;
                       actual pinned-source images support the integration gate.
NEGATIVE_CONTROL_RESULT: Six missing-metadata cases and null-sidecar crash
                         failed before repair and pass afterward. Deliberately
                         corrupted real image copy was rejected. Interrupted
                         repair left no completion marker; subsequent repair
                         restored all source bytes.
CONTRADICTORY_EVIDENCE: Two metadata defects reproduced and repaired; no
                       unresolved finding within this declared scope.
RESIDUAL_RISKS: Source came from the already cached immutable revision, so
                fresh network acquisition was not repeated. Cloud mounts,
                concurrent dataset writers, OS power loss, model inference
                and actual OCR are not certified by this audit.
VERDICT: PASS.
```

No commits were created. The original step 2 verification report is historical; this dedicated audit supplies the current assessment and exact candidate identity.
