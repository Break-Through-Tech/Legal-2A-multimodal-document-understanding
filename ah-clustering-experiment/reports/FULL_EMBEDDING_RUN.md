# Full embedding generation on Colab

Status: complete. Initial run: October 4, 2026; final OCR recovery began October 6. Verified publication completed at 2026-10-07T00:18:21.140702+00:00.

[Executed Colab notebook](https://colab.research.google.com/drive/1nc4iDTxv0hJvn_0JRYhxekUxiuzwa1qn). The notebook records real commands and outputs; all associated runtimes have been released.

| Mode | Documents | Extraction seconds | Payload MiB | Artifact |
|---|---:|---:|---:|---|
| dinov3 | 1,000 | 358.4 | 294.9 | `embeddings-9ef1694d986576ba0acf` |
| layoutlmv3-image | 1,000 | 320.6 | 580.7 | `embeddings-af5fe5262fbc515a18be` |
| colpali | 1,000 | 779.0 | 263.0 | `embeddings-d987f31cbdb073fed394` |
| colqwen2 | 1,000 | 1088.8 | 189.9 | `embeddings-17d7e1b5ec380770c824` |
| layoutlmv3-ocr | 1,000 | 239.9 | 1733.2 | `embeddings-14f0c9c4cc8cb38ae601` |

The extraction durations include model loading, inference, cache writing and validation; some earlier modes ran alongside OCR. They are observed durations, not isolated performance benchmarks. All models used batch size 1. DINOv3 and LayoutLMv3 used float32; ColPali and ColQwen2 used float16.

## Inputs and validation

- Dataset: `getomni-ai/ocr-benchmark@4ed0d95271ca00107726230f7a0944ed9e90d897`; artifact `dataset-9c71de8a0c0e5e4aab1d`.
- Every cache contains exactly IDs 0 through 999 and preserves the shared 700/150/150 assignments. No clustering or classifier fitting was performed.
- Frozen source digest: `7acf6ea0273b332ba63421772ff955530c739242a5f049a256d0e9a7a2801335`. Immutable model revisions and processor settings are recorded in each artifact configuration.
- Downloaded files passed SHA-256 checks and the normal artifact/embedding readers: exact coverage, schemas, model tensor contracts and finite values. Import checked the local dataset-manifest hash and all published payload hashes.
- Native outputs remain unpooled and unnormalized. Cache integrity does not establish downstream clustering quality.

## OCR recovery

Actual Tesseract 5.3.4 recognized 329,869 words across 1,000 pages; 3 pages have explicitly empty recognition. OCR artifact: `ocr-6a40a0906e051e963e73`.

The original final-mode export was lost with its runtime. Recovery used the frozen Step 4 source and an external helper that saved native per-document OCR artifacts. Each downloaded checkpoint was reopened and validated locally. A later runtime loss was recovered from 425 saved pages. The final CPU invocation took 2491.1 seconds including checkpoint reuse and full-artifact assembly; this is not the total time of all attempts.

The helper matched the original whole-cohort OCR path on two real pages; a separate fixture test rejected a deliberately corrupted checkpoint. Original resolution, English LSTM/PSM 3, a 600-second per-page timeout and `OMP_THREAD_LIMIT=1` were retained. Reference text was never substituted.

Recovery used a saved Step 4 source snapshot while a separate Step 5 task edited the workspace. This report describes the generated caches and observed recovery; it does not certify the separate Step 5 implementation.

## Saved outputs

- Project caches: `outputs/embeddings/`, `outputs/ocr/`, and `outputs/metadata/`.
- Verified exports and per-mode proofs: `D:/ColabRuns/verified-20261004/<mode>/ah-clustering-experiment/`.
- Final transfer parts and checksums: `D:/ColabRuns/recovery-20261006/final-bundle/`.
- Durable OCR checkpoints and operational evidence: `D:/ColabRuns/recovery-20261006/`.

The local Colab connector was also repaired for stalled adapter calls and oversized worker responses. Validation passed 196 tests with one skip, plus formatting, type checks and package checks. A fresh supervised connection successfully returned the real 383,319-byte OCR log. Existing Codex clients need a refresh to load the supervisor fix.
