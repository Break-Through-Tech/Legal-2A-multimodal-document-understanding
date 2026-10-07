# Step 4 verification

Verified on 2026-10-04 against Git base `d0650e86581b6ca89eb1ae2b6f0bef45c73727b7` plus the uncommitted source digest `7acf6ea0273b332ba63421772ff955530c739242a5f049a256d0e9a7a2801335`.

The implementation adds five extraction presets, frozen native Transformers adapters, actual Tesseract OCR, a batched runner, immutable checkpoint pins, native tensor persistence, and completed-cache reuse. Full-dataset extraction and interruption recovery remain step 5. No clustering or classification ran.

## Oracle and scope

`STEP_4_INDEPENDENT.md` freezes the expected behavior from the accepted plan, artifact contract and official model sources. `experiments/verify_step4.py` independently loads each official processor/model and compares every saved tensor, shape, dtype and mask against a direct upstream forward. The verifier also observes real adapter forwards to check evaluation mode, frozen parameters and `torch.inference_mode()`.

The intended evidence grade is `LIVE_VERIFIED` for the small pilots. Numeric and mask mutations must fail the comparison. A deliberately failing model loader is used only to prove completed cache reuse; no model inference or OCR engine is mocked in the live pilots. Hand-authored fixtures remain boundary checks, not pretrained evidence.

The runtime is Colab T4, Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.17.0, Hugging Face Hub 1.30.0, NumPy 2.5.3, Pillow 12.3.0, and Tesseract 5.3.4. Exact installed dependencies and hardware observations are retained in artifact manifests. `requirements-embeddings.txt` pins the direct dependencies.

The first unauthenticated DINOv3 request returned 401. The existing approved local Hugging Face login downloaded the pinned official checkpoint; only checkpoint files were copied to Colab. No credentials were exported. Final verification uses those downloaded caches offline.

## Real-image checks

Pilots use training documents 0 and 1, with document 2 added for multimodal LayoutLMv3. Batch size is 2, including a final partial batch for the three-document run. Float32 comparisons use absolute and relative tolerance `1e-5`; float16 uses `3e-3`. Integer tensors and masks require exact equality. Storage preserves native dtypes and values without application-level pooling or normalization.

| Mode | Precision | Observed retained tensors |
| --- | --- | --- |
| DINOv3 ViT-S/16 | float32 | CLS `1 x 384`, registers `4 x 384`, patches `196 x 384` |
| LayoutLMv3 image-only | float32 | CLS `1 x 768`; visual sequence `197 x 768` with a 197-position mask |
| LayoutLMv3 with actual OCR | float32 | CLS `1 x 768`; sequences `304 x 768` for documents 0/1 and `709 x 768` for document 2; aligned text IDs, boxes, word IDs and full output mask |
| ColPali | float16 | Native `1030 x 128` vectors, IDs, padding/image/special masks |
| ColQwen2 | float16 | Batched `755 x 128` vectors with variable valid lengths and exact padding/image/special masks |

Both retrieval checkpoints already normalize in their native forward. The pipeline saves that output unchanged, including prompt/special positions and padding masks. DINOv3 register vectors are retained separately. Image-only LayoutLMv3 uses visual CLS at index zero; changing even malformed OCR input leaves every tensor unchanged.

Multimodal OCR inputs are actual engine results. Successful empty OCR and 600-word replay of recognized words are additional constructed encoder-boundary cases, independently compared with native upstream processing. They produce respectively 2/512 text positions and 199/709 total positions, retain the parent ID, and record truncation. Document 2 also naturally exceeds 512 text subtokens and exercises the truncation policy on actual OCR.

## OCR inspection

Tesseract recognized 73, 76 and 399 words in documents 0, 1 and 2. The final OCR artifact is `ocr-f130820b2fdbbe43b9b6`. Its metadata pins the engine, executable, traineddata, Pillow decoder, configuration, original image hashes, and coordinate convention. Box overlays were inspected on all three original images.

Document 0's boxes broadly track its table text, but flags and some rules produce spurious boxes. Document 1's recognized black text is localized correctly, while much white text over the blue infographic is missed. Document 2 has useful paragraph boxes with some spurious chart marks. These are recorded OCR limitations; no reference transcription fills the gaps. Engine tests also cover a truly blank rendered page, explicit failures and changed-image rejection.

## Tests and negative controls

- Local full suite: `python -m unittest discover -s ah-clustering-experiment/tests -q` ran 118 tests successfully, with one live Tesseract test skipped because the local executable is absent.
- Colab step 4 suite: 37 tests passed, zero skips. This includes real Tesseract, all new adapter/pipeline boundary tests and the independent oracle boundaries.
- Changed tokenizer, torchvision or SentencePiece versions force a new extraction request. The review found and fixed missing preprocessing dependencies in the cache fingerprint; the OCR cache also pins the image decoder version.
- Numeric value mutation, shortened masks and flipped mask values were detected. Parser checks reject invalid coordinates, confidence, IDs, mask axes and nonfinite vectors. Real stored-shard corruption and image/manifest corruption are rejected.
- Completed caches reuse the same artifact even when requested IDs are reversed; the model-loading boundary is forced to fail if called.
- A verifier replay initially passed NumPy scalar confidences from Parquet into the strict Python-scalar input contract. The verifier's independent decoding was corrected and the live checks rerun; production OCR decoding already produced valid native scalars.
- `git diff --check` passed. Shared `data/split.csv` remains SHA256 `0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c`.

A broad Colab regression run on exported source hit one existing step 2 assertion requiring a non-null Git commit. Colab received a source export without `.git`, which the production provenance API explicitly permits. That assertion passes in the real local checkout; the 37 relevant Colab tests pass. No fake Git revision was inserted. Installation also reported conflicts with unused preinstalled Colab packages (`google-colab`, Numba and cuDF); the pinned extraction stack executed successfully. Use an isolated experiment environment for repeatable setup.

## Reproduction and retained evidence

From the experiment directory, with the prepared dataset and encoder requirements installed:

```text
python experiments/verify_step4.py --config configs/models/dinov3.json --dataset-dir outputs/metadata/dataset-9c71de8a0c0e5e4aab1d --ids 0 1 --batch-size 2 --device cuda:0 --report outputs/logs/oracle-dinov3.json
```

Repeat for `layoutlmv3-image`, `colpali` and `colqwen2`; use IDs `0 1 2` for `layoutlmv3-ocr`. Approved checkpoint access or an already downloaded cache is required. The pilot's full source digest is checked at start and end. Report copies under `reports/step4-evidence/` retain artifact IDs, settings, tensor shapes, checks and the exact source digest. Numeric artifacts, OCR overlays, extraction indexes and detailed runtime manifests are saved under ignored `outputs/`.

## Limits

This evidence covers real small pilots at the configured precision and tested batch size. It does not establish all-document quality, maximum batch size, throughput benchmarks, or recovery on a mounted persistent backend. The Colab source export has no Git metadata; the matching source digest identifies the executed code. Full 1,000-document extraction, interrupted-write recovery, storage sizing and production compute setup remain step 5. Clustering begins in step 6.

## Evidence ledger

CLAIM: every requested extraction mode produces finite, correctly mapped native outputs with verified token semantics and persisted cache round-trips on real document images.

ORACLE_SOURCE: the accepted Step 4 plan, Step 3 artifact contract, official pinned Transformers implementations and separately invoked direct model forwards.

EVIDENCE_GRADE: `LIVE_VERIFIED`.

CHECKS_RUN: five final-source T4 differential pilots at batch size 2; frozen inference observations; OCR mutation invariance; empty/truncated OCR boundaries; native-value/mask negative controls; completed-cache model-loading prohibition; local 118-test suite and Colab 37-test step 4 suite.

MOCKS_AND_LOST_COVERAGE: only the completed-cache loading boundary is replaced by a deliberate failure during live verification. Fixture-based unit tests do not certify pretrained behavior. No full-dataset or mounted-storage recovery claim is made.

CONTRADICTORY_EVIDENCE: the exported-source Git assertion, verifier scalar conversion, and cache dependency omission are accounted for above. No unresolved contradiction remains for the Step 4 pilot claim.

VERDICT: PASS for Step 4. Full extraction and recovery remain Step 5.

The five final embedding artifacts and their OCR dependency were exported and passed local readers' checksum, configuration and document-ID validation; see `step4-evidence/local-readback.json`. Export recovery explicitly selected the metadata directory after the connector's prefix-style glob handling omitted it from the first transfer. All required artifacts are preserved locally. The `legal-step4` T4 assignment was released after read-back; the connector's session store confirms it is no longer allocated. Changes remain uncommitted.
