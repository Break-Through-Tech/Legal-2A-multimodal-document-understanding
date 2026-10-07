# Step 4 independent oracle and evidence

Oracle frozen on 2026-10-04 before reading the candidate encoder or pipeline.

## Acceptance contract

The authority is Step 4 of `IMPLEMENTATION_PLAN.md`, the Step 3 `ARTIFACT_FORMAT.md`, and the official Transformers 5.17.0 sources linked below. The required grade is `LIVE_VERIFIED`: prepared real images, immutable real pretrained checkpoints, actual OCR for multimodal inputs, native processors and models, and numeric/Parquet/JSON persistence must execute. Model mocks do not establish this grade. Mocks are permitted only to test failure boundaries such as proving an already complete cache does not load an encoder.

Expected invariants, independent of the candidate:

1. Encoders run in evaluation mode with no trainable parameters or gradients. Document IDs map explicitly to their image-derived outputs, including when order changes.
2. DINOv3 sequence index zero is CLS. Register vectors follow CLS; patch vectors follow the registers. Native vectors must be retained without averaging or additional normalization.
3. LayoutLMv3 image-only calls contain only image inputs, with automatic OCR disabled. The first output is its visual CLS, followed by 196 patches for a 224-by-224 input with 16-by-16 patches. Changing OCR leaves these values and identity unchanged.
4. Multimodal LayoutLMv3 outputs contain the text sequence followed by visual CLS and patches. The output attention mask must include the visual positions; input text mask alone is insufficient. Actual OCR words, boxes, word/token alignment, empty text policy, and truncation must remain traceable to the parent image and document.
5. ColPali and ColQwen2 retain the returned multi-vector sequence, including prompt and special tokens. The native models normalize vectors themselves. Saving those outputs is not an extra normalization step. Output masks must match returned sequence lengths and padding semantics, proven against the pinned native implementation rather than assumed.
6. Saved tensors must equal the native outputs with explicit dtype conversion only where required for NumPy storage. Finite checks, masks, tensor dimensions, and exact document membership are mandatory. Mutated mask length, native vector values, and OCR-dependent identity are negative controls.

## Primary sources

- [DINOv3 documentation](https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/dinov3): CLS, registers, and patch sequence order.
- [LayoutLMv3 5.17.0 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py): image-only dispatch, visual CLS insertion, text/visual concatenation, and output masks.
- [ColPali 5.17.0 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/colpali/modeling_colpali.py): retrieval projection, native normalization, padding mask.
- [ColQwen2 5.17.0 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/colqwen2/modeling_colqwen2.py): processor image-grid unpadding and retrieval output sequence.

## Preliminary environment observations

Read-only local inspection found an RTX 3050 with 4 GiB VRAM. The root environment has no Transformers. A historical LayoutLMv3 checkpoint cache exists at revision `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`; it and the historical environments were not modified by this verifier. The lead owns dependency installation, checkpoint access, and Colab T4 allocation. This report does not claim those historical caches establish current inference evidence.

## Independent boundary checks

`tests/test_step4_independent.py` passed 12 tests with `.venv/Scripts/python.exe -B -m unittest discover -s ah-clustering-experiment/tests -p test_step4_independent.py -v`.

The tests use hand-calculated asymmetric pixel boxes, coordinate flooring, empty OCR output, malformed extents, confidence limits, multiple-page rejection, and explicit record mutations. Output validation checks cover variable padding, a text-only mask incorrectly applied to longer returned sequences, nonbinary masks, nonfinite vectors, and nonintegral token IDs. These checks run real parser/validation code on literal fixtures. They do not run OCR or model inference and establish `INDEPENDENTLY_CHECKED` for those boundaries only.

Independent review also identified two integration defects before live execution: OCR identity used a records-only checksum instead of the specified validated completion-manifest checksum; and the encoder expected a confidence list under the scalar mean-confidence field. The owners corrected both contracts.

## Real differential harness

`experiments/verify_step4.py` invokes the production extraction runner and reads its persisted tensors, then separately loads the pinned official processor and model and invokes their public APIs on the same prepared images. It compares every output tensor, shape and dtype; checks padding masks; injects numerical and mask mutations; and proves completed-cache reuse with the model-loading boundary made to fail. LayoutLMv3 image-only additionally runs with deliberately corrupted OCR inputs and must retain every output value. Actual OCR is read from the persisted OCR artifact for the multimodal reference call. No model forward is mocked.

The tolerance is declared before execution: absolute and relative `1e-5` for float32, `3e-3` for float16. Masks and integer tensors require exact equality. The script aborts if the source digest changes during its run. Its CLI help imports successfully in the local dataset environment; live execution is coordinated by the lead on the allocated T4.

The harness supports `--batch-size 2` to exercise true batching and variable padding where the model's image tokenization varies. It also observes a real forward on every adapter and asserts evaluation mode, disabled parameter gradients, `torch.inference_mode()`, and no accumulated gradients. Multimodal LayoutLMv3 additionally receives two explicitly constructed edge inputs: an empty successful OCR record and 600 words replayed from actual recognized OCR. Those cases must agree with the official processor/model, produce respectively 2 and 512 text positions, and preserve parent IDs and the truncation flag. They establish encoder behavior at those boundaries, not OCR-engine accuracy on naturally empty or overlong pages.

## Evidence ledger

REVISION: base Git commit `d0650e86581b6ca89eb1ae2b6f0bef45c73727b7` plus uncommitted implementation; each live report records the exact source digest.

CLAIM: native pretrained extraction, persistence, and verified output semantics for all five requested modes.

ORACLE_SOURCE: frozen requirements and official sources above.

EVIDENCE_GRADE: `LIVE_VERIFIED` for all five small real-image pilots; `INDEPENDENTLY_CHECKED` for parser/output boundaries. Final source digest: `7acf6ea0273b332ba63421772ff955530c739242a5f049a256d0e9a7a2801335`.

REAL_COMPONENTS_EXECUTED: all five real pretrained modes on Colab T4, official processors, actual Tesseract 5.3.4, native numeric/Parquet/JSON storage and cache readers. See `STEP_4_VERIFICATION.md` and the exact reports in `step4-evidence/`.

MOCKS_AND_LOST_COVERAGE: no mocks in the 12 independent tests; their literal arrays and TSV records do not establish model or engine behavior. The differential harness uses one named boundary mock solely for completed-cache reuse.

NEGATIVE_CONTROL_RESULT: malformed geometry, mismatched IDs and masks, invalid confidence, and nonfinite vectors were rejected.

CONTRADICTORY_EVIDENCE: none unresolved in the boundary checks.

RESIDUAL_RISKS: small pilots do not establish all-document quality or maximum resource requirements. Persistence on a mounted Drive backend, interruption recovery, and the full 1,000-document runs belong to Step 5. OCR misses some infographic text; details are recorded in the lead evidence report.

VERDICT: PASS for Step 4's extraction implementation and small real-image acceptance checks. Full extraction remains Step 5.
