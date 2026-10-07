# Step 4 model API decisions

Inspected on 2026-10-04. This note records source-level contracts; the separate Step 4 verification report records actual pretrained execution. Boundary tests do not count as live encoder evidence.

## Checkpoints

| Family | Native Transformers checkpoint | Immutable Hub revision | Access |
| --- | --- | --- | --- |
| DINOv3 ViT-S/16 | `facebook/dinov3-vits16-pretrain-lvd1689m` | `114c1379950215c8b35dfcd4e90a5c251dde0d32` | Manual approval required |
| ColPali | `vidore/colpali-v1.3-hf` | `133a9eb02947310513f52f8f0d39d622e0eab8dc` | Public |
| ColQwen2 | `vidore/colqwen2-v1.0-hf` | `ddc07d2317c80f75fc742b7362ee9ad1912908f9` | Public |
| LayoutLMv3, both modes | `microsoft/layoutlmv3-base` | `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4` | Public |

Revisions were read from the official Hub model API and the matching model cards. Model and processor loads both receive the same immutable revision. The Col models use the fully merged `-hf` checkpoints; loading an adapter whose base model follows a mutable revision is unnecessary.

The verified T4 runtime uses Python 3.13, `torch==2.11.0+cu130`, `transformers==5.17.0`, and `huggingface_hub==1.30.0`. These versions expose all four native classes. Exact runtime dependencies are retained in each execution manifest. No `colpali-engine`, PEFT, bitsandbytes, or custom remote model code is required. The global local environment has Transformers 4.57.3, but is not the evidence runtime. In particular, installing old Transformers 4.x into the experiment's Hub 1.x environment would introduce a dependency conflict.

## Native output contracts

- [DINOv3 documentation](https://huggingface.co/docs/transformers/model_doc/dinov3) and [5.17.0 source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py): `DINOv3ViTModel.last_hidden_state` orders CLS, register tokens, then image patches. The adapter preserves all three separately. It checks the token count against the actual processed image dimensions and configured patch and register sizes. It does not average patches or call `pooler_output`.
- [ColPali checkpoint](https://huggingface.co/vidore/colpali-v1.3-hf) and [5.17.0 source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/colpali/modeling_colpali.py): `ColPaliForRetrieval.embeddings` contains the retrieval projection. Normalization occurs inside the official model forward; it is retained exactly.
- [ColQwen2 documentation](https://huggingface.co/docs/transformers/model_doc/colqwen2) and [5.17.0 source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/colqwen2/modeling_colqwen2.py): `ColQwen2ForRetrieval` and `ColQwen2Processor` are the native pair. The expanded image tokens and fixed visual prompt have one returned embedding each. The adapter checks both output batch and token axes before accepting the processor mask. Full vectors, input IDs, padding mask, image mask, and special mask are retained; prompt tokens are not silently discarded. Grid metadata records variable image patch arrangements.
- [LayoutLMv3 documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) and [5.17.0 source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py): bare `LayoutLMv3Model` accepts `pixel_values` without `input_ids`. Its visual embedding path prepends a learned visual CLS before patches. Thus image-only output index zero is visual CLS; no text tokenizer is used in that mode. Multimodal sequence order is text tokens, visual CLS, patches. Its index zero is text CLS, and the complete output mask appends valid visual positions to the text attention mask.

## OCR and numerical policies

Layout's processor OCR is disabled explicitly in both modes. Image-only never examines the optional OCR argument. Multimodal inputs require cached actual OCR; no reference transcriptions enter the adapter. The OCR contract uses `word_confidences` in Tesseract's 0..100 scale, words, normalized integer xyxy boxes in 0..1000, status, document ID, and image checksum.

OCR mode keeps the first 512 text subtokens including special tokens. It records the full untruncated subtoken count, truncation flag, original and retained word counts, and each retained token's original word index (`-1` for non-word tokens). A partially truncated final word can have retained subtokens; `retained_word_count` counts words with any retained subtoken. Empty successful OCR produces the tokenizer's text special tokens plus the image. Failed OCR stops the run. This mode uses one parent document record, with no chunks.

Models are frozen, placed in evaluation mode, and called inside `torch.inference_mode()`. Only configured float32 or float16 outputs are accepted; float16 suitability must be measured on the intended GPU. NumPy conversion preserves the returned floating dtype exactly and rejects nonfinite values. No adapter-level normalization, pooling, quantization, or dtype fallback is applied. Padding positions remain in native outputs with an explicit mask. Processor image transformations, tokenizer identity, fixed prompt, model configuration, batch size, and precision are recorded.

## Boundary checks and remaining gates

`python -m unittest discover -s ah-clustering-experiment/tests -p test_embedding_models.py` passed 10 tests. Negative controls reject mutable revisions, unsupported precision, automatic OCR injection, invalid batch sizes, incompatible output/mask/token-ID axes, nonbinary masks, nonfinite vectors, and malformed OCR. These use synthetic boundary inputs only and do not establish pretrained inference success.

Real-image native-forward comparisons, exact artifact round trips, empty-OCR and truncation execution, image-only OCR invariance, DINO checkpoint access, and T4 precision/capacity are recorded by the lead and independent verifier. Until their evidence is available, these remain live verification gates rather than claims established by this note.
