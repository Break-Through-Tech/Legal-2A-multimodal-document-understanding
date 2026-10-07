# Step 5 compute and existing-cache audit

Audit date: 2026-10-06. This report separates retained evidence from verification of the new recovery implementation. Existing caches were not rewritten.

## Existing native caches

The audit loaded the pinned dataset manifest and all 1,000 original images through `load_documents`, then opened all nine published embedding artifacts through both `read_artifact` and `read_embeddings`. The APIs checked artifact/configuration identity, payload checksums, unique document mapping, native tensor shapes/dtypes and aligned offsets/masks. An additional pass required every numeric tensor value to be finite and every mask value to be binary.

| Existing mode | Artifact | Expected IDs | Validated payload bytes |
| --- | --- | --- | ---: |
| DINOv3 | `embeddings-9ef1694d986576ba0acf` | All 1,000 | 309,224,564 |
| LayoutLMv3 image-only | `embeddings-af5fe5262fbc515a18be` | All 1,000 | 608,944,885 |
| ColPali | `embeddings-d987f31cbdb073fed394` | All 1,000 | 275,764,459 |
| ColQwen2 | `embeddings-17d7e1b5ec380770c824` | All 1,000 | 199,081,397 |
| LayoutLMv3 with OCR | `embeddings-6c93da1854e5670cfe3d` | Only pilot IDs 0, 1, 2 | 4,113,699 |

All four full caches match the exact manifest ID set 0–999, with no missing or unexpected IDs. The remaining four image-only pilot artifacts also passed validation. All nine have historical source digest `7acf6ea0273b332ba63421772ff955530c739242a5f049a256d0e9a7a2801335`; they do not establish recovery behavior for the new implementation. DINOv3 retains CLS, patches and register vectors; its format does not need an attention mask.

The local OCR artifact `ocr-f130820b2fdbbe43b9b6` passed `read_artifact` and the OCR record validator for IDs 0, 1 and 2, with 21,144 payload bytes. There is no published full OCR artifact in this experiment. The historical external `D:/ColabRuns/legal-20261004/bundles/layoutlmv3-ocr/` transfer contains a 1,690,174,931-byte partial file that fails `zipfile.is_zipfile`; it was not extracted or treated as valid data. `D:/ColabRuns/verified-20261004/` contains only the four image-only modes.

The ignored machine-readable audit is `outputs/logs/step5-audit-existing-caches.json`; the full read took 179.3 seconds. It records all artifact IDs, tensor layouts, source identities, sizes and missing-ID lists for the pilots.

## Colab availability

A fresh session `step5-20261006` allocated successfully and executed Python on a real Tesla T4. Observed versions were Python 3.13.15 and PyTorch 2.11.0+cu130, with CUDA available. `/content/drive` and `/content/drive/MyDrive` were absent and `/proc/mounts` had no Drive/FUSE/GCS mount. No credentials were requested or inspected. The session was stopped immediately after this audit; the connector confirmed the active runtime was released.

This proves T4 allocation works. It does not establish a durable Colab output backend. Extraction into `/content` alone would repeat the loss that affected the historical OCR transfer. Mounted-storage interruption/reconnect testing remains separate from local filesystem testing.

## Local pilot environment

The local machine exposes an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4,096 MiB. The original root `.venv` has storage libraries but no torch or Transformers. A new environment inside `ah-clustering-experiment/.venv` uses Python 3.11.15 and an explicitly recorded read-only `.pth` reference to the existing root `cache/local-models-venv/Lib/site-packages`. That supplies torch 2.6.0+cu124 and torchvision 0.21.0+cu124 without modifying the older environment. All newly installed packages and their installer cache are experiment-owned.

Transformers 5.17.0, tokenizers 0.23.2, safetensors 0.8.0, sentencepiece 0.2.2, datasets 5.0.1, huggingface_hub 1.30.0, pandas 3.0.5, pyarrow 25.0.1 and Pillow 12.3.0 were installed. NumPy is 2.4.6: the documented 2.5.3 pin requires Python >=3.12 and cannot run in the reused Python 3.11 torch environment. This local stack is a distinct measured configuration, not an exact reproduction of the Colab stack.

The existing pinned LayoutLMv3 safetensors checkpoint and six supporting files were copied from the historical root cache into the experiment cache; the source files were unchanged. Revision `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4` loaded successfully offline on `cuda:0` in 22.1 seconds. The experiment already contains the DINOv3 weights at revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`. No local Tesseract was present on PATH or in the standard Windows installation path at audit time.

`experiments/verify_step5_live.py` is the real pretrained three-document probe. It compares direct uninterrupted encoder output to a subprocess extraction terminated before its second batch checkpoint rename and then restarted in a fresh process. It declares float tolerance `rtol=atol=1e-5`, exact integer/mask comparison, an inference trace proving the completed first document was not inferred again, and completed-cache reuse with model loading forbidden.

The local LayoutLMv3 image-only probe passed at source digest `19f95fe14111b8abff529bad79c673e054befb50859b71e68efec76c5e4c63d6`. It used training IDs 0, 1 and 2, batch size 1 and float32. The deliberately terminated child exited with code 74 before `batch-00001.bin` was committed. The recorded inference calls were crash: `[0]`, `[1]`; resumed process: `[1]`, `[2]`. Document 0's completed checkpoint therefore survived without a second inference. All output tensors and token metadata matched direct uninterrupted real pretrained inference, with maximum absolute tensor error **0.0**, no missing/duplicate IDs and exact masks. A third process reused the published cache while its model-loader instrumentation rejected any encoder loading.

The resulting artifact is `embeddings-324ac06ad4d360c926c7`; complete evidence is saved at `outputs/logs/step5-live-local-layoutlmv3-20261006.json` with phase-specific output and inference trace alongside it. Measured peak GPU allocated memory was 528,624,128 bytes and reserved memory was 566,231,040 bytes. The resumed run inferred two documents at 4.736 documents/second and reused one document. Final retained tensors occupied 1,825,359 bytes, published payloads 1,835,531 bytes, and the checkpoint directory 2,445,384 bytes including the uncommitted temporary write left by the killed process. The fixed image-only layout extrapolates to 608,453,000 tensor bytes for 1,000 documents, before file overhead and checkpoint duplication.

This is real process-termination recovery on the actual local D: filesystem with a frozen source digest and pretrained GPU inference. It does not establish Google Drive recovery, multimodal OCR recovery, or new full-corpus extraction.

## Storage observations

At initial audit, the experiment held 378,089,858 bytes of original images and the four full native caches totaled 1,393,015,305 payload bytes. All embedding files together occupied 1,413,170,556 bytes. The local D: volume had approximately 209.9 GB free. These figures exclude new recovery checkpoints, model-download growth, and any additional OCR result.

The recovery format retains checkpoint payloads as well as final artifacts, so capacity planning must include both copies, metadata, original images, model weights and temporary writes. Historical full-cache bytes provide measured image-only sizing; the three-page OCR pilot cannot guarantee a full-corpus OCR tensor estimate because token lengths vary. These numbers are capacity observations, not a claim that the five-mode step 5 completion predicate has passed.
