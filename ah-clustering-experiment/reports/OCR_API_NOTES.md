# Actual OCR contract

`src.embeddings.ocr.prepare_ocr(storage_root, documents, source_root=..., config=None)`
returns an embedding-compatible OCR identity and a document-ID-indexed record map.
Image input rows contain exactly `document_id`, `image_path`, `image_sha256`.
`source_root` points to the experiment code for source provenance; image paths
are relative to `storage_root`.

The adapter invokes the actual Tesseract CLI and parses its word-level TSV. It
requires Tesseract and requested traineddata on the host, plus existing Pillow.
No pytesseract package is needed. On Debian/Colab install `tesseract-ocr` and
`tesseract-ocr-eng`; the executable is discovered through PATH. On Windows an
explicit `executable` path or the standard Program Files installation also works.
Installation is an environment setup action, not performed by the module.

Options default to `language=eng`, `psm=3`, `oem=1`, `timeout_seconds=120`.
Optional `executable` and `tessdata_dir` choose installations. The cache records
the resolved engine version, executable SHA256, traineddata SHA256, Pillow decoder version, options,
input image identities, coordinate/image transformations and source digest.
Changing any of those creates a new cache identity. Completed corrupt caches
raise errors rather than silently becoming a cache hit.

Records are persisted in `documents.parquet`; `records.json` contains their
canonical content checksum. The embedding-facing `content_sha256` is the
validated completion manifest's `manifest_sha256`, as required by the Step 3
contract. All payloads and OCR semantics are read back before completion.

The original image is decoded to RGB at its original dimensions; no resize,
deskew, orientation transform or reference transcription is applied. Multi-frame
images fail explicitly. Words retain Tesseract TSV order. Pixel XYXY coordinates
are scaled by floor division to integer 0..1000, with top-left origin. The record
contains `words`, `boxes`, `word_confidences` (0..100), mean `confidence` (float),
`status`, original dimensions and image identity. Successful empty pages contain
empty sequences, `status=empty` and `confidence=0.0`. Engine failures raise and
publish no completion marker. OCR does not truncate; the model adapter owns its
document/chunk policy.

`render_ocr_overlay(storage_root, record, artifact_id)` writes a PNG to
`outputs/ocr/<artifact_id>/overlay-<document_id>.png`. These are inspection aids,
not inputs or immutable artifact payloads. Original image checksums are verified
before drawing. Red boxes are projected back from normalized coordinates.

For a three-document training smoke run:

```console
python experiments/prepare_ocr.py --manifest outputs/metadata/<dataset-id>/manifest.parquet --limit 3
python -m unittest discover -s tests -p test_ocr.py -v
```

The parser acceptance example uses independently calculated coordinates from
[Tesseract's documented TSV example](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).
The engine install requirements follow the
[official installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).
Parser and validation tests are self-checks. The optional live test exercises
actual Tesseract on rendered text and a blank page, persistence, cache reuse,
overlay generation and changed-image rejection. Dataset OCR and visual overlay
inspection are separate evidence required for the Step 4 live gate.
