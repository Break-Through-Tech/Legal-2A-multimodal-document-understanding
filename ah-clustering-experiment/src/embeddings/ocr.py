"""Actual Tesseract OCR on original pixels, with checksummed persistent records.

No reference transcription is accepted. Engine failures are errors, while a
successful page with no recognized words is an explicit ``empty`` record.
"""

from __future__ import annotations

import csv
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from PIL import Image, ImageDraw
import pandas as pd

from ..artifacts import ArtifactWriter, artifact_identity, read_artifact
from ..contracts import document_ids, portable_path, require, sha256
from ..dataset import _atomic_bytes, _atomic_json, file_sha256, fingerprint, owned_path, validate_storage_root
from ..provenance import capture_provenance, source_identity
from ..recovery import RecoveryJournal


COORDINATES = "XYXY; origin top-left; floor(pixel_coordinate * 1000 / image_axis); range 0..1000"
DEFAULT_CONFIG = {"language": "eng", "psm": 3, "oem": 1, "timeout_seconds": 120}


def validate_ocr_config(config: dict | None) -> dict:
    require(config is None or isinstance(config, dict), "OCR config must be an object")
    result = {**DEFAULT_CONFIG, **(config or {})}
    require(not set(result) - {*DEFAULT_CONFIG, "executable", "tessdata_dir"}, "Unknown OCR option")
    require(isinstance(result["language"], str) and re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", result["language"]) is not None,
            "OCR language must be one or more Tesseract language codes")
    require(type(result["psm"]) is int and result["psm"] in range(3, 14), "OCR psm must be 3..13 (text recognition, without OSD)")
    require(type(result["oem"]) is int and result["oem"] in range(4), "OCR oem must be 0..3")
    require(type(result["timeout_seconds"]) in {int, float} and math.isfinite(result["timeout_seconds"]) and result["timeout_seconds"] > 0,
            "OCR timeout_seconds must be finite and positive")
    for option in ("executable", "tessdata_dir"):
        require(option not in result or isinstance(result[option], str) and bool(result[option].strip()), f"Invalid OCR {option}")
    return result


def _run(command: list[str], *, timeout: float, payload: bytes | None = None) -> str:
    try:
        result = subprocess.run(command, input=payload, capture_output=True, timeout=timeout, check=True)
        return result.stdout.decode("utf-8-sig")
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        detail = getattr(error, "stderr", b"") or b""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"Actual OCR engine failed: {error}; {detail[:1000]}") from error


def _engine(config: dict) -> tuple[list[str], dict]:
    executable = config.get("executable") or shutil.which("tesseract")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR/tesseract.exe"
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        raise RuntimeError("Actual OCR requires the Tesseract executable and traineddata; install Tesseract or set OCR executable. Reference text is never substituted.")
    resolved = shutil.which(executable) or str(Path(executable).resolve())
    version = _run([resolved, "--version"], timeout=10).splitlines()[0].strip()
    require(version.lower().startswith("tesseract "), "Executable is not Tesseract")
    command = [resolved]
    if config.get("tessdata_dir"):
        command += ["--tessdata-dir", str(Path(config["tessdata_dir"]).resolve())]
    languages = _run([*command, "--list-langs"], timeout=10)
    match = re.search(r'List of available languages in "(.+?)"', languages)
    require(match is not None, "Cannot establish Tesseract traineddata directory")
    data_dir = Path(match.group(1))
    hashes = {}
    for language in config["language"].split("+"):
        path = data_dir / f"{language}.traineddata"
        require(path.is_file(), f"Missing actual OCR traineddata: {language}")
        hashes[language] = file_sha256(path)
    # Pin the resolved directory explicitly so ambient TESSDATA_PREFIX cannot
    # silently alter the recognized language data after identity discovery.
    command = [resolved, "--tessdata-dir", str(data_dir.resolve())]
    return command, {"engine": "tesseract", "engine_version": version,
                     "executable_sha256": file_sha256(Path(resolved)), "traineddata_sha256": hashes}


def parse_tsv(tsv: str, width: int, height: int) -> tuple[list[str], list[list[int]], list[float]]:
    """Read only word rows from the engine's documented TSV output."""
    require(type(width) is int and type(height) is int and width > 0 and height > 0, "Invalid image dimensions")
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    require(reader.fieldnames == ["level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"],
            "Unexpected actual OCR TSV schema")
    words, boxes, confidences = [], [], []
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), "Malformed actual OCR TSV row")
        if int(row["level"]) != 5 or not row["text"].strip():
            continue
        require(int(row["page_num"]) == 1, "OCR expects a single image page")
        left, top, box_width, box_height = (int(row[key]) for key in ("left", "top", "width", "height"))
        require(0 <= left < left + box_width <= width and 0 <= top < top + box_height <= height,
                "OCR word box lies outside the original image or has invalid extent")
        confidence = float(row["conf"])
        require(math.isfinite(confidence) and 0 <= confidence <= 100, "Invalid OCR word confidence")
        words.append(row["text"].strip())
        boxes.append([left * 1000 // width, top * 1000 // height,
                      (left + box_width) * 1000 // width, (top + box_height) * 1000 // height])
        confidences.append(confidence)
    return words, boxes, confidences


def validate_ocr_record(record: dict, document: dict) -> None:
    require(type(record.get("document_id")) is int and record["document_id"] >= 0, "Invalid OCR document ID")
    require(record.get("document_id") == document["document_id"] and record.get("image_sha256") == document["image_sha256"], "OCR image identity mismatch")
    require(sha256(record.get("image_sha256")), "Invalid OCR image checksum")
    require(record.get("image_path") == document["image_path"], "OCR image path mismatch")
    require(type(record.get("image_width")) is int and record["image_width"] > 0 and type(record.get("image_height")) is int and record["image_height"] > 0,
            "Invalid OCR image dimensions")
    words, boxes, confidences = (record.get(key) for key in ("words", "boxes", "word_confidences"))
    require(all(isinstance(value, list) for value in (words, boxes, confidences)), "OCR sequences must be lists")
    require(len(words) == len(boxes) == len(confidences), "OCR word/box/confidence lengths disagree")
    require(all(isinstance(word, str) and bool(word.strip()) and word == word.strip() for word in words), "Invalid OCR word")
    for box in boxes:
        require(isinstance(box, list) and len(box) == 4 and all(type(v) is int and 0 <= v <= 1000 for v in box)
                and box[0] <= box[2] and box[1] <= box[3], "Invalid normalized OCR box")
    require(all(type(value) in {float, int} and math.isfinite(value) and 0 <= value <= 100 for value in confidences), "Invalid OCR confidence")
    require(record.get("status") == ("ok" if words else "empty"), "OCR status disagrees with word count")
    expected_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    require(type(record.get("confidence")) is float and record["confidence"] == expected_confidence, "OCR mean confidence mismatch")


def _validated_records(bundle: dict) -> tuple[dict, dict[int, dict]]:
    payload = bundle["json"]["records"]
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if hasattr(value, "tolist"):
            return plain(value.tolist())
        return value
    records = plain(bundle["tables"]["documents"].to_dict("records"))
    expected = bundle["config"]["documents"]
    require(isinstance(records, list) and len(records) == len(expected), "OCR document count mismatch")
    document_ids(record["document_id"] for record in records)
    for record, document in zip(records, expected):
        validate_ocr_record(record, document)
    require(payload["content_sha256"] == fingerprint(records), "OCR content checksum mismatch")
    config = bundle["config"]
    identity = {"artifact_id": bundle["manifest"]["artifact_id"], "content_sha256": bundle["manifest"]["manifest_sha256"],
                "engine": config["engine"], "engine_version": config["engine_version"],
                "config": {key: config[key] for key in ("options", "coordinate_transform", "image_transform", "image_decoder", "executable_sha256", "traineddata_sha256")}}
    return identity, {record["document_id"]: record for record in records}


def prepare_ocr(storage_root: Path, documents: list[dict], *, source_root: Path,
                config: dict | None = None) -> tuple[dict, dict[int, dict]]:
    """Run or reuse actual OCR. Input rows contain only ID, path and image SHA."""
    started = time.perf_counter()
    root = validate_storage_root(storage_root)
    request_id = fingerprint({"documents": documents, "options": config,
                              "source_digest": source_identity(source_root)["source_digest"]})
    log_path = owned_path(root, f"outputs/logs/ocr-request-{request_id[:20]}-{uuid.uuid4().hex[:12]}.json")
    progress = {"status": "running", "request_id": request_id, "artifact_id": None,
                "expected_document_count": len(documents), "completed_document_ids": [],
                "completed_document_count": 0, "reused_document_count": 0,
                "current_document_id": None}

    def report():
        progress["elapsed_seconds"] = time.perf_counter() - started
        progress["completed_document_count"] = len(progress["completed_document_ids"])
        _atomic_json(log_path, progress)

    try:
        report()
        result = _prepare_ocr(root, documents, source_root=source_root, config=config,
                              started=started, progress=progress, report=report)
        progress.update(status="complete", artifact_id=result[0]["artifact_id"],
                        completed_document_ids=sorted(result[1]), current_document_id=None)
        report()
        return result
    except BaseException as error:
        progress.update(status="failed", error_type=type(error).__name__, error=str(error))
        try:
            report()
        except Exception as log_error:
            error.add_note(f"Could not persist OCR failure log: {log_error}")
        raise


def _prepare_ocr(root: Path, documents: list[dict], *, source_root: Path, config: dict | None,
                 started: float, progress: dict, report) -> tuple[dict, dict[int, dict]]:
    options = validate_ocr_config(config)
    document_ids(row["document_id"] for row in documents)
    documents = sorted([dict(row) for row in documents], key=lambda row: row["document_id"])
    for row in documents:
        require(set(row) == {"document_id", "image_path", "image_sha256"}, "OCR accepts image identity only, never reference text")
        portable_path(row["image_path"])
        require(sha256(row["image_sha256"]), "Invalid image SHA256")
        require(file_sha256(owned_path(root, row["image_path"])) == row["image_sha256"], "OCR source image checksum mismatch")
    command, engine = _engine(options)
    identity_config = {**engine, "documents": documents,
                       "image_decoder": {"Pillow": importlib.metadata.version("Pillow")},
                       "options": {key: options[key] for key in DEFAULT_CONFIG},
                       "coordinate_transform": COORDINATES,
                       "image_transform": "single-page decoded RGB PNG; original size; no EXIF rotation, resize or deskew",
                       "source_digest": source_identity(source_root)["source_digest"]}
    artifact_id, _ = artifact_identity("ocr", identity_config)
    progress["artifact_id"] = artifact_id
    report()
    marker = owned_path(root, f"outputs/metadata/{artifact_id}/artifact.json")
    with RecoveryJournal(root, artifact_id, identity_config) as journal:
        if marker.exists():
            restored = _validated_records(read_artifact(root, artifact_id, expected_config=identity_config))
            progress["reused_document_count"] = len(restored[1])
            return restored
        journal.clear_artifact_lock(artifact_id)
        records = []
        reused_documents = 0
        for row in documents:
            progress["current_document_id"] = row["document_id"]
            report()
            path = owned_path(root, row["image_path"])
            require(file_sha256(path) == row["image_sha256"], "OCR image changed during processing")
            name = f"document-{row['document_id']}"
            payload = journal.get(name)
            record = None
            if payload is not None:
                try:
                    candidate = json.loads(payload)
                    require(isinstance(candidate, dict), "OCR checkpoint must be an object")
                    validate_ocr_record(candidate, row)
                    record = candidate
                except (ValueError, TypeError, KeyError, UnicodeError):
                    # A checksum-valid but invalid record is unfinished too.
                    record = None
            with Image.open(path) as image:
                require(getattr(image, "n_frames", 1) == 1, "OCR requires a single-frame source image")
                width, height = image.size
                if record is not None and (record["image_width"], record["image_height"]) == (width, height):
                    require(file_sha256(path) == row["image_sha256"], "OCR image changed during processing")
                    records.append(record)
                    reused_documents += 1
                    progress["completed_document_ids"].append(row["document_id"])
                    progress["reused_document_count"] = reused_documents
                    report()
                    continue
                encoded = io.BytesIO()
                image.convert("RGB").save(encoded, format="PNG")
            tsv = _run([*command, "stdin", "stdout", "-l", options["language"], "--psm", str(options["psm"]),
                        "--oem", str(options["oem"]), "-c", "tessedit_create_tsv=1"],
                       timeout=options["timeout_seconds"], payload=encoded.getvalue())
            words, boxes, confidences = parse_tsv(tsv, width, height)
            record = {**row, "image_width": width, "image_height": height, "words": words, "boxes": boxes,
                      "word_confidences": confidences, "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                      "status": "ok" if words else "empty"}
            validate_ocr_record(record, row)
            require(file_sha256(path) == row["image_sha256"], "OCR image changed during processing")
            journal.put(name, json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8"))
            records.append(record)
            progress["completed_document_ids"].append(row["document_id"])
            report()
        with ArtifactWriter(root, "ocr", identity_config) as writer:
            journal.mark_artifact_writer(artifact_id)
            writer.table("documents", pd.DataFrame(records))
            writer.json("records", {"content_sha256": fingerprint(records)})
            writer.complete(capture_provenance(source_root, started, {"engine": engine, "documents": len(records),
                             "reused_documents": reused_documents,
                             "processed_documents": len(records) - reused_documents,
                             "words": sum(len(record["words"]) for record in records)}), validator=_validated_records)
        return _validated_records(read_artifact(root, artifact_id, expected_config=identity_config))


def render_ocr_overlay(storage_root: Path, record: dict, artifact_id: str) -> Path:
    """Draw normalized word boxes on original pixels for visual inspection."""
    root = validate_storage_root(storage_root)
    require(re.fullmatch(r"ocr-[0-9a-f]{20}", artifact_id) is not None, "Invalid OCR artifact ID")
    validate_ocr_record(record, record)
    portable_path(record["image_path"])
    source = owned_path(root, record["image_path"])
    require(file_sha256(source) == record["image_sha256"], "OCR overlay source checksum mismatch")
    with Image.open(source) as image:
        require(image.size == (record["image_width"], record["image_height"]), "OCR overlay dimensions mismatch")
        overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for box in record["boxes"]:
        draw.rectangle([box[0] * overlay.width / 1000, box[1] * overlay.height / 1000,
                        box[2] * overlay.width / 1000, box[3] * overlay.height / 1000], outline="#ee2222", width=2)
    path = owned_path(root, f"outputs/ocr/{artifact_id}/overlay-{record['document_id']}.png")
    encoded = io.BytesIO()
    overlay.save(encoded, format="PNG")
    _atomic_bytes(path, encoded.getvalue())
    return path
