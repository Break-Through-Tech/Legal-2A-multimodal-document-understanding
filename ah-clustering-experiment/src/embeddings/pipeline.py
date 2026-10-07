"""Label-free, batched extraction into the existing native tensor artifact format."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

from PIL import Image
import pandas as pd

from . import read_embeddings, write_embeddings
from .checkpoints import encode_batch, decode_batch
from ..artifacts import read_artifact, artifact_identity
from ..contracts import document_ids as checked_ids, embedding_config, require, portable_path, sha256, validate_embedding_config
from ..dataset import (_atomic_json, configure_cache, file_sha256, fingerprint,
                       owned_path, validate_assignments, validate_storage_root,
                       _validate_preparation_metadata, DATASET_NAME, DATASET_REVISION)
from ..provenance import source_identity
from ..recovery import RecoveryJournal


def validate_settings(settings: dict) -> None:
    required = {"family", "checkpoint", "revision", "input_mode", "dtype", "batch_size", "processor_options"}
    require(isinstance(settings, dict) and set(settings) == required, "Unexpected model configuration fields")
    require(settings["family"] in {"dinov3", "colpali", "colqwen2", "layoutlmv3"}, "Unknown encoder family")
    require(isinstance(settings["checkpoint"], str) and bool(settings["checkpoint"].strip()), "Missing checkpoint")
    require(isinstance(settings["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", settings["revision"]) is not None,
            "Checkpoint revision must be an immutable 40-character commit")
    require(settings["input_mode"] in {"image_only", "image_and_ocr"}, "Unknown input mode")
    require(settings["family"] == "layoutlmv3" or settings["input_mode"] == "image_only", "OCR is only supported by LayoutLMv3")
    require(settings["dtype"] in {"float32", "float16"}, "Use an exactly serializable float32 or float16 dtype")
    require(type(settings["batch_size"]) is int and settings["batch_size"] > 0, "batch_size must be positive")
    require(isinstance(settings["processor_options"], dict), "processor_options must be an object")
    fingerprint(settings)


def load_documents(storage_root: Path, dataset_dir: Path, ids: list[int] | None = None) -> tuple[dict, list[dict]]:
    """Verify manifest provenance, then expose only ID/image identity to encoders.

    Only selected images need to be present for a pilot. The complete 1,000-row
    manifest remains authoritative; labels and partitions never leave this function.
    """
    root = validate_storage_root(storage_root)
    directory = Path(dataset_dir).resolve()
    require(directory.is_relative_to(root), "Dataset metadata must stay inside storage root")
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    require(isinstance(metadata, dict) and metadata.get("status") == "complete", "Dataset is not complete")
    _validate_preparation_metadata(metadata)
    identity = metadata["identity"]
    require(identity["dataset"] == {"name": DATASET_NAME, "revision": DATASET_REVISION, "upstream_split": "test"},
            "Unexpected dataset source")
    require(metadata["identity_sha256"] == fingerprint(identity)
            and metadata["dataset_id"] == "dataset-" + fingerprint(identity)[:20], "Dataset identity mismatch")
    manifest_path = directory / "manifest.parquet"
    require(file_sha256(manifest_path) == metadata["manifest_sha256"], "Dataset manifest checksum mismatch")
    frame = pd.read_parquet(manifest_path)
    validate_assignments(frame.rename(columns={"document_id": "id"}))
    require(metadata["document_count"] == len(frame) and metadata["final_label_count"] == frame.label.nunique()
            and metadata["partition_counts"] == frame.split.value_counts().to_dict(), "Dataset summary mismatch")
    require((frame.dataset_revision == DATASET_REVISION).all()
            and (frame.shared_csv_sha256 == identity["shared_csv_sha256"]).all(), "Dataset row provenance mismatch")
    selected = checked_ids(ids) if ids is not None else checked_ids(frame.document_id)
    require(set(selected).issubset(set(frame.document_id)), "Requested document IDs are absent from dataset")
    rows = frame.loc[frame.document_id.isin(selected), ["document_id", "image_path", "image_sha256"]]
    documents = rows.sort_values("document_id").to_dict("records")
    for document in documents:
        portable_path(document["image_path"])
        require(sha256(document["image_sha256"]), "Missing image checksum")
        path = owned_path(root, document["image_path"])
        require(file_sha256(path) == document["image_sha256"], f"Image checksum mismatch for {document['document_id']}")
        row = frame.loc[frame.document_id == document["document_id"]].iloc[0]
        with Image.open(path) as image:
            require(getattr(image, "n_frames", 1) == 1, "Extraction requires a single image page")
            require((image.width, image.height, image.format) == (row.image_width, row.image_height, row.image_format),
                    "Image dimensions or format differ from the dataset manifest")
    dataset = {key: metadata[key] for key in ("dataset_id", "identity_sha256", "manifest_sha256")}
    return dataset, documents


def extract_embeddings(settings: dict, storage_root: Path, dataset_dir: Path, *, source_root: Path,
                       document_ids: list[int] | None = None, device: str = "cpu",
                       ocr_config: dict | None = None) -> dict:
    """Run a configured frozen encoder, or verify and reuse an exact completed cache.

    Checkpoints preserve complete inference batches across process interruptions.
    A corrupt or unfinished checkpoint is inferred again with the original batch.
    """
    validate_settings(settings)
    root = validate_storage_root(storage_root)
    configure_cache(root)
    dataset, documents = load_documents(root, dataset_dir, document_ids)
    ocr_identity, ocr_records = None, None
    if settings["input_mode"] == "image_and_ocr":
        from .ocr import prepare_ocr
        ocr_identity, ocr_records = prepare_ocr(root, documents, source_root=source_root, config=ocr_config)
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("torch", "torchvision", "transformers", "tokenizers", "sentencepiece",
                     "safetensors", "huggingface_hub", "Pillow", "numpy")}
    request = {"settings": settings, "dataset": dataset, "documents": documents,
               "ocr": ocr_identity, "device": device, "dependencies": dependencies,
               "source_digest": source_identity(source_root)["source_digest"]}
    request_id = fingerprint(request)
    with RecoveryJournal(root, f"extraction-{request_id[:20]}", request) as journal:
        return _extract_locked(settings, root, source_root, dataset, documents, device,
                               ocr_identity, ocr_records, request, request_id, journal)


def _extract_locked(settings, root, source_root, dataset, documents, device,
                    ocr_identity, ocr_records, request, request_id, journal):
    index_path = owned_path(root, f"outputs/metadata/extraction-{request_id[:20]}.json")
    log_path = owned_path(root, f"outputs/logs/extraction-{request_id[:20]}.json")
    started = time.perf_counter()
    try:
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            require(index["request"] == request, "Extraction index configuration mismatch")
            bundle = read_artifact(root, index["artifact_id"])
            config = bundle["config"]
            require(config["processor"]["extraction_request"] == request_id
                    and config["documents"] == documents and config["dataset"] == dataset
                    and config["source_digest"] == request["source_digest"] and config["ocr"] == ocr_identity,
                    "Extraction index points at incompatible embeddings")
            restored = read_embeddings(root, config)
            return {"artifact_id": index["artifact_id"], "document_count": len(restored), "reused": True,
                    "config": config, "request_id": request_id}

        encoder = None
        saved_config = journal.get("embedding-config")
        config = None
        if saved_config is not None:
            try:
                candidate = json.loads(saved_config)
                validate_embedding_config(candidate)
                require(candidate["processor"]["execution"] == request
                        and candidate["processor"]["extraction_request"] == request_id
                        and candidate["dataset"] == dataset and candidate["documents"] == documents
                        and candidate["source_digest"] == request["source_digest"]
                        and candidate["ocr"] == ocr_identity, "Restart configuration differs")
                config = candidate
            except (ValueError, TypeError, KeyError, AttributeError):
                pass

        def load():
            from .models import load_encoder
            return load_encoder(settings, cache_dir=owned_path(root, "cache/huggingface/hub"), device=device)

        def configuration(loaded):
            return embedding_config(
            model=loaded.metadata["model"],
            processor={**loaded.metadata["processor"], "extraction_request": request_id, "execution": request},
            dataset=dataset, documents=documents, input_mode=settings["input_mode"],
            transformations=loaded.metadata["transformations"], tensors=loaded.tensor_schema,
            token_policy=loaded.metadata["token_policy"], source_digest=request["source_digest"], ocr=ocr_identity)

        if config is None:
            encoder = load()
            config = configuration(encoder)
            journal.put("embedding-config", json.dumps(config, allow_nan=False).encode())
        artifact_id, _ = artifact_identity("embeddings", config)
        marker = owned_path(root, f"outputs/metadata/{artifact_id}/artifact.json")
        # Recover the publication/index gap without another model load or write.
        if marker.exists():
            restored = read_embeddings(root, config)
            _atomic_json(index_path, {"request": request, "artifact_id": artifact_id})
            return {"artifact_id": artifact_id, "document_count": len(restored), "reused": True,
                    "config": config, "request_id": request_id}
        journal.clear_artifact_lock(artifact_id)
        measurements = {"device": device, "batch_size": settings["batch_size"], "precision": settings["dtype"],
                        "resumed_documents": 0, "inferred_documents": 0, "inference_seconds": 0.0,
                        "retained_tensor_bytes": 0}

        def batches():
            nonlocal encoder
            for offset in range(0, len(documents), settings["batch_size"]):
                batch = documents[offset:offset + settings["batch_size"]]
                ids = [row["document_id"] for row in batch]
                name = f"batch-{offset:05d}"
                cached = decode_batch(journal.get(name), config, ids)
                if cached is not None:
                    measurements["resumed_documents"] += len(ids)
                    measurements["retained_tensor_bytes"] += sum(a.nbytes for row in cached for a in row["tensors"].values())
                    yield from cached
                    continue
                if encoder is None:
                    encoder = load()
                    require(configuration(encoder) == config, "Encoder differs from restart configuration")
                images = []
                try:
                    for row in batch:
                        path = owned_path(root, row["image_path"])
                        require(file_sha256(path) == row["image_sha256"], "Image changed during extraction")
                        with Image.open(path) as original:
                            images.append(original.convert("RGB"))
                    records = [ocr_records[doc_id] for doc_id in ids] if ocr_records is not None else None
                    if records is not None:
                        from .ocr import validate_ocr_record
                        for record, row in zip(records, batch):
                            validate_ocr_record(record, row)
                    inference_started = time.perf_counter()
                    outputs = encoder.encode(images, ids, ocr_records=records)
                    measurements["inference_seconds"] += time.perf_counter() - inference_started
                    require([output["document_id"] for output in outputs] == ids, "Encoder changed document IDs or batch order")
                    journal.put(name, encode_batch(outputs, config, ids))
                    measurements["inferred_documents"] += len(ids)
                    measurements["retained_tensor_bytes"] += sum(a.nbytes for row in outputs for a in row["tensors"].values())
                    yield from outputs
                finally:
                    for image in images:
                        image.close()
            measurements["inference_documents_per_second"] = (
                measurements["inferred_documents"] / measurements["inference_seconds"]
                if measurements["inference_seconds"] else None)
            measurements["estimated_1000_document_tensor_bytes"] = (
                measurements["retained_tensor_bytes"] * 1000 / len(documents))
            # Optional hardware observations, without importing torch on cache-only runs.
            torch = sys.modules.get("torch")
            measurements["cuda_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device) if torch is not None and device.startswith("cuda") else None)
            measurements["cuda_peak_reserved_bytes"] = (
                torch.cuda.max_memory_reserved(device) if torch is not None and device.startswith("cuda") else None)

        artifact_id = write_embeddings(root, config, batches(), source_root=source_root,
                                       shard_size=settings["batch_size"],
                                       measurements=measurements, journal=journal)
        restored = read_embeddings(root, config)
        _atomic_json(index_path, {"request": request, "artifact_id": artifact_id})
        _atomic_json(log_path, {"status": "complete", "artifact_id": artifact_id,
                               "document_count": len(restored), "measurements": measurements,
                               "elapsed_seconds": time.perf_counter() - started})
        return {"artifact_id": artifact_id, "document_count": len(restored), "reused": False,
                "config": config, "request_id": request_id}
    except Exception as error:
        _atomic_json(log_path, {"status": "failed", "request_id": request_id,
                               "error_type": type(error).__name__, "error": str(error),
                               "elapsed_seconds": time.perf_counter() - started})
        raise
