"""Native tensor persistence only; this module performs no inference or pooling."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..artifacts import ArtifactWriter, artifact_identity, read_artifact
from ..contracts import document_ids, numeric_array, require, validate_embedding_config
from ..dataset import fingerprint
from ..provenance import capture_provenance


def _validate_document(document: dict, config: dict) -> None:
    document_ids([document["document_id"]])
    require(set(document) == {"document_id", "tensors", "token_metadata"}, "Unexpected embedding document fields")
    require(isinstance(document["token_metadata"], dict) and bool(document["token_metadata"]), "Per-document token metadata is required")
    fingerprint(document["token_metadata"])
    require(set(document["tensors"]) == set(config["tensors"]), "Missing or unexpected output tensor")
    lengths = {}
    for name, spec in config["tensors"].items():
        array = document["tensors"][name]
        numeric_array(array)
        require(array.dtype == np.dtype(spec["dtype"]) and list(array.shape[1:]) == spec["tail_shape"], f"Tensor dtype/shape mismatch: {name}")
        group = spec["alignment"]
        require(group not in lengths or lengths[group] == len(array), f"Output mask/token alignment mismatch: {name}")
        lengths[group] = len(array)


def _decode(bundle: dict) -> dict[int, dict]:
    config = bundle["config"]
    validate_embedding_config(config)
    require(bundle["manifest"]["kind"] == "embeddings", "Not an embedding artifact")
    require(set(bundle["tables"]) == {"documents"} and not bundle["json"], "Unexpected embedding payloads")
    frame = bundle["tables"]["documents"]
    expected_columns = {"document_id", "image_path", "image_sha256", "shard", "token_metadata"}
    for name in config["tensors"]:
        expected_columns.update({f"{name}__offset", f"{name}__length"})
    require(set(frame.columns) == expected_columns, "Unexpected document record schema")
    ids = document_ids(frame.document_id)
    images = {row["document_id"]: row for row in config["documents"]}
    require(set(ids) == set(images), "Missing or unexpected embedding document IDs")
    # Pandas groupby drops null keys by default. Validate every record before
    # grouping so a malformed shard cannot silently remove a document.
    require(all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0
                for value in frame["shard"]), "Every document must have a nonnegative integer shard index")
    used = set()
    result = {}
    for shard, rows in frame.groupby("shard", sort=True):
        require(isinstance(shard, (int, np.integer)) and not isinstance(shard, (bool, np.bool_)) and shard >= 0, "Invalid shard index")
        # Offsets are checked independently of record ordering.
        for name in config["tensors"]:
            key = f"shard-{shard:05d}-{name}"
            require(key in bundle["arrays"], "Missing tensor shard")
            used.add(key)
            array = bundle["arrays"][key]
            cursor = 0
            for row in rows.sort_values([f"{name}__offset", f"{name}__length"]).to_dict("records"):
                offset, length = row[f"{name}__offset"], row[f"{name}__length"]
                require(type(offset) is int and type(length) is int and length >= 0 and offset == cursor,
                        "Invalid, overlapping or gapped tensor offsets")
                cursor += length
                require(cursor <= len(array), "Tensor span exceeds shard")
            require(cursor == len(array), "Unreferenced tensor values")
        for row in rows.to_dict("records"):
            doc_id = row["document_id"]
            require(all(row[k] == images[doc_id][k] for k in ("image_path", "image_sha256")), "Image identity mismatch")
            tensors = {}
            for name in config["tensors"]:
                start, length = row[f"{name}__offset"], row[f"{name}__length"]
                tensors[name] = bundle["arrays"][f"shard-{shard:05d}-{name}"][start:start + length]
            document = {"document_id": doc_id, "tensors": tensors, "token_metadata": json.loads(row["token_metadata"])}
            _validate_document(document, config)
            result[doc_id] = document
    require(used == set(bundle["arrays"]), "Unexpected tensor shard")
    require(set(result) == set(images), "Missing or unexpected decoded document IDs")
    return result


def write_embeddings(storage_root: str | Path, config: dict, documents, *, source_root: Path,
                     shard_size: int = 32, measurements: dict | None = None) -> str:
    """Persist an iterable of ID-keyed native outputs in bounded document batches.

    All expected documents must be present before completion. Partial files are
    unpublished; inference resume and failure records are a later pipeline step.
    """
    validate_embedding_config(config)
    require(type(shard_size) is int and shard_size > 0, "shard_size must be positive")
    started = time.perf_counter()
    images = {row["document_id"]: row for row in config["documents"]}
    seen = set()
    records = []
    with ArtifactWriter(storage_root, "embeddings", config) as writer:
        batch = []
        shard = 0

        def flush():
            offsets = {name: 0 for name in config["tensors"]}
            for doc in batch:
                row = {**images[doc["document_id"]], "shard": shard,
                       "token_metadata": json.dumps(doc["token_metadata"], sort_keys=True, allow_nan=False)}
                for name, array in doc["tensors"].items():
                    row[f"{name}__offset"] = offsets[name]
                    row[f"{name}__length"] = len(array)
                    offsets[name] += len(array)
                records.append(row)
            for name in config["tensors"]:
                # Without dtype, concatenate normalizes non-native byte order.
                writer.array(f"shard-{shard:05d}-{name}", np.concatenate(
                    [doc["tensors"][name] for doc in batch], axis=0,
                    dtype=np.dtype(config["tensors"][name]["dtype"]), casting="no"))
            batch.clear()

        for document in documents:
            _validate_document(document, config)
            doc_id = document["document_id"]
            require(doc_id in images and doc_id not in seen, "Duplicate or unexpected embedding ID")
            seen.add(doc_id)
            # Inference iterators may reuse output buffers for subsequent batches.
            batch.append({"document_id": int(doc_id), "token_metadata": json.loads(json.dumps(document["token_metadata"])),
                          "tensors": {name: value.copy() for name, value in document["tensors"].items()}})
            if len(batch) == shard_size:
                flush()
                shard += 1
        if batch:
            flush()
        require(seen == set(images), "Missing embedding document IDs")
        writer.table("documents", pd.DataFrame(records).sort_values("document_id"))
        writer.complete(capture_provenance(source_root, started, measurements), validator=_decode)
        return writer.artifact_id


def read_embeddings(storage_root: str | Path, config: dict) -> dict[int, dict]:
    """Return native tensors keyed by document ID, after full validation."""
    validate_embedding_config(config)
    artifact_id, _ = artifact_identity("embeddings", config)
    return _decode(read_artifact(storage_root, artifact_id, expected_config=config))
