"""Versioned identities shared by extraction, clustering and result readers."""

from __future__ import annotations

import copy
import numbers
from pathlib import PurePosixPath
import re

import numpy as np

from .dataset import fingerprint

SCHEMA_VERSION = 1
OUTPUT_KINDS = ("embeddings", "ocr", "similarities", "clusters", "metrics", "projections", "metadata", "logs")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def document_ids(values) -> list[int]:
    result = list(values)
    require(bool(result), "Document IDs must not be empty")
    require(all(isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_)) and v >= 0 for v in result),
            "Document IDs must be nonnegative integers")
    require(len(result) == len(set(result)), "Duplicate document IDs")
    return [int(v) for v in result]


def portable_path(value: str) -> None:
    require(isinstance(value, str) and bool(value) and "\\" not in value and ":" not in value,
            "Paths must be relative POSIX paths")
    require(not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts,
            "Paths must stay inside the artifact")


def numeric_array(array: np.ndarray) -> None:
    require(isinstance(array, np.ndarray) and array.ndim >= 1, "Expected a numeric array with at least one axis")
    require(array.dtype.kind in "bifu" and np.isfinite(array).all(), "Arrays must contain finite numeric or boolean values")


def embedding_config(*, model: dict, processor: dict, dataset: dict, documents: list[dict],
                     input_mode: str, transformations: dict, tensors: dict,
                     token_policy: dict, source_digest: str, ocr: dict | None = None) -> dict:
    """Create a label-free identity. Documents contain image identity, never reference text.

    OCR identity must cover actual words, boxes, engine configuration and status.
    Image-only modes deliberately ignore any supplied OCR identity.
    """
    config = {"schema_version": SCHEMA_VERSION, "model": model, "processor": processor,
              "dataset": dataset, "documents": sorted(documents, key=lambda row: row["document_id"]),
              "input_mode": input_mode, "transformations": transformations, "tensors": tensors,
              "token_policy": token_policy, "source_digest": source_digest,
              "ocr": ocr if input_mode == "image_and_ocr" else None}
    validate_embedding_config(config)
    return copy.deepcopy(config)


def validate_embedding_config(config: dict) -> None:
    require(config.get("schema_version") == SCHEMA_VERSION, "Unsupported embedding schema")
    model = config["model"]
    require(isinstance(model.get("checkpoint"), str) and bool(model["checkpoint"].strip()), "Missing checkpoint")
    require(isinstance(model.get("revision"), str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model["revision"]) is not None,
            "Model revision must be an immutable commit or content SHA")
    for name in ("processor", "transformations", "token_policy"):
        require(isinstance(config[name], dict) and bool(config[name]), f"Missing {name}")
    require(sha256(config["source_digest"]), "Missing source digest")
    dataset = config["dataset"]
    require(bool(dataset.get("dataset_id")) and sha256(dataset.get("identity_sha256")) and sha256(dataset.get("manifest_sha256")),
            "Dataset identity and manifest checksum are required")
    documents = config["documents"]
    document_ids(row["document_id"] for row in documents)
    require(documents == sorted(documents, key=lambda r: r["document_id"]), "Document identities must be sorted by ID")
    for row in documents:
        require(set(row) == {"document_id", "image_path", "image_sha256"}, "Use only ID and image identity in model inputs")
        portable_path(row["image_path"])
        require(sha256(row["image_sha256"]), "Missing image checksum")
    require(config["input_mode"] in {"image_only", "image_and_ocr"}, "Unknown input mode")
    if config["input_mode"] == "image_and_ocr":
        ocr = config["ocr"]
        require(isinstance(ocr, dict) and bool(ocr.get("artifact_id")) and sha256(ocr.get("content_sha256")),
                "Actual OCR content identity is required")
        require(bool(ocr.get("engine")) and bool(ocr.get("engine_version")) and isinstance(ocr.get("config"), dict),
                "OCR engine, version and configuration are required")
        require(bool(config["transformations"].get("word_coordinates")), "Declare word coordinate transformations")
    else:
        require(config["ocr"] is None, "Image-only embeddings must not depend on OCR")
    require(bool(config["transformations"].get("image")), "Declare image transformations")
    tensors = config["tensors"]
    require(isinstance(tensors, dict) and bool(tensors), "Output tensor schema is required")
    for name, tensor in tensors.items():
        require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, "Invalid tensor name")
        require(set(tensor) == {"dtype", "tail_shape", "role", "alignment"}, "Unexpected tensor schema")
        require(np.dtype(tensor["dtype"]).kind in "bifu", "Unsupported numeric dtype")
        require(isinstance(tensor["tail_shape"], list) and all(type(v) is int and v > 0 for v in tensor["tail_shape"]),
                "Tensor tail_shape must contain positive integer dimensions")
        require(tensor["role"] in {"embedding", "mask", "token_ids", "coordinates"}, "Unknown tensor role")
        require(isinstance(tensor["alignment"], str) and bool(tensor["alignment"]), "Declare output alignment group")
        if tensor["role"] == "mask":
            require(np.dtype(tensor["dtype"]).kind == "b" and tensor["tail_shape"] == [], "Masks must be boolean vectors")
        if tensor["role"] == "token_ids":
            require(np.dtype(tensor["dtype"]).kind in "iu", "Token IDs must be integers")
    fingerprint(config)


def clustering_config(*, embedding_artifact: str, embedding_identity: str, model: dict,
                      representation: str, token_selection: dict, pooling: dict,
                      normalization: dict, similarity: dict | None, algorithm: str,
                      parameters: dict, cohort: dict, seed: int,
                      cluster_count_policy: dict, source_digest: str) -> dict:
    """Identify a run before fitting. Labels are excluded; count selection is disclosed."""
    require(bool(embedding_artifact) and sha256(embedding_identity), "Missing embedding artifact identity")
    require(bool(model.get("checkpoint")) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model.get("revision", "")) is not None,
            "Missing immutable checkpoint identity")
    require(bool(representation) and bool(algorithm), "Representation and algorithm are required")
    require(all(isinstance(v, dict) and v for v in (token_selection, pooling, normalization)), "Declare selection, pooling and normalization (including none)")
    require(isinstance(parameters, dict) and type(seed) is int, "Parameters and an integer seed are required")
    require(sha256(source_digest), "Missing source digest")
    ids = document_ids(cohort["document_ids"])
    require(cohort.get("designation") in {"initial_training_partition", "exploratory"}, "Declare cohort designation")
    require(cohort.get("cohort") in {"train", "val", "test", "all"} and bool(cohort.get("dataset_id")), "Missing cohort identity")
    require(cohort["cohort"] == "train" or cohort["designation"] == "exploratory", "Held-out cohorts must be exploratory")
    require(cluster_count_policy.get("mode") in {"independently_chosen", "class_count_informed"}
            and bool(cluster_count_policy.get("rationale")), "Declare cluster count selection")
    count = cluster_count_policy.get("n_clusters")
    require(count is None or (type(count) is int and count >= 2), "Cluster count must be null or an integer of at least two")
    require("n_clusters" not in parameters or parameters["n_clusters"] == count, "Algorithm and declared cluster count disagree")
    require(cluster_count_policy["mode"] != "class_count_informed" or count is not None, "Class-count-informed runs must declare a count")
    if similarity is not None:
        require(bool(similarity.get("artifact_id")) and sha256(similarity.get("identity_sha256"))
                and bool(similarity.get("definition")), "Similarity needs an artifact identity and definition")
    config = {"schema_version": SCHEMA_VERSION, "embedding_artifact": embedding_artifact,
              "embedding_identity": embedding_identity, "model": model, "representation": representation,
              "token_selection": token_selection, "pooling": pooling, "normalization": normalization,
              "similarity": similarity, "algorithm": algorithm, "parameters": parameters,
              "cohort": {**cohort, "document_ids": sorted(ids)}, "seed": seed,
              "cluster_count_policy": cluster_count_policy, "source_digest": source_digest}
    fingerprint(config)
    return copy.deepcopy(config)
