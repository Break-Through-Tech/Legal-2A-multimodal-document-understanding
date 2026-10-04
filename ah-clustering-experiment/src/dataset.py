"""Prepare the pinned benchmark without changing shared labels or partitions.

The manifest contains image references and evaluation metadata only. Reference
transcriptions, extraction targets and schemas stay in the upstream cache.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import numbers
import os
from pathlib import Path
import platform
import subprocess
import uuid
from datetime import datetime, timezone

import pandas as pd
from PIL import Image


DATASET_NAME = "getomni-ai/ocr-benchmark"
DATASET_REVISION = "4ed0d95271ca00107726230f7a0944ed9e90d897"
PARTITIONS = {"train": 700, "val": 150, "test": 150}
REQUIRED_COLUMNS = {"id", "split", "label", "original_label", "document_quality"}
SCHEMA_VERSION = 1
LIMITATIONS = [
    "The test partition has previously been explored; it is not an untouched holdout.",
    "Unknown merges rare formats; it is neither clustering noise nor evidence of unseen-class detection.",
    "The benchmark includes business documents beyond the legal subset.",
    "Shared partitions do not group source or template families.",
]


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def preparation_provenance() -> dict:
    source_root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), source_root / "experiments/prepare_dataset.py"]
    digests = {path.relative_to(source_root).as_posix(): file_sha256(path) for path in files}
    result = {"source_files_sha256": digests, "source_digest": fingerprint(digests), "git_commit": None, "experiment_dirty": None}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, stderr=subprocess.DEVNULL, text=True).strip()
        result["experiment_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=source_root, stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass  # An exported source directory may have no Git checkout.
    return result


def validate_storage_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.name != "ah-clustering-experiment":
        raise ValueError("Storage root must be a directory named ah-clustering-experiment")
    return root


def owned_path(root: Path, relative: str) -> Path:
    """Reject output paths redirected outside the experiment by traversal or links."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes experiment root: {relative}")
    return path


def configure_cache(root: Path) -> None:
    for name, relative in {
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_DATASETS_CACHE": "cache/datasets",
        "XDG_CACHE_HOME": "cache",
        "TMPDIR": "cache/tmp",
        "TMP": "cache/tmp",
        "TEMP": "cache/tmp",
    }.items():
        path = owned_path(root, relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def validate_assignments(
    frame: pd.DataFrame,
    expected_count: int = 1000,
    expected_labels: int = 19,
    expected_partitions: dict | None = None,
) -> pd.DataFrame:
    expected_partitions = PARTITIONS if expected_partitions is None else expected_partitions
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing shared manifest columns: {sorted(missing)}")
    result = frame.loc[:, ["id", "split", "label", "original_label", "document_quality"]].copy()
    if result.isna().any().any():
        raise ValueError("Shared manifest contains missing values")
    if not all(isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0 for value in result["id"]):
        raise ValueError("Document IDs must be nonnegative integers")
    if result["id"].duplicated().any():
        raise ValueError("Duplicate document IDs or overlapping partitions")
    if len(result) != expected_count:
        raise ValueError(f"Expected {expected_count} documents, received {len(result)}")
    for column in ("split", "label", "original_label", "document_quality"):
        if not all(isinstance(value, str) and value and value == value.strip() for value in result[column]):
            raise ValueError(f"Invalid or unnormalized {column}")
    if result["split"].value_counts().to_dict() != expected_partitions:
        raise ValueError(f"Partition counts must be {expected_partitions}")
    if result["label"].nunique() != expected_labels:
        raise ValueError(f"Expected {expected_labels} final labels")
    for name in expected_partitions:
        if result.loc[result["split"] == name, "label"].nunique() != expected_labels:
            raise ValueError(f"Partition {name} does not contain all final labels")
    return result.sort_values("id").reset_index(drop=True)


def join_source_records(assignments: pd.DataFrame, source_records) -> list[dict]:
    """Join by ID, verifying the original source metadata against the shared CSV."""
    if assignments["id"].duplicated().any():
        raise ValueError("Duplicate assignment IDs")
    by_id = assignments.set_index("id").to_dict(orient="index")
    seen = set()
    result = []
    for record in source_records:
        doc_id = record.get("id")
        if not isinstance(doc_id, numbers.Integral) or isinstance(doc_id, bool):
            raise ValueError("Source document ID must be an integer")
        if doc_id in seen:
            raise ValueError(f"Duplicate source document ID: {doc_id}")
        if doc_id not in by_id:
            raise ValueError(f"Unexpected source document ID: {doc_id}")
        seen.add(doc_id)
        assignment = by_id[doc_id]
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing source metadata for {doc_id}")
        for source_key, column in (("format", "original_label"), ("documentQuality", "document_quality")):
            value = metadata.get(source_key)
            if not isinstance(value, str) or value.strip().upper() != assignment[column]:
                raise ValueError(f"Source {source_key} disagrees with shared {column} for {doc_id}")
        if "image" not in record:
            raise ValueError(f"Missing image for {doc_id}")
        result.append({"document_id": int(doc_id), **assignment, "image": record["image"]})
    if seen != set(by_id):
        raise ValueError(f"Missing source documents: {sorted(set(by_id) - seen)[:10]}")
    return sorted(result, key=lambda row: row["document_id"])


def select_cohort(manifest: pd.DataFrame, cohort: str = "train") -> list[int]:
    if cohort not in {*PARTITIONS, "all"}:
        raise ValueError("Cohort must be train, val, test, or all")
    if manifest["document_id"].duplicated().any():
        raise ValueError("Duplicate document IDs in manifest")
    if not manifest["split"].isin(PARTITIONS).all():
        raise ValueError("Invalid partition in manifest")
    selected = manifest if cohort == "all" else manifest.loc[manifest["split"] == cohort]
    return sorted(int(value) for value in selected["document_id"])


def validate_config(config: dict) -> None:
    if config.get("dataset") != {"name": DATASET_NAME, "revision": DATASET_REVISION, "upstream_split": "test"}:
        raise ValueError("Dataset configuration must use the documented pinned benchmark revision and upstream test split")
    if config.get("default_cohort", "train") not in {*PARTITIONS, "all"}:
        raise ValueError("Invalid default_cohort")
    if config.get("label_usage") != "evaluation_only":
        raise ValueError("Labels must be evaluation_only; cluster count disclosure is recorded separately")
    policy = config.get("cluster_count_policy", {})
    if policy.get("mode") not in {"class_count_informed", "independently_chosen"}:
        raise ValueError("Declare class_count_informed or independently_chosen cluster_count_policy")
    count = policy.get("n_clusters")
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 2):
        raise ValueError("n_clusters must be null or an integer of at least two")
    if policy["mode"] == "class_count_informed" and count != 19:
        raise ValueError("A class-count-informed baseline must declare the 19 known final classes")
    if not isinstance(policy.get("rationale"), str) or not policy["rationale"].strip():
        raise ValueError("Cluster count policy needs a rationale")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep sibling names short enough for typical Windows persistent roots.
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _image_info(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        width, height, image_format = image.width, image.height, image.format
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
    if width <= 0 or height <= 0 or not image_format:
        raise ValueError("Invalid image dimensions or format")
    return width, height, image_format


def _image_bytes(encoded_image: dict, root: Path) -> bytes:
    if not isinstance(encoded_image, dict):
        raise ValueError("Source images must be encoded dictionaries (datasets.Image(decode=False))")
    payload = encoded_image.get("bytes")
    if payload is not None:
        return bytes(payload)
    source_path = encoded_image.get("path")
    if not source_path:
        raise ValueError("Source image has neither bytes nor path")
    path = Path(source_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source image file must be inside the experiment storage root")
    return path.read_bytes()


def _load_source(root: Path):
    # All HF imports happen after configure_cache; preserve original encoded bytes.
    from datasets import Image as DatasetImage, load_dataset

    # datasets embeds cache_dir in one Windows lock filename. A relative cache
    # path avoids doubling the long repository path in that filename.
    from contextlib import chdir

    with chdir(root):
        source = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="test", cache_dir="cache/datasets")
    return source.select_columns(["id", "image", "metadata"]).cast_column("image", DatasetImage(decode=False))


def validate_artifact(root: Path, manifest_path: Path, metadata_path: Path, identity: dict, assignments: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    if metadata is None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("identity") != identity:
        raise ValueError("Incomplete or incompatible dataset artifact")
    if metadata.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Manifest checksum mismatch")
    manifest = pd.read_parquet(manifest_path)
    required = {"document_id", "split", "label", "original_label", "document_quality", "image_path", "image_sha256", "image_width", "image_height", "image_format", "dataset_revision", "shared_csv_sha256"}
    if set(manifest.columns) != required:
        raise ValueError("Unexpected manifest columns")
    actual = validate_assignments(manifest.rename(columns={"document_id": "id"}))
    if not actual.equals(assignments):
        raise ValueError("Cached manifest no longer matches shared assignments")
    expected_summary = {
        "dataset_id": "dataset-" + fingerprint(identity)[:20],
        "identity_sha256": fingerprint(identity),
        "document_count": len(manifest),
        "partition_counts": manifest["split"].value_counts().to_dict(),
        "final_label_count": manifest["label"].nunique(),
        "extraction_scope": "all_1000_documents",
        "label_usage": "evaluation_only",
    }
    for field, expected in expected_summary.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Cached dataset metadata disagrees with manifest: {field}")
    if not (manifest["dataset_revision"] == DATASET_REVISION).all() or not (manifest["shared_csv_sha256"] == identity["shared_csv_sha256"]).all():
        raise ValueError("Cached manifest provenance mismatch")
    for row in manifest.to_dict(orient="records"):
        relative = row["image_path"]
        if Path(relative).is_absolute() or "\\" in relative:
            raise ValueError("Image paths must be portable relative POSIX paths")
        path = owned_path(root, relative)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch for {row['document_id']}")
        if _image_info(payload) != (row["image_width"], row["image_height"], row["image_format"]):
            raise ValueError(f"Image metadata mismatch for {row['document_id']}")
    return manifest


def prepare_dataset(config: dict, storage_root: str | Path, assignments_path: str | Path, source_records=None, cohort: str | None = None) -> dict:
    """Publish a validated all-document manifest and separately identified cohort.

    source_records is an injectable iterable for independent tests; production
    loads the immutable upstream revision. Concurrent writes are not supported.
    """
    validate_config(config)
    root = validate_storage_root(storage_root)
    configure_cache(root)
    chosen_cohort = cohort or config.get("default_cohort", "train")
    if chosen_cohort not in {*PARTITIONS, "all"}:
        raise ValueError("Cohort must be train, val, test, or all")
    csv_path = Path(assignments_path).resolve()
    csv_bytes = csv_path.read_bytes()
    assignments = validate_assignments(pd.read_csv(io.BytesIO(csv_bytes)))
    identity = {"schema_version": SCHEMA_VERSION, "dataset": config["dataset"], "shared_csv_sha256": hashlib.sha256(csv_bytes).hexdigest()}
    dataset_id = "dataset-" + fingerprint(identity)[:20]
    artifact_dir = owned_path(root, f"outputs/metadata/{dataset_id}")
    manifest_path = artifact_dir / "manifest.parquet"
    metadata_path = artifact_dir / "dataset.json"
    reused = False
    try:
        manifest = validate_artifact(root, manifest_path, metadata_path, identity, assignments)
        reused = True
    except (OSError, ValueError, KeyError, TypeError):
        # A stale completion marker cannot represent the repairing artifact.
        metadata_path.unlink(missing_ok=True)
        source = _load_source(root) if source_records is None else source_records
        records = join_source_records(assignments, source)
        rows = []
        for record in records:
            payload = _image_bytes(record.pop("image"), root)
            width, height, image_format = _image_info(payload)
            digest = hashlib.sha256(payload).hexdigest()
            extension = {"JPEG": "jpg", "TIFF": "tif"}.get(image_format, image_format.lower())
            relative = f"outputs/images/{dataset_id}/{record['document_id']:04d}.{extension}"
            path = owned_path(root, relative)
            if not path.is_file() or file_sha256(path) != digest:
                _atomic_bytes(path, payload)
            rows.append({**record, "image_path": relative, "image_sha256": digest, "image_width": width, "image_height": height, "image_format": image_format, "dataset_revision": DATASET_REVISION, "shared_csv_sha256": identity["shared_csv_sha256"]})
        manifest = pd.DataFrame(rows)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        parquet_buffer = io.BytesIO()
        manifest.to_parquet(parquet_buffer, index=False)
        _atomic_bytes(manifest_path, parquet_buffer.getvalue())
        metadata = {
            "status": "complete", "identity": identity, "dataset_id": dataset_id,
            "identity_sha256": fingerprint(identity),
            "manifest_sha256": file_sha256(manifest_path), "document_count": len(manifest),
            "partition_counts": manifest["split"].value_counts().to_dict(), "final_label_count": manifest["label"].nunique(),
            "extraction_scope": "all_1000_documents", "reference_text_policy": "Kept only in raw source cache; excluded from manifest and model inputs; not actual OCR.",
            "label_usage": "evaluation_only", "limitations": LIMITATIONS,
            "created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
            "preparation_provenance": preparation_provenance(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("datasets", "pandas", "pyarrow", "Pillow")},
        }
        # The marker is the last dataset write, after a full read-back. Readers
        # never discover an artifact marked complete while validation is pending.
        manifest = validate_artifact(root, manifest_path, metadata_path, identity, assignments, metadata=metadata)
        _atomic_json(metadata_path, metadata)
    cohort_identity = {
        "dataset_id": dataset_id, "cohort": chosen_cohort,
        "document_ids": select_cohort(manifest, chosen_cohort),
        "designation": "exploratory" if chosen_cohort != "train" else "initial_training_partition",
        "label_usage": config["label_usage"], "cluster_count_policy": config["cluster_count_policy"],
    }
    cohort_id = "cohort-" + fingerprint(cohort_identity)[:20]
    cohort_path = owned_path(root, f"outputs/metadata/{dataset_id}/cohorts/{cohort_id}/cohort.json")
    _atomic_json(cohort_path, {"cohort_id": cohort_id, "identity_sha256": fingerprint(cohort_identity), **cohort_identity})
    return {"dataset_id": dataset_id, "cohort_id": cohort_id, "reused": reused, "manifest": str(manifest_path), "metadata": str(metadata_path), "cohort": str(cohort_path), "document_count": len(manifest), "cohort_count": len(cohort_identity["document_ids"])}
