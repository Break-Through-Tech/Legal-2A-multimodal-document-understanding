"""Checksummed numeric/Parquet bundles, published only after read-back validation.

One writer owns each identity. A failed write has no completion manifest and is
never a cache hit. The extraction pipeline owns inference restart checkpoints.
"""

from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

from .contracts import OUTPUT_KINDS, SCHEMA_VERSION, numeric_array, portable_path, require
from .dataset import _atomic_bytes, _atomic_json, file_sha256, fingerprint, owned_path, validate_storage_root


def initialize_storage(storage_root: str | Path) -> Path:
    root = validate_storage_root(storage_root)
    for kind in OUTPUT_KINDS:
        owned_path(root, f"outputs/{kind}").mkdir(parents=True, exist_ok=True)
    return root


def artifact_identity(kind: str, config: dict) -> tuple[str, str]:
    require(kind in OUTPUT_KINDS and kind not in {"metadata", "logs"}, "Unsupported artifact kind")
    require(isinstance(config, dict) and bool(config), "Configuration must be a nonempty JSON object")
    digest = fingerprint({"schema_version": SCHEMA_VERSION, "kind": kind, "config": config})
    return f"{kind}-{digest[:20]}", digest


def _metadata_path(root: Path, artifact_id: str) -> Path:
    require(isinstance(artifact_id, str) and re.fullmatch(r"[a-z]+-[0-9a-f]{20}", artifact_id) is not None,
            "Invalid artifact ID")
    return owned_path(root, f"outputs/metadata/{artifact_id}/artifact.json")


def _name(name: str) -> None:
    require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_-]*", name) is not None, "Invalid payload name")


def _validate_provenance(provenance: dict) -> None:
    require(isinstance(provenance, dict) and all(k in provenance for k in ("source", "dependencies", "runtime", "hardware", "measurements")),
            "Source, dependency, runtime and hardware provenance are required")
    source = provenance["source"]
    require(source.get("source_digest") == fingerprint(source.get("source_files_sha256")), "Source digest mismatch")
    require(all(k in source for k in ("git_commit", "repository_dirty", "experiment_dirty", "git_status")), "Git state is required (null if unavailable)")
    runtime = provenance["runtime"]
    require(isinstance(runtime.get("elapsed_seconds"), (int, float)) and runtime["elapsed_seconds"] >= 0,
            "Measured elapsed runtime is required")
    require(bool(provenance["dependencies"]) and bool(provenance["hardware"]), "Missing dependency or hardware observations")
    fingerprint(provenance)


class ArtifactWriter:
    """Context-managed single-writer publication. Existing complete artifacts are immutable.

    Files live in outputs/<kind>/<id>; config and completion live in metadata/<id>.
    Local rename and exclusive-file semantics are assumed, not claimed for cloud mounts.
    """

    def __init__(self, storage_root: str | Path, kind: str, config: dict):
        self.root = initialize_storage(storage_root)
        self.kind = kind
        self.config = copy.deepcopy(config)
        self.artifact_id, self.identity_sha256 = artifact_identity(kind, self.config)
        self.metadata_path = _metadata_path(self.root, self.artifact_id)
        self.lock_path = self.metadata_path.with_name("writer.lock")
        self.files = {}
        self.active = False
        self.completed = False

    def __enter__(self):
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "artifact_id": self.artifact_id, "token": uuid.uuid4().hex}))
        except FileExistsError as error:
            raise ValueError("Artifact already has a writer lock; verify the owner is stopped before removing a stale lock") from error
        self.active = True
        try:
            require(not self.metadata_path.exists(), "Completed artifacts are immutable; read/validate before reuse")
            config_path = self.metadata_path.with_name("config.json")
            if config_path.exists():
                require(json.loads(config_path.read_text(encoding="utf-8")) == self.config, "Artifact ID collision or incompatible partial configuration")
            _atomic_json(config_path, self.config)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.active:
            self.lock_path.unlink(missing_ok=True)
            self.active = False

    def _write(self, name: str, extension: str, payload: bytes, description: dict):
        require(self.active and not self.completed, "Writer is not open")
        _name(name)
        require(name not in self.files, "Duplicate payload name")
        relative = f"outputs/{self.kind}/{self.artifact_id}/{name}.{extension}"
        path = owned_path(self.root, relative)
        _atomic_bytes(path, payload)
        self.files[name] = {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size, **description}

    def array(self, name: str, value: np.ndarray) -> None:
        numeric_array(value)
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        self._write(name, "npy", buffer.getvalue(), {"format": "npy", "dtype": value.dtype.str, "shape": list(value.shape)})

    def table(self, name: str, frame: pd.DataFrame, *, unique_key: str | None = "document_id") -> None:
        require(isinstance(frame, pd.DataFrame) and not frame.columns.duplicated().any(), "Invalid table columns")
        require(all(isinstance(c, str) for c in frame.columns), "Table column names must be strings")
        if unique_key is not None:
            require(unique_key in frame and not frame[unique_key].isna().any() and not frame[unique_key].duplicated().any(),
                    "Table key must be present, non-null and unique")
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self._write(name, "parquet", buffer.getvalue(), {"format": "parquet", "rows": len(frame),
                                                      "columns": list(frame.columns), "unique_key": unique_key})

    def json(self, name: str, value: dict) -> None:
        payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        self._write(name, "json", payload, {"format": "json"})

    def complete(self, provenance: dict, validator=None) -> dict:
        require(self.active and not self.completed and bool(self.files), "No writable payload to complete")
        _validate_provenance(provenance)
        if "source_digest" in self.config:
            require(self.config["source_digest"] == provenance["source"]["source_digest"], "Source changed since configuration was created")
        manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "kind": self.kind,
                    "artifact_id": self.artifact_id, "identity_sha256": self.identity_sha256,
                    "config_sha256": file_sha256(self.metadata_path.with_name("config.json")),
                    "files": self.files, "provenance": provenance}
        manifest["manifest_sha256"] = fingerprint(manifest)
        bundle = _read(self.root, self.artifact_id, self.config, manifest)
        if validator is not None:
            validator(bundle)
        _atomic_json(self.metadata_path, manifest)  # Completion is always the last write.
        self.completed = True
        return copy.deepcopy(manifest)


def _read(root: Path, artifact_id: str, expected_config: dict | None, manifest: dict) -> dict:
    require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("status") == "complete", "Incomplete or unsupported artifact")
    require(manifest.get("manifest_sha256") == fingerprint({k: v for k, v in manifest.items() if k != "manifest_sha256"}), "Manifest checksum mismatch")
    config_path = _metadata_path(root, artifact_id).with_name("config.json")
    require(file_sha256(config_path) == manifest["config_sha256"], "Configuration checksum mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_id, expected_digest = artifact_identity(manifest["kind"], config)
    require(artifact_id == manifest["artifact_id"] == expected_id and manifest["identity_sha256"] == expected_digest,
            "Artifact identity mismatch")
    require(expected_config is None or fingerprint(config) == fingerprint(expected_config), "Incompatible configuration")
    _validate_provenance(manifest["provenance"])
    if "source_digest" in config:
        require(config["source_digest"] == manifest["provenance"]["source"]["source_digest"], "Source identity mismatch")
    result = {"manifest": manifest, "config": config, "arrays": {}, "tables": {}, "json": {}}
    require(isinstance(manifest["files"], dict) and bool(manifest["files"]), "Missing payloads")
    for name, item in manifest["files"].items():
        _name(name)
        portable_path(item["path"])
        extension = {"npy": "npy", "parquet": "parquet", "json": "json"}.get(item["format"])
        require(extension is not None and item["path"] == f"outputs/{manifest['kind']}/{artifact_id}/{name}.{extension}", "Invalid payload path")
        path = owned_path(root, item["path"])
        require(path.stat().st_size == item["bytes"] and file_sha256(path) == item["sha256"], f"Payload checksum mismatch: {name}")
        if item["format"] == "npy":
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            numeric_array(value)
            require(value.dtype.str == item["dtype"] and list(value.shape) == item["shape"], "Array schema mismatch")
            result["arrays"][name] = value
        elif item["format"] == "parquet":
            frame = pd.read_parquet(path)
            require(len(frame) == item["rows"] and list(frame.columns) == item["columns"], "Table schema mismatch")
            key = item["unique_key"]
            require(key is None or (key in frame and not frame[key].isna().any() and not frame[key].duplicated().any()), "Invalid table key")
            result["tables"][name] = frame
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            fingerprint(value)
            result["json"][name] = value
    return result


def read_artifact(storage_root: str | Path, artifact_id: str, *, expected_config: dict | None = None) -> dict:
    """Read a completed artifact, checking every payload before returning any data."""
    root = validate_storage_root(storage_root)
    try:
        manifest = json.loads(_metadata_path(root, artifact_id).read_text(encoding="utf-8"))
        return _read(root, artifact_id, expected_config, manifest)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unreadable, incomplete or incompatible artifact {artifact_id}: {error}") from error
