"""Checksummed restart checkpoints guarded by a process-lifetime filesystem lock.

Locks require a filesystem implementing OS advisory locking and atomic replace.
Validate those operations on a mount before using it for extraction.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from .contracts import require
from .dataset import _atomic_bytes, _atomic_json, file_sha256, fingerprint, owned_path, validate_storage_root


class RecoveryJournal:
    def __init__(self, root: Path, key: str, identity: dict):
        require(re.fullmatch(r"[a-z0-9_-]+", key) is not None, "Invalid recovery key")
        self.root = validate_storage_root(root)
        self.directory = owned_path(self.root, f"outputs/checkpoints/{key}")
        self.identity = fingerprint(identity)
        self.handle = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = owned_path(self.root, (self.directory / "process.lock").relative_to(self.root).as_posix()).open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise ValueError("Recovery journal already has a writer or filesystem locking is unsupported") from error
        self.handle = handle
        try:
            path = self._path("identity", ".json")
            if path.exists():
                require(json.loads(path.read_text()) == {"identity_sha256": self.identity}, "Recovery identity mismatch")
            else:
                _atomic_json(path, {"identity_sha256": self.identity})
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def _path(self, name: str, suffix: str) -> Path:
        require(self.handle is not None, "Recovery journal is not locked")
        require(re.fullmatch(r"[a-z0-9_-]+", name) is not None, "Invalid checkpoint name")
        return owned_path(self.root, (self.directory / (name + suffix)).relative_to(self.root).as_posix())

    def get(self, name: str) -> bytes | None:
        payload = self._path(name, ".bin")
        receipt = self._path(name, ".json")
        try:
            saved = json.loads(receipt.read_text())
            data = payload.read_bytes()
            if saved != {"identity_sha256": self.identity, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()}:
                return None
            return data
        except (OSError, ValueError, TypeError):
            return None

    def put(self, name: str, data: bytes) -> None:
        _atomic_bytes(self._path(name, ".bin"), data)
        _atomic_json(self._path(name, ".json"), {"identity_sha256": self.identity,
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def mark_artifact_writer(self, artifact_id: str) -> None:
        path = self._artifact_lock(artifact_id)
        self.put("artifact-writer", json.dumps({"artifact_id": artifact_id,
                 "lock_sha256": file_sha256(path)}).encode())

    def _artifact_lock(self, artifact_id: str) -> Path:
        require(re.fullmatch(r"[a-z]+-[0-9a-f]{20}", artifact_id) is not None, "Invalid artifact ID")
        return owned_path(self.root, f"outputs/metadata/{artifact_id}/writer.lock")

    def clear_artifact_lock(self, artifact_id: str) -> None:
        path = self._artifact_lock(artifact_id)
        if path.exists():
            saved = self.get("artifact-writer")
            require(saved is not None and json.loads(saved) == {"artifact_id": artifact_id,
                    "lock_sha256": file_sha256(path)},
                    "Unknown artifact writer lock; confirm its owner stopped before manual removal")
            path.unlink()
