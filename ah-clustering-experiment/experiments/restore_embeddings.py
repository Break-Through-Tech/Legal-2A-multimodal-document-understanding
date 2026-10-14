"""Restore published native embeddings without loading encoders or running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "saved-embeddings"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(model, entry, root, snapshots=SNAPSHOTS):
    if "parts" not in entry:
        return restore_archive(model, entry, root, snapshots / entry["archive"])
    with tempfile.TemporaryDirectory(prefix=".assemble-embeddings-", dir=root) as assembly:
        archive_path = Path(assembly) / "archive.zip"
        with archive_path.open("wb") as target:
            for part in entry["parts"]:
                name = part["name"]
                if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
                    raise ValueError(f"Invalid archive part name: {name}")
                path = snapshots / name
                if not path.is_file() or path.stat().st_size != part["bytes"]:
                    raise ValueError(f"Missing full archive part {name}. Run git lfs pull first.")
                if sha256(path) != part["sha256"]:
                    raise ValueError(f"Archive part checksum differs: {name}")
                with path.open("rb") as source:
                    shutil.copyfileobj(source, target)
        return restore_archive(model, entry, root, archive_path)


def restore_archive(model, entry, root, archive_path):
    if not archive_path.is_file() or archive_path.stat().st_size != entry["archive_bytes"]:
        raise ValueError(f"Missing full archive for {model}. Run git lfs pull first.")
    if sha256(archive_path) != entry["archive_sha256"]:
        raise ValueError(f"Archive checksum differs for {model}. Download it again with Git LFS.")
    files = entry["files"]
    destinations = {}
    for name, expected in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or ":" in name or not name.startswith("outputs/")):
            raise ValueError(f"Invalid snapshot path: {name}")
        destination = (root / name).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Snapshot path escapes storage root: {name}")
        if destination.exists():
            if (not destination.is_file() or destination.stat().st_size != expected["bytes"]
                    or sha256(destination) != expected["sha256"]):
                raise ValueError(f"Existing file differs; left untouched: {destination}")
        destinations[name] = destination

    restored = 0
    # Validate all expanded bytes before publishing any file from this model.
    with tempfile.TemporaryDirectory(prefix=".restore-embeddings-", dir=root) as staging:
        stage = Path(staging)
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(files) or set(archive.namelist()) != set(files):
                raise ValueError(f"Archive members differ for {model}")
            for name, expected in files.items():
                if archive.getinfo(name).file_size != expected["bytes"]:
                    raise ValueError(f"Expanded size differs: {name}")
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                if sha256(target) != expected["sha256"]:
                    raise ValueError(f"Expanded checksum differs: {name}")
        # Completion manifests are published after their payloads.
        for name in sorted(files, key=lambda value: (value.endswith("/artifact.json"), value)):
            destination = destinations[name]
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / name).replace(destination)
            restored += 1
    return {"model": model, "artifact_id": entry["artifact_id"],
            "documents": entry["document_count"], "restored_files": restored,
            "reused_files": len(files) - restored}


def main():
    index = json.loads((SNAPSHOTS / "index.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(index["models"]),
                        help="Restore only these models; default: all five")
    parser.add_argument("--storage-root", type=Path, default=ROOT,
                        help="Destination directory, named ah-clustering-experiment")
    args = parser.parse_args()
    root = args.storage_root.resolve()
    if root.name != "ah-clustering-experiment":
        parser.error("Storage root must be named ah-clustering-experiment")
    root.mkdir(parents=True, exist_ok=True)
    try:
        for model in args.models or index["models"]:
            print(json.dumps(restore(model, index["models"][model], root)), flush=True)
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f"Embedding restore failed: {error}\n")


if __name__ == "__main__":
    main()
