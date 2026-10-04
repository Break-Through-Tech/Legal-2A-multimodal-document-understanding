"""Content identities for uncommitted code and measured execution context."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import time

from .dataset import file_sha256, fingerprint


def source_identity(source_root: Path) -> dict:
    """Hash runnable source, tests, configs and dependency declarations, including untracked files.

    Generated outputs, caches, environments and reports never enter the digest.
    Git state is descriptive; the content digest identifies the actual source.
    """
    root = Path(source_root).resolve()
    paths = set()
    for directory in ("src", "experiments", "configs", "tests", "dashboard"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".toml", ".yaml", ".yml"}:
                paths.add(path)
    for name in ("requirements.txt", "pyproject.toml", "uv.lock", "poetry.lock", ".gitignore"):
        if (root / name).is_file():
            paths.add(root / name)
    hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}
    result = {"source_files_sha256": hashes, "source_digest": fingerprint(hashes),
              "git_commit": None, "repository_dirty": None, "experiment_dirty": None,
              "git_status": "unavailable"}
    try:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL, timeout=10).strip()
        result.update(git_commit=git("rev-parse", "HEAD"),
                      repository_dirty=bool(git("status", "--porcelain", "--untracked-files=all")),
                      experiment_dirty=bool(git("status", "--porcelain", "--untracked-files=all", "--", ".")),
                      git_status="available")
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def capture_provenance(source_root: Path, started: float, measurements: dict | None = None) -> dict:
    """Record installed versions and available hardware; never load a model or require CUDA.

    started is a time.perf_counter() value. Callers can supply peak memory,
    throughput and device measurements collected during the operation.
    """
    dependencies = {dist.metadata["Name"]: dist.version
                    for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    hardware = {"machine": platform.machine(), "processor": platform.processor() or None,
                "gpu_query": "unavailable", "gpus": []}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        hardware.update(gpu_query="available", gpus=[line.strip() for line in output.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        pass
    result = {"source": source_identity(source_root), "dependencies": dict(sorted(dependencies.items())),
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "finished_at": datetime.now(timezone.utc).isoformat(),
                          "elapsed_seconds": time.perf_counter() - started},
              "hardware": hardware, "measurements": measurements or {}}
    fingerprint(result)  # Refuse non-JSON or non-finite measurements.
    return result
