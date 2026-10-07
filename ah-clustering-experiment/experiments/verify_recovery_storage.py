"""Probe locking, interrupted writes and restart on the selected storage backend.

This tests filesystem operations, not model inference or runtime-loss durability.
Run on the actual mount before extraction and retain the JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from src.dataset import _atomic_bytes, _atomic_json, owned_path, validate_storage_root
from src.provenance import capture_provenance
from src.recovery import RecoveryJournal


def child(root: Path, key: str, mode: str):
    with RecoveryJournal(root, key, {"probe": 1}) as journal:
        if mode == "hold":
            journal.put("completed", b"native checkpoint fixture")
            journal.put("ready", b"ready")
            while True:
                time.sleep(0.1)
        else:
            # Reproduce death after the data replacement but before its receipt.
            _atomic_bytes(journal.directory / "interrupted.bin", b"unfinished")
            os._exit(74)


def verify(root: Path, report: Path):
    started = time.perf_counter()
    key = "storage-probe-" + uuid.uuid4().hex
    process = subprocess.Popen([sys.executable, __file__, "--storage-root", str(root),
                                "--child", "hold", "--key", key])

    def kill_child():
        # Windows venv python.exe can be a launcher with an interpreter child.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           check=True, capture_output=True)
        else:
            process.kill()
        process.wait(timeout=10)
    ready = owned_path(root, f"outputs/checkpoints/{key}/ready.json")
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Storage probe child failed to acquire and write before deadline")
            time.sleep(0.05)
        try:
            with RecoveryJournal(root, key, {"probe": 1}):
                raise AssertionError("Second writer acquired a live journal lock")
        except ValueError as error:
            if "already has a writer" not in str(error):
                raise
        kill_child()
    finally:
        if process.poll() is None:
            kill_child()
    with RecoveryJournal(root, key, {"probe": 1}) as journal:
        assert journal.get("completed") == b"native checkpoint fixture"
    interrupted = subprocess.run([sys.executable, __file__, "--storage-root", str(root),
                                  "--child", "interrupt", "--key", key], timeout=30)
    assert interrupted.returncode == 74
    with RecoveryJournal(root, key, {"probe": 1}) as journal:
        assert journal.get("interrupted") is None
        assert journal.get("completed") == b"native checkpoint fixture"
        journal.put("interrupted", b"repaired")
        assert journal.get("interrupted") == b"repaired"
        # Negative control: retain the valid receipt while changing its payload.
        _atomic_bytes(journal.directory / "completed.bin", b"deliberate corruption")
        assert journal.get("completed") is None
    _atomic_json(report, {"status": "passed", "storage_root": str(root), "probe_key": key,
                 "checks": ["live writer exclusion", "lock released after kill", "completed payload survives kill",
                            "unfinished payload rejected", "retry succeeds", "corrupt payload rejected"],
                 "limits": "One host/process restart only. Does not prove remount, power-loss, or cross-host locking.",
                 "provenance": capture_provenance(ROOT, started)})
    print(json.dumps({"status": "passed", "report": str(report)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--child", choices=["hold", "interrupt"], help=argparse.SUPPRESS)
    parser.add_argument("--key", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = validate_storage_root(args.storage_root)
    if args.child:
        child(root, args.key, args.child)
    else:
        report = args.report or owned_path(root, "outputs/logs/step5-storage-probe.json")
        if not report.resolve().is_relative_to(root):
            parser.error("Report must remain inside the storage root")
        verify(root, report)


if __name__ == "__main__":
    main()
