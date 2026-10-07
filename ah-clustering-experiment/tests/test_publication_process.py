"""Exercise single-writer exclusion and incomplete publication in real processes.

Oracle: a second live writer cannot acquire the first writer's artifact; killing
that writer before completion must not make the partial payload a readable cache.
No storage API or process boundary is mocked. Inference resume is not tested.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import ArtifactWriter, artifact_identity, read_artifact
from src.provenance import capture_provenance

CONFIG = {"scenario": "actual-process-interruption", "schema_version": 1}


def hold_partial_writer(root):
    with ArtifactWriter(root, "similarities", CONFIG) as writer:
        writer.array("matrix", np.array([[1., .5], [.5, 1.]], dtype="float32"))
        (root / "ready").write_text(writer.artifact_id, encoding="utf-8")
        # The parent deliberately terminates this test-only process.
        while True:
            time.sleep(.1)


class ProcessPublicationContract(unittest.TestCase):
    def test_second_process_exclusion_and_killed_writer_remains_incomplete(self):
        scratch = ROOT / "cache" / "process-publication"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary) / "ah-clustering-experiment"
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--hold-partial", str(root)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                deadline = time.monotonic() + 15
                while not (root / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue((root / "ready").exists(), "Child did not reach the partial write boundary")
                artifact_id, _ = artifact_identity("similarities", CONFIG)
                with self.assertRaisesRegex(ValueError, "writer lock"):
                    with ArtifactWriter(root, "similarities", CONFIG):
                        self.fail("Two processes acquired the same artifact")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    read_artifact(root, artifact_id)
                process.kill()
                process.wait(timeout=10)
                marker = root / "outputs" / "metadata" / artifact_id / "artifact.json"
                self.assertFalse(marker.exists())
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    read_artifact(root, artifact_id)
                lock = marker.with_name("writer.lock")
                self.assertTrue(lock.is_file())
                # The process has exited: this follows the documented stale-lock procedure.
                lock.unlink()
                started = time.perf_counter()
                with ArtifactWriter(root, "similarities", CONFIG) as writer:
                    writer.array("matrix", np.array([[1., .5], [.5, 1.]], dtype="float32"))
                    writer.complete(capture_provenance(ROOT, started))
                restored = read_artifact(root, artifact_id)
                np.testing.assert_array_equal(restored["arrays"]["matrix"], [[1., .5], [.5, 1.]])
                del restored  # Release mmap before Windows temporary directory cleanup.
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--hold-partial":
        hold_partial_writer(Path(sys.argv[2]))
    else:
        unittest.main()
