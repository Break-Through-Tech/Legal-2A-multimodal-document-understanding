"""Requirement-derived recovery checks; encoder execution is an explicit fixture.

The pipeline, image loading, dataset validation, checkpoints, publication and
readers are real. These tests establish local orchestration/storage behavior,
not pretrained-model fidelity or mounted Drive crash durability.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (DATASET_NAME, DATASET_REVISION, LIMITATIONS, REFERENCE_TEXT_POLICY,
                         file_sha256, fingerprint, preparation_provenance)
from src.embeddings import read_embeddings
from src.embeddings.pipeline import extract_embeddings


SETTINGS = {"family": "dinov3", "checkpoint": "independent/synthetic-encoder", "revision": "a" * 40,
            "input_mode": "image_only", "dtype": "float32", "batch_size": 2, "processor_options": {}}
# Literal oracle fixed before inspection of the recovery implementation.
GOLDEN = {
    0: [[0.25, -0.5]],
    1: [[1.25, -1.5], [2.25, -2.5]],
    2: [[2.25, -2.5], [3.25, -3.5], [4.25, -4.5]],
    3: [[3.25, -3.5]],
    4: [[4.25, -4.5], [5.25, -5.5]],
    5: [[5.25, -5.5], [6.25, -6.5], [7.25, -7.5]],
}


class FixtureEncoder:
    """Replace only costly pretrained inference, never checkpoint code."""

    def __init__(self, trace: Path, *, fail_call=None, terminate=False, wrong_ids=False):
        self.trace, self.fail_call, self.terminate = trace, fail_call, terminate
        self.wrong_ids, self.calls = wrong_ids, 0
        self.metadata = {
            "model": {"checkpoint": SETTINGS["checkpoint"], "revision": SETTINGS["revision"]},
            "processor": {"fixture": "independent RGB encoder"},
            "transformations": {"image": "RGB, no resizing"},
            "token_policy": {"special_indices": [], "fixture": True},
        }
        self.tensor_schema = {
            "vectors": {"dtype": "float32", "tail_shape": [2], "role": "embedding", "alignment": "output"},
            "mask": {"dtype": "bool", "tail_shape": [], "role": "mask", "alignment": "output"},
        }

    def encode(self, images, ids, *, ocr_records=None):
        self.calls += 1
        if self.calls == self.fail_call:
            if self.terminate:
                os._exit(73)  # No finally handlers: reproduce a lost process.
            raise RuntimeError("independent injected interruption")
        assert len(images) == len(ids) and all(im.mode == "RGB" for im in images)
        assert ocr_records is None
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(ids) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        outputs = []
        for key in ids:
            count = key % 3 + 1
            outputs.append({
                "document_id": key,
                "tensors": {
                    "vectors": np.array([[key + n + .25, -key - n - .5] for n in range(count)], dtype=np.float32),
                    "mask": np.array([n % 2 == 0 for n in range(count)], dtype=bool),
                },
                "token_metadata": {"parent_document_id": key, "special_indices": [], "fixture": True},
            })
        if self.wrong_ids:
            outputs[-1]["document_id"] = outputs[0]["document_id"]
        return outputs


def run_fixture(root, directory, source, trace, *, settings=None, **encoder_options):
    encoder = FixtureEncoder(trace, **encoder_options)
    with patch("src.embeddings.pipeline.importlib.metadata.version", return_value="fixture-version"), \
         patch("src.embeddings.models.load_encoder", return_value=encoder):
        return extract_embeddings(copy.deepcopy(settings or SETTINGS), root, directory,
                                  source_root=source, document_ids=[5, 2, 0, 4, 1, 3])


class IndependentStep5Recovery(unittest.TestCase):
    def setUp(self):
        cache = ROOT / "cache/test-step5-independent"
        cache.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.base = Path(temporary.name)
        self.root = self.base / "ah-clustering-experiment"
        self.source = self.base / "source"
        (self.source / "src").mkdir(parents=True)
        (self.source / "src/fixture.py").write_text("# Independent storage fixture\n", encoding="utf-8")
        (self.root / "outputs/images").mkdir(parents=True)
        for key in GOLDEN:
            Image.new("RGB", (17, 23), "white").save(self.root / f"outputs/images/{key}.png")
        image_sha = file_sha256(self.root / "outputs/images/0.png")
        csv = ROOT.parent / "data/split.csv"
        frame = pd.read_csv(csv).rename(columns={"id": "document_id"})
        frame["image_path"] = frame.document_id.map(lambda key: f"outputs/images/{key}.png")
        frame["image_sha256"] = image_sha
        frame["image_width"], frame["image_height"], frame["image_format"] = 17, 23, "PNG"
        frame["dataset_revision"] = DATASET_REVISION
        frame["shared_csv_sha256"] = file_sha256(csv)
        identity = {"schema_version": 1,
                    "dataset": {"name": DATASET_NAME, "revision": DATASET_REVISION, "upstream_split": "test"},
                    "shared_csv_sha256": file_sha256(csv)}
        dataset_id = "dataset-" + fingerprint(identity)[:20]
        self.directory = self.root / f"outputs/metadata/{dataset_id}"
        self.directory.mkdir(parents=True)
        frame.to_parquet(self.directory / "manifest.parquet", index=False)
        metadata = {
            "status": "complete", "identity": identity, "identity_sha256": fingerprint(identity),
            "dataset_id": dataset_id, "manifest_sha256": file_sha256(self.directory / "manifest.parquet"),
            "document_count": 1000, "final_label_count": 19,
            "partition_counts": {"train": 700, "val": 150, "test": 150},
            "reference_text_policy": REFERENCE_TEXT_POLICY, "limitations": LIMITATIONS,
            "python_version": "fixture", "created_at": "2026-10-06T00:00:00+00:00",
            "dependencies": {key: "fixture" for key in ("datasets", "pandas", "pyarrow", "Pillow")},
            "preparation_provenance": preparation_provenance(),
        }
        (self.directory / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.trace = self.root / "inference-trace.jsonl"

    def extract(self, **options):
        return run_fixture(self.root, self.directory, self.source, self.trace, **options)

    def inferred_ids(self):
        return [key for line in self.trace.read_text(encoding="utf-8").splitlines() for key in json.loads(line)]

    def assert_native(self, result):
        restored = read_embeddings(self.root, result["config"])
        self.assertEqual(set(restored), set(GOLDEN))
        self.assertEqual(result["document_count"], 6)
        for key, expected in GOLDEN.items():
            actual = restored[key]
            self.assertEqual(actual["document_id"], key)
            self.assertEqual(actual["tensors"]["vectors"].dtype, np.dtype("float32"))
            np.testing.assert_array_equal(actual["tensors"]["vectors"], np.asarray(expected, dtype=np.float32))
            self.assertEqual(actual["tensors"]["mask"].dtype, np.dtype("bool"))
            np.testing.assert_array_equal(actual["tensors"]["mask"], [True, False, True][:len(expected)])
            self.assertEqual(actual["token_metadata"],
                             {"parent_document_id": key, "special_indices": [], "fixture": True})
        # Release memory maps before Windows tempfile cleanup.
        del restored

    def test_uninterrupted_native_outputs_match_literal_oracle(self):
        result = self.extract()
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 4, 5])
        self.assert_native(result)

    def test_exception_resume_does_not_repeat_completed_inference(self):
        with self.assertRaisesRegex(RuntimeError, "independent injected"):
            self.extract(fail_call=2)
        self.assertEqual(self.inferred_ids(), [0, 1])
        self.assertFalse(list((self.root / "outputs/metadata").glob("extraction-*.json")))
        result = self.extract()
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 4, 5])
        self.assert_native(result)
        with patch("src.embeddings.pipeline.importlib.metadata.version", return_value="fixture-version"), \
             patch("src.embeddings.models.load_encoder", side_effect=AssertionError("Completed cache loaded encoder")):
            reused = extract_embeddings(SETTINGS, self.root, self.directory, source_root=self.source,
                                        document_ids=[0, 1, 2, 3, 4, 5])
        self.assertTrue(reused["reused"])
        self.assert_native(reused)

    def test_real_process_exit_preserves_completed_inference(self):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--crash-child",
                                  str(self.root), str(self.directory), str(self.source), str(self.trace)],
                                 capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 73, process.stdout + process.stderr)
        self.assertEqual(self.inferred_ids(), [0, 1])
        result = self.extract()
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 4, 5])
        self.assert_native(result)

    def test_process_exit_during_payload_publication_recovers_unfinished_batch(self):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--write-crash-child",
                                  str(self.root), str(self.directory), str(self.source), str(self.trace)],
                                 capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 74, process.stdout + process.stderr)
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3])
        result = self.extract()
        # Batch 2 completed inference but its write was killed before publication.
        # Repeating it is necessary; the previously published batch stays reused.
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 2, 3, 4, 5])
        self.assert_native(result)

    def check_partial_payload_damage(self, *, remove):
        with self.assertRaisesRegex(RuntimeError, "independent injected"):
            self.extract(fail_call=3)
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3])
        # Public journal layout supplied by the contract owner; no implementation
        # internals are read or replaced to select the persisted payload.
        partial = sorted((self.root / "outputs/checkpoints").rglob("batch-*.bin"))
        self.assertTrue(partial, "Completed batches must have durable numeric files")
        if remove:
            partial[0].unlink()
        else:
            partial[0].write_bytes(partial[0].read_bytes() + b"independent corruption")
        result = self.extract()
        counts = {key: self.inferred_ids().count(key) for key in GOLDEN}
        # Whichever saved batch owned the corrupt file is unfinished; the other
        # successful batch is still reusable. This does not depend on filenames.
        self.assertIn(counts, [dict(enumerate([2, 2, 1, 1, 1, 1])),
                               dict(enumerate([1, 1, 2, 2, 1, 1]))])
        self.assert_native(result)

    def test_corrupt_partial_shard_is_recomputed_without_losing_other_batch(self):
        self.check_partial_payload_damage(remove=False)

    def test_missing_partial_shard_is_recomputed_without_losing_other_batch(self):
        self.check_partial_payload_damage(remove=True)

    def test_unpublished_output_damage_repairs_from_valid_checkpoint_without_inference(self):
        with self.assertRaisesRegex(RuntimeError, "independent injected"):
            self.extract(fail_call=3)
        final_arrays = sorted((self.root / "outputs/embeddings").rglob("*.npy"))
        self.assertTrue(final_arrays, "Incremental final shards should exist after two batches")
        final_arrays[0].write_bytes(final_arrays[0].read_bytes() + b"damaged final materialization")
        result = self.extract()
        self.assertEqual(self.inferred_ids(), [0, 1, 2, 3, 4, 5])
        self.assert_native(result)

    def test_changed_configuration_never_reuses_partial_inference(self):
        with self.assertRaisesRegex(RuntimeError, "independent injected"):
            self.extract(fail_call=2)
        changed = copy.deepcopy(SETTINGS)
        changed["processor_options"] = {"do_resize": False}
        result = self.extract(settings=changed)
        self.assertEqual(self.inferred_ids(), [0, 1, 0, 1, 2, 3, 4, 5])
        self.assert_native(result)

    def test_duplicate_and_missing_encoder_ids_never_publish_completion(self):
        with self.assertRaises(ValueError):
            self.extract(wrong_ids=True)
        self.assertFalse(list((self.root / "outputs/metadata").glob("extraction-*.json")))
        result = self.extract()
        self.assert_native(result)

    def test_literal_tensor_oracle_detects_perturbation_negative_control(self):
        result = self.extract()
        # Deliberately perturb the observable reader result, leaving all shapes,
        # IDs and dtypes valid. The exact native-output oracle must reject it.
        restored = read_embeddings(self.root, result["config"])
        altered = copy.deepcopy(restored)
        del restored
        altered[2]["tensors"]["vectors"][0, 0] += 1
        with patch(__name__ + ".read_embeddings", return_value=altered):
            with self.assertRaises(AssertionError):
                self.assert_native(result)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--crash-child":
        run_fixture(*(Path(value) for value in sys.argv[2:]), fail_call=2, terminate=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--write-crash-child":
        original_replace = os.replace

        def crash_before_batch_publication(source, destination, *args, **kwargs):
            if Path(destination).name == "batch-00002.bin":
                os._exit(74)
            return original_replace(source, destination, *args, **kwargs)

        with patch("os.replace", side_effect=crash_before_batch_publication):
            run_fixture(*(Path(value) for value in sys.argv[2:]))
    else:
        unittest.main()
