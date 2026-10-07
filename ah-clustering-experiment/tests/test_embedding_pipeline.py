"""Self-checked boundary tests: actual local storage, no model inference claim.

Synthetic images and tensors are deliberate fixtures. The cache test substitutes
dependency-version lookup because storage-only environments need not install
torch, and makes model loading fail if called. Dataset assignments come from the
shared CSV; metadata and image artifacts are isolated in an experiment tempfile.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import embedding_config
from src.dataset import (DATASET_NAME, DATASET_REVISION, LIMITATIONS, REFERENCE_TEXT_POLICY,
                         _atomic_json, file_sha256, fingerprint, preparation_provenance)
from src.embeddings import write_embeddings
from src.embeddings.pipeline import extract_embeddings, load_documents, validate_settings
from src.provenance import source_identity


SETTINGS = {"family": "dinov3", "checkpoint": "fixture/never-loaded", "revision": "a" * 40,
            "input_mode": "image_only", "dtype": "float32", "batch_size": 1, "processor_options": {}}


class ExtractionBoundary(unittest.TestCase):
    def setUp(self):
        cache = ROOT / "cache/test-pipeline"
        cache.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.root = Path(temporary.name) / "ah-clustering-experiment"
        self.source = Path(temporary.name) / "source"
        (self.source / "src").mkdir(parents=True)
        (self.source / "src/fixture.py").write_text("# Cache boundary fixture; no model inference.\n")
        (self.root / "outputs/images").mkdir(parents=True)
        for doc_id in (0, 1):
            Image.new("RGB", (20, 30), "white").save(self.root / f"outputs/images/{doc_id}.png")
        image_sha = file_sha256(self.root / "outputs/images/0.png")
        csv = ROOT.parent / "data/split.csv"
        frame = pd.read_csv(csv).rename(columns={"id": "document_id"})
        frame["image_path"] = frame.document_id.map(lambda key: f"outputs/images/{key}.png")
        frame["image_sha256"] = image_sha
        frame["image_width"], frame["image_height"], frame["image_format"] = 20, 30, "PNG"
        frame["dataset_revision"] = DATASET_REVISION
        frame["shared_csv_sha256"] = file_sha256(csv)
        identity = {"schema_version": 1, "dataset": {"name": DATASET_NAME, "revision": DATASET_REVISION, "upstream_split": "test"},
                    "shared_csv_sha256": file_sha256(csv)}
        dataset_id = "dataset-" + fingerprint(identity)[:20]
        self.directory = self.root / f"outputs/metadata/{dataset_id}"
        self.directory.mkdir(parents=True)
        frame.to_parquet(self.directory / "manifest.parquet", index=False)
        self.metadata = {"status": "complete", "identity": identity, "identity_sha256": fingerprint(identity),
                         "dataset_id": dataset_id, "manifest_sha256": file_sha256(self.directory / "manifest.parquet"),
                         "document_count": 1000, "final_label_count": 19,
                         "partition_counts": {"train": 700, "val": 150, "test": 150},
                         "reference_text_policy": REFERENCE_TEXT_POLICY, "limitations": LIMITATIONS,
                         "python_version": "fixture", "created_at": "2026-10-04T00:00:00+00:00",
                         "dependencies": {key: "fixture" for key in ("datasets", "pandas", "pyarrow", "Pillow")},
                         "preparation_provenance": preparation_provenance()}
        _atomic_json(self.directory / "dataset.json", self.metadata)

    def test_load_exposes_only_sorted_image_identity(self):
        _, documents = load_documents(self.root, self.directory, [1, 0])
        self.assertEqual([row["document_id"] for row in documents], [0, 1])
        self.assertTrue(all(set(row) == {"document_id", "image_path", "image_sha256"} for row in documents))

    def test_bad_selection_and_manifest_checksum_fail(self):
        for ids in ([0, 0], [1001], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                load_documents(self.root, self.directory, ids)
        path = self.directory / "manifest.parquet"
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            load_documents(self.root, self.directory, [0])

    def test_changed_original_image_fails_before_encoding(self):
        Image.new("RGB", (20, 30), "black").save(self.root / "outputs/images/0.png")
        with self.assertRaisesRegex(ValueError, "Image checksum"):
            load_documents(self.root, self.directory, [0])

    def test_manifest_semantics_checked_even_with_recomputed_checksum(self):
        path = self.directory / "manifest.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "split"] = "unknown"
        frame.to_parquet(path, index=False)
        self.metadata["manifest_sha256"] = file_sha256(path)
        _atomic_json(self.directory / "dataset.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "Partition counts"):
            load_documents(self.root, self.directory, [0])

    def seed_completed_cache(self):
        dataset, documents = load_documents(self.root, self.directory, [0])
        request = {"settings": SETTINGS, "dataset": dataset, "documents": documents, "ocr": None, "device": "cpu",
                   "dependencies": {name: "fixture" for name in ("torch", "torchvision", "transformers", "tokenizers",
                                                                 "sentencepiece", "safetensors", "huggingface_hub", "Pillow", "numpy")},
                   "source_digest": source_identity(self.source)["source_digest"]}
        request_id = fingerprint(request)
        config = embedding_config(
            model={"checkpoint": SETTINGS["checkpoint"], "revision": SETTINGS["revision"]},
            processor={"extraction_request": request_id, "execution": request}, dataset=dataset, documents=documents,
            input_mode="image_only", transformations={"image": "fixture pixels"},
            tensors={"vectors": {"dtype": "float32", "tail_shape": [2], "role": "embedding", "alignment": "output"}},
            token_policy={"fixture": True}, source_digest=request["source_digest"])
        artifact_id = write_embeddings(self.root, config, [{"document_id": 0, "tensors": {"vectors": np.array([[1, 2]], dtype=np.float32)},
                                                           "token_metadata": {"fixture": True}}], source_root=self.source)
        _atomic_json(self.root / f"outputs/metadata/extraction-{request_id[:20]}.json", {"request": request, "artifact_id": artifact_id})
        return artifact_id

    def test_cache_hit_validates_actual_payload_without_model_loading(self):
        artifact_id = self.seed_completed_cache()
        with patch("src.embeddings.pipeline.importlib.metadata.version", return_value="fixture"), \
             patch("src.embeddings.models.load_encoder", side_effect=AssertionError("Cache hit loaded a model")) as loader:
            result = extract_embeddings(SETTINGS, self.root, self.directory, source_root=self.source, document_ids=[0])
        self.assertTrue(result["reused"])
        self.assertEqual(result["artifact_id"], artifact_id)
        self.assertEqual(result["document_count"], 1)
        loader.assert_not_called()

    def test_corrupt_cached_tensor_fails_without_loading_model(self):
        artifact_id = self.seed_completed_cache()
        tensor = next((self.root / f"outputs/embeddings/{artifact_id}").glob("*.npy"))
        tensor.write_bytes(tensor.read_bytes() + b"corruption")
        with patch("src.embeddings.pipeline.importlib.metadata.version", return_value="fixture"), \
             patch("src.embeddings.models.load_encoder", side_effect=AssertionError("Corrupt cache loaded a model")) as loader:
            with self.assertRaisesRegex(ValueError, "checksum"):
                extract_embeddings(SETTINGS, self.root, self.directory, source_root=self.source, document_ids=[0])
        loader.assert_not_called()
        logs = list((self.root / "outputs/logs").glob("extraction-*.json"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(json.loads(logs[0].read_text())["status"], "failed")

    def test_changed_preprocessing_dependency_cannot_reuse_cache(self):
        self.seed_completed_cache()
        for dependency in ("tokenizers", "torchvision", "sentencepiece"):
            with self.subTest(dependency=dependency), \
                 patch("src.embeddings.pipeline.importlib.metadata.version",
                       side_effect=lambda name: "changed" if name == dependency else "fixture"), \
                 patch("src.embeddings.models.load_encoder", side_effect=RuntimeError("new inference required")) as loader:
                with self.assertRaisesRegex(RuntimeError, "new inference required"):
                    extract_embeddings(SETTINGS, self.root, self.directory, source_root=self.source, document_ids=[0])
                loader.assert_called_once()

    def test_invalid_settings_fail(self):
        for change in ({"revision": "main"}, {"batch_size": 0}, {"batch_size": True},
                       {"input_mode": "image_and_ocr"}, {"dtype": "bfloat16"}, {"extra": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_settings({**SETTINGS, **change})


if __name__ == "__main__":
    unittest.main()
