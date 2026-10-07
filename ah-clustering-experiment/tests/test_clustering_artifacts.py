"""Real storage checks for clustering-only reuse and image-free cache consumption."""

import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_embedding_pipeline as fixtures
from src.clustering.experiment import load_cohort, similarity_artifact


class ClusteringArtifactChecks(unittest.TestCase):
    setUp = fixtures.ExtractionBoundary.setUp
    def test_cohort_reads_saved_metadata_without_original_pixels(self):
        # This fixture has 1,000 manifest rows but only two actual images.
        cohort, rows = load_cohort(self.root, self.directory, "train")
        self.assertEqual(len(rows), 700)
        self.assertEqual(cohort["document_ids"], sorted(row["document_id"] for row in rows))
        self.assertTrue(all(set(row) == {"document_id", "image_path", "image_sha256"} for row in rows))

    def test_clustering_parameter_change_reuses_similarity_without_computing(self):
        for name in ("src/clustering/similarity.py", "src/clustering/experiment.py", "src/artifacts.py",
                     "src/contracts.py", "src/dataset.py", "src/embeddings/__init__.py"):
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        configs = self.source / "configs"
        configs.mkdir()
        (configs / "clustering.json").write_text('{"n_clusters": 2}')
        embedding = {"manifest": {"artifact_id": "embeddings-" + "a"*20,
                     "identity_sha256": "b"*64, "manifest_sha256": "c"*64}}
        documents = {key: {"tensors": {"vectors": np.array(value, dtype=np.float32),
                      "attention_mask": np.ones(len(value), dtype=bool), "image_mask": np.ones(len(value), dtype=bool),
                      "special_mask": np.ones(len(value), dtype=bool)}}
                     for key, value in {0: [[1, 0]], 1: [[0, 1]]}.items()}
        cohort = {"document_ids": [0, 1], "cohort": "train"}
        first = similarity_artifact(self.root, self.source, embedding, documents, cohort, "cpu", {"numpy": np.__version__})
        (configs / "clustering.json").write_text('{"n_clusters": 3}')
        with patch("src.clustering.experiment.compute_similarity", side_effect=AssertionError("Similarity recomputed")):
            second = similarity_artifact(self.root, self.source, embedding, documents, cohort, "cpu", {"numpy": np.__version__})
        self.assertEqual(first["manifest"]["artifact_id"], second["manifest"]["artifact_id"])
        np.testing.assert_array_equal(second["arrays"]["distance"], [[0, 1], [1, 0]])


if __name__ == "__main__":
    import unittest
    unittest.main()
