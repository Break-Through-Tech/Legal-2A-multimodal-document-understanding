"""Storage acceptance cases use exact hand-specified values and actual disk I/O.

These arrays are fixtures, not claims about model output. Direct NumPy/Parquet
reads independently check the encoding; negative controls also bypass checksums
to test semantic validation rather than only detecting changed bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import ArtifactWriter, artifact_identity, initialize_storage, read_artifact
from src.contracts import clustering_config, embedding_config
from src.embeddings import read_embeddings, write_embeddings
from src.provenance import capture_provenance, source_identity


def digest(value):
    # Independent implementation of the documented canonical JSON digest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ArtifactContract(unittest.TestCase):
    def setUp(self):
        cache = ROOT / "cache" / "test-artifacts"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "ah-clustering-experiment"
        self.source = Path(self.temporary.name) / "source"
        (self.source / "src").mkdir(parents=True)
        (self.source / "src" / "encoder.py").write_text("OUTPUT_POLICY = 'native'\n")
        self.source_digest = source_identity(self.source)["source_digest"]
        self.arguments = dict(
            model={"checkpoint": "fixture/native-encoder", "revision": "a" * 40},
            processor={"resize": [2, 2], "revision": "a" * 40},
            dataset={"dataset_id": "dataset-fixture", "identity_sha256": "b" * 64, "manifest_sha256": "c" * 64},
            documents=[{"document_id": i, "image_path": f"outputs/images/{i}.png", "image_sha256": str(i % 10) * 64} for i in (19, 7, 42)],
            input_mode="image_only", transformations={"image": {"resize": [2, 2], "coordinates": "unchanged"}},
            tensors={"vectors": {"dtype": "float32", "tail_shape": [2], "role": "embedding", "alignment": "output"},
                     "mask": {"dtype": "bool", "tail_shape": [], "role": "mask", "alignment": "output"},
                     "cls": {"dtype": "float32", "tail_shape": [2], "role": "embedding", "alignment": "cls"}},
            token_policy={"special_tokens": "retained", "register_tokens": "none", "prompt_tokens": "none"},
            source_digest=self.source_digest,
        )
        self.config = embedding_config(**self.arguments)
        self.documents = [
            {"document_id": 19, "tensors": {"vectors": np.array([[1, 2], [3, 4]], dtype="float32"),
             "mask": np.array([True, False]), "cls": np.array([[91, 92]], dtype="float32")}, "token_metadata": {"special_indices": [0]}},
            {"document_id": 7, "tensors": {"vectors": np.array([[5, 6]], dtype="float32"),
             "mask": np.array([True]), "cls": np.array([[71, 72]], dtype="float32")}, "token_metadata": {"special_indices": []}},
            {"document_id": 42, "tensors": {"vectors": np.empty((0, 2), dtype="float32"),
             "mask": np.array([], dtype=bool), "cls": np.array([[41, 42]], dtype="float32")}, "token_metadata": {"empty_reason": "fixture empty sequence"}},
        ]

    def write(self, documents=None):
        return write_embeddings(self.root, self.config, self.documents if documents is None else documents,
                                source_root=self.source, shard_size=2)

    def metadata_path(self, artifact_id):
        return self.root / "outputs" / "metadata" / artifact_id / "artifact.json"

    def rewrite_metadata(self, artifact_id, mutate):
        path = self.metadata_path(artifact_id)
        manifest = json.loads(path.read_text())
        mutate(manifest)
        manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        path.write_text(json.dumps(manifest))

    def rewrite_table(self, artifact_id, mutate):
        bundle = read_artifact(self.root, artifact_id)
        path = self.root / bundle["manifest"]["files"]["documents"]["path"]
        frame = pd.read_parquet(path)
        mutate(frame)
        frame.to_parquet(path, index=False)
        self.rewrite_metadata(artifact_id, lambda m: m["files"]["documents"].update(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))

    def test_native_round_trip_and_independent_encoding(self):
        artifact_id = self.write()
        restored = read_embeddings(self.root, self.config)
        self.assertEqual(set(restored), {7, 19, 42})
        for original in self.documents:
            actual = restored[original["document_id"]]
            self.assertEqual(actual["token_metadata"], original["token_metadata"])
            for name, array in original["tensors"].items():
                np.testing.assert_array_equal(actual["tensors"][name], array)
                self.assertEqual(actual["tensors"][name].dtype, array.dtype)
        data = self.root / "outputs/embeddings" / artifact_id
        np.testing.assert_array_equal(np.load(data / "shard-00000-vectors.npy", allow_pickle=False), [[1, 2], [3, 4], [5, 6]])
        records = pd.read_parquet(data / "documents.parquet").set_index("document_id")
        self.assertEqual((records.at[19, "vectors__offset"], records.at[19, "vectors__length"]), (0, 2))
        self.assertEqual((records.at[7, "vectors__offset"], records.at[7, "vectors__length"]), (2, 1))
        self.assertEqual((records.at[42, "shard"], records.at[42, "vectors__length"]), (1, 0))
        manifest = json.loads(self.metadata_path(artifact_id).read_text())
        self.assertEqual(manifest["provenance"]["source"]["source_digest"], self.source_digest)
        self.assertIn("numpy", {key.lower() for key in manifest["provenance"]["dependencies"]})
        self.assertGreaterEqual(manifest["provenance"]["runtime"]["elapsed_seconds"], 0)

    def test_document_order_is_not_identity_or_join(self):
        config = embedding_config(**{**self.arguments, "documents": list(reversed(self.arguments["documents"]))})
        self.assertEqual(config, self.config)
        artifact_id = self.write()
        self.rewrite_table(artifact_id, lambda frame: frame.sort_values("document_id", ascending=False, inplace=True))
        np.testing.assert_array_equal(read_embeddings(self.root, config)[7]["tensors"]["vectors"], [[5, 6]])

    def test_corrupt_and_missing_shards_rejected(self):
        artifact_id = self.write()
        path = self.root / "outputs/embeddings" / artifact_id / "shard-00000-vectors.npy"
        path.write_bytes(b"not a numeric array")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_embeddings(self.root, self.config)
        path.unlink()
        with self.assertRaises(ValueError):
            read_embeddings(self.root, self.config)

    def test_manifest_corruption_and_unsupported_schema_rejected(self):
        artifact_id = self.write()
        path = self.metadata_path(artifact_id)
        original = path.read_text()
        manifest = json.loads(original)
        manifest["provenance"]["runtime"]["elapsed_seconds"] = 999
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Manifest checksum"):
            read_embeddings(self.root, self.config)
        path.write_text(original)
        self.rewrite_metadata(artifact_id, lambda m: m.update(schema_version=999))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            read_embeddings(self.root, self.config)

    def test_checksum_correct_wrong_dtype_and_nonfinite_values_rejected(self):
        artifact_id = self.write()
        path = self.root / "outputs/embeddings" / artifact_id / "shard-00000-vectors.npy"
        for array in (np.array([[1, 2], [3, 4], [5, 6]], dtype="float64"),
                      np.array([[np.inf, 2], [3, 4], [5, 6]], dtype="float32")):
            np.save(path, array, allow_pickle=False)
            self.rewrite_metadata(artifact_id, lambda m: m["files"]["shard-00000-vectors"].update(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                dtype=array.dtype.str))
            with self.subTest(dtype=array.dtype), self.assertRaises(ValueError):
                read_embeddings(self.root, self.config)

    def test_incompatible_config_and_corrupt_snapshot_rejected(self):
        artifact_id = self.write()
        wrong = copy.deepcopy(self.config)
        wrong["processor"]["resize"] = [3, 3]
        with self.assertRaisesRegex(ValueError, "Incompatible configuration"):
            read_artifact(self.root, artifact_id, expected_config=wrong)
        self.metadata_path(artifact_id).with_name("config.json").write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "Configuration checksum"):
            read_artifact(self.root, artifact_id)

    def test_checksum_correct_invalid_offsets_rejected(self):
        artifact_id = self.write()
        self.rewrite_table(artifact_id, lambda frame: frame.loc.__setitem__((frame.document_id == 7, "vectors__offset"), 0))
        with self.assertRaisesRegex(ValueError, "offsets"):
            read_embeddings(self.root, self.config)

    def test_checksum_correct_wrong_ids_rejected(self):
        artifact_id = self.write()
        self.rewrite_table(artifact_id, lambda frame: frame.loc.__setitem__((frame.document_id == 7, "document_id"), 8))
        with self.assertRaisesRegex(ValueError, "document IDs"):
            read_embeddings(self.root, self.config)

    def test_mask_alignment_dtype_and_nonfinite_inputs_rejected(self):
        for mutation in (lambda d: d["tensors"].update(mask=np.array([True])),
                         lambda d: d["tensors"].update(vectors=d["tensors"]["vectors"].astype("float64")),
                         lambda d: d["tensors"]["vectors"].__setitem__((0, 0), np.nan)):
            bad = copy.deepcopy(self.documents)
            mutation(bad[0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.write(bad)

    def test_missing_duplicate_and_unexpected_documents_rejected(self):
        for documents in (self.documents[:-1], self.documents + self.documents[:1],
                          [{**self.documents[0], "document_id": 999}, *self.documents[1:]]):
            with self.subTest(ids=[d["document_id"] for d in documents]), self.assertRaises(ValueError):
                self.write(documents)
            artifact_id, _ = artifact_identity("embeddings", self.config)
            self.assertFalse(self.metadata_path(artifact_id).exists())

    def test_interruption_never_publishes_completion_and_retry_works(self):
        def interrupted():
            yield from self.documents[:2]
            raise RuntimeError("simulated producer interruption after first shard")
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.write(interrupted())
        artifact_id, _ = artifact_identity("embeddings", self.config)
        self.assertFalse(self.metadata_path(artifact_id).exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_embeddings(self.root, self.config)
        self.write()
        self.assertEqual(set(read_embeddings(self.root, self.config)), {7, 19, 42})

    def test_reused_producer_buffers_are_snapshotted(self):
        vector = np.zeros((1, 2), dtype="float32")
        def documents():
            for doc_id in (19, 7, 42):
                vector[:] = doc_id
                yield {"document_id": doc_id, "tensors": {"vectors": vector, "cls": vector,
                       "mask": np.array([True])}, "token_metadata": {"special_indices": []}}
        self.write(documents())
        restored = read_embeddings(self.root, self.config)
        for doc_id in (19, 7, 42):
            np.testing.assert_array_equal(restored[doc_id]["tensors"]["vectors"], [[doc_id, doc_id]])

    def test_one_writer_and_completed_immutability(self):
        with ArtifactWriter(self.root, "embeddings", self.config):
            with self.assertRaisesRegex(ValueError, "writer lock"):
                with ArtifactWriter(self.root, "embeddings", self.config):
                    pass
        self.write()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.write()

    def test_ocr_and_preprocessing_cache_invalidation(self):
        ocr = {"artifact_id": "ocr-fixture", "content_sha256": "d" * 64,
               "engine": "fixture-only", "engine_version": "1", "config": {"language": "en"}}
        args = {**self.arguments, "input_mode": "image_and_ocr", "ocr": ocr,
                "transformations": {"image": {"resize": [2, 2]}, "word_coordinates": {"from": "pixels", "to": "0..1000"}}}
        original = embedding_config(**args)
        changed_ocr = {**ocr, "content_sha256": "e" * 64}
        changed = embedding_config(**{**args, "ocr": changed_ocr})
        self.assertNotEqual(artifact_identity("embeddings", original), artifact_identity("embeddings", changed))
        self.assertEqual(embedding_config(**{**self.arguments, "ocr": ocr}),
                         embedding_config(**{**self.arguments, "ocr": changed_ocr}))
        for field, value in (("processor", {"resize": [4, 4]}),
                             ("transformations", {"image": {"resize": [8, 8]}}),
                             ("model", {"checkpoint": "fixture/native-encoder", "revision": "b" * 40})):
            self.assertNotEqual(artifact_identity("embeddings", self.config),
                                artifact_identity("embeddings", embedding_config(**{**self.arguments, field: value})))
        with self.assertRaisesRegex(ValueError, "OCR"):
            embedding_config(**{**args, "ocr": None})

    def test_floating_revision_and_label_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            embedding_config(**{**self.arguments, "model": {"checkpoint": "fixture/encoder", "revision": "main"}})
        documents = copy.deepcopy(self.arguments["documents"])
        documents[0]["label"] = "Do not embed me"
        with self.assertRaisesRegex(ValueError, "only ID"):
            embedding_config(**{**self.arguments, "documents": documents})

    def test_output_layout_and_other_result_formats(self):
        initialize_storage(self.root)
        for kind in ("embeddings", "ocr", "similarities", "clusters", "metrics", "projections", "metadata", "logs"):
            self.assertTrue((self.root / "outputs" / kind).is_dir())
        provenance = capture_provenance(self.source, time.perf_counter())
        for kind in ("ocr", "similarities", "clusters", "metrics", "projections"):
            with self.subTest(kind=kind), ArtifactWriter(self.root, kind, {"fixture": kind}) as writer:
                if kind == "similarities":
                    writer.array("matrix", np.array([[1., .25], [.25, 1.]], dtype="float32"))
                    writer.table("ordered_ids", pd.DataFrame({"document_id": [19, 7], "position": [0, 1]}))
                elif kind == "metrics":
                    writer.json("scores", {"silhouette": {"value": None, "reason": "one cluster", "coverage": 1.0}})
                elif kind == "ocr":
                    writer.table("documents", pd.DataFrame({"document_id": [19, 7], "status": ["complete", "empty"],
                                 "words_json": ['["word"]', '[]'], "boxes_json": ['[[0,0,1,1]]', '[]']}))
                else:
                    writer.table("documents", pd.DataFrame({"document_id": [19, 7], "cluster_id" if kind == "clusters" else "x": [0, 1]}))
                writer.complete(provenance)
                bundle = read_artifact(self.root, writer.artifact_id, expected_config={"fixture": kind})
                self.assertEqual(bundle["manifest"]["kind"], kind)

    def test_object_arrays_path_escape_and_duplicate_table_keys_rejected(self):
        with ArtifactWriter(self.root, "ocr", {"fixture": True}) as writer:
            with self.assertRaises(ValueError):
                writer.array("unsafe", np.array([{"x": 1}], dtype=object))
            with self.assertRaises(ValueError):
                writer.array("../outside", np.array([1]))
            with self.assertRaises(ValueError):
                writer.table("documents", pd.DataFrame({"document_id": [7, 7]}))
        with self.assertRaises(ValueError):
            read_artifact(self.root, "../../outside")

    def test_uncommitted_source_digest_and_generated_exclusion(self):
        before = source_identity(self.source)
        (self.source / "outputs").mkdir()
        (self.source / "outputs" / "ignore.py").write_text("generated")
        self.assertEqual(source_identity(self.source)["source_digest"], before["source_digest"])
        (self.source / "src" / "new.py").write_text("NEW = True\n")
        after = source_identity(self.source)
        self.assertNotEqual(after["source_digest"], before["source_digest"])
        self.assertIn("src/new.py", after["source_files_sha256"])
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.write()

    def test_clustering_choices_change_run_identity(self):
        arguments = dict(embedding_artifact="embeddings-fixture", embedding_identity="a" * 64,
                         model=self.config["model"], representation="cls", token_selection={"kind": "cls"},
                         pooling={"kind": "none"}, normalization={"kind": "none"}, similarity=None,
                         algorithm="kmeans", parameters={"n_clusters": 2}, seed=9,
                         cohort={"dataset_id": "dataset-fixture", "cohort": "train", "designation": "initial_training_partition", "document_ids": [19, 7]},
                         cluster_count_policy={"mode": "independently_chosen", "n_clusters": 2, "rationale": "fixture"},
                         source_digest=self.source_digest)
        baseline = artifact_identity("clusters", clustering_config(**arguments))
        for field, value in (("seed", 10), ("representation", "patches"), ("normalization", {"kind": "l2"}),
                             ("pooling", {"kind": "mean"}), ("parameters", {"n_clusters": 2, "n_init": 10}),
                             ("algorithm", "agglomerative"), ("token_selection", {"kind": "all"}),
                             ("similarity", {"artifact_id": "similarities-fixture", "identity_sha256": "e" * 64, "definition": "cosine"})):
            with self.subTest(field=field):
                self.assertNotEqual(baseline, artifact_identity("clusters", clustering_config(**{**arguments, field: value})))
        with self.assertRaisesRegex(ValueError, "exploratory"):
            clustering_config(**{**arguments, "cohort": {**arguments["cohort"], "cohort": "all"}})
        with self.assertRaisesRegex(ValueError, "count disagree"):
            clustering_config(**{**arguments, "parameters": {"n_clusters": 3}})


if __name__ == "__main__":
    unittest.main()
