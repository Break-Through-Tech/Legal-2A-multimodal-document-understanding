"""Independent Step 3 oracle: plan/spec and literal arrays, never production expectations.

Scratch stays under cache/independent-step3. No encoders or OCR are mocked or run.
"""

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

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from src.artifacts import ArtifactWriter, artifact_identity, read_artifact
from src.contracts import embedding_config
from src.embeddings import read_embeddings, write_embeddings
from src.provenance import capture_provenance, source_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IndependentStep3Checks(unittest.TestCase):
    def setUp(self):
        scratch = EXPERIMENT / "cache" / "independent-step3"
        scratch.mkdir(parents=True, exist_ok=True)
        # Keep payload paths below Windows MAX_PATH in this deeply nested checkout.
        self.case = Path(tempfile.mkdtemp(prefix="i-", dir=scratch))
        self.root = self.case / "ah-clustering-experiment"
        self.root.mkdir()
        self.source = self.case / "source"
        (self.source / "src").mkdir(parents=True)
        (self.source / "src" / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.provenance = capture_provenance(self.source, time.perf_counter())
        self.kwargs = dict(
            model={"checkpoint": "independent/synthetic-tensors", "revision": "a" * 40},
            processor={"size": [17, 29], "rescale": False},
            dataset={"dataset_id": "independent-fixture", "identity_sha256": "b" * 64, "manifest_sha256": "c" * 64},
            documents=[{"document_id": i, "image_path": f"outputs/images/{i}.png", "image_sha256": str(i % 10) * 64} for i in [901, 4, 72]],
            input_mode="image_only", transformations={"image": {"kind": "none"}, "word_coordinates": {"kind": "none"}},
            tensors={
                "vectors": {"dtype": "float32", "tail_shape": [2], "role": "embedding", "alignment": "tokens"},
                "mask": {"dtype": "bool", "tail_shape": [], "role": "mask", "alignment": "tokens"},
                "ids": {"dtype": "int64", "tail_shape": [], "role": "token_ids", "alignment": "tokens"},
                "tiles": {"dtype": "float16", "tail_shape": [2, 2], "role": "embedding", "alignment": "tiles"},
            },
            token_policy={"special_tokens": "retained", "prompt_tokens": "none"},
            source_digest=self.provenance["source"]["source_digest"],
        )
        self.config = embedding_config(**self.kwargs)
        self.documents = [
            self.document(901, [[-3.5, 2.25], [0.0, -0.125]], [True, False], [2**40 + 3, -7], [[[1, 2], [3, 4]]]),
            self.document(4, [], [], [], []),
            self.document(72, [[16.5, -8.0]], [True], [123], [[[0.5, -0.5], [8, -8]], [[4, 2], [1, 0]]]),
        ]

    def document(self, ident, vectors, mask, ids, tiles):
        return {"document_id": ident, "tensors": {
            "vectors": np.array(vectors, dtype=np.float32).reshape(-1, 2),
            "mask": np.array(mask, dtype=bool), "ids": np.array(ids, dtype=np.int64),
            "tiles": np.array(tiles, dtype=np.float16).reshape(-1, 2, 2),
        }, "token_metadata": {"special_indices": [], "register_indices": [], "prompt_indices": [],
                               "ocr_word_alignment": None, "empty_reason": "no retained tokens" if not vectors else None}}

    def write(self, documents=None, shard_size=2):
        ident = write_embeddings(self.root, self.config, self.documents if documents is None else documents,
                                 source_root=self.source, shard_size=shard_size)
        self.manifest_path = self.root / "outputs" / "metadata" / ident / "artifact.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        return ident

    def resign(self):
        for item in self.manifest["files"].values():
            path = self.root / item["path"]
            item["sha256"] = file_digest(path)
            item["bytes"] = path.stat().st_size
        self.manifest.pop("manifest_sha256", None)
        self.manifest["manifest_sha256"] = digest(self.manifest)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def edit_records(self, mutate):
        path = self.root / self.manifest["files"]["documents"]["path"]
        frame = pd.read_parquet(path)
        mutate(frame)
        frame.to_parquet(path, index=False)
        self.resign()

    def assert_documents(self, actual):
        self.assertEqual(set(actual), {901, 4, 72})
        for expected in self.documents:
            found = actual[expected["document_id"]]
            self.assertEqual(found["token_metadata"], expected["token_metadata"])
            for name, value in expected["tensors"].items():
                self.assertEqual(found["tensors"][name].dtype, value.dtype)
                np.testing.assert_array_equal(found["tensors"][name], value)
                self.assertFalse(found["tensors"][name].flags.writeable)

    def test_literal_native_values_and_direct_formats(self):
        ident = self.write()
        self.assert_documents(read_embeddings(self.root, self.config))
        expected_hash = digest({"schema_version": 1, "kind": "embeddings", "config": self.config})
        self.assertEqual(ident, "embeddings-" + expected_hash[:20])
        self.assertEqual(self.manifest["identity_sha256"], expected_hash)
        records = pd.read_parquet(self.root / self.manifest["files"]["documents"]["path"]).set_index("document_id")
        self.assertEqual(records.loc[901, "vectors__offset"], 0)
        self.assertEqual(records.loc[4, "vectors__offset"], 2)
        self.assertEqual(records.loc[4, "vectors__length"], 0)
        self.assertEqual(records.loc[72, "shard"], 1)
        raw = np.load(self.root / self.manifest["files"]["shard-00000-vectors"]["path"], allow_pickle=False)
        np.testing.assert_array_equal(raw, np.array([[-3.5, 2.25], [0, -0.125]], dtype=np.float32))
        saved_config = json.loads(self.manifest_path.with_name("config.json").read_text())
        self.assertEqual(saved_config, self.config)

    def test_record_order_and_sparse_shard_numbers(self):
        self.write()
        mapping = {0: 3, 1: 41}
        for old_name in list(self.manifest["files"]):
            if old_name.startswith("shard-"):
                old_index = int(old_name.split("-")[1])
                new_name = old_name.replace(f"{old_index:05d}", f"{mapping[old_index]:05d}", 1)
                item = self.manifest["files"].pop(old_name)
                old_path = self.root / item["path"]
                new_path = old_path.with_name(new_name + ".npy")
                old_path.rename(new_path)
                item["path"] = new_path.relative_to(self.root).as_posix()
                self.manifest["files"][new_name] = item
        path = self.root / self.manifest["files"]["documents"]["path"]
        frame = pd.read_parquet(path).iloc[::-1].copy()
        frame["shard"] = frame["shard"].map(mapping)
        frame.to_parquet(path, index=False)
        self.resign()
        self.assert_documents(read_embeddings(self.root, self.config))

    def test_all_empty_sequences_in_own_shards(self):
        self.documents = [self.document(i, [], [], [], []) for i in [901, 4, 72]]
        self.write(shard_size=1)
        self.assert_documents(read_embeddings(self.root, self.config))

    def test_generic_diverse_numeric_types(self):
        values = {"unsigned": np.array([0, 2**64 - 1], dtype=np.uint64),
                  "wide": np.array([[1 / 8, -1024.5]], dtype=np.float64),
                  "narrow": np.array([-128, 127], dtype=np.int8),
                  "bigendian": np.array([1.25, -2.5], dtype=">f4")}
        with ArtifactWriter(self.root, "similarities", {"fixture": "native-types"}) as writer:
            for name, value in values.items():
                writer.array(name, value)
            writer.complete(self.provenance)
        found = read_artifact(self.root, writer.artifact_id)
        for name, value in values.items():
            self.assertEqual(found["arrays"][name].dtype, value.dtype)
            np.testing.assert_array_equal(found["arrays"][name], value)

    def test_embedding_preserves_nonnative_endian_dtype(self):
        self.kwargs["tensors"]["vectors"]["dtype"] = ">f4"
        self.config = embedding_config(**self.kwargs)
        for document in self.documents:
            document["tensors"]["vectors"] = document["tensors"]["vectors"].astype(">f4")
        self.write()
        self.assert_documents(read_embeddings(self.root, self.config))

    def test_reader_rejects_actual_corrupt_bytes(self):
        self.write()
        path = self.root / self.manifest["files"]["shard-00000-vectors"]["path"]
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            original = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([original[0] ^ 1]))
        with self.assertRaises(ValueError):
            read_embeddings(self.root, self.config)

    def test_publication_rejects_actual_corrupt_bytes(self):
        with ArtifactWriter(self.root, "metrics", {"fixture": "corrupt-before-publish"}) as writer:
            writer.json("summary", {"value": None, "reason": "fixture"})
            path = self.root / writer.files["summary"]["path"]
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                writer.complete(self.provenance)
            self.assertFalse(writer.metadata_path.exists())

    def test_rehashed_invalid_span_variants(self):
        self.write()
        path = self.root / self.manifest["files"]["documents"]["path"]
        original = pd.read_parquet(path)
        for field, value in [("vectors__offset", -1), ("vectors__offset", 1),
                             ("vectors__length", -1), ("vectors__length", 2**62),
                             ("vectors__offset", 0.5), ("vectors__length", True)]:
            with self.subTest(field=field, value=value):
                frame = original.copy()
                if isinstance(value, float) or isinstance(value, bool):
                    frame[field] = frame[field].astype(float if isinstance(value, float) else bool)
                frame.loc[frame.document_id == 901, field] = value
                frame.to_parquet(path, index=False)
                self.resign()
                with self.assertRaises(ValueError):
                    read_embeddings(self.root, self.config)

    def test_rehashed_duplicate_and_unexpected_document_keys(self):
        self.write()
        for bad_id in [4, 999, -1]:
            with self.subTest(bad_id=bad_id):
                self.edit_records(lambda frame: frame.loc.__setitem__((frame.document_id == 901, "document_id"), bad_id))
                with self.assertRaises(ValueError):
                    read_embeddings(self.root, self.config)
                self.edit_records(lambda frame: frame.loc.__setitem__((frame.image_path == "outputs/images/901.png", "document_id"), 901))

    def test_rehashed_nullable_shard_cannot_silently_drop_empty_document(self):
        self.write()
        def mutate(frame):
            frame["shard"] = frame["shard"].astype("Int64")
            frame.loc[frame.document_id == 4, "shard"] = pd.NA
        self.edit_records(mutate)
        with self.assertRaises(ValueError):
            read_embeddings(self.root, self.config)

    def test_rehashed_mask_alignment_mismatch(self):
        self.write()
        def mutate(frame):
            frame.loc[frame.document_id == 901, "mask__length"] = 1
            frame.loc[frame.document_id == 4, "mask__offset"] = 1
            frame.loc[frame.document_id == 4, "mask__length"] = 1
        self.edit_records(mutate)
        with self.assertRaises(ValueError):
            read_embeddings(self.root, self.config)

    def test_rehashed_invalid_token_metadata(self):
        self.write()
        path = self.root / self.manifest["files"]["documents"]["path"]
        original = pd.read_parquet(path)
        # Token-index semantics have no common tensor target in v1; test the
        # documented nonempty-object boundary, not an invented encoder schema.
        for bad in [[], {}]:
            with self.subTest(metadata=bad):
                frame = original.copy()
                frame.loc[frame.document_id == 901, "token_metadata"] = json.dumps(bad)
                frame.to_parquet(path, index=False)
                self.resign()
                with self.assertRaises(ValueError):
                    read_embeddings(self.root, self.config)

    def test_writer_rejects_misaligned_input(self):
        self.documents[0]["tensors"]["mask"] = np.array([True])
        with self.assertRaises(ValueError):
            self.write()
        ident, _ = artifact_identity("embeddings", self.config)
        self.assertFalse((self.root / "outputs" / "metadata" / ident / "artifact.json").exists())

    def test_interrupted_iterable_is_unpublished_and_retryable(self):
        def interrupted():
            yield self.documents[0]
            raise InterruptedError("injected after first real shard write")
        ident, _ = artifact_identity("embeddings", self.config)
        with self.assertRaises(InterruptedError):
            self.write(interrupted(), shard_size=1)
        metadata = self.root / "outputs" / "metadata" / ident
        self.assertFalse((metadata / "artifact.json").exists())
        self.assertFalse((metadata / "writer.lock").exists())
        self.assertTrue(list((self.root / "outputs" / "embeddings" / ident).glob("*.npy")))
        with self.assertRaises(ValueError):
            read_embeddings(self.root, self.config)
        self.write(shard_size=1)
        self.assert_documents(read_embeddings(self.root, self.config))

    def test_producer_reused_buffers_are_snapshotted(self):
        self.documents = [self.document(i, [[value, -value]], [True], [i], [[[1, 2], [3, 4]]])
                          for i, value in [(901, 1.25), (4, 8.5), (72, -16.0)]]
        shared = copy.deepcopy(self.documents[0])
        def producer():
            for expected in self.documents:
                shared["document_id"] = expected["document_id"]
                for name, values in expected["tensors"].items():
                    shared["tensors"][name][...] = values
                yield shared
            # Mutation when the iterator finishes must not change saved buffers.
            for values in shared["tensors"].values():
                values[...] = 0
        self.write(producer(), shard_size=10)
        self.assert_documents(read_embeddings(self.root, self.config))

    def test_single_writer_and_immutable_completion(self):
        config = {"fixture": "exclusive"}
        with ArtifactWriter(self.root, "metrics", config) as writer:
            with self.assertRaises(ValueError):
                with ArtifactWriter(self.root, "metrics", config):
                    pass
            writer.json("summary", {"value": 0})
            writer.complete(self.provenance)
        before = writer.metadata_path.read_bytes()
        with self.assertRaises(ValueError):
            with ArtifactWriter(self.root, "metrics", config):
                pass
        self.assertEqual(writer.metadata_path.read_bytes(), before)

    def test_config_snapshot_and_rejection_of_incompatible_config(self):
        config = {"fixture": {"preprocess": [1, 2]}}
        expected = copy.deepcopy(config)
        writer = ArtifactWriter(self.root, "metrics", config)
        config["fixture"]["preprocess"][0] = 90
        with writer:
            writer.json("summary", {"value": 0})
            writer.complete(self.provenance)
        self.assertEqual(read_artifact(self.root, writer.artifact_id)["config"], expected)
        with self.assertRaises(ValueError):
            read_artifact(self.root, writer.artifact_id, expected_config=config)

    def test_cache_identity_changes_and_ocr_separation(self):
        base = artifact_identity("embeddings", self.config)
        mutations = [lambda k: k["processor"].update(size=[9, 9]),
                     lambda k: k["documents"][0].update(image_sha256="f" * 64),
                     lambda k: k["model"].update(revision="e" * 40),
                     lambda k: k.update(source_digest="d" * 64),
                     lambda k: k["transformations"].update(image={"kind": "resize"})]
        for mutate in mutations:
            kwargs = copy.deepcopy(self.kwargs)
            mutate(kwargs)
            self.assertNotEqual(base, artifact_identity("embeddings", embedding_config(**kwargs)))
        first = {"artifact_id": "actual-ocr-fixture", "content_sha256": "1" * 64,
                 "engine": "fixture", "engine_version": "1", "config": {"language": "en"}}
        second = {**first, "content_sha256": "2" * 64}
        self.assertEqual(base, artifact_identity("embeddings", embedding_config(**self.kwargs, ocr=first)))
        self.assertEqual(base, artifact_identity("embeddings", embedding_config(**self.kwargs, ocr=second)))
        kwargs = {**self.kwargs, "input_mode": "image_and_ocr"}
        self.assertNotEqual(artifact_identity("embeddings", embedding_config(**kwargs, ocr=first)),
                            artifact_identity("embeddings", embedding_config(**kwargs, ocr=second)))

    def test_source_files_change_identity_generated_content_does_not(self):
        before = source_identity(self.source)["source_digest"]
        for folder in ["cache", "outputs", "reports"]:
            (self.source / folder).mkdir()
            (self.source / folder / "generated.py").write_text("noise", encoding="utf-8")
        self.assertEqual(source_identity(self.source)["source_digest"], before)
        (self.source / "src" / "untracked.py").write_text("X = 3\n", encoding="utf-8")
        after = source_identity(self.source)["source_digest"]
        self.assertNotEqual(after, before)
        (self.source / "src" / "fixture.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.assertNotEqual(source_identity(self.source)["source_digest"], after)

    def test_changed_source_blocks_publication(self):
        (self.source / "src" / "fixture.py").write_text("VALUE = 99\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.write()
        ident, _ = artifact_identity("embeddings", self.config)
        self.assertFalse((self.root / "outputs" / "metadata" / ident / "artifact.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
