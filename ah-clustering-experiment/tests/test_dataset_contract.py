"""Independent acceptance checks grounded in data/split.csv and data/README.md.

The source fixture below injects failures without downloading images. It does not
certify Hugging Face loading or actual image persistence; verify_real_dataset.py
checks those outputs separately after the real preparation command has run.
"""
from __future__ import annotations

import json
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.dataset import join_source_records, prepare_dataset, select_cohort, validate_assignments


class SharedManifestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = pd.read_csv(REPOSITORY_ROOT / "data" / "split.csv")

    def source_fixture(self):
        # Distinct payloads make an accidental positional image join observable.
        return [
            {
                "id": int(row.id),
                "metadata": json.dumps({
                    "format": " " + row.original_label.lower() + " ",
                    "documentQuality": " " + row.document_quality.lower() + " ",
                }),
                "image": {"bytes": f"image-for-{row.id}".encode(), "path": None},
                "true_markdown_output": f"REFERENCE ONLY {row.id}",
                "true_json_output": json.dumps({"reference": int(row.id)}),
                "json_schema": '{"type":"object"}',
            }
            for row in self.shared.itertuples(index=False)
        ]

    def test_shared_csv_matches_documented_contract(self):
        self.assertEqual(len(self.shared), 1000)
        self.assertEqual(set(self.shared.id), set(range(1000)))
        self.assertEqual(self.shared.id.nunique(), 1000)
        self.assertEqual(self.shared.label.nunique(), 19)
        self.assertEqual(self.shared.split.value_counts().to_dict(),
                         {"train": 700, "val": 150, "test": 150})
        self.assertEqual((self.shared.label == "Unknown").sum(), 75)
        for partition in ("train", "val", "test"):
            self.assertEqual(self.shared[self.shared.split == partition].label.nunique(), 19)
        result = validate_assignments(self.shared)
        self.assertEqual(set(result.id), set(self.shared.id))

    def test_join_is_by_id_after_source_and_csv_permutations(self):
        source = list(reversed(self.source_fixture()))
        assignments = self.shared.sample(frac=1, random_state=91)
        records = join_source_records(assignments, source)
        expected = self.shared.set_index("id")
        self.assertEqual(len(records), 1000)
        self.assertEqual({record["document_id"] for record in records}, set(expected.index))
        for record in records:
            doc_id = record["document_id"]
            for column in ("split", "label", "original_label", "document_quality"):
                self.assertEqual(record[column], expected.at[doc_id, column])
            self.assertEqual(record["image"]["bytes"], f"image-for-{doc_id}".encode())

    def test_duplicate_assignment_ids_fail(self):
        bad = self.shared.copy()
        bad.loc[1, "id"] = bad.loc[0, "id"]
        with self.assertRaises(ValueError):
            validate_assignments(bad)

    def test_missing_assignment_fails(self):
        with self.assertRaises(ValueError):
            validate_assignments(self.shared.iloc[:-1])

    def test_invalid_partition_fails(self):
        bad = self.shared.copy()
        bad.loc[0, "split"] = "train,val"
        with self.assertRaises(ValueError):
            validate_assignments(bad)

    def test_partition_counts_cannot_be_changed(self):
        bad = self.shared.copy()
        bad.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            validate_assignments(bad)

    def test_missing_label_fails(self):
        bad = self.shared.copy()
        bad.loc[0, "label"] = None
        with self.assertRaises(ValueError):
            validate_assignments(bad)

    def test_missing_source_id_fails(self):
        with self.assertRaises(ValueError):
            join_source_records(self.shared, self.source_fixture()[:-1])

    def test_duplicate_source_id_fails(self):
        source = self.source_fixture()
        source[1]["id"] = source[0]["id"]
        with self.assertRaises(ValueError):
            join_source_records(self.shared, source)

    def test_unexpected_source_id_fails(self):
        source = self.source_fixture()
        source[0]["id"] = 1001
        with self.assertRaises(ValueError):
            join_source_records(self.shared, source)

    def test_conflicting_source_format_fails(self):
        source = self.source_fixture()
        source[0]["metadata"] = json.dumps({"format": "PATENT", "documentQuality": "PHOTO"})
        with self.assertRaises(ValueError):
            join_source_records(self.shared, source)

    def test_conflicting_source_quality_fails(self):
        source = self.source_fixture()
        source[0]["metadata"] = json.dumps({"format": "TABLE", "documentQuality": "CLEAN"})
        with self.assertRaises(ValueError):
            join_source_records(self.shared, source)

    def test_default_cohort_contains_only_shared_training_ids(self):
        manifest = pd.DataFrame(join_source_records(self.shared, self.source_fixture()))
        actual = select_cohort(manifest)
        expected = set(self.shared.loc[self.shared.split == "train", "id"])
        self.assertEqual(len(actual), 700)
        self.assertEqual(set(actual), expected)
        holdout = set(self.shared.loc[self.shared.split != "train", "id"])
        self.assertFalse(set(actual) & holdout)

    def test_all_cohort_requires_explicit_selection(self):
        manifest = pd.DataFrame(join_source_records(self.shared, self.source_fixture()))
        self.assertEqual(set(select_cohort(manifest, "all")), set(self.shared.id))
        for partition in ("val", "test"):
            self.assertEqual(set(select_cohort(manifest, partition)),
                             set(self.shared.loc[self.shared.split == partition, "id"]))
        with self.assertRaises(ValueError):
            select_cohort(manifest, "validation")


class PersistenceContract(unittest.TestCase):
    """Actual filesystem/Parquet integration; only the upstream images are fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.environment = os.environ.copy()
        temporary_parent = EXPERIMENT_ROOT / "cache" / "test-dataset"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=temporary_parent)
        cls.root = Path(cls.temporary.name) / "ah-clustering-experiment"
        cls.assignments = REPOSITORY_ROOT / "data" / "split.csv"
        cls.config = {
            "dataset": {"name": "getomni-ai/ocr-benchmark",
                        "revision": "4ed0d95271ca00107726230f7a0944ed9e90d897",
                        "upstream_split": "test"},
            "default_cohort": "train", "label_usage": "evaluation_only",
            "cluster_count_policy": {"mode": "independently_chosen", "n_clusters": None,
                                     "rationale": "No clustering is fitted in dataset preparation."},
        }
        image = io.BytesIO()
        Image.new("RGB", (2, 3), (16, 32, 64)).save(image, format="PNG")
        cls.image_payload = image.getvalue()
        shared = pd.read_csv(cls.assignments)
        cls.source = [
            {"id": int(row.id), "metadata": json.dumps({"format": row.original_label,
             "documentQuality": row.document_quality}), "image": {"bytes": cls.image_payload},
             "true_markdown_output": "This is reference text, never OCR.",
             "true_json_output": '{"sensitive_reference":"target"}', "json_schema": '{}'}
            for row in shared.itertuples(index=False)
        ]
        cls.initial = prepare_dataset(cls.config, cls.root, cls.assignments, cls.source)

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls.environment)
        cls.temporary.cleanup()

    def test_bytes_metadata_and_references_remain_separate(self):
        manifest = pd.read_parquet(self.initial["manifest"])
        self.assertEqual(len(manifest), 1000)
        self.assertFalse({"true_markdown_output", "true_json_output", "json_schema", "ocr", "words"}
                         & set(manifest.columns))
        self.assertEqual(set(manifest.image_sha256), {hashlib.sha256(self.image_payload).hexdigest()})
        self.assertEqual(set(manifest.image_width), {2})
        self.assertEqual(set(manifest.image_height), {3})
        self.assertEqual(set(manifest.image_format), {"PNG"})
        for relative in manifest.image_path:
            self.assertFalse(Path(relative).is_absolute())
            self.assertNotIn("\\", relative)
            resolved = (self.root / relative).resolve()
            self.assertTrue(resolved.is_relative_to(self.root.resolve()))
            self.assertEqual(resolved.read_bytes(), self.image_payload)

    def test_reuse_needs_no_source_iteration(self):
        def unavailable_source():
            raise AssertionError("Valid completed artifacts must not reload upstream images")
            yield
        result = prepare_dataset(self.config, self.root, self.assignments, unavailable_source())
        self.assertTrue(result["reused"])
        self.assertEqual(result["dataset_id"], self.initial["dataset_id"])

    def test_all_cohort_changes_identity_and_declares_exploration(self):
        result = prepare_dataset(self.config, self.root, self.assignments, self.source, cohort="all")
        self.assertEqual(result["dataset_id"], self.initial["dataset_id"])
        self.assertNotEqual(result["cohort_id"], self.initial["cohort_id"])
        cohort = json.loads(Path(result["cohort"]).read_text())
        self.assertEqual(cohort["designation"], "exploratory")
        self.assertEqual(set(cohort["document_ids"]), set(range(1000)))

    def test_changed_cluster_policy_changes_cohort_identity(self):
        changed = json.loads(json.dumps(self.config))
        changed["cluster_count_policy"] = {"mode": "class_count_informed", "n_clusters": 19,
                                            "rationale": "Uses the known number of final classes."}
        result = prepare_dataset(changed, self.root, self.assignments, self.source)
        self.assertEqual(result["dataset_id"], self.initial["dataset_id"])
        self.assertNotEqual(result["cohort_id"], self.initial["cohort_id"])

    def test_corrupted_image_is_detected_and_repaired(self):
        manifest = pd.read_parquet(self.initial["manifest"])
        image_path = self.root / manifest.iloc[0].image_path
        image_path.write_bytes(b"deliberately corrupt image negative control")
        try:
            result = prepare_dataset(self.config, self.root, self.assignments, self.source)
            self.assertFalse(result["reused"])
            self.assertEqual(image_path.read_bytes(), self.image_payload)
        finally:
            image_path.write_bytes(self.image_payload)

    def test_corrupted_manifest_is_detected_and_repaired(self):
        manifest_path = Path(self.initial["manifest"])
        original = manifest_path.read_bytes()
        manifest_path.write_bytes(b"deliberately corrupt parquet negative control")
        try:
            result = prepare_dataset(self.config, self.root, self.assignments, self.source)
            self.assertFalse(result["reused"])
            restored = pd.read_parquet(manifest_path)
            self.assertEqual(len(restored), 1000)
            self.assertEqual(set(restored.document_id), set(range(1000)))
            completion = json.loads(Path(result["metadata"]).read_text())
            self.assertEqual(completion["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        except Exception:
            manifest_path.write_bytes(original)
            raise

    def test_incomplete_metadata_is_rebuilt(self):
        metadata_path = Path(self.initial["metadata"])
        original = metadata_path.read_bytes()
        metadata = json.loads(original)
        metadata["status"] = "incomplete"
        metadata_path.write_text(json.dumps(metadata))
        try:
            result = prepare_dataset(self.config, self.root, self.assignments, self.source)
            self.assertFalse(result["reused"])
            self.assertEqual(json.loads(metadata_path.read_text())["status"], "complete")
        finally:
            metadata_path.write_bytes(original)

    def test_corrupted_metadata_summary_is_rebuilt(self):
        metadata_path = Path(self.initial["metadata"])
        original = metadata_path.read_bytes()
        metadata = json.loads(original)
        metadata["document_count"] = 999
        metadata_path.write_text(json.dumps(metadata))
        try:
            result = prepare_dataset(self.config, self.root, self.assignments, self.source)
            self.assertFalse(result["reused"])
            self.assertEqual(json.loads(metadata_path.read_text())["document_count"], 1000)
        finally:
            metadata_path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
