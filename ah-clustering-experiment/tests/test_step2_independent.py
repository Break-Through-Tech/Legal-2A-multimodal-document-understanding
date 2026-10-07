"""Independent Step 2 oracle: data README, shared CSV, and plan requirements.

Never imports the existing contract tests. Fixtures preserve every real CSV ID
and assignment while giving each ID independently generated, distinct PNG bytes.
This exercises storage, not authenticity of upstream images (a separate gate).
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parent
sys.path.insert(0, str(EXPERIMENT))
from src import dataset as candidate

CSV = REPOSITORY / "data/split.csv"
REVISION = "4ed0d95271ca00107726230f7a0944ed9e90d897"
CSV_SHA = "0f4c5d9e29795744d7eff33e192460a56ae4651b536b1723900fd2fe30e82e0c"
COUNTS = {"train": 700, "val": 150, "test": 150}
EXPECTED_COLUMNS = {"document_id", "split", "label", "original_label", "document_quality",
                    "image_path", "image_sha256", "image_width", "image_height", "image_format",
                    "dataset_revision", "shared_csv_sha256"}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class UnavailableSource:
    def __iter__(self):
        raise RuntimeError("independent source unavailable")


class Step2Independent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = dict(os.environ)
        cls.assignments = pd.read_csv(CSV)
        cls.config = {
            "dataset": {"name": "getomni-ai/ocr-benchmark", "revision": REVISION, "upstream_split": "test"},
            "default_cohort": "train", "label_usage": "evaluation_only",
            "cluster_count_policy": {"mode": "independently_chosen", "n_clusters": None,
                                     "rationale": "Count is explicitly chosen before clustering."},
        }
        cls.images = {}
        cls.source = []
        for row in cls.assignments.to_dict("records"):
            doc_id = row["id"]
            stream = io.BytesIO()
            Image.new("RGB", (2 + doc_id % 3, 3 + doc_id % 5),
                      (doc_id % 256, doc_id // 256, (doc_id * 37) % 256)).save(stream, format="PNG")
            payload = stream.getvalue()
            cls.images[doc_id] = payload
            cls.source.append({"id": doc_id, "image": {"bytes": payload, "path": None},
                               "metadata": json.dumps({"format": " " + row["original_label"].lower() + " ",
                                                       "documentQuality": row["document_quality"].lower()}),
                               "true_markdown_output": "REFERENCE ONLY", "true_json_output": "REFERENCE ONLY"})
        # Independent deterministic permutation makes position-based joins fail.
        cls.source = [cls.source[int(i)] for i in np.random.default_rng(851).permutation(1000)]
        scratch = EXPERIMENT / "cache/independent-step2"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.sandbox = Path(tempfile.mkdtemp(prefix="t", dir=scratch))
        cls.root = cls.sandbox / "ah-clustering-experiment"
        cls.result = candidate.prepare_dataset(cls.config, cls.root, CSV, source_records=cls.source)
        cls.manifest_path = Path(cls.result["manifest"])
        cls.metadata_path = Path(cls.result["metadata"])
        cls.manifest_bytes = cls.manifest_path.read_bytes()
        cls.metadata_bytes = cls.metadata_path.read_bytes()
        cls.manifest = pd.read_parquet(cls.manifest_path)
        cls.metadata = json.loads(cls.metadata_bytes)

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls.env)
        scratch = (EXPERIMENT / "cache/independent-step2").resolve()
        if not cls.sandbox.resolve().is_relative_to(scratch) or cls.sandbox.resolve() == scratch:
            raise RuntimeError("Refusing cleanup outside independent test scratch")
        shutil.rmtree(cls.sandbox)

    @contextmanager
    def altered(self):
        """One process owns this fixture; restore originals even on failed assertions."""
        try:
            yield
        finally:
            self.manifest_path.write_bytes(self.manifest_bytes)
            self.metadata_path.write_bytes(self.metadata_bytes)
            for row in self.manifest.to_dict("records"):
                path = self.root / row["image_path"]
                original = self.images[row["document_id"]]
                if not path.exists() or path.read_bytes() != original:
                    path.write_bytes(original)

    def prepare(self, source=None, **kwargs):
        return candidate.prepare_dataset(self.config, self.root, CSV,
                                         source_records=UnavailableSource() if source is None else source, **kwargs)

    def validate(self):
        return candidate.validate_artifact(self.root, self.manifest_path, self.metadata_path,
                                           self.metadata["identity"], self.assignments)

    def write_manifest_with_fresh_hash(self, frame):
        frame.to_parquet(self.manifest_path, index=False)
        metadata = copy.deepcopy(self.metadata)
        metadata["manifest_sha256"] = sha(self.manifest_path.read_bytes())
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_01_oracle_shared_csv(self):
        self.assertEqual(sha(CSV.read_bytes()), CSV_SHA)
        self.assertEqual(set(self.assignments.id), set(range(1000)))
        self.assertTrue(self.assignments.id.is_unique)
        self.assertEqual(self.assignments.split.value_counts().to_dict(), COUNTS)
        self.assertEqual(self.assignments.label.nunique(), 19)
        self.assertEqual((self.assignments.label == "Unknown").sum(), 75)
        for partition in COUNTS:
            self.assertEqual(self.assignments[self.assignments.split == partition].label.nunique(), 19)

    def test_02_all_1000_original_encoded_bytes_and_metadata_by_id(self):
        self.assertEqual(len(set(self.images.values())), 1000)
        self.assertEqual(set(self.manifest.columns), EXPECTED_COLUMNS)
        self.assertEqual(set(self.manifest.document_id), set(range(1000)))
        self.assertTrue(self.manifest.document_id.is_unique)
        by_id = self.assignments.set_index("id")
        for row in self.manifest.to_dict("records"):
            doc_id = row["document_id"]
            self.assertEqual((self.root / row["image_path"]).read_bytes(), self.images[doc_id])
            self.assertEqual(row["image_sha256"], sha(self.images[doc_id]))
            self.assertEqual(row["dataset_revision"], REVISION)
            self.assertEqual(row["shared_csv_sha256"], CSV_SHA)
            for field in ("split", "label", "original_label", "document_quality"):
                self.assertEqual(row[field], by_id.at[doc_id, field])
            with Image.open(io.BytesIO(self.images[doc_id])) as image:
                self.assertEqual((row["image_width"], row["image_height"], row["image_format"]),
                                 (image.width, image.height, image.format))

    def test_03_all_cohorts_and_offline_reuse(self):
        cohort_ids = set()
        for name, count in {**COUNTS, "all": 1000}.items():
            result = self.prepare(cohort=name)
            self.assertTrue(result["reused"])
            self.assertEqual(result["document_count"], 1000)
            self.assertEqual(result["cohort_count"], count)
            cohort = json.loads(Path(result["cohort"]).read_bytes())
            expected = self.assignments if name == "all" else self.assignments[self.assignments.split == name]
            self.assertEqual(cohort["document_ids"], sorted(expected.id.tolist()))
            self.assertEqual(cohort["designation"], "initial_training_partition" if name == "train" else "exploratory")
            cohort_ids.add(result["cohort_id"])
        self.assertEqual(len(cohort_ids), 4)
        self.assertEqual(self.prepare()["cohort_count"], 700)

    def test_04_shuffled_assignment_and_manifest_order(self):
        shuffled = self.assignments.sample(frac=1, random_state=17)
        joined = candidate.join_source_records(shuffled, self.source)
        self.assertEqual([row["document_id"] for row in joined], list(range(1000)))
        with self.altered():
            self.write_manifest_with_fresh_hash(self.manifest.sample(frac=1, random_state=31))
            self.assertEqual(set(self.validate().document_id), set(range(1000)))
            self.assertTrue(self.prepare()["reused"])

    def test_05_assignment_id_type_boundaries(self):
        for invalid in (True, False, 0.0, "0", -1, None, np.nan):
            with self.subTest(invalid=repr(invalid)):
                frame = self.assignments.astype({"id": object})
                frame.at[0, "id"] = invalid
                with self.assertRaises((ValueError, TypeError)):
                    candidate.validate_assignments(frame)
        frame = self.assignments.astype({"id": object})
        frame["id"] = pd.Series([np.int64(v) for v in frame.id], dtype=object)
        self.assertEqual(len(candidate.validate_assignments(frame)), 1000)

    def test_06_missing_duplicate_and_malformed_assignments(self):
        mutations = [lambda f: f.iloc[:-1], lambda f: pd.concat([f.iloc[:-1], f.iloc[:1]]),
                     lambda f: f.drop(columns="label")]
        for field, value in (("split", "holdout"), ("label", ""), ("original_label", None),
                             ("document_quality", " CLEAN"), ("label", "Unknown ")):
            def mutate(f, field=field, value=value):
                f.at[0, field] = value
                return f
            mutations.append(mutate)
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises((ValueError, TypeError)):
                candidate.validate_assignments(mutation(self.assignments.copy()))

    def test_07_source_id_and_metadata_negative_controls(self):
        for invalid in (True, 0.0, "0", None, -1, 1000):
            records = copy.deepcopy(self.source)
            records[0]["id"] = invalid
            with self.subTest(id=repr(invalid)), self.assertRaises((ValueError, TypeError)):
                candidate.join_source_records(self.assignments, records)
        for mutation in ("missing", "duplicate", "metadata", "format", "quality", "image"):
            records = copy.deepcopy(self.source)
            if mutation == "missing": records.pop()
            elif mutation == "duplicate": records[-1] = records[0]
            elif mutation == "metadata": records[0]["metadata"] = None
            elif mutation == "image": del records[0]["image"]
            else:
                metadata = json.loads(records[0]["metadata"])
                metadata["format" if mutation == "format" else "documentQuality"] = "WRONG"
                records[0]["metadata"] = metadata
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, TypeError)):
                candidate.join_source_records(self.assignments, records)

    def test_08_config_invalidity(self):
        cases = [("dataset", "revision", "main"), ("dataset", "name", "another/dataset"),
                 ("dataset", "upstream_split", "train"), (None, "label_usage", "fit"),
                 (None, "default_cohort", "holdout"), ("cluster_count_policy", "mode", "automatic"),
                 ("cluster_count_policy", "rationale", " "), ("cluster_count_policy", "n_clusters", True),
                 ("cluster_count_policy", "n_clusters", 1), ("cluster_count_policy", "n_clusters", 2.0)]
        for section, field, value in cases:
            config = copy.deepcopy(self.config)
            (config if section is None else config[section])[field] = value
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, TypeError)):
                candidate.validate_config(config)
        config = copy.deepcopy(self.config)
        config["cluster_count_policy"].update(mode="class_count_informed", n_clusters=18)
        with self.assertRaises(ValueError): candidate.validate_config(config)
        config["cluster_count_policy"]["n_clusters"] = 19
        candidate.validate_config(config)

    def test_09_images_and_manifest_corruption_rejected(self):
        image_path = self.root / self.manifest.iloc[0].image_path
        for mutation in ("missing_image", "broken_image", "changed_image", "manifest_bytes"):
            with self.subTest(mutation=mutation), self.altered():
                if mutation == "missing_image": image_path.unlink()
                elif mutation == "broken_image": image_path.write_bytes(b"broken")
                elif mutation == "changed_image": image_path.write_bytes(self.images[1])
                else: self.manifest_path.write_bytes(b"not parquet")
                with self.assertRaises((ValueError, OSError)): self.validate()

    def test_10_semantically_invalid_manifest_even_with_recomputed_hash(self):
        cases = [("document_id", 1), ("split", "val"), ("label", "WRONG"),
                 ("original_label", "WRONG"), ("document_quality", "WRONG"),
                 ("dataset_revision", "main"), ("shared_csv_sha256", "0" * 64),
                 ("image_width", -1), ("image_path", "../outside.png"), ("image_sha256", "0" * 64)]
        for field, value in cases:
            with self.subTest(field=field), self.altered():
                frame = self.manifest.copy()
                frame.at[0, field] = value
                self.write_manifest_with_fresh_hash(frame)
                with self.assertRaises((ValueError, OSError)): self.validate()

    def test_11_required_metadata_and_provenance_not_silently_accepted(self):
        fields = ("identity", "manifest_sha256", "document_count", "partition_counts", "final_label_count",
                  "reference_text_policy", "limitations", "preparation_provenance", "dependencies", "created_at", "python_version")
        for field in fields:
            with self.subTest(missing=field), self.altered():
                metadata = copy.deepcopy(self.metadata)
                del metadata[field]
                self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises((ValueError, TypeError)): self.validate()

    def test_12_null_sidecar_repairs(self):
        with self.altered():
            self.metadata_path.write_text("null", encoding="utf-8")
            result = self.prepare(self.source)
            self.assertFalse(result["reused"])
            self.assertEqual(json.loads(self.metadata_path.read_bytes())["status"], "complete")
            self.assertEqual(len(self.validate()), 1000)

    def test_13_interrupted_repair_removes_completion_marker(self):
        def interrupted():
            yield self.source[0]
            raise RuntimeError("independent interrupted source")
        with self.altered():
            self.manifest_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "independent interrupted source"):
                self.prepare(interrupted())
            self.assertFalse(self.metadata_path.exists())
            repaired = self.prepare(self.source)
            self.assertFalse(repaired["reused"])
            self.assertEqual(len(self.validate()), 1000)

    def test_14_invalid_image_source_fails_without_marker(self):
        with self.altered():
            self.manifest_path.write_bytes(b"corrupt")
            source = copy.deepcopy(self.source)
            source[0]["image"] = {"bytes": b"not an image"}
            with self.assertRaises((ValueError, OSError)):
                self.prepare(source)
            self.assertFalse(self.metadata_path.exists())

    def test_15_unavailable_source_after_corruption_leaves_no_marker(self):
        with self.altered():
            self.manifest_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "independent source unavailable"):
                self.prepare()
            self.assertFalse(self.metadata_path.exists())

    def test_16_sidecar_provenance_matches_independent_files(self):
        self.assertEqual(self.metadata["identity"]["dataset"], self.config["dataset"])
        self.assertEqual(self.metadata["identity"]["shared_csv_sha256"], CSV_SHA)
        provenance = self.metadata["preparation_provenance"]
        for relative, digest in provenance["source_files_sha256"].items():
            self.assertEqual(sha((EXPERIMENT / relative).read_bytes()), digest)
        self.assertIn("src/dataset.py", provenance["source_files_sha256"])
        self.assertIn("experiments/prepare_dataset.py", provenance["source_files_sha256"])
        self.assertTrue(provenance["git_commit"])
        self.assertIn("not actual OCR", self.metadata["reference_text_policy"])
        self.assertEqual(len(self.metadata["limitations"]), 4)

    def test_17_required_policy_repair_and_unavailable_source(self):
        for field in ("limitations", "preparation_provenance", "reference_text_policy"):
            with self.subTest(field=field), self.altered():
                metadata = copy.deepcopy(self.metadata)
                del metadata[field]
                self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "independent source unavailable"):
                    self.prepare()
                self.assertFalse(self.metadata_path.exists())
                self.assertFalse(self.prepare(self.source)["reused"])
                self.assertIn(field, json.loads(self.metadata_path.read_bytes()))

    def test_18_nonobject_sidecar_and_null_unavailable_source(self):
        for payload in ("null", "[]", "true", '"complete"', "42", "{truncated"):
            with self.subTest(payload=payload), self.altered():
                self.metadata_path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "independent source unavailable"):
                    self.prepare()
                self.assertFalse(self.metadata_path.exists())

    def test_19_semantically_invalid_provenance(self):
        cases = [(None, "dependencies", None), (None, "preparation_provenance", None),
                 (None, "reference_text_policy", "Use reference text as actual OCR"),
                 (None, "limitations", []), (None, "created_at", "not-a-timestamp"),
                 (None, "python_version", ""),
                 ("preparation_provenance", "source_digest", "0" * 64),
                 ("preparation_provenance", "source_files_sha256", {}),
                 ("preparation_provenance", "experiment_dirty", "yes"),
                 ("preparation_provenance", "git_commit", "not-a-commit")]
        for section, field, value in cases:
            with self.subTest(field=field), self.altered():
                metadata = copy.deepcopy(self.metadata)
                (metadata if section is None else metadata[section])[field] = value
                self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises((ValueError, TypeError)):
                    self.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
