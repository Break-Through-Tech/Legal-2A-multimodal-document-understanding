"""Read saved artifacts and source Arrow directly, without the candidate reader.

Run with --storage-root EXPERIMENT --manifest PATH --metadata PATH
--train-cohort PATH [--all-cohort PATH] --source-arrow PATH.
Expectations come from the shared CSV, its documentation, and original bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd
from datasets import Dataset, Image as DatasetImage
from PIL import Image

REVISION = "4ed0d95271ca00107726230f7a0944ed9e90d897"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify(args):
    root = args.storage_root.resolve()
    csv_path = REPOSITORY_ROOT / "data" / "split.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        expected = {int(row["id"]): row for row in csv.DictReader(handle)}
    assert len(expected) == 1000
    assert Counter(row["split"] for row in expected.values()) == {"train": 700, "val": 150, "test": 150}
    shared_digest = digest(csv_path.read_bytes())
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["identity"]["dataset"] == {"name": "getomni-ai/ocr-benchmark", "revision": REVISION, "upstream_split": "test"}
    assert metadata["identity"]["shared_csv_sha256"] == shared_digest
    assert metadata["manifest_sha256"] == digest(args.manifest.read_bytes())
    assert metadata["document_count"] == 1000
    assert metadata["final_label_count"] == 19
    assert metadata["partition_counts"] == {"train": 700, "val": 150, "test": 150}
    assert metadata["extraction_scope"] == "all_1000_documents"

    manifest = pd.read_parquet(args.manifest)
    assert len(manifest) == 1000 and manifest.document_id.nunique() == 1000
    assert set(manifest.document_id) == set(expected)
    assert not {"true_markdown_output", "true_json_output", "json_schema", "ocr", "words"} & set(manifest.columns)
    assert set(manifest.dataset_revision) == {REVISION}
    assert set(manifest.shared_csv_sha256) == {shared_digest}

    # Explicit offline source; never invokes load_dataset or calls the network.
    original = Dataset.from_file(str(args.source_arrow)).cast_column("image", DatasetImage(decode=False))
    source = {int(record["id"]): record for record in original}
    assert len(original) == len(source) == 1000 and set(source) == set(expected)
    byte_count = 0
    for row in manifest.itertuples(index=False):
        doc_id = row.document_id
        shared = expected[doc_id]
        for name in ("split", "label", "original_label", "document_quality"):
            assert getattr(row, name) == shared[name], (doc_id, name)
        raw_meta = source[doc_id]["metadata"]
        raw_meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        assert raw_meta["format"].strip().upper() == shared["original_label"]
        assert raw_meta["documentQuality"].strip().upper() == shared["document_quality"]
        relative = PurePosixPath(row.image_path)
        assert not relative.is_absolute() and ".." not in relative.parts and "\\" not in row.image_path
        image_path = (root / row.image_path).resolve()
        assert image_path.is_relative_to(root)
        payload = image_path.read_bytes()
        original_image = source[doc_id]["image"]
        original_payload = original_image.get("bytes")
        if original_payload is None:
            original_payload = Path(original_image["path"]).read_bytes()
        assert payload == original_payload, f"Original image bytes changed for {doc_id}"
        assert digest(payload) == row.image_sha256
        with Image.open(image_path) as image:
            assert (image.width, image.height, image.format) == (row.image_width, row.image_height, row.image_format)
            image.verify()
        byte_count += len(payload)

    train = json.loads(args.train_cohort.read_text(encoding="utf-8"))
    expected_train = {doc_id for doc_id, row in expected.items() if row["split"] == "train"}
    assert len(train["document_ids"]) == 700 and set(train["document_ids"]) == expected_train
    assert train["dataset_id"] == metadata["dataset_id"] and train["cohort"] == "train"
    assert train["designation"] == "initial_training_partition" and train["label_usage"] == "evaluation_only"
    assert "cluster_count_policy" in train
    if args.all_cohort:
        all_docs = json.loads(args.all_cohort.read_text(encoding="utf-8"))
        assert all_docs["dataset_id"] == train["dataset_id"]
        assert all_docs["cohort_id"] != train["cohort_id"]
        assert all_docs["cohort"] == "all" and all_docs["designation"] == "exploratory"
        assert len(all_docs["document_ids"]) == 1000 and set(all_docs["document_ids"]) == set(expected)
    return {"status": "PASS", "documents": 1000, "images_byte_equal_to_pinned_source": 1000,
            "original_image_bytes": byte_count, "labels": 19,
            "partitions": {"train": 700, "val": 150, "test": 150},
            "train_cohort": 700, "all_cohort_checked": bool(args.all_cohort),
            "shared_csv_sha256": shared_digest, "dataset_revision": REVISION}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("storage-root", "manifest", "metadata", "train-cohort", "source-arrow"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--all-cohort", type=Path)
    print(json.dumps(verify(parser.parse_args()), indent=2))
