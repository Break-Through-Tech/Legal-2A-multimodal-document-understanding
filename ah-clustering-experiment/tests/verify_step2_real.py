"""Rebuild and repair an isolated dataset from the actual cached pinned source.

Never corrupt production artifacts. Expected labels/splits come from the shared
CSV; the independent verifier compares every original image byte to source Arrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from datasets import Dataset, Image as DatasetImage

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from src.dataset import prepare_dataset
from verify_real_dataset import verify


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arrow", required=True, type=Path)
    args = parser.parse_args()
    source_path = args.source_arrow.resolve()
    config = json.loads((EXPERIMENT / "configs/dataset.json").read_text(encoding="utf-8"))
    shared_csv = EXPERIMENT.parent / "data/split.csv"
    before_csv = hashlib.sha256(shared_csv.read_bytes()).hexdigest()
    source = Dataset.from_file(str(source_path)).cast_column("image", DatasetImage(decode=False))
    scratch = EXPERIMENT / "cache/s2-real"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="r-", dir=scratch) as temporary:
            root = Path(temporary) / "ah-clustering-experiment"
            # Reverse upstream iteration: a positional join cannot pass the verifier.
            def real_records():
                for record in source.select(list(reversed(range(len(source))))):
                    encoded = record["image"]
                    payload = encoded.get("bytes")
                    if payload is None:
                        # The source Arrow may reference files in the original
                        # cache. Supply their exact bytes to the isolated writer.
                        payload = Path(encoded["path"]).read_bytes()
                    yield {"id": record["id"], "metadata": record["metadata"],
                           "image": {"bytes": payload}}

            print("Preparing isolated original images from pinned source...", file=sys.stderr, flush=True)
            fresh = prepare_dataset(config, root, shared_csv, source_records=real_records())
            assert fresh["reused"] is False and fresh["document_count"] == 1000

            def unavailable():
                raise AssertionError("A valid cache must not request upstream source records")
                yield

            print("Checking source-unavailable reuse and every original image...", file=sys.stderr, flush=True)
            all_docs = prepare_dataset(config, root, shared_csv, source_records=unavailable(), cohort="all")
            assert all_docs["reused"] is True
            check_args = argparse.Namespace(storage_root=root, manifest=Path(fresh["manifest"]),
                metadata=Path(fresh["metadata"]), train_cohort=Path(fresh["cohort"]),
                all_cohort=Path(all_docs["cohort"]), source_arrow=source_path)
            fresh_check = verify(check_args)

            print("Checking real-image corruption and interrupted repair...", file=sys.stderr, flush=True)
            # Corrupt only the isolated copy and prove the independent oracle notices.
            import pandas as pd
            image_path = root / pd.read_parquet(fresh["manifest"]).iloc[0].image_path
            original = image_path.read_bytes()
            image_path.write_bytes(b"step2 audit: deliberately corrupted encoded image")
            detected = False
            try:
                verify(check_args)
            except AssertionError as error:
                assert "Original image bytes changed" in str(error), str(error)
                detected = True
            assert detected, "The real-image verifier missed deliberate corruption"

            # Source interruption cannot leave a valid completion marker for repair.
            def interrupted():
                yield next(real_records())
                raise InterruptedError("step2 audit: interrupted upstream source")

            try:
                prepare_dataset(config, root, shared_csv, source_records=interrupted())
            except InterruptedError:
                pass
            else:
                raise AssertionError("Interrupted repair unexpectedly succeeded")
            assert not Path(fresh["metadata"]).exists(), "Interrupted repair retained completion"
            print("Rebuilding after interruption and comparing all images again...", file=sys.stderr, flush=True)
            repaired = prepare_dataset(config, root, shared_csv, source_records=real_records())
            assert repaired["reused"] is False
            assert image_path.read_bytes() == original
            repair_check = verify(check_args)
            assert hashlib.sha256(shared_csv.read_bytes()).hexdigest() == before_csv
            result = {"status": "PASS", "fresh_preparation": fresh_check,
                      "source_unavailable_cache_reuse": True,
                      "real_image_corruption_detected": detected,
                      "interrupted_repair_left_no_completion": True,
                      "full_repair": repair_check,
                      "shared_csv_unchanged": True,
                      "production_artifacts_modified": False,
                      "elapsed_seconds": round(time.perf_counter() - started, 3)}
    finally:
        os.environ.clear()
        os.environ.update(environment)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
