"""Prepare actual OCR from the Step 2 image manifest and save inspection overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.dont_write_bytecode = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared Step 2 manifest.parquet")
    parser.add_argument("--cohort", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--limit", type=int, help="Optional bounded smoke sample, in ascending document ID order")
    parser.add_argument("--config", type=Path, help="JSON OCR options; defaults to English LSTM OCR, PSM 3")
    parser.add_argument("--overlay-count", type=int, default=3)
    args = parser.parse_args()
    from src.dataset import select_cohort
    from src.embeddings.ocr import prepare_ocr, render_ocr_overlay
    from src.embeddings.pipeline import load_documents
    import pandas as pd

    try:
        if args.limit is not None and args.limit < 1 or args.overlay_count < 0:
            raise ValueError("limit must be positive; overlay-count must be nonnegative")
        manifest = pd.read_parquet(args.manifest)
        ids = select_cohort(manifest, args.cohort)
        if args.limit is not None:
            ids = ids[:args.limit]
        _, documents = load_documents(args.storage_root, args.manifest.parent, ids)
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        identity, records = prepare_ocr(args.storage_root, documents, source_root=EXPERIMENT_ROOT, config=config)
        overlays = [str(render_ocr_overlay(args.storage_root, record, identity["artifact_id"]))
                    for record in list(records.values())[:args.overlay_count]]
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Actual OCR preparation failed: {error}\n")
    print(json.dumps({"identity": identity, "documents": len(records),
                      "word_counts": {str(key): len(record["words"]) for key, record in records.items()},
                      "empty_documents": [key for key, record in records.items() if record["status"] == "empty"],
                      "overlays": overlays}, indent=2))


if __name__ == "__main__":
    main()
