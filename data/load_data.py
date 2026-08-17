"""Download the OmniAI OCR Benchmark and parse its labels into a flat CSV.

Usage:
    python data/load_data.py
    python data/load_data.py --summary
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "getomni-ai/ocr-benchmark"
DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / "raw"
LABELS_CSV = DATA_DIR / "labels.csv"

# The head classes carry ~50 examples each; everything else is a long tail down to
# single examples. See Challenge-Project-Overview.md before changing this threshold.
HEAD_CLASS_MIN_COUNT = 30


def build_labels() -> pd.DataFrame:
    ds = load_dataset(DATASET, split="test", cache_dir=str(CACHE_DIR))

    rows = []
    for record_id, metadata in zip(ds["id"], ds["metadata"]):
        meta = json.loads(metadata)
        rows.append(
            {
                "id": record_id,
                # "SCANNED_TABLE " ships with a trailing space and would otherwise
                # split into a second class.
                "format": meta.get("format", "").strip().upper(),
                "document_quality": meta.get("documentQuality", "").strip().upper(),
            }
        )

    df = pd.DataFrame(rows)
    counts = df["format"].value_counts()
    df["is_head_class"] = df["format"].map(counts) >= HEAD_CLASS_MIN_COUNT
    return df


def print_summary(df: pd.DataFrame) -> None:
    counts = df["format"].value_counts()
    head = counts[counts >= HEAD_CLASS_MIN_COUNT]
    tail = counts[counts < HEAD_CLASS_MIN_COUNT]

    print(f"\n{len(df)} documents, {len(counts)} distinct format labels\n")

    print(f"HEAD — {len(head)} classes, {head.sum()} documents")
    for label, n in head.items():
        print(f"  {label:<30} {n}")

    print(f"\nTAIL — {len(tail)} classes, {tail.sum()} documents")
    for label, n in tail.items():
        print(f"  {label:<30} {n}")

    print("\nDocument quality:")
    for label, n in df["document_quality"].value_counts().items():
        print(f"  {label:<30} {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="print the class distribution")
    args = parser.parse_args()

    df = build_labels()
    df.to_csv(LABELS_CSV, index=False)
    print(f"Wrote {LABELS_CSV} ({len(df)} rows)")

    if args.summary:
        print_summary(df)


if __name__ == "__main__":
    main()
