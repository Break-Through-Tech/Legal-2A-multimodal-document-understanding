"""Score top-k retrieval classification from a saved similarity artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.dont_write_bytecode = True


def discover_similarity(root: Path) -> str:
    """Return the single available similarity artifact, or explain why not."""
    candidates = sorted(path.parent.name for path in (root / "outputs/metadata").glob("similarities-*/artifact.json"))
    if not candidates:
        raise SystemExit(
            "No similarity artifact found under outputs/metadata/.\n"
            "Produce one first, for example:\n"
            "  python experiments/run_clustering.py --models colpali --cohort all --similarity-device cuda:0"
        )
    if len(candidates) > 1:
        raise SystemExit(
            "Several similarity artifacts found; choose one with --similarity-id:\n  "
            + "\n  ".join(candidates)
        )
    return candidates[0]


def load_split(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "split", label_column}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    if frame.id.duplicated().any():
        raise SystemExit(f"{path} contains duplicate document IDs")
    return frame.set_index("id")


def slice_matrix(matrix: np.ndarray, order: list[int], query_ids: list[int],
                 neighbour_ids: list[int]) -> np.ndarray:
    """Select a query-by-neighbour block, resolving every ID through the saved order."""
    position = {document_id: index for index, document_id in enumerate(order)}
    missing = [i for i in list(query_ids) + list(neighbour_ids) if i not in position]
    if missing:
        raise SystemExit(
            f"{len(missing)} requested documents are absent from the similarity artifact, "
            f"starting with {missing[:5]}.\n"
            "This usually means the artifact was built for a single cohort. Rebuild it with "
            "--cohort all so it covers both the training and evaluation documents."
        )
    rows = np.array([position[i] for i in query_ids])
    columns = np.array([position[i] for i in neighbour_ids])
    return np.asarray(matrix[np.ix_(rows, columns)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--similarity-id", help="Similarity artifact ID; auto-detected when unique")
    parser.add_argument("--matrix", default="directed", choices=["directed", "similarity"],
                        help="directed scores query->neighbour; similarity is the symmetrised average")
    parser.add_argument("--split", type=Path, default=EXPERIMENT_ROOT.parent / "data/split.csv")
    parser.add_argument("--label-column", default="label", help="label (19 classes) or original_label (36 types)")
    parser.add_argument("--query-split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--neighbour-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out", type=Path, help="Write the full result JSON here")
    args = parser.parse_args()

    from src.artifacts import read_artifact
    from src.classification.knn import DEFINITION, evaluate
    from src.dataset import validate_storage_root

    root = validate_storage_root(args.storage_root)
    artifact_id = args.similarity_id or discover_similarity(root)
    bundle = read_artifact(root, artifact_id)
    matrix = bundle["arrays"][args.matrix]
    order = bundle["tables"]["ordered_ids"].sort_values("position").document_id.tolist()
    if matrix.shape != (len(order), len(order)):
        raise SystemExit(f"{args.matrix} is {matrix.shape} but {len(order)} ordered IDs were saved")

    frame = load_split(args.split, args.label_column)
    absent = [i for i in order if i not in frame.index]
    if absent:
        raise SystemExit(f"{len(absent)} artifact documents are absent from {args.split}, starting with {absent[:5]}")

    query_ids = sorted(frame.index[frame.split == args.query_split])
    neighbour_ids = sorted(frame.index[frame.split == args.neighbour_split])
    overlap = set(query_ids) & set(neighbour_ids)
    if overlap:
        raise SystemExit(f"Query and neighbour splits overlap on {len(overlap)} documents; a document cannot vote for itself")
    if not query_ids or not neighbour_ids:
        raise SystemExit(f"Empty split selection: {len(query_ids)} queries, {len(neighbour_ids)} neighbours")

    started = time.perf_counter()
    block = slice_matrix(matrix, order, query_ids, neighbour_ids)
    labels = {i: str(frame.at[i, args.label_column]) for i in order}
    result = evaluate(block, query_ids, {i: labels[i] for i in query_ids}, neighbour_ids,
                      {i: labels[i] for i in neighbour_ids}, k_values=args.k)
    result.update({
        "similarity_artifact": artifact_id,
        "similarity_identity_sha256": bundle["manifest"]["identity_sha256"],
        "matrix": args.matrix,
        "split_file": str(args.split),
        "label_column": args.label_column,
        "query_split": args.query_split,
        "neighbour_split": args.neighbour_split,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "vote_definition": DEFINITION,
    })

    print(f"\nsimilarity artifact : {artifact_id}")
    print(f"matrix              : {args.matrix}")
    print(f"label column        : {args.label_column}")
    print(f"neighbours          : {len(neighbour_ids)} ({args.neighbour_split})")
    print(f"queries             : {len(query_ids)} ({args.query_split})\n")
    header = f"{'k':>3}  {'macro F1':>9}  {'accuracy':>9}"
    print(header)
    print("-" * len(header))
    for row in result["results"]:
        print(f"{row['k']:>3}  {row['macro_f1']:>9.4f}  {row['accuracy']:>9.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
