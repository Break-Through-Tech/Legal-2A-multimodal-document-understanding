"""Compare the pooled arm against the retained arm on one embedding cache."""

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


def discover(storage_root: Path, kind: str, hint: str | None) -> str:
    if hint:
        return hint
    found = sorted(path.parent.name for path in (storage_root / "outputs/metadata").glob(f"{kind}-*/artifact.json"))
    if not found:
        raise SystemExit(
            f"No {kind} artifact found under outputs/metadata/.\n"
            "Produce the similarity matrix first:\n"
            "  python experiments/run_clustering.py --models colpali --cohort all --similarity-device cuda:0"
        )
    if len(found) > 1:
        raise SystemExit(f"Several {kind} artifacts found; pass one explicitly:\n  " + "\n  ".join(found))
    return found[0]


def load_split(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"id", "split", label_column} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    if frame.id.duplicated().any():
        raise SystemExit(f"{path} has duplicate document IDs")
    return frame.set_index("id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--embedding-id", help="Embedding artifact ID; auto-detected when unique")
    parser.add_argument("--similarity-id", help="Similarity artifact ID; auto-detected when unique")
    parser.add_argument("--split", type=Path, default=EXPERIMENT_ROOT.parent / "data/split.csv")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--query-split", default="test", choices=["test", "val"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--C", type=float, nargs="+", default=[1.0], help="Regularization for logistic regression")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from src.artifacts import read_artifact
    from src.classification.baselines import (
        DEFINITION as POOLING,
        PooledClassifier,
        pooled_bytes,
        pooled_matrix,
        retained_bytes,
    )
    from src.classification.knn import DEFINITION as VOTE, macro_f1, topk_labels
    from src.dataset import validate_storage_root
    from src.embeddings import read_embeddings

    root = validate_storage_root(args.storage_root)
    frame = load_split(args.split, args.label_column)

    embedding_id = discover(root, "embeddings", args.embedding_id)
    similarity_id = discover(root, "similarities", args.similarity_id)
    config = json.loads((root / "outputs/metadata" / embedding_id / "config.json").read_text(encoding="utf-8"))
    documents = read_embeddings(root, config)
    similarity = read_artifact(root, similarity_id)

    order = similarity["tables"]["ordered_ids"].sort_values("position").document_id.tolist()
    position = {document_id: index for index, document_id in enumerate(order)}
    absent = [i for i in order if i not in frame.index]
    if absent:
        raise SystemExit(f"{len(absent)} artifact documents are absent from {args.split}")

    query_ids = sorted(frame.index[frame.split == args.query_split])
    neighbour_ids = sorted(frame.index[frame.split == "train"])
    if set(query_ids) & set(neighbour_ids):
        raise SystemExit("Query and training splits overlap; a document cannot be its own neighbour")
    truth = [str(frame.at[i, args.label_column]) for i in query_ids]
    neighbour_labels = {i: str(frame.at[i, args.label_column]) for i in neighbour_ids}
    reference = {i: str(frame.at[i, args.label_column]) for i in query_ids}

    rows = []

    started = time.perf_counter()
    pooled_train = pooled_matrix(documents, neighbour_ids)
    pooled_query = pooled_matrix(documents, query_ids)
    pooled_seconds = time.perf_counter() - started
    for method in ("logistic_regression", "nearest_centroid"):
        cs = args.C if method == "logistic_regression" else [1.0]
        for C in cs:
            started = time.perf_counter()
            model = PooledClassifier(method=method, C=C).fit(pooled_train, neighbour_labels.values())
            predicted = model.predict(pooled_query)
            seconds = time.perf_counter() - started
            score, per_label = macro_f1(truth, predicted)
            rows.append({
                "arm": "pooled",
                "method": method if method == "nearest_centroid" else f"{method} (C={C:g})",
                "representation": f"mean-pool -> 1 x {pooled_train.shape[1]}",
                "macro_f1": score,
                "accuracy": sum(t == p for t, p in zip(truth, predicted)) / len(truth),
                "fit_predict_seconds": round(seconds, 4),
                "per_label": per_label,
            })

    matrix = np.asarray(similarity["arrays"]["directed"])
    if matrix.shape != (len(order), len(order)):
        raise SystemExit(f"directed matrix is {matrix.shape} but {len(order)} IDs were saved")
    block = matrix[np.ix_([position[i] for i in query_ids], [position[i] for i in neighbour_ids])]
    tokens = int(np.median([len(documents[i]["tensors"]["vectors"]) for i in query_ids]))
    for k in args.k:
        started = time.perf_counter()
        predicted = topk_labels(block, query_ids, neighbour_ids, neighbour_labels, k)
        seconds = time.perf_counter() - started
        score, per_label = macro_f1(truth, predicted)
        rows.append({
            "arm": "retained",
            "method": f"top-{k} vote",
            "representation": f"{tokens} x {pooled_train.shape[1]}",
            "macro_f1": score,
            "accuracy": sum(t == p for t, p in zip(truth, predicted)) / len(truth),
            "fit_predict_seconds": round(seconds, 4),
            "per_label": per_label,
        })

    # Token vectors live in the embedding artifact, not the similarity one.
    embedding_manifest = json.loads(
        (root / "outputs/metadata" / embedding_id / "artifact.json").read_text(encoding="utf-8")
    )
    retained = retained_bytes(embedding_manifest)
    pooled_total = pooled_bytes(len(documents), pooled_train.shape[1])
    similarity_seconds = similarity["manifest"]["provenance"]["runtime"]["elapsed_seconds"]

    best_pooled = max(row["macro_f1"] for row in rows if row["arm"] == "pooled")
    best_retained = max(row["macro_f1"] for row in rows if row["arm"] == "retained")

    width = max(len(row["method"]) for row in rows) + 2
    print(f"\nrepresentation comparison   embedding={embedding_id}   similarity={similarity_id}")
    print(f"labels={args.label_column}   train={len(neighbour_ids)}   {args.query_split}={len(query_ids)}")
    print()
    print(f"{'method':<{width}}{'representation':<24}{'macro F1':>9}{'accuracy':>10}")
    print("-" * (width + 43))
    for row in rows:
        print(f"{row['method']:<{width}}{row['representation']:<24}{row['macro_f1']:>9.4f}{row['accuracy']:>10.4f}")
    print("-" * (width + 43))
    print(f"best pooled   : {best_pooled:.4f}")
    print(f"best retained : {best_retained:.4f}")
    print(f"delta         : {best_retained - best_pooled:+.4f}")
    print()
    print(f"vectors per document   : 1 vs {tokens}  ({tokens}x)")
    print(f"vector bytes, retained : {retained / 1048576:9.1f} MiB   ({len(documents)} documents)")
    print(f"vector bytes, pooled   : {pooled_total / 1048576:9.3f} MiB")
    print(f"storage ratio          : {retained / pooled_total:9.1f}x")
    print(f"similarity matrix cost : {similarity_seconds:.1f} s  ({len(order) * (len(order) - 1) // 2} pairs)")
    print(f"pooling cost           : {pooled_seconds:.1f} s  ({len(documents)} documents)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "embedding_artifact": embedding_id,
            "embedding_identity_sha256": similarity["config"]["embedding_identity"],
            "similarity_artifact": similarity_id,
            "split_file": str(args.split),
            "label_column": args.label_column,
            "query_split": args.query_split,
            "query_count": len(query_ids),
            "train_count": len(neighbour_ids),
            "rows": rows,
            "cost": {
                "tokens_per_document_median": tokens,
                "retained_vector_bytes": retained,
                "pooled_vector_bytes": pooled_total,
                "storage_ratio": retained / pooled_total,
                "similarity_elapsed_seconds": similarity_seconds,
                "pooling_elapsed_seconds": pooled_seconds,
            },
            "definitions": {"pooling": POOLING, "vote": VOTE},
        }, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
