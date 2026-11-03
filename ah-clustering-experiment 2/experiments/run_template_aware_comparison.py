"""Re-score both arms on a split that keeps template groups on one side.

Groups are connected components of "similarity >= threshold" within one label. The
threshold is swept, not chosen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.dont_write_bytecode = True

SPLITS = ("train", "val", "test")


def discover(storage_root: Path, kind: str, hint: str | None) -> str:
    if hint:
        return hint
    found = sorted(path.parent.name for path in (storage_root / "outputs/metadata").glob(f"{kind}-*/artifact.json"))
    if not found:
        raise SystemExit(f"No {kind} artifact found under outputs/metadata/")
    if len(found) > 1:
        raise SystemExit(f"Several {kind} artifacts found; pass one explicitly:\n  " + "\n  ".join(found))
    return found[0]


def template_groups(similarity: np.ndarray, order: list[int], labels: dict[int, str],
                    threshold: float) -> dict[int, int]:
    """Connected components of 'similarity >= threshold' restricted to one label."""
    position = {document_id: index for index, document_id in enumerate(order)}
    adjacency = np.zeros((len(order), len(order)), dtype=bool)
    by_label: dict[str, list[int]] = {}
    for document_id in order:
        by_label.setdefault(labels[document_id], []).append(position[document_id])
    for indices in by_label.values():
        block = similarity[np.ix_(indices, indices)]
        adjacency[np.ix_(indices, indices)] = block >= threshold
    np.fill_diagonal(adjacency, False)
    count, component = connected_components(csr_matrix(adjacency), directed=False)
    return {document_id: int(component[position[document_id]]) for document_id in order}


def grouped_split(labels: dict[int, str], groups: dict[int, int],
                  fractions: dict[str, float] | None = None) -> dict[int, str]:
    """Assign whole template groups to one side, stratified by label."""
    fractions = fractions or {"train": 0.70, "val": 0.15, "test": 0.15}
    frames = pd.Series(labels).groupby(pd.Series(labels).values).groups
    assignment: dict[int, str] = {}
    for label, members in frames.items():
        members = list(members)
        buckets: dict[int, list[int]] = {}
        for document_id in members:
            buckets.setdefault(groups[document_id], []).append(document_id)
        ordered = sorted(buckets.values(), key=lambda group: (-len(group), group[0]))
        used = {name: 0 for name in SPLITS}
        target = {name: fractions[name] * len(members) for name in SPLITS}
        for group in ordered:
            deficits = {name: target[name] - used[name] for name in SPLITS}
            chosen = max(SPLITS, key=lambda name: (deficits[name], name == "train"))
            for document_id in group:
                assignment[document_id] = chosen
            used[chosen] += len(group)
    return assignment


def leak_rate(groups: dict[int, int], assignment: dict[int, str], side: str = "test") -> dict:
    """Share of query documents that have a same-template document in training."""
    train_groups = {groups[i] for i, s in assignment.items() if s == "train"}
    queries = [i for i, s in assignment.items() if s == side]
    if not queries:
        return {"queries": 0, "with_train_twin": 0, "share": 0.0}
    hits = sum(1 for i in queries if groups[i] in train_groups)
    return {"queries": len(queries), "with_train_twin": hits, "share": hits / len(queries)}


def score_arms(similarity: np.ndarray, order: list[int], labels: dict[int, str],
               assignment: dict[int, str], query_side: str, ks, pooled_features,
               macro_f1, topk_labels) -> list[dict]:
    from src.classification.baselines import PooledClassifier

    position = {document_id: index for index, document_id in enumerate(order)}
    train = sorted(i for i, s in assignment.items() if s == "train")
    query = sorted(i for i, s in assignment.items() if s == query_side)
    if not train or not query:
        return []
    truth = [labels[i] for i in query]
    neighbour_labels = {i: labels[i] for i in train}
    rows = []

    model = PooledClassifier(method="logistic_regression", C=1.0).fit(
        pooled_features(train), [labels[i] for i in train]
    )
    predicted = model.predict(pooled_features(document_ids=query))
    score, _ = macro_f1(truth, predicted)
    rows.append({"arm": "pooled + logistic regression (C=1)", "macro_f1": score,
                 "accuracy": sum(t == p for t, p in zip(truth, predicted)) / len(truth)})

    block = similarity[np.ix_([position[i] for i in query], [position[i] for i in train])]
    for k in ks:
        predicted = topk_labels(block, query, train, neighbour_labels, k)
        score, _ = macro_f1(truth, predicted)
        rows.append({"arm": f"retained + top-{k} vote", "macro_f1": score,
                     "accuracy": sum(t == p for t, p in zip(truth, predicted)) / len(truth)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--embedding-id")
    parser.add_argument("--similarity-id")
    parser.add_argument("--split", type=Path, default=EXPERIMENT_ROOT.parent / "data/split.csv")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from src.artifacts import read_artifact
    from src.classification.baselines import pooled_matrix
    from src.classification.knn import macro_f1, topk_labels
    from src.contracts import document_ids
    from src.dataset import validate_storage_root
    from src.embeddings import read_embeddings

    root = validate_storage_root(args.storage_root)
    frame = pd.read_csv(args.split).set_index("id")
    labels = {int(i): str(frame.at[i, args.label_column]) for i in frame.index}

    embedding_id = discover(root, "embeddings", args.embedding_id)
    similarity_id = discover(root, "similarities", args.similarity_id)
    config = json.loads((root / "outputs/metadata" / embedding_id / "config.json").read_text(encoding="utf-8"))
    documents = read_embeddings(root, config)
    similarity = read_artifact(root, similarity_id)
    order = similarity["tables"]["ordered_ids"].sort_values("position").document_id.tolist()
    matrix = np.asarray(similarity["arrays"]["directed"])
    if matrix.shape != (len(order), len(order)):
        raise SystemExit(f"directed is {matrix.shape} but {len(order)} IDs were saved")

    cache: dict[tuple, np.ndarray] = {}

    def pooled_features(document_ids):
        key = tuple(document_ids)
        if key not in cache:
            cache[key] = pooled_matrix(documents, list(document_ids))
        return cache[key]

    random_assignment = {int(i): str(frame.at[i, "split"]) for i in frame.index}
    random_rows = score_arms(matrix, order, labels, random_assignment, "test", args.k,
                             pooled_features, macro_f1, topk_labels)

    print(f"\nembedding={embedding_id}  similarity={similarity_id}  labels={args.label_column}")
    print(f"random split: train={sum(v == 'train' for v in random_assignment.values())}"
          f"  test={sum(v == 'test' for v in random_assignment.values())}")
    print(f"\n{'arm':<34}{'random split':>14}")
    print("-" * 48)
    for row in random_rows:
        print(f"{row['arm']:<34}{row['macro_f1']:>14.4f}")

    report = {"embedding_artifact": embedding_id, "similarity_artifact": similarity_id,
              "label_column": args.label_column, "random_split": random_rows, "thresholds": []}

    for threshold in args.thresholds:
        groups = template_groups(matrix, order, labels, threshold)
        grouped = grouped_split(labels, groups)
        sizes = pd.Series([groups[i] for i in order]).value_counts()
        random_leak = leak_rate(groups, random_assignment)
        grouped_leak = leak_rate(groups, grouped)
        grouped_rows = score_arms(matrix, order, labels, grouped, "test", args.k,
                                  pooled_features, macro_f1, topk_labels)
        counts = {name: sum(v == name for v in grouped.values()) for name in SPLITS}
        empty = sum(1 for _, members in pd.Series(labels).groupby(pd.Series(labels).values).groups.items()
                    if not any(grouped[i] == "test" for i in members))

        print(f"\n{'=' * 78}\nthreshold {threshold:.2f}   groups={sizes.size}   "
              f"largest={int(sizes.max())}   singletons={int((sizes == 1).sum())}")
        print(f"  grouped split: " + "  ".join(f"{n}={counts[n]}" for n in SPLITS)
              + f"   classes without test docs={empty}")
        print(f"  test docs sharing a template with train:")
        print(f"    random split : {random_leak['with_train_twin']}/{random_leak['queries']}"
              f" = {random_leak['share']:.1%}")
        print(f"    grouped split: {grouped_leak['with_train_twin']}/{grouped_leak['queries']}"
              f" = {grouped_leak['share']:.1%}")
        print(f"  {'arm':<34}{'grouped split':>14}")
        print("  " + "-" * 46)
        for row in grouped_rows:
            print(f"  {row['arm']:<34}{row['macro_f1']:>14.4f}")

        report["thresholds"].append({
            "threshold": threshold, "group_count": int(sizes.size),
            "largest_group": int(sizes.max()), "singletons": int((sizes == 1).sum()),
            "grouped_split_counts": counts, "classes_without_test": int(empty),
            "random_split_leak": random_leak, "grouped_split_leak": grouped_leak,
            "grouped_split": grouped_rows,
        })

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
