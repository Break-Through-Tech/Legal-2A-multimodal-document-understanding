"""Paired bootstrap intervals for the retained-minus-pooled difference.

Both arms are scored on the same resampled documents and the models are held fixed,
so this covers test-set variance only. Resampling is stratified by label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.dont_write_bytecode = True


def stratified_index(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    """Resample document positions, preserving each label's original count."""
    by_label: dict[str, list[int]] = {}
    for position, label in enumerate(labels):
        by_label.setdefault(label, []).append(position)
    picks = []
    for positions in by_label.values():
        pool = np.asarray(positions)
        picks.append(rng.choice(pool, size=len(pool), replace=True))
    return np.concatenate(picks)


def unstratified_index(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, count, size=count)


def summarise(samples: np.ndarray, point: float) -> dict:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(low), "ci_high": float(high),
            "mean": float(samples.mean()), "std": float(samples.std(ddof=1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--embedding-id")
    parser.add_argument("--similarity-id")
    parser.add_argument("--split", type=Path, default=EXPERIMENT_ROOT.parent / "data/split.csv")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--mode", default="random", choices=["random", "grouped"])
    parser.add_argument("--threshold", type=float, default=0.85, help="Grouping threshold for --mode grouped")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from experiments.run_template_aware_comparison import grouped_split, template_groups
    from src.artifacts import read_artifact
    from src.classification.baselines import PooledClassifier, pooled_matrix
    from src.classification.knn import macro_f1, topk_labels
    from src.dataset import validate_storage_root
    from src.embeddings import read_embeddings

    root = validate_storage_root(args.storage_root)
    frame = pd.read_csv(args.split).set_index("id")
    labels = {int(i): str(frame.at[i, args.label_column]) for i in frame.index}

    def pick(kind: str, hint: str | None) -> str:
        if hint:
            return hint
        found = sorted(p.parent.name for p in (root / "outputs/metadata").glob(f"{kind}-*/artifact.json"))
        if len(found) != 1:
            raise SystemExit(f"Pass --{kind}-id explicitly; found {found}")
        return found[0]

    embedding_id = pick("embeddings", args.embedding_id)
    similarity_id = pick("similarities", args.similarity_id)
    config = json.loads((root / "outputs/metadata" / embedding_id / "config.json").read_text(encoding="utf-8"))
    documents = read_embeddings(root, config)
    similarity = read_artifact(root, similarity_id)
    order = similarity["tables"]["ordered_ids"].sort_values("position").document_id.tolist()
    position = {document_id: index for index, document_id in enumerate(order)}
    matrix = np.asarray(similarity["arrays"]["directed"])

    if args.mode == "random":
        assignment = {int(i): str(frame.at[i, "split"]) for i in frame.index}
    else:
        groups = template_groups(matrix, order, labels, args.threshold)
        assignment = grouped_split(labels, groups)

    train = sorted(i for i, s in assignment.items() if s == "train")
    query = sorted(i for i, s in assignment.items() if s == "test")
    if not train or not query:
        raise SystemExit("Empty train or test side")
    truth = [labels[i] for i in query]
    label_set = sorted(set(labels[i] for i in order))
    neighbour_labels = {i: labels[i] for i in train}

    pooled_train = pooled_matrix(documents, train)
    pooled_query = pooled_matrix(documents, query)
    block = matrix[np.ix_([position[i] for i in query], [position[i] for i in train])]

    arms: dict[str, list[str]] = {}
    arms["pooled + logistic regression (C=1)"] = PooledClassifier(
        method="logistic_regression", C=1.0).fit(pooled_train, [labels[i] for i in train]
    ).predict(pooled_query)
    arms["pooled + nearest centroid"] = PooledClassifier(
        method="nearest_centroid").fit(pooled_train, [labels[i] for i in train]
    ).predict(pooled_query)
    for k in args.k:
        arms[f"retained + top-{k} vote"] = topk_labels(block, query, train, neighbour_labels, k)

    points = {name: macro_f1(truth, predicted, labels=label_set)[0] for name, predicted in arms.items()}

    rng = np.random.default_rng(args.seed)
    names = list(arms)
    primary = "pooled + logistic regression (C=1)"
    pairs = [f"retained + top-{k} vote" for k in args.k]
    draws = {scheme: {n: np.empty(args.replicates) for n in names} for scheme in ("stratified", "unstratified")}
    diffs = {scheme: {name: np.empty(args.replicates) for name in pairs}
             for scheme in ("stratified", "unstratified")}

    for scheme in ("stratified", "unstratified"):
        for r in range(args.replicates):
            index = (stratified_index(truth, rng) if scheme == "stratified"
                     else unstratified_index(len(truth), rng))
            resampled_truth = [truth[i] for i in index]
            for name in names:
                predicted = [arms[name][i] for i in index]
                score = macro_f1(resampled_truth, predicted, labels=label_set)[0]
                draws[scheme][name][r] = score
            for name in pairs:
                diffs[scheme][name][r] = draws[scheme][name][r] - draws[scheme][primary][r]

    print(f"\n{embedding_id}   mode={args.mode}"
          + (f" (tau={args.threshold:.2f})" if args.mode == "grouped" else "")
          + f"   test={len(query)}   train={len(train)}   replicates={args.replicates}")
    print(f"\n{'arm':<36}{'macro F1':>9}{'95% CI (stratified)':>26}")
    print("-" * 71)
    for name in names:
        s = summarise(draws["stratified"][name], points[name])
        print(f"{name:<36}{s['point']:>9.4f}   [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")

    print(f"\npaired difference against '{primary}'")
    print(f"{'comparison':<30}{'scheme':<14}{'delta':>9}{'95% CI':>22}{'P(>0)':>8}")
    print("-" * 83)
    for name in pairs:
        point = points[name] - points[primary]
        for scheme in ("stratified", "unstratified"):
            samples = diffs[scheme][name]
            s = summarise(samples, point)
            flag = "  spans 0" if s["ci_low"] <= 0 <= s["ci_high"] else ""
            share = float((samples > 0).mean())
            print(f"{name:<30}{scheme:<14}{point:>+9.4f}   [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]"
                  f"{share:>8.3f}{flag}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "embedding_artifact": embedding_id, "similarity_artifact": similarity_id,
            "mode": args.mode, "threshold": args.threshold if args.mode == "grouped" else None,
            "label_column": args.label_column, "replicates": args.replicates, "seed": args.seed,
            "test_count": len(query), "train_count": len(train), "label_set": label_set,
            "points": points,
            "intervals": {scheme: {n: summarise(draws[scheme][n], points[n]) for n in names}
                          for scheme in draws},
            "difference_vs_primary": {name: {
                scheme: {**summarise(diffs[scheme][name], points[name] - points[primary]),
                         "share_greater_than_zero": float((diffs[scheme][name] > 0).mean())}
                for scheme in diffs} for name in pairs},
            "primary_arm": primary,
        }, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
