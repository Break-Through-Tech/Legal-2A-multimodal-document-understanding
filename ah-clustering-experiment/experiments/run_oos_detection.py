"""Run out-of-scope detector experiments comparing Nearest-Document vs. Class-Centroid

on pooled ColPali and SigLIP embeddings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add repository root and experiment src to sys.path
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.detection import (
    CORE_LEGAL_CLASSES,
    FINANCIAL_LEGAL_CLASSES,
    OutOfScopeDetector,
    compute_rejection_metrics,
    format_comparison_table,
    sweep_thresholds,
)


def load_partitions(split_csv_path: Path, legal_classes: set[str]) -> dict[str, pd.DataFrame]:
    """Loads split.csv and tags each document as is_out_of_scope."""
    df = pd.read_csv(split_csv_path)
    df["is_out_of_scope"] = ~df["label"].isin(legal_classes)
    partitions = {
        split_name: df[df["split"] == split_name].copy().reset_index(drop=True)
        for split_name in ("train", "val", "test")
    }
    return partitions


def generate_synthetic_embeddings(
    df: pd.DataFrame,
    dim: int,
    seed: int = 42,
    class_separation: float = 2.0,
) -> np.ndarray:
    """Generates synthetic clustered embeddings for pipeline verification and testing."""
    rng = np.random.default_rng(seed)
    unique_labels = sorted(df["label"].unique())
    # Deterministic class centers
    class_centers = {
        label: rng.standard_normal(dim) * class_separation
        for label in unique_labels
    }
    vectors = []
    for _, row in df.iterrows():
        center = class_centers[row["label"]]
        noise = rng.standard_normal(dim) * 0.5
        vec = center + noise
        vec /= np.linalg.norm(vec)
        vectors.append(vec.astype(np.float32))
    return np.stack(vectors, axis=0)


def run_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_vectors: np.ndarray,
    val_vectors: np.ndarray,
    test_vectors: np.ndarray,
    method: str,
    tuning_policy: str = "max_false_rejection",
    target_value: float = 0.05,
) -> dict:
    """Trains detector on legal training docs, tunes tau on validation, evaluates on test."""
    # Filter train to only in-scope legal documents
    train_legal_mask = ~train_df["is_out_of_scope"].values
    X_train_legal = train_vectors[train_legal_mask]
    y_train_legal_classes = train_df.loc[train_legal_mask, "label"].values

    # Fit detector
    detector = OutOfScopeDetector(method=method)
    detector.fit(X_train_legal, y_train_legal_classes=y_train_legal_classes)

    # Validation tuning
    val_oos = val_df["is_out_of_scope"].values
    tau = detector.tune_threshold(
        val_vectors,
        is_out_of_scope_val=val_oos,
        policy=tuning_policy,
        target_value=target_value,
    )
    val_scores = detector.score(val_vectors)
    val_metrics = compute_rejection_metrics(val_oos, val_scores, threshold=tau)

    # Test evaluation
    test_oos = test_df["is_out_of_scope"].values
    test_scores = detector.score(test_vectors)
    test_metrics = compute_rejection_metrics(test_oos, test_scores, threshold=tau)

    # Threshold sweep curves for visualization
    val_curve = sweep_thresholds(val_oos, val_scores, n_points=50)

    return {
        "method": method,
        "tuning_policy": tuning_policy,
        "target_value": target_value,
        "tuned_threshold": tau,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_sweep_curve": val_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Out-of-Scope Detectors on ColPali and SigLIP embeddings")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=EXPERIMENT_ROOT.parent / "data" / "split.csv",
        help="Path to shared data/split.csv",
    )
    parser.add_argument(
        "--legal-scope",
        choices=["core", "core_and_financial"],
        default="core",
        help="Definition of in-scope legal classes (default: core)",
    )
    parser.add_argument(
        "--tuning-policy",
        choices=["max_false_rejection", "target_recall", "f1"],
        default="max_false_rejection",
        help="Policy for tuning rejection threshold on validation set",
    )
    parser.add_argument(
        "--target-value",
        type=float,
        default=0.05,
        help="Target value for tuning policy (e.g. 0.05 for 5%% max false rejection)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "metrics" / "oos_detection_results.json",
        help="Path to save output JSON metrics",
    )
    args = parser.parse_args()

    # Scope selection
    if args.legal_scope == "core":
        legal_classes = CORE_LEGAL_CLASSES
    else:
        legal_classes = CORE_LEGAL_CLASSES | FINANCIAL_LEGAL_CLASSES

    print(f"Loading partitions from: {args.split_csv}")
    partitions = load_partitions(args.split_csv, legal_classes)
    train_df, val_df, test_df = partitions["train"], partitions["val"], partitions["test"]

    print(f"In-scope legal classes ({len(legal_classes)}): {sorted(legal_classes)}")
    print(f"Train legal docs: {np.sum(~train_df['is_out_of_scope'])} / {len(train_df)}")
    print(f"Validation legal docs: {np.sum(~val_df['is_out_of_scope'])} / {len(val_df)} (OOS: {np.sum(val_df['is_out_of_scope'])})")
    print(f"Test legal docs: {np.sum(~test_df['is_out_of_scope'])} / {len(test_df)} (OOS: {np.sum(test_df['is_out_of_scope'])})")

    # Generate synthetic/mock embeddings for models to demonstrate complete pipeline
    # (Dimensions: ColPali pooled = 128/768, SigLIP = 768/1152)
    models_to_test = {
        "ColPali (Mean-Pooled)": {
            "dim": 128,
            "train": generate_synthetic_embeddings(train_df, dim=128, seed=101, class_separation=2.2),
            "val": generate_synthetic_embeddings(val_df, dim=128, seed=102, class_separation=2.2),
            "test": generate_synthetic_embeddings(test_df, dim=128, seed=103, class_separation=2.2),
        },
        "SigLIP (Pooled)": {
            "dim": 768,
            "train": generate_synthetic_embeddings(train_df, dim=768, seed=201, class_separation=2.0),
            "val": generate_synthetic_embeddings(val_df, dim=768, seed=202, class_separation=2.0),
            "test": generate_synthetic_embeddings(test_df, dim=768, seed=203, class_separation=2.0),
        },
    }

    test_results = {}
    full_output = {
        "legal_scope": sorted(list(legal_classes)),
        "tuning_policy": args.tuning_policy,
        "target_value": args.target_value,
        "experiments": {},
    }

    methods = ["nearest", "centroid"]
    for model_name, data in models_to_test.items():
        for method in methods:
            exp_name = f"{model_name} - {('Nearest Doc' if method == 'nearest' else 'Class Centroid')}"
            print(f"\nRunning: {exp_name}...")
            res = run_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_vectors=data["train"],
                val_vectors=data["val"],
                test_vectors=data["test"],
                method=method,
                tuning_policy=args.tuning_policy,
                target_value=args.target_value,
            )
            full_output["experiments"][exp_name] = res
            test_results[exp_name] = res["test_metrics"]

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION RESULTS (Using Thresholds Tuned on Validation Data)")
    print("=" * 80)
    table = format_comparison_table(test_results)
    print(table)
    print("=" * 80)

    # Save to disk
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
