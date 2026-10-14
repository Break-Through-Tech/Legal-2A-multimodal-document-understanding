"""Run multiclass classification and two-stage intake routing on legal documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.classification import (
    LegalDocumentClassifier,
    TwoStageIntakeRouter,
    compute_multiclass_metrics,
    format_multiclass_table,
)
from src.detection import (
    CORE_LEGAL_CLASSES,
    OutOfScopeDetector,
)
from experiments.run_oos_detection import generate_synthetic_embeddings, load_partitions


def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal Document Multiclass Classification")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=EXPERIMENT_ROOT.parent / "data" / "split.csv",
        help="Path to data/split.csv",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "nearest_centroid"],
        default="logistic_regression",
        help="Classification algorithm",
    )
    args = parser.parse_args()

    partitions = load_partitions(args.split_csv, CORE_LEGAL_CLASSES)
    train_df, val_df, test_df = partitions["train"], partitions["val"], partitions["test"]

    print("=" * 80)
    print("STAGE 1: PURE LEGAL MULTICLASS CLASSIFICATION (Known In-Scope Legal Documents)")
    print("=" * 80)

    # Filter to only legal documents for pure classification benchmark
    train_legal = train_df[~train_df["is_out_of_scope"]].copy().reset_index(drop=True)
    val_legal = val_df[~val_df["is_out_of_scope"]].copy().reset_index(drop=True)
    test_legal = test_df[~test_df["is_out_of_scope"]].copy().reset_index(drop=True)

    print(f"In-scope legal classes: {sorted(CORE_LEGAL_CLASSES)}")
    print(f"Legal Train count: {len(train_legal)} | Val count: {len(val_legal)} | Test count: {len(test_legal)}")

    # Embeddings
    dim = 768
    X_train_legal = generate_synthetic_embeddings(train_legal, dim=dim, seed=301, class_separation=3.0)
    X_val_legal = generate_synthetic_embeddings(val_legal, dim=dim, seed=302, class_separation=3.0)
    X_test_legal = generate_synthetic_embeddings(test_legal, dim=dim, seed=303, class_separation=3.0)

    y_train = train_legal["label"].values
    y_test = test_legal["label"].values

    classifier = LegalDocumentClassifier(method=args.classifier)
    classifier.fit(X_train_legal, y_train)

    y_test_pred = classifier.predict(X_test_legal)
    metrics = compute_multiclass_metrics(y_test, y_test_pred)

    print("\n" + format_multiclass_table(metrics))

    print("\n" + "=" * 80)
    print("STAGE 2: TWO-STAGE INTAKE ROUTER (Reject Out-of-Scope + Classify Legal)")
    print("=" * 80)

    # Detector on full train/val/test
    X_train_full = generate_synthetic_embeddings(train_df, dim=dim, seed=401, class_separation=3.0)
    X_val_full = generate_synthetic_embeddings(val_df, dim=dim, seed=402, class_separation=3.0)
    X_test_full = generate_synthetic_embeddings(test_df, dim=dim, seed=403, class_separation=3.0)

    train_legal_mask = ~train_df["is_out_of_scope"].values
    detector = OutOfScopeDetector(method="nearest")
    detector.fit(X_train_full[train_legal_mask], train_df.loc[train_legal_mask, "label"].values)

    # Tune threshold on validation set
    tau = detector.tune_threshold(
        X_val_full,
        is_out_of_scope_val=val_df["is_out_of_scope"].values,
        policy="max_false_rejection",
        target_value=0.05,
    )
    print(f"Detector threshold tuned on validation: tau = {tau:.4f}")

    # Build router
    router = TwoStageIntakeRouter(detector, classifier)
    routed_test_preds = router.predict(X_test_full)

    # Ground truth: if out of scope, expected label is 'OUT_OF_SCOPE', otherwise exact legal class
    y_test_true_routed = np.where(test_df["is_out_of_scope"].values, "OUT_OF_SCOPE", test_df["label"].values)

    routed_metrics = compute_multiclass_metrics(y_test_true_routed, routed_test_preds)
    print("\nComplete Router Performance (All 150 incoming test documents):")
    print(format_multiclass_table(routed_metrics))
    print("=" * 80)


if __name__ == "__main__":
    main()
