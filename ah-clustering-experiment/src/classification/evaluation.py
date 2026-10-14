"""Evaluation metrics and formatting for multiclass legal document classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_multiclass_metrics(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
) -> dict:
    """Computes comprehensive multiclass classification metrics."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    unique_classes = sorted(list(set(y_t) | set(y_p)))

    # Classification report as dict
    report = classification_report(y_t, y_p, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=unique_classes)

    accuracy = float(report.get("accuracy", 0.0))
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
    weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", 0.0))

    per_class = {}
    for c in unique_classes:
        if c in report:
            per_class[c] = {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classes": unique_classes,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def format_multiclass_table(metrics: dict) -> str:
    """Formats per-class classification metrics as a Markdown table."""
    per_class = metrics.get("per_class", {})
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for c, stats in per_class.items():
        rows.append([
            c,
            f"{stats['precision'] * 100:.1f}%",
            f"{stats['recall'] * 100:.1f}%",
            f"{stats['f1']:.4f}",
            str(stats["support"]),
        ])

    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    summary_line = (
        f"\n**Overall Accuracy:** {metrics['accuracy'] * 100:.2f}% | "
        f"**Macro F1:** {metrics['macro_f1']:.4f} | "
        f"**Weighted F1:** {metrics['weighted_f1']:.4f}"
    )
    return "\n".join([header_line, separator_line] + data_lines) + summary_line
