"""Evaluation metrics and threshold analysis for out-of-scope detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_rejection_metrics(
    y_true_oos: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Computes out-of-scope detection and rejection metrics.

    Args:
        y_true_oos: Binary array where 1 (or True) indicates Out-of-Scope (non-legal)
                    and 0 (or False) indicates In-Scope (legal).
        scores: 1D array of legal similarity scores (higher = more legal).
        threshold: Decision threshold tau. If score < tau, document is Rejected (predicted OOS).
                   If None, threshold-dependent metrics are omitted.

    Returns:
        dict containing ROC-AUC, PR-AUC, Out-of-Scope Recall, Legal False Rejection Rate, etc.
    """
    y_true = np.asarray(y_true_oos, dtype=bool)
    scores_arr = np.asarray(scores, dtype=np.float64)

    if len(y_true) != len(scores_arr):
        raise ValueError(f"Length mismatch: {len(y_true)} ground-truth labels vs {len(scores_arr)} scores")
    if not y_true.any() or y_true.all():
        raise ValueError("Ground-truth must contain both in-scope and out-of-scope examples")

    # Invert legal similarity so higher = more anomalous (out-of-scope)
    anomaly_scores = -scores_arr

    roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    pr_auc = float(average_precision_score(y_true, anomaly_scores))

    result = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_samples": int(len(y_true)),
        "n_oos": int(np.sum(y_true)),
        "n_legal": int(np.sum(~y_true)),
    }

    if threshold is not None:
        # Rejected as OOS if score < threshold
        pred_oos = scores_arr < threshold

        tp = int(np.sum(pred_oos & y_true))     # OOS correctly rejected
        fp = int(np.sum(pred_oos & ~y_true))    # Legal incorrectly rejected (false alarm)
        fn = int(np.sum(~pred_oos & y_true))    # OOS missed (accepted)
        tn = int(np.sum(~pred_oos & ~y_true))   # Legal correctly accepted

        total_oos = tp + fn
        total_legal = fp + tn

        oos_recall = tp / total_oos if total_oos > 0 else 0.0
        legal_false_rejection_rate = fp / total_legal if total_legal > 0 else 0.0
        oos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        oos_f1 = (2 * oos_precision * oos_recall / (oos_precision + oos_recall)
                  if (oos_precision + oos_recall) > 0 else 0.0)
        accuracy = (tp + tn) / len(y_true)

        result.update({
            "threshold": float(threshold),
            "oos_recall": float(oos_recall),
            "legal_false_rejection_rate": float(legal_false_rejection_rate),
            "oos_precision": float(oos_precision),
            "oos_f1": float(oos_f1),
            "accuracy": float(accuracy),
            "confusion_matrix": {
                "tp_oos_rejected": tp,
                "fp_legal_rejected": fp,
                "fn_oos_accepted": fn,
                "tn_legal_accepted": tn,
            },
        })

    return result


def sweep_thresholds(
    y_true_oos: np.ndarray,
    scores: np.ndarray,
    n_points: int = 101,
) -> list[dict]:
    """Sweeps candidate rejection thresholds to profile Out-of-Scope Recall vs Legal False Rejection.

    Args:
        y_true_oos: Binary array where True indicates Out-of-Scope.
        scores: 1D array of legal similarity scores.
        n_points: Number of thresholds to evaluate between min and max scores.

    Returns:
        List of dicts with threshold, oos_recall, legal_false_rejection_rate, oos_precision, oos_f1.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    min_score, max_score = float(np.min(scores_arr)), float(np.max(scores_arr))
    candidate_thresholds = np.linspace(min_score - 1e-4, max_score + 1e-4, n_points)

    curve = []
    for tau in candidate_thresholds:
        m = compute_rejection_metrics(y_true_oos, scores_arr, threshold=float(tau))
        curve.append({
            "threshold": float(tau),
            "oos_recall": m["oos_recall"],
            "legal_false_rejection_rate": m["legal_false_rejection_rate"],
            "oos_precision": m["oos_precision"],
            "oos_f1": m["oos_f1"],
        })
    return curve


def format_comparison_table(results: dict[str, dict]) -> str:
    """Formats an evaluation results mapping into a markdown comparison table.

    Expected results structure:
      {
         "ColPali - Nearest Doc": {...},
         "ColPali - Class Centroid": {...},
         "SigLIP - Nearest Doc": {...},
         "SigLIP - Class Centroid": {...},
      }
    """
    headers = [
        "Experiment",
        "ROC-AUC",
        "PR-AUC",
        "Threshold (tau)",
        "OOS Recall",
        "Legal False Rejection",
        "OOS F1",
    ]
    rows = []
    for name, r in results.items():
        row = [
            name,
            f"{r.get('roc_auc', 0.0):.4f}",
            f"{r.get('pr_auc', 0.0):.4f}",
            f"{r.get('threshold', 0.0):.4f}" if "threshold" in r else "N/A",
            f"{r.get('oos_recall', 0.0) * 100:.1f}%" if "oos_recall" in r else "N/A",
            f"{r.get('legal_false_rejection_rate', 0.0) * 100:.1f}%" if "legal_false_rejection_rate" in r else "N/A",
            f"{r.get('oos_f1', 0.0):.4f}" if "oos_f1" in r else "N/A",
        ]
        rows.append(row)

    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + data_lines)
