"""Multiclass legal document classification and two-stage intake routing."""
from __future__ import annotations

from .classifier import LegalDocumentClassifier, TwoStageIntakeRouter
from .evaluation import compute_multiclass_metrics, format_multiclass_table

__all__ = [
    "LegalDocumentClassifier",
    "TwoStageIntakeRouter",
    "compute_multiclass_metrics",
    "format_multiclass_table",
]
