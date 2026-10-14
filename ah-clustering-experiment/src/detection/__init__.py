"""Out-of-scope detection and rejection package."""
from __future__ import annotations

from .detector import OutOfScopeDetector
from .evaluation import compute_rejection_metrics, format_comparison_table, sweep_thresholds
from .pooling import pool_colpali_tokens, pool_tokens

# Recognized in-scope legal document subsets based on project specifications
CORE_LEGAL_CLASSES = {
    "REAL_ESTATE",
    "COMMERCIAL_LEASE_AGREEMENT",
    "PATENT",
    "PETITION_FORM",
    "PROXY_VOTING",
}

FINANCIAL_LEGAL_CLASSES = {
    "ACCOUNT_STATEMENT",
    "CREDIT_CARD_STATEMENT",
    "FORM_1040",
}

__all__ = [
    "OutOfScopeDetector",
    "compute_rejection_metrics",
    "sweep_thresholds",
    "format_comparison_table",
    "pool_tokens",
    "pool_colpali_tokens",
    "CORE_LEGAL_CLASSES",
    "FINANCIAL_LEGAL_CLASSES",
]
