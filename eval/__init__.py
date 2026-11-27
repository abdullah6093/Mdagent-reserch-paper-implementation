"""Evaluation module for MDAgents."""

from .metrics import (
    compute_accuracy,
    compute_consensus_entropy,
    estimate_cost,
    significance_test
)

from .analysis import (
    analyze_results,
    format_summary_table
)

__all__ = [
    "compute_accuracy",
    "compute_consensus_entropy",
    "estimate_cost",
    "significance_test",
    "analyze_results",
    "format_summary_table"
]
