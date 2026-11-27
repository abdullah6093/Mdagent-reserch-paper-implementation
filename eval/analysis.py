"""
Analysis and reporting module for MDAgents evaluation results.
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

from .metrics import compute_accuracy, estimate_cost

def analyze_results(results: List[Dict[str, Any]], model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    predictions = [r.get("predicted_choice", "") for r in results]
    ground_truths = [r.get("ground_truth", "") for r in results]

    acc_stats = compute_accuracy(predictions, ground_truths)
    complexity_counts = Counter([r.get("complexity", "unknown") for r in results])
    
    api_calls = [r.get("api_calls", 1) for r in results]
    avg_api_calls = round(sum(api_calls) / len(api_calls), 2) if api_calls else 0
    total_api_calls = sum(api_calls)

    entropies = []
    for r in results:
        history = r.get("entropy_history", [])
        if history:
            entropies.append(history[-1].get("entropy", 0.0))
    avg_final_entropy = round(sum(entropies) / len(entropies), 4) if entropies else 0.0

    summary = {
        "total_samples": len(results),
        "accuracy": acc_stats["accuracy"],
        "correct": acc_stats["correct"],
        "complexity_distribution": dict(complexity_counts),
        "avg_api_calls_per_query": avg_api_calls,
        "total_api_calls": total_api_calls,
        "avg_final_entropy": avg_final_entropy,
        "estimated_cost_usd": estimate_cost(model_name, total_api_calls * 400, total_api_calls * 150)
    }

    return summary

def format_summary_table(summary: Dict[str, Any]) -> str:
    lines = [
        "=" * 50,
        "             MDAGENTS EVALUATION REPORT          ",
        "=" * 50,
        f"Total Samples Tested:   {summary['total_samples']}",
        f"Accuracy:               {summary['accuracy'] * 100:.1f}% ({summary['correct']}/{summary['total_samples']})",
        f"Avg API Calls / Query:  {summary['avg_api_calls_per_query']}",
        f"Total API Calls:        {summary['total_api_calls']}",
        f"Avg Final Consensus H:  {summary['avg_final_entropy']}",
        f"Estimated Total Cost:   ${summary['estimated_cost_usd']:.4f}",
        "-" * 50,
        "Complexity Distribution:",
    ]
    for comp, count in summary.get("complexity_distribution", {}).items():
        pct = (count / summary['total_samples']) * 100 if summary['total_samples'] > 0 else 0
        lines.append(f"  - {comp.capitalize():<12}: {count:3d} ({pct:.1f}%)")
    lines.append("=" * 50)
    return "\n".join(lines)
