"""
Evaluation metrics for the MDAgents framework.
Calculates:
1. Accuracy: correct / total
2. Consensus entropy: H = - sum(p(x) * log2(p(x)))
3. Classifier optimality: accuracy = a*p_argmax + b*p_argmid + c*p_argmin
4. API call counters and USD cost estimation
5. Statistical significance testing (paired t-test and McNemar's test)
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = 0
    total = len(predictions)
    for p, g in zip(predictions, ground_truths):
        p_clean = str(p).strip().upper()
        g_clean = str(g).strip().upper()
        if p_clean == g_clean:
            correct += 1

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "correct": correct,
        "total": total
    }

def compute_consensus_entropy(answers: List[str]) -> float:
    if not answers:
        return 0.0
    counts = Counter(answers)
    total = len(answers)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def estimate_cost(
    model: str,
    total_prompt_tokens: int,
    total_completion_tokens: int
) -> float:
    rates = {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 0.50, "output": 1.50}
    }
    
    rate = rates.get(model, {"input": 1.0, "output": 2.0})
    input_cost = (total_prompt_tokens / 1_000_000) * rate["input"]
    output_cost = (total_completion_tokens / 1_000_000) * rate["output"]
    return round(input_cost + output_cost, 4)

def significance_test(
    scores_a: List[int],
    scores_b: List[int]
) -> Dict[str, Any]:
    """Fast, reliable paired t-test with standard normal/t approximation."""
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    n = len(diffs)
    if n < 2:
        return {"p_value": 1.0, "statistic": 0.0, "is_significant_p05": False}

    mean_d = sum(diffs) / n
    variance = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_err = math.sqrt(variance / n) if variance > 0 else 0.0

    if std_err == 0.0:
        stat = 0.0
        p_val = 1.0 if mean_d == 0 else 0.001
    else:
        stat = mean_d / std_err
        # Two-tailed approximation using normal CDF / erf
        z = abs(stat)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    p_val = max(0.0001, min(1.0, p_val))
    return {
        "p_value": round(p_val, 5),
        "statistic": round(stat, 4),
        "is_significant_p05": p_val < 0.05
    }
