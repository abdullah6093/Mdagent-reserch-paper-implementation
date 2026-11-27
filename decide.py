"""
Final decision maker and temperature ensemble module for MDAgents.
Provides robust answer parsing (regex) and multi-temperature ensembling.
"""

import re
import logging
from collections import Counter
from typing import List, Dict, Optional, Any, Union

from llm import chat
from prompts import FINAL_DECISION_PROMPT
from config import DEFAULT_MODEL, TEMPERATURE_ENSEMBLE, DEFAULT_TEMPERATURE

logger = logging.getLogger("mdagents.decide")

def extract_answer_choice(text: str, valid_options: Optional[List[str]] = None) -> str:
    if not text:
        return ""

    patterns = [
        r'(?:answer|choice|option|decision)[:\s\*\#]+\(?([A-Ea-e])\)?(?:\s|$|\.)',
        r'final answer is[:\s\*\#]+\(?([A-Ea-e])\)?',
        r'^\s*\(?([A-Ea-e])\)?[\.\:\s]',
        r'\b([A-Ea-e])\)\s',
        r'option\s+([A-Ea-e])\b',
        r'\b([A-Ea-e])\b'
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            choice = m.group(1).upper()
            if valid_options and choice in valid_options:
                return choice
            elif not valid_options:
                return choice

    lower = text.lower()
    if "yes" in lower and "no" not in lower:
        return "yes"
    if "no" in lower and "yes" not in lower:
        return "no"

    fallback = re.search(r'\b([A-E])\b', text)
    if fallback:
        return fallback.group(1)

    return text.strip()[:10]

def make_final_decision_with_ensemble(
    question: str,
    accumulated_evidence: str,
    model: str = DEFAULT_MODEL,
    temperatures: Optional[List[float]] = None
) -> Dict[str, Any]:
    if temperatures is None:
        temperatures = TEMPERATURE_ENSEMBLE

    prompt = FINAL_DECISION_PROMPT.format(
        question=question,
        answer_choices="",
        agent_answers=accumulated_evidence
    )
    messages = [
        {"role": "system", "content": "You are a senior medical decision maker reaching the final clinical decision."},
        {"role": "user", "content": prompt}
    ]

    responses = {}
    parsed_choices = []

    for t in temperatures:
        resp = chat(messages, model=model, temperature=t)
        responses[str(t)] = resp
        parsed = extract_answer_choice(resp)
        parsed_choices.append(parsed)

    counts = Counter(parsed_choices)
    majority_choice = counts.most_common(1)[0][0] if parsed_choices else ""

    return {
        "final_choice": majority_choice,
        "temperature_responses": responses,
        "temperature_votes": parsed_choices,
        "vote_counts": dict(counts)
    }
