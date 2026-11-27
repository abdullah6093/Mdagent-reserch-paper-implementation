"""
Complexity assessment module (The Moderator) for MDAgents.
Classifies medical queries into low, moderate, or high complexity.
"""

import re
import logging
from typing import Optional

from llm import chat
from prompts import COMPLEXITY_CHECK_PROMPT
from config import COMPLEXITY_LOW, COMPLEXITY_MODERATE, COMPLEXITY_HIGH

logger = logging.getLogger("mdagents.complexity")

def determine_complexity(
    question: str,
    forced_difficulty: str = "adaptive",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3
) -> str:
    """
    Determine the complexity level of the query.
    If forced_difficulty is 'low', 'moderate', or 'high', returns that directly.
    Otherwise uses the LLM moderator to classify into low, moderate, or high.
    Defaults to 'moderate' on parsing failure as specified in Kim et al.
    """
    forced = forced_difficulty.lower().strip()
    if forced in [COMPLEXITY_LOW, COMPLEXITY_MODERATE, COMPLEXITY_HIGH]:
        return forced
    if forced in ["basic", "intermediate", "advanced"]:
        mapping = {"basic": COMPLEXITY_LOW, "intermediate": COMPLEXITY_MODERATE, "advanced": COMPLEXITY_HIGH}
        return mapping[forced]

    prompt = COMPLEXITY_CHECK_PROMPT.format(question=question)
    messages = [
        {
            "role": "system",
            "content": "You are a medical expert who conducts initial assessment and decides question complexity."
        },
        {"role": "user", "content": prompt}
    ]

    try:
        response = chat(messages, model=model, temperature=temperature)
        cleaned = response.lower().strip()
        
        if re.search(r'\b(low|basic|1\))\b', cleaned):
            return COMPLEXITY_LOW
        elif re.search(r'\b(high|advanced|3\))\b', cleaned):
            return COMPLEXITY_HIGH
        elif re.search(r'\b(moderate|intermediate|2\))\b', cleaned):
            return COMPLEXITY_MODERATE
        else:
            logger.warning(f"Could not parse complexity from response: '{response}'. Defaulting to moderate.")
            return COMPLEXITY_MODERATE
    except Exception as e:
        logger.error(f"Error in complexity determination: {e}. Defaulting to moderate.")
        return COMPLEXITY_MODERATE
