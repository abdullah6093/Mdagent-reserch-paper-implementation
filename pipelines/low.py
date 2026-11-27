"""
Low-complexity pipeline for MDAgents.
Single Primary Care Clinician (PCC) with 3-shot prompt and direct answer.
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from agents import Agent
from recruiter import RecruitmentPlan
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from llm import get_api_call_count

def run_low_pipeline(
    question: str,
    recruitment_plan: RecruitmentPlan,
    model: str = DEFAULT_MODEL,
    examplers: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    spec = recruitment_plan.agents[0] if recruitment_plan.agents else None
    role = spec.role if spec else "Primary Care Clinician"
    desc = spec.description if spec else "A general physician providing comprehensive primary medical care."

    pcc = Agent(
        role=role,
        description=desc,
        model=model,
        examplers=examplers[:3] if examplers else None
    )

    query = (
        f"Please review the following medical question carefully and select the best answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Provide your answer and brief reasoning."
    )

    response = pcc.chat(query, temperature=DEFAULT_TEMPERATURE, images=images)
    total_calls = get_api_call_count() - start_calls

    return {
        "final_answer": response,
        "raw_response": response,
        "complexity": "low",
        "api_calls": total_calls,
        "interaction_log": {"pcc_response": response}
    }
