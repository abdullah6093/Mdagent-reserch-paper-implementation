"""Pipelines module for MDAgents."""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .low import run_low_pipeline
from .moderate import run_moderate_pipeline
from .high import run_high_pipeline
from recruiter import RecruitmentPlan
from config import COMPLEXITY_LOW, COMPLEXITY_MODERATE, COMPLEXITY_HIGH, DEFAULT_MODEL

def run_pipeline(
    question: str,
    complexity: str,
    recruitment_plan: RecruitmentPlan,
    model: str = DEFAULT_MODEL,
    examplers: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Union[str, Path]]] = None,
    **kwargs
) -> Dict[str, Any]:
    comp = complexity.lower().strip()
    if comp == COMPLEXITY_LOW:
        return run_low_pipeline(
            question=question,
            recruitment_plan=recruitment_plan,
            model=model,
            examplers=examplers,
            images=images
        )
    elif comp == COMPLEXITY_MODERATE:
        return run_moderate_pipeline(
            question=question,
            recruitment_plan=recruitment_plan,
            model=model,
            examplers=examplers,
            images=images,
            **kwargs
        )
    elif comp == COMPLEXITY_HIGH:
        return run_high_pipeline(
            question=question,
            recruitment_plan=recruitment_plan,
            model=model,
            examplers=examplers,
            images=images
        )
    else:
        return run_moderate_pipeline(
            question=question,
            recruitment_plan=recruitment_plan,
            model=model,
            examplers=examplers,
            images=images,
            **kwargs
        )

__all__ = [
    "run_low_pipeline",
    "run_moderate_pipeline",
    "run_high_pipeline",
    "run_pipeline"
]
