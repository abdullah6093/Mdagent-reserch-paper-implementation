"""Baselines module for MDAgents."""

from .single_agent import (
    zero_shot,
    few_shot,
    few_shot_cot,
    cot_sc,
    ensemble_refinement,
    medprompt
)

from .multi_agent import (
    majority_voting,
    weighted_voting,
    borda_count,
    medagents,
    meta_prompting,
    reconcile,
    autogen_baseline,
    dylan_baseline
)

__all__ = [
    "zero_shot",
    "few_shot",
    "few_shot_cot",
    "cot_sc",
    "ensemble_refinement",
    "medprompt",
    "majority_voting",
    "weighted_voting",
    "borda_count",
    "medagents",
    "meta_prompting",
    "reconcile",
    "autogen_baseline",
    "dylan_baseline"
]
