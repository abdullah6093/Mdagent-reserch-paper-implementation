"""
High-complexity pipeline (Integrated Care Teams - ICT) for MDAgents.
Coordinates multi-departmental teams passing accumulated synthesized reports.
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from agents import Agent, AgentTeam
from recruiter import RecruitmentPlan
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from llm import get_api_call_count, chat

def run_high_pipeline(
    question: str,
    recruitment_plan: RecruitmentPlan,
    model: str = DEFAULT_MODEL,
    examplers: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    accumulated_reports: List[Dict[str, str]] = []
    accumulated_str = ""

    teams: List[AgentTeam] = []
    for team_spec in recruitment_plan.teams:
        team_agents = [
            Agent(
                role=m.role,
                description=m.description,
                model=model,
                examplers=examplers[:1] if examplers else None
            )
            for m in team_spec.members
        ]
        lead_agent = team_agents[0] if team_agents else None
        teams.append(AgentTeam(name=team_spec.name, goal=team_spec.goal, members=team_agents, lead=lead_agent))

    for team in teams:
        team_report = team.synthesize_team_report(question=question, prior_reports=accumulated_str)
        accumulated_reports.append({
            "team_name": team.name,
            "goal": team.goal,
            "report": team_report
        })
        accumulated_str += f"\n\n=== [{team.name}] ===\n{team_report}\n"

    decision_prompt = (
        f"You are the Chief Medical Officer and Final Decision Maker.\n\n"
        f"Medical Question:\n{question}\n\n"
        f"Cumulative Clinical Reports from All Integrated Care Teams:\n"
        f"{accumulated_str}\n\n"
        f"Please review the accumulated evidence thoroughly and output your final diagnostic conclusion and chosen answer."
    )
    final_decision = chat(
        [
            {"role": "system", "content": "You are a chief physician delivering the definitive decision for a complex case."},
            {"role": "user", "content": decision_prompt}
        ],
        model=model,
        temperature=DEFAULT_TEMPERATURE,
        images=images
    )

    total_calls = get_api_call_count() - start_calls
    return {
        "final_answer": final_decision,
        "raw_response": final_decision,
        "complexity": "high",
        "accumulated_reports": accumulated_reports,
        "api_calls": total_calls
    }
