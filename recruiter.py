"""
Expert recruitment and team structure parser for MDAgents.
Builds Primary Care Clinicians (PCC), Multidisciplinary Teams (MDT),
or Integrated Care Teams (ICT) with communication graphs.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from llm import chat
from prompts import RECRUITER_PROMPT_MDT, RECRUITER_PROMPT_ICT
from config import COMPLEXITY_LOW, COMPLEXITY_MODERATE, COMPLEXITY_HIGH, DEFAULT_NUM_AGENTS

logger = logging.getLogger("mdagents.recruiter")

@dataclass
class AgentSpec:
    role: str
    description: str
    is_lead: bool = False
    peers: List[str] = field(default_factory=list)

@dataclass
class TeamSpec:
    name: str
    goal: str
    members: List[AgentSpec] = field(default_factory=list)

@dataclass
class RecruitmentPlan:
    complexity: str
    agents: List[AgentSpec] = field(default_factory=list)
    teams: List[TeamSpec] = field(default_factory=list)
    communication_graph: Dict[str, List[str]] = field(default_factory=dict)

def recruit_pcc() -> RecruitmentPlan:
    """Low complexity: Single Primary Care Clinician."""
    pcc_agent = AgentSpec(
        role="Primary Care Clinician",
        description="A general medical practitioner skilled in diagnosing common clinical conditions.",
        is_lead=True
    )
    return RecruitmentPlan(
        complexity=COMPLEXITY_LOW,
        agents=[pcc_agent],
        teams=[],
        communication_graph={"Primary Care Clinician": []}
    )

def recruit_mdt(
    question: str,
    num_agents: int = DEFAULT_NUM_AGENTS,
    model: str = "gpt-4o-mini"
) -> RecruitmentPlan:
    """
    Moderate complexity: Recruits an MDT of specialized agents.
    Parses '==' (role-description) and '>' (hierarchy) operators.
    """
    prompt = RECRUITER_PROMPT_MDT.format(question=question)
    messages = [
        {"role": "system", "content": "You are an experienced medical expert who recruits specialists for medical consultation."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = chat(messages, model=model, temperature=0.5)
        agents, graph = _parse_mdt_response(response, num_agents)
    except Exception as e:
        logger.warning(f"Error calling MDT recruiter: {e}. Using fallback specialists.")
        agents, graph = _fallback_mdt_agents()

    return RecruitmentPlan(
        complexity=COMPLEXITY_MODERATE,
        agents=agents,
        teams=[],
        communication_graph=graph
    )

def _parse_mdt_response(response: str, max_agents: int) -> Tuple[List[AgentSpec], Dict[str, List[str]]]:
    agents: List[AgentSpec] = []
    graph: Dict[str, List[str]] = {}

    chunks = re.split(r'>|\n', response)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if "==" in chunk:
            parts = chunk.split("==")
            role = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
            desc = parts[1].strip() if len(parts) > 1 else "Medical specialist"
            if role:
                agents.append(AgentSpec(role=role, description=desc))
        elif "-" in chunk and len(agents) < max_agents:
            parts = chunk.split("-")
            role = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
            desc = parts[1].strip() if len(parts) > 1 else "Medical specialist"
            if role:
                agents.append(AgentSpec(role=role, description=desc))

        if len(agents) >= max_agents:
            break

    if not agents:
        return _fallback_mdt_agents()

    all_roles = [a.role for a in agents]
    for a in agents:
        graph[a.role] = [r for r in all_roles if r != a.role]
        a.peers = graph[a.role]

    return agents, graph

def _fallback_mdt_agents() -> Tuple[List[AgentSpec], Dict[str, List[str]]]:
    agents = [
        AgentSpec(role="Internist", description="Specializes in adult internal diseases and systemic conditions."),
        AgentSpec(role="Clinical Specialist", description="Diagnoses complex disease mechanisms and clinical presentations."),
        AgentSpec(role="Diagnostic Consultant", description="Analyzes laboratory, clinical signs, and differential diagnoses.")
    ]
    graph = {
        "Internist": ["Clinical Specialist", "Diagnostic Consultant"],
        "Clinical Specialist": ["Internist", "Diagnostic Consultant"],
        "Diagnostic Consultant": ["Internist", "Clinical Specialist"]
    }
    for a in agents:
        a.peers = graph[a.role]
    return agents, graph

def recruit_ict(
    question: str,
    model: str = "gpt-4o-mini"
) -> RecruitmentPlan:
    """
    High complexity: Recruits multiple specialized teams (IAT, DET, FRDT).
    """
    prompt = RECRUITER_PROMPT_ICT.format(question=question)
    messages = [
        {"role": "system", "content": "You are a chief medical officer organizing multidisciplinary teams (MDTs) for integrated care."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = chat(messages, model=model, temperature=0.5)
        teams = _parse_ict_response(response)
    except Exception as e:
        logger.warning(f"Error calling ICT recruiter: {e}. Using fallback ICT structure.")
        teams = _fallback_ict_teams()

    all_agents = []
    for t in teams:
        all_agents.extend(t.members)

    return RecruitmentPlan(
        complexity=COMPLEXITY_HIGH,
        agents=all_agents,
        teams=teams,
        communication_graph={}
    )

def _parse_ict_response(response: str) -> List[TeamSpec]:
    teams: List[TeamSpec] = []
    current_team_name = "Initial Assessment Team (IAT)"
    current_members: List[AgentSpec] = []

    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        header_match = re.search(r'\[(.*?)\]|^(?:Group\s*\d+|Initial Assessment|Differential|Final Review)', line, re.IGNORECASE)
        if header_match and any(keyword in line.lower() for keyword in ["team", "group", "iat", "det", "frdt"]):
            if current_members:
                teams.append(TeamSpec(
                    name=current_team_name,
                    goal=f"Investigate {current_team_name}",
                    members=list(current_members)
                ))
                current_members = []
            current_team_name = line.replace("[", "").replace("]", "").strip()
            continue

        if "==" in line or "member" in line.lower() or line.startswith("-"):
            clean = re.sub(r'^(?:-\s*Member\s*\d*:\s*|-\s*)', '', line).strip()
            if "==" in clean:
                parts = clean.split("==")
                role = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else "Clinical team member"
                is_lead = "lead" in role.lower() or len(current_members) == 0
                current_members.append(AgentSpec(role=role, description=desc, is_lead=is_lead))

    if current_members:
        teams.append(TeamSpec(
            name=current_team_name,
            goal=f"Investigate {current_team_name}",
            members=list(current_members)
        ))

    if len(teams) < 2:
        return _fallback_ict_teams()

    return teams

def _fallback_ict_teams() -> List[TeamSpec]:
    team1 = TeamSpec(
        name="Initial Assessment Team (IAT)",
        goal="Gather baseline clinical evidence, patient history, and primary symptoms.",
        members=[
            AgentSpec(role="Triage Physician (Lead)", description="Evaluates acute presentation.", is_lead=True),
            AgentSpec(role="General Internist", description="Reviews baseline vital signs and system review.")
        ]
    )
    team2 = TeamSpec(
        name="Differential & Expert Team (DET)",
        goal="Formulate differential diagnoses and specialized pathophysiological analysis.",
        members=[
            AgentSpec(role="Subspecialist (Lead)", description="Provides targeted subspecialty expertise.", is_lead=True),
            AgentSpec(role="Pathology & Lab Consultant", description="Correlates clinical findings with diagnostic criteria.")
        ]
    )
    team3 = TeamSpec(
        name="Final Review & Decision Team (FRDT)",
        goal="Synthesize all evidence and deliver the definitive decision.",
        members=[
            AgentSpec(role="Senior Attending Clinician (Lead)", description="Oversees clinical decision-making.", is_lead=True),
            AgentSpec(role="Evidence Specialist", description="Validates treatment efficacy and consensus.")
        ]
    )
    return [team1, team2, team3]

def recruit(
    question: str,
    complexity: str,
    model: str = "gpt-4o-mini",
    num_agents: int = DEFAULT_NUM_AGENTS
) -> RecruitmentPlan:
    if complexity == COMPLEXITY_LOW:
        return recruit_pcc()
    elif complexity == COMPLEXITY_MODERATE:
        return recruit_mdt(question, num_agents=num_agents, model=model)
    elif complexity == COMPLEXITY_HIGH:
        return recruit_ict(question, model=model)
    else:
        return recruit_mdt(question, num_agents=num_agents, model=model)
