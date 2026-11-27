"""
Moderate-complexity pipeline (MDT) for MDAgents.
Multi-round, multi-turn collaborative discussion between specialist agents,
with consensus monitoring, entropy calculation, and moderator feedback.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from agents import Agent
from recruiter import RecruitmentPlan
from prompts import (
    PARTICIPATION_PROMPT,
    PERSUASION_PROMPT,
    MODERATOR_FEEDBACK_PROMPT
)
from config import DEFAULT_MODEL, MAX_ROUNDS, MAX_TURNS, DEFAULT_TEMPERATURE
from llm import get_api_call_count, chat

def calculate_consensus_entropy(answers: List[str]) -> float:
    if not answers:
        return 0.0
    counts = Counter(answers)
    total = len(answers)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def _extract_choice(text: str) -> str:
    match = re.search(r'\b(?:[Aa]nswer|[Oo]ption)[:\s]*\(?([A-Ea-e])\)?', text)
    if match:
        return match.group(1).upper()
    single_letter = re.search(r'\b([A-E])\b', text)
    if single_letter:
        return single_letter.group(1).upper()
    return text.strip()[:20]

def run_moderate_pipeline(
    question: str,
    recruitment_plan: RecruitmentPlan,
    model: str = DEFAULT_MODEL,
    examplers: Optional[List[Dict[str, Any]]] = None,
    max_rounds: int = MAX_ROUNDS,
    max_turns: int = MAX_TURNS,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    
    agent_dict: Dict[str, Agent] = {}
    for spec in recruitment_plan.agents:
        ag = Agent(
            role=spec.role,
            description=spec.description,
            model=model,
            examplers=examplers[:2] if examplers else None
        )
        agent_dict[spec.role] = ag

    interaction_log = []
    entropy_history = []
    
    current_opinions: Dict[str, str] = {}
    current_choices: Dict[str, str] = {}
    for role, ag in agent_dict.items():
        init_msg = (
            f"Here is a medical query under team discussion:\n{question}\n\n"
            f"From your perspective as a {role}, give your initial diagnosis/answer with justification."
        )
        opinion = ag.chat(init_msg, images=images)
        current_opinions[role] = opinion
        current_choices[role] = _extract_choice(opinion)

    initial_entropy = calculate_consensus_entropy(list(current_choices.values()))
    entropy_history.append({"round": 0, "turn": 0, "entropy": initial_entropy, "choices": dict(current_choices)})
    interaction_log.append({"stage": "initial_assessment", "opinions": dict(current_opinions)})

    if len(set(current_choices.values())) == 1:
        return _make_final_result(question, current_opinions, interaction_log, entropy_history, start_calls, model)

    for r in range(1, max_rounds + 1):
        round_log = {"round": r, "turns": []}

        for t in range(1, max_turns + 1):
            turn_active_count = 0
            turn_messages = []

            for role, ag in agent_dict.items():
                peers = recruitment_plan.communication_graph.get(role, [k for k in agent_dict.keys() if k != role])
                if not peers:
                    continue

                other_opinions_str = "\n".join([f"[{k}]: {v}" for k, v in current_opinions.items() if k != role])
                
                part_prompt = PARTICIPATION_PROMPT.format(
                    role=role,
                    question=question,
                    opinions=other_opinions_str
                )
                participate = ag.chat(part_prompt)

                if "yes" in participate.lower():
                    target_role = peers[0]
                    persuasion = ag.chat(
                        PERSUASION_PROMPT.format(
                            role=role,
                            question=question,
                            target_agent=target_role
                        )
                    )
                    if target_role in agent_dict:
                        peer_resp = agent_dict[target_role].chat(
                            f"[{role} says to you]:\n{persuasion}\n\nPlease reply with your assessment."
                        )
                        current_opinions[target_role] = peer_resp
                        current_choices[target_role] = _extract_choice(peer_resp)
                        turn_messages.append({"from": role, "to": target_role, "content": persuasion, "reply": peer_resp})
                        turn_active_count += 1

            step_entropy = calculate_consensus_entropy(list(current_choices.values()))
            entropy_history.append({"round": r, "turn": t, "entropy": step_entropy, "choices": dict(current_choices)})
            round_log["turns"].append({"turn": t, "messages": turn_messages, "entropy": step_entropy})

            if len(set(current_choices.values())) == 1 or turn_active_count == 0:
                break

        interaction_log.append(round_log)

        if len(set(current_choices.values())) == 1:
            break

        if r < max_rounds:
            discussion_summary = "\n".join([f"{k}: {v}" for k, v in current_opinions.items()])
            mod_prompt = MODERATOR_FEEDBACK_PROMPT.format(
                question=question,
                discussion_history=discussion_summary
            )
            moderator_feedback = chat(
                [{"role": "system", "content": "You are a clinical moderator guiding an MDT consensus discussion."},
                 {"role": "user", "content": mod_prompt}],
                model=model,
                temperature=0.3
            )
            for role, ag in agent_dict.items():
                ag.chat(f"[Moderator Review & Feedback]:\n{moderator_feedback}\nPlease refine your clinical judgment.")

    return _make_final_result(question, current_opinions, interaction_log, entropy_history, start_calls, model)

def _make_final_result(question, current_opinions, interaction_log, entropy_history, start_calls, model):
    opinions_str = "\n\n".join([f"[{k}]: {v}" for k, v in current_opinions.items()])
    final_prompt = (
        f"You are the chief moderator of an MDT team.\n"
        f"Question:\n{question}\n\n"
        f"Here are the final conclusions from all specialists:\n"
        f"{opinions_str}\n\n"
        f"Make the definitive clinical decision based on group consensus."
    )
    final_decision = chat(
        [{"role": "system", "content": "You are a final medical decision maker synthesizing MDT opinions."},
         {"role": "user", "content": final_prompt}],
        model=model,
        temperature=DEFAULT_TEMPERATURE
    )

    total_calls = get_api_call_count() - start_calls
    return {
        "final_answer": final_decision,
        "raw_response": final_decision,
        "complexity": "moderate",
        "opinions": current_opinions,
        "entropy_history": entropy_history,
        "interaction_log": interaction_log,
        "api_calls": total_calls
    }
