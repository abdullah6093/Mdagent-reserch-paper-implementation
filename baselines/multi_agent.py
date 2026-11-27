"""
Multi-agent baselines from Kim et al., NeurIPS 2024.
Implements:
1. Majority voting
2. Weighted voting
3. Borda count
4. MedAgents
5. Meta-Prompting
6. ReConcile
7. AutoGen
8. DyLAN
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from llm import chat, get_api_call_count
from decide import extract_answer_choice
from config import DEFAULT_MODEL

def majority_voting(
    question: str,
    model: str = DEFAULT_MODEL,
    n_agents: int = 5,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    votes = []

    for idx in range(n_agents):
        prompt = f"Agent {idx + 1}: Please solve this medical question and state your final answer choice.\n\n{question}"
        resp = chat(prompt, model=model, temperature=0.7, images=images)
        choice = extract_answer_choice(resp)
        votes.append(choice)

    counts = Counter(votes)
    winner = counts.most_common(1)[0][0] if votes else ""

    return {
        "baseline": "majority_voting",
        "answer": winner,
        "votes": votes,
        "api_calls": get_api_call_count() - start_calls
    }

def weighted_voting(
    question: str,
    model: str = DEFAULT_MODEL,
    n_agents: int = 5,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    weighted_scores: Dict[str, float] = defaultdict(float)

    for idx in range(n_agents):
        prompt = (
            f"You are Medical Expert {idx + 1}.\n"
            f"Answer this medical question:\n{question}\n\n"
            f"Provide your answer and a confidence score from 1 to 10 in this format:\n"
            f"Answer: <choice>\nConfidence: <score>"
        )
        resp = chat(prompt, model=model, temperature=0.7, images=images)
        choice = extract_answer_choice(resp)
        conf_match = re.search(r'confidence[:\s]+(\d+)', resp, re.IGNORECASE)
        conf = float(conf_match.group(1)) if conf_match else 5.0
        weighted_scores[choice] += conf

    winner = max(weighted_scores.items(), key=lambda x: x[1])[0] if weighted_scores else ""

    return {
        "baseline": "weighted_voting",
        "answer": winner,
        "scores": dict(weighted_scores),
        "api_calls": get_api_call_count() - start_calls
    }

def borda_count(
    question: str,
    model: str = DEFAULT_MODEL,
    n_agents: int = 5,
    options: Optional[List[str]] = None,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    opts = options or ["A", "B", "C", "D"]
    borda_scores: Dict[str, int] = {opt: 0 for opt in opts}

    for idx in range(n_agents):
        prompt = (
            f"Medical Expert {idx + 1}:\n"
            f"Rank the answer choices from most likely to least likely for this medical query:\n{question}\n\n"
            f"Provide your ranked list (e.g. 1. A, 2. B, 3. C, 4. D):"
        )
        resp = chat(prompt, model=model, temperature=0.7, images=images)
        found = [m.upper() for m in re.findall(r'\b([A-E])\b', resp) if m.upper() in opts]
        seen = set()
        ordered = []
        for f in found:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
        for missing in opts:
            if missing not in ordered:
                ordered.append(missing)

        n = len(ordered)
        for rank, opt in enumerate(ordered):
            borda_scores[opt] += (n - 1 - rank)

    winner = max(borda_scores.items(), key=lambda x: x[1])[0] if borda_scores else ""

    return {
        "baseline": "borda_count",
        "answer": winner,
        "borda_scores": borda_scores,
        "api_calls": get_api_call_count() - start_calls
    }

def medagents(
    question: str,
    model: str = DEFAULT_MODEL,
    n_agents: int = 5,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()

    recruitment_resp = chat(
        f"Identify {n_agents} medical specialist roles needed to solve:\n{question}\nList only the roles separated by commas.",
        model=model,
        temperature=0.3
    )
    roles = [r.strip() for r in recruitment_resp.split(",") if r.strip()][:n_agents]
    if len(roles) < 2:
        roles = ["Clinician", "Pathologist", "Pharmacologist", "Surgeon", "Radiologist"]

    analyses = {}
    for role in roles:
        ans = chat(
            f"You are a {role}. Provide your clinical analysis and answer for:\n{question}",
            model=model,
            temperature=0.5,
            images=images
        )
        analyses[role] = ans

    combined = "\n\n".join([f"[{k}]: {v}" for k, v in analyses.items()])
    discussion_resp = chat(
        f"Review the clinical analyses from all experts:\n{combined}\n\n"
        f"Medical Question:\n{question}\n\n"
        f"Synthesize the consensus answer among the experts:",
        model=model,
        temperature=0.2
    )

    choice = extract_answer_choice(discussion_resp)
    return {
        "baseline": "medagents",
        "answer": choice,
        "discussion": discussion_resp,
        "api_calls": get_api_call_count() - start_calls
    }

def meta_prompting(
    question: str,
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()

    orchestrator_prompt = (
        f"You are a Meta-Clinical Orchestrator. For this medical question:\n{question}\n\n"
        f"1. Break the question into sub-problems.\n"
        f"2. Solve each sub-problem adopting the appropriate medical persona.\n"
        f"3. Verify and combine all findings into the final answer choice."
    )
    resp = chat(orchestrator_prompt, model=model, temperature=0.3, images=images)
    choice = extract_answer_choice(resp)

    return {
        "baseline": "meta_prompting",
        "answer": choice,
        "response": resp,
        "api_calls": get_api_call_count() - start_calls
    }

def reconcile(
    question: str,
    models: Optional[List[str]] = None,
    n_rounds: int = 3,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    agent_roles = ["Internal Medicine", "Specialist Physician", "Clinical Researcher"]
    opinions = {}

    for role in agent_roles:
        ans = chat(f"You are an expert in {role}. Answer:\n{question}", temperature=0.6, images=images)
        opinions[role] = ans

    for r in range(n_rounds - 1):
        round_ops = {}
        for role in agent_roles:
            others = "\n".join([f"{k}: {v}" for k, v in opinions.items() if k != role])
            prompt = (
                f"You are {role}.\n"
                f"Here are the opinions of the other clinicians:\n{others}\n\n"
                f"Question: {question}\n\n"
                f"Defend your position or update your view if convinced."
            )
            round_ops[role] = chat(prompt, temperature=0.4)
        opinions = round_ops

    summary = "\n".join([f"{k}: {v}" for k, v in opinions.items()])
    final_resp = chat(
        f"Review the debate:\n{summary}\n\nQuestion:\n{question}\nWhat is the consensus answer?",
        temperature=0.1
    )
    choice = extract_answer_choice(final_resp)

    return {
        "baseline": "reconcile",
        "answer": choice,
        "debate_summary": summary,
        "api_calls": get_api_call_count() - start_calls
    }

def autogen_baseline(
    question: str,
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()

    clinician_resp = chat(f"As a Primary Clinician, assess this case:\n{question}", model=model, temperature=0.5, images=images)
    specialist_resp = chat(f"As a Medical Specialist, review this assessment and provide specialized differential:\nAssessment: {clinician_resp}\nQuestion: {question}", model=model, temperature=0.5)
    moderator_resp = chat(f"As the Clinical Moderator, evaluate both opinions and state the final answer:\nClinician: {clinician_resp}\nSpecialist: {specialist_resp}\nQuestion: {question}", model=model, temperature=0.1)

    choice = extract_answer_choice(moderator_resp)
    return {
        "baseline": "autogen",
        "answer": choice,
        "moderator_decision": moderator_resp,
        "api_calls": get_api_call_count() - start_calls
    }

def dylan_baseline(
    question: str,
    model: str = DEFAULT_MODEL,
    n_rounds: int = 4,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    agents = [f"Specialist Agent {i + 1}" for i in range(4)]
    agent_outputs = {}

    for ag in agents:
        agent_outputs[ag] = chat(f"You are {ag}. Answer:\n{question}", model=model, temperature=0.5, images=images)

    for r in range(1, n_rounds):
        new_outputs = {}
        for ag in agents:
            peer_context = "\n".join([f"{k}: {v}" for k, v in agent_outputs.items() if k != ag])
            prompt = f"You are {ag} in round {r + 1}. Consider peer insights:\n{peer_context}\n\nQuestion: {question}\nRefined diagnosis:"
            new_outputs[ag] = chat(prompt, model=model, temperature=0.3)
        agent_outputs = new_outputs

    all_answers = [extract_answer_choice(ans) for ans in agent_outputs.values()]
    counts = Counter(all_answers)
    final_choice = counts.most_common(1)[0][0] if all_answers else ""

    return {
        "baseline": "dylan",
        "answer": final_choice,
        "votes": all_answers,
        "api_calls": get_api_call_count() - start_calls
    }
