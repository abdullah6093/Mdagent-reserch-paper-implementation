"""
Single-agent baselines from Kim et al., NeurIPS 2024.
Implements:
1. Zero-shot
2. Few-shot (3-shot)
3. Few-shot CoT
4. CoT Self-Consistency (CoT-SC)
5. Ensemble Refinement (ER)
6. Medprompt (kNN/shuffled few-shot + CoT ensemble)
"""

import random
from collections import Counter
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from llm import chat, get_api_call_count
from decide import extract_answer_choice
from prompts import ZERO_SHOT_PROMPT, COT_PROMPT, COT_SC_PROMPT, FEWSHOT_BASE_PROMPT
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE

def zero_shot(
    question: str,
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    prompt = ZERO_SHOT_PROMPT.format(
        context="",
        question=question,
        answer_choices=""
    )
    resp = chat(prompt, model=model, temperature=0.1, images=images)
    choice = extract_answer_choice(resp)
    return {
        "baseline": "zero_shot",
        "answer": choice,
        "raw_response": resp,
        "api_calls": get_api_call_count() - start_calls
    }

def few_shot(
    question: str,
    examples: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None,
    n_shot: int = 3
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    shot_examples = examples[:n_shot] if examples else []
    ex_str = "\n\n".join([f"Q: {e['question']}\nA: {e['answer']}" for e in shot_examples])

    prompt = FEWSHOT_BASE_PROMPT.format(
        instruction="Answer the following medical questions given the examples.",
        few_shot_examples=ex_str,
        context="",
        question=question,
        answer_choices=""
    )
    resp = chat(prompt, model=model, temperature=0.1, images=images)
    choice = extract_answer_choice(resp)
    return {
        "baseline": "few_shot",
        "answer": choice,
        "raw_response": resp,
        "api_calls": get_api_call_count() - start_calls
    }

def few_shot_cot(
    question: str,
    examples: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None,
    n_shot: int = 3
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    shot_examples = examples[:n_shot] if examples else []
    ex_str = "\n\n".join([
        f"Q: {e['question']}\nReasoning: {e.get('reason', 'Step-by-step clinical analysis.')}\n{e['answer']}"
        for e in shot_examples
    ])

    prompt = (
        f"You are a medical expert. Review the examples, then think step by step to answer the question.\n\n"
        f"Examples:\n{ex_str}\n\n"
        f"Question:\n{question}\n\n"
        f"Step-by-step reasoning and Answer:"
    )
    resp = chat(prompt, model=model, temperature=0.3, images=images)
    choice = extract_answer_choice(resp)
    return {
        "baseline": "few_shot_cot",
        "answer": choice,
        "raw_response": resp,
        "api_calls": get_api_call_count() - start_calls
    }

def cot_sc(
    question: str,
    examples: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None,
    n_chains: int = 5
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    chains = []
    votes = []

    for seed_idx in range(n_chains):
        temp = 0.7
        prompt = (
            f"Solve this medical query by thinking step by step through clinical principles.\n\n"
            f"Question:\n{question}\n\n"
            f"Chain of reasoning #{seed_idx + 1}:"
        )
        resp = chat(prompt, model=model, temperature=temp, images=images, seed=seed_idx * 10)
        choice = extract_answer_choice(resp)
        chains.append(resp)
        votes.append(choice)

    counts = Counter(votes)
    majority_vote = counts.most_common(1)[0][0] if votes else ""

    return {
        "baseline": "cot_sc",
        "answer": majority_vote,
        "votes": votes,
        "chains": chains,
        "api_calls": get_api_call_count() - start_calls
    }

def ensemble_refinement(
    question: str,
    examples: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None,
    n_samples: int = 3
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    initial_solutions = []
    for i in range(n_samples):
        resp = chat(
            f"Analyze this medical query:\n{question}\nProvide your step-by-step reasoning and choice.",
            model=model,
            temperature=0.7,
            images=images
        )
        initial_solutions.append(resp)

    combined_solutions = "\n\n---\n\n".join(initial_solutions)
    refinement_prompt = (
        f"You are an expert clinician reviewing {n_samples} independent candidate reasoning paths for this question:\n"
        f"{question}\n\n"
        f"Candidate solutions:\n{combined_solutions}\n\n"
        f"Synthesize the strengths and correct any errors in these solutions to make the definitive decision."
    )
    refined_resp = chat(refinement_prompt, model=model, temperature=0.1)
    choice = extract_answer_choice(refined_resp)

    return {
        "baseline": "ensemble_refinement",
        "answer": choice,
        "refined_response": refined_resp,
        "candidate_solutions": initial_solutions,
        "api_calls": get_api_call_count() - start_calls
    }

def medprompt(
    question: str,
    train_examples: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    images: Optional[List[Union[str, Path]]] = None,
    n_shuffles: int = 3
) -> Dict[str, Any]:
    start_calls = get_api_call_count()
    shuffled_answers = []

    for s_idx in range(n_shuffles):
        rng = random.Random(s_idx + 42)
        sampled_ex = rng.sample(train_examples, min(3, len(train_examples))) if train_examples else []
        ex_str = "\n\n".join([f"Q: {e['question']}\nA: {e['answer']}" for e in sampled_ex])

        prompt = (
            f"You are a top clinical specialist. Review these medical examples:\n{ex_str}\n\n"
            f"Now think step by step and answer:\n{question}"
        )
        resp = chat(prompt, model=model, temperature=0.3, images=images)
        shuffled_answers.append(extract_answer_choice(resp))

    counts = Counter(shuffled_answers)
    final_choice = counts.most_common(1)[0][0] if shuffled_answers else ""

    return {
        "baseline": "medprompt",
        "answer": final_choice,
        "ensemble_votes": shuffled_answers,
        "api_calls": get_api_call_count() - start_calls
    }
