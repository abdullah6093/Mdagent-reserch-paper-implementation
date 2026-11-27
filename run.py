"""
Main executable runner for the MDAgents framework.
Supports:
- Adaptive collaboration (MDAgents: low / moderate / high)
- Forced difficulty ablations (low, moderate, high)
- Baselines (Zero-shot, Few-shot, CoT, CoT-SC, ER, Medprompt, Voting, MedAgents, etc.)
- Multi-seed evaluations (e.g. 50 samples x 3 seeds as in NeurIPS paper)
- Detailed metrics, API call counting, result caching, and mock simulation mode
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from config import (
    DEFAULT_MODEL, DEFAULT_NUM_AGENTS, NUM_SAMPLES,
    MAX_ROUNDS, MAX_TURNS, OUTPUT_DIR
)
from data import load_dataset, sample_dataset, create_question_text, get_few_shot_examples
from complexity import determine_complexity
from recruiter import recruit
from pipelines import run_pipeline
from decide import extract_answer_choice
from llm import set_mock_mode
import baselines
from eval import analyze_results, format_summary_table

def parse_args():
    parser = argparse.ArgumentParser(description="MDAgents: Adaptive Collaboration of LLMs for Medical Decision-Making")
    parser.add_argument("--dataset", type=str, default="medqa", help="Dataset name (e.g. medqa, pubmedqa, ddxplus, pmc_vqa)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model (e.g. gpt-4o-mini, gpt-4o, gemini-1.5-flash)")
    parser.add_argument("--difficulty", type=str, default="adaptive", choices=["adaptive", "low", "moderate", "high", "basic", "intermediate", "advanced"], help="Complexity mode")
    parser.add_argument("--mode", type=str, default="adaptive", choices=["adaptive", "baseline"], help="Execution mode")
    parser.add_argument("--baseline", type=str, default="zero_shot", help="Baseline name when mode=baseline")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of questions to test per seed (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-seed run (e.g. 42,123,456)")
    parser.add_argument("--num_agents", type=int, default=DEFAULT_NUM_AGENTS, help="Number of specialists in MDT (default: 3)")
    parser.add_argument("--max_rounds", type=int, default=MAX_ROUNDS, help="Maximum discussion rounds for MDT (default: 5)")
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS, help="Maximum turns per round for MDT (default: 5)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Directory to save JSON results")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode (requires no API keys)")
    return parser.parse_args()

def run_single_experiment(args, seed: int):
    if args.mock:
        set_mock_mode(True)
        print("[INFO] Running in MOCK SIMULATION MODE (No API credits will be consumed).")

    print(f"\n{'='*60}")
    print(f"Running Experiment: Dataset={args.dataset} | Model={args.model} | Mode={args.mode} | Seed={seed}")
    print(f"{'='*60}\n")

    all_test_samples = load_dataset(args.dataset, split="test")
    test_samples = sample_dataset(all_test_samples, n_samples=args.num_samples, seed=seed)
    train_examples = get_few_shot_examples(args.dataset, n=3, seed=seed)

    results = []

    for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        q_text, media_path = create_question_text(sample, shuffle_options=True, seed=seed + idx)
        images = [media_path] if media_path and sample.get("modality") == "image" else None

        if args.mode == "baseline":
            b_func = getattr(baselines, args.baseline, None)
            if b_func is None:
                raise ValueError(f"Unknown baseline: {args.baseline}")
            
            if args.baseline in ["zero_shot", "meta_prompting", "autogen_baseline"]:
                output = b_func(q_text, model=args.model, images=images)
            elif args.baseline in ["few_shot", "few_shot_cot", "cot_sc", "ensemble_refinement", "medprompt"]:
                output = b_func(q_text, examples=train_examples, model=args.model, images=images)
            elif args.baseline in ["majority_voting", "weighted_voting", "borda_count", "medagents", "dylan_baseline"]:
                output = b_func(q_text, model=args.model, n_agents=args.num_agents, images=images)
            else:
                output = b_func(q_text, model=args.model, images=images)

            predicted_choice = output.get("answer", "")
            complexity_level = "baseline"
            pipeline_result = output

        else:
            complexity_level = determine_complexity(
                question=q_text,
                forced_difficulty=args.difficulty,
                model=args.model
            )

            recruitment_plan = recruit(
                question=q_text,
                complexity=complexity_level,
                model=args.model,
                num_agents=args.num_agents
            )

            pipeline_result = run_pipeline(
                question=q_text,
                complexity=complexity_level,
                recruitment_plan=recruitment_plan,
                model=args.model,
                examplers=train_examples,
                images=images,
                max_rounds=args.max_rounds,
                max_turns=args.max_turns
            )

            predicted_choice = extract_answer_choice(pipeline_result.get("final_answer", ""))

        results.append({
            "sample_id": sample.get("id"),
            "question": q_text,
            "ground_truth": sample.get("answer"),
            "predicted_choice": predicted_choice,
            "complexity": complexity_level,
            "api_calls": pipeline_result.get("api_calls", 1),
            "entropy_history": pipeline_result.get("entropy_history", []),
            "pipeline_output": pipeline_result
        })

    summary = analyze_results(results, model_name=args.model)
    report_text = format_summary_table(summary)
    print("\n" + report_text)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = str(args.model).replace("/", "_").replace("\\", "_")
    out_filename = f"{safe_model}_{args.dataset}_{args.mode}_{args.difficulty}_seed{seed}.json"
    out_path = out_dir / out_filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to: {out_path}\n")
    return summary

def main():
    args = parse_args()
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seed_list = [args.seed]

    all_summaries = []
    for s in seed_list:
        summary = run_single_experiment(args, seed=s)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        avg_acc = sum(s["accuracy"] for s in all_summaries) / len(all_summaries)
        avg_calls = sum(s["avg_api_calls_per_query"] for s in all_summaries) / len(all_summaries)
        print("\n" + "#" * 60)
        print(f"MULTI-SEED AGGREGATE ({len(all_summaries)} seeds)")
        print(f"Mean Accuracy:     {avg_acc * 100:.2f}%")
        print(f"Mean API Calls:    {avg_calls:.2f} calls/query")
        print("#" * 60 + "\n")

if __name__ == "__main__":
    main()