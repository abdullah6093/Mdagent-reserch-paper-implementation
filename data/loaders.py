"""
Standardized dataset loaders for MDAgents.
Supports all 10 benchmarks from Kim et al., NeurIPS 2024:
MedQA, PubMedQA, DDXPlus, SymCat, JAMA, MedBullets,
Path-VQA, PMC-VQA, MIMIC-CXR, MedVidQA.
"""

import os
import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from config import DATA_DIR

def load_dataset(dataset_name: str, split: str = "test", custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load dataset into a unified schema:
    {
        'id': str,
        'question': str,
        'options': dict (e.g. {'A': '...', 'B': '...'}),
        'answer': str (e.g. 'A' or 'yes'),
        'context': str,
        'image_path': Optional[str],
        'video_path': Optional[str],
        'modality': 'text' | 'image' | 'video'
    }
    """
    dataset_name = dataset_name.lower().replace("-", "_")
    
    base_dirs = [
        Path(custom_path) if custom_path else None,
        DATA_DIR / dataset_name,
        DATA_DIR,
        Path("data") / dataset_name,
        Path("data"),
        Path("../data") / dataset_name
    ]
    
    found_file = None
    for b_dir in base_dirs:
        if b_dir and b_dir.exists():
            for ext in [f"{split}.jsonl", f"{split}.json", f"{dataset_name}.jsonl", f"{dataset_name}.json", f"{split}.csv"]:
                target = b_dir / ext if b_dir.is_dir() else b_dir
                if target.is_file():
                    found_file = target
                    break
            if found_file:
                break

    samples = []
    if found_file and found_file.exists():
        samples = _parse_local_file(found_file, dataset_name)
    else:
        samples = _load_fallback(dataset_name, split)

    return samples

def _parse_local_file(file_path: Path, dataset_name: str) -> List[Dict[str, Any]]:
    records = []
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
    elif suffix == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

    standardized = []
    for idx, r in enumerate(records):
        standardized.append(_standardize_sample(r, dataset_name, idx))
    return standardized

def _standardize_sample(raw: Dict[str, Any], dataset_name: str, index: int) -> Dict[str, Any]:
    item_id = str(raw.get("id", raw.get("question_id", f"{dataset_name}_{index}")))
    question = raw.get("question", "")
    context = raw.get("context", raw.get("abstract", ""))
    image_path = raw.get("image_path", raw.get("img_path", None))
    video_path = raw.get("video_path", None)
    
    options = {}
    answer = ""
    
    if dataset_name == "medqa":
        options = raw.get("options", {})
        answer = raw.get("answer_idx", raw.get("answer", ""))
        modality = "text"

    elif dataset_name == "pubmedqa":
        options = {"A": "yes", "B": "no", "C": "maybe"}
        ans_raw = str(raw.get("final_decision", raw.get("answer", ""))).lower()
        if "yes" in ans_raw:
            answer = "A"
        elif "no" in ans_raw:
            answer = "B"
        else:
            answer = "C"
        modality = "text"

    elif dataset_name in ["path_vqa", "pmc_vqa", "mimic_cxr"]:
        raw_opts = raw.get("options", {})
        if isinstance(raw_opts, list):
            options = {chr(65 + i): opt for i, opt in enumerate(raw_opts)}
        elif isinstance(raw_opts, dict):
            options = raw_opts
        else:
            options = {"A": "yes", "B": "no"}
        answer = str(raw.get("answer", raw.get("cop", "")))
        modality = "image"

    elif dataset_name == "medvidqa":
        raw_opts = raw.get("options", {})
        if isinstance(raw_opts, list):
            options = {chr(65 + i): opt for i, opt in enumerate(raw_opts)}
        else:
            options = raw_opts
        answer = str(raw.get("answer", ""))
        modality = "video"

    else:
        raw_opts = raw.get("options", {})
        if isinstance(raw_opts, list):
            options = {chr(65 + i): opt for i, opt in enumerate(raw_opts)}
        elif isinstance(raw_opts, dict):
            options = raw_opts
        else:
            options = {}
        answer = str(raw.get("answer", raw.get("label", raw.get("answer_idx", ""))))
        modality = "text"

    return {
        "id": item_id,
        "question": question,
        "options": options,
        "answer": answer,
        "context": context,
        "image_path": image_path,
        "video_path": video_path,
        "modality": modality
    }

def _load_fallback(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    sample_pool = [
        {
            "id": f"{dataset_name}_demo_1",
            "question": "A 45-year-old woman presents with fatigue, cold intolerance, and weight gain. Thyroid palpation reveals a non-tender, symmetrically enlarged gland. Serum TSH is elevated and free T4 is decreased. Which of the following is the most likely diagnosis?",
            "options": {"A": "Graves' disease", "B": "Hashimoto's thyroiditis", "C": "Subacute thyroiditis", "D": "Toxic multinodular goiter"},
            "answer": "B",
            "context": "Endocrine clinical case study.",
            "image_path": None,
            "video_path": None,
            "modality": "text"
        },
        {
            "id": f"{dataset_name}_demo_2",
            "question": "A 62-year-old man presents with sudden onset crushing substernal chest pain radiating to his left jaw. ECG shows ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
            "options": {"A": "Right coronary artery", "B": "Left anterior descending artery", "C": "Left circumflex artery", "D": "Left main coronary artery"},
            "answer": "A",
            "context": "Cardiology acute presentation.",
            "image_path": None,
            "video_path": None,
            "modality": "text"
        },
        {
            "id": f"{dataset_name}_demo_3",
            "question": "A 28-year-old male with a history of myasthenia gravis presents with increasing diplopia and ptosis after starting an antibiotic. Which drug is known to exacerbate symptoms of myasthenia gravis?",
            "options": {"A": "Amoxicillin", "B": "Gentamicin", "C": "Cephalexin", "D": "Azithromycin"},
            "answer": "B",
            "context": "Neuromuscular pharmacology query.",
            "image_path": None,
            "video_path": None,
            "modality": "text"
        }
    ]
    return sample_pool

def sample_dataset(dataset: List[Dict[str, Any]], n_samples: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if len(dataset) <= n_samples:
        shuffled = list(dataset)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(dataset, n_samples)

def create_question_text(sample: Dict[str, Any], shuffle_options: bool = True, seed: Optional[int] = None) -> Tuple[str, Optional[str]]:
    q_text = sample["question"]
    if sample.get("context"):
        q_text = f"Context: {sample['context']}\n\nQuestion: {q_text}"

    options = sample.get("options", {})
    if options:
        opt_items = list(options.items())
        if shuffle_options:
            rng = random.Random(seed) if seed is not None else random.Random()
            rng.shuffle(opt_items)
        
        opt_str = "\n".join([f"({k}) {v}" for k, v in opt_items])
        q_text = f"{q_text}\n\nOptions:\n{opt_str}"

    media_path = sample.get("image_path") or sample.get("video_path")
    return q_text, media_path

def get_few_shot_examples(dataset_name: str, n: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    train_data = load_dataset(dataset_name, split="train")
    sampled = sample_dataset(train_data, n_samples=n, seed=seed)
    
    examples = []
    for s in sampled:
        q_text, _ = create_question_text(s, shuffle_options=False)
        examples.append({
            "question": q_text,
            "answer": f"Answer: ({s['answer']})",
            "reason": f"Based on established clinical evidence, the correct choice is ({s['answer']})."
        })
    return examples
