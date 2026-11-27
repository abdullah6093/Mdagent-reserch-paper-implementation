"""
Unified LLM wrapper for the MDAgents framework.
Handles caching, retries, API counters, multimodal inputs (images/videos),
and unified interface for Groq (OpenAI-compatible), OpenAI, and Google Gemini models.
Includes mock simulation mode for testing full pipelines without API keys.
"""

import os
import re
import json
import hashlib
import sqlite3
import base64
import threading
import time
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

import requests
from dotenv import load_dotenv

from config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, MODEL_MAP, VISION_MODELS,
    CACHE_DIR, LOG_DIR, DEFAULT_TEMPERATURE
)

load_dotenv()

logger = logging.getLogger("mdagents.llm")

GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ── Global thread-safe counters ──
_lock = threading.Lock()
_api_call_count = 0
_total_prompt_tokens = 0
_total_completion_tokens = 0
_mock_mode = False


def set_mock_mode(enabled: bool = True):
    global _mock_mode
    _mock_mode = enabled


def is_mock_mode() -> bool:
    return _mock_mode


# ── Disk Cache Setup (SQLite) ──
_cache_db = CACHE_DIR / "llm_cache.db"


def _init_cache():
    conn = sqlite3.connect(str(_cache_db))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "key TEXT PRIMARY KEY, "
        "response TEXT, "
        "model TEXT, "
        "timestamp REAL)"
    )
    conn.commit()
    conn.close()


_init_cache()


def _cache_key(model: str, messages: Any, temperature: float, seed: Optional[int]) -> str:
    serialized = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed
    }, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> Optional[str]:
    try:
        conn = sqlite3.connect(str(_cache_db))
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def _set_cached(key: str, response: str, model: str):
    try:
        conn = sqlite3.connect(str(_cache_db))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO cache (key, response, model, timestamp) VALUES (?, ?, ?, ?)",
            (key, response, model, time.time())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# ── API Counters and Stats ──
def get_api_call_count() -> int:
    with _lock:
        return _api_call_count


def reset_api_call_count():
    global _api_call_count, _total_prompt_tokens, _total_completion_tokens
    with _lock:
        _api_call_count = 0
        _total_prompt_tokens = 0
        _total_completion_tokens = 0


def increment_api_call_count(prompt_tokens: int = 0, completion_tokens: int = 0):
    global _api_call_count, _total_prompt_tokens, _total_completion_tokens
    with _lock:
        _api_call_count += 1
        _total_prompt_tokens += prompt_tokens
        _total_completion_tokens += completion_tokens


def get_total_tokens() -> Dict[str, int]:
    with _lock:
        return {
            "prompt_tokens": _total_prompt_tokens,
            "completion_tokens": _total_completion_tokens,
            "total_tokens": _total_prompt_tokens + _total_completion_tokens
        }


# ── JSONL Logging ──
_log_file = LOG_DIR / "llm_interactions.jsonl"


def _log_interaction(model: str, messages: Any, response: str, temperature: float, cached: bool):
    try:
        entry = {
            "timestamp": time.time(),
            "model": model,
            "temperature": temperature,
            "cached": cached,
            "messages": messages,
            "response": response
        }
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log interaction: {e}")


# ── Media Processing Helpers ──
def encode_image_base64(image_path: Union[str, Path]) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sample_video_frames(video_path: Union[str, Path], num_frames: int = 8) -> List[str]:
    """Sample num_frames evenly from a video and return as list of base64 JPEG strings."""
    try:
        import cv2
        path = str(video_path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file {path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        interval = max(1, total_frames // num_frames)
        frames_b64 = []

        for i in range(num_frames):
            frame_idx = min(i * interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

        cap.release()
        return frames_b64
    except Exception as e:
        logger.warning(f"Video frame sampling unavailable: {e}")
        return []


# ── Mock Clinical Response Generator ──
def _generate_mock_response(messages: List[Dict[str, Any]], temperature: float) -> str:
    """Generate realistic medical mock response for pipeline testing without API keys."""
    increment_api_call_count(prompt_tokens=300, completion_tokens=120)
    last_msg = messages[-1]["content"] if messages else ""
    last_text = str(last_msg).lower()

    if "difficulty/complexity" in last_text or "assess the complexity" in last_text:
        return random.choice(["1) low", "2) moderate", "3) high", "moderate"])

    if "assemble a multi-disciplinary team" in last_text or "recruiter" in last_text:
        return (
            "Cardiologist == Specialist in cardiovascular physiology and diagnostic criteria. > "
            "Endocrinologist == Expert in hormone regulation, thyroid function, and metabolic pathways. > "
            "Internal Medicine Specialist == General internist providing comprehensive differential diagnosis."
        )

    if "integrated care team (ict)" in last_text:
        return (
            "[Initial Assessment Team (IAT)]\n"
            "- Member: Triage Physician (Lead) == Evaluates acute clinical presentation.\n"
            "- Member: General Internist == Performs systemic review of patient symptoms.\n\n"
            "[Differential & Expert Team (DET)]\n"
            "- Member: Subspecialist (Lead) == Formulates disease-specific differentials.\n"
            "- Member: Pathologist == Correlates laboratory markers and histology.\n\n"
            "[Final Review & Decision Team (FRDT)]\n"
            "- Member: Senior Attending (Lead) == Synthesizes all team reports for definitive decision.\n"
            "- Member: Clinical Pharmacologist == Evaluates contraindications and pharmacology."
        )

    if "participating in a multidisciplinary team discussion" in last_text or "indicate whether you want to talk" in last_text:
        return "yes, I would like to consult the Endocrinologist regarding the biomarker findings."

    if "remind your medical expertise and leave your opinion" in last_text or "persuasion" in last_text:
        return "Colleagues, based on the elevated TSH and low free T4 with a non-tender goiter, this presentation is classic for Hashimoto's thyroiditis. Answer: (B)."

    if "moderator" in last_text and "feedback" in last_text:
        return "Both agents should reconsider the autoimmune etiology versus toxic nodular goiter. Please converge towards consensus on the clinical signs."

    if "synthesize these findings" in last_text or "report synthesizer" in last_text:
        return "Comprehensive analysis indicates an autoimmune destruction of thyroid follicular cells. The findings strongly support Answer: (B)."

    choices = ["A", "B", "C", "D"]
    chosen = random.choice(choices)
    return f"Based on thorough clinical reasoning and established guidelines, the most supported diagnosis is Answer: ({chosen})."


def _is_groq_model(model_name: str) -> bool:
    lowered = model_name.lower()
    groq_prefixes = (
        "openai/gpt-oss",
        "qwen/",
        "llama",
        "meta-llama/",
        "mixtral",
        "gemma",
        "deepseek",
        "groq/",
    )
    return any(lowered.startswith(p) or f"/{p}" in lowered for p in groq_prefixes)


def _parse_groq_wait_seconds(resp: requests.Response, attempt: int) -> float:
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after) + 1.0
        except ValueError:
            pass

    match = re.search(r"try again in ([0-9.]+)(ms|s)", resp.text)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        seconds = (val / 1000.0) if unit == "ms" else val
        return seconds + 1.0

    return min(2 ** attempt, 30)


# ── Unified Chat Function ──
def chat(
    messages: Union[str, List[Dict[str, Any]]],
    model: str = "openai/gpt-oss-20b",
    temperature: float = DEFAULT_TEMPERATURE,
    images: Optional[List[Union[str, Path]]] = None,
    seed: Optional[int] = None,
    use_cache: bool = True
) -> str:
    """
    Unified chat function for all LLM calls.
    Supports Groq, OpenAI, Google Gemini, caching, retries, mock mode, and multimodal inputs.
    """
    resolved_model = MODEL_MAP.get(model, model)

    if isinstance(messages, str):
        msg_list = [{"role": "user", "content": messages}]
    else:
        msg_list = list(messages)

    if _mock_mode:
        return _generate_mock_response(msg_list, temperature)

    cache_key = _cache_key(resolved_model, msg_list, temperature, seed)
    if use_cache:
        cached_resp = _get_cached(cache_key)
        if cached_resp is not None:
            _log_interaction(resolved_model, msg_list, cached_resp, temperature, cached=True)
            return cached_resp

    response_text = ""
    if "gemini" in resolved_model.lower():
        response_text = _call_gemini(msg_list, resolved_model, temperature, images)
    elif _is_groq_model(resolved_model):
        response_text = _call_groq(msg_list, resolved_model, temperature, images, seed)
    else:
        response_text = _call_openai(msg_list, resolved_model, temperature, images, seed)

    if use_cache and response_text:
        _set_cached(cache_key, response_text, resolved_model)

    _log_interaction(resolved_model, msg_list, response_text, temperature, cached=False)
    return response_text


def _build_openai_compatible_messages(
    messages: List[Dict[str, Any]],
    images: Optional[List[Union[str, Path]]]
) -> List[Dict[str, Any]]:
    formatted_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
        if "content" in msg:
            formatted_messages.append({"role": role, "content": msg["content"]})
        elif "parts" in msg and isinstance(msg["parts"], list):
            text = "\n".join(
                p.get("text", "") for p in msg["parts"] if isinstance(p, dict) and p.get("text")
            )
            formatted_messages.append({"role": role, "content": text})

    if images:
        last_user_idx = -1
        for idx in range(len(formatted_messages) - 1, -1, -1):
            if formatted_messages[idx]["role"] == "user":
                last_user_idx = idx
                break

        if last_user_idx != -1:
            orig_content = formatted_messages[last_user_idx]["content"]
            content_list: List[Dict[str, Any]] = [{"type": "text", "text": str(orig_content)}]
            for img_path in images:
                b64 = encode_image_base64(img_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            formatted_messages[last_user_idx]["content"] = content_list

    return formatted_messages


def _call_groq(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    images: Optional[List[Union[str, Path]]],
    seed: Optional[int]
) -> str:
    """Make call to Groq Chat Completions API with Retry-After and TPM backoff."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set. Please set it in .env file or use --mock for simulated testing.")

    formatted_messages = _build_openai_compatible_messages(messages, images)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_retries = 8
    for attempt in range(max_retries):
        try:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": 512,
            }
            if seed is not None:
                payload["seed"] = seed

            resp = requests.post(GROQ_CHAT_ENDPOINT, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage") or {}
                prompt_toks = int(usage.get("prompt_tokens") or 250)
                comp_toks = int(usage.get("completion_tokens") or 150)

                choices = data.get("choices", [])
                if choices:
                    text_resp = choices[0].get("message", {}).get("content", "") or ""
                    if text_resp:
                        increment_api_call_count(prompt_toks, comp_toks)
                        time.sleep(2.0)
                        return text_resp

            if resp.status_code == 429:
                wait_s = _parse_groq_wait_seconds(resp, attempt)
                logger.warning(
                    f"Groq TPM rate limit (429). Waiting {wait_s:.1f}s before retry {attempt + 1}/{max_retries}..."
                )
                time.sleep(wait_s)
                continue

            logger.warning(f"Groq REST attempt {attempt + 1} status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Groq REST attempt {attempt + 1} failed: {e}")

        if attempt == max_retries - 1:
            raise RuntimeError(f"Groq API request failed after {max_retries} attempts.")
        time.sleep(min(2 ** attempt, 20))

    return ""


def _call_openai(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    images: Optional[List[Union[str, Path]]],
    seed: Optional[int]
) -> str:
    """Make call to OpenAI API with exponential backoff retries."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai package: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        api_key = os.getenv("openai_api_key", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in .env file or use --mock for simulated testing.")

    client = OpenAI(api_key=api_key)
    formatted_messages = _build_openai_compatible_messages(messages, images)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": formatted_messages,
                "temperature": temperature,
            }
            if seed is not None:
                kwargs["seed"] = seed

            completion = client.chat.completions.create(**kwargs)
            prompt_toks = completion.usage.prompt_tokens if completion.usage else 0
            comp_toks = completion.usage.completion_tokens if completion.usage else 0
            increment_api_call_count(prompt_toks, comp_toks)

            return completion.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

    return ""


def _call_gemini(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    images: Optional[List[Union[str, Path]]]
) -> str:
    """Make call to Google Gemini via direct REST API (clean, pure HTTP, zero DLL dependency)."""
    api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    if not api_key:
        api_key = os.getenv("genai_api_key", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in .env file or use --mock for simulated testing.")

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append({"text": f"[{role.upper()}]:\n{content}\n"})

    if images:
        for img_path in images:
            b64_data = encode_image_base64(img_path)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_data
                }
            })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": temperature}
    }

    gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(gemini_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if content_parts:
                        text_resp = content_parts[0].get("text", "")
                        increment_api_call_count(prompt_tokens=250, completion_tokens=150)
                        return text_resp
            logger.warning(f"Gemini REST attempt {attempt + 1} status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Gemini REST attempt {attempt + 1} failed: {e}")

        if attempt == max_retries - 1:
            raise RuntimeError(f"Gemini API request failed after {max_retries} attempts.")
        time.sleep(2 ** attempt)

    return ""