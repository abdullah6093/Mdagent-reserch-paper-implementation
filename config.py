"""MDAgents Configuration - All constants from Kim et al., NeurIPS 2024."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── API Keys ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Model Mappings ──
MODEL_MAP = {
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "gpt-4v": "gpt-4-vision-preview",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-pro": "gemini-pro",
    "gemini-pro-vision": "gemini-pro-vision",
    "gemini-1.5-flash": "gemini-1.5-flash",
}

# Models that support vision/images
VISION_MODELS = {"gpt-4v", "gpt-4o", "gpt-4o-mini", "gemini-pro-vision", "gemini-1.5-flash"}

# ── Default Parameters (from paper) ──
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
TEMPERATURE_ENSEMBLE = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2]
MAX_ROUNDS = 5         # R: max discussion rounds (MDT)
MAX_TURNS = 5          # T: max turns per round
DEFAULT_NUM_AGENTS = 3 # N: paper's best N=3
DEFAULT_NUM_TEAMS = 3  # number of ICT teams
NUM_FEWSHOT = 3        # 3-shot for low complexity
NUM_SAMPLES = 50       # 50 samples per dataset per seed
SEEDS = [42, 123, 456] # 3 random seeds

# ── Complexity levels ──
COMPLEXITY_LOW = "low"
COMPLEXITY_MODERATE = "moderate"
COMPLEXITY_HIGH = "high"

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in [DATA_DIR, OUTPUT_DIR, CACHE_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Supported datasets ──
DATASETS = [
    "medqa", "pubmedqa", "ddxplus", "symcat",
    "jama", "medbullets", "path_vqa", "pmc_vqa",
    "mimic_cxr", "medvidqa"
]

TEXT_DATASETS = ["medqa", "pubmedqa", "ddxplus", "symcat", "jama", "medbullets"]
IMAGE_DATASETS = ["path_vqa", "pmc_vqa", "mimic_cxr"]
VIDEO_DATASETS = ["medvidqa"]
