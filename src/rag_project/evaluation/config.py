from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv() 


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_DATASET = (
    PROJECT_ROOT / "articles" / "golden_dataset.txt"
)

RESULTS_DIR = (
    PROJECT_ROOT / "evaluation" / "results"
)


# ---------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------

LANGUAGES = ["en", "ar"]

MAX_WORKERS = 2

RAGAS_MODEL = "openai/gpt-oss-120b"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY2 = os.environ.get("GROQ_API_KEY2")



# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )