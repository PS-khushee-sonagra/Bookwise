# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent


# Load environment variables first
load_dotenv(ROOT_DIR / ".env")


# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add it to your .env file."
    )


# Gemini model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-flash-latest"
)


# Data directories
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
FAISS_DIR = ROOT_DIR / "data" / "faiss_index"


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# Caching and persistence
PERSISTENT_INDEXING = False

# Similarity threshold for document chunk filtering
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
