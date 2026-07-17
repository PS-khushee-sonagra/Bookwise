import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.config import GEMINI_API_KEY, UPLOAD_DIR, FAISS_DIR

print("API Key Loaded:", bool(GEMINI_API_KEY))
print("Upload Directory:", UPLOAD_DIR)
print("FAISS Directory:", FAISS_DIR)
