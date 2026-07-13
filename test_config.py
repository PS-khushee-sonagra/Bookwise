from app.config import GEMINI_API_KEY, UPLOAD_DIR, FAISS_DIR

print("API Key Loaded:", (GEMINI_API_KEY[:10] + "...") if GEMINI_API_KEY else "None")
print("Upload Directory:", UPLOAD_DIR)
print("FAISS Directory:", FAISS_DIR)