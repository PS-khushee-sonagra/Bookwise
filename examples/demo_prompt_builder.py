import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.config import FAISS_DIR
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.prompt_builder import PromptBuilder

embedder = EmbeddingGenerator()

store = VectorStore(FAISS_DIR)
store.load()

retriever = Retriever(embedder, store)

builder = PromptBuilder()

results = retriever.retrieve(
    "What is financial freedom?",
    top_k=5,
)

prompt = builder.build_prompt(
    "What is financial freedom?",
    results,
)

print(prompt)
