import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.config import FAISS_DIR
from app.document_processor import DocumentProcessor
from app.chunker import Chunker
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore

processor = DocumentProcessor()
chunker = Chunker()
embedder = EmbeddingGenerator()

pdf_path = os.path.join(project_root, "data", "uploads", "rich-dad-poor-dad.pdf")
pages = processor.extract_text(pdf_path)

chunks = chunker.create_chunks(
    pages,
    "rich-dad-poor-dad.pdf",
)

embeddings = embedder.embed_documents(chunks)

store = VectorStore(FAISS_DIR)

store.add_documents(
    embeddings,
    chunks,
)

store.save()

print("Index saved!")

store.load()

query = embedder.embed_query(
    "What is financial freedom?"
)

results = store.search(
    query,
    top_k=5,
)


print()

print("Top Results:\n")

for result in results:

    print("=" * 70)

    print(result["page_number"])

    print(result["score"])

    print(result["text"][:250])
