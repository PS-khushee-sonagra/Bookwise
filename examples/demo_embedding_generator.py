import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.document_processor import DocumentProcessor
from app.chunker import Chunker
from app.embedding_generator import EmbeddingGenerator

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

print()

print(f"Chunks: {len(chunks)}")

print(f"Embedding Shape: {embeddings.shape}")

print()

print("First Chunk")

print(chunks[0])

print()

print("First Embedding (first 10 values)")

print(embeddings[0][:10])

query_embedding = embedder.embed_query(
    "What is financial freedom?"
)

print()

print("Query Shape:")

print(query_embedding.shape)
