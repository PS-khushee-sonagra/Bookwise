from app.document_processor import DocumentProcessor
from app.chunker import Chunker
from app.embedding_generator import EmbeddingGenerator

processor = DocumentProcessor()
chunker = Chunker()
embedder = EmbeddingGenerator()

pages = processor.extract_text(
    "data/uploads/rich-dad-poor-dad.pdf"
)

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