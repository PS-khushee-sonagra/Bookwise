from app.config import FAISS_DIR
from app.document_processor import DocumentProcessor
from app.chunker import Chunker
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore

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