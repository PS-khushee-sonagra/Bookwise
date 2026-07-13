from app.config import FAISS_DIR
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore
from app.retriever import Retriever

embedder = EmbeddingGenerator()

store = VectorStore(FAISS_DIR)
store.load()

retriever = Retriever(
    embedder=embedder,
    vector_store=store,
)

results = retriever.retrieve(
    "What is financial freedom?",
    top_k=5,
)

print("\nTop Results:\n")

for i, result in enumerate(results, start=1):

    print("=" * 70)

    print(f"Rank: {i}")

    print(f"Score: {result['score']:.4f}")

    print(f"Page: {result['page_number']}")

    print()

    print(result["text"][:400])