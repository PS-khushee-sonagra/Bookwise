import numpy as np
from app.retriever import Retriever
from app.vector_store import VectorStore
from app.embedding_generator import EmbeddingGenerator

def test_retriever_flow(tmp_path):
    embedder = EmbeddingGenerator()
    store = VectorStore(index_dir=tmp_path)
    
    # Add dummy document so search works
    chunks = [
        {
            "chunk_id": "doc1_p1_c1",
            "text": "Retrieved content.",
            "page_number": 1,
            "metadata": {"source": "doc1.pdf", "page_number": 1}
        }
    ]
    embeddings = np.ones((1, 384), dtype=np.float32)
    store.add_documents(embeddings, chunks)
    
    retriever = Retriever(embedder=embedder, vector_store=store)
    results = retriever.retrieve("test query", top_k=1)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1_p1_c1"
