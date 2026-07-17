import pytest
import numpy as np
from app.vector_store import VectorStore

def test_vector_store_operations(tmp_path):
    store = VectorStore(index_dir=tmp_path)
    
    # 1. Test empty index load failure
    with pytest.raises(FileNotFoundError):
        store.load()
        
    # 2. Add documents
    chunks = [
        {
            "chunk_id": "doc1_p1_c1",
            "text": "First chunk content.",
            "page_number": 1,
            "metadata": {"source": "doc1.pdf", "page_number": 1}
        },
        {
            "chunk_id": "doc1_p1_c2",
            "text": "Second chunk content.",
            "page_number": 1,
            "metadata": {"source": "doc1.pdf", "page_number": 1}
        }
    ]
    
    embeddings = np.array([
        [1.0] + [0.0] * 383,
        [0.0, 1.0] + [0.0] * 382
    ], dtype=np.float32)
    
    store.add_documents(embeddings, chunks)
    assert store.index is not None
    assert len(store.metadata) == 2
    assert store.index.ntotal == 2
    
    # 2.5 Add more documents and assert index size increases
    more_chunks = [
        {
            "chunk_id": "doc1_p1_c3",
            "text": "Third chunk content.",
            "page_number": 1,
            "metadata": {"source": "doc1.pdf", "page_number": 1}
        }
    ]
    more_embeddings = np.array([
        [0.0, 0.0, 1.0] + [0.0] * 381
    ], dtype=np.float32)
    
    store.add_documents(more_embeddings, more_chunks)
    assert store.index.ntotal == 3
    assert len(store.metadata) == 3
    
    # 3. Save index
    store.save()
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.pkl").exists()
    
    # 4. Load index in a new store instance
    new_store = VectorStore(index_dir=tmp_path)
    new_store.load()
    assert new_store.index is not None
    assert len(new_store.metadata) == 3
    assert new_store.metadata[0]["chunk_id"] == "doc1_p1_c1"
    
    # 5. Search
    query_emb = np.array([1.0] + [0.0] * 383, dtype=np.float32)
    results = new_store.search(query_emb, top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc1_p1_c1"
    assert "score" in results[0]
    
    # 6. Clear
    new_store.clear()
    assert new_store.index is None
    assert len(new_store.metadata) == 0
    assert not (tmp_path / "index.faiss").exists()
