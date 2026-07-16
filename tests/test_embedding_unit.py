import numpy as np
from app.embedding_generator import EmbeddingGenerator

def test_embedding_generator_interface():
    generator = EmbeddingGenerator()
    chunks = [{"text": "Hello world"}, {"text": "Testing embeddings"}]
    
    embeddings = generator.embed_documents(chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    
    query_emb = generator.embed_query("test query")
    assert isinstance(query_emb, np.ndarray)
    assert query_emb.shape == (384,)
