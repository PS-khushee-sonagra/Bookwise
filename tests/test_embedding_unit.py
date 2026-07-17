import numpy as np
from app.embedding_generator import EmbeddingGenerator

def test_embedding_generator_interface():
    generator = EmbeddingGenerator()
    chunks = [{"text": "Hello world"}, {"text": "Testing embeddings"}]
    
    embeddings = generator.embed_documents(chunks)
    
    # Verify dimensions
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    
    # Verify embeddings are normalized (L2 norm should be close to 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    
    # Verify there are no NaN values
    assert not np.isnan(embeddings).any()
    
    query_emb = generator.embed_query("test query")
    
    # Verify dimensions
    assert isinstance(query_emb, np.ndarray)
    assert query_emb.shape == (384,)
    
    # Verify query embedding is normalized and contains no NaN values
    assert np.allclose(np.linalg.norm(query_emb), 1.0, atol=1e-5)
    assert not np.isnan(query_emb).any()

