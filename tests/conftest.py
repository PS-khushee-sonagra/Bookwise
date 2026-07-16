import pytest
import numpy as np
from unittest.mock import MagicMock
from app.embedding_generator import EmbeddingGenerator
from app.llm import LLMGenerator

@pytest.fixture(autouse=True)
def mock_embedding_generator(monkeypatch):
    """
    Mock the EmbeddingGenerator class methods to return mock vectors
    and avoid loading the actual sentence-transformers model.
    """
    def fake_embed_documents(self, chunks):
        # Return mock normalized embeddings (vectors of 1s normalized)
        dim = 384
        vectors = np.ones((len(chunks), dim), dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def fake_embed_query(self, query):
        dim = 384
        vector = np.ones(dim, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm

    monkeypatch.setattr(EmbeddingGenerator, "__init__", lambda self, model_name=None: None)
    monkeypatch.setattr(EmbeddingGenerator, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(EmbeddingGenerator, "embed_query", fake_embed_query)

@pytest.fixture(autouse=True)
def mock_llm_generator(monkeypatch):
    """
    Mock LLMGenerator to return mock answers from Gemini and bypass API key validation.
    """
    monkeypatch.setattr(LLMGenerator, "__init__", lambda self, model_name=None: None)
    monkeypatch.setattr(LLMGenerator, "generate", lambda self, prompt: "This is a mocked answer from Gemini.")
