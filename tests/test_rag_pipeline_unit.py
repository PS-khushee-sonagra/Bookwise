import pytest
from app.rag_pipeline import RAGPipeline

def test_rag_pipeline_empty_index(tmp_path, monkeypatch):
    # Redirect FAISS_DIR to a temporary folder
    monkeypatch.setattr("app.rag_pipeline.FAISS_DIR", tmp_path)
    
    pipeline = RAGPipeline()
    
    # Test asking when index is empty/missing
    response = pipeline.ask("What is RAG?")
    assert "I don't know based on the uploaded document" in response["answer"]
    assert response["pages"] == []

def test_rag_pipeline_filtering_and_citations(tmp_path, monkeypatch):
    # Redirect FAISS_DIR to a temporary folder
    monkeypatch.setattr("app.rag_pipeline.FAISS_DIR", tmp_path)
    
    pipeline = RAGPipeline(similarity_threshold=0.35)
    
    # Index a mock document
    pages = [
        {"page_number": 1, "text": "This page talks about RAG pipelines and vector stores."},
        {"page_number": 2, "text": "This page is about python development."}
    ]
    pipeline.index_document(pages, "bookwise.pdf")
    
    # 1. Normal question: Mock score is 1.0, which is >= 0.35 threshold
    response = pipeline.ask("What is RAG?")
    assert "mocked answer" in response["answer"]
    assert len(response["pages"]) > 0
    assert "bookwise.pdf — Page 1" in response["pages"][0]
    
    # 2. Filtered out query: Use a high threshold override of 1.1 to simulate no chunk passing the threshold
    response_fallback = pipeline.ask("What is RAG?", threshold=1.1)
    assert response_fallback["answer"] == "I couldn't find relevant information in the uploaded documents."
    assert response_fallback["pages"] == []

def test_rag_pipeline_persistence_on_startup(tmp_path, monkeypatch):
    # Redirect FAISS_DIR to a temporary folder
    monkeypatch.setattr("app.rag_pipeline.FAISS_DIR", tmp_path)
    
    # Create and index in pipeline 1
    pipeline1 = RAGPipeline()
    pages = [{"page_number": 1, "text": "Persistent knowledge text."}]
    pipeline1.index_document(pages, "persistence.pdf")
    
    # Pipeline 2 initialization should load from disk instead of clearing
    pipeline2 = RAGPipeline()
    assert pipeline2.vector_store.index is not None
    assert len(pipeline2.vector_store.metadata) > 0
    
    response = pipeline2.ask("Persistent query")
    assert "mocked answer" in response["answer"]
