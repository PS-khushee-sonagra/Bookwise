from app.chunker import Chunker

def test_chunker_create_chunks():
    # Test base chunking and metadata preservation
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"page_number": 1, "text": "This is page one. Sentence one is here."},
        {"page_number": 2, "text": "This is page two. Sentence two is here."},
    ]
    chunks = chunker.create_chunks(pages, "test_doc.pdf")
    
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source"] == "test_doc.pdf"
    assert chunks[0]["page_number"] == 1
    assert "chunk_id" in chunks[0]
    assert "text" in chunks[0]

def test_chunker_overlap_validation():
    # Test that consecutive chunks overlap correctly
    chunker = Chunker(chunk_size=20, chunk_overlap=10)
    pages = [
        {"page_number": 1, "text": "One. Two. Three. Four. Five. Six. Seven. Eight."}
    ]
    chunks = chunker.create_chunks(pages, "overlap_test.pdf")
    
    assert len(chunks) >= 2
    # First chunk should contain "Three."
    # Second chunk should start with "Three." as it is carried over by overlap
    assert "Three." in chunks[0]["text"]
    assert "Three." in chunks[1]["text"]

