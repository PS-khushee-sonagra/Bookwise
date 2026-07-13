from app.document_processor import DocumentProcessor
from app.chunker import Chunker

processor = DocumentProcessor()
chunker = Chunker()

pages = processor.extract_text(
    "data/uploads/rich-dad-poor-dad.pdf"
)

chunks = chunker.create_chunks(
    pages,
    "rich-dad-poor-dad.pdf"
)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")

print()

print(chunks[0])

print()

print(chunks[1])