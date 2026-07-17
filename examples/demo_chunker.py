import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.document_processor import DocumentProcessor
from app.chunker import Chunker

processor = DocumentProcessor()
chunker = Chunker()

pdf_path = os.path.join(project_root, "data", "uploads", "rich-dad-poor-dad.pdf")
pages = processor.extract_text(pdf_path)

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
