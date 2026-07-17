import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.document_processor import DocumentProcessor

processor = DocumentProcessor()

pdf_path = os.path.join(project_root, "data", "uploads", "rich-dad-poor-dad.pdf")
pages = processor.extract_text(pdf_path)

print(f"\nTotal Pages: {len(pages)}\n")

print("-" * 60)

print("First Page Preview:\n")

print(pages[0]["text"][:1000])

print("-" * 60)

print("\nReturned Object:\n")

print(pages[0])
