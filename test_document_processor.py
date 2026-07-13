from app.document_processor import DocumentProcessor

processor = DocumentProcessor()

pages = processor.extract_text(
    "data/uploads/rich-dad-poor-dad.pdf"
)

print(f"\nTotal Pages: {len(pages)}\n")

print("-" * 60)

print("First Page Preview:\n")

print(pages[0]["text"][:1000])

print("-" * 60)

print("\nReturned Object:\n")

print(pages[0])