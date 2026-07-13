"""
Test script for BookWise RAG Pipeline

Flow tested:

PDF
 ↓
Document Processor
 ↓
Chunker
 ↓
Embedding Generator
 ↓
FAISS Vector Store
 ↓
Retriever
 ↓
Prompt Builder
 ↓
Gemini LLM
"""

import logging

from app.rag_pipeline import RAGPipeline
from app.document_processor import DocumentProcessor


logging.basicConfig(
    level=logging.INFO
)


logger = logging.getLogger(__name__)



def main():

    try:

        # Initialize components
        pipeline = RAGPipeline()

        processor = DocumentProcessor()


        # -------------------------
        # 1. Process PDF
        # -------------------------

        print("\nProcessing PDF...\n")


        pages = processor.extract_text(
            "test.pdf"
        )


        print(
            f"Extracted pages: {len(pages)}"
        )


        # Optional verification
        print("\nFirst page preview:")
        print(
            pages[0]["text"][:300]
        )



        # -------------------------
        # 2. Index document
        # -------------------------

        print("\nIndexing document...\n")


        indexed_chunks = pipeline.index_document(
            pages=pages,
            filename="test.pdf"
        )


        print(
            f"Indexed chunks: {indexed_chunks}"
        )



        # -------------------------
        # 3. Ask question
        # -------------------------

        print("\nAsking question...\n")


        question = "What is RAG?"


        answer = pipeline.ask(
            question
        )


        print("\nQUESTION:")
        print(question)


        print("\nANSWER:")
        print(answer)



    except Exception as e:

        logger.exception(
            "Test failed"
        )

        print(
            f"\nERROR: {e}"
        )



if __name__ == "__main__":
    main()