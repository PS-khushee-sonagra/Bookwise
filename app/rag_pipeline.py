"""
RAG Pipeline for BookWise

Responsibilities:
- Process documents.
- Create chunks.
- Generate embeddings.
- Store embeddings in FAISS.
- Retrieve relevant chunks.
- Build grounded prompts.
- Generate answers using LLM.
"""

import logging

from app.chunker import Chunker
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore
from app.prompt_builder import PromptBuilder
from app.llm import LLMGenerator
from app.config import FAISS_DIR


logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self):

        self.chunker = Chunker()

        self.embedder = EmbeddingGenerator()

        self.vector_store = VectorStore(
            FAISS_DIR
        )

        self.prompt_builder = PromptBuilder()

        self.llm = LLMGenerator()

        self.clear_index()

    def clear_index(self, force: bool = False) -> None:
        """
        Clear the vector store index.
        """
        from app.config import PERSISTENT_INDEXING
        if force or not PERSISTENT_INDEXING:
            self.vector_store.clear()
            logger.info("RAGPipeline index cleared.")


    def index_document(
        self,
        pages,
        filename: str
    ) -> int:
        """
        Index document into FAISS.

        Flow:

        PDF Pages
            ↓
        Chunks
            ↓
        Embeddings
            ↓
        FAISS
            ↓
        Save
        """

        try:

            # 1. Create chunks with page metadata
            chunks = self.chunker.create_chunks(
                pages=pages,
                filename=filename
            )

            logger.info(
                "Created %d chunks",
                len(chunks)
            )


            # 2. Generate embeddings
            embeddings = self.embedder.embed_documents(
                chunks
            )

            logger.info(
                "Generated document embeddings"
            )


            # 3. Add embeddings to FAISS
            self.vector_store.add_documents(
                embeddings,
                chunks
            )

            logger.info(
                "Documents added to vector store"
            )


            # 4. Save FAISS index
            self.vector_store.save()

            logger.info(
                "Vector store saved successfully"
            )


            return len(chunks)


        except Exception as e:

            logger.exception(
                "Document indexing failed"
            )

            raise RuntimeError(
                f"Indexing failed: {e}"
            ) from e



    def ask(
        self,
        question: str
    ) -> dict:
        """
        Answer question using RAG.

        Flow:

        Question
            ↓
        Query Embedding
            ↓
        FAISS Search
            ↓
        Prompt Builder
            ↓
        Gemini
            ↓
        Answer + Sources
        """

        try:

            # Validate question
            if not question.strip():

                raise ValueError(
                    "Question cannot be empty."
                )


            # 1. Load FAISS if not loaded
            if self.vector_store.index is None:
                try:
                    self.vector_store.load()
                except FileNotFoundError:
                    logger.info("No index files found on disk. Returning empty answer.")
                    return {
                        "answer": "I don't know based on the uploaded document.",
                        "pages": []
                    }

            if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
                logger.info("FAISS index is empty. Returning empty answer.")
                return {
                    "answer": "I don't know based on the uploaded document.",
                    "pages": []
                }


            # 2. Generate query embedding
            query_embedding = self.embedder.embed_query(
                question
            )


            # 3. Retrieve relevant chunks
            results = self.vector_store.search(
                query_embedding,
                top_k=8
            )


            logger.info(
                "Retrieved %d chunks",
                len(results)
            )


            # 4. Build grounded prompt
            prompt = self.prompt_builder.build_prompt(
                question,
                results
            )


            # 5. Generate answer
            answer = self.llm.generate(
                prompt
            )


            # 6. Extract source pages and document names
            unique_sources = set()

            for chunk in results:

                if isinstance(chunk, dict):

                    metadata = chunk.get(
                        "metadata",
                        {}
                    )

                    source = metadata.get(
                        "source"
                    )

                    page_number = metadata.get(
                        "page_number"
                    )

                    if source and page_number:
                        unique_sources.add(
                            (source, page_number)
                        )
                    elif page_number:
                        unique_sources.add(
                            ("Unknown Document", page_number)
                        )


            # Sort by document name first, then by page number
            sorted_sources = sorted(
                list(unique_sources),
                key=lambda x: (x[0], x[1])
            )


            # Format citations
            formatted_pages = []
            for source, page_number in sorted_sources:
                if source != "Unknown Document":
                    formatted_pages.append(
                        f"{source} — Page {page_number}"
                    )
                else:
                    formatted_pages.append(
                        f"Page {page_number}"
                    )


            return {
                "answer": answer,
                "pages": formatted_pages
            }

        except Exception as e:

            logger.exception(
                "Question answering failed"
            )

            raise RuntimeError(
                f"RAG answer failed: {e}"
            ) from e