"""
Retriever for BookWise

Responsibilities:
- Convert user queries into embeddings.
- Retrieve the most relevant document chunks.
"""

from typing import Dict, List
import logging

from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves the most relevant chunks for a user query.
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks.

        Args:
            query: User question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of retrieved chunk dictionaries.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info("Retrieving top %d chunks.", top_k)

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        logger.info(
            "Retrieved %d chunks.",
            len(results),
        )

        return results