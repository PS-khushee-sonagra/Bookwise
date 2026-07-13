"""
Embedding Generator for BookWise

Responsibilities:
- Load the embedding model once.
- Generate embeddings for document chunks.
- Generate embeddings for user queries.
"""

from typing import List, Dict

import logging
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Load embedding model.
        """

        logger.info("Loading embedding model: %s", model_name)

        self.model = SentenceTransformer(model_name)

        logger.info("Embedding model loaded successfully.")

    def embed_documents(
        self,
        chunks: List[Dict],
    ) -> np.ndarray:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            NumPy array of embeddings.
        """

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        logger.info(
            "Generated %d document embeddings.",
            len(embeddings),
        )

        return embeddings

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Generate embedding for a user query.

        Args:
            query: User question.

        Returns:
            Query embedding.
        """

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info("Generated query embedding.")

        return embedding