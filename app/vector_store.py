"""
Vector Store for BookWise

Responsibilities:
- Store embeddings in FAISS.
- Store metadata.
- Save and load the index.
- Perform similarity search.
"""

import sentence_transformers
from pathlib import Path
from typing import Dict, List
import logging
import pickle

# pyrefly: ignore [missing-import]
import faiss
# pyrefly: ignore [missing-import]
import numpy as np

from app.config import (
    PERSISTENT_INDEXING,
    SIMILARITY_THRESHOLD,
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store.
    """

    def __init__(self, index_dir: str | Path):

        self.index_dir = Path(index_dir)

        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.pkl"

        self.index = None
        self.metadata: List[Dict] = []

    def add_documents(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict],
    ) -> None:
        """
        Add embeddings and metadata.
        """

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embeddings and chunks must have the same length."
            )

        # Load existing index and metadata from disk if they exist, aren't already loaded in memory, and PERSISTENT_INDEXING is enabled
        if PERSISTENT_INDEXING and self.index is None and self.index_path.exists() and self.metadata_path.exists():
            try:
                self.load()
            except Exception as e:
                logger.warning("Failed to load existing index/metadata: %s", e)

        dimension = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(
                embeddings.astype(np.float32)
            )
            self.metadata = chunks.copy()
        else:
            self.index.add(
                embeddings.astype(np.float32)
            )
            self.metadata.extend(chunks)

        logger.info(
            "Indexed %d chunks. Total chunks in index: %d",
            len(chunks),
            len(self.metadata),
        )

    def save(self) -> None:
        """
        Save FAISS index and metadata.
        """

        if self.index is None:
            raise RuntimeError("No index to save.")

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "wb",
        ) as f:
            # NOTE: pickle is used here for storing metadata.
            # Only serialize/deserialize data from trusted local storage.
            pickle.dump(self.metadata, f)

        logger.info("Vector store saved.")

    def load(self) -> None:
        """
        Load FAISS index and metadata.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                "FAISS index not found."
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Metadata file not found."
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "rb",
        ) as f:
            # SECURITY NOTE: pickle.load is vulnerable to arbitrary code execution
            # if loading an untrusted file. Ensure this file is only loaded from
            # a trusted local source (e.g. locally indexed metadata).
            self.metadata = pickle.load(f)

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"FAISS index size ({self.index.ntotal}) does not match metadata size ({len(self.metadata)})."
            )

        logger.info(
            "Loaded %d chunks.",
            len(self.metadata),
        )

    def clear(self) -> None:
        """
        Clear the in-memory index and metadata, and delete the saved files on disk.
        """
        self.index = None
        self.metadata = []

        if self.index_path.exists():
            try:
                self.index_path.unlink()
                logger.info("Deleted FAISS index file: %s", self.index_path)
            except Exception as e:
                logger.warning("Failed to delete FAISS index file: %s", e)

        if self.metadata_path.exists():
            try:
                self.metadata_path.unlink()
                logger.info("Deleted metadata file: %s", self.metadata_path)
            except Exception as e:
                logger.warning("Failed to delete metadata file: %s", e)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search for similar chunks.
        """

        if self.index is None:
            raise RuntimeError(
                "Vector store is not loaded."
            )

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            top_k,
        )

        results = []

        MIN_SCORE = SIMILARITY_THRESHOLD
        for score, idx in zip(
            distances[0],
            indices[0],
            
        ):
            if score < MIN_SCORE:
              continue


            # Check for invalid index (like -1 which is returned by FAISS when it can't find matches)
            # or index out of range to prevent incorrect metadata mapping
            if idx < 0 or idx >= len(self.metadata):
                logger.warning("Skipping invalid FAISS index retrieved: %d", idx)
                continue

            chunk = self.metadata[idx].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results