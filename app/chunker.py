"""
Chunker for BookWise

Responsibilities:
- Split PDF pages into smaller chunks.
- Generate unique chunk IDs.
- Preserve page metadata.
"""

import re
from typing import List, Dict
import logging


logger = logging.getLogger(__name__)


class Chunker:
    """
    Splits document pages into chunks for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def create_chunks(
    self,
    pages: List[Dict],
    filename: str,
) -> List[Dict]:
        """
        Create chunks from PDF pages.

        Args:
            pages:
                List of dictionaries:
                {
                    "page_number": int,
                    "text": str
                }

            filename:
                PDF file name.

        Returns:
            List of chunk dictionaries.
        """

        if not pages:
            raise ValueError(
                "Cannot chunk empty document."
            )

        chunks = []
        global_chunk_number = 1

        for page in pages:
            page_number = page.get(
                "page_number",
                None
            )
            text = page.get(
                "text",
                ""
            )

            if not text.strip():
                continue

            # Split text into sentences using regex boundary detection
            raw_sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in raw_sentences if s.strip()]

            page_chunks_text = []
            current_chunk_sentences = []
            current_chunk_len = 0

            for sentence in sentences:
                  sentence_len = len(sentence)

                  # Fallback: if sentence is longer than chunk_size, split by characters
                  if sentence_len > self.chunk_size:
                      if current_chunk_sentences:
                          page_chunks_text.append(" ".join(current_chunk_sentences))
                          current_chunk_sentences = []
                          current_chunk_len = 0

                      sub_start = 0
                      while sub_start < sentence_len:
                          sub_end = sub_start + self.chunk_size
                          page_chunks_text.append(sentence[sub_start:sub_end])
                          sub_start = sub_end - self.chunk_overlap
                      continue

                  # Normal sentence grouping
                  if current_chunk_len + sentence_len > self.chunk_size and current_chunk_sentences:
                      page_chunks_text.append(" ".join(current_chunk_sentences))

                      # Handle overlap: keep last sentences that fit within chunk_overlap
                      overlap_sentences = []
                      overlap_len = 0
                      for s in reversed(current_chunk_sentences):
                          if overlap_len + len(s) <= self.chunk_overlap:
                              overlap_sentences.insert(0, s)
                              overlap_len += len(s) + 1  # +1 for space
                          else:
                              break
                      current_chunk_sentences = overlap_sentences
                      current_chunk_len = overlap_len

                  current_chunk_sentences.append(sentence)
                  current_chunk_len += sentence_len + (1 if current_chunk_len > 0 else 0)

            if current_chunk_sentences:
                page_chunks_text.append(" ".join(current_chunk_sentences))

            # Build chunk dicts
            page_chunk_number = 1
            for chunk_text in page_chunks_text:
                if not chunk_text.strip():
                    continue

                chunk = {
                    "chunk_id": (
                        f"{filename}"
                        f"_p{page_number}"
                        f"_c{page_chunk_number}"
                    ),
                    "text": chunk_text,
                    "page_number": page_number,
                    "metadata": {
                        "source": filename,
                        "page_number": page_number,
                        "chunk_number": global_chunk_number,
                    },
                }
                chunks.append(chunk)
                global_chunk_number += 1
                page_chunk_number += 1

        logger.info(
            "Created %d chunks from %s",
            len(chunks),
            filename,
        )

        return chunks